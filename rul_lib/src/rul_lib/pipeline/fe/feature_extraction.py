import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import asdict
from tsfresh import extract_relevant_features, extract_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import from_columns
from typing import Any, Callable
from torch.utils.data import DataLoader
# local
from rul_lib.training.trainer import FeTrainer
from rul_lib.training.train_logger import TrainLogger
from rul_lib.pipeline.fe import models
from rul_lib.pipeline.fe.parser import AEParams, TsfreshParams, DilatedCnnAEParams
from rul_lib.pipeline.parser.scheduler import build_scheduler
from rul_lib.training.tune_reporter import make_tune_cfg
from rul_lib.utils.dict_utils import flatten_conditional
from rul_lib.utils.json_utils import json_safe_dict
from rul_lib.pipeline.fe.cnn.datasets import AEDataset
from rul_lib.pipeline.fe.cnn.encode import WindowEncoder, TemporalWindowEncoder
from rul_lib.pipeline.fe.cnn.encode import compute_latents
from rul_lib.gls.gls import logger


def make_registry() -> tuple[dict[str, type], Callable[..., Any]]:
  ''' Create a registry and a decorator to register classes. '''
  reg: dict[str, type] = {}

  def register(name: str, params_cls: type | None = None) -> Callable[[type], type]:

    def deco(cls: type) -> type:
      cls._registry_key = name
      if params_cls is not None:
        cls.params_cls = staticmethod(lambda: params_cls)
      reg[name] = cls
      return cls

    return deco

  return reg, register


_FE_REGISTRY, register_fe_model = make_registry()


def _deep_update(d: dict, u: dict) -> dict:
  for k, v in u.items():
    if isinstance(v, dict) and isinstance(d.get(k), dict):
      d[k] = _deep_update(d[k], v)
    else:
      d[k] = v
  return d


class BaseFeAdapter(ABC):
  ''' Base class for feature-extraction adapters. '''

  @abstractmethod
  def fit_encode(self,
                 data: dict[str, Any],
                 params: dict,
                 tracking: dict,
                 train_logger: TrainLogger | None = None) -> dict[str, Any]:
    ''' Fit feature-extraction model and encode data. '''
    ...

  @classmethod
  @abstractmethod
  def describe(cls) -> dict[str, Any]:
    ''' Describe the feature-extraction model. '''
    ...


class TorchFeAdapter(BaseFeAdapter):
  ''' Base class for torch-based feature-extraction adapters. '''

  name = 'base'
  model_cls: type | None = None
  encoder_cls: type | None = None
  temporal: bool = False
  register_prefix: str = 'fe_torch'

  def build_model(self, p: Any, feats: int, win: int) -> nn.Module:
    raise NotImplementedError

  def build_encoder(self, trained_model: nn.Module, device: torch.device) -> nn.Module:
    raise NotImplementedError

  @classmethod
  def params_cls(cls) -> type:
    raise NotImplementedError('register_fe_model must inject params_cls')

  def _set_seed(self, seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(seed)

  def _device(self, p: Any) -> torch.device:
    return torch.device(p.device)

  def _check_input(self, x_train: np.ndarray, x_test: np.ndarray) -> tuple[int, int]:
    if x_train.ndim != 3 or x_test.ndim != 3:
      raise ValueError('expected x_train/x_test with shape [N, W, F]')
    win = int(x_train.shape[1])
    feats = int(x_train.shape[2])
    return win, feats

  def _dataloaders(self, x_train: np.ndarray, x_test: np.ndarray, batch_size: int) -> tuple[DataLoader, DataLoader]:
    dl_tr = DataLoader(AEDataset(x_train), batch_size=batch_size, shuffle=True, drop_last=False)
    dl_va = DataLoader(AEDataset(x_test), batch_size=batch_size, shuffle=False, drop_last=False)
    return dl_tr, dl_va

  def _train(self,
             model: nn.Module,
             p: Any,
             params: dict,
             tracking: dict,
             train_loader: DataLoader,
             val_loader: DataLoader,
             win: int,
             feats: int,
             register_name: str,
             train_logger: TrainLogger | None = None) -> nn.Module:
    device = self._device(p)
    opt = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
    sched = build_scheduler(optimizer=opt, cfg=p)
    cfg = dict(params or {})
    cfg.update({'window_size': win, 'n_features': feats, 'patience': p.patience, 'scheduler': p.scheduler})
    # keep common keys if present
    for k in ('latent_dim', 'hidden_ch', 'batch_size', 'epochs', 'learning_rate', 'weight_decay', 'random_seed', 'device'): # yapf: disable
      if hasattr(p, k):
        cfg[k] = getattr(p, k)
    trainer = FeTrainer(model=model,
                        run_name=f'{register_name}_v{tracking.get("version", "0.0.0")}',
                        criterion=nn.MSELoss(),
                        optimizer=opt,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        num_epochs=p.epochs,
                        device=device,
                        version=str(tracking.get('version', '0.0.0')),
                        scheduler=sched,
                        config=cfg,
                        patience=p.patience,
                        train_logger=train_logger,
                        tune_cfg=make_tune_cfg(phase='fe', fe_epochs=0))
    trained = trainer.train(
        register_name=register_name,
        tags={
            'stage': 'feature_extraction',
            'win': str(win),
            'latent_dim': str(getattr(p, 'latent_dim', 'na')),
        },
        code_paths=[models.__file__],
    )
    return trained

  def fit_encode(self,
                 data: dict[str, Any],
                 params: dict,
                 tracking: dict,
                 train_logger: TrainLogger | None = None) -> dict[str, Any]:
    ''' Fit feature-extraction model and encode data. '''
    P = self.params_cls()
    p = P.from_dict(flatten_conditional(params) or {})
    x_train = np.asarray(data['x_train'], dtype=np.float32)
    x_test = np.asarray(data['x_test'], dtype=np.float32)
    win, feats = self._check_input(x_train, x_test)
    self._set_seed(int(getattr(p, 'random_seed', 42)))
    device = self._device(p)
    dl_tr, dl_va = self._dataloaders(x_train, x_test, batch_size=int(getattr(p, 'batch_size', 64)))
    model = self.build_model(p=p, feats=feats, win=win).to(device)
    model_name = str(getattr(p, 'model_name', getattr(self, '_registry_key', self.name)))
    register_name = f'{self.register_prefix}_{model_name}_w{win}_z{int(getattr(p, "latent_dim", 0))}'
    trained_model = self._train(model=model,
                                p=p,
                                params=params,
                                tracking=tracking,
                                train_loader=dl_tr,
                                val_loader=dl_va,
                                win=win,
                                feats=feats,
                                register_name=register_name,
                                train_logger=train_logger)

    encoder = self.build_encoder(trained_model, device).to(device).eval()
    z_train, z_test = compute_latents(encoder=encoder, windowed={'x_train': x_train, 'x_test': x_test})
    return {
        'encoder': {
            'name': model_name,
            'model': encoder,
            'model_type': 'torch',
        },
        'z_train': z_train,
        'z_test': z_test,
        'metrics': {
            'latent_dim': int(z_train.shape[1]),
            'window_size': int(win),
            'n_features': int(feats),
            'temporal': bool(self.temporal),
        },
    }

  @classmethod
  def describe(cls) -> dict[str, Any]:
    P = cls.params_cls()
    p = P()
    fe_name = getattr(cls, '_registry_key', cls.name)
    base = {
        'description': f'{fe_name} torch-based feature extractor.',
        'inputs': {
            'x_train': '[N_train, W, F] float32',
            'x_test': '[N_test, W, F] float32',
        },
        'outputs': {
            'z_train': '[N_train, latent_dim]',
            'z_test': '[N_test, latent_dim]',
        },
        'params': json_safe_dict(p),
        'trainer': {
            'criterion': 'MSELoss',
            'optimizer': 'Adam',
            'scheduler': 'optional',
            'early_stopping': True,
        },
        'encoder': {
            'type': 'torch',
            'temporal': bool(getattr(cls, 'temporal', False)),
        },
        'source': {
            'model_class': cls.model_cls.__name__ if cls.model_cls else None,
            'encoder_class': cls.encoder_cls.__name__ if cls.encoder_cls else None,
            'module': cls.model_cls.__module__ if cls.model_cls else None,
        },
    }
    override = cls.describe_override()
    if override:
      base = _deep_update(base, override)
    return base

  @classmethod
  def describe_override(cls) -> dict[str, Any]:
    return {}


@register_fe_model('cnn-ae', params_cls=AEParams)
class CnnAe(TorchFeAdapter):
  name = 'cnn-ae'
  model_cls = models.ConvAe
  encoder_cls = WindowEncoder
  temporal = False
  register_prefix = 'fe_torch'

  def build_model(self, p: AEParams, feats: int, win: int) -> nn.Module:
    return self.model_cls(in_feats=feats, **asdict(p))

  def build_encoder(self, trained_model: nn.Module, device: torch.device) -> nn.Module:
    return WindowEncoder(trained_model)

  @classmethod
  def describe_override(cls) -> dict[str, Any]:
    return {'description': 'CNN autoencoder with strided convolutions for window-wise feature extraction.'}


@register_fe_model('dilated-cnn-ae', params_cls=DilatedCnnAEParams)
class DilatedCnnAe(TorchFeAdapter):
  name = 'dilated-cnn-ae'
  model_cls = models.DilatedCnnAE
  encoder_cls = WindowEncoder
  temporal = False
  register_prefix = 'fe_torch'

  def build_model(self, p: DilatedCnnAEParams, feats: int, win: int) -> nn.Module:
    kwargs = asdict(p).copy()
    kwargs.pop('in_ch', None)
    return self.model_cls(in_ch=feats, **kwargs)

  def build_encoder(self, trained_model: nn.Module, device: torch.device) -> nn.Module:
    return WindowEncoder(trained_model)

  @classmethod
  def describe_override(cls) -> dict[str, Any]:
    return {
        'description': 'Dilated CNN autoencoder capturing multi-scale temporal context.',
        'architecture': {
            'downsampling': 'strided conv',
            'context': 'dilated residual blocks',
        },
    }


@register_fe_model('cnn-ae-temp', params_cls=AEParams)
class CnnAeTemp(TorchFeAdapter):
  name = 'cnn-ae-temp'
  model_cls = models.ConvAeTemp
  encoder_cls = TemporalWindowEncoder
  temporal = True
  register_prefix = 'fe_torch'

  def build_model(self, p: AEParams, feats: int, win: int) -> nn.Module:
    return self.model_cls(in_feats=feats, latent_dim=p.latent_dim, hidden=p.hidden_ch)

  def build_encoder(self, trained_model: nn.Module, device: torch.device) -> nn.Module:
    return TemporalWindowEncoder(trained_model)

  @classmethod
  def describe_override(cls) -> dict[str, Any]:
    return {
        'description': 'Temporal CNN autoencoder preserving temporal order in latent space.',
        'encoder': {
            'temporal': True
        },
    }


@register_fe_model('flatten')
class FlattenFe(BaseFeAdapter):

  def fit_encode(self,
                 data: dict[str, Any],
                 params: dict,
                 tracking: dict,
                 train_logger: TrainLogger | None = None) -> dict[str, Any]:
    x_tr = np.asarray(data['x_train'], dtype=np.float32)
    x_te = np.asarray(data['x_test'], dtype=np.float32)
    if x_tr.ndim != 3 or x_te.ndim != 3:
      raise ValueError('expected x_train/x_test with shape [N, W, F]')

    n_tr, w, f = x_tr.shape
    model = models.Flatten(window_size=w, n_features=f)
    z_train = model.predict(model_input=x_tr)
    z_test = model.predict(model_input=x_te)
    metrics = {'latent_dim': int(w * f), 'window_size': int(w), 'n_features': int(f)}
    return {
        'encoder': {
            'name': 'flatten',
            'model': model,
            'model_type': 'tabular'
        },
        'z_train': z_train,
        'z_test': z_test,
        'metrics': metrics,
    }

  @classmethod
  def describe(cls) -> dict[str, Any]:
    return {
        'description': 'Flatten each window [W, F] into a single vector of length W*F. No training.',
        'inputs': {
            'x_train': '[N, W, F] float32',
            'x_test': '[N, W, F] float32'
        },
        'outputs': {
            'z_train': '[N, W*F]',
            'z_test': '[N, W*F]'
        },
        'params': {},
        'trainer': {},
        'source': {
            'module': models.__name__,
            'class': 'Flatten'
        },
    }


@register_fe_model('tsfresh')
class TsfreshFe(BaseFeAdapter):

  def fit_encode(self,
                 data: dict[str, np.ndarray],
                 params: dict,
                 tracking: dict,
                 train_logger: TrainLogger | None = None) -> dict[str, any]:
    x_train = np.asarray(data['x_train'], dtype=np.float32)
    x_test = np.asarray(data['x_test'], dtype=np.float32)
    y_train = np.asarray(data['y_train'], dtype=np.float32)
    if x_train.ndim != 3 or x_test.ndim != 3:
      raise ValueError('expected x_train/x_test with shape [N, W, F]')
    p = TsfreshParams.from_dict(flatten_conditional(params) or {})
    fc_params = MinimalFCParameters() if p.fc_params == 'minimal' else EfficientFCParameters()
    # --- choose channels: drop meta like unit/cycle/os if present ---
    n, w, f = x_train.shape
    keep_idx = list(range(f))
    # heuristic: drop channel 0 if it looks like integer unit ids (constant over time, mostly integers)
    ch0 = x_train[..., 0]
    if np.allclose(ch0, np.round(ch0)) and np.all(ch0.std(axis=1) < 1e-9):
      keep_idx = [i for i in keep_idx if i != 0]
    # allow explicit override via params (optional)
    if hasattr(p, 'drop_channels') and p.drop_channels:
      keep_idx = [i for i in keep_idx if i not in set(p.drop_channels)]
    if len(keep_idx) == 0:
      raise ValueError('no sensor channels left after dropping meta channels')
    # names for kinds (stable)
    kind_names = [f'c{i}' for i in keep_idx]

    def _to_long_df(x):
      n, w, f = x.shape
      # slice kept channels
      xk = x[:, :, keep_idx]  # [N, W, K]
      k = xk.shape[2]
      ids = np.repeat(np.arange(n), w * k)
      times = np.tile(np.repeat(np.arange(w), k), n)
      kinds = np.tile(np.array(kind_names), n * w)
      vals = xk.reshape(-1)
      return pd.DataFrame({'id': ids, 'time': times, 'kind': kinds, 'value': vals})

    # y must have index matching 'id'
    y_ser = pd.Series(y_train, index=np.arange(len(y_train)))
    # ---- TRAIN: relevant features ----
    df_tr_long = _to_long_df(x_train)
    xtr = extract_relevant_features(df_tr_long,
                                    y_ser,
                                    column_id='id',
                                    column_sort='time',
                                    column_kind='kind',
                                    column_value='value',
                                    default_fc_parameters=fc_params,
                                    n_jobs=getattr(p, 'n_jobs', 0),
                                    disable_progressbar=True)
    impute(xtr)
    selected_cols = list(xtr.columns)
    # optional hard cap using p-values
    max_k = getattr(p, 'max_features', None)
    if isinstance(max_k, int) and max_k > 0 and xtr.shape[1] > max_k:
      rel = calculate_relevance_table(xtr, y_ser)
      rel = rel.sort_values('p_value', ascending=True)
      keep = rel['feature'].head(max_k).tolist()
      xtr = xtr[keep]
      selected_cols = keep
    # ---- TEST: same calculators + identical column order ----
    df_te_long = _to_long_df(x_test)
    reduced_fc = from_columns(selected_cols)
    xte = extract_features(df_te_long,
                           column_id='id',
                           column_sort='time',
                           column_kind='kind',
                           column_value='value',
                           kind_to_fc_parameters=reduced_fc,
                           n_jobs=getattr(p, 'n_jobs', 0),
                           disable_progressbar=True)
    # ensure identical order and impute missing cols (if any)
    xte = xte.reindex(columns=selected_cols)
    impute(xte)
    scaler_stats: dict[str, list[float]] | None = None
    z_train = xtr.values.astype(np.float32, copy=False)
    z_test = xte.values.astype(np.float32, copy=False)
    metrics = {
        'latent_dim': int(z_train.shape[1]),
        'window_size': int(x_train.shape[1]),
        'fc_params': p.fc_params,
        'features_selected': int(len(selected_cols)),
        'max_features_cap': int(getattr(p, 'max_features', 0) or 0),
        'dropped_meta_channels': [i for i in range(f) if i not in keep_idx],
        'normalized': bool(p.normalize)
    }
    pyfunc_enc = models.TsfreshEncoderPyfunc(fc_params=p.fc_params,
                                             selected_cols=selected_cols,
                                             fc_parameters=fc_params,
                                             n_jobs=getattr(p, 'n_jobs', 0),
                                             normalize=p.normalize,
                                             scaler_mean=None if not scaler_stats else scaler_stats.get('mean'),
                                             scaler_scale=None if not scaler_stats else scaler_stats.get('scale'))
    spec = {
        'w': int(x_train.shape[1]),
        'f': int(x_train.shape[2]),
        'kept_channels': keep_idx,
        'kind_names': kind_names,
        'latent_dim': int(z_train.shape[1]),
        'fc_params': p.fc_params,
        'selected_features': selected_cols,
        'normalized': bool(p.normalize),
        'scaler_mean': None if not scaler_stats else scaler_stats.get('mean'),
        'scaler_scale': None if not scaler_stats else scaler_stats.get('scale')
    }
    return {
        'encoder': {
            'name': 'tsfresh',
            'model_type': 'tabular',
            'model': pyfunc_enc,
            'spec': spec
        },
        'z_train': z_train,
        'z_test': z_test,
        'metrics': metrics
    }

  @staticmethod
  def describe() -> dict:
    return {
        'description': 'TSFresh: handcrafted time-series features per window.',
        'inputs': {
            'x_train': '[N_train, W, F] float32',
            'x_test': '[N_test, W, F] float32'
        },
        'outputs': {
            'z_train': '[N_train, D_features]',
            'z_test': '[N_test, D_features]'
        },
        'params': json_safe_dict(TsfreshParams()),
        'trainer': {
            'criterion': None,
            'optimizer': None,
            'metric': None
        },
        'source': {
            'module': 'tsfresh',
            'class': 'extract_features'
        },
    }


def train_feature_extraction(data: dict,
                             config: dict,
                             tracking: dict,
                             train_logger: TrainLogger | None = None) -> dict[str, Any]:
  ''' Train feature-extraction model specified by config on data. '''
  name = str(config.get('model', 'cnn-ae'))
  if name not in _FE_REGISTRY:
    raise ValueError(f'unknown feature-extraction model_name={name}. available={list(_FE_REGISTRY)}')
  adapter = _FE_REGISTRY[name]()
  logger.info(f'train feature-extraction model={name}')
  return adapter.fit_encode(data=data, params=config, train_logger=train_logger, tracking=tracking)

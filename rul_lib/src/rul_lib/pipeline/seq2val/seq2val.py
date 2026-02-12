import torch
import importlib
import pkgutil
from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import Any
# local
import rul_lib.pipeline.seq2val.models as models
import rul_lib.pipeline.seq2val.param_parser as pp
from rul_lib.training.trainer import RegressionTrainer, TrainLogger
from rul_lib.training.tune_reporter import make_tune_cfg
from rul_lib.utils.json_utils import json_safe_dict
from rul_lib.gls.gls import logger
from rul_lib.training.loss import ConfigCriterion, compute_metrics
from rul_lib.pipeline.seq2val.utils import build_seq_loaders, eval_torch_model, optimizer_and_sched, to_numpy
from rul_lib.training.utils import select_device
from rul_lib.pipeline.shared.registry import make_registry

_SEQ2VAL_REGISTRY, register_seq2val = make_registry()
_PLUGINS_LOADED = False


def _try_import(module: str) -> None:
  try:
    importlib.import_module(module)
    logger.info(f'seq2val plugin loaded: {module}')
  except ModuleNotFoundError:
    logger.debug(f'seq2val plugin not found: {module}')
  except Exception as e:
    logger.warning(f'failed to import seq2val plugin {module}: {e}')


def _load_seq2val_plugins(extra_modules: list[str] | tuple[str, ...] | None = None,
                          discover_filter: str | list[str] | tuple[str, ...] | None = None,
                          scan_namespace: bool = True) -> None:
  ''' Load seq2val adapter plugins dynamically. '''
  global _PLUGINS_LOADED
  if _PLUGINS_LOADED:
    return
  _PLUGINS_LOADED = True
  # 1) Explicit modules (from config)
  if extra_modules:
    for m in extra_modules:
      if isinstance(m, str) and m:
        _try_import(m)
    if not scan_namespace:
      return
  # 2) Namespace package discovery (generated adapters live here)
  for ns in ('agent_rul.gen_models',):
    try:
      pkg = importlib.import_module(ns)
    except ModuleNotFoundError:
      continue
    paths = getattr(pkg, '__path__', None)
    if not paths:
      continue
    # normalize filters
    filters: tuple[str, ...]
    if isinstance(discover_filter, str):
      filters = (discover_filter,)
    elif isinstance(discover_filter, (list, tuple)):
      filters = tuple(x for x in discover_filter if isinstance(x, str))
    else:
      filters = tuple()
    for mod in pkgutil.iter_modules(paths):
      name = f'{ns}.{mod.name}'
      if filters and not any(f == mod.name or f in name for f in filters):
        continue
      _try_import(name)


class BaseSeq2ValAdapter(ABC):

  @abstractmethod
  def fit_eval(self, windowed: dict, params: dict, train_logger: TrainLogger, tracking: dict,
               cfg_criterion: ConfigCriterion, cfg: dict) -> dict[str, Any]:
    raise NotImplementedError

  @staticmethod
  @abstractmethod
  def describe(self) -> dict:
    ...


class TorchSeq2Val(BaseSeq2ValAdapter):
  name = 'base'  # overwritten by decorator
  model_cls: type | None = None  # set in subclass

  def build_model(self, in_dim: int, p, device: torch.device):
    raise NotImplementedError

  @classmethod
  def describe(cls) -> dict:
    P = cls.params_cls()  # set by decorator
    p = P()  # default instance
    reg_name = getattr(cls, '_registry_key', cls.name)
    d = {
        'description': f'{reg_name} seq2val over windows.',
        'inputs': {
            'x_train': '[N_train, W, F]',
            'x_test': '[N_test, W, F]'
        },
        'outputs': {
            'y_pred': '[N_test, 1]'
        },
        'params': json_safe_dict(p),
        'metric': 'rmse'
    }
    if cls.model_cls is not None:
      d['source'] = {'module': cls.model_cls.__module__, 'class': cls.model_cls.__name__}
    return d

  def fit_eval(self, windowed: dict, params: dict, train_logger: TrainLogger, tracking: dict,
               cfg_criterion: ConfigCriterion, cfg: dict) -> dict[str, Any]:
    P = self.params_cls()
    p = P.from_dict(params)
    device = select_device(pref='cuda')
    in_feats = int(to_numpy(windowed['x_train']).shape[-1])
    dl_tr, dl_va = build_seq_loaders(windowed=windowed, batch_size=p.batch_size, cfg=cfg.get('multi_window', {}))
    model = self.build_model(in_dim=in_feats, p=p, device=device).to(device)
    optimizer, scheduler = optimizer_and_sched(model, p)
    reg_name = getattr(self.__class__, '_registry_key', self.__class__.name)
    register_name = f'{reg_name}_seq2val_f{in_feats}'
    run_name = f'{register_name}_v{tracking.get("version", "0.0.0")}'
    trainer = RegressionTrainer(model=model,
                                run_name=run_name,
                                optimizer=optimizer,
                                train_loader=dl_tr,
                                val_loader=dl_va,
                                num_epochs=p.epochs,
                                device=device,
                                version=str(tracking.get('version', '0.0.0')),
                                scheduler=scheduler,
                                config={
                                    'model': reg_name,
                                    'params': asdict(p)
                                },
                                patience=p.patience,
                                train_logger=train_logger,
                                tune_cfg=make_tune_cfg(phase='seq2val', fe_epochs=0, cfg_criterion=cfg_criterion),
                                cfg_criterion=cfg_criterion)
    trained_model = trainer.train(register_name=register_name,
                                  tags={
                                      'stage': 'seq2val',
                                      'input_dim': in_feats
                                  },
                                  code_paths=[models.__file__])
    pred_val, yv = eval_torch_model(model=trained_model, dl_va=dl_va, device=device)
    metrics = compute_metrics(pred=pred_val, y=yv, config=cfg_criterion, eval=True)
    return {'model': trained_model, 'pred_val': pred_val, 'y_val': yv, **metrics}

  @classmethod
  def params_cls(cls) -> type[pp.BaseParams]:
    raise NotImplementedError('register_seq2val must inject params_cls')


@register_seq2val('cnn', params_cls=pp.CNNParams)
class CNNAdapter(TorchSeq2Val):
  model_cls = models.CNNRegressor

  def build_model(self, in_dim: int, p: pp.CNNParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


@register_seq2val('lstm', params_cls=pp.LSTMParams)
class LSTMAdapter(TorchSeq2Val):
  model_cls = models.LSTMRegressor

  def build_model(self, in_dim: int, p: pp.LSTMParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


@register_seq2val('gru', params_cls=pp.GRUParams)
class GRUAdapter(TorchSeq2Val):
  model_cls = models.GRURegressor

  def build_model(self, in_dim: int, p: pp.GRUParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


@register_seq2val('tcn', params_cls=pp.TCNParams)
class TCNAdapter(TorchSeq2Val):
  model_cls = models.TCNRegressor

  def build_model(self, in_dim: int, p: pp.TCNParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


@register_seq2val('gru_tcn', params_cls=pp.GRUTCNParams)
class GRUTCNAdapter(TorchSeq2Val):
  model_cls = models.GRUTCNRegressor

  def build_model(self, in_dim: int, p: pp.GRUTCNParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


@register_seq2val('cnn_lstm', params_cls=pp.CNNLSTMParams)
class CNNLSTMAdapter(TorchSeq2Val):
  model_cls = models.CNNLSTMRegressor

  def build_model(self, in_dim: int, p: pp.CNNLSTMParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


@register_seq2val('adv_cnn', params_cls=pp.AdvCNNParams)
class AdvCNNAdapter(TorchSeq2Val):
  model_cls = models.AdvCnnRegressor

  def build_model(self, in_dim: int, p: pp.AdvCNNParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


@register_seq2val('adv_tcn', params_cls=pp.AdvTCNParams)
class AdvTCNAdapter(TorchSeq2Val):
  model_cls = models.AdvTCN

  def build_model(self, in_dim: int, p: pp.AdvTCNParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


@register_seq2val('gru_attn', params_cls=pp.GRUAttnParams)
class GRUAttnAdapter(TorchSeq2Val):
  model_cls = models.GRUAttnRegressor

  def build_model(self, in_dim: int, p: pp.GRUAttnParams, device: torch.device):
    return self.model_cls(in_feats=in_dim, **asdict(p)).to(device)


def train_seq2val(windowed: dict, cfg: dict, tracking: dict, train_logger: TrainLogger) -> dict[str, Any]:
  ''' Train and evaluate a seq2val model specified in cfg. '''
  # Load dynamically generated adapters restricted to current version if possible
  plugins = cfg.get('plugins', [])
  only_plugins = bool(cfg.get('plugins_only', False))
  discover_filter = cfg.get('plugin_filter', None)
  if not discover_filter and isinstance(tracking, dict):
    ds = (tracking.get('dataset') or {}).get('name', '')
    ver = str(tracking.get('version', '')).replace('.', '_')
    if ds and ver:
      discover_filter = f'{ds}_{ver}'
  if isinstance(plugins, (list, tuple)) and plugins:
    _load_seq2val_plugins(extra_modules=list(plugins), discover_filter=discover_filter, scan_namespace=not only_plugins)
  else:
    _load_seq2val_plugins(discover_filter=discover_filter, scan_namespace=True)
  name = str(cfg.get('model')).lower()
  if name not in _SEQ2VAL_REGISTRY:
    raise ValueError(f'unknown seq2val model={name}. available={list(_SEQ2VAL_REGISTRY)}')
  params = cfg.get('models', {}).get(name, {})
  cfg_criterion = ConfigCriterion.from_dict(cfg.get('criterion', {}))
  adapter = _SEQ2VAL_REGISTRY[name]()
  logger.info(f'train seq2val model={name}')
  return adapter.fit_eval(windowed=windowed,
                          params=params,
                          train_logger=train_logger,
                          tracking=tracking,
                          cfg_criterion=cfg_criterion,
                          cfg=cfg)

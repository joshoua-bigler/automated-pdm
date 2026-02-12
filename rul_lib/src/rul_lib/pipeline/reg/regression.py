import importlib
import numpy as np
import pkgutil
import torch
import xgboost as xgb
from abc import ABC, abstractmethod
from typing import Any
from dataclasses import asdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from rul_lib.training.tune_reporter import make_tune_cfg, TuneReporter
from rul_lib.training.loss import ConfigCriterion, compute_metrics
# local
import rul_lib.pipeline.reg.models as models
from rul_lib.training.trainer import RegressionTrainer, TrainLogger
import rul_lib.pipeline.reg.param_parser as pp
from rul_lib.pipeline.parser.scheduler import build_scheduler
from rul_lib.training.tune_reporter import make_tune_cfg
from rul_lib.utils.json_utils import json_safe_dict
from rul_lib.gls.gls import logger
from rul_lib.training.loss import ConfigCriterion
from rul_lib.pipeline.reg.utils import eval_torch_model, build_loaders
from rul_lib.training.utils import select_device
from rul_lib.pipeline.shared.registry import make_registry
from rul_lib.training.loss import compute_metrics

_REG_REGISTRY, register_model = make_registry()
_REG_PLUGINS_LOADED = False


def _try_import(module: str) -> None:
  ''' Try to import a module and log the result. '''
  try:
    importlib.import_module(module)
    logger.info(f'regression plugin loaded: {module}')
  except ModuleNotFoundError:
    logger.debug(f'regression plugin not found: {module}')
  except Exception as e:
    logger.warning(f'failed to import regression plugin {module}: {e}')


def _load_reg_plugins(extra_modules: list[str] | tuple[str, ...] | None = None,
                      discover_filter: str | list[str] | tuple[str, ...] | None = None,
                      scan_namespace: bool = True) -> None:
  ''' Load regression plugins from specified modules and by namespace discovery. '''
  global _REG_PLUGINS_LOADED
  if _REG_PLUGINS_LOADED:
    return
  _REG_PLUGINS_LOADED = True
  # 1) Explicit modules (from config)
  if extra_modules:
    for m in extra_modules:
      if isinstance(m, str) and m:
        _try_import(m)
    if not scan_namespace:
      return
  # 2) Namespace discovery for generated adapters
  for ns in ('agent_rul.gen_models',):
    try:
      pkg = importlib.import_module(ns)
    except ModuleNotFoundError:
      continue
    paths = getattr(pkg, '__path__', None)
    if not paths:
      continue
    # normalize filters
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


class BaseRegAdapter(ABC):
  ''' Base class for regression model adapters. '''

  @abstractmethod
  def fit_eval(self, z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray, params: dict,
               train_logger: TrainLogger | None, tracking: dict | None, cfg: dict) -> dict[str, Any]:
    raise NotImplementedError

  @classmethod
  @abstractmethod
  def describe(cls) -> dict:
    ...


class TorchRegBase(BaseRegAdapter):
  ''' Base class for PyTorch regression model adapters. '''
  name = 'base'
  model_cls: type | None = None

  def build_model(self, in_dim: int, p: pp.MlpParams | pp.AdvMlpParams, device: torch.device):
    raise NotImplementedError

  @classmethod
  def describe(cls) -> dict:
    P = cls.params_cls()
    p = P()
    reg_name = getattr(cls, '_registry_key', cls.name)
    d = {
        'description': f'{reg_name} regressor on latent features.',
        'inputs': {
            'z_train': '[n_train, d]',
            'y_train': '[n_train]',
            'z_test': '[n_test, d]',
            'y_test': '[n_test]'
        },
        'outputs': {
            'model': 'torch.nn.Module',
            'obj': 'float',
            'pred_val': '[n_test]',
            'y_val': '[n_test]'
        },
        'params': json_safe_dict(p) if p else {},
        'metric': 'rmse'
    }
    if cls.model_cls is not None:
      d['source'] = {'module': cls.model_cls.__module__, 'class': cls.model_cls.__name__}
    return d

  def fit_eval(self, z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray, params: dict,
               train_logger: TrainLogger, tracking: dict, cfg: dict) -> dict[str, Any]:
    params = dict(params or {})
    P = self.params_cls()
    p = P.from_dict(params)
    in_dim = int(z_train.shape[1])
    device = select_device(params.get('device') if isinstance(params, dict) else None)
    dl_train, dl_va = build_loaders(z_train, y_train, z_test, y_test, batch_size=p.batch_size, device=device, cfg=cfg.get('multi_window', {})) # yapf: disable
    model = self.build_model(in_dim, p, device)
    model.in_dim = in_dim
    p.model_name = params.get('model', self.name) if isinstance(params, dict) else self.name
    optimizer = torch.optim.Adam(model.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay)
    scheduler = build_scheduler(optimizer=optimizer, cfg=p)
    register_name = f'{p.model_name}_f{model.in_dim}'
    run_name = f"{register_name}_v{tracking.get('version', '0.0.0')}"
    cgf_criterion = ConfigCriterion.from_dict(d=params.get('criterion', {}))
    trainer = RegressionTrainer(model=model,
                                run_name=run_name,
                                optimizer=optimizer,
                                train_loader=dl_train,
                                val_loader=dl_va,
                                num_epochs=p.epochs,
                                device=device,
                                version=str(tracking.get('version', '0.0.0')),
                                scheduler=scheduler,
                                config=p.as_dict() if hasattr(p, 'as_dict') else json_safe_dict(p),
                                patience=p.patience,
                                train_logger=train_logger,
                                tune_cfg=make_tune_cfg(phase='reg', fe_epochs=0, cfg_criterion=cgf_criterion),
                                cfg_criterion=cgf_criterion)
    trained = trainer.train(register_name=register_name,
                            tags={
                                'stage': 'regression',
                                'input_dim': model.in_dim
                            },
                            code_paths=[models.__file__])
    pred_val, yv = eval_torch_model(model=trained, dl_va=dl_va, device=device)
    metrics = compute_metrics(pred=pred_val, y=yv, config=trainer.cfg_criterion, eval=True)
    return {'model': trained, 'pred_val': pred_val, 'y_val': yv, **metrics}

  @classmethod
  def params_cls(cls) -> type[pp.MlpParams | pp.AdvMlpParams]:
    raise NotImplementedError('register_seq2val must inject params_cls')


class SklearnRegBase(BaseRegAdapter):
  ''' Base class for scikit-learn regression model adapters. '''
  name = 'base'
  model_cls: type | None = None

  @classmethod
  def describe(cls) -> dict:
    P = cls.params_cls()
    p = P()
    reg_name = getattr(cls, '_registry_key', cls.name)
    d = {
        'description': f'{reg_name} regressor on latent features.',
        'inputs': {
            'z_train': '[n_train, d]',
            'y_train': '[n_train]',
            'z_test': '[n_test, d]',
            'y_test': '[n_test]'
        },
        'outputs': {
            'model': (cls.model_cls.__name__ if cls.model_cls else 'sklearn.BaseEstimator'),
            'obj': 'float',
            'pred_val': '[n_test]',
            'y_val': '[n_test]'
        },
        'params': json_safe_dict(p),
        'metric': 'rmse'
    }
    if cls.model_cls is not None:
      d['source'] = {'module': cls.model_cls.__module__, 'class': cls.model_cls.__name__}
    return d

  def build_model(self, p):
    if self.model_cls is None:
      raise NotImplementedError('model_cls must be set in subclass')
    if hasattr(p, 'as_dict'):
      return self.model_cls(**p.as_dict())
    try:
      return self.model_cls(**asdict(p))
    except Exception:
      return self.model_cls()

  def fit_eval(self, z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray, params: dict,
               train_logger: TrainLogger, tracking: dict, cfg: dict) -> dict[str, Any]:
    P = self.params_cls()
    p = P.from_dict(params or {})
    cfg_criterion = ConfigCriterion.from_dict(d=params.get('criterion', {}) if isinstance(params, dict) else {})
    mdl = self.build_model(p)
    mdl.fit(z_train, y_train)
    pred_tr = mdl.predict(z_train).astype(np.float32)
    pred_val = mdl.predict(z_test).astype(np.float32)
    train_metrics = compute_metrics(pred=pred_tr, y=y_train, config=cfg_criterion, eval=False)
    val_metrics = compute_metrics(pred=pred_val, y=y_test, config=cfg_criterion, eval=True)
    reporter = TuneReporter(make_tune_cfg(phase='reg', fe_epochs=0, cfg_criterion=cfg_criterion))
    reporter.report(step=1, train_metrics=train_metrics, val_metrics=val_metrics)
    return {'model': mdl, 'pred_val': pred_val, 'y_val': y_test, **val_metrics}

  @classmethod
  def params_cls(cls) -> type[pp.MlpParams | pp.AdvMlpParams]:
    raise NotImplementedError('register_model must inject params_cls')


@register_model('mlp_torch', params_cls=pp.MlpParams)
class TorchMLPAdapter(TorchRegBase):
  model_cls = models.MlpRegressor

  def build_model(self, in_dim: int, p: pp.MlpParams, device: torch.device):
    return self.model_cls(in_dim=in_dim, **asdict(p)).to(device)


@register_model('mlp_torch_adv', params_cls=pp.AdvMlpParams)
class AdvTorchMLPAdapter(TorchRegBase):
  model_cls = models.AdvMlpRegressor

  def build_model(self, in_dim: int, p: pp.AdvMlpParams, device: torch.device):
    return self.model_cls(in_dim=in_dim, **asdict(p)).to(device)


@register_model('tcn_torch', params_cls=pp.TcnParams)
class TCNRegAdapter(TorchRegBase):
  model_cls = models.TCNRegressor

  def build_model(self, in_dim: int, p: pp.TcnParams, device: torch.device):
    kwargs = asdict(p)
    return self.model_cls(in_dim=in_dim, **kwargs).to(device)


@register_model('random_forest', params_cls=pp.RfParams)
class RandomForestAdapter(SklearnRegBase):
  model_cls = RandomForestRegressor


@register_model('svr', params_cls=pp.SVRParams)
class SVRAdapter(SklearnRegBase):
  model_cls = SVR


@register_model('xgboost', params_cls=pp.XgbParams)
class XGBoostAdapter(SklearnRegBase):
  model_cls = None 

  def build_model(self, p: pp.XgbParams):
    if xgb is None:
      raise ImportError('xgboost is not installed. Please install it with "pip install xgboost".')
    kwargs = asdict(p).copy()
    kwargs.pop('model', None)
    return xgb.XGBRegressor(objective='reg:squarederror', **kwargs)


def train_regression(z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray, y_test: np.ndarray, config: dict,
                     tracking: dict, train_logger: TrainLogger) -> dict[str, Any]:
  ''' Train and evaluate a regression model based on the provided configuration. '''
  plugins = config.get('plugins', [])
  only_plugins = bool(config.get('plugins_only', False))
  discover_filter = config.get('plugin_filter', None)
  if not discover_filter and isinstance(tracking, dict):
    ds = (tracking.get('dataset') or {}).get('name', '')
    ver = str(tracking.get('version', '')).replace('.', '_')
    if ds and ver:
      discover_filter = f'{ds}_{ver}'
  if isinstance(plugins, (list, tuple)) and plugins:
    _load_reg_plugins(extra_modules=list(plugins), discover_filter=discover_filter, scan_namespace=not only_plugins)
  else:
    _load_reg_plugins(discover_filter=discover_filter, scan_namespace=True)
  name = str(config.get('model', 'mlp_torch'))
  if name not in _REG_REGISTRY:
    raise ValueError(f'unknown regression model_name={name}. available={list(_REG_REGISTRY)}')
  adapter = _REG_REGISTRY[name]()
  params = config.get('models', {}).get(name, {})
  params['criterion'] = config.get('criterion', {})
  if not params:
    logger.warning(f'no params found for model {name}, using defaults')
  logger.info(f'train regression model={name}')
  return adapter.fit_eval(z_train=z_train,
                          y_train=y_train,
                          z_test=z_test,
                          y_test=y_test,
                          params=params,
                          train_logger=train_logger,
                          tracking=tracking,
                          cfg=config)

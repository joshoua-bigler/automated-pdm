import torch
from dataclasses import dataclass, asdict, field
# local
from rul_lib.utils.dict_utils import flatten_conditional


def _to_int(v, default: int) -> int:
  try:
    return int(v)
  except Exception:
    return default


def _to_float(v, default: float) -> float:
  try:
    return float(v)
  except Exception:
    return default


def _to_int_list(v, default: list[int]) -> list[int]:
  if v is None:
    return list(default)
  if isinstance(v, (list, tuple)):
    try:
      return [int(x) for x in v]
    except Exception:
      return list(default)
  try:
    return [int(x.strip()) for x in str(v).split(',') if x.strip()]
  except Exception:
    return list(default)


def pick_device(s):
  return 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class MlpParams:
  hidden: int = 64
  dropout: float = 0.0
  learning_rate: float = 1e-3
  weight_decay: float = 1e-4
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  scheduler: str = 'steplr'  # 'none' | 'steplr' | 'cosine'
  step_size: int = 20
  gamma: float = 0.5
  device: str = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 'auto' | 'cpu' | 'cuda'
  model: str = 'mlp_torch'  # for naming only

  @classmethod
  def from_dict(cls, d: dict) -> 'MlpParams':
    d = {k.lower(): v for k, v in d.items()} if d else {}
    if 'lr' in d and 'learning_rate' not in d:
      d['learning_rate'] = d.pop('lr')
    p = cls()
    p.hidden = _to_int(d.get('hidden', p.hidden), p.hidden)
    p.dropout = max(0.0, min(0.95, _to_float(d.get('dropout', p.dropout), p.dropout)))
    p.learning_rate = max(1e-6, _to_float(d.get('learning_rate', p.learning_rate), p.learning_rate))
    p.weight_decay = max(0.0, _to_float(d.get('weight_decay', p.weight_decay), p.weight_decay))
    p.batch_size = max(1, _to_int(d.get('batch_size', p.batch_size), p.batch_size))
    p.epochs = max(1, _to_int(d.get('epochs', p.epochs), p.epochs))
    p.patience = max(1, _to_int(d.get('patience', p.patience), p.patience))
    p.scheduler = str(d.get('scheduler', p.scheduler)).lower()
    p.step_size = max(1, _to_int(d.get('step_size', p.step_size), p.step_size))
    p.gamma = max(0.0, min(1.0, _to_float(d.get('gamma', p.gamma), p.gamma)))
    p.device = pick_device(d.get('device', p.device))
    p.model = str(d.get('model', p.model))
    return p


@dataclass
class AdvMlpParams:
  # model hyperparameters
  width: int = 128
  depth: int = 3
  dropout: float = 0.1
  stoch_depth: float = 0.0
  # trainer hyperparameters
  batch_size: int = 128
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  epochs: int = 200
  patience: int = 20
  scheduler: str = 'none'  # 'none' | 'steplr' | 'cosine'
  step_size: int = 20  # StepLR
  gamma: float = 0.5  # StepLR
  t_max: int = 50  # CosineAnnealingLR

  @classmethod
  def from_dict(cls, d: dict) -> 'AdvMlpParams':
    d = flatten_conditional(d or {})
    return cls(
        width=_to_int(d.get('width', cls.width), cls.width),
        depth=_to_int(d.get('depth', cls.depth), cls.depth),
        dropout=_to_float(d.get('dropout', cls.dropout), cls.dropout),
        stoch_depth=_to_float(d.get('stoch_depth', cls.stoch_depth), cls.stoch_depth),
        batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
        learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
        weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
        epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
        patience=_to_int(d.get('patience', cls.patience), cls.patience),
        scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
        step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
        gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
        t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max),
    )

  def as_dict(self) -> dict:
    return asdict(self)


@dataclass
class TcnParams:
  channels: list[int] = field(default_factory=lambda: [64, 32, 16])
  kernel_size: int = 5
  dropout: float = 0.1
  hidden_mlp: int = 64
  # trainer hyperparameters
  batch_size: int = 64
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  epochs: int = 200
  patience: int = 20
  scheduler: str = 'none'  # 'none' | 'steplr' | 'cosine'
  step_size: int = 20  # StepLR
  gamma: float = 0.5  # StepLR
  t_max: int = 50  # CosineAnnealingLR

  @classmethod
  def from_dict(cls, d: dict) -> 'TcnParams':
    d = flatten_conditional(d or {})
    default_channels = cls().channels
    channels = _to_int_list(d.get('channels', default_channels), default_channels)
    return cls(
        channels=channels,
        kernel_size=_to_int(d.get('kernel_size', cls.kernel_size), cls.kernel_size),
        dropout=max(0.0, min(0.95, _to_float(d.get('dropout', cls.dropout), cls.dropout))),
        hidden_mlp=_to_int(d.get('hidden_mlp', cls.hidden_mlp), cls.hidden_mlp),
        batch_size=max(1, _to_int(d.get('batch_size', cls.batch_size), cls.batch_size)),
        learning_rate=max(1e-6, _to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate)),
        weight_decay=max(0.0, _to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay)),
        epochs=max(1, _to_int(d.get('epochs', cls.epochs), cls.epochs)),
        patience=max(1, _to_int(d.get('patience', cls.patience), cls.patience)),
        scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
        step_size=max(1, _to_int(d.get('step_size', cls.step_size), cls.step_size)),
        gamma=max(0.0, min(1.0, _to_float(d.get('gamma', cls.gamma), cls.gamma))),
        t_max=max(1, _to_int(d.get('t_max', cls.t_max), cls.t_max)),
    )

  def as_dict(self) -> dict:
    return asdict(self)


@dataclass
class RfParams:
  n_estimators: int = 200
  max_depth: int | None = None
  min_samples_split: int = 2
  min_samples_leaf: int = 1
  max_features: str | int | float | None = 'sqrt'
  random_state: int = 42
  n_jobs: int = -1
  model: str = 'random_forest'

  @classmethod
  def from_dict(cls, d: dict) -> 'RfParams':
    d = {k.lower(): v for k, v in (d or {}).items()}

    def _to_int(v, default):
      try:
        return int(v)
      except:
        return default

    p = cls()
    p.n_estimators = max(1, _to_int(d.get('n_estimators', p.n_estimators), p.n_estimators))
    p.max_depth = None if d.get('max_depth', p.max_depth) in (None, 'none') else _to_int(
        d.get('max_depth', p.max_depth), p.max_depth or 0)
    p.min_samples_split = max(2, _to_int(d.get('min_samples_split', p.min_samples_split), p.min_samples_split))
    p.min_samples_leaf = max(1, _to_int(d.get('min_samples_leaf', p.min_samples_leaf), p.min_samples_leaf))
    p.max_features = d.get('max_features', p.max_features)
    p.random_state = _to_int(d.get('random_state', p.random_state), p.random_state)
    p.n_jobs = _to_int(d.get('n_jobs', p.n_jobs), p.n_jobs)
    p.model = str(d.get('model', p.model))
    return p


@dataclass
class SVRParams:
  kernel: str = 'rbf'
  C: float = 1.0
  epsilon: float = 0.1
  gamma: str | float = 'scale'
  degree: int = 3
  coef0: float = 0.0
  tol: float = 1e-3
  shrinking: bool = True
  max_iter: int = -1
  cache_size: float = 200.0
  model: str = 'svr'

  @classmethod
  def from_dict(cls, d: dict) -> 'SVRParams':
    d = flatten_conditional(d or {})
    p = cls()

    def get_float(name, default):
      v = d.get(name, default)
      try:
        return float(v)
      except Exception:
        return default

    def get_int(name, default):
      v = d.get(name, default)
      try:
        return int(v)
      except Exception:
        return default

    p.kernel = str(d.get('kernel', p.kernel)).lower()
    p.C = get_float('C', get_float('c', p.C))
    p.epsilon = get_float('epsilon', p.epsilon)
    p.coef0 = get_float('coef0', p.coef0)
    p.tol = get_float('tol', p.tol)
    p.cache_size = get_float('cache_size', p.cache_size)
    g = d.get('gamma', p.gamma)
    try:
      p.gamma = float(g)
    except Exception:
      p.gamma = g
    p.degree = get_int('degree', p.degree)
    p.max_iter = get_int('max_iter', p.max_iter)
    p.shrinking = bool(d.get('shrinking', p.shrinking))
    p.model = str(d.get('model', p.model))
    return p


@dataclass
class XgbParams:
  n_estimators: int = 300
  max_depth: int = 6
  learning_rate: float = 0.05  # eta
  subsample: float = 0.8
  colsample_bytree: float = 0.8
  reg_lambda: float = 1.0
  reg_alpha: float = 0.0
  min_child_weight: float = 1.0
  gamma: float = 0.0
  n_jobs: int = -1
  random_state: int = 42
  tree_method: str = 'auto'  # will auto-fix below
  model: str = 'xgboost'
  device: str = 'cpu'  # 'cuda' or 'cpu' for xgboost>=2

  @classmethod
  def from_dict(cls, d: dict) -> 'XgbParams':
    d = flatten_conditional(d or {})
    p = cls()

    def get_int(name, default):
      v = d.get(name, default)
      try:
        return int(v)
      except Exception:
        return default

    def get_float(name, default):
      v = d.get(name, default)
      try:
        return float(v)
      except Exception:
        return default

    p.n_estimators = max(1, get_int('n_estimators', p.n_estimators))
    p.max_depth = max(1, get_int('max_depth', p.max_depth))
    p.learning_rate = max(1e-4, get_float('learning_rate', d.get('eta', p.learning_rate)))
    p.subsample = get_float('subsample', p.subsample)
    p.colsample_bytree = get_float('colsample_bytree', p.colsample_bytree)
    p.reg_lambda = get_float('reg_lambda', p.reg_lambda)
    p.reg_alpha = get_float('reg_alpha', p.reg_alpha)
    p.min_child_weight = get_float('min_child_weight', p.min_child_weight)
    p.gamma = get_float('gamma', p.gamma)
    p.n_jobs = get_int('n_jobs', p.n_jobs)
    p.random_state = get_int('random_state', p.random_state)
    p.tree_method = str(d.get('tree_method', p.tree_method))
    p.model = str(d.get('model', p.model))
    device = str(d.get('device', p.device)).lower()
    if device in ('gpu', 'cuda'):
      p.device = 'cuda'
    elif device in ('cpu',):
      p.device = 'cpu'
    else:
      p.device = 'cuda' 
    if p.device == 'cuda' and p.tree_method == 'auto':
      p.tree_method = 'hist'  
    return p

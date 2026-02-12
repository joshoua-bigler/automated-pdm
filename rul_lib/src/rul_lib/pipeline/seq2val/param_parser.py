from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any
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


def _to_opt_float(v, default: float | None) -> float | None:
  if v is None:
    return default
  s = str(v).strip().lower()
  if s in {"none", "null", ""}:
    return None
  try:
    return float(v)
  except Exception:
    return default


def _to_bool(v, default: bool) -> bool:
  if isinstance(v, bool):
    return v
  s = str(v).strip().lower()
  if s in {'true', '1', 'yes'}:
    return True
  if s in {'false', '0', 'no'}:
    return False
  return default


def _to_int_list(v, default: list[int]) -> list[int]:
  if v is None:
    return list(default) if default is not None else None
  if isinstance(v, (list, tuple)):
    try:
      return [int(x) for x in v]
    except Exception:
      return list(default) if default is not None else None
  try:
    return [int(x.strip()) for x in str(v).split(',') if x.strip()]
  except Exception:
    return list(default) if default is not None else None


class BaseParams(ABC):
  ''' Base class for model/trainer hyperparameters. '''

  @property
  @abstractmethod
  def learning_rate(self) -> float:
    ...

  @property
  @abstractmethod
  def weight_decay(self) -> float:
    ...

  @property
  @abstractmethod
  def batch_size(self) -> int:
    ...

  @property
  @abstractmethod
  def epochs(self) -> int:
    ...

  @property
  @abstractmethod
  def patience(self) -> int:
    ...

  @abstractmethod
  def from_dict(self, d: dict[str, Any]) -> dict[str, Any]:
    ...

  def as_dict(self) -> dict[str, Any]:
    return asdict(self)


@dataclass
class CNNParams:
  kernel_size: int = 3
  channels: int = 32
  dropout: float = 0.0
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'  # 'none' | 'steplr' | 'cosine'
  step_size: int = 20  # StepLR
  gamma: float = 0.5  # StepLR
  t_max: int = 50  # CosineAnnealingLR

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'CNNParams':
    d = flatten_conditional(d or {})
    return cls(
        kernel_size=_to_int(d.get('kernel_size', cls.kernel_size), cls.kernel_size),
        channels=_to_int(d.get('channels', cls.channels), cls.channels),
        dropout=_to_float(d.get('dropout', cls.dropout), cls.dropout),
        batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
        epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
        patience=_to_int(d.get('patience', cls.patience), cls.patience),
        learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
        weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
        scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
        step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
        gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
        t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max),
    )


@dataclass
class LSTMParams:
  hidden: int = 128
  num_layers: int = 2
  dropout: float = 0.1
  bidirectional: bool = True
  fc_hidden: int = 64
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'
  step_size: int = 20
  gamma: float = 0.5
  t_max: int = 50

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'LSTMParams':
    d = flatten_conditional(d or {})
    return cls(
        hidden=_to_int(d.get('hidden', cls.hidden), cls.hidden),
        fc_hidden=_to_int(d.get('fc_hidden', cls.fc_hidden), cls.fc_hidden),
        num_layers=_to_int(d.get('num_layers', cls.num_layers), cls.num_layers),
        dropout=_to_float(d.get('dropout', cls.dropout), cls.dropout),
        bidirectional=_to_bool(d.get('bidirectional', cls.bidirectional), cls.bidirectional),
        batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
        epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
        patience=_to_int(d.get('patience', cls.patience), cls.patience),
        learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
        weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
        scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
        step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
        gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
        t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max),
    )


@dataclass
class GRUParams:
  hidden: int = 128
  num_layers: int = 2
  dropout: float = 0.1
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'
  step_size: int = 20
  gamma: float = 0.5
  t_max: int = 50

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'GRUParams':
    d = flatten_conditional(d or {})
    return cls(
        hidden=_to_int(d.get('hidden', cls.hidden), cls.hidden),
        num_layers=_to_int(d.get('num_layers', cls.num_layers), cls.num_layers),
        dropout=_to_float(d.get('dropout', cls.dropout), cls.dropout),
        batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
        epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
        patience=_to_int(d.get('patience', cls.patience), cls.patience),
        learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
        weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
        scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
        step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
        gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
        t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max),
    )


@dataclass
class TCNParams:
  channels: tuple[int] = field(default_factory=lambda: (32, 64))
  kernel_size: int = 3
  dropout: float = 0.1
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'
  step_size: int = 20
  gamma: float = 0.5
  t_max: int = 50
  dilations: list[int] | None = None

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'TCNParams':
    d = flatten_conditional(d or {})
    return cls(channels=_to_int_list(d.get('channels',
                                           cls().channels), (32, 64)),
               kernel_size=_to_int(d.get('kernel_size', cls.kernel_size), cls.kernel_size),
               dropout=_to_float(d.get('dropout', cls.dropout), cls.dropout),
               batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
               epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
               patience=_to_int(d.get('patience', cls.patience), cls.patience),
               learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
               weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
               scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
               step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
               gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
               t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max),
               dilations=_to_int_list(d.get('dilations', cls.dilations), None))


@dataclass
class GRUTCNParams:
  tcn_channels: tuple[int] = field(default_factory=lambda: (32, 64))
  tcn_kernel_size: int = 3
  tcn_dropout: float = 0.1
  gru_hidden: int = 128
  gru_layers: int = 2
  gru_dropout: float = 0.1
  bidirectional: bool = False
  dilations: list[int] | None = None
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'
  step_size: int = 20
  gamma: float = 0.5
  t_max: int = 50

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'GRUTCNParams':
    d = flatten_conditional(d or {})
    channels = _to_int_list(d.get('tcn_channels', cls().tcn_channels), (32, 64))
    dils = _to_int_list(d.get('dilations', cls.dilations), None)
    return cls(tcn_channels=tuple(channels) if channels is not None else cls().tcn_channels,
               tcn_kernel_size=_to_int(d.get('tcn_kernel_size', cls.tcn_kernel_size), cls.tcn_kernel_size),
               tcn_dropout=_to_float(d.get('tcn_dropout', cls.tcn_dropout), cls.tcn_dropout),
               gru_hidden=_to_int(d.get('gru_hidden', cls.gru_hidden), cls.gru_hidden),
               gru_layers=_to_int(d.get('gru_layers', cls.gru_layers), cls.gru_layers),
               gru_dropout=_to_float(d.get('gru_dropout', cls.gru_dropout), cls.gru_dropout),
               bidirectional=_to_bool(d.get('bidirectional', cls.bidirectional), cls.bidirectional),
               batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
               epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
               patience=_to_int(d.get('patience', cls.patience), cls.patience),
               learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
               weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
               scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
               step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
               gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
               t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max),
               dilations=dils)


@dataclass
class CNNLSTMParams:
  conv_channels: int = 32
  kernel_size: int = 3
  hidden: int = 128
  num_layers: int = 2
  dropout: float = 0.1
  bidirectional: bool = False
  lstm_hidden: int = 64
  lstm_layers: int = 1
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'
  step_size: int = 20
  gamma: float = 0.5
  t_max: int = 50

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'CNNLSTMParams':
    d = flatten_conditional(d or {})
    return cls(
        conv_channels=_to_int(d.get('conv_channels', cls.conv_channels), cls.conv_channels),
        kernel_size=_to_int(d.get('kernel_size', cls.kernel_size), cls.kernel_size),
        hidden=_to_int(d.get('hidden', cls.hidden), cls.hidden),
        num_layers=_to_int(d.get('num_layers', cls.num_layers), cls.num_layers),
        dropout=_to_float(d.get('dropout', cls.dropout), cls.dropout),
        bidirectional=_to_bool(d.get('bidirectional', cls.bidirectional), cls.bidirectional),
        batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
        epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
        patience=_to_int(d.get('patience', cls.patience), cls.patience),
        learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
        weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
        scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
        step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
        gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
        t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max),
        lstm_hidden=_to_int(d.get('lstm_hidden', cls.lstm_hidden), cls.lstm_hidden),
        lstm_layers=_to_int(d.get('lstm_layers', cls.lstm_layers), cls.lstm_layers),
    )


@dataclass
class AdvCNNParams:
  kernel_size: int = 5
  channels: int = 64
  blocks: int = 3
  dropout: float = 0.1
  clip: float | None = None
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'
  step_size: int = 20
  gamma: float = 0.5
  t_max: int = 50

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'AdvCNNParams':
    d = flatten_conditional(d or {})
    return cls(kernel_size=_to_int(d.get('kernel_size', cls.kernel_size), cls.kernel_size),
               channels=_to_int(d.get('channels', cls.channels), cls.channels),
               blocks=_to_int(d.get('blocks', cls.blocks), cls.blocks),
               dropout=_to_float(d.get('dropout', cls.dropout), cls.dropout),
               clip=_to_opt_float(d.get('clip', cls.clip), cls.clip),
               batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
               epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
               patience=_to_int(d.get('patience', cls.patience), cls.patience),
               learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
               weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
               scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
               step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
               gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
               t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max))


@dataclass
class AdvTCNParams:
  # model hyperparameters
  channels: tuple[int, ...] = field(default_factory=lambda: (16, 16, 32, 32, 64, 64))
  kernel_size: int = 7
  dropout: float = 0.2
  dilations: list[int] | None = None
  stem_stride: int = 4
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'
  step_size: int = 20
  gamma: float = 0.5
  t_max: int = 50

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'AdvTCNParams':
    d = flatten_conditional(d or {})
    ch = _to_int_list(d.get('channels', list(cls().channels)), list(cls().channels))
    ch_t = tuple(ch) if ch is not None else cls().channels
    dils_raw = d.get('dilations', cls.dilations)
    dils = _to_int_list(dils_raw, None) if dils_raw is not None else None
    # allow a single dilation cycle to be repeated to match channels
    if dils is not None and len(dils) > 0 and len(dils) != len(ch_t):
      if len(ch_t) % len(dils) == 0:
        rep = len(ch_t) // len(dils)
        dils = list(dils) * rep
      else:
        raise ValueError(f'len(dilations)={len(dils)} must match len(channels)={len(ch_t)} or divide it evenly')
    return cls(channels=ch_t,
               kernel_size=_to_int(d.get('kernel_size', cls.kernel_size), cls.kernel_size),
               dropout=_to_float(d.get('dropout', cls.dropout), cls.dropout),
               dilations=dils,
               stem_stride=_to_int(d.get('stem_stride', cls.stem_stride), cls.stem_stride),
               batch_size=_to_int(d.get('batch_size', cls.batch_size), cls.batch_size),
               epochs=_to_int(d.get('epochs', cls.epochs), cls.epochs),
               patience=_to_int(d.get('patience', cls.patience), cls.patience),
               learning_rate=_to_float(d.get('learning_rate', cls.learning_rate), cls.learning_rate),
               weight_decay=_to_float(d.get('weight_decay', cls.weight_decay), cls.weight_decay),
               scheduler=str(d.get('scheduler', cls.scheduler)).lower(),
               step_size=_to_int(d.get('step_size', cls.step_size), cls.step_size),
               gamma=_to_float(d.get('gamma', cls.gamma), cls.gamma),
               t_max=_to_int(d.get('t_max', cls.t_max), cls.t_max))


@dataclass
class GRUAttnParams:
  # model hyperparameters
  hidden: int = 128
  num_layers: int = 2
  dropout: float = 0.1
  bidirectional: bool = False
  # attention hyperparameters (temporal attention pooling)
  attn_hidden: int = 128
  attn_heads: int = 1
  attn_dropout: float = 0.0
  # trainer hyperparameters
  batch_size: int = 64
  epochs: int = 200
  patience: int = 20
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  scheduler: str = 'none'
  step_size: int = 20
  gamma: float = 0.5
  t_max: int = 50

  @classmethod
  def from_dict(cls, d: dict[str, Any] | None) -> 'GRUAttnParams':
    d = flatten_conditional(d or {})
    p = cls()
    return cls(hidden=_to_int(d.get('hidden', p.hidden), p.hidden),
               num_layers=_to_int(d.get('num_layers', p.num_layers), p.num_layers),
               dropout=_to_float(d.get('dropout', p.dropout), p.dropout),
               bidirectional=_to_bool(d.get('bidirectional', p.bidirectional), p.bidirectional),
               attn_hidden=_to_int(d.get('attn_hidden', p.attn_hidden), p.attn_hidden),
               attn_heads=max(1, _to_int(d.get('attn_heads', p.attn_heads), p.attn_heads)),
               attn_dropout=_to_float(d.get('attn_dropout', p.attn_dropout), p.attn_dropout),
               batch_size=_to_int(d.get('batch_size', p.batch_size), p.batch_size),
               epochs=_to_int(d.get('epochs', p.epochs), p.epochs),
               patience=_to_int(d.get('patience', p.patience), p.patience),
               learning_rate=_to_float(d.get('learning_rate', p.learning_rate), p.learning_rate),
               weight_decay=_to_float(d.get('weight_decay', p.weight_decay), p.weight_decay),
               scheduler=str(d.get('scheduler', p.scheduler)).lower(),
               step_size=_to_int(d.get('step_size', p.step_size), p.step_size),
               gamma=_to_float(d.get('gamma', p.gamma), p.gamma),
               t_max=_to_int(d.get('t_max', p.t_max), p.t_max))

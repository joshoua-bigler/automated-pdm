import torch
from dataclasses import dataclass


@dataclass
class AEParams:
  latent_dim: int = 8
  hidden_ch: int = 64
  epochs: int = 100
  batch_size: int = 64
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  patience: int = 10
  scheduler: str = 'none'  # 'none' | 'steplr' | 'cosine'
  step_size: int = 20
  gamma: float = 0.5
  random_seed: int = 42
  device: str = 'cuda'  # 'auto' | 'cpu' | 'cuda'
  model_name: str = 'cnn-ae'

  @classmethod
  def from_dict(cls, d: dict) -> 'AEParams':
    d = d or {}
    p = cls()
    p.latent_dim = int(d.get('latent_dim', p.latent_dim))
    p.hidden_ch = int(d.get('hidden_ch', p.hidden_ch))
    p.epochs = int(d.get('epochs', p.epochs))
    p.batch_size = max(1, int(d.get('batch_size', p.batch_size)))
    p.learning_rate = float(d.get('learning_rate', p.learning_rate))
    p.weight_decay = float(d.get('weight_decay', p.weight_decay))
    p.patience = int(d.get('patience', p.patience))
    p.scheduler = str(d.get('scheduler', p.scheduler)).lower()
    p.step_size = int(d.get('step_size', p.step_size))
    p.gamma = float(d.get('gamma', p.gamma))
    p.random_seed = int(d.get('random_seed', p.random_seed))
    p.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p.model_name = str(d.get('model_name', p.model_name))
    return p


@dataclass
class DilatedCnnAEParams:
  # model architecture
  in_ch: int = 2
  base_ch: int = 32
  latent_dim: int = 32
  n_down: int = 3
  k_down: int = 15
  k_dil: int = 3
  dilations: tuple = (1, 2, 4, 8)
  dropout: float = 0.0
  out_len_hint: int | None = None
  # training
  epochs: int = 100
  batch_size: int = 64
  learning_rate: float = 1e-3
  weight_decay: float = 0.0
  patience: int = 10
  # scheduler
  scheduler: str = 'none'  # 'none' | 'steplr' | 'cosine'
  step_size: int = 20
  gamma: float = 0.5
  # misc
  random_seed: int = 42
  device: str = 'auto'  # 'auto' | 'cpu' | 'cuda'
  model_name: str = 'dilated-cnn-ae'

  @classmethod
  def from_dict(cls, d: dict) -> 'DilatedCnnAEParams':
    d = d or {}
    p = cls()
    # architecture
    p.in_ch = int(d.get('in_ch', p.in_ch))
    p.base_ch = int(d.get('base_ch', p.base_ch))
    p.latent_dim = int(d.get('latent_dim', p.latent_dim))
    p.n_down = int(d.get('n_down', p.n_down))
    p.k_down = int(d.get('k_down', p.k_down))
    p.k_dil = int(d.get('k_dil', p.k_dil))
    p.dilations = tuple(d.get('dilations', p.dilations))
    p.dropout = float(d.get('dropout', p.dropout))
    p.out_len_hint = d.get('out_len_hint', p.out_len_hint)
    # training
    p.epochs = int(d.get('epochs', p.epochs))
    p.batch_size = max(1, int(d.get('batch_size', p.batch_size)))
    p.learning_rate = float(d.get('learning_rate', p.learning_rate))
    p.weight_decay = float(d.get('weight_decay', p.weight_decay))
    p.patience = int(d.get('patience', p.patience))
    # scheduler
    p.scheduler = str(d.get('scheduler', p.scheduler)).lower()
    p.step_size = int(d.get('step_size', p.step_size))
    p.gamma = float(d.get('gamma', p.gamma))
    # misc
    p.random_seed = int(d.get('random_seed', p.random_seed))
    dev = str(d.get('device', p.device)).lower()
    if dev == 'auto':
      p.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
      p.device = dev
    p.model_name = str(d.get('model_name', p.model_name))
    return p


@dataclass
class TsfreshParams:
  fc_params: str = 'minimal'
  n_jobs: int = 72
  max_features: int = 128
  normalize: bool = False

  @classmethod
  def from_dict(cls, d: dict | None) -> 'TsfreshParams':
    d = d or {}
    p = cls()
    p.fc_params = str(d.get('fc_params', p.fc_params)).lower()
    p.max_features = int(d.get('max_features', p.max_features))
    p.n_jobs = int(d.get('n_jobs', p.n_jobs))
    norm = d.get('normalize', p.normalize)
    if isinstance(norm, str):
      norm = norm.strip().lower() in ['1', 'true', 'yes', 'y', 'on']
    p.normalize = bool(norm)
    return p

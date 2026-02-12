import torch
from dataclasses import is_dataclass, asdict
from typing import Any, Mapping

AVAILABLE_SCHEDULERS: list[str] = [
    'none',
    'steplr',
    'cosine',
]


def build_scheduler(optimizer: torch.optim.Optimizer,
                    cfg: Mapping[str, Any] | Any | None = None) -> torch.optim.lr_scheduler._LRScheduler | None:
  ''' Build LR scheduler from config. '''
  if cfg is None:
    return None
  if is_dataclass(cfg):
    cfg = asdict(cfg)
  elif isinstance(cfg, Mapping):
    cfg = dict(cfg)
  else:
    return None
  name = str(cfg.get('scheduler', 'none')).lower()
  if name == 'steplr':
    step_size = int(cfg.get('step_size', 20))
    gamma = float(cfg.get('gamma', 0.5))
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
  if name == 'cosine':
    t_max = int(cfg.get('t_max', 50))
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
  return None

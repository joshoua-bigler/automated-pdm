import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ConfigCriterion:
  kind: str = 'mse'
  lam: float = 0.0
  reduction: str = 'mean'
  # nasa
  clip: float | None = 10.0
  k_late: float = 10.0
  k_early: float = 13.0
  huber_delta: float = 10.0
  tau: float = 0.5

  @classmethod
  def from_dict(cls, d: dict | None = None) -> 'ConfigCriterion':
    if not d:
      return cls()
    d = {str(k).lower(): v for k, v in d.items()}
    p = cls()
    # kind with aliases + safe fallback
    kind = str(d.get('kind', p.kind)).strip().lower()
    aliases = {
        'mse': 'mse',
        'l2': 'mse',
        'nasa': 'nasa',
        'huber': 'huber',
        'smoothl1': 'huber',
        'smooth_l1': 'huber',
    }
    p.kind = aliases.get(kind, 'mse')
    # reduction safe fallback
    red = str(d.get('reduction', p.reduction)).strip().lower()
    p.reduction = red if red in ('mean', 'sum', 'none') else 'mean'
    p.lam = max(0.0, float(d.get('lam', p.lam)))
    clip = d.get('clip', p.clip)
    p.clip = float(clip) if clip is not None else None
    p.k_late = max(1.0, float(d.get('k_late', p.k_late)))
    p.k_early = max(1.0, float(d.get('k_early', p.k_early)))
    p.huber_delta = max(1e-6, float(d.get('huber_delta', p.huber_delta)))
    p.tau = float(d.get('tau', p.tau))
    return p


class QuantileLoss(nn.Module):
  ''' Pinball / quantile loss. tau in (0,1). '''

  def __init__(self, tau: float = 0.2, reduction: str = 'mean'):
    super().__init__()
    assert 0.0 < tau < 1.0
    self.tau = float(tau)
    self.reduction = reduction

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    e = y_true - y_pred
    loss = torch.maximum(self.tau * e, (self.tau - 1.0) * e)
    if self.reduction == 'mean':
      return loss.mean()
    if self.reduction == 'sum':
      return loss.sum()
    return loss


class NasaScoreLoss(nn.Module):
  ''' NASA score loss as defined in the C-MAPSS dataset paper. '''

  def __init__(self, k_late=10.0, k_early=13.0, clip=10.0, reduction='mean'):
    super().__init__()
    self.k_late = float(k_late)
    self.k_early = float(k_early)
    self.clip = float(clip) if clip is not None else None
    self.reduction = reduction if reduction in ('mean', 'sum', 'none') else 'mean'

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    d = y_pred - y_true
    if self.clip is not None:
      d = torch.clamp(d, -self.clip, self.clip)
    pos = torch.expm1(torch.clamp(d, min=0.0) / self.k_late)
    neg = torch.expm1(torch.clamp(-d, min=0.0) / self.k_early)
    loss = torch.where(d >= 0, pos, neg)
    if self.reduction == 'mean':
      return loss.mean()
    if self.reduction == 'sum':
      return loss.sum()
    return loss


class MseNasaLoss(nn.Module):
  ''' Combined Mean Squared Error and NASA score loss. '''

  def __init__(self, config: ConfigCriterion):
    super().__init__()
    self.lam = float(config.lam)
    self.mse = nn.MSELoss(reduction=config.reduction)
    self.nasa = NasaScoreLoss(k_late=config.k_late,
                              k_early=config.k_early,
                              clip=config.clip,
                              reduction=config.reduction)

  def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    mse = self.mse(y_pred, y_true)
    nasa = self.nasa(y_pred, y_true)
    if mse.ndim > 0:
      mse = mse.mean()
    if nasa.ndim > 0:
      nasa = nasa.mean()
    return mse + self.lam * nasa


def make_criterion(config: ConfigCriterion) -> nn.Module:
  k = (config.kind or 'mse').strip().lower()
  if k in ('quantile', 'pinball'):
    return QuantileLoss(tau=0.2, reduction=config.reduction)
  if k == 'nasa':
    return NasaScoreLoss(k_late=config.k_late, k_early=config.k_early, clip=config.clip, reduction=config.reduction)
  if k in ('mse+nasa', 'combo'):
    return MseNasaLoss(config=config)
  if k == 'huber':
    return nn.SmoothL1Loss(beta=float(config.huber_delta), reduction=config.reduction)
  return nn.MSELoss(reduction=config.reduction)


def compute_metrics(pred: torch.Tensor | np.ndarray,
                    y: torch.Tensor | np.ndarray,
                    config: ConfigCriterion,
                    eval: bool = False) -> dict:
  ''' Compute evaluation metrics based on predictions and ground truth. '''
  if isinstance(pred, np.ndarray):
    pred = torch.as_tensor(pred, dtype=torch.float32)
  if isinstance(y, np.ndarray):
    y = torch.as_tensor(y, dtype=torch.float32)
  pred = pred.detach().cpu().float()
  y = y.detach().cpu().float()
  diff = pred - y
  mse_t = (diff * diff).mean()
  mse = float(mse_t.item())
  rmse = float(np.sqrt(mse))
  kind = (config.kind or 'mse').strip().lower()
  if kind not in ('mse', 'nasa', 'mse+nasa', 'combo', 'huber'):
    kind = 'mse'
  metrics: dict[str, float] = {'mse': mse, 'rmse': rmse}
  if kind == 'huber':
    huber_loss = nn.SmoothL1Loss(beta=float(config.huber_delta), reduction='mean')
    huber = float(huber_loss(pred, y).item())
    metrics.update({'huber': huber, 'obj': huber})
  elif kind == 'mse':
    metrics.update({'obj': mse})
  else:
    nasa_loss = NasaScoreLoss(k_late=config.k_late, k_early=config.k_early, clip=config.clip, reduction='mean')
    nasa = float(nasa_loss(pred, y).item())
    if kind == 'nasa':
      metrics.update({'nasa': nasa, 'obj': nasa})
    else:
      obj = mse + float(config.lam) * nasa
      metrics.update({'nasa': nasa, 'obj': obj})
  if eval:
    return {f'val_{k}': v for k, v in metrics.items()}
  return metrics

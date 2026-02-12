import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import explained_variance_score
from typing import Any
# local
from rul_lib.pipeline.pre.normalize import invert_target


def nasa_score(y_pred: np.ndarray, y_true: np.ndarray, alpha_neg: float = 13.0, alpha_pos: float = 10.0) -> float:
  ''' NASA (PHM08) scoring function.

      h_j = y_pred - y_true
      s_j = exp(-h_j / alpha_neg) - 1, if h_j < 0
          = exp( h_j / alpha_pos) - 1, if h_j >= 0

      Final score = sum(s_j)
  '''
  h = y_pred - y_true
  s = np.where(h < 0, np.exp(-h / alpha_neg) - 1, np.exp(h / alpha_pos) - 1)
  return float(np.sum(s))


def r2_score_np(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> float:
  ''' Compute R^2 (coefficient of determination) in numpy.
      R2 = 1 - (SS_res / SS_tot)
  '''
  ss_res = np.sum((y_true - y_pred)**2)
  ss_tot = np.sum((y_true - np.mean(y_true))**2) + eps
  return 1.0 - ss_res / ss_tot


def _infer_device(model: nn.Module) -> torch.device:
  try:
    return next(model.parameters()).device
  except StopIteration:
    pass
  for attr in ("encoder", "regressor", "backbone", "net", "model"):
    sub = getattr(model, attr, None)
    if sub is not None and hasattr(sub, "parameters"):
      try:
        return next(sub.parameters()).device
      except StopIteration:
        continue
  return torch.device("cpu")


def eval_model(model: nn.Module,
               x_test: np.ndarray,
               y_test: np.ndarray,
               meta: dict,
               os_test: np.ndarray | None = None) -> dict[str, Any]:
  ''' Evaluate RUL model on test data. '''
  device = _infer_device(model)
  model.eval()
  x = torch.from_numpy(x_test).float().to(device)
  y_s = torch.from_numpy(y_test).float().to(device)
  use_os = getattr(model, 'use_os', False)
  # ---- forward pass ----
  with torch.no_grad():
    if use_os:
      if os_test is None:
        raise ValueError('Model expects OS features (use_os=True) but os_test=None.')
      os_t = torch.as_tensor(os_test, dtype=torch.float32, device=device)
      p_s = model(x, os=os_t).squeeze(-1).detach().cpu().numpy()
    else:
      p_s = model(x).squeeze(-1).detach().cpu().numpy()
  # ---- align target with multi-window ----
  m = getattr(model, 'm', 1)
  multi_window = getattr(model, 'multi_window', False)
  y_arr = y_s.detach().cpu().numpy()
  if multi_window and m > 1:
    y_arr = y_arr[m - 1:]
  # ---- invert scaling ----
  y_pred = invert_target(p_s, meta)
  y_true = invert_target(y_arr, meta)
  # ---- compute metrics ----
  errors = y_pred - y_true
  abs_err = np.abs(errors)
  rmse = float(np.sqrt(np.mean(errors**2)))
  mae = float(np.mean(abs_err))
  mse = float(np.mean(errors**2))
  bias = float(np.mean(errors))
  medae = float(np.median(abs_err))
  r2 = r2_score_np(y_pred, y_true)
  nasa = nasa_score(y_pred, y_true)
  ev = explained_variance_score(y_true, y_pred)
  return {
      'y_pred': y_pred,
      'y_true': y_true,
      'eval_metrics': {
          'rmse': rmse,
          'mae': mae,
          'mse': mse,
          'bias': bias,
          'medae': medae,
          'r2': r2,
          'nasa': nasa,
          'ev': ev
      }
  }

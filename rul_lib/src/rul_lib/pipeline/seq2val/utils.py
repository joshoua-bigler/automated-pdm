import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Any
# local
import rul_lib.pipeline.seq2val.param_parser as pp
from rul_lib.pipeline.parser.scheduler import build_scheduler


def to_numpy(x: Any) -> np.ndarray:
  if hasattr(x, 'to_numpy'):
    return x.to_numpy()
  return np.asarray(x)


def make_multiwindow_seq(x,
                         y,
                         m: int = 4,
                         flatten: bool = False,
                         as_sequence: bool = True) -> tuple[np.ndarray, np.ndarray]:
  ''' Build stacked multi-window sequences.

      Parameters
      ----------
      x : np.ndarray
          Shape (N, L, F) or (B, W, L, F). If 4D, (B, W) is merged into a single
          window index axis before stacking.
      y : np.ndarray
          Targets aligned with the *last* window of each stack.
      m : int
          Number of consecutive windows to stack.
      flatten : bool, default=False
          If True, each stacked sample is flattened to shape (m*L*F,). Use this
          for non-sequence models (MLP, XGBoost, etc.).
      as_sequence : bool, default=False
          If True, stacked windows are merged along the time axis and returned
          with shape (m*L, F), suitable for sequence models like TCN/LSTM.
          If both flatten and as_sequence are True, a ValueError is raised.

      Returns
      -------
      X : np.ndarray
          Stacked input samples. Shape depends on flags:
            - flatten=True:   (N_eff, m*L*F)
            - as_sequence=True: (N_eff, m*L, F)
            - else:           (N_eff, m, L, F)
      y_out : np.ndarray
          Targets aligned with the last window of each stack. Shape (N_eff,).
  '''
  if flatten and as_sequence:
    raise ValueError('flatten and as_sequence cannot both be True')

  # Case 1: x = (B, W, L, F) → merge B and W first
  if x.ndim == 4:
    b, w, l, f = x.shape
    x = x.reshape(b * w, l, f)
    y = y.reshape(b * w)
  # Now x = (N, L, F)
  n, l, f = x.shape
  X_list, y_list = [], []
  for i in range(m - 1, n):
    # windows: (m, L, F)
    windows = x[i - m + 1:i + 1]
    if flatten:
      X_i = windows.reshape(-1)  # (m*L*F,)
    elif as_sequence:
      X_i = windows.reshape(m * l, f)  # (m*L, F)
    else:
      X_i = windows  # (m, L, F)
    X_list.append(X_i)
    y_list.append(y[i])  # target = last window's target
  return np.array(X_list), np.array(y_list)


def build_seq_loaders(windowed: dict, batch_size: int, cfg: dict):
  ''' Build DataLoaders for sequence models with optional multi-window stacking.

      Parameters
      ----------
      windowed : dict
          Dictionary containing 'x_train', 'y_train', 'x_test', 'y_test'.
          x arrays may have shape (N, L, F) for single windows or (B, W, L, F)
          for multiple windows per unit.

      batch_size : int
          Batch size for the returned DataLoaders.

      m : int, default=4
          Number of consecutive windows to stack for each training sample.
          For input (N, L, F), this produces output of shape
          (N - m + 1, m, L, F) before any reshaping.

      Returns
      -------
      dl_tr : DataLoader
          Training DataLoader returning (x, y) batches.

      dl_va : DataLoader
          Validation DataLoader returning (x, y) batches.

      Notes
      -----
      - If x has shape (B, W, L, F), it is first flattened into (B*W, L, F)
        so windows are treated as a continuous sequence before multi-window stacking.
      - Sequence models typically reshape (m, L, F) into (m*L, F) before forward pass.
  '''
  apply = bool(cfg.get('apply', False))
  m = int(cfg.get('m', 0))
  x_tr = to_numpy(windowed['x_train']).astype(np.float32)
  y_tr = to_numpy(windowed['y_train']).astype(np.float32).reshape(-1)
  x_va = to_numpy(windowed['x_test']).astype(np.float32)
  y_va = to_numpy(windowed['y_test']).astype(np.float32).reshape(-1)
  # Build multi-window sequences
  if apply and m > 0:
    x_tr, y_tr = make_multiwindow_seq(x=x_tr, y=y_tr, m=m)
    x_va, y_va = make_multiwindow_seq(x=x_va, y=y_va, m=m)
  dl_tr = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                     batch_size=int(batch_size),
                     shuffle=True,
                     drop_last=False)
  dl_va = DataLoader(TensorDataset(torch.from_numpy(x_va), torch.from_numpy(y_va)),
                     batch_size=int(batch_size),
                     shuffle=False,
                     drop_last=False)
  return dl_tr, dl_va


def optimizer_and_sched(model: nn.Module, p_params: pp.BaseParams) -> tuple[torch.optim.Optimizer, Any]:
  ''' Build optimizer and scheduler for training. '''
  optimizer = torch.optim.Adam(model.parameters(), lr=p_params.learning_rate, weight_decay=p_params.weight_decay)
  scheduler = build_scheduler(optimizer=optimizer, cfg=p_params)
  return optimizer, scheduler


def eval_torch_model(model: nn.Module, dl_va: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
  ''' Evaluate a PyTorch model on validation data loader. '''
  model.eval()
  preds, ys = [], []
  with torch.no_grad():
    for xb, yb in dl_va:
      xb = xb.to(device)
      preds.append(model(xb).cpu())
      ys.append(yb.cpu())
  pred_val = torch.cat(preds).squeeze(-1).numpy()
  yv = torch.cat(ys).squeeze(-1).numpy()
  return pred_val, yv

import numpy as np
import torch
import torch.nn as nn
from typing import Any


def _flatten_latent(z: torch.Tensor) -> torch.Tensor:
  if z.dim() > 2:
    z = z.view(z.size(0), -1)
  return z


def _stack_multi_windows(z_flat: torch.Tensor, m: int) -> torch.Tensor:
  ''' PyTorch equivalent of the feature-part of make_multiwindow_latent.

      Input
      -----
      z_flat : (N, D) or higher → flattened to (N, D_flat)
      m      : window length

      Returns
      -------
      Z_out  : (N_eff, m*D_flat) with N_eff = N - m + 1
  '''
  z_flat = _flatten_latent(z_flat)
  if z_flat.dim() != 2:
    raise ValueError(f'z_flat must be 2D after flatten, got shape={z_flat.shape}')
  n, d = z_flat.shape
  if m <= 1:
    return z_flat
  if n < m:
    raise ValueError(f'n={n} < m={m} in multi-window stacking')
  # build sliding indices: for i in [m-1 .. n-1] we want rows [i-m+1 .. i]
  device = z_flat.device
  dtype = torch.long
  n_eff = n - m + 1
  start_idx = torch.arange(n_eff, device=device, dtype=dtype).unsqueeze(1)  # (N_eff, 1)
  offsets = torch.arange(m, device=device, dtype=dtype).unsqueeze(0)  # (1, m)
  idx = start_idx + offsets  # (N_eff, m)
  # gather windows: (N_eff, m, D)
  blocks = z_flat[idx]  # index on dim 0
  # flatten windows: (N_eff, m*D)
  Z_out = blocks.reshape(n_eff, m * d)
  return Z_out


class RulModel(nn.Module):
  ''' RUL model with encoder and regressor. '''

  def __init__(self,
               encoder: nn.Module,
               regressor: nn.Module | object,
               multi_window: bool = False,
               m: int = 1,
               use_os: bool = False):
    ''' encoder :
          Encoder model (e.g., autoencoder).
        regressor :
          Regressor model (e.g., torch nn.Module or sklearn model).
        multi_window :
          Whether sliding multi-window processing is used in latent space.
        m :
          Window length (number of consecutive latent vectors).
        use_os :
          If True, expects an additional OS vector per sample in forward(..., os=...).
    '''
    super().__init__()
    self.encoder = encoder
    self.regressor = regressor
    self.multi_window = bool(multi_window)
    self.m = int(m)
    self.use_os = bool(use_os)

  def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
    # x: (N, ...) → z: (N, D_flat)
    with torch.no_grad():
      enc_out = self.encoder(x)
      z = enc_out[1] if isinstance(enc_out, (tuple, list)) else enc_out
    return _flatten_latent(z)

  def _concat_os(self, z: torch.Tensor, os: torch.Tensor | np.ndarray | None) -> torch.Tensor:
    if os is None:
      raise ValueError('use_os=True but os is None in forward(...)')
    os_t = torch.as_tensor(os, dtype=z.dtype, device=z.device)
    if os_t.ndim == 1:
      os_t = os_t.unsqueeze(-1)  # (N, 1)
    if os_t.shape[0] != z.shape[0]:
      raise ValueError(f'os batch size {os_t.shape[0]} != latent batch size {z.shape[0]}')
    return torch.cat([z, os_t], dim=-1)  # (N, D_flat [+ os_dim])

  def forward(self, x: torch.Tensor, os: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
    z = self._encode_batch(x)  # (N, D_flat)
    if self.multi_window and self.m > 1:
      z = _stack_multi_windows(z, self.m)  # (N_eff, m * D_flat)
    if self.use_os:
      z = self._concat_os(z, os)  # (N_eff, m * D_flat + os_dim)
    # Torch regressor path
    if hasattr(self.regressor, 'forward') or hasattr(self.regressor, 'parameters'):
      return self.regressor(z)
    # Sklearn/tabular regressor path
    if hasattr(self.regressor, 'predict'):
      z_np = z.detach().cpu().numpy()
      pred = self.regressor.predict(z_np).astype(np.float32)
      if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
      return torch.from_numpy(pred).to(z.device)
    raise TypeError(f'unsupported regressor type: {type(self.regressor)}')


class TabularRulModel(torch.nn.Module):
  ''' RUL model with non-torch encoder and regressor. '''

  def __init__(self, encoder: Any, regressor: Any, multi_window: bool = False, m: int = 1, use_os: bool = False):
    super().__init__()
    self.encoder = encoder
    self.regressor = regressor
    self.multi_window = bool(multi_window)
    self.m = int(m)
    self.use_os = bool(use_os)

  def _get_regressor_device(self, x: torch.Tensor) -> torch.device:
    if hasattr(self.regressor, 'parameters'):
      try:
        p = next(self.regressor.parameters())
        return p.device
      except StopIteration:
        return x.device
    return x.device

  def _encode_batch(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
      z_np = self.encoder.predict(x)  # (N, D) numpy
    z = torch.as_tensor(z_np, dtype=torch.float32, device=device)
    return _flatten_latent(z)  # (N, D_flat)

  def _concat_os(self, z: torch.Tensor, os: torch.Tensor | np.ndarray, device: torch.device) -> torch.Tensor:
    if os is None:
      raise ValueError('use_os=True but os is None in forward(...)')
    os_t = torch.as_tensor(os, dtype=z.dtype, device=device)
    if os_t.ndim == 1:
      os_t = os_t.unsqueeze(-1)  # (batch, 1)
    if os_t.shape[0] != z.shape[0]:
      raise ValueError(f'os batch size {os_t.shape[0]} != z batch size {z.shape[0]}')
    return torch.cat([z, os_t], dim=-1)

  def forward(self, x: torch.Tensor, os: torch.Tensor | np.ndarray | None = None) -> torch.Tensor:
    reg_dev = self._get_regressor_device(x)
    # 1) encode with non-torch encoder
    z = self._encode_batch(x, device=reg_dev)  # (N, D_flat)
    # 2) prepare OS and align with multi-window if needed
    os_aligned = None
    if self.use_os:
      if os is None:
        raise ValueError('use_os=True but os is None in forward(...)')
      os_t = torch.as_tensor(os, dtype=z.dtype, device=reg_dev)
      # if OS still has original length N and we apply multi-window,
      # trim first m-1 entries so that OS refers to the window "end" index
      if self.multi_window and self.m > 1 and os_t.shape[0] == z.shape[0]:
        os_t = os_t[self.m - 1:]  # (N_eff, ...) like y[m-1:]
      os_aligned = os_t
    # 3) optionally stack multiple windows in latent space
    if self.multi_window and self.m > 1:
      z = _stack_multi_windows(z, self.m)  # (N_eff, m * D_flat)
    # 4) late-fuse OS per stacked sample
    if self.use_os:
      z = self._concat_os(z, os_aligned, device=reg_dev)  # (N_eff, m*D_flat + os_dim)
    # 5) Torch regressor path
    if hasattr(self.regressor, 'forward') or hasattr(self.regressor, 'parameters'):
      return self.regressor(z)
    # 6) Sklearn/tabular regressor path
    if hasattr(self.regressor, 'predict'):
      z_np = z.detach().cpu().numpy()
      pred = self.regressor.predict(z_np).astype(np.float32)
      if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
      return torch.from_numpy(pred).to(reg_dev)
    raise TypeError(f'unsupported regressor type: {type(self.regressor)}')

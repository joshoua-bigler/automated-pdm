import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mlflow.pyfunc import PythonModel
from typing import Any
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute


class ConvAe(nn.Module):

  def __init__(self, in_feats: int, latent_dim: int, hidden_ch: int = 64, **_: Any):
    super().__init__()
    self.encoder = nn.Sequential(nn.Conv1d(in_feats, hidden_ch, kernel_size=5, padding=2), nn.ReLU(inplace=True),
                                 nn.Conv1d(hidden_ch, hidden_ch, kernel_size=5, padding=2), nn.ReLU(inplace=True),
                                 nn.AdaptiveAvgPool1d(1))
    self.to_latent = nn.Linear(hidden_ch, latent_dim)
    self.from_latent = nn.Linear(latent_dim, hidden_ch)
    self.decoder = nn.Sequential(nn.Conv1d(hidden_ch, hidden_ch, kernel_size=5, padding=2), nn.ReLU(inplace=True),
                                 nn.Conv1d(hidden_ch, in_feats, kernel_size=5, padding=2))

  def encode(self, x_ch_first: torch.Tensor) -> torch.Tensor:
    h = self.encoder(x_ch_first).squeeze(-1)
    return self.to_latent(h)

  def decode(self, z: torch.Tensor, T: int) -> torch.Tensor:
    h = self.from_latent(z).unsqueeze(-1)  # (b, hidden, 1)
    # memory-light: broadcast, then make contiguous for conv
    h = h.expand(-1, -1, T).contiguous()  # (b, hidden, T)
    return self.decoder(h)  # (b, feats, T)

  def forward(self, x_win_feats: torch.Tensor) -> torch.Tensor:
    T = x_win_feats.size(1)
    x_cf = x_win_feats.permute(0, 2, 1)  # (b, feats, T)
    z = self.encode(x_cf)
    xhat_cf = self.decode(z, T)
    xhat = xhat_cf.permute(0, 2, 1)  # (b, T, feats)
    return xhat


class ConvAeTemp(nn.Module):
  ''' Convolutional Autoencoder with strided conv/convtranspose layers. '''

  def __init__(self, in_feats: int, latent_dim: int, hidden: int = 64, **_: Any):
    super().__init__()
    self.enc = nn.Sequential(
        nn.Conv1d(in_feats, hidden, 5, padding=2, stride=2),
        nn.ReLU(inplace=True),  # T -> T/2
        nn.Conv1d(hidden, hidden, 5, padding=2, stride=2),
        nn.ReLU(inplace=True)  # T/2 -> T/4
    )
    self.to_lat = nn.Conv1d(hidden, latent_dim, 1)  # preserve small T_bottleneck
    self.from_lat = nn.Conv1d(latent_dim, hidden, 1)
    self.dec = nn.Sequential(
        nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1),
        nn.ReLU(inplace=True),  # x2
        nn.ConvTranspose1d(hidden, in_feats, 4, stride=2, padding=1)  # x2
    )

  def encode(self, x_ch_first: torch.Tensor) -> torch.Tensor:
    h = self.enc(x_ch_first)  # (b, hidden, T/4)
    z = self.to_lat(h)  # (b, latent_dim, T/4)
    return z

  def decode(self, z: torch.Tensor) -> torch.Tensor:
    h = self.from_lat(z)  # (b, hidden, T/4)
    return self.dec(h)  # (b, in_feats, T)

  def forward(self, x_win_feats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x_win_feats.permute(0, 2, 1)  # (b, feats, T)
    z = self.encode(x)
    xhat = self.decode(z).permute(0, 2, 1)  # (b, T, feats)
    return xhat, z.mean(dim=2)  # optional: pool z over time for a vector


class Flatten(PythonModel):

  def __init__(self, window_size: int, n_features: int):
    super().__init__()
    self.w = int(window_size)
    self.f = int(n_features)

  def _to_numpy(self, x: np.ndarray | torch.Tensor) -> tuple[np.ndarray, bool, tuple[torch.device, torch.dtype] | None]:
    try:
      if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32, copy=False), True, (x.device, x.dtype)
    except ImportError:
      pass
    if hasattr(x, 'to_numpy'):
      return x.to_numpy().astype(np.float32, copy=False), False, None
    return np.asarray(x, dtype=np.float32), False, None

  def predict(self, model_input, context=None) -> np.ndarray | torch.Tensor:
    x_np, was_tensor, torch_meta = self._to_numpy(model_input)
    if x_np.ndim != 3:
      raise ValueError('expected input shape [n, w, f]')
    n, w, f = x_np.shape
    if w != self.w or f != self.f:
      raise ValueError(f'expected [*, {self.w}, {self.f}], got {tuple(x_np.shape)}')
    z_np = x_np.reshape(n, self.w * self.f).astype(np.float32, copy=False)
    if was_tensor:
      dev, dt = torch_meta
      return torch.from_numpy(z_np).to(device=dev, dtype=dt)
    return z_np

  def __str__(self) -> str:
    return f'{self.__class__.__name__}(window_size={self.w}, n_features={self.f})'

  def __repr__(self) -> str:
    return self.__str__()


class TsfreshEncoderPyfunc(PythonModel):

  def __init__(self,
               fc_params: str,
               selected_cols: list,
               fc_parameters: MinimalFCParameters | EfficientFCParameters,
               n_jobs: int = 0,
               normalize: bool = False,
               scaler_mean: list[float] | None = None,
               scaler_scale: list[float] | None = None):
    super().__init__()
    self.fc_name = str(fc_params)
    self.selected_cols = list(selected_cols)
    self.n_jobs = int(n_jobs)
    self.fc = fc_parameters
    self.normalize = bool(normalize)
    if self.normalize:
      self.scaler_mean = np.asarray(scaler_mean, dtype=np.float64) if scaler_mean is not None else None
      self.scaler_scale = np.asarray(scaler_scale, dtype=np.float64) if scaler_scale is not None else None
      if self.scaler_mean is None or self.scaler_scale is None:
        raise ValueError('normalize=True but scaler stats are missing')
      self.scaler_scale = np.where(self.scaler_scale == 0.0, 1.0, self.scaler_scale)
    else:
      self.scaler_mean = None
      self.scaler_scale = None

  def _to_numpy(self, model_input) -> np.ndarray:
    if isinstance(model_input, torch.Tensor):
      return model_input.detach().cpu().numpy().astype(np.float32, copy=False)
    elif hasattr(model_input, 'to_numpy'):
      return model_input.to_numpy().astype(np.float32, copy=False)
    return np.asarray(model_input, dtype=np.float32)

  def _to_long_df(self, x_np: np.ndarray) -> pd.DataFrame:
    if x_np.ndim != 3:
      raise ValueError('expected input shape [n, w, f]')
    n, w, f = x_np.shape
    ids = np.repeat(np.arange(n), w * f)
    times = np.tile(np.repeat(np.arange(w), f), n)
    kinds = np.tile(np.arange(f), n * w)
    vals = x_np.reshape(-1)
    return pd.DataFrame({'id': ids, 'time': times, 'kind': kinds, 'value': vals})

  def predict(self, model_input, context=None) -> np.ndarray | torch.Tensor:
    model_input = self._to_numpy(model_input)
    df_long = self._to_long_df(model_input)
    feats = extract_features(df_long,
                             column_id='id',
                             column_sort='time',
                             column_kind='kind',
                             column_value='value',
                             default_fc_parameters=self.fc,
                             n_jobs=self.n_jobs,
                             disable_progressbar=True)
    impute(feats)
    feats = feats.reindex(columns=self.selected_cols, fill_value=0.0)
    feats_arr = feats.to_numpy(dtype=np.float64, copy=False)
    if self.normalize and self.scaler_mean is not None and self.scaler_scale is not None:
      feats_arr = (feats_arr - self.scaler_mean) / self.scaler_scale
    return feats_arr.astype(np.float32, copy=False)

  def __str__(self) -> str:
    return f'{self.__class__.__name__}(fc_params={self.fc_name}, n_jobs={self.n_jobs}, selected_cols={len(self.selected_cols)})'

  def __repr__(self) -> str:
    return self.__str__()


def _same_pad(k: int, d: int = 1) -> int:
  # for stride=1 conv, keeps length (approximately) same
  return (k // 2) * d


class ConvBlock1d(nn.Module):

  def __init__(self, cin: int, cout: int, k: int, stride: int = 1, dilation: int = 1, dropout: float = 0.0):
    super().__init__()
    pad = _same_pad(k, dilation)
    self.conv = nn.Conv1d(cin, cout, kernel_size=k, stride=stride, dilation=dilation, padding=pad, bias=False)
    self.bn = nn.BatchNorm1d(cout)
    self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = self.bn(x)
    x = F.silu(x)
    x = self.drop(x)
    return x


class ResDilatedBlock1d(nn.Module):

  def __init__(self, c: int, k: int = 3, dilation: int = 1, dropout: float = 0.0):
    super().__init__()
    pad = _same_pad(k, dilation)
    self.conv1 = nn.Conv1d(c, c, kernel_size=k, dilation=dilation, padding=pad, bias=False)
    self.bn1 = nn.BatchNorm1d(c)
    self.conv2 = nn.Conv1d(c, c, kernel_size=k, dilation=dilation, padding=pad, bias=False)
    self.bn2 = nn.BatchNorm1d(c)
    self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = self.conv1(x)
    y = self.bn1(y)
    y = F.silu(y)
    y = self.drop(y)
    y = self.conv2(y)
    y = self.bn2(y)
    return F.silu(x + y)


class UpsampleBlock1d(nn.Module):

  def __init__(self, cin: int, cout: int, k: int = 9, scale: int = 2, dropout: float = 0.0):
    super().__init__()
    self.scale = scale
    self.conv = nn.Conv1d(cin, cout, kernel_size=k, padding=_same_pad(k), bias=False)
    self.bn = nn.BatchNorm1d(cout)
    self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(x, scale_factor=self.scale, mode='linear', align_corners=False)
    x = self.conv(x)
    x = self.bn(x)
    x = F.silu(x)
    x = self.drop(x)
    return x


class DilatedCnnAE(nn.Module):

  def __init__(self,
               in_ch: int = 2,
               base_ch: int = 32,
               latent_dim: int = 32,
               n_down: int = 3,
               k_down: int = 15,
               k_dil: int = 3,
               dilations: tuple = (1, 2, 4, 8),
               dropout: float = 0.0,
               out_len_hint: int | None = None,
               **_: Any):
    super().__init__()
    self.in_ch = in_ch
    self.base_ch = base_ch
    self.latent_dim = latent_dim
    self.n_down = n_down
    self.out_len_hint = out_len_hint
    enc = []
    c = in_ch
    for i in range(n_down):
      c2 = base_ch * (2**i)
      enc.append(ConvBlock1d(c, c2, k=k_down, stride=2, dilation=1, dropout=dropout))
      c = c2
    self.enc_down = nn.Sequential(*enc)
    self.enc_dil = nn.Sequential(*[ResDilatedBlock1d(c, k=k_dil, dilation=int(d), dropout=dropout) for d in dilations])
    self.enc_pool = nn.AdaptiveAvgPool1d(1)
    self.encoder = nn.Sequential(self.enc_down, self.enc_dil, self.enc_pool)
    self.to_latent = nn.Linear(c, latent_dim)
    self.from_latent = nn.Linear(latent_dim, c)
    if out_len_hint is not None:
      seed_len = max(4, int(math.ceil(out_len_hint / (2**n_down))))
    else:
      seed_len = 16
    self.seed_len = seed_len
    self.dec_seed = nn.Parameter(torch.zeros(1, c, seed_len))
    dec = []
    for i in reversed(range(n_down)):
      c2 = base_ch * (2**i)
      dec.append(UpsampleBlock1d(c, c2, k=k_down, scale=2, dropout=dropout))
      c = c2
    self.dec_up = nn.Sequential(*dec)
    self.dec_out = nn.Conv1d(c, in_ch, kernel_size=9, padding=_same_pad(9), bias=True)

  def _infer_layout(self, x: torch.Tensor) -> str:
    # returns 'bcl' or 'blc'
    if x.ndim != 3:
      raise ValueError(f'expected 3d tensor, got shape {tuple(x.shape)}')
    if x.shape[1] == self.in_ch:
      return 'bcl'
    if x.shape[-1] == self.in_ch:
      return 'blc'
    raise ValueError(f'cannot infer layout for shape {tuple(x.shape)} with in_ch={self.in_ch}')

  def _to_bcl(self, x: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == 'bcl':
      return x
    # blc -> bcl
    return x.permute(0, 2, 1).contiguous()

  def _from_bcl(self, x: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == 'bcl':
      return x
    # bcl -> blc
    return x.permute(0, 2, 1).contiguous()

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    layout = self._infer_layout(x)
    x = self._to_bcl(x, layout)
    x = self.encoder(x).squeeze(-1)  # (b, c)
    z = self.to_latent(x)
    return z

  def decode(self, z: torch.Tensor, target_len: int) -> torch.Tensor:
    b = z.shape[0]
    h = self.from_latent(z).view(b, -1, 1)
    seed = self.dec_seed.expand(b, -1, -1)
    x = seed + h
    x = self.dec_up(x)
    x = self.dec_out(x)
    if x.shape[-1] != target_len:
      x = F.interpolate(x, size=target_len, mode='linear', align_corners=False)
    return x  # (b, c, L)

  def forward(self, x: torch.Tensor, return_latent: bool = False):
    layout = self._infer_layout(x)
    x_bcl = self._to_bcl(x, layout)
    target_len = x_bcl.shape[-1]
    z = self.encode(x_bcl)
    xr_bcl = self.decode(z, target_len=target_len)
    xr = self._from_bcl(xr_bcl, layout)
    if return_latent:
      return xr, z
    return xr

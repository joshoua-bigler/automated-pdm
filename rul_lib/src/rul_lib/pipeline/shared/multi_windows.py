import numpy as np
# local
from rul_lib.enums import PipelineType


def infer_base_seq_len(windowed: dict) -> int:
  ' Infer base sequence length from windowed data. '
  x_tr = windowed['x_train']
  if x_tr.ndim == 3:  # (N, L, C) or (N, C, L)
    if x_tr.shape[-1] == windowed['meta'].get('n_features', x_tr.shape[-1]):  # (N, L, C)
      return int(x_tr.shape[1])
    else:  # (N, C, L)
      return int(x_tr.shape[-1])
  raise ValueError(f'unexpected x_train shape={x_tr.shape}')


def infer_seq_len(windowed: dict, cfg: dict, pipeline_type: PipelineType) -> int:
  ' Infer effective sequence length considering multi-window settings. '
  L = infer_base_seq_len(windowed)
  if pipeline_type is PipelineType.FE_REG:
    return L
  apply_mw = bool(cfg.get('apply', False))
  m = int(cfg.get('m', 1)) if apply_mw else 1
  L_eff = L * m
  return L_eff


def make_multiwindow_latent(z: np.ndarray, y: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
  ' Create multi-window latent representations. '
  if z.ndim != 2:
    raise ValueError(f'z must be 2D, got shape={z.shape}')
  n, d = z.shape
  z = z.astype(np.float32)
  y = y.astype(np.float32).reshape(-1)
  if n != y.shape[0]:
    raise ValueError(f'len(z)={n} != len(y)={y.shape[0]}')
  if m <= 1:
    return z, y
  X_list, y_list = [], []
  for i in range(m - 1, n):
    block = z[i - m + 1:i + 1]  # (m, D)
    X_list.append(block.reshape(-1))  # (m*D,)
    y_list.append(y[i])  # target of last element
  Z_out = np.stack(X_list, axis=0)  # (N_eff, m*D)
  y_out = np.array(y_list, dtype=np.float32)
  return Z_out, y_out


def append_os_features(z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, y_te: np.ndarray, windowed: dict,
                       apply_mw: bool, m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  ''' Append OS features to latent representations. '''
  os_tr_raw = np.asarray(windowed['os_int_train'])
  os_te_raw = np.asarray(windowed['os_int_test'])
  if os_tr_raw.ndim == 1:
    os_tr_raw = os_tr_raw.reshape(-1, 1)
    os_te_raw = os_te_raw.reshape(-1, 1)
  if apply_mw and m > 1:
    os_tr_feat = os_tr_raw[m - 1:]  # (N_eff, os_dim)
    os_te_feat = os_te_raw[m - 1:]
  else:
    os_tr_feat = os_tr_raw  # (N, os_dim)
    os_te_feat = os_te_raw
  if z_tr.shape[0] != os_tr_feat.shape[0]:
    raise ValueError(f'z_train rows {z_tr.shape[0]} != os_train rows {os_tr_feat.shape[0]}')
  if z_te.shape[0] != os_te_feat.shape[0]:
    raise ValueError(f'z_test rows {z_te.shape[0]} != os_test rows {os_te_feat.shape[0]}')
  z_tr_ext = np.concatenate([z_tr, os_tr_feat], axis=1)
  z_te_ext = np.concatenate([z_te, os_te_feat], axis=1)
  return z_tr_ext, y_tr, z_te_ext, y_te

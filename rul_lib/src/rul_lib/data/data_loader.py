import numpy as np
import pandas as pd
import yaml
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
# local
from rul_lib.pipeline.pre.normalize import invert_target


def load_config(config_path: Path) -> dict:
  with open(config_path, 'r') as f:
    config = yaml.safe_load(f)
  return config


def create_cmapps_eval_data(data: dict[str, pd.DataFrame],
                            window_size: str | int) -> tuple[np.ndarray, np.ndarray, list[int]]:
  ''' data['x_test']: rows = time steps, columns include 'unit' + sensor cols (no cycle column).
      data['y_test']: has 'unit' and 'rul' (per row). We'll take the last rul per unit.

      Returns
      -------
      X: (n_units, W, n_feats) numpy array
      y: (n_units,) numpy array of last-window targets (one per unit)
      units: list of unit ids in the returned order
  '''
  x = data['x_test'].copy()
  y = data['y_test'].copy()
  W = int(window_size)
  feat_cols = [c for c in x.columns if c not in ('unit', 'rul', 'fault', 'timestamp')]
  counts = x.groupby('unit').size()
  ok_units = counts[counts >= W].index
  units = ok_units.tolist()
  last_df = (x[x['unit'].isin(ok_units)].groupby('unit', group_keys=False).tail(W))
  # ensure rows per unit are in chronological row-order
  last_df = last_df.set_index('unit')
  # build 3D tensor
  X = (last_df.groupby(level=0)[feat_cols].apply(lambda g: g.to_numpy()).to_list())
  X = np.stack(X, axis=0)
  # target: last label per unit (matches official CMAPSS eval)
  y_last = (y.groupby('unit', as_index=True)['rul'].last().reindex(units)) # yapf: disable
  y_arr = y_last.to_numpy(dtype=float)
  return X, y_arr, units


def _feat_cols(df: pd.DataFrame) -> list[str]:
  return [c for c in df.columns if c not in ('unit', 'rul', 'fault', 'timestamp', 'rec', 'cycle')]


def _nearest_distinct(values: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> np.ndarray:
  chosen = []
  used = np.zeros(values.shape[0], dtype=bool)
  for t in targets:
    m = mask & ~used
    if not m.any():
      continue
    d = np.abs(values - float(t))
    d[~m] = np.inf
    i = int(np.argmin(d))
    if np.isinf(d[i]):
      continue
    chosen.append(i)
    used[i] = True
  return np.array(chosen, dtype=int)


def create_femto_eval_data(data: dict,
                           window_size: int,
                           samples_per_unit: int = 20,
                           return_scaled: bool = True,
                           windowed_meta: dict | None = None,
                           encode_os: bool = False) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray | None]:
  x = data['x_test'].copy()
  y = data['y_test'][['unit', 'cycle', 'rul']].copy()
  meta = data.get('meta', {}) or {}
  ys = meta.get('y_scaler', {}) or {}
  clip = float(ys.get('clip', ys.get('params', {}).get('clip', 125.0)))
  W = int(window_size)
  try:
    df = (x.set_index(['unit', 'cycle']).join(y.set_index(['unit', 'cycle'])[['rul']], how='inner').reset_index())
  except Exception:
    df = x.merge(y, on=['unit', 'cycle'], how='inner')
    df = df.sort_values(['unit', 'cycle'], kind='mergesort').reset_index(drop=True)
  feats = _feat_cols(df)
  wm = windowed_meta or {}
  os_mapping = wm.get('os_mapping')
  has_os = encode_os and os_mapping is not None and 'os' in df.columns
  grid = np.linspace(0.0, clip, int(samples_per_unit), dtype=np.float32)
  X_list = []
  y_list = []
  units = []
  os_list = [] if has_os else None
  for unit, du in df.groupby('unit', sort=False):
    # keep du sorted; if you used join via index, it's usually already sorted by original order
    du = du.sort_values('cycle', kind='mergesort')
    m = len(du)
    if m < W:
      continue
    rul_scaled_u = du['rul'].to_numpy(dtype=np.float32)
    rul_raw_u = invert_target(rul_scaled_u.astype(np.float64), meta).astype(np.float32)
    rul_raw_u = np.clip(rul_raw_u, 0.0, clip)
    ends_ok = (np.arange(m) >= (W - 1)) & (rul_raw_u >= 0.0) & (rul_raw_u <= clip)
    if not ends_ok.any():
      continue
    ends = _nearest_distinct(rul_raw_u, grid, ends_ok)
    if ends.size == 0:
      continue
    xv = du[feats].to_numpy(dtype=np.float32)  # (m, n_feats)
    # build all windows as a view: (m-W+1, W, n_feats)
    win = sliding_window_view(xv, window_shape=W, axis=0)
    # sliding_window_view gives shape (m-W+1, W, n_feats) already for axis=0
    # map end indices e -> window index (start) = e-(W-1)
    starts = ends.astype(np.int64) - (W - 1)
    X_u = win[starts]  # (n_sel, W, n_feats)
    X_list.append(X_u)
    if return_scaled:
      y_u = rul_scaled_u[ends.astype(np.int64)]
    else:
      y_u = rul_raw_u[ends.astype(np.int64)]
    y_list.append(y_u.astype(np.float32, copy=False))
    units.extend([unit] * int(len(ends)))
    if has_os:
      os_u = du['os'].to_numpy()
      # vectorized mapping (safe, but can be replaced by an array lookup if os is dense)
      try:
        os_codes = np.fromiter((os_mapping[int(v)] for v in os_u[ends.astype(np.int64)]),
                               dtype=np.int64,
                               count=len(ends))
      except KeyError as e:
        raise ValueError(f'os value {e.args[0]} not in os_mapping {os_mapping}')
      os_list.append(os_codes)
  if not X_list:
    raise ValueError('no eval windows built')
  X = np.concatenate(X_list, axis=0)
  y_out = np.concatenate(y_list, axis=0)
  os_eval = None
  if has_os and os_list:
    os_eval = np.concatenate(os_list, axis=0)
  return X, y_out, units, os_eval

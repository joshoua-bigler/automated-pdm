import pandas as pd
import numpy as np


def downsample(data: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
  ''' Downsample time-series data by a specified factor.

      Parameters  
      ----------
      data: 
        Dictionary containing dataframes for different splits (e.g., 'x_train', 'y_train', etc.).
      config : 
        Configuration dictionary with downsampling parameters:
        - apply: bool, whether to apply downsampling.
        - factor: int, downsampling factor.
        - method: str, downsampling method ('decimate' or 'poly').
        - group_by: str | list[str], columns to group by when downsampling.
        - sensor_cols: list[str], columns to downsample (if None, auto-detect).
        - meta_cols: list[str], additional meta columns to retain.  

      Returns
      -------
      dict:
        Downsampled dataframes.
  '''
  apply = bool(config.get('apply', True))
  factor = int(config.get('factor', 8))
  method = (config.get('method') or 'poly').lower()  # 'decimate' | 'poly'
  group_by = config.get('group_by', 'unit')
  sensor_cols_cfg = config.get('sensor_cols', ['vib_h', 'vib_v', 'temperature'])  # yapf: disable; e.g., ['vib_h', 'vib_v', 'timestamp']
  meta_cols_cfg = config.get('meta_cols', ['unit', 'cycle', 'os'])  # optional override
  if not apply or factor <= 1:
    return data
  # normalize group_by into a list
  if isinstance(group_by, str):
    group_by = [group_by]
  elif not isinstance(group_by, list):
    raise TypeError('group_by must be str or list[str]')
  default_meta = ['unit', 'cycle', 'os', 'split']
  meta_cols = list(dict.fromkeys((meta_cols_cfg or default_meta)))  # preserve order, dedupe

  def _group_cols(df: pd.DataFrame) -> list[str]:
    # Prefer (unit, cycle) if available, else use provided group_by columns
    if 'unit' in df.columns and 'cycle' in df.columns:
      return ['unit', 'cycle']
    missing = [c for c in group_by if c not in df.columns]
    if missing:
      raise KeyError(f'missing group_by columns: {missing}')
    return group_by

  def _pick_signal_cols(df: pd.DataFrame) -> list[str]:
    if sensor_cols_cfg:
      return [c for c in sensor_cols_cfg if c in df.columns]
    # fallback: all numeric columns excluding meta/id/label-like columns
    exclude = set(meta_cols + ['rul', 'label', 'target'])
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

  def _downsample_group_frame(g: pd.DataFrame, sig_cols: list[str], grp_cols: list[str]) -> pd.DataFrame:
    if method == 'poly':
      from scipy.signal import resample_poly
      x = g[sig_cols].to_numpy()
      x_ds = resample_poly(x, up=1, down=factor, axis=0)
      g_ds = pd.DataFrame(x_ds, columns=sig_cols)
      # carry meta: constants for grouping keys; for others, stride
      for c in meta_cols:
        if c not in g.columns:
          continue
        if c in grp_cols:  # constant within group
          g_ds[c] = g[c].iloc[0]
        else:
          v = g[c].iloc[::factor].reset_index(drop=True)
          if len(v) < len(g_ds):
            v = pd.concat([v, pd.Series([v.iloc[-1]] * (len(g_ds) - len(v)))], ignore_index=True)
          g_ds[c] = v.iloc[:len(g_ds)].to_numpy()
      # ensure grouping columns are first, then meta, then signals (stable order)
      lead = [c for c in grp_cols if c in g_ds.columns]
      metas = [c for c in meta_cols if c in g_ds.columns and c not in lead]
      cols = lead + metas + sig_cols
      return g_ds[cols]
    elif method == 'decimate':
      g_ds = g.iloc[::factor].copy().reset_index(drop=True)
      # ensure grouping keys are constant (they should already be)
      for c in meta_cols:
        if c in g_ds.columns and c in grp_cols:
          g_ds[c] = g[c].iloc[0]
      # keep only meta + signals (avoid accidental extra numeric)
      keep = [c for c in (grp_cols + [c for c in meta_cols if c not in grp_cols] + sig_cols) if c in g_ds.columns]
      return g_ds[keep]
    else:
      raise ValueError(f'unknown method: {method}')

  def _downsample_x(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = _group_cols(df)
    sig_cols = _pick_signal_cols(df)
    if not sig_cols:
      return df.copy()
    parts = []
    for _, g in df.groupby(grp_cols, sort=False, as_index=False):
      parts.append(_downsample_group_frame(g, sig_cols, grp_cols))
    out = pd.concat(parts, ignore_index=True)
    # Invariants: os constant per unit (if present)
    if 'unit' in out.columns and 'os' in out.columns:
      bad = out.groupby('unit')['os'].nunique()
      bad = bad[bad != 1]
      if not bad.empty:
        raise AssertionError(f'os not constant per unit for: {list(bad.index)}')
    return out

  def _subsample_y(df: pd.DataFrame) -> pd.DataFrame:
    grp_cols = _group_cols(df) if any(c in df.columns for c in group_by) else []
    if not grp_cols:
      return df
    # if labels are per-group (e.g., one row per unit/cycle), leave as is
    sizes = df.groupby(grp_cols).size()
    if sizes.min() <= 2:
      return df
    parts = []
    for _, g in df.groupby(grp_cols, sort=False, as_index=False):
      parts.append(g.iloc[::factor, :].reset_index(drop=True))
    return pd.concat(parts, ignore_index=True)

  out = {}
  for split, df in data.items():
    if split in ('x_train', 'x_val', 'x_test'):
      out[split] = _downsample_x(df)
    elif split in ('y_train', 'y_val', 'y_test'):
      out[split] = _subsample_y(df)
    else:
      out[split] = df.copy()
  return out

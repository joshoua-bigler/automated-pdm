import pandas as pd
import numpy as np
# local
from rul_lib.gls.gls import logger

DEFAULT_META_COLS = {'unit', 'cycle', 'rul', 'label', 'target', 'split', 'timestamp', 'os'}


def _select_feature_cols(df: pd.DataFrame, columns: list[str] | None, ignore: list[str]) -> list[str]:
  ignore_set = set(ignore or []) | DEFAULT_META_COLS
  if columns:
    return [c for c in columns if c in df.columns and c not in ignore_set and pd.api.types.is_numeric_dtype(df[c])]
  return [c for c in df.columns if c not in ignore_set and pd.api.types.is_numeric_dtype(df[c])]


def _moving_average(df: pd.DataFrame, cols: list[str], group_by: str | None, window: int) -> pd.DataFrame:
  dfc = df.copy()
  if not cols:
    return dfc
  if group_by and group_by in dfc.columns:
    dfc[cols] = (dfc.groupby(
        group_by, group_keys=False)[cols].apply(lambda g: g.rolling(window=window, min_periods=1, center=True).mean()))
  else:
    dfc[cols] = dfc[cols].rolling(window=window, min_periods=1, center=True).mean()
  return dfc


def _exponential(df: pd.DataFrame, cols: list[str], group_by: str | None, alpha: float) -> pd.DataFrame:
  '''Simple exponential smoothing via pandas EWM (S_t = α x_t + (1-α) S_{t-1}).'''
  dfc = df.copy()
  if not cols:
    return dfc
  if not (0.0 < float(alpha) < 1.0):
    raise ValueError('exponential smoothing requires 0 < alpha < 1')
  if group_by and group_by in dfc.columns:
    dfc[cols] = (dfc.groupby(group_by, group_keys=False)[cols].apply(lambda g: g.ewm(alpha=float(alpha), adjust=False).mean()))  # yapf: disable
  else:
    dfc[cols] = dfc[cols].ewm(alpha=float(alpha), adjust=False).mean()
  return dfc


def _tsma_block(values: np.ndarray, period: int, m_cycles: int) -> np.ndarray:
  ''' Time Synchronous Moving Average (TSMA) on a 2D array.

      Parameters
      ----------
      values : np.ndarray
          Array of shape (n_samples, n_features), ordered in time.
      period : int
          Number of samples per cycle (assumed constant for this block).
      m_cycles : int
          Number of neighbouring cycles to average (M in TSMA).

      Returns
      -------
      np.ndarray
          TSMA-processed array of shape ((N - M + 1) * period, n_features),
          where N is the number of full cycles (N = floor(n_samples / period)).
          Any tail samples that do not form a complete cycle are dropped.
  '''
  if period <= 0:
    raise ValueError('tsma requires period > 0')
  if m_cycles <= 0:
    raise ValueError('tsma requires m_cycles > 0')
  n_samples, n_features = values.shape
  n_cycles = n_samples // period
  if n_cycles < m_cycles:
    return values
  n_full = n_cycles * period
  vals_full = values[:n_full].reshape(n_cycles, period, n_features)  # (N, L, F)
  out_cycles = n_cycles - m_cycles + 1
  out = np.empty((out_cycles, period, n_features), dtype=values.dtype)
  for r in range(out_cycles):
    out[r] = vals_full[r:r + m_cycles].mean(axis=0)
  out_flat = out.reshape(out_cycles * period, n_features)
  return out_flat


def _tsma(df: pd.DataFrame, cols: list[str], group_by: str | None, period_by_group: dict,
          m_cycles: int) -> pd.DataFrame:
  ''' TSMA over M neighbouring cycles with period resolved per group.

      Parameters
      ----------
      df : pd.DataFrame
          Input data, assumed sorted in time within each group.
      cols : list[str]
          Columns to apply TSMA on.
      group_by : str | None
          Grouping column (e.g., 'unit'). If None or not in df, TSMA is applied
          to the full DataFrame using the period stored under key None.
      period_by_group : dict
          Mapping from group key -> period (samples per cycle).
          If group_by is None, the key None is used.
      m_cycles : int
          Number of neighbouring cycles to average.
  '''
  if not cols:
    return df
  if m_cycles <= 0:
    raise ValueError('tsma requires m_cycles > 0')

  def _apply_block(block: pd.DataFrame) -> pd.DataFrame:
    if block.empty:
      return block
    if group_by and group_by in block.columns:
      key = block[group_by].iloc[0]
    else:
      key = None
    if key not in period_by_group:
      raise KeyError(f'No TSMA period found for group {key!r}')
    period = int(period_by_group[key])
    vals = block[cols].to_numpy()
    out_vals = _tsma_block(vals, period=period, m_cycles=m_cycles)
    n_out = out_vals.shape[0]
    out_block = block.iloc[:n_out].copy()
    out_block[cols] = out_vals
    return out_block

  if group_by and group_by in df.columns:
    df_out = df.groupby(group_by, group_keys=False).apply(_apply_block)
  else:
    df_out = _apply_block(df)
  return df_out


def _build_tsma_period_map(base_df: pd.DataFrame, group_by: str | None, os_col: str, os_period_map: dict | None,
                           os_rpm_map: dict | None, fs: float | None) -> dict:
  if os_period_map is None and os_rpm_map is None:
    if group_by and group_by in base_df.columns:
      period_by_group: dict = {}
      if 'cycle' not in base_df.columns:
        raise ValueError('tsma fallback without period map requires column "cycle" in data')
      for key, g in base_df.groupby(group_by):
        if g.empty:
          period_by_group[key] = 0
          continue
        first_cycle = g['cycle'].iloc[0]
        period = int((g['cycle'] == first_cycle).sum())
        if period <= 0:
          raise ValueError(f'computed non-positive period {period} for group {key!r}')
        period_by_group[key] = period
      return period_by_group
    # no grouping → single global period
    if base_df.empty:
      return {None: 0}
    if 'cycle' in base_df.columns:
      first_cycle = base_df['cycle'].iloc[0]
      period = int((base_df['cycle'] == first_cycle).sum())
    else:
      period = int(base_df.shape[0])
    if period <= 0:
      raise ValueError(f'computed non-positive period {period} for global group')
    return {None: period}
  # --- original behaviour: use os_period_map / os_rpm_map ---
  if os_rpm_map is not None and fs is None:
    raise ValueError('tsma with os_rpm_map requires params["fs"] (sampling frequency in Hz)')
  if group_by and group_by in base_df.columns:
    os_series = base_df.groupby(group_by)[os_col].first()
    keys = list(os_series.index)
    os_values = list(os_series.values)
  else:
    if os_col not in base_df.columns:
      raise ValueError(f'tsma requires os column "{os_col}" in data')
    if base_df.empty:
      return {None: 0}
    keys = [None]
    os_values = [base_df[os_col].iloc[0]]

  def _resolve_period(os_val):
    # generate candidate keys: original, int, str
    candidates = [os_val]
    try:
      candidates.append(int(os_val))
    except (TypeError, ValueError):
      pass
    candidates.append(str(os_val))
    for cand in candidates:
      if os_period_map is not None and cand in os_period_map:
        return int(os_period_map[cand])
      if os_rpm_map is not None and cand in os_rpm_map:
        rpm = float(os_rpm_map[cand])
        if rpm <= 0.0:
          raise ValueError(f'invalid rpm {rpm} for operating state {os_val!r}')
        return int(round(fs * 60.0 / rpm))
    raise KeyError(f'no period mapping found for operating state {os_val!r}')

  period_by_group: dict = {}
  for key, os_val in zip(keys, os_values):
    period = _resolve_period(os_val)
    if period <= 0:
      raise ValueError(f'computed non-positive period {period} for group {key!r}')
    period_by_group[key] = period
  return period_by_group


def denoise(data: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
  ''' Apply denoising to selected x_* splits in the dataset dictionary according to config
      and return the updated dict. Adds a "meta.denoising" entry describing what was applied.
  '''
  method = config.get('method', None)
  if method is None or method == 'none':
    logger.info('No denoising is applied')
    return data
  params = config.get('params', {}) or {}
  group_by = config.get('group_by', 'unit')
  ignore_used = config.get('ignore', []) or []
  cols_cfg = config.get('columns', None)
  if 'x_train' not in data:
    raise ValueError('data must include x_train')
  xtr = data['x_train']
  if not isinstance(xtr, pd.DataFrame):
    raise ValueError('x_train must be a DataFrame')
  x_keys = (config.get('apply_keys',
                       tuple(k for k, v in data.items() if isinstance(v, pd.DataFrame) and k.startswith('x_'))))
  cols = _select_feature_cols(xtr, cols_cfg, ignore_used)
  logger.info(f'Denoising method={method}, group_by={group_by}, cols={len(cols)}')
  # Method-specific precomputation (e.g., TSMA period map)
  tsma_period_by_group: dict | None = None
  if method == 'tsma':
    os_col = params.get('os_column', 'os')
    os_period_map = params.get('os_period_map', None)
    os_rpm_map = params.get('os_rpm_map', None)
    fs = params.get('fs', None)
    if fs is not None:
      fs = float(fs)
    # build period map from ALL x_* frames (train/val/test), so every unit is covered
    base_frames: list[pd.DataFrame] = []
    for k, v in data.items():
      if isinstance(v, pd.DataFrame) and k in x_keys:
        base_frames.append(v)
    if not base_frames:
      raise ValueError('tsma: no DataFrames found for apply_keys')
    base_df = pd.concat(base_frames, ignore_index=True)
    tsma_period_by_group = _build_tsma_period_map(
        base_df=base_df,
        group_by=group_by,
        os_col=os_col,
        os_period_map=os_period_map,
        os_rpm_map=os_rpm_map,
        fs=fs,
    )
  out: dict[str, pd.DataFrame] = {}
  for k, v in data.items():
    if isinstance(v, pd.DataFrame) and k in x_keys:
      if method == 'moving_average':
        dfc = _moving_average(v, cols, group_by=group_by, window=int(params.get('window', 5)))
      elif method == 'exponential':
        dfc = _exponential(v, cols, group_by=group_by, alpha=float(params.get('alpha', 0.5)))
      elif method == 'tsma':
        if tsma_period_by_group is None:
          raise RuntimeError('internal error: tsma_period_by_group is not initialized')
        m_cycles = int(params.get('m_cycles', 3))
        dfc = _tsma(v, cols, group_by=group_by, period_by_group=tsma_period_by_group, m_cycles=m_cycles)
      else:
        logger.warning(f'Unknown denoising method "{method}", passing data through unchanged')
        dfc = v
      out[k] = dfc
    else:
      out[k] = v
  # --- NEW: align y_* to x_* for tsma ---
  if method == 'tsma':
    for x_key in x_keys:
      if x_key in out:
        suffix = x_key.split('x_', 1)[1] if x_key.startswith('x_') else None
        if suffix:
          y_key = 'y_' + suffix
          if y_key in out:
            idx = out[x_key].index
            out[y_key] = out[y_key].loc[idx].copy()
  # build meta entry (also store resolved TSMA periods if available)
  params_meta = dict(params)
  if method == 'tsma' and tsma_period_by_group is not None:
    params_meta = dict(params_meta)
    params_meta['resolved_period_by_group'] = {
        (str(k) if k is not None else None): int(v) for k, v in tsma_period_by_group.items()
    }
  meta = out.get('meta', {}).copy()
  meta['denoising'] = {
      'method': method,
      'columns': cols,
      'ignore': ignore_used,
      'group_by': group_by,
      'params': {
          k: (int(v) if isinstance(v, np.integer) else v) for k, v in params_meta.items()
      },
      'applied_on': list(x_keys),
  }
  out['meta'] = meta
  logger.info(f'Denoising method {method} applied on keys: {list(x_keys)}')
  return out

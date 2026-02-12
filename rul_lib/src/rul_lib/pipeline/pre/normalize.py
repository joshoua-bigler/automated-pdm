import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Any
# local
from rul_lib.gls.gls import logger
from rul_lib.pipeline.pre.cluster import ClusterNorm


def _feature_cols_from_config(df: pd.DataFrame, ignore: list[str], columns: list[str] | None = None) -> list[str]:
  '''Determine feature columns from config and dataframe '''
  if columns == 'all':
    candidates = [c for c in df.columns if c not in ignore]
  else:
    candidates = [c for c in columns if c in df.columns and c not in ignore]
  return candidates


def _pick_scaler(method: str) -> StandardScaler | MinMaxScaler:
  ''' Return a scaler instance based on method name. '''
  if method == 'standard':
    return StandardScaler()
  if method == 'minmax':
    return MinMaxScaler()
  raise ValueError(f'unknown normalization method: {method}')


def classical_normalize(config: dict, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
  ''' Classical normalization (standard or minmax) on x_* splits. '''
  if 'x_train' not in data:
    raise ValueError('data must include x_train')
  method = config.get('norm_type', 'standard')
  group_by = config.get('group_by', None)
  ignore_cfg = config.get('ignore', [])
  cols_cfg = config.get('columns', 'all')
  logger.info(f'normalizing with method={method}, group_by={group_by}')

  # make a factory so each group gets a fresh scaler
  def make_scaler():
    return _pick_scaler(method=method)

  xtr = data['x_train']
  # robust feature selection: never normalize obvious meta
  default_ignore = ['unit', 'cycle', 'os', 'split', 'rul', 'label', 'target', 'timestamp']  # keep timestamp unscaled by default
  ignore = list(dict.fromkeys(default_ignore + list(ignore_cfg or [])))  
  feature_cols = _feature_cols_from_config(df=xtr, ignore=ignore, columns=cols_cfg)
  if not feature_cols:
    logger.warning('no feature columns selected for normalization')

  def _to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    dfc = df.copy()
    if cols:
      dfc[cols] = dfc[cols].astype('float64')
    return dfc

  def _as_float_array(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    x = df[cols].to_numpy(dtype='float64', copy=False)
    # replace non-finite with column medians (fit-time) or zeros (transform-time fallback)
    return x

  # list x_* keys once
  x_keys = [k for k in data.keys() if isinstance(data[k], pd.DataFrame) and k.startswith('x_')]
  out: dict[str, pd.DataFrame] = {}

  # helper to sanitize X for fit/transform
  def _sanitize_for_fit(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
      col_med = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0)
      col_med[~np.isfinite(col_med)] = 0.0
      bad = ~np.isfinite(x)
      x = x.copy()
      x[bad] = np.take(col_med, np.where(bad)[1])
    return x

  def _sanitize_for_transform(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
      x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    return x

  if not group_by or group_by == 'none':
    # ------- global fit on train -------
    scaler = make_scaler()
    x_fit = _sanitize_for_fit(_as_float_array(xtr, feature_cols))
    scaler.fit(x_fit)
    for k in x_keys:
      dfc = _to_float(data[k], feature_cols)
      x_arr = _sanitize_for_transform(_as_float_array(dfc, feature_cols))
      dfc[feature_cols] = scaler.transform(x_arr)
      out[k] = dfc
    meta = {
      'mode': 'global',
      'method': method,
      'columns': feature_cols,
      'mean': getattr(scaler, 'mean_', None).tolist() if hasattr(scaler, 'mean_') else None,
      'scale': getattr(scaler, 'scale_', None).tolist() if hasattr(scaler, 'scale_') else None,
      'min': getattr(scaler, 'min_', None).tolist() if hasattr(scaler, 'min_') else None,
      'data_min': getattr(scaler, 'data_min_', None).tolist() if hasattr(scaler, 'data_min_') else None,
      'data_max': getattr(scaler, 'data_max_', None).tolist() if hasattr(scaler, 'data_max_') else None,
    }
  else:
    # ------- grouped fit on train -------
    by = group_by if isinstance(group_by, list) else [group_by]
    missing = [c for c in by if c not in xtr.columns]
    if missing:
      raise ValueError(f'group_by columns not in x_train: {missing}')
    scalers: dict = {}
    for gid, gdf in xtr.groupby(by, dropna=False, sort=False):
      sc = make_scaler()
      x_fit = _sanitize_for_fit(_as_float_array(gdf, feature_cols))
      sc.fit(x_fit)
      scalers[gid] = sc
    if 'x_test' in data:
      unseen = []
      seen = set(scalers.keys())
      for gid in data['x_test'].groupby(by, dropna=False, sort=False).groups:
        if gid not in seen:
          unseen.append(gid)
      if unseen:
        logger.warning(f'unseen groups in x_test -> using global fallback: {unseen}')
    global_fallback = make_scaler()
    global_fallback.fit(_sanitize_for_fit(_as_float_array(xtr, feature_cols)))
    for k in x_keys:
      df = data[k]
      miss_any = [c for c in by if c not in df.columns]
      if miss_any:
        raise ValueError(f'split {k} is missing group_by columns: {miss_any}')
      dfc = _to_float(df, feature_cols)
      # group once, assign by index
      for gid, idx in dfc.groupby(by, dropna=False, sort=False).groups.items():
        sc = scalers.get(gid, global_fallback)
        x_arr = _sanitize_for_transform(dfc.loc[idx, feature_cols].to_numpy(dtype='float64', copy=False))
        dfc.loc[idx, feature_cols] = sc.transform(x_arr)
      out[k] = dfc

    def _arr(x):
      return x.tolist() if isinstance(x, np.ndarray) else (x.tolist() if hasattr(x, 'tolist') else x)

    meta = {
      'mode': 'grouped',
      'method': method,
      'columns': feature_cols,
      'group_by': by,
      'scalers': {
        str(gid): {
          'mean': _arr(getattr(sc, 'mean_', None)) if hasattr(sc, 'mean_') else None,
          'scale': _arr(getattr(sc, 'scale_', None)) if hasattr(sc, 'scale_') else None,
          'min': _arr(getattr(sc, 'min_', None)) if hasattr(sc, 'min_') else None,
          'data_min': _arr(getattr(sc, 'data_min_', None)) if hasattr(sc, 'data_min_') else None,
          'data_max': _arr(getattr(sc, 'data_max_', None)) if hasattr(sc, 'data_max_') else None,
        } for gid, sc in scalers.items()
      }
    }
  # pass-through other keys
  for k, v in data.items():
    if k not in out:
      out[k] = v
  out['meta'] = data.get('meta', {}).copy()
  out['meta']['normalization'] = meta
  # strong invariant checks (cheap)
  for k in [kk for kk in out.keys() if kk.startswith('x_')]:
    df = out[k]
    # ensure meta untouched
    for m in ['unit', 'cycle', 'os', 'split', 'rul', 'label', 'target']:
      if m in df.columns:
        assert not np.issubdtype(df[m].dtype, np.floating) or np.isfinite(df[m]).all(), f'{k}:{m} has non-finite values after norm'
    # ensure features finite
    if feature_cols:
      x_arr = df[feature_cols].to_numpy(dtype='float64', copy=False)
      assert np.isfinite(x_arr).all(), f'{k}: non-finite values in normalized features'
  return out


def scale_target(config: dict,
                 data: dict[str, pd.DataFrame],
                 target_col: str | None = None,
                 ignore: list[str] | None = None) -> dict[str, pd.DataFrame]:
  ''' Scale target y_* using stats from y_train only.

      Features
      --------
      - Always clips targets at clip before scaling.
      - Scales ONLY the target column; other columns (e.g., unit, cycle) are preserved.
      - Stores params in meta['y_scaler'] and the chosen target column in meta['y_target'].

      Config:
        preprocessing:
          target:
            column: rul
          target_scaling:
            method: none | standard | minmax
            clip: 130
  '''
  if 'y_train' not in data:
    raise ValueError('y_train missing')
  method = config.get('method', 'none')
  clip = config.get('clip', None)
  logger.info(f'Scaling target with method={method}, clip={clip}, target_col={target_col}')
  # --- resolve target column ---
  ytr = data['y_train']
  if isinstance(ytr, pd.Series):
    tgt = ytr.name if ytr.name is not None else 'target'
    ytr_series = ytr
  else:
    tgt = target_col or config.get('column')
    if tgt is None:
      num_cols = [c for c in ytr.columns if pd.api.types.is_numeric_dtype(ytr[c].dtype)]
      if 'rul' in ytr.columns:
        tgt = 'rul'
      elif len(num_cols) == 1:
        tgt = num_cols[0]
      else:
        raise ValueError(
          f'could not infer target column; provide config.preprocessing.target.column or target_col; '
          f'candidates={num_cols}'
        )
    ytr_series = ytr[tgt]

  # --- clip before scaling ---
  def clip_clip(s: pd.Series) -> pd.Series:
    return s if clip is None or float(clip) == 0 else s.clip(upper=float(clip))
    
  # --- define transforms (fit on train only) ---
  if method == 'none':
    params = {}
    fwd = lambda s: clip_clip(s)
  elif method == 'standard':
    s_clipped = clip_clip(ytr_series)
    mu = float(s_clipped.mean())
    sd = float(s_clipped.std(ddof=0)) or 1.0
    params = {'mu': mu, 'sd': sd}
    fwd = lambda s: (clip_clip(s) - mu) / sd
  elif method == 'minmax':
    mn, mx = 0.0, clip
    rng = (mx - mn) or 1.0
    params = {'min': mn, 'max': mx}
    fwd = lambda s: (clip_clip(s) - mn) / rng
  else:
    raise ValueError(f'unknown target scaling method: {method}')
  # --- apply transform to all y_* splits ---
  out = {k: v for k, v in data.items()}
  for k, v in list(out.items()):
    if not k.startswith('y_'):
      continue
    if isinstance(v, pd.Series):
      out[k] = fwd(v)
    elif isinstance(v, pd.DataFrame):
      if tgt not in v.columns:
        out[k] = v.copy()
      else:
        dfc = v.copy()
        dfc[tgt] = fwd(dfc[tgt])
        out[k] = dfc
    else:
      out[k] = v
  # --- metadata ---
  meta = out.get('meta', {}).copy()
  meta['y_target'] = tgt
  meta['y_scaler'] = {'method': method, 'params': params, 'clip': clip}
  out['meta'] = meta
  return out



def invert_target(y_scaled: np.ndarray, meta: dict) -> np.ndarray:
  ''' Invert target scaling using meta['y_scaler'].

      Params:
        y_scaled: scaled target values (array-like)
        meta: dataset meta dictionary containing 'y_scaler' entry

      Returns:
        np.ndarray of unscaled target values
  '''
  ys = (meta or {}).get('y_scaler') or {}
  method = ys.get('method', 'none')
  p = ys.get('params', {}) or {}
  mu = p.get('mu', 0.0)
  sd = p.get('sd', p.get('std', 1.0))
  mn = p.get('min', 0.0)
  mx = p.get('max', 1.0)
  a = np.asarray(y_scaled, dtype='float64')
  if method == 'none':
    out = a
  elif method == 'standard':
    out = a * sd + mu
  elif method == 'minmax':
    out = a * (mx - mn) + mn
  else:
    raise ValueError(f'unknown scaling: {method}')
  return np.maximum(out, 0.0)


def normalize(data: dict[str, Any], config: dict, drop_os_cols: bool = True) -> dict[str, Any]:
  ''' Normalize x_* splits using clustering-based normalization (ClusterNorm).

      Config:
        normalization:
          method: opc_mode
          os_cols: [s1, s2, ...]         # required for opc_mode
          sensor_cols: 'auto'|[s1, s2,...]
          kmax: 6
          round_decimals: 1
          min_level_frac: 0.05
          random_state: 42
          enable_gmm: False
          add_mode_onehot: False

      If os_cols is not provided or method != 'opc_mode', falls back to classical normalization.
  '''
  p = dict(config.get('params', {}))

  def g(key, default):
    return config.get(key, p.get(key, default))

  os_cols = set(config.get('os_cols', {}))
  if config.get('method') == 'none':
    logger.info('Normalization method is "none"; skipping normalization.')
    return data
  if not os_cols or config.get('method') != 'opc_mode':
    logger.info('Using classical normalization instead.')
    return classical_normalize(config=config, data=data)
  os_cols = [col for col in data['x_train'].columns if col in os_cols] # yapf: disable
  sensor_cols = config.get('sensor_cols', 'auto')
  if sensor_cols == 'auto':
    numeric = data['x_train'].select_dtypes(include=[np.number]).columns.tolist()
    ignore = set(os_cols) | ({'unit', 'cycle', 'time', 'timestamp', 'rul', 'target'} | set(config.get('ignore', {})))
    sensor_cols = [c for c in numeric if c not in ignore]
  else:
    ignore = set(os_cols) | set(config.get('ignore', []))
  smn = ClusterNorm(os_cols=os_cols,
                    sensor_cols=sensor_cols,
                    kmax=int(g('kmax', 6)),
                    round_decimals=int(g('round_decimals', 1)),
                    min_level_frac=float(g('min_level_frac', 0.05)),
                    random_state=int(g('random_state', 42)),
                    enable_gmm=bool(g('enable_gmm', False)),
                    add_mode_onehot=bool(g('add_mode_onehot', False)))
  t0 = time.perf_counter()
  smn.fit(data['x_train'])
  keys = [k for k in ('x_train', 'x_val', 'x_test') if k in data]
  out = data.copy()
  for k in keys:
    out[k] = smn.transform(df=data[k], add_mode_col=True)
    if drop_os_cols:
      out[k].drop(columns=os_cols, inplace=True)  
  time_delta = time.perf_counter() - t0
  logger.info(f'Finding Operating Modes via Clustering Methods took {time_delta:.2f}s: method={smn.method}, k={smn.meta.get('k', 'N/A')}') # yapf: disable
  out['meta'] = data.get('meta', {}).copy()
  out['meta']['normalization'] = smn.meta
  out['meta']['normalization']['cluster_search_time'] = time_delta
  return out

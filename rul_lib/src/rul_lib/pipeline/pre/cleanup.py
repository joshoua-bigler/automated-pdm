import numpy as np
import pandas as pd
# local
from rul_lib.gls.gls import logger


def cmapps_preprocessing(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
  ''' Simple cleanup: remove constant and low-variance features '''
  data_reduced = {}
  for key, df in data.items():
    data_only = df.drop(columns=['unit'])
    variances = data_only.var()
    constant_cols = variances[variances == 0].index.tolist()
    low_var_cols = variances[variances < 1e-3].index.tolist()
    data_reduced[key] = df.drop(columns=set(constant_cols + low_var_cols))
    logger.info(f'{key}: Dropped {len(constant_cols)} constant columns and {len(low_var_cols)} low variance columns')
  return data_reduced


def drop_constant_features(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
  ''' Drop constant features detected on x_train and apply to all splits.

      Safety improvements
      -------------------
      - Never consider obvious id/meta columns (e.g., 'unit', 'cycle', 'os', 'split',
        'timestamp', 'rul', 'label', 'target') for dropping.
      - Only detect constants among numeric columns (feature-like).
      - If meta.id_columns is provided, also protect those from being dropped.

      Returns
      -------
      dict with same keys as input (x_train, x_val, x_test, …) and an
      additional 'meta' entry containing the dropped columns.
  '''
  if 'x_train' not in data:
    raise ValueError('data must include x_train')
  xtr = data['x_train']
  if not isinstance(xtr, pd.DataFrame):
    raise TypeError('x_train must be a DataFrame')
  # Build an ignore set for id/meta columns
  default_ignore = ['unit', 'cycle', 'os', 'split', 'timestamp', 'rul', 'label', 'target']
  meta = data.get('meta', {}) or {}
  id_cols = meta.get('id_columns', []) or []
  ignore: set[str] = set(default_ignore) | {c for c in id_cols if isinstance(c, str)}
  # Consider only numeric, non-ignored columns for constant detection
  candidates = [c for c in xtr.columns if c not in ignore and pd.api.types.is_numeric_dtype(xtr[c])]
  if not candidates:
    logger.info('drop_constant_features: no candidate numeric columns to check')
    return data
  nunique = xtr[candidates].nunique(dropna=False)
  const_cols = nunique[nunique <= 1].index.tolist()
  out: dict[str, pd.DataFrame] = {}
  for k, df in data.items():
    if isinstance(df, pd.DataFrame):
      out[k] = df.drop(columns=const_cols, errors='ignore').copy()
    else:
      out[k] = df
  meta_out = data.get('meta', {}).copy()
  meta_out['dropped_constant'] = const_cols
  meta_out['dropped_constant_details'] = {'ignored_columns': sorted(ignore), 'checked_candidates': candidates}
  logger.info(f"Dropped {len(const_cols)} constant numeric columns (ignored id/meta: {sorted(ignore)}): {const_cols}")
  out['meta'] = meta_out
  return out


def handle_missing_values(data: dict[str, pd.DataFrame], method: str = 'ffill_bfill') -> dict[str, pd.DataFrame]:
  ''' Handle missing values in the data.
  
      Parameters
      ----------
      data: dict with keys like 'x_train', 'x_val', 'x_test'.
      method: strategy for imputation.
        - 'ffill_bfill': forward fill, then backward fill if still missing
        - 'mean': fill with column mean (computed from train only)
        - 'median': fill with column median (from train only)
        - 'zero': fill with 0
      
      Returns
      -------
      dict with same keys as input, with missing values filled.
  '''
  if 'x_train' not in data:
    raise ValueError('data must include x_train')
  logger.info(f'Imputing missing values using method: {method}')
  out = {}
  xtr = data.get('x_train')
  if method == 'none':
    data['meta'] = {'impute_method': 'none'}
    return data
  if method in ['mean', 'median']:
    stats = {}
    for c in xtr.columns:
      if method == 'mean':
        stats[c] = xtr[c].mean(skipna=True)
      else:
        stats[c] = xtr[c].median(skipna=True)
  for k, df in data.items():
    if not isinstance(df, pd.DataFrame):
      out[k] = df  # carry over non-DataFrame entries untouched
      continue
    if method == 'ffill_bfill':
      out[k] = df.ffill().bfill().copy()
    elif method in ['mean', 'median']:
      out[k] = df.fillna(value=stats).copy()
    elif method == 'zero':
      out[k] = df.fillna(0).copy()
    else:
      logger.warning(f'Unknown imputation method: {method}. Skipping imputation.')
      return data
  out['meta'] = {'impute_method': method}
  return out


def cleanup_features(data: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
  ''' Cleanup features by dropping specified columns.

      Config:
        drop: list of column names to drop

      Returns
      -------
      dict with same keys as input, with specified columns dropped.
  '''
  drop = config.get('drop', [])
  if not drop:
    return data
  logger.info(f'Dropping columns: {drop}')
  out = {}
  for k, df in data.items():
    if isinstance(df, pd.DataFrame):
      out[k] = df.drop(columns=drop, errors='ignore').copy()
    else:
      out[k] = df
  return out


def resample(data: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
  ''' Resample TRAIN (and optionally TEST).

      Modes
      -----
      1) Row-level (default, per_cycle=False):
         - Keep all degrading rows (RUL <= clip).
         - Keep a fraction of healthy rows per unit.

      2) Cycle-level (per_cycle=True, requires cycle_col):
         - Define cycles by (unit_col, cycle_col).
         - Keep ALL rows of degrading cycles (RUL <= clip).
         - Keep ALL rows of a fraction of healthy cycles per unit.

      Expected keys in data:
        x_train, y_train[, x_test, y_test]

      Config:
        apply: bool (default True)
        apply_test: bool (default = apply)
        clip: float (default 125)
        keep_healthy_fraction: float (default 0.2)
        keep_healthy_fraction_test: float (default = train)
        seed: int (default 42)
        group_by: str (default 'unit')       -> column in X
        rul_col: str (default 'rul')         -> column in y (if DataFrame)
        per_cycle: bool (default False)
        cycle_col: str (default 'cycle')
  '''

  def _to_1d_target(y, target_col: str) -> np.ndarray:
    if isinstance(y, pd.DataFrame):
      if target_col not in y.columns:
        raise ValueError(f'missing target_col "{target_col}"')
      return y[target_col].to_numpy()
    if isinstance(y, pd.Series):
      return y.to_numpy()
    return np.asarray(y)

  def _putback_y(y_orig, y_vec_new, keep_pos_sorted):
    if isinstance(y_orig, pd.DataFrame):
      return y_orig.iloc[keep_pos_sorted].reset_index(drop=True)
    if isinstance(y_orig, pd.Series):
      return y_orig.iloc[keep_pos_sorted].reset_index(drop=True)
    return y_vec_new[keep_pos_sorted]

  def _resample_pair_row_level(x: pd.DataFrame, y, unit_col: str, target_col: str, clip: float, frac: float, seed: int,
                               split_name: str) -> tuple[pd.DataFrame, object, np.ndarray, dict]:
    if not isinstance(x, pd.DataFrame):
      raise TypeError(f'{split_name}: X must be a DataFrame')
    if unit_col not in x.columns:
      raise ValueError(f'{split_name}: missing column in X: {unit_col}')
    y_vec = _to_1d_target(y, target_col)
    if len(y_vec) != len(x):
      raise ValueError(f'{split_name}: length mismatch between X and y')
    pos = np.arange(len(x))
    meta = pd.DataFrame({'pos': pos, unit_col: x[unit_col].to_numpy(), 'rul_end': y_vec})
    degr_mask = meta['rul_end'] <= clip
    healthy_mask = ~degr_mask
    degr_pos = meta.loc[degr_mask, 'pos'].to_numpy()
    if healthy_mask.any() and frac > 0.0:
      healthy_sub = meta.loc[healthy_mask].groupby(unit_col, group_keys=False).sample(frac=frac, random_state=seed)
      healthy_pos = healthy_sub['pos'].to_numpy()
    else:
      healthy_pos = np.empty(0, dtype=int)
    keep_pos = np.concatenate([degr_pos, healthy_pos])
    kept_meta = meta.loc[meta['pos'].isin(keep_pos)].copy()
    kept_meta.sort_values([unit_col, 'pos'], inplace=True)
    keep_pos_sorted = kept_meta['pos'].to_numpy()
    x_new = x.iloc[keep_pos_sorted].reset_index(drop=True)
    y_new = _putback_y(y, y_vec, keep_pos_sorted)
    counts = {
        f'{split_name}_orig': int(len(x)),
        f'{split_name}_degr_kept': int(len(degr_pos)),
        f'{split_name}_healthy_orig': int(healthy_mask.sum()),
        f'{split_name}_healthy_kept': int(len(healthy_pos)),
        f'{split_name}_final': int(len(keep_pos_sorted)),
    }
    return x_new, y_new, keep_pos_sorted, counts

  def _resample_pair_cycle_level(x: pd.DataFrame, y, unit_col: str, cycle_col: str, target_col: str, clip: float,
                                 frac: float, seed: int,
                                 split_name: str) -> tuple[pd.DataFrame, object, np.ndarray, dict]:
    if not isinstance(x, pd.DataFrame):
      raise TypeError(f'{split_name}: X must be a DataFrame')
    if unit_col not in x.columns:
      raise ValueError(f'{split_name}: missing column in X: {unit_col}')
    if cycle_col not in x.columns:
      raise ValueError(f'{split_name}: missing cycle_col "{cycle_col}" in X for cycle-level resampling')

    y_vec = _to_1d_target(y, target_col)
    if len(y_vec) != len(x):
      raise ValueError(f'{split_name}: length mismatch between X and y')

    pos = np.arange(len(x))
    meta_rows = pd.DataFrame({
        'pos': pos,
        unit_col: x[unit_col].to_numpy(),
        cycle_col: x[cycle_col].to_numpy(),
        'rul_end': y_vec
    })

    # One RUL per (unit, cycle): use first row's RUL within that cycle
    meta_cycles = (meta_rows.groupby([unit_col, cycle_col], as_index=False).agg(first_pos=('pos', 'min'),
                                                                                rul_end=('rul_end', 'first'),
                                                                                n_rows=('pos', 'size')))

    degr_mask = meta_cycles['rul_end'] <= clip
    healthy_mask = ~degr_mask

    degr_cycles = meta_cycles.loc[degr_mask, [unit_col, cycle_col]]
    if healthy_mask.any() and frac > 0.0:
      healthy_cycles = meta_cycles.loc[healthy_mask, [unit_col, cycle_col]]
      healthy_sub = healthy_cycles.groupby(unit_col, group_keys=False).sample(frac=frac, random_state=seed)
    else:
      healthy_sub = meta_cycles.iloc[0:0][[unit_col, cycle_col]]

    keep_cycles = pd.concat([degr_cycles, healthy_sub], ignore_index=True).drop_duplicates()

    # Map kept (unit, cycle) pairs back to row positions
    meta_keep = meta_rows.merge(keep_cycles, on=[unit_col, cycle_col], how='inner')
    meta_keep.sort_values([unit_col, 'pos'], inplace=True)
    keep_pos_sorted = meta_keep['pos'].to_numpy()

    x_new = x.iloc[keep_pos_sorted].reset_index(drop=True)
    y_new = _putback_y(y, y_vec, keep_pos_sorted)

    counts = {
        f'{split_name}_orig_rows': int(len(x)),
        f'{split_name}_orig_cycles': int(len(meta_cycles)),
        f'{split_name}_degr_cycles_kept': int(len(degr_cycles)),
        f'{split_name}_healthy_cycles_orig': int(healthy_mask.sum()),
        f'{split_name}_healthy_cycles_kept': int(len(healthy_sub)),
        f'{split_name}_final_rows': int(len(keep_pos_sorted)),
    }
    return x_new, y_new, keep_pos_sorted, counts

  # --- config ---
  apply_train = bool(config.get('apply', True))
  apply_test = bool(config.get('apply_test', apply_train))
  unit_col = str(config.get('group_by', 'unit'))
  target_col = str(config.get('rul_col', 'rul'))
  clip = float(config.get('clip', 125))
  frac_tr = float(config.get('keep_healthy_fraction', 0.2))
  frac_te = float(config.get('keep_healthy_fraction_test', frac_tr))
  seed = int(config.get('seed', 42))
  per_cycle = bool(config.get('per_cycle', False))
  cycle_col = str(config.get('cycle_col', 'cycle'))

  if not apply_train and not apply_test:
    return data

  out = dict(data)
  meta_out = out.get('meta', {}).copy()
  meta_out['columns'] = {'unit_col': unit_col, 'target_col': target_col}
  meta_out['resampling_params'] = {
      'clip': clip,
      'seed': seed,
      'train_keep_healthy_fraction': frac_tr,
      'test_keep_healthy_fraction': frac_te,
      'per_cycle': per_cycle,
      'cycle_col': cycle_col,
  }

  # --- helper selector ---
  def _resample_pair(x, y, frac, split_name: str):
    if per_cycle:
      return _resample_pair_cycle_level(x=x,
                                        y=y,
                                        unit_col=unit_col,
                                        cycle_col=cycle_col,
                                        target_col=target_col,
                                        clip=clip,
                                        frac=frac,
                                        seed=seed,
                                        split_name=split_name)
    return _resample_pair_row_level(x=x,
                                    y=y,
                                    unit_col=unit_col,
                                    target_col=target_col,
                                    clip=clip,
                                    frac=frac,
                                    seed=seed,
                                    split_name=split_name)

  # --- train ---
  if apply_train:
    xtr, ytr = out['x_train'], out['y_train']
    xtr_new, ytr_new, keep_tr, counts_tr = _resample_pair(xtr, ytr, frac_tr, 'train')
    out['x_train'], out['y_train'] = xtr_new, ytr_new
    out['train_keep_pos'] = keep_tr
    meta_out.update(counts_tr)

  # --- test ---
  has_test = ('x_test' in out) and ('y_test' in out)
  if apply_test and has_test:
    xte, yte = out['x_test'], out['y_test']
    xte_new, yte_new, keep_te, counts_te = _resample_pair(xte, yte, frac_te, 'test')
    out['x_test'], out['y_test'] = xte_new, yte_new
    out['test_keep_pos'] = keep_te
    meta_out.update(counts_te)

  out['meta'] = meta_out

  if 'x_train' in out and unit_col not in out['x_train'].columns:
    raise RuntimeError(f'missing required column {unit_col!r} after train resampling')
  if 'x_test' in out and apply_test and has_test and unit_col not in out['x_test'].columns:
    raise RuntimeError(f'missing required column {unit_col!r} after test resampling')
  return out

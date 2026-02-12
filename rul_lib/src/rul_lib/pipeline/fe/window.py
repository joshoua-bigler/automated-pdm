import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from typing import Any
# local
from rul_lib.gls.gls import logger


def grouped_sliding_windows(data: dict[str, pd.DataFrame], params: dict[str, Any]) -> dict[str, np.ndarray]:
  ''' Create grouped sliding windows from time series data.

      Parameters
      ----------
      data :
      Input data containing 'x_train', 'y_train', 'x_test', 'y_test' DataFrames.

      params :
      - size : int, window size (default: 64)
      - stride : int, window stride (default: 1)
      - group_by : str, column name to group by (default: 'unit')
      - by_cycle : bool, whether to create windows per cycle (default: False)
      - fast : bool, if True and by_cycle=True, creates one window per cycle (default: False)
      - drop_first : bool, whether to drop the first partial window (default: False)
      - drop_last : bool, whether to drop the last partial window (default: False)
      - encode_unit : bool, whether to encode unit IDs as integers (default: False)
      - os_col : str, column name for operating settings (default: 'os')
      - encode_os : bool, whether to encode operating settings as integers (default: False)

      Returns
      -------
      dict[str, np.ndarray]
  '''
  size = int(params.get('size', 64))
  stride = int(params.get('stride', 1))
  group_by = params.get('group_by', 'unit')
  by_cycle = bool(params.get('by_cycle', False))
  fast = bool(params.get('fast', False))
  drop_first = bool(params.get('drop_first', False))
  drop_last = bool(params.get('drop_last', False))
  encode_unit = bool(params.get('encode_unit', False))
  os_col = params.get('os_col', 'os')
  encode_os = bool(params.get('encode_os', False))

  if fast and not by_cycle:
    raise ValueError('fast=True currently requires by_cycle=True')

  required = {'x_train', 'y_train', 'x_test', 'y_test'}
  if not required.issubset(data.keys()):
    raise ValueError(f'data must contain {required}')

  def _select_feat_cols(df: pd.DataFrame, split: str) -> list[str]:
    exclude = {group_by, 'cycle', 'rul', os_col}
    feat_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
      raise ValueError(f'no numeric feature columns found in {split}')
    return feat_cols

  def _compute_starts(n: int, win_size: int) -> tuple[np.ndarray, int]:
    max_start = n - win_size
    if max_start < 0:
      return np.empty((0,), dtype=int), 0

    if drop_first:
      first_start = max_start % stride
      dropped = first_start
      starts = np.arange(first_start, max_start + 1, stride, dtype=int)
      return starts, dropped

    if drop_last:
      last_valid = (max_start // stride) * stride
      dropped = max_start - last_valid
      starts = np.arange(0, last_valid + 1, stride, dtype=int)
      return starts, dropped

    dropped = 0 if (max_start % stride == 0) else stride - (max_start % stride)
    starts = np.arange(0, max_start + 1, stride, dtype=int)
    if starts.size and starts[-1] != max_start:
      starts = np.concatenate([starts, np.array([max_start], dtype=int)])
    return starts, dropped

  def _make_windows(x_df: pd.DataFrame, y_df: pd.DataFrame, feat_cols: list[str],
                    split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if group_by not in x_df.columns or group_by not in y_df.columns:
      raise ValueError(f'missing {group_by!r} column in input data')
    if 'rul' not in y_df.columns:
      raise ValueError('y_* must contain a \'rul\' column')

    missing = [c for c in feat_cols if c not in x_df.columns]
    if missing:
      raise ValueError(f'{split}: missing feature columns: {missing}')

    non_num = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(x_df[c])]
    if non_num:
      raise ValueError(f'{split}: non-numeric feature columns: {non_num}')

    if by_cycle:
      if 'cycle' not in x_df.columns or 'cycle' not in y_df.columns:
        raise ValueError('by_cycle=True requires a \'cycle\' column in x_* and y_*')
      rul_map = y_df.set_index([group_by, 'cycle'])['rul'].to_dict()
      grouped = x_df.groupby([group_by, 'cycle'], sort=False)
    else:
      y_grouped = y_df.groupby(group_by, sort=False)['rul']
      rul_arr_map = {k: v.to_numpy() for k, v in y_grouped}
      grouped = x_df.groupby(group_by, sort=False)

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    unit_blocks: list[np.ndarray] = []
    lengths: list[int] = []

    total_points = 0
    total_windows = 0
    total_dropped = 0

    for key, gx in grouped:
      if by_cycle:
        unit_id, cycle_id = key
        if (unit_id, cycle_id) not in rul_map:
          logger.warning(f'unit {unit_id}, cycle {cycle_id}: no matching label, skipped')
          continue
        rul_val = float(rul_map[(unit_id, cycle_id)])
      else:
        unit_id = key
        if unit_id not in rul_arr_map:
          logger.warning(f'unit {unit_id}: no matching labels, skipped')
          continue
        rul_arr = rul_arr_map[unit_id]

      x = gx[feat_cols].to_numpy(copy=False)
      n = x.shape[0]
      total_points += n

      if fast and by_cycle:
        if n == 0:
          logger.warning(f'unit {unit_id}: empty segment, skipped')
          continue
        X_blocks.append(x[None, :, :])
        y_blocks.append(np.array([rul_val], dtype=float))
        unit_blocks.append(np.array([unit_id], dtype=object))
        lengths.append(n)
        total_windows += 1
        logger.debug(f'unit {unit_id}: length={n}, windows=1, dropped_points=0, fast={fast}')
        continue

      if n < size:
        logger.warning(f'unit {unit_id}: segment length {n} < window_size {size}, skipped')
        continue

      starts, dropped = _compute_starts(n=n, win_size=size)
      if starts.size == 0:
        continue

      total_dropped += dropped
      total_windows += int(starts.size)

      w_all = sliding_window_view(x, window_shape=(size, x.shape[1]))[:, 0, :, :]
      w_sel = w_all[starts]

      if by_cycle:
        y_sel = np.full((w_sel.shape[0],), rul_val, dtype=float)
      else:
        end_idx = np.minimum(starts + size - 1, rul_arr.shape[0] - 1)
        y_sel = rul_arr[end_idx].astype(float, copy=False)

      u_sel = np.full((w_sel.shape[0],), unit_id, dtype=object)

      X_blocks.append(w_sel)
      y_blocks.append(y_sel)
      unit_blocks.append(u_sel)

      logger.debug(f'unit {unit_id}: length={n}, windows={starts.size}, dropped_points={dropped}, fast={fast}')

    if total_points == 0:
      raise RuntimeError('no data available after windowing')

    pct = 100.0 * total_dropped / total_points
    logger.info(f'window summary: total_points={total_points}, total_windows={total_windows}, '
                f'total_dropped={total_dropped} ({pct:.2f}% of points)')

    if not X_blocks:
      empty_x = np.empty((0, size, len(feat_cols)), dtype=float)
      empty_y = np.empty((0,), dtype=float)
      empty_u = np.empty((0,), dtype=object)
      return empty_x, empty_y, empty_u

    if fast and by_cycle:
      max_len = int(max(lengths)) if lengths else 0
      n_feat = int(X_blocks[0].shape[2])
      X = np.zeros((len(X_blocks), max_len, n_feat), dtype=float)
      for i, w in enumerate(X_blocks):
        L = int(w.shape[1])
        X[i, :L, :] = w[0]
      y = np.concatenate(y_blocks, axis=0)
      units = np.concatenate(unit_blocks, axis=0)
      return X, y, units

    X = np.concatenate(X_blocks, axis=0)
    y = np.concatenate(y_blocks, axis=0)
    units = np.concatenate(unit_blocks, axis=0)
    return X, y, units

  feat_cols_tr = _select_feat_cols(data['x_train'], split='x_train')
  feat_cols_te = _select_feat_cols(data['x_test'], split='x_test')

  missing_in_test = [c for c in feat_cols_tr if c not in feat_cols_te]
  if missing_in_test:
    logger.warning(f'x_test is missing feature columns present in x_train: {missing_in_test}. '
                   'Using intersection to align features.')

  feat_cols = [c for c in feat_cols_tr if c in feat_cols_te]
  if not feat_cols:
    raise ValueError('no common numeric feature columns between x_train and x_test')

  x_tr, y_tr, u_tr = _make_windows(data['x_train'], data['y_train'], feat_cols=feat_cols, split='x_train')
  x_te, y_te, u_te = _make_windows(data['x_test'], data['y_test'], feat_cols=feat_cols, split='x_test')

  result: dict[str, Any] = {
      'x_train': x_tr,
      'y_train': y_tr,
      'unit_train': u_tr,
      'x_test': x_te,
      'y_test': y_te,
      'unit_test': u_te,
      'meta': {
          'features': feat_cols
      },
  }

  if encode_unit:
    all_units = pd.Index(np.concatenate([u_tr, u_te])).unique()
    unit_categories = all_units.tolist()
    unit_mapping = {u: i for i, u in enumerate(unit_categories)}

    def _encode_units(u_arr: np.ndarray) -> np.ndarray:
      return np.array([unit_mapping[u] for u in u_arr], dtype=int)

    result['unit_int_train'] = _encode_units(u_tr)
    result['unit_int_test'] = _encode_units(u_te)
    result['meta']['unit_categories'] = unit_categories
    result['meta']['unit_mapping'] = unit_mapping

  if encode_os:
    if os_col not in data['x_train'].columns or os_col not in data['x_test'].columns:
      raise ValueError(f'encode_os=True but {os_col!r} not found in x_train/x_test')

    df_os = pd.concat([data['x_train'][[group_by, os_col]], data['x_test'][[group_by, os_col]]],
                      axis=0,
                      ignore_index=True)
    df_os = df_os.drop_duplicates(subset=[group_by]).set_index(group_by)
    unit_to_os = df_os[os_col].to_dict()

    os_tr = np.array([unit_to_os[u] for u in u_tr])
    os_te = np.array([unit_to_os[u] for u in u_te])

    all_os = pd.Index(np.concatenate([os_tr, os_te])).unique()
    os_categories = all_os.tolist()
    os_mapping = {o: i for i, o in enumerate(os_categories)}

    def _encode_os(os_arr: np.ndarray) -> np.ndarray:
      return np.array([os_mapping[o] for o in os_arr], dtype=int)

    result['os_train'] = os_tr
    result['os_test'] = os_te
    result['os_int_train'] = _encode_os(os_tr)
    result['os_int_test'] = _encode_os(os_te)
    result['meta']['os_categories'] = os_categories
    result['meta']['os_mapping'] = os_mapping

  return result

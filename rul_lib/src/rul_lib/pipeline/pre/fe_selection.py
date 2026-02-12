import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
# local
from rul_lib.gls.gls import logger


def select_by_correlation(data: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
  ''' Drop redundant features using Pearson correlation on x_train.

      Behavior
      --------
      - Compute absolute Pearson correlation matrix on numeric features.
      - Compute global dependence score d_i = mean_j |corr(i,j)|.
      - Greedy prune: keep feature with highest d_i, drop its partners >= threshold.
      - Apply selected columns to all x_* splits and store metadata.

      Config
      ------
      apply: bool = True
      correlation_threshold: float = 0.9
      ignore: list[str] = []
  '''
  if 'x_train' not in data:
    raise ValueError('x_train required')
  if not config.get('apply', True):
    return data
  thr = float(config.get('correlation_threshold', 0.9))
  ignore = set(config.get('ignore', []) or [])
  xtr = data['x_train']
  feats = [c for c in xtr.columns if c not in ignore and pd.api.types.is_numeric_dtype(xtr[c])]
  if len(feats) < 2:
    return data
  X_df = xtr[feats]
  Xc = X_df.dropna(axis=0, how='any')
  if Xc.shape[0] < 2:
    return data
  C = Xc.corr(method='pearson').astype('float64').to_numpy()
  A = np.abs(C)
  np.fill_diagonal(A, 0.0)
  feats_use = feats
  mean_dep = A.mean(axis=0)
  order = [feats_use[i] for i in np.argsort(-mean_dep)]
  keep = set(feats_use)
  dropped = []
  for c in order:
    if c not in keep:
      continue
    i = feats_use.index(c)
    partners = [feats_use[j] for j in range(len(feats_use)) if A[i, j] >= thr and feats_use[j] in keep]
    for p in partners:
      if p != c and p in keep:
        keep.remove(p)
        dropped.append(p)
        logger.info(f'dropping correlated feature {p!r} (corr >= {thr} with {c!r})')
  kept_cols = [c for c in xtr.columns if (c in keep) or (c not in feats)]
  out: dict[str, pd.DataFrame] = {}
  for k, df in data.items():
    if isinstance(df, pd.DataFrame) and k.startswith('x_'):
      out[k] = df[kept_cols].copy()
    else:
      out[k] = df
  meta = out.get('meta', {}).copy()
  meta['sensor_selection_corr'] = {
      'method': 'pearson',
      'corr_threshold': thr,
      'ignored': sorted(ignore),
      'kept': [c for c in kept_cols if c in feats_use],
      'dropped_corr': sorted(set(dropped)),
      'n_rows_used': int(Xc.shape[0])
  }
  out['meta'] = meta
  return out


def select_topk_mi(data: dict[str, pd.DataFrame], config: dict) -> dict[str, pd.DataFrame]:
  ''' Keep top-k features by mutual information with y_train[target_col]; computed on x_train only.

      Config
      ------
      apply: bool = True
      target_col: str = 'rul'
      top_k: int = 12
      ignore: list[str] = []           # e.g. ['unit', 'cycle']
      n_neighbors: int = 3
      random_state: int | None = 0
  '''
  if 'x_train' not in data or 'y_train' not in data:
    raise ValueError('x_train and y_train required')
  if not config.get('apply', True):
    return data
  target_col = str(config.get('target_col', 'rul'))
  top_k = int(config.get('top_k', 12))
  ignore = set(config.get('ignore', []) or [])
  n_neighbors = int(config.get('n_neighbors', 3))
  random_state = config.get('random_state', 0)
  xtr = data['x_train']
  ytr = data['y_train']
  # --- select candidate features
  feats = [c for c in xtr.columns if c not in ignore and pd.api.types.is_numeric_dtype(xtr[c])]
  if len(feats) < 1:
    return data
  # --- prepare target vector
  if isinstance(ytr, pd.DataFrame):
    if target_col not in ytr.columns:
      raise ValueError(f'target column "{target_col}" not in y_train')
    y = ytr[target_col].to_numpy(dtype='float64', copy=True)
  else:
    y = ytr.to_numpy(dtype='float64', copy=True)
  # --- prepare X matrix and impute NaNs (mean)
  X = xtr[feats].to_numpy(dtype='float64', copy=True)
  if np.isnan(X).any():
    col_means = np.nanmean(X, axis=0)
    ridx, cidx = np.where(np.isnan(X))
    X[ridx, cidx] = np.take(col_means, cidx)
  # --- compute mutual information
  mi = mutual_info_regression(X, y, n_neighbors=n_neighbors, random_state=random_state)
  pairs = sorted(zip(feats, mi), key=lambda kv: kv[1], reverse=True)
  k = min(top_k, len(pairs))
  kept_feats = [f for f, _ in pairs[:k]]
  dropped_feats = [f for f, _ in pairs[k:]]
  # --- apply selection to all splits
  out: dict[str, pd.DataFrame] = {}
  for key, df in data.items():
    if isinstance(df, pd.DataFrame) and key.startswith('x_'):
      extra = [c for c in df.columns if c not in feats]  # preserve id/meta cols
      out[key] = df[kept_feats + extra].copy()
    else:
      out[key] = df
  # --- record in meta
  meta = out.get('meta', {}).copy()
  meta['sensor_selection_mi'] = {
      'target': target_col,
      'top_k': int(top_k),
      'n_neighbors': int(n_neighbors),
      'random_state': random_state,
      'kept': kept_feats,
      'dropped': dropped_feats,
      'mi_scores': {
          f: float(s) for f, s in pairs
      },
      'n_rows_used': int(X.shape[0]),
  }
  out['meta'] = meta
  return out

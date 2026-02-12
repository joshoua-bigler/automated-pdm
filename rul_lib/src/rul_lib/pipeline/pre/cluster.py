import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import calinski_harabasz_score, silhouette_score


@dataclass
class ClusterNorm:
  ''' Cluster-based normalizer for sensor data.
      Clusters operating modes based on given operating setting columns, and normalizes sensor data within each cluster.
      If operating mode cannot be determined (unseen setting), falls back to global normalization.
      Methods supported: 'unique' (exact unique modes), 'kmeans', 'gmm' (Gaussian Mixture Model with BIC model selection).


      Parameters
      ----------
      os_cols: 
        List of operating setting column names.
      sensor_cols: 
        List of sensor column names to be normalized.
      kmax: 
        Maximum number of clusters to consider.
      round_decimals: 
        Number of decimal places to round operating settings for 'unique' method.
      min_level_frac: 
        Minimum fraction of data points per cluster for 'unique' method.
      random_state: 
        Random state for reproducibility.
      enable_gmm: 
        Whether to enable GMM clustering with BIC model selection.
      add_mode_onehot: 
        Whether to add one-hot encoded operating mode columns to output.
      
      Fitted attributes
      -----------------
      method: 
        The clustering method used ('unique', 'kmeans', 'gmm').
      os_scaler: 
        StandardScaler for operating settings (if applicable).
      kmeans: 
        Fitted KMeans model (if applicable).
      gmm: 
        Fitted GaussianMixture model (if applicable).
      unique_map: 
        Mapping of unique operating settings to mode indices (if applicable).
      scalers: 
        Dictionary of StandardScalers for each operating mode.
      global_scaler: 
        StandardScaler for global normalization fallback.
      meta: 
        Dictionary of metadata including clustering diagnostics (CH index, silhouette score, etc.).
  '''
  os_cols: list[str]
  sensor_cols: list[str]
  kmax: int = 6
  round_decimals: int = 1
  min_level_frac: float = 0.05
  random_state: int = 42
  enable_gmm: bool = False
  add_mode_onehot: bool = False
  # fitted
  method: str | None = None  # 'unique'|'kmeans'|'gmm'
  os_scaler: StandardScaler | None = None
  kmeans: KMeans | None = None
  gmm: GaussianMixture | None = None
  unique_map: dict[tuple, int] | None = None
  scalers: dict[int, StandardScaler] | None = None
  global_scaler: StandardScaler | None = None
  meta: dict = None

  def fit(self, df_train: pd.DataFrame) -> 'ClusterNorm':
    self.meta = {}
    Xraw = df_train[self.os_cols].to_numpy(dtype='float64', copy=False)
    # try exact unique modes first
    rounded = np.round(Xraw, self.round_decimals)
    uniq, inv, cnts = np.unique(rounded, axis=0, return_inverse=True, return_counts=True)
    if len(uniq) <= self.kmax and cnts.min() / len(rounded) >= self.min_level_frac:
      self.method = 'unique'
      self.unique_map = {tuple(u): int(i) for i, u in enumerate(uniq)}
      modes = inv.astype(int)
      self.os_scaler = None
      self.kmeans = None
      self.gmm = None
      self.meta['k'] = len(uniq)
    else:
      # standardize os
      self.os_scaler = StandardScaler().fit(Xraw)
      X = self.os_scaler.transform(Xraw)
      # optionally GMM with BIC model selection
      if self.enable_gmm:
        models, bics = [], []
        for k in range(1, max(2, self.kmax) + 1):
          gm = GaussianMixture(n_components=k, covariance_type='full', random_state=self.random_state)
          gm.fit(X)
          models.append(gm)
          bics.append(gm.bic(X))
        best = int(np.argmin(bics))
        gm = models[best]
        if gm.n_components > 1:
          self.method = 'gmm'
          self.gmm = gm
          self.kmeans = None
          modes = gm.predict(X).astype(int)
          self.meta['k'] = gm.n_components
          self.meta['bic'] = bics
        else:
          self.gmm = None
          modes, k = self._fit_kmeans_with_validity(X)
          self.method = 'kmeans'
          self.meta['k'] = k
      else:
        modes, k = self._fit_kmeans_with_validity(X)
        self.method = 'kmeans'
        self.meta['k'] = k
    self.scalers = {}
    for m in np.unique(modes):
      g = df_train.loc[modes == m, self.sensor_cols]
      sc = StandardScaler().fit(g.to_numpy(dtype='float64', copy=False))
      self.scalers[int(m)] = sc
    self.global_scaler = StandardScaler().fit(df_train[self.sensor_cols].to_numpy(dtype='float64', copy=False))

    # diagnostics
    self.meta['method'] = self.method
    self.meta['ch'] = self._safe_ch(Xraw if self.os_scaler is None else self.os_scaler.transform(Xraw), modes)
    self.meta['sil'] = self._safe_sil(Xraw if self.os_scaler is None else self.os_scaler.transform(Xraw), modes)
    return self

  def _fit_kmeans_with_validity(self, X: np.ndarray) -> tuple[np.ndarray, int]:
    best, best_score, best_labels = None, -np.inf, None
    for k in range(2, max(2, self.kmax) + 1):
      km = KMeans(n_clusters=k, n_init='auto', random_state=self.random_state).fit(X)
      labels = km.labels_.astype(int)
      ch = self._safe_ch(X, labels) or 0.0
      sil = self._safe_sil(X, labels)
      siln = ((sil + 1.0) / 2.0) if sil is not None else 0.0
      score = 0.7 * ch + 0.3 * siln
      if score > best_score:
        best, best_score, best_labels = km, score, labels
    self.kmeans = best
    return best_labels, int(best.n_clusters)

  def _assign_modes(self, df: pd.DataFrame) -> np.ndarray:
    Xraw = df[self.os_cols].to_numpy(dtype='float64', copy=False)
    if self.method == 'unique':
      r = np.round(Xraw, self.round_decimals)
      modes = np.array([self.unique_map.get(tuple(row), -1) for row in r], dtype=int)
      if (modes < 0).any():
        uniq_arr = np.array(list(self.unique_map.keys()), dtype='float64')
        for i in np.where(modes < 0)[0]:
          d = np.linalg.norm(r[i] - uniq_arr, axis=1)
          modes[i] = int(np.argmin(d))
      return modes
    X = self.os_scaler.transform(Xraw) if self.os_scaler is not None else Xraw
    if self.method == 'gmm':
      return self.gmm.predict(X).astype(int)
    if self.method == 'kmeans':
      return self.kmeans.predict(X).astype(int)
    return np.zeros(len(df), dtype=int)

  def _ensure_fitted(self) -> None:
    if self.meta is None:
      raise ValueError('SmartModeNormalizer instance is not fitted yet. Call `fit` first.')

  def transform(self, df: pd.DataFrame, add_mode_col: bool = True) -> pd.DataFrame:
    self._ensure_fitted()
    out = df.copy()
    modes = self._assign_modes(out)
    # normalize sensors (ensure float dtypes before assignment)
    arr = out[self.sensor_cols].to_numpy(dtype='float64', copy=False)
    for m, sc in self.scalers.items():
      mask = modes == int(m)
      if mask.any():
        arr[mask] = sc.transform(arr[mask])
    unseen = ~np.isin(modes, list(self.scalers.keys()))
    if unseen.any():
      arr[unseen] = self.global_scaler.transform(arr[unseen])
    out[self.sensor_cols] = out[self.sensor_cols].astype('float64', copy=False)
    out.loc[:, self.sensor_cols] = arr
    if add_mode_col and self.meta.get('k', 0) > 1:
      out['ops_mode'] = modes.astype(np.int64, copy=False)
    if self.add_mode_onehot:
      dummies = pd.get_dummies(modes, prefix='mode', dtype=np.int8)
      out = pd.concat([out, dummies], axis=1)
    return out

  def _safe_ch(self, X: np.ndarray, labels: np.ndarray) -> float | None:
    try:
      return float(calinski_harabasz_score(X, labels))
    except Exception:
      return None

  def _safe_sil(self, X: np.ndarray, labels: np.ndarray) -> float | None:
    try:
      return float(silhouette_score(X, labels))
    except Exception:
      return None

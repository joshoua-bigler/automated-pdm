import hashlib, json
import pandas as pd
from pathlib import Path
from typing import Protocol
# local
from rul_lib.data.dowsample import downsample

R_ADD_SEC = {
    'Bearing1_3': 5730,
    'Bearing1_4': 2900,
    'Bearing1_5': 1610,
    'Bearing1_6': 1460,
    'Bearing1_7': 7570,
    'Bearing2_3': 7530,
    'Bearing2_4': 1390,
    'Bearing2_5': 3090,
    'Bearing2_6': 1290,
    'Bearing2_7': 580,
    'Bearing3_3': 820
}

SEC_PER_CYCLE = 10
OFFSET_CYC = {k: v / SEC_PER_CYCLE for k, v in R_ADD_SEC.items()}


class DatasetLoader(Protocol):

  def __call__(self, root_path, config, **kwargs) -> dict:
    ...


def load_cmapps(root_path: str | Path, config: dict, **kwargs) -> dict:
  ''' Load CMAPPS dataset from given root path. '''
  fd_number = config.get('fd_number', None)
  if not fd_number:
    raise ValueError('config must contain fd_number')
  if fd_number not in (1, 2, 3, 4):
    raise ValueError('fd_number must be one of 1, 2, 3, 4')
  n_sensors = 21 if fd_number in (1, 2) else 26
  cols = ['unit', 'cycle', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, n_sensors + 1)]
  df_train = pd.read_csv(root_path / f'data/cmapps/train_FD00{fd_number}.txt',
                         sep=r'\s+',
                         header=None,
                         names=cols,
                         engine='python')
  df_test = pd.read_csv(root_path / f'data/cmapps/test_FD00{fd_number}.txt',
                        sep=r'\s+',
                        header=None,
                        names=cols,
                        engine='python')
  df_rul = pd.read_csv(root_path / f'data/cmapps/RUL_FD00{fd_number}.txt',
                       sep=r'\s+',
                       header=None,
                       names=['rul'],
                       engine='python')
  # --- label train rul -------------------------------------------------------
  df_train = df_train.copy()
  df_train['rul'] = df_train.groupby('unit')['cycle'].transform(lambda c: c.max() - c)
  # --- label test rul --------------------------------------------------------
  last_cycles = df_test.groupby('unit')['cycle'].max().reset_index()
  last_cycles = last_cycles.rename(columns={'cycle': 'last_cycle'})
  last_cycles['rul_offset'] = df_rul['rul']
  df_test = df_test.merge(last_cycles, on='unit')
  df_test['rul'] = df_test['last_cycle'] - df_test['cycle'] + df_test['rul_offset']
  df_test = df_test.drop(columns=['last_cycle', 'rul_offset'])
  # --- split into x / y ------------------------------------------------------
  x_train = df_train.drop(columns=['rul', 'cycle'])
  y_train = df_train[['unit', 'rul']]
  x_test = df_test.drop(columns=['rul', 'cycle'])
  y_test = df_test[['unit', 'rul']]
  return {
      'x_train': x_train,
      'y_train': y_train,
      'x_test': x_test,
      'y_test': y_test,
  }


def load_femto(root_path: str | Path, config: dict, use_timestamp: bool = False, **kwargs) -> dict:
  ''' Load FEMTO dataset from given root path. 

      Parameters
      ----------
      root_path: 
        base path to FEMTO data
      config: dict
        dataset configuration dictionary
      use_timestamp:
        if True, compute timestamps from sample indices and sampling rate
        
      Returns
      -------
      dict with:
        - x_train, x_test : pd.DataFrame
        - y_train, y_test : pd.DataFrame
        - meta : dict with dataset info
  '''
  fs = 25600.0  # sampling frequency in Hz
  p = config
  base = Path(root_path) / p.get('relative_path', 'data/femto')
  train_dir, test_dir = base / 'train', base / 'test'
  cache_dir = base / '.cache'

  # simple downsampling cfg
  ds_cfg = dict(p.get('downsampling', {}))
  ds_apply = bool(ds_cfg.get('apply', True))
  ds_factor = int(ds_cfg.get('factor', 8)) if ds_apply else 1
  fs_out = fs / float(ds_factor) if ds_apply and ds_factor > 1 else fs

  if not train_dir.exists() or not test_dir.exists():
    missing = [str(d) for d in (train_dir, test_dir) if not d.exists()]
    raise FileNotFoundError(f'missing split dir(s): {", ".join(missing)}')

  def _units(d):
    return sorted(x.name for x in d.iterdir() if x.is_dir())

  def _idx(path):
    try:
      return int(path.stem.split('_')[-1])
    except Exception:
      return 0

  def _read_csv_robust(fp):
    df = pd.read_csv(fp, header=None, engine='c', on_bad_lines='skip')
    if df.shape[1] == 1:
      df = pd.read_csv(fp, header=None, sep=None, engine='python', on_bad_lines='skip')
    empty = [c for c in df.columns if df[c].isna().all()]
    if empty:
      df = df.drop(columns=empty)
    return df

  def _fingerprint(split_dir):
    items = []
    for pth in sorted(split_dir.rglob('acc_*.csv')):
      try:
        st = pth.stat()
        items.append((str(pth.relative_to(split_dir)), st.st_size, int(st.st_mtime_ns)))
      except FileNotFoundError:
        pass
    payload = json.dumps(items, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha1(payload.encode('utf-8')).hexdigest()

  # ---------- caches ----------
  def _raw_cache_paths(split):
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_ts' if use_timestamp else '_raw'
    return cache_dir / f'{split}{suffix}.parquet', cache_dir / f'{split}{suffix}.manifest.json'

  def _ds_cache_paths(split, kind):  # kind: 'x' or 'y'
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = '_ts' if use_timestamp else '_raw'
    tag_ds = f'_ds{int(ds_apply)}_f{ds_factor}'
    return cache_dir / f'{split}_{kind}{suffix}{tag_ds}.parquet', cache_dir / f'{split}_{kind}{suffix}{tag_ds}.manifest.json'

  def _try_load_raw(split_dir, split):
    pq, man = _raw_cache_paths(split)
    if not (pq.exists() and man.exists()):
      return None
    with man.open('r', encoding='utf-8') as f:
      meta = json.load(f)
    if meta.get('fingerprint') != _fingerprint(split_dir):
      return None
    if meta.get('use_timestamp', True) != use_timestamp:
      return None
    return pd.read_parquet(pq, engine='pyarrow')

  def _write_raw(split_dir, split, df):
    pq, man = _raw_cache_paths(split)
    df.to_parquet(pq, index=False, engine='pyarrow', compression='zstd')
    with man.open('w', encoding='utf-8') as f:
      json.dump(
          {
              'fingerprint': _fingerprint(split_dir),
              'rows': int(len(df)),
              'cols': list(df.columns),
              'use_timestamp': use_timestamp,
          },
          f,
          separators=(',', ':'))

  def _try_load_ds(split, kind):
    if not ds_apply or ds_factor <= 1:
      return None
    pq, man = _ds_cache_paths(split, kind)
    if not (pq.exists() and man.exists()):
      return None
    with man.open('r', encoding='utf-8') as f:
      meta = json.load(f)
    if bool(meta.get('ds_apply', False)) != ds_apply:
      return None
    if int(meta.get('ds_factor', 1)) != ds_factor:
      return None
    if meta.get('use_timestamp', True) != use_timestamp:
      return None
    return pd.read_parquet(pq, engine='pyarrow')

  def _write_ds(split, kind, df):
    if not ds_apply or ds_factor <= 1:
      return
    pq, man = _ds_cache_paths(split, kind)
    df.to_parquet(pq, index=False, engine='pyarrow', compression='zstd')
    with man.open('w', encoding='utf-8') as f:
      json.dump(
          {
              'rows': int(len(df)),
              'cols': list(df.columns),
              'use_timestamp': use_timestamp,
              'ds_apply': ds_apply,
              'ds_factor': ds_factor,
          },
          f,
          separators=(',', ':'))

  # ---------- helpers ----------
  def _cond(unit):
    u = unit.lower()
    if u.startswith('bearing1_'):
      return 1
    if u.startswith('bearing2_'):
      return 2
    if u.startswith('bearing3_'):
      return 3
    return 0

  def _load(split_dir, split_name, units):
    frames = []
    for u in units:
      files = sorted((split_dir / u).glob('acc_*.csv'), key=_idx)
      if not files:
        continue
      total = len(files)
      for i, fp in enumerate(files, start=1):
        df = _read_csv_robust(fp)
        if df.shape[1] >= 3:
          df = df.iloc[:, -3:]
        df.columns = ['sample', 'vib_h', 'vib_v']
        df['sample'] = pd.to_numeric(df['sample'], errors='coerce').astype('float64')
        df['vib_h'] = pd.to_numeric(df['vib_h'], errors='coerce')
        df['vib_v'] = pd.to_numeric(df['vib_v'], errors='coerce')

        cycle = int(i)
        if use_timestamp:
          df['timestamp'] = df['sample'] / fs
          sig_cols = ['timestamp', 'vib_h', 'vib_v']
        else:
          sig_cols = ['vib_h', 'vib_v']

        out = df[sig_cols].copy()
        out.insert(0, 'unit', u)
        out.insert(1, 'cycle', cycle)
        out.insert(0, 'split', split_name)
        out['os'] = _cond(u)
        extra = OFFSET_CYC.get(u, 0.0)  # 0 for train units, >0 for test units
        out['rul'] = float(max(0, total - i + extra))
        frames.append(out)

    if not frames:
      return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(['unit', 'cycle']).reset_index(drop=True)

  # ---------- load raw ----------
  tr_units = p.get('train_units', _units(train_dir))
  te_units = p.get('test_units', _units(test_dir))

  df_tr = _try_load_raw(train_dir, 'train')
  if df_tr is None:
    df_tr = _load(train_dir, 'train', tr_units)
    _write_raw(train_dir, 'train', df_tr)

  df_te = _try_load_raw(test_dir, 'test')
  if df_te is None:
    df_te = _load(test_dir, 'test', te_units)
    _write_raw(test_dir, 'test', df_te)

  # ---------- split x/y ----------
  drop_meta = [c for c in ('rul', 'split') if c in df_tr.columns]
  x_train = df_tr.drop(columns=drop_meta).reset_index(drop=True)
  y_train = df_tr[['unit', 'cycle', 'rul']].reset_index(drop=True)
  x_test = df_te.drop(columns=drop_meta).reset_index(drop=True)
  y_test = df_te[['unit', 'cycle', 'rul']].reset_index(drop=True)

  # ---------- downsample using your function + cache ----------
  if ds_apply and ds_factor > 1:
    x_tr_cached = _try_load_ds('train', 'x')
    y_tr_cached = _try_load_ds('train', 'y')
    x_te_cached = _try_load_ds('test', 'x')
    y_te_cached = _try_load_ds('test', 'y')

    if all(obj is not None for obj in (x_tr_cached, y_tr_cached, x_te_cached, y_te_cached)):
      x_train, y_train, x_test, y_test = x_tr_cached, y_tr_cached, x_te_cached, y_te_cached
    else:
      data_xy = {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}
      ds_effective = {'apply': True, 'factor': ds_factor, 'group_by': 'unit'}
      data_xy = downsample(data_xy, ds_effective)
      x_train, y_train, x_test, y_test = data_xy['x_train'], data_xy['y_train'], data_xy['x_test'], data_xy['y_test']

      if use_timestamp and 'timestamp' in x_train.columns:

        def _fix_ts(df):
          parts = []
          for (u, c), g in df.groupby(['unit', 'cycle'], sort=False):
            n = len(g)
            gg = g.drop(columns=['timestamp']).copy()
            gg['timestamp'] = np.arange(n, dtype=float) / fs_out
            parts.append(gg)
          return pd.concat(parts, ignore_index=True)

        x_train = _fix_ts(x_train)
        x_test = _fix_ts(x_test)

      _write_ds('train', 'x', x_train)
      _write_ds('train', 'y', y_train)
      _write_ds('test', 'x', x_test)
      _write_ds('test', 'y', y_test)

  meta = {
      'dataset': 'femto',
      'root': str(base),
      'train_units': list(tr_units),
      'test_units': list(te_units),
      'use_timestamp': use_timestamp,
      'sampling_rate_hz': fs_out,
      'raw_columns': (['timestamp', 'vib_h', 'vib_v'] if use_timestamp else ['vib_h', 'vib_v']),
      'id_columns': ['unit', 'cycle'],
      'downsampling': {
          'apply': bool(ds_apply),
          'factor': int(ds_factor),
      },
  }
  return {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test, 'meta': meta}


DATASET_LOADERS: dict[str, DatasetLoader] = {'cmapps': load_cmapps, 'femto': load_femto}


def get_dataset_loader(name: str) -> DatasetLoader:
  if name not in DATASET_LOADERS:
    raise ValueError(f'unknown dataset loader: {name}')
  return DATASET_LOADERS[name]

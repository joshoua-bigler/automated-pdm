import re
import pandas as pd
from dataclasses import dataclass


@dataclass
class SchemaConfig:
  head_n: int = 3
  profiles: tuple[str, ...] = ('auto',)
  unit_col: str | None = None
  time_col: str | None = None
  target_cols: list[str] | None = None


def _detect_time_col(cols: list[str]) -> str | None:
  candidates = ('cycle', 'time', 'timestamp', 't', 'sample', 'index')
  for c in candidates:
    if c in cols:
      return c
  for c in ('run', 'ticks', 'frame'):
    if c in cols:
      return c
  return None


def _detect_unit_col(cols: list[str]) -> str | None:
  for c in ('unit', 'id', 'unit_id', 'engine_id', 'bearing_id', 'machine', 'asset'):
    if c in cols:
      return c
  return None


def _detect_target_cols(cols: list[str]) -> list[str]:
  tnames: list[str] = []
  for c in cols:
    lc = c.lower()
    if lc in ('rul', 'target', 'y', 'label'):
      tnames.append(c)
    elif 'rul' in lc or 'life' in lc or 'failure' in lc:
      tnames.append(c)
  return sorted(set(tnames))


def _detect_os_cols(cols: list[str]) -> list[str]:
  os_pref = [c for c in cols if re.fullmatch(r'os\d+', c)]
  if os_pref:
    return sorted(os_pref, key=lambda x: int(x[2:]))
  generic: list[str] = []
  for c in cols:
    if c.lower().startswith('os'):
      generic.append(c)
  return generic


def _detect_sensor_cols(cols: list[str]) -> list[str]:
  s_num = [c for c in cols if re.fullmatch(r's\d+', c)]
  if s_num:
    return sorted(s_num, key=lambda x: int(x[1:]))
  prefixes = ('sensor', 'vib', 'acc', 'ch', 'rpm', 'current', 'voltage', 'temperature')
  return [c for c in cols if any(c.lower().startswith(p) for p in prefixes)]


def _range_token(seq: list[str], prefix: str) -> str | None:
  xs = [c for c in seq if re.fullmatch(fr'{prefix}\d+', c)]
  if not xs:
    return None
  idx = sorted(int(c[len(prefix):]) for c in xs)
  return f'{prefix}{idx[0]}:{prefix}{idx[-1]}'


def _apply_profile_hints(name: str, cols: list[str], hints: dict) -> None:
  if name == 'cmapps':
    if not hints.get('time_col') and 'cycle' in cols:
      hints['time_col'] = 'cycle'
    if not hints.get('unit_col') and 'unit' in cols:
      hints['unit_col'] = 'unit'
  elif name in ('femto', 'pronostia'):
    if not hints.get('unit_col'):
      for k in ('bearing_id', 'id'):
        if k in cols:
          hints['unit_col'] = k
          break
    if not hints.get('time_col'):
      for k in ('time', 'timestamp', 'sample', 'window', 'window_index'):
        if k in cols:
          hints['time_col'] = k
          break
  elif name in ('nasa_bearing', 'ims_bearing'):
    if not hints.get('unit_col'):
      for k in ('bearing_id', 'asset', 'id'):
        if k in cols:
          hints['unit_col'] = k
          break
    if not hints.get('time_col'):
      for k in ('time', 'sample', 'file_index', 'window_index'):
        if k in cols:
          hints['time_col'] = k
          break


def make_generic_schema(df: pd.DataFrame, cfg: SchemaConfig | None = None) -> dict:
  ''' Make a generic data schema from a pandas DataFrame. '''
  cfg = cfg or SchemaConfig()
  cols = list(df.columns)
  hints: dict = {'unit_col': cfg.unit_col, 'time_col': cfg.time_col}
  for prof in cfg.profiles:
    if prof != 'auto':
      _apply_profile_hints(prof, cols, hints)
  unit_col = hints.get('unit_col') or _detect_unit_col(cols)
  time_col = hints.get('time_col') or _detect_time_col(cols)
  target_cols = (cfg.target_cols or []) or _detect_target_cols(cols)
  os_cols = _detect_os_cols(cols)
  sensor_cols = _detect_sensor_cols(cols)
  dtypes = {c: str(t) for c, t in df.dtypes.items()}
  num_cols = [c for c, t in df.dtypes.items() if str(t).startswith(('int', 'float'))]
  cat_cols = [c for c in cols if c not in num_cols and c != time_col]
  os_range = _range_token(cols, 'os')
  s_range = _range_token(cols, 's')
  select_columns = [x for x in (os_range, s_range) if x]
  id_cols: list[str] = [c for c in ('run', 'sequence', 'bearing_id', 'engine_id') if c in cols]
  if unit_col and unit_col not in id_cols:
    id_cols = [unit_col] + id_cols
  head = df.head(cfg.head_n).to_dict(orient='list')
  if time_col and unit_col:
    try:
      ok = df[[unit_col, time_col]].dropna().groupby(unit_col)[time_col].apply(lambda s: (s.diff().dropna() >= 0).all())
      time_monotonic_by_unit = bool(ok.all())
    except Exception:
      time_monotonic_by_unit = None
  else:
    time_monotonic_by_unit = None
  schema = {
      'schema_version': '1.0',
      'n_rows': int(len(df)),
      'n_units': int(df[unit_col].nunique()) if unit_col else None,
      'columns': cols,
      'dtypes': dtypes,
      'head': head,
      'unit_col': unit_col,
      'time_col': time_col,
      'target_cols': target_cols,
      'os_cols': os_cols,
      'sensor_cols': sensor_cols,
      'id_cols': id_cols,
      'num_cols': num_cols,
      'cat_cols': cat_cols,
      'time_monotonic_by_unit': time_monotonic_by_unit,
      'has_os': len(os_cols) > 0,
      'select_columns': select_columns,
      'profiles': cfg.profiles,
  }
  return schema

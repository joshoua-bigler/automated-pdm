import numpy as np
import pandas as pd
import torch
from datetime import datetime, date
from dataclasses import asdict
from pathlib import Path


def json_default(o: object) -> object:
  ''' Default JSON serializer supporting numpy, pandas, datetime, Path, etc. '''
  # numpy
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, (np.floating,)):
    return float(o)
  if isinstance(o, (np.ndarray,)):
    return o.tolist()
  # pandas
  if isinstance(o, pd.Categorical):
    return o.tolist()
  if isinstance(o, pd.Series):
    # handles category dtype too
    return o.astype(object).where(o.notna(), None).tolist()
  if isinstance(o, pd.DataFrame):
    return o.to_dict(orient='records')
  if isinstance(o, (pd.Timestamp,)):
    return o.isoformat()
  # common extras
  if isinstance(o, (datetime, date)):
    return o.isoformat()
  if isinstance(o, Path):
    return str(o)
  # ray tune domains / other odd objects → fallback
  return str(o)


def json_safe_dict(obj: object) -> dict:
  ''' Convert a dataclass or similar object to a JSON-serializable dict. '''
  d = asdict(obj)
  for k, v in d.items():
    if isinstance(v, torch.device):
      d[k] = str(v)  # e.g. "cuda:0"
  return d

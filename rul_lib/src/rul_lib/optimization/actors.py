import os, json, hashlib
import ray
import json
import torch
import numpy as np
from typing import Any


def create_seed() -> int:
  ''' Create a random seed, optionally from environment variable PY_SEED. '''
  seed = int(os.environ.get('PY_SEED', '1337'))
  np.random.seed(seed)
  torch.manual_seed(seed)
  return seed


def enc_key(cfg: dict) -> str:
  ''' Generate a unique key for a configuration dictionary. '''
  s = json.dumps(cfg, sort_keys=True, separators=(',', ':'))
  return hashlib.sha1(s.encode('utf-8')).hexdigest()


def ray_resolve(x: Any) -> Any:
  ''' Resolve a Ray ObjectRef to its value, or return the value if not an ObjectRef. '''
  try:
    from ray._raylet import ObjectRef
    return ray.get(x) if isinstance(x, ObjectRef) else x
  except Exception:
    try:
      return ray.get(x)
    except Exception:
      return x


def fe_key(fe_cfg: dict) -> str:
  ''' Generate a unique key for a feature extraction configuration. '''
  return f'fe:{enc_key(fe_cfg)}'


@ray.remote
class FeatureCache:
  ''' Cache for feature extraction results, with a maximum number of items. '''

  def __init__(self, max_items=256):
    self.store = {}
    self.order = []
    self.max_items = max_items

  def get(self, key: str) -> dict | None:
    return self.store.get(key)

  def put(self, key: str, z_tr_ref: Any, y_tr_ref: Any, z_te_ref: Any, y_te_ref: Any, meta: dict) -> bool:
    if key not in self.store and len(self.order) >= self.max_items:
      old = self.order.pop(0)
      self.store.pop(old, None)
    self.store[key] = {'z_train': z_tr_ref, 'y_train': y_tr_ref, 'z_test': z_te_ref, 'y_test': y_te_ref, 'meta': meta}
    if key not in self.order:
      self.order.append(key)
    return True

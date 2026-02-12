import numpy as np
from copy import deepcopy
from ray import tune


def _qrandint(low: int, high: int, q: int) -> tune.sample_from:
  if hasattr(tune, 'qrandint'):
    return tune.qrandint(int(low), int(high), int(q))
  n = (high - low) // q + 1
  return tune.sample_from(lambda _: low + q * np.random.randint(0, n))


def _quniform(low: float, high: float, q: float) -> tune.sample_from:
  if hasattr(tune, 'quniform'):
    return tune.quniform(float(low), float(high), float(q))
  return tune.sample_from(lambda _: round(np.random.uniform(low, high) / q) * q)


def _make_domain(spec: dict) -> tune.sample_from:
  kind = str(spec['space']).lower()
  if kind == 'choice':
    values = spec['values']
    if isinstance(values, (list, tuple)):
      values = list(values)
    return tune.choice(values)
  if kind == 'uniform':
    return tune.uniform(float(spec['low']), float(spec['high']))
  if kind == 'loguniform':
    return tune.loguniform(float(spec['low']), float(spec['high']))
  if kind == 'quniform':
    return _quniform(float(spec['low']), float(spec['high']), float(spec['q']))
  if kind == 'randint':
    return tune.randint(int(spec['low']), int(spec['high']) + 1)
  if kind == 'qrandint':
    return _qrandint(int(spec['low']), int(spec['high']), int(spec.get('q', 1)))
  raise ValueError(f'unknown space kind: {kind}')


def _walk_build(node: object) -> object:
  if isinstance(node, dict):
    if 'space' in node:
      return _make_domain(node)
    out = {}
    for k, v in node.items():
      k_str = str(k)
      out[k_str] = _walk_build(v)
    return out
  if isinstance(node, (list, tuple)):
    return type(node)(_walk_build(v) for v in node)
  return node


def build_search_space(cfg: dict) -> dict:
  '''Build a Ray Tune search space from a configuration dictionary.'''
  cfg = deepcopy(cfg)
  return _walk_build(cfg)

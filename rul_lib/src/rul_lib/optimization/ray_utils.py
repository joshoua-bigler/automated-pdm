import json
import numpy as np
import os
import ray
import tempfile
from ray import tune
from ray.tune import Checkpoint
# local
from rul_lib.optimization.actors import FeatureCache
from rul_lib.utils.json_utils import json_default


def ray_feat_cache(feat_cache: FeatureCache, fkey: str, z_train: np.ndarray, y_train: np.ndarray, z_test: np.ndarray,
                   y_test: np.ndarray, fe_cfg_full: dict) -> None:
  ''' Cache feature extraction outputs in Ray object store. '''
  meta = {'feature_engineering': fe_cfg_full, 'latent_dim': int(z_train.shape[1])}
  ray.get(
      feat_cache.put.remote(fkey, ray.put(z_train.astype(np.float32)), ray.put(y_train.astype(np.float32)),
                            ray.put(z_test.astype(np.float32)), ray.put(y_test.astype(np.float32)), meta))


def ray_checkpoint(cfg: dict, val_obj: float, seed: int) -> None:
  with tempfile.TemporaryDirectory() as tmp:
    payload = {
        'pipeline': cfg['pipeline'],
        'feature_engineering': cfg['feature_engineering'],
        'regression_config': cfg.get('regression', {}),
        'seq2val_config': cfg.get('seq2val', {}),
        'val_obj': val_obj,
        'seed': seed
    }
    with open(os.path.join(tmp, 'trial.json'), 'w') as f:
      json.dump(payload, f, separators=(',', ':'), sort_keys=True, default=json_default)
    tune.report({'val_obj': val_obj}, checkpoint=Checkpoint.from_directory(tmp))

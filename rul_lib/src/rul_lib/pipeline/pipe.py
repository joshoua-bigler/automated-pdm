import numpy as np
import torch
from ray import tune
from ray.tune import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.logger import TBXLoggerCallback
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
# local
from rul_lib.pipeline.fe.feature_extraction import train_feature_extraction
from rul_lib.pipeline.rul.models import RulModel, TabularRulModel
from rul_lib.pipeline.fe.cnn.datasets import AEDataset
from rul_lib.optimization.actors import FeatureCache, ray_resolve, fe_key
from rul_lib.optimization.ray_utils import ray_feat_cache
from rul_lib.training.train_logger import NullLogger


def with_criterion(cfg: dict, crit: dict) -> dict:
  ''' Add criterion configuration to the pipeline configuration. '''
  out = dict(cfg)
  out['criterion'] = crit
  return out


def to_tensor_xy(x: np.ndarray, y: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
  ''' Convert numpy arrays to PyTorch tensors for features and targets. '''
  x_t = torch.tensor(np.asarray(x), dtype=torch.float32)
  y_t = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
  return x_t, y_t


def make_train_loader_from_encoder(encoder_meta: dict,
                                   windowed: dict,
                                   cfg: dict,
                                   reg_model,
                                   use_os: bool = False) -> tuple[torch.nn.Module, DataLoader]:
  ''' Create a training data loader based on the encoder type. '''
  enc = encoder_meta
  if enc['model_type'] == 'torch':
    model = RulModel(encoder=enc['model'], regressor=reg_model, multi_window=cfg.get('apply', False), m=cfg.get('m', 1))
    loader = DataLoader(AEDataset(windowed['x_train']), batch_size=32, shuffle=False, drop_last=False)
    return model, loader
  if enc['model_type'] == 'tabular':
    x_tr, y_tr = to_tensor_xy(windowed['x_train'], windowed['y_train'])
    loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=False, drop_last=False)
    model = TabularRulModel(encoder=enc['model'], regressor=reg_model, multi_window=cfg.get('apply', False), m=cfg.get('m', 1), use_os=use_os)  # yapf: disable
    return model, loader
  raise ValueError(f'unknown encoder model type {enc["model_type"]}')


def make_supervised_loader(windowed: dict) -> DataLoader:
  ''' Create a supervised training data loader from windowed data. '''
  x_tr, y_tr = to_tensor_xy(windowed['x_train'], windowed['y_train'])
  return DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=False, drop_last=False)


def run_fe_with_cache(windowed: dict, cfg: dict, tracking: dict,
                      feat_cache: FeatureCache) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  ''' Run feature extraction with caching support. '''
  fe_cfg_full = {'feature_engineering': cfg['feature_engineering'], 'feature_extraction': cfg['feature_extraction']}
  fkey = fe_key(fe_cfg=fe_cfg_full)
  cached = ray_resolve(feat_cache.get.remote(fkey)) if feat_cache else None
  if cached is not None:
    z_train = ray_resolve(cached['z_train']).astype(np.float32)
    y_train = ray_resolve(cached['y_train']).astype(np.float32)
    z_test = ray_resolve(cached['z_test']).astype(np.float32)
    y_test = ray_resolve(cached['y_test']).astype(np.float32)
    return z_train, y_train, z_test, y_test
  art = train_feature_extraction(data=windowed,
                                 config=cfg['feature_extraction'],
                                 tracking=tracking,
                                 train_logger=NullLogger())
  z_train, z_test = art['z_train'], art['z_test']
  y_train = windowed['y_train'].astype(np.float32)
  y_test = windowed['y_test'].astype(np.float32)
  if feat_cache:
    ray_feat_cache(feat_cache=feat_cache,
                   fkey=fkey,
                   z_train=z_train,
                   y_train=y_train,
                   z_test=z_test,
                   y_test=y_test,
                   fe_cfg_full=fe_cfg_full)
  return z_train, y_train, z_test, y_test

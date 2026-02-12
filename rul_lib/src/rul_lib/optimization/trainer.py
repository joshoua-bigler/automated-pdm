import ray
import torch
# local
import rul_lib.pipeline.pipe as pipe
from rul_lib.enums import PipelineType
from rul_lib.gls.gls import logger
from rul_lib.pipeline.shared.config_view import make_pipeline_view, parse_pipeline_type
from rul_lib.pipeline.fe.window import grouped_sliding_windows
from rul_lib.pipeline.reg.regression import train_regression
from rul_lib.pipeline.seq2val.seq2val import train_seq2val
from rul_lib.optimization.actors import FeatureCache, create_seed
from rul_lib.optimization.ray_utils import ray_checkpoint
from rul_lib.training.train_logger import NullLogger
from rul_lib.optimization.ray_logger import write_run_summary
from rul_lib.pipeline.shared.multi_windows import make_multiwindow_latent, append_os_features


def train_pipeline(cfg: dict, feat_cache: FeatureCache, tracking: dict, data_ref: dict) -> None:
  ''' Train a pipeline specified by cfg using data in data_ref and feature cache feat_cache. '''
  logger.info('starting trial')
  write_run_summary(cfg=cfg)
  device = 'cuda' if torch.cuda.is_available() else 'cpu'
  logger.info(f'device: {device}')
  seed = create_seed()
  data = {k: ray.get(v) if isinstance(v, ray.ObjectRef) else v for k, v in data_ref.items()}
  cfgv = make_pipeline_view(cfg=cfg)
  ptype = parse_pipeline_type(cfg=cfgv)
  windowed = grouped_sliding_windows(data=data, params=cfg['feature_engineering']['window'])
  if ptype == PipelineType.FE_REG:
    logger.info(f'pipeline=fe_reg fe={cfgv.get("feature_extraction", {}).get("model", "N/A")} reg={cfgv.get("regression", {}).get("model", "N/A")}') # yapf: disable
    z_tr, y_tr, z_te, y_te = pipe.run_fe_with_cache(windowed=windowed, cfg=cfgv, tracking=tracking, feat_cache=feat_cache) # yapf: disable
    multi_window_cfg = cfgv['regression'].get('multi_window', {}) or {}
    apply_mw = bool(multi_window_cfg.get('apply', False))
    m = int(multi_window_cfg.get('m', 0))
    if apply_mw and m > 1:
      z_tr, y_tr = make_multiwindow_latent(z=z_tr, y=y_tr, m=m)
      z_te, y_te = make_multiwindow_latent(z=z_te, y=y_te, m=m)
    encode_os = cfgv.get('feature_engineering', {}).get('window', {}).get('encode_os', False)
    if encode_os:
      z_tr, y_tr, z_te, y_te = append_os_features(z_tr=z_tr, y_tr=y_tr, z_te=z_te, y_te=y_te, windowed=windowed, apply_mw=apply_mw, m=m) # yapf: disable
    reg_cfg = pipe.with_criterion(cfgv['regression'], cfgv['criterion'])
    res = train_regression(z_train=z_tr,
                           y_train=y_tr,
                           z_test=z_te,
                           y_test=y_te,
                           config=reg_cfg,
                           tracking=tracking,
                           train_logger=NullLogger())
  elif ptype == PipelineType.SEQ2VAL:
    logger.info(f'pipeline=seq2val model={cfgv.get("seq2val", {}).get("model", "N/A")}')
    s2v_cfg = pipe.with_criterion(cfgv['seq2val'], cfgv['criterion'])
    # seq2val handles its own multi-window / OS usage internally if configured
    res = train_seq2val(windowed=windowed, cfg=s2v_cfg, tracking=tracking, train_logger=NullLogger())
  else:
    raise ValueError(f'unknown pipeline type {cfgv["pipeline"]["type"]}')
  ray_checkpoint(cfg=cfgv, val_obj=float(res['val_obj']), seed=seed)

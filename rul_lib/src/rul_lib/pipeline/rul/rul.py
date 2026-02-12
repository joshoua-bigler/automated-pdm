import json
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import torch.nn as nn
import yaml
from pathlib import Path
from ray.tune import ResultGrid
from torch.utils.data import DataLoader
# local
import rul_lib.pipeline.pipe as pipe
import rul_lib.pipeline.rul.models as models
from rul_lib.training.train_logger import log_model
from rul_lib.gls.gls import logger
from rul_lib.utils.dict_utils import flatten_conditional
from rul_lib.optimization.ray_logger import log_ray_results_summary
from rul_lib.model_registry.utils import make_model_meta
from rul_lib.enums import PipelineType
from rul_lib.pipeline.shared.names import _flatten_for_params, get_model_name, get_run_name
from rul_lib.pipeline.shared.config_view import make_pipeline_view, parse_pipeline_type
from rul_lib.pipeline.fe.window import grouped_sliding_windows
from rul_lib.pipeline.fe.feature_extraction import train_feature_extraction
from rul_lib.pipeline.reg.regression import train_regression
from rul_lib.pipeline.seq2val.seq2val import train_seq2val
from rul_lib.eval.eval_rul import eval_model
from rul_lib.training.train_logger import MlflowLogger
from rul_lib.utils.visualize import scatter_plot
from rul_lib.pipeline.shared.multi_windows import make_multiwindow_latent, append_os_features, infer_seq_len


def log_rul(results: ResultGrid | None,
            rul: nn.Module,
            train_loader: DataLoader,
            config: dict,
            tracking: dict,
            meta: dict,
            window_size: int = None,
            metrics: dict = None,
            artifacts: dict = None,
            metric: str = 'val_obj',
            mode: str = 'min',
            topk: int = 10,
            log_ray_results: bool = True,
            ray_storage_path: str | Path = None) -> None:
  ''' Log RUL model and artifacts to MLflow. '''
  ptype = parse_pipeline_type(config)
  cfg_view = make_pipeline_view(config)  # prune inactive branches
  version = tracking.get('version')
  exp_name = tracking.get('experiment_name', '')
  feats_in = 'unk'
  if ptype is PipelineType.FE_REG:
    fe = flatten_conditional(cfg_view.get('feature_extraction') or {})
    feats_in = fe.get('latent_dim', 'unk')
  model_meta = make_model_meta(full_meta=meta)
  with mlflow.start_run(run_name=get_run_name(config=cfg_view, tracking=tracking, window_size=window_size)):
    mlflow.log_text(yaml.safe_dump(config, sort_keys=False), 'artifacts/config.yml')
    mlflow.log_dict(meta, 'artifacts/preprocessing_meta.json')
    mlflow.log_dict(tracking, 'artifacts/tracking_meta.json')
    pre_cfg = cfg_view.get('preprocessing', {})
    small_params = {
        'pipeline_type': ptype.value,
        'window_size': window_size,
        'stride': cfg_view.get('feature_engineering', {}).get('window', {}).get('stride', 'UNK'),
        'version': version,
        'pre_denoise': pre_cfg.get('denoising', {}).get('method', 'UNK'),
        'pre_target_scaling': pre_cfg.get('target_scaling', {}).get('method', 'UNK'),
    }
    if ptype is PipelineType.FE_REG:
      small_params['fe_latent_dim'] = feats_in
      small_params['fe_model'] = cfg_view.get('feature_extraction', {}).get('model', 'UNK')
      small_params['reg_model'] = cfg_view.get('regression', {}).get('model', 'UNK')
    else:
      s2v = flatten_conditional(cfg_view.get('seq2val') or {})
      small_params['seq2val_model'] = s2v.get('model', 'seq')
    mlflow.log_params(_flatten_for_params(small_params))
    if metrics:
      mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}, step=0)
    mlflow.set_tags({
        'pipeline_type': ptype.value,
        'model_name': tracking['mlflow']['rul_model_name'],
        'experiment_name': exp_name
    })
    log_model(model=rul,
              train_loader=train_loader,
              code_paths=[models.__file__],
              registered_model_name=get_model_name(cfg_view, tracking),
              tags={'meta': json.dumps(model_meta)})
    if artifacts:
      for key, art in artifacts.items():
        if isinstance(art, plt.Figure):
          mlflow.log_figure(art, f'artifacts/{key}.png')
        elif isinstance(art, (str, Path)) and Path(art).exists():
          mlflow.log_artifact(str(art), artifact_path='artifacts')
        else:
          logger.warning(f'unsupported artifact type: {type(art)} for key={key}')
    if log_ray_results and ray_storage_path:
      ray_dir = Path(ray_storage_path) / f'{exp_name}_v{version}'
      if ray_dir.exists():
        mlflow.log_artifact(str(ray_dir), artifact_path='artifacts/ray')
    if log_ray_results and results is not None:
      log_ray_results_summary(results=results, artifact_path='artifacts/configs', metric=metric, mode=mode, topk=topk)


def create_rul(results: ResultGrid | None,
               data: dict,
               tracking: dict,
               ray_storage_path: str | Path = None,
               metrics: dict | None = None,
               cfg: dict | None = None) -> dict | None:
  ''' Create and train RUL model based on Ray Tune results or config. '''
  logger.info('assemble best model -> mlflow')
  metrics = metrics or {}
  cfgv: dict
  best_metric = 'val_obj'
  best_mode = 'min'
  if results is not None:
    # Be robust to missing/NaN metrics from trials
    def _select_best(res_grid: ResultGrid) -> tuple:
      candidates = [
          ('val_obj', 'min', True),
          ('val_obj', 'min', False),  # allow NaN/Inf if needed
          ('val_rmse', 'min', True),
          ('val_loss', 'min', True),
          ('val_mse', 'min', True)
      ]
      last_err: Exception | None = None
      for metric_name, mode_name, filt in candidates:
        try:
          r = res_grid.get_best_result(metric=metric_name, mode=mode_name, filter_nan_and_inf=filt)
          return r, metric_name, mode_name
        except Exception as e:
          last_err = e
          continue
      # final fallback: first non-error trial if any
      try:
        for r in res_grid:
          if getattr(r, 'error', None) is None:
            return r, None, None
      except Exception:
        pass
      raise RuntimeError(f'Could not select best trial; last error: {last_err}')

    best, best_metric, best_mode = _select_best(results)
    cfgv = make_pipeline_view(cfg=best.config)
  else:
    if not cfg:
      raise ValueError('create_rul: either results or cfg must be provided')
    cfgv = make_pipeline_view(cfg=cfg)
  ptype = parse_pipeline_type(cfg=cfgv)
  windowed = grouped_sliding_windows(data=data, params=cfgv['feature_engineering']['window'])
  mlf = MlflowLogger()
  encode_os = cfgv.get('feature_engineering', {}).get('window', {}).get('encode_os', False)
  if ptype == PipelineType.FE_REG:
    fe_art = train_feature_extraction(data=windowed, config=cfgv['feature_extraction'], tracking=tracking, train_logger=mlf) # yapf: disable
    z_tr, z_te = fe_art['z_train'], fe_art['z_test']
    y_tr = windowed['y_train'].astype(np.float32)
    y_te = windowed['y_test'].astype(np.float32)
    multi_window_cfg = cfgv['regression'].get('multi_window', {})
    apply_mw = bool(multi_window_cfg.get('apply', False))
    m = int(multi_window_cfg.get('m', 0))
    # 1) multi-window latent once
    if apply_mw and m > 1:
      z_tr, y_tr = make_multiwindow_latent(z=z_tr, y=y_tr, m=m)
      z_te, y_te = make_multiwindow_latent(z=z_te, y=y_te, m=m)
    encode_os = cfgv.get('feature_engineering', {}).get('window', {}).get('encode_os', False)
    if encode_os:
      z_tr, y_tr, z_te, y_te = append_os_features(z_tr=z_tr, y_tr=y_tr, z_te=z_te, y_te=y_te, windowed=windowed, apply_mw=apply_mw, m=m) # yapf: disable
    reg_cfg = pipe.with_criterion(cfgv['regression'], cfgv['criterion'])
    reg_art = train_regression(z_train=z_tr, y_train=y_tr, z_test=z_te, y_test=y_te, config=reg_cfg, tracking=tracking, train_logger=mlf) # yapf: disable
    rul_model, train_loader = pipe.make_train_loader_from_encoder(encoder_meta=fe_art['encoder'], windowed=windowed, cfg=multi_window_cfg, reg_model=reg_art['model'], use_os=encode_os) # yapf: disable
  else:
    s2v_cfg = pipe.with_criterion(cfgv['seq2val'], cfgv['criterion'])
    res = train_seq2val(windowed=windowed, cfg=s2v_cfg, tracking=tracking, train_logger=mlf)
    rul_model = res['model']
    train_loader = pipe.make_supervised_loader(windowed)
    multi_window_cfg = cfgv['seq2val'].get('multi_window', {})
  ws = infer_seq_len(windowed=windowed, cfg=multi_window_cfg, pipeline_type=ptype)
  ev = eval_model(model=rul_model,
                  x_test=windowed['x_test'],
                  y_test=windowed['y_test'],
                  meta=data['meta'],
                  os_test=windowed.get('os_int_test', None) if encode_os else None)
  rmse = ev['eval_metrics']['rmse']
  logger.info(f'RUL RMSE: {rmse:.4f}')
  fig = scatter_plot(pred=ev['y_pred'], target=ev['y_true'], rmse=rmse)
  # yapf: disable
  log_rul(results=results,
          rul=rul_model,
          train_loader=train_loader,
          config=cfgv,
          tracking=tracking,
          window_size=ws,
          artifacts={'parity_plot': fig},
          metrics={**metrics, **ev['eval_metrics']},
          ray_storage_path=ray_storage_path,
          meta=data['meta'],
          metric=best_metric or 'val_obj',
          mode=best_mode or 'min',
          log_ray_results=(results is not None))
  # yapf: enable
  return {'val_obj': float(rmse), **ev.get('eval_metrics', {})}

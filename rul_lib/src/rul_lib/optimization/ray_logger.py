import os, json, tempfile
import mlflow
from pathlib import Path
from ray.tune import ResultGrid
from ray.air import session
from torch.utils.tensorboard import SummaryWriter
from typing import Any
# local
from rul_lib.utils.json_utils import json_default
from rul_lib.utils.dict_utils import flatten_conditional
from rul_lib.pipeline.shared.config_view import make_pipeline_view, parse_pipeline_type
from rul_lib.pipeline.shared.names import _win_size, _flatten_for_params


def make_config_view(cfg: dict) -> dict:
  ''' Return a pruned config view with flattened conditional blocks. '''
  view = make_pipeline_view(cfg)
  if 'feature_extraction' in view:
    view = dict(view)
    view['feature_extraction'] = flatten_conditional(view['feature_extraction'])
  if 'seq2val' in view:
    view = dict(view)
    view['seq2val'] = flatten_conditional(view['seq2val'])
  if 'regression' in view:
    view = dict(view)
    view['regression'] = flatten_conditional(view['regression'])
  return view


def make_small_params(view: dict) -> dict:
  '''Compact, UI-friendly params derived from the config view.'''
  try:
    ptype = parse_pipeline_type(view).value
  except Exception:
    ptype = 'unknown'
  wsize = _win_size(view)
  small = {'pipeline': {'type': ptype}, 'window': {'size': wsize}}
  if 'feature_extraction' in view:
    ld = (view.get('feature_extraction') or {}).get('latent_dim', 'unk')
    small['fe'] = {'latent_dim': ld}
  elif 'seq2val' in view:
    s2v = view.get('seq2val') or {}
    small['seq2val'] = {'model': s2v.get('model', 'seq')}
  return small


def log_ray_results_summary(results: ResultGrid,
                            artifact_path: str = 'ray',
                            metric: str = 'obj',
                            mode: str = 'min',
                            topk: int = 5):
  ''' Log a summary of Ray Tune results to MLflow. '''
  trials = []
  for r in results:
    df = r.metrics_dataframe
    if metric not in df:
      continue
    best_val = float(df[metric].min() if mode == 'min' else df[metric].max())
    last_iter = None
    if 'epoch' in df:
      last_iter = int(df['epoch'].max())
    elif 'training_iteration' in df:
      last_iter = int(df['training_iteration'].max())
    cfg_view = make_config_view(r.config)
    small = make_small_params(cfg_view)
    trials.append({
        'trial_path': r.path,
        'best_metric': best_val,
        'last_iter': last_iter,
        'config': r.config,
        'config_view': cfg_view,
        'small_params': _flatten_for_params(small),
    })
  trials.sort(key=lambda x: x['best_metric'], reverse=(mode == 'max'))
  top = trials[:topk]
  best_view = top[0]['config_view'] if top else {}
  best_small = top[0]['small_params'] if top else {}
  with tempfile.TemporaryDirectory() as tmp:
    with open(os.path.join(tmp, 'topk_trials.json'), 'w') as f:
      json.dump(top, f, indent=2, default=json_default)
    with open(os.path.join(tmp, 'best_config_view.json'), 'w') as f:
      json.dump(best_view, f, indent=2, default=json_default)
    with open(os.path.join(tmp, 'best_small_params.json'), 'w') as f:
      json.dump(best_small, f, indent=2, default=json_default)
    try:
      best = results.get_best_result(metric=metric, mode=mode)
      if best.checkpoint is not None:
        ckpt_dir = os.path.join(tmp, 'best_checkpoint')
        best.checkpoint.to_directory(ckpt_dir)
    except Exception:
      pass
    mlflow.log_artifacts(tmp, artifact_path=artifact_path)


def _get(d: dict, path: str, default='na') -> Any:
  ''' Get a nested value from a dictionary using a dot-separated path. '''
  cur = d
  for p in path.split('.'):
    if not isinstance(cur, dict) or p not in cur:
      return default
    cur = cur[p]
  return cur


def make_trial_name(trial: Any) -> str:
  ''' Generate a name for a trial. '''
  cfg = trial.config
  ptype = _get(cfg, 'pipeline.type', 'na')
  win = _get(cfg, 'feature_engineering.window.size', 'na')
  stride = _get(cfg, 'feature_engineering.window.stride', 'na')
  if ptype == 'fe_reg':
    fe_model = _get(cfg, 'feature_extraction.model', 'na')
    reg_model = _get(cfg, 'regression.model', 'na')
    base = f'{ptype}-win{win}-s{stride}-fe={fe_model}-reg={reg_model}'
  else:
    seq_model = _get(cfg, 'seq2val.model', 'na')
    base = f'{ptype}-win{win}-s{stride}-model={seq_model}'
  return f'{base}-{trial.trial_id[-4:]}'


def make_trial_dirname(trial: Any) -> str:
  ''' Generate a directory name for a trial. '''
  return make_trial_name(trial).replace('/', '_').replace(' ', '_')


def _brief_pre(cfg: dict) -> dict:
  ''' Generate a brief summary of the preprocessing configuration. '''
  pre = cfg.get('preprocessing', {})
  return {
      'cleanup.drop': ','.join(pre.get('cleanup', {}).get('drop', [])) or '[]',
      'imputation.method': pre.get('imputation', {}).get('method', 'na'),
      'denoising.method': pre.get('denoising', {}).get('method', 'na'),
      'denoising.group_by': pre.get('denoising', {}).get('group_by', 'na'),
      'normalization.method': pre.get('normalization', {}).get('method', 'na'),
      'normalization.group_by': pre.get('normalization', {}).get('group_by', 'na'),
      'select_by_correlation.apply': pre.get('select_by_correlation', {}).get('apply', False),
      'select_by_correlation.method': pre.get('select_by_correlation', {}).get('method', 'na'),
      'select_by_correlation.corr_thr': pre.get('select_by_correlation', {}).get('correlation_threshold', 'na'),
      'select_topk_mi.apply': pre.get('select_topk_mi', {}).get('apply', False),
      'select_topk_mi.top_k': pre.get('select_topk_mi', {}).get('top_k', 'na'),
      'target_scaling.method': pre.get('target_scaling', {}).get('method', 'na'),
      'target_scaling.clip': pre.get('target_scaling', {}).get('clip', 'na'),
  }


def _model_brief(cfg: dict) -> dict:
  ''' Generate a brief summary of the model configuration. '''
  ptype = _get(cfg, 'pipeline.type', 'na')
  if ptype == 'fe_reg':
    return {
        'pipeline': ptype,
        'fe.model': _get(cfg, 'feature_extraction.model', 'na'),
        'reg.model': _get(cfg, 'regression.model', 'na'),
    }
  return {
      'pipeline': ptype,
      'seq2val.model': _get(cfg, 'seq2val.model', 'na'),
  }


def write_run_summary(cfg: dict) -> None:
  ''' Write a run summary (markdown and JSON) and TensorBoard text summary. '''
  trial_dir = Path(session.get_trial_dir())
  writer = SummaryWriter(log_dir=str(trial_dir))
  win = _get(cfg, 'feature_engineering.window.size', 'na')
  stride = _get(cfg, 'feature_engineering.window.stride', 'na')
  pre = _brief_pre(cfg)
  models = _model_brief(cfg)
  lines = []
  lines.append('# run summary')
  lines.append(f'- window.size: {win}')
  lines.append(f'- window.stride: {stride}')
  lines.append('')
  lines.append('## models')
  for k, v in models.items():
    lines.append(f'- {k}: {v}')
  lines.append('')
  lines.append('## preprocessing')
  for k, v in pre.items():
    lines.append(f'- {k}: {v}')
  md = '\n'.join(lines)
  writer.add_text('config_summary', md, 0)
  writer.flush()
  writer.close()
  (trial_dir / 'run_summary.md').write_text(md)
  (trial_dir / 'run_summary.json').write_text(
      json.dumps({
          'window': {
              'size': win,
              'stride': stride
          },
          'models': models,
          'preprocessing': pre
      }, indent=2))

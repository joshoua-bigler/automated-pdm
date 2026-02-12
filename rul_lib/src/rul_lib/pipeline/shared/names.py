# local
from rul_lib.utils.dict_utils import flatten_conditional
from rul_lib.enums import PipelineType
from rul_lib.pipeline.shared.config_view import parse_pipeline_type


def _win_size(cfg: dict) -> str | int | float:
  ''' Extract window size from config, or NaN if not found. '''
  fe = cfg.get('feature_engineering') or {}
  w = fe.get('window') or {}
  return w.get('size', 'NaN')


def get_run_name(config: dict, tracking: dict, window_size: str | int | float) -> str:
  ''' Generate a unique run name based on the config and tracking information. '''
  ptype = parse_pipeline_type(config)
  version = tracking.get('version')
  base = tracking['mlflow']['rul_model_name']
  if ptype is PipelineType.FE_REG:
    fe = flatten_conditional(config.get('feature_extraction') or {})
    feats = fe.get('latent_dim', None)
    return f'{base}_w{window_size}{'_f' + str(feats) if feats is not None else ''}_v{version}'
  s2v = flatten_conditional(config.get('seq2val') or {})
  s2v_model = s2v.get('model', 'seq')
  return f'{base}_{s2v_model}_w{window_size}_v{version}'


def get_model_name(config: dict, tracking: dict) -> str:
  ''' Generate a model name based on the config and tracking information. '''
  ptype = parse_pipeline_type(config)
  wsize = _win_size(config)
  base = tracking['mlflow']['rul_model_name']
  if ptype is PipelineType.FE_REG:
    fe = flatten_conditional(config.get('feature_extraction') or {})
    feats = fe.get('latent_dim', None)
    return f'{base}_w{wsize}{'_f' + str(feats) if feats is not None else ''}'
  s2v = flatten_conditional(config.get('seq2val') or {})
  s2v_model = s2v.get('model', 'seq')
  return f'{base}_{s2v_model}_w{wsize}'


def _flatten_for_params(d: dict, prefix: str = '') -> dict:
  ''' Flatten a nested dict for parameter logging.

      Non-primitive values are converted to strings.

      Parameters
      ----------
      d: 
        Input dict
      prefix:
        Prefix for keys (used in recursion)
      
      Returns
      -------
      Flattened dict
  '''
  out = {}
  for k, v in d.items():
    kk = f'{prefix}.{k}' if prefix else k
    if isinstance(v, dict):
      out.update(_flatten_for_params(v, kk))
    elif isinstance(v, (str, int, float, bool)):
      out[kk] = v
    else:
      out[kk] = str(v)
  return out

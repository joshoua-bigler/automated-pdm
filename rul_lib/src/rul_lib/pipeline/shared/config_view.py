from copy import deepcopy
# local
from rul_lib.enums import PipelineType

_REQUIRED = {
    PipelineType.FE_REG: ['feature_engineering', 'feature_extraction', 'regression'],
    PipelineType.SEQ2VAL: ['feature_engineering', 'seq2val'],
}

_DROP = {
    PipelineType.FE_REG: ['seq2val'],
    PipelineType.SEQ2VAL: ['feature_extraction', 'feature_selection', 'regression'],
}


def parse_pipeline_type(cfg: dict) -> PipelineType:
  t = ((cfg.get('pipeline') or {}).get('type') or '').strip()
  return PipelineType(t)


def validate_for_pipeline(cfg: dict, ptype: PipelineType) -> None:
  ''' Validate configuration for the specified pipeline type. '''
  missing = [k for k in _REQUIRED[ptype] if k not in cfg]
  if missing:
    raise ValueError(f'missing required sections for {ptype.value}: {missing}')


def make_pipeline_view(cfg: dict) -> dict:
  ''' Create a view of the configuration suitable for the specified pipeline type. '''
  ptype = parse_pipeline_type(cfg)
  validate_for_pipeline(cfg, ptype)
  view = deepcopy(cfg)
  for k in _DROP[ptype]:
    view.pop(k, None)
  return view

import json
from typing import Literal
from langchain_core.tools import tool
from rul_lib.pipeline.fe import feature_extraction as fe_mod
from rul_lib.pipeline.reg import regression as reg_mod
from rul_lib.pipeline.seq2val import seq2val as s2v_mod
from rul_lib.gls.gls import logger
from typing import Any


@tool
def list_available_models() -> str:
  ''' Return available model names for FE, regression, and seq2val as JSON. '''
  out = {
      'fe': sorted(list(fe_mod._FE_REGISTRY.keys())),
      'regression': sorted(list(reg_mod._REG_REGISTRY.keys())),
      'seq2val': sorted(list(s2v_mod._SEQ2VAL_REGISTRY.keys())),
  }
  return json.dumps(out, indent=2)


@tool
def list_available_schedulers() -> str:
  ''' Return the valid trainer LR schedulers as JSON.

      This is a canonical list shared by adapters and parser logic.
  '''
  try:
    from rul_lib.pipeline.parser.scheduler import AVAILABLE_SCHEDULERS
    scheds = list(AVAILABLE_SCHEDULERS)
  except Exception:
    logger.error('could not import AVAILABLE_SCHEDULERS from rul_lib.pipeline.parser.scheduler')
    scheds = ['unknown']
  return json.dumps({'schedulers': scheds}, indent=2)


@tool
def describe_model(area: Literal['fe', 'regression', 'seq2val'],
                   name: str,
                   with_source: bool = True,
                   max_lines: int = 200) -> str:
  ''' Return adapter-provided description with optional model source.

      Parameters
      ----------
      area : {'fe', 'regression', 'seq2val'}
        Registry area to search
      name: 
        Model/adapter name
      with_source: 
        If True and adapter provides source metadata, include class source code
      max_lines: 
        Max number of source lines to return (truncate if longer)
  '''
  area = str(area).lower().strip()
  name = str(name).lower().strip()
  registries = {
      'fe': fe_mod._FE_REGISTRY,
      'regression': reg_mod._REG_REGISTRY,
      'seq2val': s2v_mod._SEQ2VAL_REGISTRY,
  }
  reg = registries.get(area)
  if reg is None:
    return json.dumps({'error': f'unknown area: {area} (use fe|regression|seq2val)'})
  adapter_cls = reg.get(name)
  if adapter_cls is None:
    return json.dumps({'error': f'unknown {area} model: {name}'})
  desc_attr = getattr(adapter_cls, 'describe', None)
  if not callable(desc_attr):
    return json.dumps({'error': f'{area}:{name} has no class/static describe()'})
  try:
    res = desc_attr()  # must be @classmethod or @staticmethod
  except TypeError:
    return json.dumps({'error': f'{area}:{name} describe() must be @classmethod/@staticmethod (no self)'})
  except Exception as e:
    return json.dumps({'error': f'describe() failed: {e}'})
  if isinstance(res, dict):
    out: dict[str, Any] = {'area': area, 'name': name}
    out.update(res)
    # Optionally include source code if adapter provided metadata
    if with_source:
      src_meta = res.get('source') if isinstance(res, dict) else None
      source_code = None
      if isinstance(src_meta, dict):
        mod_name = src_meta.get('module')
        cls_name = src_meta.get('class')
        if mod_name and cls_name:
          try:
            import importlib, inspect
            mod = importlib.import_module(mod_name)
            cls = getattr(mod, cls_name)
            src = inspect.getsource(cls)
            lines = src.splitlines()
            if len(lines) > max_lines:
              src = '\n'.join(lines[:max_lines]) + f"\n# ... truncated {len(lines) - max_lines} lines ..."
            source_code = src
          except Exception as e:
            source_code = f"<error getting source {mod_name}.{cls_name}: {e}>"
        elif src_meta.get('file') and src_meta.get('class'):
          # Fallback: read file and include note; exact class extraction is not guaranteed
          try:
            path = src_meta['file']
            with open(path, 'r') as f:
              content = f.read()
            lines = content.splitlines()
            if len(lines) > max_lines:
              content = '\n'.join(lines[:max_lines]) + f"\n# ... truncated {len(lines) - max_lines} lines ..."
            source_code = content
          except Exception as e:
            source_code = f"<error reading file {src_meta['file']}: {e}>"
      if source_code is not None:
        out['source_code'] = source_code
    return json.dumps(out, indent=2)
  # Allow string (or other types) from describe(); wrap as description
  return json.dumps({'area': area, 'name': name, 'description': str(res)}, indent=2)


@tool
def describe_models(area: str = 'all', with_source: bool = False, max_lines: int = 200) -> str:
  ''' Describe all registered models for a given area, or all areas.

      Parameters
      ----------
      area:
        One of 'fe' | 'regression' | 'seq2val' | 'all' (default: 'all').
      with_source:
        If True and adapter provides source metadata, include (truncated) source code.
      max_lines:
        Max number of source lines when with_source is True.

      Returns
      -------
      JSON mapping area -> list of model descriptions.
  '''
  areas = ['fe', 'regression', 'seq2val'] if str(area).lower() in ('all', '', 'any', 'none') else [str(area).lower()]
  registries = {
      'fe': fe_mod._FE_REGISTRY,
      'regression': reg_mod._REG_REGISTRY,
      'seq2val': s2v_mod._SEQ2VAL_REGISTRY,
  }
  out: dict[str, list[dict[str, Any]]] = {}
  for a in areas:
    reg = registries.get(a)
    if not reg:
      out[a] = [{'error': f'unknown area: {a} (use fe|regression|seq2val|all)'}]
      continue
    models: list[dict[str, Any]] = []
    for name, adapter_cls in reg.items():
      desc_attr = getattr(adapter_cls, 'describe', None)
      if not callable(desc_attr):
        models.append({'area': a, 'name': name, 'error': 'no describe()'})
        continue
      try:
        res = desc_attr()
      except Exception as e:
        models.append({'area': a, 'name': name, 'error': f'describe() failed: {e}'})
        continue
      info: dict[str, Any] = {'area': a, 'name': name}
      if isinstance(res, dict):
        info.update(res)
        if with_source:
          src_meta = res.get('source') if isinstance(res, dict) else None
          source_code = None
          if isinstance(src_meta, dict):
            mod_name = src_meta.get('module')
            cls_name = src_meta.get('class')
            if mod_name and cls_name:
              try:
                import importlib, inspect
                mod = importlib.import_module(mod_name)
                cls = getattr(mod, cls_name)
                src = inspect.getsource(cls)
                lines = src.splitlines()
                if len(lines) > max_lines:
                  src = '\n'.join(lines[:max_lines]) + f"\n# ... truncated {len(lines) - max_lines} lines ..."
                source_code = src
              except Exception as e:
                source_code = f"<error getting source {mod_name}.{cls_name}: {e}>"
          if source_code is not None:
            info['source_code'] = source_code
      else:
        info['description'] = str(res)
      models.append(info)
    out[a] = models
  return json.dumps(out, indent=2)

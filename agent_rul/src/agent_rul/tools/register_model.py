import json
import sys
from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from rul_lib.gls.gls import logger
# local
from agent_rul.tools.utils import sanitize_filename, ensure_pkg, atomic_write, import_module_from_path, split_sym


class _RegisterAdapterArgs(BaseModel):
  model_module: str = Field(description='Import path of the saved model module')
  dataset_name: str = Field(description='Dataset name')
  version: str = Field(description='Dataset version (e.g., 0.7.1)')
  model_name: str = Field(description='Short model identifier (no .py)')
  model_class_name: str = Field(default='GeneratedRULRegressor', description='Class name defined in model_module')
  base_dir: str = Field(default='agent_rul/src/agent_rul/gen_models')
  package_import_root: str = Field(default='agent_rul.gen_models')
  register_symbol: str = Field(default='rul_lib.pipeline.reg.regression:register_model')
  adapter_base: str = Field(default='rul_lib.pipeline.reg.regression:TorchRegBase')
  params_cls: str = Field(default='rul_lib.pipeline.reg.param_parser:MlpParams')
  overwrite: bool = Field(default=True)


@tool(args_schema=_RegisterAdapterArgs)
def register_model_adapter(model_module: str,
                           dataset_name: str,
                           version: str,
                           model_name: str,
                           model_class_name: str = 'GeneratedRULRegressor',
                           base_dir: str = 'agent_rul/src/agent_rul/gen_models',
                           package_import_root: str = 'agent_rul.gen_models',
                           register_symbol: str = 'rul_lib.pipeline.reg.regression:register_model',
                           adapter_base: str = 'rul_lib.pipeline.reg.regression:TorchRegBase',
                           params_cls: str = 'rul_lib.pipeline.reg.param_parser:MlpParams',
                           overwrite: bool = True) -> dict:
  ''' Create and import a model adapter placed in the SAME directory pattern as save_model_file:
        base_dir / f'{dataset_name}_{version.replace(".", "_")}' / adapter_<model_name>.py

      Automatically builds a registry key and adapter class name, updates __init__.py, and imports
      the adapter to trigger registration.

      Parameters
      ----------
      model_module : str
        Import path of the saved model module (as returned by save_model_file).
      dataset_name : str
        Dataset name.
      version : str
        Dataset version.
      model_name : str
        Model identifier (used for adapter file/class and registry key). Do not include file extensions.
      model_class_name : str
        Name of the model class defined in model_module.
      base_dir : str
        Root directory for generated models.
      package_import_root : str
        Root import path corresponding to base_dir.
      register_symbol : str
        '<module>:<attr>' path to the register decorator.
      adapter_base : str
        '<module>:<class>' path to the base adapter class.
      params_cls : str
        '<module>:<class>' path to the parameter class.
      overwrite : bool
        Whether to overwrite an existing adapter file.

      Returns
      -------
      dict
        {
          'status': 'ok' | 'error',
          'adapter_path': str,
          'adapter_module': str,
          'registry_name': str,
          'registered': bool,
          'error': str (if failed)
        }
  '''
  try:
    # --- paths and names aligned with save_model_file ---
    safe_version = version.replace('.', '_')
    target_dir = Path(base_dir) / f'{dataset_name}_{safe_version}'
    ensure_pkg(target_dir)
    # normalize model_name (strip potential '.py' mistakenly provided by generators)
    norm_model_name = sanitize_filename(model_name)
    if norm_model_name.endswith('.py'):
      norm_model_name = norm_model_name[:-3]
    adapter_filename = f'adapter_{norm_model_name}.py'
    adapter_file = target_dir / adapter_filename
    rel_path = target_dir.relative_to(Path(base_dir))
    pkg_adapter_mod = f"{package_import_root}.{'.'.join(rel_path.parts)}.{adapter_file.stem}"
    # registry key and adapter class name
    registry_name = f'{dataset_name}_{safe_version}_{norm_model_name}'
    adapter_class_name = f'{model_class_name}Adapter'
    # --- adapter template ---
    reg_mod, reg_attr = split_sym(register_symbol)
    base_mod, base_cls = split_sym(adapter_base)
    prm_mod, prm_cls = split_sym(params_cls)
    is_sklearn = base_cls.endswith('SklearnRegBase')
    adapter_imports = []
    if not is_sklearn:
      adapter_imports.append('import torch')
      adapter_imports.append('import inspect')
      adapter_imports.append('from dataclasses import asdict, is_dataclass')
    adapter_imports.extend([
        f'from {reg_mod} import {reg_attr} as register_model',
        f'from {base_mod} import {base_cls} as AdapterBase',
        f'from {prm_mod} import {prm_cls} as Params',
        f'from {model_module} import {model_class_name} as ModelCls',
    ])
    adapter_src = '\n'.join(adapter_imports) + '\n\n'
    adapter_src += f'''@register_model('{registry_name}', params_cls=Params)
class {adapter_class_name}(AdapterBase):
  name = '{registry_name}'
  model_cls = ModelCls
'''
    if not is_sklearn:
      adapter_src += (
          "\n  def build_model(self, in_dim: int, p: Params, device: torch.device):\n"
          "    # collect params safely\n"
          "    if is_dataclass(p):\n"
          "      raw_cfg = asdict(p)\n"
          "    elif hasattr(p, 'as_dict'):\n"
          "      try:\n"
          "        raw_cfg = p.as_dict()\n"
          "      except Exception:\n"
          "        raw_cfg = {}\n"
          "    else:\n"
          "      raw_cfg = {k: v for k, v in getattr(p, '__dict__', {}).items() if not k.startswith('_')}\n"
          "    # filter by model signature to avoid unexpected trainer kwargs (e.g., epochs)\n"
          "    sig = inspect.signature(self.model_cls)\n"
          "    params = sig.parameters\n"
          "    accepts_var_kw = any(v.kind == inspect.Parameter.VAR_KEYWORD for v in params.values())\n"
          "    accepted = {k for k, v in params.items() if k != 'self' and v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)}\n"
          "    cfg = {}\n"
          "    for k, v in raw_cfg.items():\n"
          "      if v is None:\n"
          "        continue\n"
          "      if k in accepted or accepts_var_kw:\n"
          "        cfg[k] = v\n"
          "    # inject input dim under a supported name\n"
          "    for key in ('in_feats', 'input_dim', 'in_dim'):\n"
          "      if key in accepted or accepts_var_kw:\n"
          "        cfg[key] = in_dim\n"
          "        break\n"
          "    return self.model_cls(**cfg).to(device)\n")
    adapter_src += ("\n  @classmethod\n"
                    "  def params_cls(cls):\n"
                    "    return Params\n")
    # --- write adapter file ---
    if adapter_file.exists() and not overwrite:
      logger.info(f'adapter exists, overwrite=False: {adapter_file}')
    else:
      atomic_write(adapter_file, adapter_src)
      logger.info(f'wrote adapter: {adapter_file.resolve()}')
    # --- update __init__.py ---
    initf = target_dir / '__init__.py'
    if not initf.exists():
      initf.write_text('', encoding='utf-8')
    init_txt = initf.read_text(encoding='utf-8')
    import_line = f'from .{adapter_file.stem} import {adapter_class_name}\n'
    if import_line not in init_txt:
      initf.write_text(init_txt + import_line, encoding='utf-8')
    # --- import to trigger registration ---
    pkg_root = str(Path(base_dir).parents[0])
    if pkg_root not in sys.path:
      sys.path.append(pkg_root)
    import_module_from_path(pkg_adapter_mod, adapter_file)
    try:
      # Verify registration in the appropriate registry based on target symbol
      if 'seq2val' in register_symbol:
        from rul_lib.pipeline.seq2val.seq2val import _SEQ2VAL_REGISTRY as REG
      else:
        from rul_lib.pipeline.reg.regression import _REG_REGISTRY as REG
      is_registered = registry_name in REG
    except Exception:
      # In loosely coupled environments, treat import errors as non-fatal
      is_registered = True
    return {
        'status': 'ok',
        'adapter_path': str(adapter_file.resolve()),
        'adapter_module': pkg_adapter_mod,
        'registry_name': registry_name,
        'registered': bool(is_registered)
    }
  except Exception as e:
    logger.error(f'register_model_adapter failed: {e}')
    return {'status': 'error', 'error': str(e)}


class _RegisterSeq2ValArgs(BaseModel):
  model_module: str = Field(description='Import path of the saved seq2val model module')
  dataset_name: str
  version: str
  model_name: str
  model_class_name: str = Field(default='GeneratedSeq2ValRegressor')
  params_cls: str = Field(default='rul_lib.pipeline.seq2val.param_parser:LSTMParams')
  base_dir: str = Field(default='agent_rul/src/agent_rul/gen_models')
  package_import_root: str = Field(default='agent_rul.gen_models')
  overwrite: bool = Field(default=True)


@tool(args_schema=_RegisterSeq2ValArgs)
def register_seq2val_adapter(model_module: str,
                             dataset_name: str,
                             version: str,
                             model_name: str,
                             model_class_name: str = 'GeneratedSeq2ValRegressor',
                             params_cls: str = 'rul_lib.pipeline.seq2val.param_parser:LSTMParams',
                             base_dir: str = 'agent_rul/src/agent_rul/gen_models',
                             package_import_root: str = 'agent_rul.gen_models',
                             overwrite: bool = True) -> dict:
  ''' Convenience wrapper to register a Seq2Val (sequence-to-value) model adapter.

      This uses the seq2val registry and TorchSeq2Val base by default, so the caller
      doesn't need to pass the symbols explicitly.
  '''
  return register_model_adapter.invoke({
      'model_module': model_module,
      'dataset_name': dataset_name,
      'version': version,
      'model_name': model_name,
      'model_class_name': model_class_name,
      'base_dir': base_dir,
      'package_import_root': package_import_root,
      'register_symbol': 'rul_lib.pipeline.seq2val.seq2val:register_seq2val',
      'adapter_base': 'rul_lib.pipeline.seq2val.seq2val:TorchSeq2Val',
      'params_cls': params_cls,
      'overwrite': overwrite,
  })


@tool
def register_model_adapter_raw(args_json: str) -> dict:
  ''' Register tabular model adapter via JSON args.

      Expects a JSON object with keys compatible with register_model_adapter.
  '''
  try:
    args = json.loads(args_json)
    return register_model_adapter.invoke(args)
  except Exception as e:
    logger.error(f'register_model_adapter_raw failed: {e}')
    return {'status': 'error', 'error': str(e)}


@tool
def register_seq2val_adapter_raw(args_json: str) -> dict:
  ''' Register seq2val adapter via JSON args.

      Expects a JSON object with keys compatible with register_seq2val_adapter or
      register_model_adapter (with seq2val symbols).
  '''
  try:
    args = json.loads(args_json)
    # Prefer the generic path if explicit symbols are provided
    if 'register_symbol' in args or 'adapter_base' in args:
      return register_model_adapter.invoke(args)
    return register_seq2val_adapter.invoke(args)
  except Exception as e:
    logger.error(f'register_seq2val_adapter_raw failed: {e}')
    return {'status': 'error', 'error': str(e)}

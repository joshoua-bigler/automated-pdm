import sys, hashlib
from pathlib import Path
from langchain_core.tools import tool
from rul_lib.gls.gls import logger
# local
from agent_rul.tools.utils import strip_fences, sanitize_filename, ensure_pkg, atomic_write, import_module_from_path


@tool
def save_model_file(content: str,
                    dataset_name: str,
                    version: str,
                    model_name: str,
                    base_dir: str = 'agent_rul/src/agent_rul/gen_models',
                    model_filename: str = 'generated_model.py',
                    package_import_root: str = 'agent_rul.gen_models',
                    overwrite: bool = True) -> dict:
  ''' Save a generated model source file into a structured directory layout.

      This function writes the provided model code to:
        base_dir / (dataset_name_<version>) / model_filename

      It ensures that each directory level is importable as a Python package,
      validates basic syntax (requires at least one class definition),
      and dynamically imports the module once to confirm importability.

      Parameters
      ----------
      content : str
        Python source code of the model (may include ``` fences).
      dataset_name : str
        Dataset name used to form the directory prefix.
      version : str
        Dataset version appended to dataset_name, sanitized for import safety.
      model_name : str
        Model identifier, used only for logging and introspection.
      base_dir : str, default='agent_rul/src/agent_rul/gen_models'
        Root directory where generated models are stored.
      model_filename : str, default='generated_model.py'
        File name of the model source file.
      package_import_root : str, default='agent_rul.gen_models'
        Root Python import path corresponding to base_dir.
      overwrite : bool, default=True
        If False, prevents overwriting existing model files.

      Returns
      -------
      dict
        {
          'status': 'ok' | 'exists' | 'error',
          'model_path': absolute path to the saved file,
          'model_module': importable Python module path,
          'sha256': file content hash,
          'error': error message (only if status='error')
        }
  '''
  try:
    safe_version = version.replace('.', '_')
    target_dir = Path(base_dir) / f'{dataset_name}_{safe_version}'
    ensure_pkg(target_dir)
    rel_path = target_dir.relative_to(Path(base_dir))
    model_module = f"{package_import_root}.{'.'.join(rel_path.parts)}.{Path(model_filename).stem}"
    model_file = target_dir / sanitize_filename(model_filename)
    code = strip_fences(content)
    if not code or 'class ' not in code:
      raise ValueError('model content looks empty or lacks a class definition')
    if model_file.exists() and not overwrite:
      sha = hashlib.sha256(model_file.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
      return {'status': 'exists', 'model_path': str(model_file.resolve()), 'model_module': model_module, 'sha256': sha}
    atomic_write(model_file, code)
    sha = hashlib.sha256(code.encode('utf-8')).hexdigest()
    logger.info(f'wrote model for {dataset_name} v{version}: {model_file.resolve()} (sha256={sha})')
    pkg_root = str(Path(base_dir).parents[0])
    if pkg_root not in sys.path:
      sys.path.append(pkg_root)
    import_module_from_path(model_module, model_file)
    return {'status': 'ok', 'model_path': str(model_file.resolve()), 'model_module': model_module, 'sha256': sha}
  except Exception as e:
    logger.error(f'save_model_file failed: {e}')
    return {'status': 'error', 'error': str(e)}

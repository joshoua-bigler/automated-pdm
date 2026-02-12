import re, importlib, importlib.util, sys
from pathlib import Path
from tempfile import NamedTemporaryFile


def strip_fences(text: str) -> str:
  ' Remove markdown code fences from a text block if present. '
  s = text.strip()
  if s.startswith('```'):
    m = re.match(r'^```[a-zA-Z0-9_+-]*\s*(.*?)\s*```', s, flags=re.DOTALL)
    if m:
      return m.group(1).strip()
  return s


def sanitize_filename(name: str) -> str:
  ' Sanitize a string to be a safe filename. '
  name = re.sub(r'[^A-Za-z0-9._-]', '_', name.strip().replace(' ', '_'))
  return name if name.endswith('.py') else f'{name}.py'


def ensure_pkg(path: Path) -> None:
  ' Ensure that the given path exists and is a Python package (has __init__.py). '
  path.mkdir(parents=True, exist_ok=True)
  initf = path / '__init__.py'
  if not initf.exists():
    initf.write_text('', encoding='utf-8')


def atomic_write(path: Path, content: str) -> None:
  ' Atomically write content to a file at the given path. '
  with NamedTemporaryFile('w', encoding='utf-8', dir=str(path.parent), delete=False) as tmp:
    tmp.write(content.rstrip() + '\n')
    tmp_path = Path(tmp.name)
  tmp_path.replace(path)


def import_module_from_path(mod_name: str, file_path: Path):
  ''' Dynamically import a module from the given file path.

      Parameters
      ----------
      mod_name : str
        Name to assign to the imported module.
      file_path : Path
        Path to the .py file to import.

      Returns
      -------
      module
        The imported Python module.
  '''
  spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
  module = importlib.util.module_from_spec(spec)
  sys.modules[mod_name] = module
  assert spec.loader is not None
  spec.loader.exec_module(module)
  return module


def split_sym(sym: str) -> tuple[str, str]:
  mod, attr = sym.split(':', 1)
  return mod, attr

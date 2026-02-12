import os
import json
from pathlib import Path
from importlib import resources as res
from langchain_core.tools import tool
# local
from agent_rul.gls import USER_INFORMATIONS_DIR


def _prompts_dir() -> Path:
  ''' Resolve base directory for user information markdown files.

      Resolution order:
      1) Environment variable AGENT_RUL_USER_INFO_DIR (if set)
      2) USER_INFORMATION_DIR from agent_rul.gls (if exists)
      3) Package prompts directory (agent_rul.prompts)
      4) Current working directory (fallback)
  '''
  env_dir = os.getenv('AGENT_RUL_USER_INFO_DIR')
  if env_dir:
    p = Path(env_dir)
    if p.exists():
      return p
  try:
    if isinstance(USER_INFORMATIONS_DIR, Path) and USER_INFORMATIONS_DIR.exists():
      return USER_INFORMATIONS_DIR
  except Exception:
    pass
  try:
    p = Path(res.files('agent_rul.prompts'))
    if p.exists():
      return p
  except Exception:
    pass
  # Fallback to current working directory if package resources are unavailable
  return Path.cwd()


@tool
def list_user_info_markdown() -> str:
  ''' List available user information markdown files in the prompts directory. 
  
      Returns 
      -------
      A JSON string with base directory and list of files.
  '''
  base = _prompts_dir()
  if not base.exists() or not base.is_dir():
    return json.dumps({"base": str(base), "files": []}, indent=2)
  files = []
  for entry in sorted(base.glob('*.md')):
    files.append({"name": entry.name, "path": str(entry)})
  return json.dumps({"base": str(base), "files": files}, indent=2)


@tool
def load_user_instruction(name: str) -> str:
  ''' Load a user instruction markdown file from the prompts directory.

      Accepts either a plain name (e.g., "user_information") or a filename
      (e.g., "user_information.md"). Only files within the prompts directory or
      its direct subdirectory specified in the name are allowed.
  '''
  base = _prompts_dir()
  fn = Path(name)
  if fn.suffix.lower() != '.md':
    fn = fn.with_suffix('.md')
  # Normalize against base to prevent path traversal
  path = fn if fn.is_absolute() else (base / fn)
  path = path.resolve()
  try:
    base_resolved = base.resolve()
  except Exception:
    base_resolved = base
  if base_resolved not in path.parents and path != base_resolved:
    raise PermissionError('access outside prompts directory is not allowed')
  if not path.exists() or not path.is_file():
    raise FileNotFoundError(f'no markdown file found at {path}')
  return path.read_text(encoding='utf-8')

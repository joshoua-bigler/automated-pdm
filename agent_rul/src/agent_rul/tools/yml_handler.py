import io
from langchain_core.tools import tool
from pathlib import Path
from ruamel.yaml import YAML
# local
from agent_rul.gls import CONFIG_DIR, DEFAULT_TEMPLATE

yaml = YAML(typ='safe')


def _resolve_template(name: str) -> Path:
  p = Path(name)
  if not p.suffix:
    p = p.with_suffix('.yml')
  if not p.is_absolute():
    p = CONFIG_DIR / p
  return p


@tool
def list_yaml_templates() -> str:
  ''' Return available YAML template names (newline-separated). '''
  if not CONFIG_DIR.exists():
    return ''
  return '\n'.join(sorted(p.name for p in CONFIG_DIR.glob('*.yml')))


@tool
def load_yaml_template(name: str = DEFAULT_TEMPLATE) -> str:
  ''' Load a YAML template by name and return its contents. '''
  path = _resolve_template(name)
  if not path.exists():
    raise FileNotFoundError(f'no YAML template found at {path}')
  return path.read_text()


@tool
def write_yml(content: str, file_name: str) -> str:
  ''' Write YAML content to pipeline_configs/gen/<file_name> and return written path. '''
  out_dir = CONFIG_DIR / 'gen'
  out_dir.mkdir(parents=True, exist_ok=True)
  path = out_dir / Path(file_name)
  with open(path, 'w') as f:
    f.write(content)
  return str(path)


@tool
def validate_yaml(content: str) -> str:
  ''' Validate YAML syntax and minimal schema. Return "ok" or an error message. '''
  try:
    data = yaml.load(io.StringIO(content))
    if not isinstance(data, dict):
      return 'error: top-level yaml must be a mapping'
    for k in ('dataset', 'preprocessing', 'feature_engineering', 'regression'):
      if k not in data:
        return f'error: missing key: {k}'
    return 'ok'
  except Exception as e:
    return f'error: {e}'


@tool
def inspect_yaml_template(name: str = DEFAULT_TEMPLATE) -> dict:
  ''' Load a YAML template and return a compact structural summary.
      Used by the Prompt Agent to understand available sections and tunable fields.
  '''
  path = _resolve_template(name)
  if not path.exists():
    raise FileNotFoundError(f'no YAML template found at {path}')
  data = yaml.load(path.read_text())
  if not isinstance(data, dict):
    return {'error': 'template not a mapping'}

  def has_space_field(v):
    if isinstance(v, dict):
      if 'space' in v:
        return True
      return any(has_space_field(x) for x in v.values())
    if isinstance(v, list):
      return any(has_space_field(x) for x in v)
    return False

  summary = {
      'path': str(path),
      'sections': list(data.keys()),
      'has_seq2val': 'seq2val' in data,
      'has_regression': 'regression' in data,
      'space_fields': [k for k, v in data.items() if has_space_field(v)],
  }
  return summary

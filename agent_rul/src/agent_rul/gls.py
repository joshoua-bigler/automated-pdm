from pathlib import Path


def find_dir(marker: str):
  p = Path(__file__).resolve()
  for parent in p.parents:
    if (parent / marker).exists():
      return parent / marker
  return p.parents[-1]


CONFIG_DIR = find_dir(marker='pipeline_configs')
DEFAULT_TEMPLATE = 'rul_pipeline_template.yml'
USER_INFORMATIONS_DIR = find_dir(marker='prompts') / Path('user_informations')
RAY_RESULTS_DIR = find_dir(marker='local_data') / Path('ray_results')

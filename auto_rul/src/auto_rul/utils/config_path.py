from pathlib import Path

def get_config_path(tracking: dict) -> Path:
  config_path = Path(__file__).parent.parent / 'pipeline_configs'
  dataset = tracking.get('dataset', {})
  if dataset.get('name', '') == 'cmapps':
    return config_path / f'cmapps/fd{dataset['fd_number']}_{tracking['version']}.yml'
  elif dataset.get('name', '') == 'femto':
    return config_path / f'femto/{tracking['version']}.yml'
from enum import Enum


class Approach(Enum):
  AUTOML = 'automl'
  HYBRID = 'hybrid'
  AGENTIC = 'agentic'


def create_experiment_name(tracking: dict, approach: Approach) -> str:
  ''' Create an experiment name based on the dataset information in tracking. '''
  dataset = tracking.get('dataset', {})
  if not dataset:
    raise ValueError('tracking must contain dataset information')
  dataset_name = dataset.get('name', 'unknown')
  if dataset_name == 'cmapps':
    fd_number = dataset.get('fd_number', 1)
    experiment_name = f'{approach.value}_{dataset_name}_fd{fd_number}'
    return experiment_name
  return f'{approach.value}_{dataset_name}'

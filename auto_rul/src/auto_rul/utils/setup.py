import mlflow
from pathlib import Path
# local
from rul_lib.gls.gls import logger

def setup(tracking: dict)-> tuple[Path, Path]:
  logger.info(f'Run agent-rul experiment {tracking['experiment_name']} v{tracking['version']}')
  mlflow.set_tracking_uri(tracking['mlflow']['tracking_uri'])
  mlflow.set_experiment(tracking['experiment_name'])
  root_path = Path(__file__).resolve().parents[4]
  ray_storage_path = root_path / Path('local_data/ray_results')
  return root_path, ray_storage_path


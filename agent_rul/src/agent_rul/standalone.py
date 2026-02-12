import mlflow
from pathlib import Path
from dotenv import load_dotenv
from rul_lib.gls.gls import logger
from rul_lib.utils.tracking import create_experiment_name, Approach
from rul_lib.data.registry import get_dataset_loader
from rul_lib.pipeline.pre.pre import preprocessing
from rul_lib.pipeline.rul.rul import create_rul
from nb_utils.styles import set_thesis_style
# local
from agent_rul.agents.manager_agent import run_manager
from agent_rul.utils.setup import setup
from agent_rul.gls import CONFIG_DIR


def run_once(tracking: dict, config_template: str | Path = None) -> dict:
  ''' Run a single pipeline (no Ray Tune) end-to-end.

      Steps:
        - load dataset
        - generate concrete config via manager agent
        - preprocess data
        - build windows
        - run either FE+REG or Seq2Val directly (no HPO)
      Returns metrics dict with at least 'val_obj'.
  '''
  root_path, _ = setup(tracking=tracking)
  dataset_loader = get_dataset_loader(name=tracking.get('dataset', {}).get('name', ''))
  raw = dataset_loader(root_path=root_path, config=tracking.get('dataset', {}))
  logger.info(f"Loaded dataset: {tracking.get('dataset', {}).get('name', '')}")
  cfg0 = run_manager(tracking=tracking,
                     data=raw,
                     config_template=CONFIG_DIR / 'gen' / config_template,
                     out_dir=CONFIG_DIR / 'gen',
                     use_cache=True)
  cfg = cfg0.get('config')
  if not cfg:
    raise ValueError('failed to create configuration')
  data = preprocessing(data=raw, config=cfg)
  ml_uri = (tracking.get('mlflow') or {}).get('tracking_uri')
  if ml_uri:
    mlflow.set_tracking_uri(ml_uri)
  exp_name = tracking.get('experiment_name') or create_experiment_name(tracking)
  mlflow.set_experiment(exp_name)
  metrics_out = create_rul(results=None, data=data, tracking=tracking, cfg=cfg)
  if metrics_out and 'val_obj' in metrics_out:
    logger.info(f"Validation objective: {metrics_out['val_obj']}")
  return metrics_out or {}


if __name__ == '__main__':
  load_dotenv()
  set_thesis_style()
  # yapf: disable
  tracking = {
      'mlflow': {'tracking_uri': 'http://localhost:5000', 'rul_model_name': 'rul'},
      'version': '1.0.5',
      'dataset': {'name': 'cmapps', 'fd_number': 4}, # {'name': 'cmapps', 'fd_number': 1}, {'name': 'femto', 'downsampling': {'apply': False, 'factor': 1, 'group_by': 'unit'}}
      'baseline_result': {}, # {'version': '0.9.2'}
      'standalone': True,
  }
  # yapf: enable
  tracking['experiment_name'] = create_experiment_name(tracking=tracking, approach=Approach.AGENTIC)
  run_once(tracking=tracking, config_template='rul_pipeline_template.yml')

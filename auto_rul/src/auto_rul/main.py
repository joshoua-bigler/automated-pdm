import optuna
import ray
import time
from dotenv import load_dotenv
from pathlib import Path
from ray import tune
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler
from ray.tune import RunConfig
from ray.tune.logger import TBXLoggerCallback
from rul_lib.gls.gls import logger
from rul_lib.optimization.actors import FeatureCache
from rul_lib.utils.tracking import create_experiment_name, Approach
from rul_lib.pipeline.pre.pre import preprocessing
from rul_lib.utils.search_space import build_search_space
from rul_lib.data.registry import get_dataset_loader
from rul_lib.optimization.ray_logger import make_trial_name, make_trial_dirname
from rul_lib.optimization.trainer import train_pipeline
from rul_lib.pipeline.rul.rul import create_rul
from rul_lib.data.data_loader import load_config
# local
from auto_rul.utils.setup import setup
from auto_rul.utils.config_path import get_config_path
from nb_utils.styles import set_thesis_style
set_thesis_style()


def main(tracking: dict):
  root_path, ray_storage_path = setup(tracking=tracking)
  dataset_loader = get_dataset_loader(name=tracking.get('dataset', {}).get('name', ''))
  raw = dataset_loader(root_path=root_path, config=tracking.get('dataset', {}))
  cfg0 = load_config(config_path=get_config_path(tracking=tracking))
  if not cfg0:
    raise ValueError('failed to create configuration')
  data = preprocessing(data=raw, config=cfg0)
  search_space = build_search_space(cfg=cfg0)
  data_ref = {k: ray.put(v) for k, v in data.items()}
  tracking_ref = ray.put(tracking)
  feat_cache = FeatureCache.remote()
  db_path = root_path / Path('local_data/optuna/bo.db').absolute()
  storage = optuna.storages.RDBStorage(url=f'sqlite:///{db_path}')
  tpe = optuna.samplers.TPESampler(seed=42)
  study_name = f'{tracking['experiment_name']}_v{tracking['version']}_{time.strftime('%Y%m%d%H%M%S')}'
  alg = OptunaSearch(storage=storage, sampler=tpe, metric='val_obj', mode='min', study_name=study_name)
  alg = ConcurrencyLimiter(alg, max_concurrent=4)
  asha = ASHAScheduler(time_attr='training_iteration', metric='val_obj', mode='min', grace_period=5, reduction_factor=3)
  ray_run_name = f'{tracking["experiment_name"]}_v{tracking["version"]}'
  # yapf: disable
  tuner = tune.Tuner(trainable=tune.with_resources(tune.with_parameters(train_pipeline, data_ref=data_ref, feat_cache=feat_cache, tracking=tracking_ref), {'gpu': 1, 'cpu': 4}),
                     param_space=search_space,
                     tune_config=tune.TuneConfig(num_samples=tracking['num_samples'],
                     scheduler=asha,
                     search_alg=alg,
                     time_budget_s=tracking.get('time_budget_s', 10 * 3600),
                     trial_name_creator=make_trial_name,
                     trial_dirname_creator=make_trial_dirname),
                     run_config=RunConfig(storage_path=ray_storage_path, name=ray_run_name, verbose=0, log_to_file=True, callbacks=[TBXLoggerCallback()])
                    )
  t0 = time.perf_counter()
  results = tuner.fit()
  metrics = {'ray_search_time': time.perf_counter() - t0}
  create_rul(results=results, data=data, tracking=tracking, ray_storage_path=ray_storage_path, metrics=metrics)

if __name__ == '__main__':
  load_dotenv()
  tracking = {
      'mlflow': {
          'tracking_uri': 'http://localhost:5000',
          'rul_model_name': 'rul',
      },
      'version': '1.0.10',
      'dataset': {'name': 'cmapps', 'fd_number': 1}, # {'name': 'cmapps', 'fd_number': 1}, #{'name': 'femto', 'downsampling': {'apply': False, 'factor': 1, 'group_by': 'unit'}},
      'num_samples': 2,
      'time_budget_s': 2 * 3600,
  }
  tracking['experiment_name'] = create_experiment_name(tracking=tracking, approach=Approach.AUTOML)
  if False:
    logger.warning('RUNNING IN RAY LOCAL MODE!')
    ray.init(local_mode=True)
  main(tracking=tracking)
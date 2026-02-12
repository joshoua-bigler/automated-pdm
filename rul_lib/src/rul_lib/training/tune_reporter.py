import time
from dataclasses import dataclass, field
from ray import tune
# local
from rul_lib.training.loss import ConfigCriterion


@dataclass
class CustomTuneConfig:
  enabled: bool = True
  time_attr: str = 'training_iteration'  # must match ASHAScheduler.time_attr
  report_keys: list[str] = field(default_factory=list)  # if empty → send all
  aliases: dict[str, str] = field(default_factory=dict)  # optional key renames
  objective_key: str = ''  # '' → fall back to trainer default
  step_offset: int = 0  # e.g., pretrain epochs to offset steps


class TuneReporter:
  ''' Reports training metrics to Ray Tune. '''

  def __init__(self, cfg: CustomTuneConfig):
    self.cfg = cfg
    self._t0 = None

  def start(self) -> None:
    self._t0 = time.perf_counter()

  def report(self, step: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
    if not self.cfg.enabled:
      return
    if self._t0 is None:
      self.start()
    payload = {**train_metrics, **val_metrics}
    if self.cfg.report_keys:
      payload = {k: payload[k] for k in self.cfg.report_keys if k in payload}
    payload = {self.cfg.aliases.get(k, k): v for k, v in payload.items()}
    payload[self.cfg.time_attr] = int(step) + int(self.cfg.step_offset)
    payload['time_total_s'] = float(time.perf_counter() - self._t0)

    tune.report(payload)


def _extend_keys_for_criterion(keys: list[str], cfg_criterion: ConfigCriterion | None) -> list[str]:
  if not cfg_criterion or not cfg_criterion.kind:
    return keys
  k = (cfg_criterion.kind or 'mse').strip().lower()
  if k in ('nasa', 'mse+nasa', 'combo', 'quantile+nasa', 'q+nasa'):
    keys += ['train_nasa', 'val_nasa']
  if k == 'huber':
    keys += ['train_huber', 'val_huber']
  if k in ('quantile', 'pinball', 'wquantile', 'rul_quantile', 'rul-quantile'):
    keys += ['train_quantile', 'val_quantile']
  if k in ('nll', 'gaussian_nll', 'gauss_nll'):
    keys += ['train_nll', 'val_nll']
  return keys


def _default_objective_for_criterion(cfg_criterion: ConfigCriterion | None) -> str:
  if not cfg_criterion or not cfg_criterion.kind:
    return 'val_obj'
  k = (cfg_criterion.kind or 'mse').strip().lower()
  # if your compute_metrics sets obj already for these, keep val_obj
  if k in ('mse', 'huber', 'nasa', 'mse+nasa', 'combo', 'quantile+nasa', 'q+nasa'):
    return 'val_obj'
  # if you want to optimize the raw loss term instead of obj:
  if k in ('quantile', 'pinball', 'wquantile', 'rul_quantile', 'rul-quantile'):
    return 'val_quantile'
  if k in ('nll', 'gaussian_nll', 'gauss_nll'):
    return 'val_nll'
  return 'val_obj'


def make_tune_cfg(phase: str, fe_epochs: int = 0, cfg_criterion: ConfigCriterion | None = None) -> CustomTuneConfig:
  ''' Create a CustomTuneConfig for the given training phase. '''
  if phase == 'fe':
    return CustomTuneConfig(
        enabled=False,
        report_keys=['train_loss', 'val_loss', 'lr'],
        aliases={
            'train_loss': 'fe_train_loss',
            'val_loss': 'fe_val_loss',
            'lr': 'fe_lr',
        },
        objective_key='val_loss',
        step_offset=0,
    )
  common_keys = [
      'train_loss',
      'train_rmse',
      'train_obj',
      'val_loss',
      'val_rmse',
      'val_obj',
      'lr',
      'train_mse',
      'val_mse',
  ]
  common_keys = _extend_keys_for_criterion(common_keys, cfg_criterion)
  if phase == 'reg':
    return CustomTuneConfig(report_keys=common_keys,
                            objective_key=_default_objective_for_criterion(cfg_criterion),
                            step_offset=fe_epochs)
  if phase == 'seq2val':
    return CustomTuneConfig(report_keys=common_keys,
                            objective_key=_default_objective_for_criterion(cfg_criterion),
                            step_offset=0)

  raise ValueError('phase must be "fe", "reg" or "seq2val"')

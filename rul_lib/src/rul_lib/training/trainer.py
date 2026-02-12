import math, json
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any
from torch.optim.lr_scheduler import _LRScheduler
# local
from rul_lib.training.train_logger import TrainLogger, NullLogger
from rul_lib.training.tune_reporter import CustomTuneConfig, TuneReporter
from rul_lib.gls.gls import logger
from rul_lib.training.loss import make_criterion, ConfigCriterion, compute_metrics


class BaseTrainer(ABC):
  ''' PyTorch training loop with pluggable logger. '''

  def __init__(self,
               model: nn.Module,
               run_name: str,
               optimizer: torch.optim.Optimizer,
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               num_epochs: int,
               device: torch.device,
               version: str = '0.0.0',
               scheduler: _LRScheduler | None = None,
               config: dict[str, Any] | None = None,
               patience: int = 20,
               train_logger: TrainLogger | None = None,
               tune_cfg: CustomTuneConfig | None = None):
    self.model = model.to(device)
    self.run_name = run_name
    self.optimizer = optimizer
    self.train_loader = train_loader
    self.val_loader = val_loader
    self.num_epochs = num_epochs
    self.device = device
    self.scheduler = scheduler
    self.config = config or {}
    self.patience = patience
    self.best_val_score = math.inf
    self.best_state = None
    self.counter = 0
    self.version = version
    self.train_logger = train_logger or NullLogger()
    self.tune_reporter = TuneReporter(tune_cfg or CustomTuneConfig())
    self._objective_key = (tune_cfg.objective_key if tune_cfg and tune_cfg.objective_key else None)

  @abstractmethod
  def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    pass

  @abstractmethod
  def val_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    pass

  @abstractmethod
  def primary_metric_name(self) -> str:
    pass

  def _epoch_loop(self, epoch: int) -> tuple[dict[str, float], dict[str, float]]:
    self.model.train()
    train_agg: dict[str, float] = {}
    train_count = 0
    for batch in self.train_loader:
      m = self.train_step(batch)
      for k, v in m.items():
        train_agg[k] = train_agg.get(k, 0.0) + float(v)
      train_count += 1
    train_metrics = {f'train_{k}': v / max(1, train_count) for k, v in train_agg.items()}
    if self.optimizer.param_groups:
      train_metrics['lr'] = float(self.optimizer.param_groups[0].get('lr', None))
    if self.scheduler:
      self.scheduler.step()
    # val
    self.model.eval()
    val_agg: dict[str, float] = {}
    val_count = 0
    with torch.no_grad():
      for batch in self.val_loader:
        m = self.val_step(batch)
        for k, v in m.items():
          val_agg[k] = val_agg.get(k, 0.0) + float(v)
        val_count += 1
    val_metrics = {k: v / max(1, val_count) for k, v in val_agg.items()}
    return train_metrics, val_metrics

  def _maybe_early_stop(self, val_metrics: dict[str, float]) -> bool:
    key = self.primary_metric_name()
    score = val_metrics.get(key)
    if score is None:
      raise ValueError(f'primary metric {key!r} not found in val metrics: {val_metrics}')
    if score < self.best_val_score:
      self.best_val_score = score
      self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
      self.counter = 0
      return False
    self.counter += 1
    if self.counter >= self.patience:
      if self.best_state:
        self.model.load_state_dict(self.best_state)
      return True
    return False

  def train(self,
            register_name: str | None = None,
            tags: dict[str, str] | None = None,
            code_paths: list[str] | None = None) -> nn.Module:
    self.train_logger.start(run_name=self.run_name)
    pg = self.optimizer.param_groups[0] if self.optimizer.param_groups else {}
    base_params = {
        'epochs': self.num_epochs,
        'batch_size': self.train_loader.batch_size,
        'train_size': len(self.train_loader.dataset),
        'val_size': len(self.val_loader.dataset),
        'learning_rate': pg.get('lr'),
        'optimizer_type': self.optimizer.__class__.__name__,
        'weight_decay': pg.get('weight_decay', 0.0),
        'model': self.model.__class__.__name__,
        # 'num_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
        'version': self.version
    }
    if self.scheduler:
      base_params.update({
          'scheduler_type': self.scheduler.__class__.__name__,
          'scheduler_state': json.dumps(self.scheduler.state_dict(), default=str)
      })
    self.train_logger.log_params({**base_params, **self.config})
    # epochs
    for epoch in range(self.num_epochs):
      train_metrics, val_metrics = self._epoch_loop(epoch)
      self.train_logger.log_metrics(train_metrics, step=epoch)
      self.train_logger.log_metrics(val_metrics, step=epoch)
      try:
        self.tune_reporter.report(step=epoch + 1, train_metrics=train_metrics, val_metrics=val_metrics)
      except Exception:
        logger.error('Error reporting to Tune', exc_info=True)
      stop = self._maybe_early_stop(val_metrics)
      if stop:
        break
    if self.best_state is not None:
      self.model.load_state_dict(self.best_state)
    self.train_logger.log_model(model=self.model,
                                train_loader=self.train_loader,
                                code_paths=code_paths or [],
                                registered_model_name=register_name,
                                tags=tags)
    self.train_logger.close()
    return self.model


class RegressionTrainer(BaseTrainer):
  ''' Trainer for regression tasks with configurable criterion. '''

  def __init__(self, cfg_criterion: ConfigCriterion, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.cfg_criterion = cfg_criterion
    self.criterion = make_criterion(config=cfg_criterion)

  def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    x, y = batch
    x, y = x.to(self.device), y.to(self.device)
    self.optimizer.zero_grad()
    pred = self.model(x)
    # Align shapes to avoid broadcasting (e.g., pred=[B] vs y=[B,1])
    if hasattr(pred, 'ndim') and pred.ndim == 2 and pred.shape[-1] == 1:
      pred = pred.squeeze(-1)
    if hasattr(y, 'ndim') and y.ndim == 2 and y.shape[-1] == 1:
      y = y.squeeze(-1)
    loss = self.criterion(pred, y)
    loss.backward()
    self.optimizer.step()
    metrics = compute_metrics(pred=pred.detach(), y=y.detach(), config=self.cfg_criterion)
    return {'loss': float(loss.detach()), **metrics}

  def val_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    x, y = batch
    x, y = x.to(self.device), y.to(self.device)
    with torch.no_grad():
      pred = self.model(x)
      if hasattr(pred, 'ndim') and pred.ndim == 2 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
      if hasattr(y, 'ndim') and y.ndim == 2 and y.shape[-1] == 1:
        y = y.squeeze(-1)
      loss = self.criterion(pred, y)
      metrics = compute_metrics(pred=pred, y=y, config=self.cfg_criterion, eval=True)
    metrics['val_loss'] = float(loss.item())
    return metrics

  def primary_metric_name(self) -> str:
    return self._objective_key or 'val_obj'


class FeTrainer(BaseTrainer):
  ''' Trainer for feature extractors with configurable criterion. '''

  def __init__(self, criterion: nn.Module, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.criterion = criterion

  def train_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    x, y = batch
    x, y = x.to(self.device), y.to(self.device)
    self.optimizer.zero_grad()
    pred = self.model(x)
    loss = self.criterion(pred, y)
    loss.backward()
    self.optimizer.step()
    rmse = float(torch.sqrt(loss.detach()))
    return {'loss': float(loss.detach()), 'rmse': rmse}

  def val_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    x, y = batch
    x, y = x.to(self.device), y.to(self.device)
    with torch.no_grad():
      pred = self.model(x)
      loss = self.criterion(pred, y)
      rmse = float(torch.sqrt(loss))
    return {'val_loss': float(loss), 'val_rmse': rmse}

  def primary_metric_name(self) -> str:
    return self._objective_key or 'val_rmse'

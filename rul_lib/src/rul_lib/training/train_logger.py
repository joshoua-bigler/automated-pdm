import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from abc import ABC
from mlflow.models import infer_signature


def log_model(model: nn.Module,
              train_loader: torch.utils.data.DataLoader,
              code_paths: list[str] | None = None,
              registered_model_name: str | None = None,
              tags: dict[str, str] | None = None) -> None:
  # Build a tiny input example
  xb, _ = next(iter(train_loader))
  xb = xb[:1].cpu()
  x_example = xb.numpy()
  signature = None
  # Detect composite RUL model with a non-torch (e.g., sklearn) regressor
  reg = getattr(model, 'regressor', None)
  has_sklearn_reg = (reg is not None) and hasattr(reg, 'predict') and not hasattr(reg, 'parameters')
  if not has_sklearn_reg:
    # Pure Torch model path: temporarily move model to CPU for signature/logging, then restore to its original device.
    try:
      orig_device = next(model.parameters()).device
    except StopIteration:
      orig_device = torch.device('cpu')
    try:
      with torch.inference_mode():
        model.to('cpu')
        yb = model(xb).detach().numpy()
      signature = infer_signature(x_example, yb)
    except Exception:
      signature = None
    model_info = mlflow.pytorch.log_model(model.cpu(),
                                          artifact_path='model',
                                          input_example=x_example,
                                          signature=signature,
                                          code_paths=code_paths or [],
                                          registered_model_name=registered_model_name)
    if tags:
      try:
        mlflow.register_model(model_uri=model_info.model_uri, name=registered_model_name or 'model', tags=tags)
      except Exception:
        pass
    try:
      model.to(orig_device)
    except Exception:
      pass
    return


class TrainLogger(ABC):
  ''' Abstract base class for training loggers. '''

  def start(self, run_name: str) -> None:
    pass

  def log_params(self, params: dict) -> None:
    pass

  def log_metrics(self, metrics: dict[str, float], step: int) -> None:
    pass

  def log_model(self,
                model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                code_paths: list[str] | None = None,
                registered_model_name: str | None = None,
                tags: dict[str, str] | None = None) -> None:
    pass

  def log_text(self, text: str, artifact_file: str) -> None:
    pass

  def close(self) -> None:
    pass


class NullLogger(TrainLogger):
  ''' A no-op logger that does nothing. '''
  pass


class MlflowLogger(TrainLogger):
  ''' MLflow-based training logger. '''

  def __init__(self):
    self.mlflow = mlflow
    self._run = None

  def start(self, run_name: str) -> None:
    if self._run:
      return
    self._run = self.mlflow.start_run(run_name=run_name)

  def log_params(self, params: dict) -> None:
    if not self._run:
      return
    run = self.mlflow.active_run()
    already = run.data.params if run else {}
    new_params = {k: str(v) for k, v in params.items() if k not in already}
    if new_params:
      self.mlflow.log_params(new_params)

  def log_metrics(self, metrics: dict[str, float], step: int) -> None:
    if not self._run:
      return
    for k, v in metrics.items():
      try:
        self.mlflow.log_metric(k, float(v), step=step)
      except (TypeError, ValueError):
        # Fall back to logging string-like values as params to avoid mlflow metric errors
        self.log_params({k: v})

  def log_model(self,
                model: nn.Module,
                train_loader: torch.utils.data.DataLoader,
                code_paths: list[str] | None = None,
                registered_model_name: str | None = None,
                tags: dict[str, str] | None = None) -> None:
    if not self._run:
      return
    log_model(model=model,
              train_loader=train_loader,
              code_paths=code_paths,
              registered_model_name=registered_model_name,
              tags=tags)

  def close(self) -> None:
    if self._run:
      self.mlflow.end_run()
      self._run = None

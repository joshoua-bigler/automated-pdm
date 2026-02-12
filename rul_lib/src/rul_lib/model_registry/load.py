import mlflow.pytorch
import json
import torch.nn as nn
from mlflow.tracking import MlflowClient

_model_cache = {}


def load_model(model_name: str, version: str, tracking_uri: str) -> nn.Module:
  ''' Load a model from MLflow tracking server based on the provided ModelInput.'''
  cache_key = (model_name, version)
  if cache_key in _model_cache:
    return _model_cache[cache_key]
  mlflow.set_tracking_uri(tracking_uri)
  model_uri = f'models:/{model_name}/{version}'
  model = mlflow.pytorch.load_model(model_uri=model_uri)
  client = MlflowClient(tracking_uri=tracking_uri)
  version = client.get_model_version(name=model_name, version=version)
  tags = version.tags
  meta = json.loads(tags['meta']) if 'meta' in tags else {}
  model.meta = meta
  _model_cache[cache_key] = model
  return model

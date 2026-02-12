from __future__ import annotations

import json
from langchain_core.tools import tool
from mlflow.tracking import MlflowClient
from pathlib import Path
from typing import Any


def _json_safe(x: Any):
  try:
    import numpy as np
  except Exception:
    np = None  # type: ignore
  if x is None or isinstance(x, (bool, int, float, str)):
    return x
  if isinstance(x, Path):
    return str(x)
  if np is not None and isinstance(x, (np.integer, np.floating)):
    return x.item()
  if isinstance(x, dict):
    return {str(k): _json_safe(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [_json_safe(v) for v in x]
  return str(x)


@tool
def summarize_mlflow_results(tracking_uri: str,
                             experiment_name: str,
                             version: str,
                             metric: str = 'val_obj',
                             mode: str = 'min',
                             top_k: int = 3) -> dict:
  ''' Summarize Ray Tune top trials from MLflow artifacts.

      Looks for the latest run in the given MLflow experiment having param `version == <version>`,
      then loads the artifact written by the training step at `artifacts/configs/topk_trials.json`.

      Returns
      -------
      dict
        {
          'status': 'ok'|'error',
          'experiment_id': str,
          'run_id': str,
          'metric': {'name': str, 'mode': 'min'|'max'},
          'best_score': float | None,
          'topk': [
            {'id': str, 'score': float | None, 'config': dict, 'source': 'mlflow'}
          ]
        }
  '''
  client = MlflowClient(tracking_uri=tracking_uri)
  exp = client.get_experiment_by_name(experiment_name)
  if exp is None:
    return {'status': 'error', 'message': f'experiment not found: {experiment_name}'}
  filt = f"params.version = '{version}'"
  runs = client.search_runs(experiment_ids=[exp.experiment_id],
                            filter_string=filt,
                            order_by=['attributes.start_time DESC'],
                            max_results=5)
  if not runs:
    return {'status': 'error', 'message': f'no runs with params.version={version} in experiment {experiment_name}'}
  run = runs[0]
  run_id = run.info.run_id
  rel_art = 'artifacts/configs/topk_trials.json'
  try:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
      local = client.download_artifacts(run_id, rel_art, tmp)
      data = json.loads(Path(local).read_text(encoding='utf-8'))
  except Exception as e:
    # Fallback for local file-based artifact URIs
    try:
      from urllib.parse import urlparse
      run_info = client.get_run(run_id).info
      art_uri = run_info.artifact_uri
      parsed = urlparse(art_uri)
      if parsed.scheme == 'file':
        base = Path(parsed.path)
        p = base / 'configs' / 'topk_trials.json'
        data = json.loads(p.read_text(encoding='utf-8'))
      else:
        raise RuntimeError(str(e))
    except Exception:
      return {
          'status': 'error',
          'message': f'failed to load artifact {rel_art} for run {run_id}: {e}',
          'experiment_id': exp.experiment_id,
          'run_id': run_id,
      }
  # data is a list of trials with keys: trial_path, best_metric, config, ...
  # Map to the common schema used by summarize_ray_results
  sorted_trials = sorted(data, key=lambda t: float(t.get('best_metric', float('inf'))), reverse=(mode == 'max'))
  top = sorted_trials[:top_k]
  topk = [{
      'id': Path(t.get('trial_path', '')).name or str(i),
      'score': _json_safe(t.get('best_metric')),
      'config': _json_safe(t.get('config') or {}),
      'source': 'mlflow',
  } for i, t in enumerate(top)]
  best_score = topk[0]['score'] if topk else None
  return {
      'status': 'ok',
      'experiment_id': exp.experiment_id,
      'run_id': run_id,
      'metric': {
          'name': metric,
          'mode': mode
      },
      'best_score': best_score,
      'topk': topk,
  }

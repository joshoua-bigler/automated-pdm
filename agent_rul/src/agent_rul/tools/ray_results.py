from ray.tune import ExperimentAnalysis
from pathlib import Path
from langchain_core.tools import tool
# local
from agent_rul.gls import RAY_RESULTS_DIR


@tool
def summarize_ray_results(dataset_name: str,
                          version: str,
                          fd_number: str | None = None,
                          metric: str = 'val_obj',
                          mode: str = 'min',
                          top_k: int = 3) -> dict:
  ''' Summarize a Ray Tune experiment directory for a given dataset and version.

      The function collects and ranks all trials found under the corresponding Ray results
      directory (e.g., `RAY_RESULTS_DIR / f'agent_{dataset_name}_v{version}'`). It returns
      a compact overview of the top-k trials based on the specified metric.

      Parameters
      ----------
      dataset_name :
        Dataset identifier, e.g., 'cmapps' or 'femto'.
      version : 
        Experiment version string, e.g., '0.6.7'.
      fd_number : 
        Optional dataset sub-identifier (e.g., for C-MAPSS 'fd001'). If given, it is
        appended to the dataset name as '<dataset>_fd<fd_number>'.
      metric : 
        Metric key used to rank trials (default: 'val_obj').
      mode : 
        Optimization mode: 'min' for minimization or 'max' for maximization (default: 'min').
      top_k : 
        Number of top trials to include in the summary (default: 3).

      Returns
      -------
      dict
        A compact summary containing the top-k trials and key statistics:
        {
          'status': 'ok' | 'error',
          'path': str,                   # absolute path to experiment directory
          'metric': {'name': str, 'mode': str},
          'total_trials': int,
          'best_score': float | None,
          'topk': [                      # ranked list of top-k trials
            {
              'id': str,
              'local_path': str,
              'score': float | None,
              'config': dict
            },
            ...
          ]
        }

      Notes
      -----
      - The path is inferred as `RAY_RESULTS_DIR / f'agent_{dataset_name}_v{version}'`.
      - Trials without the specified metric in their results are ignored.
      - Only minimal information is included in the top-k entries for compactness.
  '''
  dataset_name = dataset_name if fd_number is None else f'{dataset_name}_fd{fd_number}'
  base = Path(RAY_RESULTS_DIR) / f'agent_{dataset_name}_v{version}'  # yapf: disable
  if not base.exists():
    return {'status': 'error', 'message': f'Experiment directory not found: {str(base)}'}

  ea = ExperimentAnalysis(str(base), default_metric=metric, default_mode=mode)
  trials = sorted([t for t in ea.trials if metric in (t.last_result or {})],
                  key=lambda t: t.last_result[metric],
                  reverse=(mode == 'max'))[:top_k]

  summary = [{
      'id': t.trial_id,
      'score': t.last_result.get(metric),
      'config': t.config,
  } for t in trials]

  return {
      'status': 'ok',
      'path': str(base),
      'metric': {
          'name': metric,
          'mode': mode
      },
      'total_trials': len(ea.trials),
      'best_score': summary[0]['score'] if summary else None,
      'topk': summary
  }

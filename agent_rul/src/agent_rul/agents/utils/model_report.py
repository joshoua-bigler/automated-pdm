import os
import json
from langchain_core.messages import ToolMessage


def _extract_creation_payloads(messages: list) -> list[dict]:
  ''' Scan tool messages and extract one or more save/register results.

      Supports either a combined 'save_and_register_model' tool or a sequence of
      'save_model_file' followed by 'register_model_adapter'.
  '''
  results: list[dict] = []
  pending: dict | None = None
  for m in messages:
    if not isinstance(m, ToolMessage):
      continue
    if m.name == 'save_and_register_model':
      try:
        d = json.loads(m.content)
      except Exception:
        d = None
      if isinstance(d, dict):
        out = {
            'status': d.get('status', 'unknown'),
            'model_path': d.get('model_path'),
            'adapter_path': d.get('adapter_path'),
            'model_module': d.get('model_module'),
            'adapter_module': d.get('adapter_module'),
            'registry_name': d.get('registry_name'),
            'registered': d.get('registered'),
            'sha256': d.get('sha256'),
        }
        results.append(out)
      continue
    # Split tools path
    if m.name == 'save_model_file':
      try:
        d = json.loads(m.content)
      except Exception:
        d = None
      if isinstance(d, dict):
        pending = {
            'status': d.get('status', 'unknown'),
            'model_path': d.get('model_path'),
            'model_module': d.get('model_module'),
            'sha256': d.get('sha256'),
            'adapter_path': None,
            'adapter_module': None,
            'registry_name': None,
            'registered': None,
        }
      continue
    if m.name == 'register_model_adapter' and pending is not None:
      try:
        d = json.loads(m.content)
      except Exception:
        d = None
      if isinstance(d, dict):
        pending['adapter_path'] = d.get('adapter_path')
        pending['adapter_module'] = d.get('adapter_module')
        pending['registry_name'] = d.get('registry_name')
        pending['registered'] = d.get('registered')
        # finalize one record
        results.append(pending)
        pending = None
  # If a save happened without a register, still record it
  if pending is not None:
    results.append(pending)
  return results


def summarize_model_report(report: dict) -> dict:
  ''' Compact, machine-readable summary for manager routing. '''
  created = report.get('created', {}) or {}
  arts = report.get('artifacts', {}) or {}
  ds = report.get('dataset', {}) or {}
  return {
      'agent': 'model',
      'kind': 'model_report_summary',
      'status': report.get('status'),
      'headline': report.get('headline'),
      'dataset': ds,
      'registry_name': created.get('registry_name'),
      'registry_names': [c.get('registry_name') for c in (report.get('created_list') or []) if c.get('registry_name')],
      'model_module': created.get('model_module'),
      'adapter_module': created.get('adapter_module'),
      'sha256': arts.get('sha256'),
      'artifact_bytes': arts.get('bytes'),
      'num_models': len(report.get('created_list') or ([] if not created else [created])),
      'next': {
          'phase': 'need_training',
          'target_config': report.get('target_path') or '',
      },
  }


def _file_bytes(path: str) -> int | None:
  ''' Return file size in bytes if path exists, else None. '''
  try:
    return int(os.path.getsize(path))
  except Exception:
    return None


def _fmt_bytes(n: int | None) -> str:
  ''' Human-readable byte size. '''
  if not isinstance(n, int):
    return 'n/a'
  for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
    if n < 1024:
      return f'{n:.0f} {unit}'
    n = n / 1024.0
  return f'{n:.1f} PB'


def _build_creation_report(answer: dict) -> dict:
  ''' Build a rich JSON report from sub-graph answer (no messages side effects).'''
  msgs = answer.get('messages', [])
  ctx = answer.get('context', {}) or {}
  tracking = ctx.get('tracking', {}) or {}
  dataset = tracking.get('dataset', {}) or {}
  payloads = _extract_creation_payloads(msgs)
  payload = payloads[-1] if payloads else {
      'status': 'unknown',
      'model_path': None,
      'adapter_path': None,
      'model_module': None,
      'adapter_module': None,
      'registry_name': None,
      'registered': None,
      'sha256': None,
  }
  model_bytes = _file_bytes(payload['model_path']) if payload['model_path'] else None
  adapter_bytes = _file_bytes(payload['adapter_path']) if payload['adapter_path'] else None
  total_bytes = (model_bytes or 0) + (adapter_bytes or 0) if (model_bytes or adapter_bytes) else None
  status = 'ok' if payload['registered'] else 'warn'
  headline = 'model generated and registered' if payload['registered'] else 'model generated (not registered)'
  report = {
      'kind': 'model_creation',
      'status': status,
      'headline': headline,
      'dataset': dataset,
      'created': {
          'registry_name': payload['registry_name'],
          'model_module': payload['model_module'],
          'adapter_module': payload['adapter_module'],
      },
      'created_list': [{
          'registry_name': p.get('registry_name'),
          'model_module': p.get('model_module'),
          'adapter_module': p.get('adapter_module'),
      } for p in payloads],
      'artifacts': {
          'model_path': payload['model_path'],
          'adapter_path': payload['adapter_path'],
          'sha256': payload['sha256'],
          'bytes': total_bytes,
      },
      'artifacts_list': [{
          'model_path': p.get('model_path'),
          'adapter_path': p.get('adapter_path'),
          'sha256': p.get('sha256'),
      } for p in payloads],
      'registry': {
          'registered': bool(payload['registered']),
      },
      'notes': [],
      'issues': [],
      'next_actions': [],
  }
  target_path = ctx.get('target_path') or tracking.get('target_path')
  if target_path:
    report['target_path'] = target_path
    report['next_actions'].append(f'train the model using pipeline config at {target_path}')
  else:
    report['next_actions'].append('train the model using your RUL pipeline (set trainer, criterion, and loaders)')
  report['notes'].append('checksum allows integrity verification of the generated source artifacts')
  if total_bytes:
    report['notes'].append(f'approx artifact size: {_fmt_bytes(total_bytes)}')
  return report


def attach_model_creation_report(answer: dict, state: dict) -> dict:
  ''' Build and return the JSON report only.
      No messages are appended here. The caller decides how to surface it.
  '''
  return _build_creation_report(answer)

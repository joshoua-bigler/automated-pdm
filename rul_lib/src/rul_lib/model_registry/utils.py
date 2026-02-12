def make_model_meta(full_meta: dict) -> dict:
  ''' Create model metadata for logging. '''
  norm = full_meta.get('normalization', {})
  return {
      'schema_version': 1,
      'y_target': full_meta.get('y_target', 'rul'),
      'y_scaler': full_meta.get('y_scaler'),
      'normalization': {
          'method': norm.get('method'),
          'mode': norm.get('mode'),
          'columns': norm.get('columns'),
          'group_by': norm.get('group_by')
      }
  }

def flatten_conditional(block: dict) -> dict:
  ''' Flatten conditional blocks in the config.

      Supports both styles of conditional blocks:
      - {'model': 'x', 'models': {'x': {...}}}
      - {'model': 'x', 'params': {...}}

      Parameters
      ----------
      block:
        Input config block
      
      Returns
      -------
      Flattened config block
  '''
  if not isinstance(block, dict) or 'model' not in block:
    return block
  name = block['model']
  if 'models' in block:
    params = block.get('models', {}).get(name, {})
  else:
    params = block.get('params', {})
  out = {'model': name}
  if isinstance(params, dict):
    out.update(params)
  return out

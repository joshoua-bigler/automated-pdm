import pandas as pd
from typing import Any
from agent_rul.data.data_schema import make_generic_schema, SchemaConfig


def _first_frame(data: dict[str, Any]) -> pd.DataFrame:
  df = data.get('x_train')
  if isinstance(df, pd.DataFrame):
    return df
  for value in data.values():
    if isinstance(value, pd.DataFrame):
      return value
  raise ValueError('data does not contain a pandas.DataFrame')


def analyze_dataset(tracking: dict[str, Any],
                    data: dict[str, Any],
                    schema_config: SchemaConfig | None = None) -> dict[str, Any]:
  ''' Analyze dataset and return schema and known values.

      Parameters
      ----------
      tracking:
        Tracking dictionary with dataset configuration
      data:
        Data dictionary containing at least one pandas.DataFrame
      schema_config:
        SchemaConfig to customize schema inference
  '''
  meta = data.get('meta', {})
  sampling_rate = meta.get('sampling_rate_hz')
  downsampling = meta.get('downsampling', {})
  factor = downsampling.get('factor') if isinstance(downsampling, dict) else None
  df = _first_frame(data)
  schema = make_generic_schema(df=df, cfg=schema_config)
  known_values = {
      'n_rows': schema.get('n_rows'),
      'n_units': schema.get('n_units'),
      'sensor_count': len(schema.get('sensor_cols', [])),
      'time_monotonic_by_unit': schema.get('time_monotonic_by_unit')
  }
  schema.setdefault('sampling', {})['fs_hz'] = sampling_rate
  schema['sampling']['downsample_factor'] = factor
  if sampling_rate and factor:
    schema['sampling']['original_fs_hz'] = sampling_rate * factor
  return {'schema': schema, 'known_values': known_values}

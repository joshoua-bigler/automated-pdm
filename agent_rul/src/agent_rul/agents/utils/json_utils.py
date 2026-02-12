import json
import numpy as np


def safe_json(obj):
  try:
    return json.dumps(obj, ensure_ascii=False, indent=2)
  except TypeError:

    def default(o):
      if isinstance(o, (np.integer, np.floating)):
        return o.item()
      return str(o)

    return json.dumps(obj, ensure_ascii=False, indent=2, default=default)

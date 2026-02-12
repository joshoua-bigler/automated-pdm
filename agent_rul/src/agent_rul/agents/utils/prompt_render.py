import json


def render_template(s: str, values: dict) -> str:
  out = s
  for k, v in values.items():
    repl = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    out = out.replace(k, repl)
  return out

import os, json, yaml, re
from importlib import resources as res
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langfuse import get_client, observe
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from typing import Literal
# local
from rul_lib.gls.gls import logger
from agent_rul.tools.model_inspection import list_available_models, describe_model, list_available_schedulers, describe_models
from agent_rul.tools.yml_handler import load_yaml_template, validate_yaml, write_yml
from agent_rul.agents.utils.message import add_messages, prepare_messages_for_model
from agent_rul.agents.shared import State
from agent_rul.agents.utils.prompt_render import render_template
from agent_rul.agents.shared import State
from agent_rul.agents.utils.json_utils import safe_json

TOOLS = [validate_yaml, describe_model, list_available_models, list_available_schedulers, load_yaml_template, write_yml, describe_models] # yapf: disable

SYSTEM = res.files('agent_rul.prompts').joinpath('config_agent.system.md').read_text(encoding='utf-8')
TASK_TEMPLATE = res.files('agent_rul.prompts').joinpath('config_agent.task.md').read_text(encoding='utf-8')

langfuse = get_client()
langfuse_handler = LangfuseCallbackHandler()

llm = ChatOpenAI(model='gpt-5', temperature=0, callbacks=[langfuse_handler], api_key=os.getenv('OPENAI_API_KEY'))
llm_with_tools = llm.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)


def _strip_inline_comments(s: str) -> str:
  return re.sub(r'(?m)\s+#.*$', '', s)


def get_config(x: dict | str, strip_comments=True, debug=False):
  ''' Extract config text and object from input dict or string. '''

  def _try_yaml(s: str):
    try:
      return yaml.safe_load(s)
    except Exception as e1:
      try:
        return yaml.load(s, Loader=yaml.FullLoader)
      except Exception as e2:
        if debug:
          print(f'yaml.safe_load error: {e1}')
          print(f'yaml.FullLoader error: {e2}')
        return None

  if isinstance(x, dict):
    if isinstance(x.get('candidate_text'), str):
      text = x['candidate_text']
      obj = _try_yaml(text) if not strip_comments else _try_yaml(_strip_inline_comments(text))
      return text, obj
    text = yaml.safe_dump(x, sort_keys=False, allow_unicode=True)
    return text, x
  if isinstance(x, str):
    text = x
    obj = _try_yaml(text)
    if obj is None and strip_comments:
      obj = _try_yaml(_strip_inline_comments(text))
    if obj is not None:
      return text, obj
    try:
      j = json.loads(text)
      return get_config(j, strip_comments=strip_comments, debug=debug)
    except Exception:
      return text, None
  raise TypeError('candidate_config must be dict or str')


def route(state: dict) -> str:
  msgs = state.get('messages', [])
  if not msgs:
    return 'end'
  last = msgs[-1]
  if isinstance(last, AIMessage) and getattr(last, 'tool_calls', None):
    return 'tools'
  return 'end'


def _ensure_system_first(msgs: list, system_text: str) -> list:
  kept = [m for m in msgs if not isinstance(m, SystemMessage)]
  return [SystemMessage(content=system_text)] + kept


def config_context(state: dict) -> list:
  ctx = state.get('context') or {}
  system_text = state.get('system_prompt')
  if not system_text:
    system_text = SYSTEM
  msgs = state.get('messages', [])
  return _ensure_system_first(msgs, system_text)


def config_agent(state: dict) -> dict:
  # msgs = config_context(state)
  prepared = prepare_messages_for_model(state['messages'])
  resp = llm_with_tools.invoke(prepared, config={'callbacks': [langfuse_handler], 'configurable': {}})
  return add_messages(state, [resp])


def tools(state: dict) -> dict:
  result = tool_node.invoke(state)
  msgs = result.get('messages', [])
  merged = dict(state)
  if msgs and isinstance(msgs[0], ToolMessage):
    merged['messages'] = state['messages'] + msgs
  else:
    merged['messages'] = state.get('messages', [])
  for k, v in result.items():
    if k == 'messages':
      continue
    merged[k] = v
  if 'system_prompt' in state:
    merged['system_prompt'] = state['system_prompt']
  if 'context' in state:
    merged['context'] = state['context']
  return merged


graph = StateGraph(dict)
graph.add_node('config_agent', config_agent)
graph.add_node('tools', tools)
graph.set_entry_point('config_agent')
graph.add_conditional_edges('config_agent', route, {'tools': 'tools', 'end': END})
graph.add_edge('tools', 'config_agent')
app = graph.compile()


@observe()
def config(state: State) -> Command[Literal['manager']]:
  ''' Config agent to generate RUL pipeline configuration based on dataset available models and tracking info. '''
  logger.info('running config agent...')
  ctx = state.get('context') or {}
  tracking = ctx.get('tracking') or {}
  tpl = ctx.get('config_template_path')
  p_ctx = ctx.get('prompt_agent', {})
  art = state.get('artifacts', {})
  replacements = {
      '{prompt_agent_context}': safe_json(p_ctx.get('config_prompt', {})),
      '{prompt_agent_hints}': safe_json(p_ctx.get('hints', {})),
      '{dataset_schema}': safe_json(ctx.get('dataset_schema', {})),
      '{model_agent_report}': safe_json(art.get('model_creation_report', {})),
  }
  system_prompt = render_template(s=SYSTEM, values=replacements)
  user_task = TASK_TEMPLATE.format(dataset_name=tracking.get('dataset', {}).get('name'),
                                   version=tracking.get('version'),
                                   config_template_path=str(tpl),
                                   config_file_name=ctx.get('config_file_name', 'unk_0.0.0.yml'),
                                   standalone=str(tracking.get('standalone', False)))
  sub_state = {
      'messages': [SystemMessage(content=system_prompt),
                   HumanMessage(content=user_task)],
      'context': ctx,
      'system_prompt': system_prompt,
  }
  final_state = app.invoke(sub_state, config={'callbacks': [langfuse_handler], 'recursion_limit': 100})
  msg = final_state.get('messages', [])[-1]
  if not msg:
    raise RuntimeError('no AIMessage produced')
  try:
    out = json.loads(msg.content.strip())
    cfg = out['scratch']['config']
  except Exception as e:
    raise RuntimeError(f'invalid config output: {msg.content}') from e
  cfg_text, cfg_obj = get_config(cfg)
  art = {
      'config_text': cfg_text,
      'config': cfg_obj,
      'config_meta': {
          'template': str(tpl),
          'dataset_name': (tracking.get('dataset') or {}).get('name'),
          'version': tracking.get('version'),
      },
  }
  # yapf: disable
  return Command(
    goto='manager',
    update={
      **state,
      'messages': state.get('messages', []) + [AIMessage(content='{"event":"config candidate_ready"}')],
      'artifacts': {**state.get('artifacts', {}), **art},
      'context': {**state.get('context', {}), **out.get('context', {})},
      'scratch': {**state.get('scratch', {}), 'phase': 'candidate_ready'},
      'phase': 'done'
    },
  )
  # yapf: enable

import os
import json
from importlib import resources as res
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langfuse import observe
from rul_lib.gls.gls import logger
from typing import Literal
# local
from agent_rul.agents.shared import State
from agent_rul.agents.utils.message import add_messages, prepare_messages_for_model
from agent_rul.agents.utils.json_utils import safe_json
from agent_rul.agents.utils.prompt_render import render_template
from agent_rul.tools.user_info import list_user_info_markdown, load_user_instruction
from agent_rul.tools.yml_handler import inspect_yaml_template
from agent_rul.tools.model_inspection import list_available_models
from agent_rul.tools.mlflow_results import summarize_mlflow_results

SYSTEM_PROMPT = res.files('agent_rul.prompts').joinpath('prompt_agent.system.md').read_text(encoding='utf-8')

langfuse_handler = LangfuseCallbackHandler()
llm = ChatOpenAI(model='gpt-5', temperature=0, callbacks=[langfuse_handler], api_key=os.getenv('OPENAI_API_KEY'))
TOOLS = [
    list_user_info_markdown, load_user_instruction, summarize_mlflow_results, list_available_models,
    inspect_yaml_template
]
llm_with_tools = llm.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)


def prompt_route(state: State) -> str:
  msgs = state.get('messages', [])
  if not msgs:
    return 'end'
  last = msgs[-1]
  if isinstance(last, AIMessage) and getattr(last, 'tool_calls', None):
    return 'prompt_tools'
  return 'end'


def prompt_node(state: State) -> dict:
  prepared = prepare_messages_for_model(state['messages'])
  resp = llm_with_tools.invoke(prepared, config={'callbacks': [langfuse_handler], 'configurable': {}})
  return add_messages(state, [resp])


def tools_node(state: State) -> dict:
  # Run tool, then merge results back into full state without clobbering keys.
  result = tool_node.invoke(state)
  msgs = result.get('messages', [])
  merged = dict(state)
  # Always append tool messages (reducers handle messages too)
  if msgs and isinstance(msgs[0], ToolMessage):
    merged['messages'] = state.get('messages', []) + msgs
  # Merge dict-like fields instead of overwriting
  for k, v in result.items():
    if k == 'messages':
      continue
    if isinstance(v, dict) and isinstance(merged.get(k), dict):
      merged[k] = {**merged[k], **v}
    else:
      merged[k] = v
  return merged


graph = StateGraph(State)
graph.add_node('prompt', prompt_node)
graph.add_node('prompt_tools', tools_node)
graph.set_entry_point('prompt')
graph.add_conditional_edges('prompt', prompt_route, {'prompt_tools': 'prompt_tools', 'end': END})
graph.add_edge('prompt_tools', 'prompt')
app = graph.compile()


@observe()
def prompt(state: State) -> Command[Literal['manager']]:
  ''' Prompt agent with tools and structured handoff via context.

      - Prepends a system prompt for prominence.
      - Allows tool calls (user info browsing, Ray summary).
      - If the model returns strict JSON with keys: messages, scratch, hints, phase,
        store outputs under context['prompt_agent'] for downstream agents.
  '''
  logger.info('running prompt agent...')
  ctx = state.get('context', {})
  tracking = ctx.get('tracking', {}) or {}
  dataset = tracking.get('dataset', {})
  ds_schema = ctx.get('dataset_schema', {})
  replacements = {
      '{dataset}': safe_json(dataset),
      '{dataset_name}': str(tracking.get('dataset', {}).get('name')),
      '{version}': str(tracking.get('version')),
      '{baseline_result}': safe_json(tracking.get('baseline_result', {})),
      '{tracking_uri}': str(tracking.get('mlflow', {}).get('tracking_uri', '')),
      '{experiment_name}': str(tracking.get('experiment_name', '')),
      '{dataset_schema}': safe_json(ds_schema),
      '{snr}': safe_json(ctx.get('dataset_snr')),
      '{standalone}': str(tracking.get('standalone', False)),
  }
  system_prompt = render_template(s=SYSTEM_PROMPT, values=replacements)
  sub_state = {'messages': [SystemMessage(content=system_prompt)], 'context': ctx}
  final_state = app.invoke(sub_state, config={'callbacks': [langfuse_handler], 'recursion_limit': 100})
  msgs = final_state.get('messages', [])
  reply = msgs[-1] if msgs else AIMessage(content='')
  # Attempt to parse structured JSON
  new_messages = []
  new_context = dict(ctx)
  new_phase = None
  try:
    data = json.loads(str(getattr(reply, 'content', '')).strip())
    if isinstance(data, dict):
      for m in data.get('messages', []):
        role = (m or {}).get('role')
        content = (m or {}).get('content', '')
        if role == 'ai':
          new_messages.append(AIMessage(content=content, metadata={'agent': 'prompt', 'kind': 'event'}))
        elif role == 'human':
          new_messages.append(HumanMessage(content=content))
        elif role == 'system':
          new_messages.append(SystemMessage(content=content))
      scratch = data.get('scratch', {}) or {}
      hints = data.get('hints', {}) or {}
      pa = dict(new_context.get('prompt_agent', {}))
      if 'config_prompt' in scratch:
        pa['config_prompt'] = scratch['config_prompt']
      if 'model_prompt' in scratch:
        pa['model_prompt'] = scratch['model_prompt']
      if hints:
        pa['hints'] = hints
      if pa:
        new_context['prompt_agent'] = pa
      new_phase = data.get('phase')
  except Exception:
    pass
  updated_msgs = state.get('messages', []) + ([reply] if reply else []) + new_messages
  update = {'messages': updated_msgs, 'context': new_context}
  if new_phase:
    update['phase'] = new_phase
  return Command(goto='manager', update=update)

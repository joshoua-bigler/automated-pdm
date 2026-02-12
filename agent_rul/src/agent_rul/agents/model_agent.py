import os
from langchain_core.messages import AIMessage
from importlib import resources as res
from langgraph.types import Command
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, ToolMessage
from langfuse import observe
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langgraph.graph import StateGraph, END
from rul_lib.gls.gls import logger
from typing import Literal
# local
from agent_rul.agents.shared import State
from agent_rul.agents.utils.json_utils import safe_json
from agent_rul.tools.model_inspection import list_available_models, describe_model, list_available_schedulers, describe_models
from agent_rul.tools.save_model import save_model_file
from agent_rul.tools.register_model import (register_model_adapter_raw, register_seq2val_adapter_raw)
from agent_rul.agents.utils.message import add_messages, prepare_messages_for_model
from agent_rul.agents.utils.model_report import attach_model_creation_report, summarize_model_report
from agent_rul.agents.utils.prompt_render import render_template

TOOLS = [
    list_available_models,
    describe_model,
    describe_models,
    list_available_schedulers,
    save_model_file,
    register_model_adapter_raw,
    register_seq2val_adapter_raw,
]

SYSTEM_PROMPT = res.files('agent_rul.prompts').joinpath('model_agent.system.md').read_text(encoding='utf-8')

langfuse_handler = LangfuseCallbackHandler()

llm = ChatOpenAI(model='gpt-5', temperature=0, callbacks=[langfuse_handler], api_key=os.getenv('OPENAI_API_KEY'))
llm_with_tools = llm.bind_tools(TOOLS)
tool_node = ToolNode(TOOLS)


def route(state: State) -> str:
  msgs = state.get('messages', [])
  if not msgs:
    return 'end'
  last = msgs[-1]
  if isinstance(last, AIMessage) and getattr(last, 'tool_calls', None):
    return 'model_tools'
  return 'end'


def model_agent(state: State) -> dict:
  prepared = prepare_messages_for_model(state['messages'])
  resp = llm_with_tools.invoke(prepared, config={'callbacks': [langfuse_handler], 'configurable': {}})
  return add_messages(state, [resp])


def tools_node(state: State) -> dict:
  # Run tool and merge outputs without dropping existing state fields.
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
  return merged


graph = StateGraph(State)
graph.add_node('model_agent', model_agent)
graph.add_node('model_tools', tools_node)
graph.set_entry_point('model_agent')
graph.add_conditional_edges('model_agent', route, {'model_tools': 'model_tools', 'end': END})
graph.add_edge('model_tools', 'model_agent')
app = graph.compile()


@observe()
def model(state: State) -> Command[Literal['manager']]:
  logger.info('running model agent...')
  ctx = state.get('context', {})
  tracking = ctx.get('tracking', {}) or {}
  dataset = tracking.get('dataset', {})
  ds_schema = ctx.get('dataset_schema', {})
  p_ctx = ctx.get('prompt_agent', {})
  replacements = {
      '{dataset}': safe_json(dataset),
      '{dataset_name}': tracking.get('dataset', {}).get('name'),
      '{version}': str(tracking.get('version')),
      '{prompt_agent_context}': safe_json(p_ctx.get('model_prompt', {})),
      '{prompt_agent_hints}': safe_json(p_ctx.get('hints', {})),
      '{dataset_schema}': safe_json(p_ctx.get('dataset_schema', {})),
      '{standalone}': str(tracking.get('standalone', False)),
  }
  system_prompt = render_template(s=SYSTEM_PROMPT, values=replacements)
  sub_state = {'messages': [SystemMessage(content=system_prompt)], 'context': ctx, 'tracking': ctx.get('tracking', {})}
  answer = app.invoke(sub_state, config={'callbacks': [langfuse_handler], 'recursion_limit': 100})
  report = attach_model_creation_report(answer=answer, state=state)
  summary = summarize_model_report(report=report)
  summary_msg = AIMessage(content=safe_json(summary), metadata={'agent': 'model', 'kind': 'model_report_summary'})
  # yapf: disable
  return Command(
    goto='manager',
    update={
      **state,
      'messages': state.get('messages', []) + [summary_msg],
      'artifacts': {**state.get('artifacts', {}), 'model_creation_report': report},
      'phase': 'need_training',
      'status': 'ok',
      'context': ctx,
    },
  )
  # yapf: enable

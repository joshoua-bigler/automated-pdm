import os
import json
import yaml
from itertools import count
from importlib import resources as res
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command
from langfuse import get_client, observe
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from pydantic import BaseModel, Field
from pathlib import Path
from rul_lib.gls.gls import logger
from rul_lib.data.data_loader import load_config
from typing import Literal, get_args
# local
from agent_rul.tools.yml_handler import load_yaml_template, validate_yaml, write_yml
from agent_rul.data.data_inspection import analyze_dataset
from agent_rul.data.data_schema import make_generic_schema
from agent_rul.agents.config_agent import config
from agent_rul.agents.model_agent import model
from agent_rul.agents.prompt_agent import prompt
from agent_rul.agents.shared import State
from agent_rul.agents.utils.json_utils import safe_json

MANAGER_SYSTEM_PROMPT = res.files('agent_rul.prompts').joinpath('manager_agent.system.md').read_text(encoding='utf-8')
MANAGER_TASK_PROMPT = res.files('agent_rul.prompts').joinpath('manager_agent.task.md').read_text(encoding='utf-8')

langfuse = get_client()
langfuse_handler = LangfuseCallbackHandler()

TOOLS = [validate_yaml, write_yml, load_yaml_template]

llm = ChatOpenAI(model='gpt-5', temperature=0, callbacks=[langfuse_handler], api_key=os.getenv('OPENAI_API_KEY'))


class RouterOut(BaseModel):
  next_step: Literal['prompt', 'config', 'model', 'FINISH'] = Field(description='next worker to call')


def _router_sys(md: str) -> SystemMessage:
  routes = ', '.join(get_args(RouterOut.model_fields['next_step'].annotation))
  return SystemMessage(content=md.replace('{routes}', routes))


def _is_bulky(text: str) -> bool:
  s = text or ''
  return any(tok in s for tok in ['```', 'tool_calls', 'save_and_register_model', '# model creation summary'])


def _prune_tail(msgs: list, n: int = 4) -> list:
  tail = []
  for m in msgs[-n:]:
    c = str(getattr(m, 'content', ''))
    if _is_bulky(c):
      continue
    tail.append(m)
  return tail


def _last_model_summary(msgs: list) -> AIMessage | None:
  for m in reversed(msgs):
    if isinstance(m, AIMessage):
      meta = getattr(m, 'metadata', {}) or {}
      if meta.get('kind') == 'model_report_summary':
        return m
      try:
        d = json.loads(str(m.content))
        if isinstance(d, dict) and d.get('kind') == 'model_report_summary':
          return m
      except Exception:
        pass
  return None


def manager_context(state: State) -> list:
  ''' Construct context messages for the manager '''
  msgs = state.get('messages', [])
  ctx = [_router_sys(MANAGER_SYSTEM_PROMPT)]
  s = _last_model_summary(msgs)
  if s is not None:
    ctx.append(s)
  ctx.extend(_prune_tail(msgs, n=4))
  return ctx


_counter = count(0)


def determ_choice() -> str:
  i = next(_counter)
  if i == 0:
    return 'prompt'
  elif i == 1:
    return 'model'
  elif i == 2:
    return 'config'
  return 'FINISH'


def manager(state: State) -> Command[Literal['prompt', 'config', 'model', '__end__']]:
  logger.info('running manager agent...')
  phase = state.get('phase', 'model')
  router_msgs = manager_context(state)
  determ = True
  if determ:
    choice = determ_choice()
  else:
    try:
      out = llm.with_structured_output(RouterOut).invoke(router_msgs)
      choice = out.next_step
      logger.info(f'manager routing to: {choice}')
    except Exception:
      choice = determ_choice()
      logger.info(f'manager router fallback -> {choice}')
  if choice == 'FINISH':
    return Command(goto=END, update={'messages': state.get('messages', []) + [AIMessage(content='manager: finished')]})
  note = AIMessage(content=f'manager: routing to {choice}')
  return Command(goto=choice, update={'messages': state.get('messages', []) + [note]})


graph = StateGraph(State)
graph.add_node('manager', manager)
graph.add_node('config', config)
graph.add_node('model', model)
graph.add_node('prompt', prompt)
graph.add_edge(START, 'manager')
app = graph.compile()


def run_manager(tracking: dict, data: dict, config_template: str | Path, use_cache: bool, out_dir: str | Path) -> dict: # yapf: disable
  data = data.copy()
  if use_cache:
    ds_name = tracking.get('dataset', {}).get('name') if isinstance(tracking, dict) else None
    version = tracking.get('version') if isinstance(tracking, dict) else None
    if ds_name and version:
      cached = Path(out_dir) / f'{ds_name}_{version}.yml'
      if cached.exists():
        return {'status': 'cached', 'path': str(cached), 'config': load_config(config_path=cached)}
  return _run_manager(tracking=tracking, data=data, config_template=config_template)


@observe()
def _run_manager(tracking: dict, data: dict, config_template: str | Path) -> dict:
  ds = tracking['dataset']['name']
  ver = tracking['version']
  dataset = analyze_dataset(tracking=tracking, data=data)
  config_file_name = f'{ds}_{ver}.yml'
  user_task = MANAGER_TASK_PROMPT.format(dataset_name=ds, version=ver, config_template=str(config_template), config_file_name=config_file_name) # yapf: disable
  messages = [HumanMessage(content=user_task)]
  # yapf: disable
  state = State(messages=messages,
                context={'tracking': tracking,
                         'dataset_snr': dataset.get('snr'),
                         'dataset_schema': dataset.get('schema'),
                         'config_template_path': str(config_template),
                         'config_file_name': config_file_name,
                         'standalone': tracking.get('standalone', False)},
                scratch={'manager': {},'config': {},'data': {}},
                phase='need_config',
                status=None,
                artifacts={})
  # yapf: enable
  final_state = app.invoke(
      state,
      config={
          'callbacks': [langfuse_handler],
          'run_name': tracking.get('experiment_name', f'agent_{ds}_v{ver}'),
          'configurable': {},
          'recursion_limit': 100
      },
  )
  artifacts = final_state.get('artifacts', {}) or {}
  config = artifacts.get('config')
  path = final_state.get('context', {}).get('path')
  if config is None:
    p = Path(path)
    if not p.exists():
      raise RuntimeError(f'written config not found: {p}')
    config = yaml.safe_load(p.read_text(encoding='utf-8'))
  return {'status': 'ok', 'path': path, 'config': config}

from langgraph.graph import MessagesState
from operator import or_ as dict_union
from pydantic import Field
from typing import Literal
from typing_extensions import Annotated


class State(MessagesState):
  phase: Literal['need_prompt', 'need_config', 'need_model', 'done', 'error'] = 'need_prompt'
  status: Literal['ok', 'failed'] | None = None
  artifacts: Annotated[dict, dict_union] = Field(default_factory=dict)
  scratch: Annotated[dict, dict_union] = Field(default_factory=dict)
  context: Annotated[dict, dict_union] = Field(default_factory=dict)

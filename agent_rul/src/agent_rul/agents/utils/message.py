from langchain_core.messages import ToolMessage


def add_messages(state: dict, new_msgs: list) -> dict:
  return {'messages': state['messages'] + new_msgs}


def prepare_messages_for_model(msgs: list) -> list:
  ''' ensure first message to model is not a ToolMessage (prevents OpenAI 400) '''
  i = 0
  while i < len(msgs) and isinstance(msgs[i], ToolMessage):
    i += 1
  prepared = msgs[i:] if i < len(msgs) else []
  if not prepared:
    raise RuntimeError('no non-tool messages available to send to the model')
  return prepared

import torch


def select_device(pref: str = None):
  ''' Select a torch.device based on preference string. '''
  if isinstance(pref, str):
    p = pref.lower()
    if p == 'cpu':
      return torch.device('cpu')
    if p == 'cuda':
      return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

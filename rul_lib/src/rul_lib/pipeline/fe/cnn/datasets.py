import numpy as np
import torch
from torch.utils.data import Dataset


class AEDataset(Dataset):
  ''' Dataset for autoencoder training. '''

  def __init__(self, x: np.ndarray):
    self.x = torch.as_tensor(x, dtype=torch.float32)

  def __len__(self) -> int:
    return self.x.shape[0]

  def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    xb = self.x[idx]  # (win, feats)
    return xb, xb  # target == input

  @property
  def window_size(self) -> int:
    return self.x.shape[1]

  @property
  def n_features(self) -> int:
    return self.x.shape[2]

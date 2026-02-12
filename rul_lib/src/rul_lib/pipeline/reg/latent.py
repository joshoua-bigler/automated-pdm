import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LatentDataset(Dataset):
  ''' PyTorch Dataset for latent representations. '''

  def __init__(self, z: np.ndarray, y: np.ndarray):
    z = z if z.ndim == 2 else z.reshape(len(z), -1)
    y = y.astype(np.float32).reshape(-1)
    self.z = torch.as_tensor(z, dtype=torch.float32)
    self.y = torch.as_tensor(y, dtype=torch.float32)

  def __len__(self) -> int:
    return self.z.shape[0]

  def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
    return self.z[i], self.y[i]


def build_loaders_from_latent(z_train: np.ndarray,
                              y_train: np.ndarray,
                              z_val: np.ndarray,
                              y_val: np.ndarray,
                              cfg: dict,
                              batch_size: int = 64,
                              num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
  ''' Build PyTorch DataLoaders from latent representations. '''
  ds_tr = LatentDataset(z_train, y_train)
  ds_va = LatentDataset(z_val, y_val)
  dl_tr = DataLoader(ds_tr,
                     batch_size=batch_size,
                     shuffle=True,
                     drop_last=False,
                     num_workers=num_workers,
                     pin_memory=torch.cuda.is_available())
  dl_va = DataLoader(ds_va,
                     batch_size=batch_size,
                     shuffle=False,
                     drop_last=False,
                     num_workers=num_workers,
                     pin_memory=torch.cuda.is_available())
  return dl_tr, dl_va

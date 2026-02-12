import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# local
from rul_lib.pipeline.reg.latent import build_loaders_from_latent


def eval_torch_model(model: nn.Module, dl_va: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
  ''' Evaluate a PyTorch model on validation data loader. '''
  model.eval()
  preds, ys = [], []
  with torch.no_grad():
    for xb, yb in dl_va:
      xb = xb.to(device)
      preds.append(model(xb).cpu())
      ys.append(yb.cpu())
  pred_val = torch.cat(preds).squeeze(-1).numpy()
  yv = torch.cat(ys).squeeze(-1).numpy()
  return pred_val, yv


def build_loaders(z_tr: np.ndarray, y_tr: np.ndarray, z_te: np.ndarray, y_te: np.ndarray, batch_size: int,
                  device: torch.device, cfg: dict) -> tuple[DataLoader, DataLoader]:
  ''' Build PyTorch data loaders from latent representations and targets. '''
  pin = device.type == 'cuda'
  nw = 2 if pin else 0
  return build_loaders_from_latent(z_train=z_tr,
                                   y_train=y_tr,
                                   z_val=z_te,
                                   y_val=y_te,
                                   batch_size=batch_size,
                                   num_workers=nw,
                                   cfg=cfg)

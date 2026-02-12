import numpy as np
import torch
import torch.nn as nn
# local
from rul_lib.pipeline.fe import models


def encode(encoder: nn.Module, data: np.ndarray) -> np.ndarray:
  ''' Encode data using the given encoder model. '''
  encoder.eval()
  dev = next(encoder.parameters()).device
  with torch.no_grad():
    x = torch.from_numpy(data).to(dev, dtype=torch.float32)
    return encoder(x).cpu().numpy()


def compute_latents(encoder: nn.Module, windowed: dict) -> tuple[np.ndarray, np.ndarray]:
  ''' Compute latent representations for train and test sets using the given encoder. '''
  z_tr = encode(encoder=encoder, data=windowed['x_train'])
  z_te = encode(encoder=encoder, data=windowed['x_test'])
  return z_tr, z_te


class WindowEncoder(nn.Module):
  ''' Encode windowed features using a convolutional autoencoder. '''

  def __init__(self, ae: models.ConvAe):
    super().__init__()
    self.encoder = ae.encoder
    self.to_latent = ae.to_latent

  def forward(self, x_win_feats: torch.Tensor) -> torch.Tensor:
    x_cf = x_win_feats.permute(0, 2, 1)  # (B, feats, win)
    h = self.encoder(x_cf).squeeze(-1)  # (B, hidden)
    return self.to_latent(h)  # (B, latent)


class TemporalWindowEncoder(nn.Module):

  def __init__(self, ae: nn.Module):
    super().__init__()
    self.ae = ae

  def forward(self, x_win_feats: torch.Tensor) -> torch.Tensor:
    # x_win_feats: (b, W, F)
    x_cf = x_win_feats.permute(0, 2, 1)  # (b, F, T)
    z_seq = self.ae.encode(x_cf)  # (b, latent_dim, T')
    z_vec = z_seq.mean(dim=2)  # (b, latent_dim)
    return z_vec

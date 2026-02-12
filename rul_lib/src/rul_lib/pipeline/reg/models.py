import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class MlpRegressor(nn.Module):
  ''' Simple feedforward MLP regressor. '''

  def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.0, **_: Any):
    super().__init__()
    self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(p=dropout),
                             nn.Linear(hidden, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x).squeeze(-1)


class MlpBlock(nn.Module):

  def __init__(self, d: int, p_drop: float = 0.1, p_stoch: float = 0.0, **_: Any):
    super().__init__()
    self.norm = nn.LayerNorm(d)
    self.fc1 = nn.Linear(d, 4 * d)
    self.fc2 = nn.Linear(4 * d, d)
    self.drop = nn.Dropout(p_drop)
    self.p_stoch = p_stoch

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.norm(x)
    h = F.gelu(self.fc1(h))
    h = self.drop(h)
    h = self.fc2(h)
    if self.training and self.p_stoch > 0.0:
      if torch.rand(()) < self.p_stoch:
        return x  # skip block
    return x + h


class AdvMlpRegressor(nn.Module):

  def __init__(self,
               in_dim: int,
               width: int = 128,
               depth: int = 3,
               dropout: float = 0.1,
               stoch_depth: float = 0.0,
               **_: Any):
    super().__init__()
    self.embed = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(dropout))
    blocks = []
    for i in range(depth):
      p_stoch = stoch_depth * float(i + 1) / depth
      blocks.append(MlpBlock(width, p_drop=dropout, p_stoch=p_stoch))
    self.blocks = nn.Sequential(*blocks)
    self.head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width // 2), nn.GELU(), nn.Dropout(dropout),
                              nn.Linear(width // 2, 1))
    self.reset_head_bias(0.0)

  def reset_head_bias(self, target_mean: float):
    with torch.no_grad():
      self.head[-1].bias.fill_(target_mean)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    h = self.embed(x)
    h = self.blocks(h)
    y = self.head(h).squeeze(-1)
    return y


class CausalConv1d(nn.Conv1d):

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, bias: bool = True):
    padding = (kernel_size - 1) * dilation
    super().__init__(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=kernel_size,
                     dilation=dilation,
                     padding=padding,
                     bias=bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = super().forward(x)
    cut = (self.kernel_size[0] - 1) * self.dilation[0]
    if cut > 0:
      out = out[..., :-cut]
    return out


class TCNBlock(nn.Module):

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
    super().__init__()
    self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
    self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
    self.dropout = nn.Dropout(dropout)
    self.act = nn.LeakyReLU()
    self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    out = self.conv1(x)
    out = self.act(out)
    out = self.dropout(out)
    out = self.conv2(out)
    out = self.act(out)
    out = self.dropout(out)
    if self.res_conv is not None:
      residual = self.res_conv(residual)
    out = out + residual
    out = self.act(out)
    return out


class TCNRegressor(nn.Module):

  def __init__(self,
               in_dim: int,
               channels: list[int] | tuple[int, ...] | None = None,
               kernel_size: int = 5,
               dropout: float = 0.1,
               hidden_mlp: int = 64,
               **_: Any):
    super().__init__()
    if channels is None:
      channels = [64, 32, 16]
    layers = []
    c_in = in_dim
    for i, c_out in enumerate(channels):
      dilation = 2**i
      layers.append(
          TCNBlock(in_channels=c_in, out_channels=c_out, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
      c_in = c_out
    self.tcn = nn.Sequential(*layers)
    self.pool = nn.AdaptiveAvgPool1d(1)
    self.fc1 = nn.Linear(c_in, hidden_mlp)
    self.fc2 = nn.Linear(hidden_mlp, 1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # accept flat (B, F) or sequence (B, T, F)
    if x.dim() == 2:
      x = x.unsqueeze(1)  # (B, 1, F)
    x = x.transpose(1, 2)  # (B, F, T)
    x = self.tcn(x)
    x = self.pool(x).squeeze(-1)
    x = F.leaky_relu(self.fc1(x))
    x = self.fc2(x).squeeze(-1)
    return x

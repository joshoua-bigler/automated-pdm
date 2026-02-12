import torch
import torch.nn as nn
from typing import Any


class CNNLSTMRegressor(nn.Module):

  def __init__(self, in_feats: int, conv_channels: int, kernel_size: int, lstm_hidden: int, lstm_layers: int,
               dropout: float, bidirectional: bool, **_: Any):
    super().__init__()
    # conv over time with sensors as channels: input (B, T, F) -> (B, F, T) for conv1d
    self.permute = True
    self.conv = nn.Sequential(
        nn.Conv1d(in_feats, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2), nn.ReLU(),
        nn.Conv1d(conv_channels, conv_channels, kernel_size=kernel_size, padding=kernel_size // 2), nn.ReLU(),
        nn.Dropout(dropout))
    # LSTM expects (B, T, C); after conv we have (B, C, T) -> transpose back
    self.lstm = nn.LSTM(input_size=conv_channels,
                        hidden_size=lstm_hidden,
                        num_layers=lstm_layers,
                        batch_first=True,
                        dropout=dropout if lstm_layers > 1 else 0.0,
                        bidirectional=bidirectional)
    out_dim = lstm_hidden * (2 if bidirectional else 1)
    self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 1))

  def forward(self, x):
    if self.permute:
      x = x.transpose(1, 2)  # (B, F, T)
    x = self.conv(x)  # (B, C, T)
    x = x.transpose(1, 2)  # (B, T, C)
    out, (hn, cn) = self.lstm(x)  # hn: (layers*dirs, B, H)
    h_last = hn[-1] if self.lstm.bidirectional is False else torch.cat((hn[-2], hn[-1]), dim=-1)
    return self.head(h_last)


class TCNBlock(nn.Module):

  def __init__(self, c_in: int, c_out: int, k: int, d: int, dropout: float, **_: Any):
    super().__init__()
    pad = (k - 1) * d
    self.net = nn.Sequential(
        nn.Conv1d(c_in, c_out, kernel_size=k, padding=pad, dilation=d),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Conv1d(c_out, c_out, kernel_size=k, padding=pad, dilation=d),
        nn.ReLU(),
        nn.Dropout(dropout),
    )
    self.down = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()

  def forward(self, x):
    out = self.net(x)
    cut = out.size(-1) - x.size(-1)
    if cut > 0:
      out = out[..., :-cut]
    return out + self.down(x)


class TCNRegressor(nn.Module):

  def __init__(self,
               in_feats: int,
               channels: tuple[int],
               kernel_size: int,
               dropout: float,
               dilations: list[int] | None = None,
               **_: Any):
    super().__init__()
    self.permute = True
    layers = []
    c_prev = in_feats
    # if no dilations passed, keep: d = 2**i
    for i, c in enumerate(channels):
      if dilations is not None:
        d = int(dilations[i])
      else:
        d = 2**i
      layers.append(TCNBlock(c_prev, c, kernel_size, d=d, dropout=dropout))
      c_prev = c

    self.tcn = nn.Sequential(*layers)
    self.head = nn.Sequential(
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(c_prev, 1),
    )

  def forward(self, x):
    if self.permute:
      x = x.transpose(1, 2)  # (B, F, T)
    x = self.tcn(x)
    return self.head(x)


class Chomp1d(nn.Module):

  def __init__(self, chomp: int):
    super().__init__()
    self.chomp = int(chomp)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.chomp <= 0:
      return x
    return x[..., :-self.chomp]


class AdvTCNBlock(nn.Module):

  def __init__(self, c_in: int, c_out: int, k: int, d: int, dropout: float, **_: Any):
    super().__init__()
    pad = (k - 1) * d
    conv1 = nn.utils.weight_norm(nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad))
    conv2 = nn.utils.weight_norm(nn.Conv1d(c_out, c_out, kernel_size=k, dilation=d, padding=pad))
    self.net = nn.Sequential(conv1, Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout), conv2, Chomp1d(pad), nn.ReLU(),
                             nn.Dropout(dropout))
    self.down = nn.Conv1d(c_in, c_out, kernel_size=1) if c_in != c_out else nn.Identity()
    self.out_act = nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.net(x)
    res = self.down(x)
    return self.out_act(out + res)


class AttnPool1d(nn.Module):

  def __init__(self, c: int):
    super().__init__()
    self.score = nn.Conv1d(c, 1, kernel_size=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (B, C, T)
    w = torch.softmax(self.score(x), dim=-1)  # (B, 1, T)
    return (x * w).sum(dim=-1)  # (B, C)


class AdvTCN(nn.Module):

  def __init__(self,
               in_feats: int,
               channels: tuple[int, ...],
               kernel_size: int,
               dropout: float,
               dilations: list[int] | None = None,
               stem_stride: int = 4,
               **_: Any):
    super().__init__()
    self.permute = True

    self.stem = (nn.Sequential(nn.Conv1d(in_feats, in_feats, kernel_size=9, stride=stem_stride, padding=4), nn.ReLU())
                 if stem_stride > 1 else nn.Identity())

    layers = []
    c_prev = in_feats

    if dilations is not None and len(dilations) != len(channels):
      raise ValueError(f'len(dilations)={len(dilations)} must match len(channels)={len(channels)}')

    for i, c in enumerate(channels):
      d = int(dilations[i]) if dilations is not None else 2**i
      layers.append(AdvTCNBlock(c_in=c_prev, c_out=c, k=kernel_size, d=d, dropout=dropout))
      c_prev = c

    self.tcn = nn.Sequential(*layers)
    self.pool = AttnPool1d(c_prev)
    self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_prev, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # input: (B, T, F)
    if self.permute:
      x = x.transpose(1, 2)  # (B, F, T)
    x = self.stem(x)
    x = self.tcn(x)
    x = self.pool(x)
    return self.head(x)


class GRUTCNRegressor(nn.Module):

  def __init__(self,
               in_feats: int,
               tcn_channels: tuple[int],
               tcn_kernel_size: int,
               tcn_dropout: float,
               gru_hidden: int,
               gru_layers: int,
               gru_dropout: float,
               bidirectional: bool = False,
               dilations: list[int] | None = None,
               **_: Any):
    super().__init__()
    self.permute = True
    layers = []
    c_prev = in_feats
    for i, c in enumerate(tcn_channels):
      if dilations:
        d = int(dilations[i]) if i < len(dilations) else int(dilations[-1])
      else:
        d = 2**i
      layers.append(TCNBlock(c_prev, c, tcn_kernel_size, d=d, dropout=tcn_dropout))
      c_prev = c
    self.tcn = nn.Sequential(*layers)
    self.rnn = nn.GRU(input_size=c_prev,
                      hidden_size=gru_hidden,
                      num_layers=gru_layers,
                      batch_first=True,
                      dropout=gru_dropout if gru_layers > 1 else 0.0,
                      bidirectional=bidirectional)
    out_dim = gru_hidden * (2 if bidirectional else 1)
    self.head = nn.Sequential(nn.Dropout(gru_dropout), nn.Linear(out_dim, 1))

  def forward(self, x):
    if self.permute:
      x = x.transpose(1, 2)
    x = self.tcn(x)
    x = x.transpose(1, 2)
    out, hn = self.rnn(x)
    if self.rnn.bidirectional:
      h_last = torch.cat((hn[-2], hn[-1]), dim=-1)
    else:
      h_last = hn[-1]
    return self.head(h_last)


class LSTMRegressor(nn.Module):

  def __init__(self,
               in_feats: int,
               hidden: int = 128,
               num_layers: int = 2,
               fc_hidden: int = 64,
               dropout: float = 0.2,
               bidirectional: bool = False,
               **_: Any):
    super().__init__()
    self.bidirectional = bidirectional
    self.hidden = hidden
    self.num_layers = num_layers
    self.num_dirs = 2 if bidirectional else 1
    # stacked LSTMs (batch_first, dropout between layers if num_layers > 1)
    self.rnn = nn.LSTM(input_size=in_feats,
                       hidden_size=hidden,
                       num_layers=num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.0,
                       bidirectional=bidirectional)
    # regression head: norm → relu → dropout → linear
    self.head = nn.Sequential(nn.LayerNorm(hidden * self.num_dirs), nn.ReLU(), nn.Dropout(dropout),
                              nn.Linear(hidden * self.num_dirs, fc_hidden), nn.ReLU(), nn.Dropout(dropout),
                              nn.Linear(fc_hidden, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch, seq_len, in_feats)
    out, _ = self.rnn(x)  # out: (batch, seq_len, hidden*num_dirs)
    h_last = out[:, -1, :]  # last timestep embedding
    return self.head(h_last)  # (batch, 1)


class GRURegressor(nn.Module):

  def __init__(self, in_feats: int, hidden: int, num_layers: int, dropout: float, **_: Any):
    super().__init__()
    self.rnn = nn.GRU(input_size=in_feats,
                      hidden_size=hidden,
                      num_layers=num_layers,
                      batch_first=True,
                      dropout=dropout if num_layers > 1 else 0.0)
    self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out, hn = self.rnn(x)
    h_last = hn[-1]
    return self.head(h_last)


class CNNRegressor(nn.Module):

  def __init__(self, in_feats: int, kernel_size: int, channels: int, dropout: float, **_: Any):
    super().__init__()
    self.permute = True
    self.net = nn.Sequential(
        nn.Conv1d(in_feats, channels, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.ReLU(),
        nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=kernel_size // 2),
        nn.ReLU(),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(channels, 1),
    )

  def forward(self, x):
    if self.permute:
      x = x.transpose(1, 2)  # (B, F, T)
    return self.net(x)


class ConvDW(nn.Module):

  def __init__(self, in_ch: int, out_ch: int, k: int, dilation: int = 1, p: float = 0.0, **_: Any):
    super().__init__()
    pad = (k // 2) * dilation
    self.net = nn.Sequential(
        nn.Conv1d(in_ch, in_ch, kernel_size=k, padding=pad, dilation=dilation, groups=in_ch, bias=False),
        nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False), nn.BatchNorm1d(out_ch), nn.GELU(), nn.Dropout(p))

  def forward(self, x):
    return self.net(x)


class SE(nn.Module):

  def __init__(self, ch: int, r: int = 8, **_: Any):
    super().__init__()
    mid = max(1, ch // r)
    self.net = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Conv1d(ch, mid, 1), nn.GELU(), nn.Conv1d(mid, ch, 1),
                             nn.Sigmoid())

  def forward(self, x):
    w = self.net(x)
    return x * w


class ResBlock(nn.Module):

  def __init__(self, ch: int, k: int, dilation: int, p: float, **_: Any):
    super().__init__()
    self.conv1 = ConvDW(ch, ch, k, dilation=dilation, p=p)
    self.conv2 = ConvDW(ch, ch, k, dilation=1, p=p)
    self.se = SE(ch)

  def forward(self, x):
    y = self.conv2(self.conv1(x))
    y = self.se(y)
    return x + y


class AdvCnnRegressor(nn.Module):

  def __init__(self,
               in_feats: int,
               kernel_size: int = 5,
               channels: int = 64,
               blocks: int = 3,
               dropout: float = 0.1,
               clip: float | None = None,
               **_: Any):
    super().__init__()
    self.permute = True
    self.clip = clip
    self.stem = nn.Sequential(nn.Conv1d(in_feats, channels, kernel_size=1, bias=False), nn.BatchNorm1d(channels),
                              nn.GELU())
    layers = []
    for i in range(blocks):
      dil = 2**i  # 1,2,4,...
      layers.append(ResBlock(channels, k=kernel_size, dilation=dil, p=dropout))
    self.backbone = nn.Sequential(*layers)
    self.gap = nn.AdaptiveAvgPool1d(1)
    self.gmp = nn.AdaptiveMaxPool1d(1)
    self.head = nn.Sequential(nn.Flatten(), nn.Linear(2 * channels, channels), nn.GELU(), nn.Dropout(dropout),
                              nn.Linear(channels, 1))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.permute:
      x = x.transpose(1, 2)  # (B, F, T)
    x = self.stem(x)
    x = self.backbone(x)
    x = torch.cat([self.gap(x), self.gmp(x)], dim=1)  # (B, 2C, 1)
    y = self.head(x)
    if self.clip is not None:
      y = self.clip * torch.sigmoid(y)
    return y


class TemporalAttention(nn.Module):

  def __init__(self, hidden: int):
    super().__init__()
    self.W = nn.Linear(hidden, hidden, bias=True)
    self.v = nn.Linear(hidden, 1, bias=False)

  def forward(self, h):
    # h: [B, T, H]
    scores = self.v(torch.tanh(self.W(h)))  # [B, T, 1]
    alpha = torch.softmax(scores, dim=1)  # [B, T, 1]
    context = (alpha * h).sum(dim=1)  # [B, H]
    return context, alpha


class GRUAttnRegressor(nn.Module):

  def __init__(self,
               in_feats: int,
               hidden: int,
               num_layers: int,
               dropout: float,
               bidirectional: bool = False,
               **_: Any):
    super().__init__()
    self.rnn = nn.GRU(input_size=in_feats,
                      hidden_size=hidden,
                      num_layers=num_layers,
                      batch_first=True,
                      dropout=dropout if num_layers > 1 else 0.0,
                      bidirectional=bidirectional)
    h_dim = hidden * (2 if bidirectional else 1)
    self.attn = TemporalAttention(h_dim)
    self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(h_dim, 1))

  def forward(self, x):
    # x: [B, T, F]
    h, _ = self.rnn(x)  # [B, T, H]
    context, alpha = self.attn(h)
    return self.head(context)

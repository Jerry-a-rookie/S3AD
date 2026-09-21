"""
KAN-AD: Time Series Anomaly Detection with Kolmogorov-Arnold Networks.
ICML 2025 — Zhou et al.

Replaces B-spline KAN with truncated Fourier basis.
Only 274 parameters. Window=96, order=2, MSE loss.

Usage:
  from models.kan_ad import KANADModel
  model = KANADModel(window=96, order=2)
  output = model(x)  # x: (B, window)
"""
import torch as th
import torch.nn as nn


class KANADModel(nn.Module):
    """
    KAN-AD: Fourier-KAN for univariate time series anomaly detection.
    Projects input to Fourier basis, passes through Conv1d with residuals,
    predicts the next point. Anomaly score = |pred - actual|.
    """
    def __init__(self, window: int = 96, order: int = 2):
        super().__init__()
        self.window = window
        self.order = order
        self.channels = 2 * order + 1

        self.register_buffer(
            "orders", self._create_custom_periodic_cosine().unsqueeze(0))

        self.out_conv = nn.Conv1d(self.channels, 1, 1, bias=False)
        self.act = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.channels)
        self.bn2 = nn.BatchNorm1d(self.channels)
        self.bn3 = nn.BatchNorm1d(1)
        self.init_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.inner_conv = nn.Conv1d(self.channels, self.channels, 3, 1, 1, bias=False)
        self.final_conv = nn.Conv1d(1, 1, window, padding=0, stride=1, dilation=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        res = [x.unsqueeze(1)]
        ff = th.concat(
            [self.orders.repeat(x.size(0), 1, 1)]
            + [th.cos(order * x.unsqueeze(1)) for order in range(1, self.order + 1)]
            + [x.unsqueeze(1)],
            dim=1,
        )
        res.append(ff)
        ff = self.init_conv(ff)
        ff = self.bn1(ff)
        ff = self.act(ff)
        ff = self.inner_conv(ff) + res.pop()
        ff = self.bn2(ff)
        ff = self.act(ff)
        ff = self.out_conv(ff) + res.pop()
        ff = self.bn3(ff)
        ff = self.act(ff)
        ff = self.final_conv(ff)
        return ff.squeeze(1)

    def _create_custom_periodic_cosine(self) -> th.Tensor:
        result = th.empty(self.order, self.window, dtype=th.float32)
        for i in range(1, self.order + 1):
            range_value = th.arange(self.window, dtype=th.float32)
            result[i - 1, :] = th.cos(2 * th.pi * range_value * i / self.window)
        return result

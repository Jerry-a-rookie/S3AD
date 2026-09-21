"""
DLinear: Are Transformers Effective for Time Series Forecasting? (AAAI 2023)

A simple yet effective baseline: decompose the input window into trend (moving
average) and seasonality (residual), apply two independent linear layers, and sum
the results. For anomaly detection, we predict the next point and score by |pred - actual|.
"""

import torch as th
import torch.nn as nn


class moving_avg(nn.Module):
    """Moving average for trend extraction."""

    def __init__(self, kernel_size: int, stride: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: (B, L) -> (B, L)  with padding on both sides to preserve length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) if x.dim() == 3 else x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        end = x[:, -1:, :].repeat(1, (self.kernel_size) // 2, 1) if x.dim() == 3 else x[:, -1:].repeat(1, self.kernel_size // 2)

        if x.dim() == 3:
            pass  # handle below
        else:
            x = x.unsqueeze(1)  # (B, L) -> (B, 1, L)
            front = front.unsqueeze(1)
            end = end.unsqueeze(1)

        x = th.cat([front, x, end], dim=2)
        x = self.avg(x)
        x = x.squeeze(1)  # (B, L)
        return x


class series_decomp(nn.Module):
    """Series decomposition: trend = MA(x), seasonal = x - trend."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """DLinear: Decomposition Linear model for time series.

    Args:
        window: input window size (lookback length).
        pred_len: prediction length (default 1 for anomaly detection).
        individual: if True, each channel has its own linear weights.
        kernel_size: moving average kernel size for decomposition.
    """

    def __init__(self, window: int = 96, pred_len: int = 1,
                 individual: bool = False, kernel_size: int = 25):
        super().__init__()
        self.window = window
        self.pred_len = pred_len
        self.individual = individual
        self.kernel_size = kernel_size

        self.decomp = series_decomp(kernel_size)

        if individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(1):  # univariate
                self.Linear_Seasonal.append(nn.Linear(window, pred_len))
                self.Linear_Trend.append(nn.Linear(window, pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(window, pred_len)
            self.Linear_Trend = nn.Linear(window, pred_len)

    def forward(self, x):
        """
        Args:
            x: (B, window) univariate input.

        Returns:
            out: (B, pred_len) next-value prediction.
        """
        # Decompose
        seasonal, trend = self.decomp(x)

        if self.individual:
            seasonal_out = self.Linear_Seasonal[0](seasonal)
            trend_out = self.Linear_Trend[0](trend)
        else:
            seasonal_out = self.Linear_Seasonal(seasonal)
            trend_out = self.Linear_Trend(trend)

        out = seasonal_out + trend_out
        return out.squeeze(-1) if self.pred_len == 1 else out

    def count_params(self):
        return sum(p.numel() for p in self.parameters()), \
               sum(p.numel() for p in self.parameters() if p.requires_grad)

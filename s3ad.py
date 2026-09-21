"""
S3AD: Selective State Space for Anomaly Detection

A pure SSM encoder for time series anomaly detection with three
coordinated mathematical redesigns of the standard SSM kernel:

  HiPPO + MultiRes — optimal multi-scale memory allocation (§3.3)
  InputDelta       — input-dependent adaptive discretization (§3.4)
  SpecReg          — spectral regularization against channel degradation (§3.5)

Architecture:
  x → [CWT] → [S³ Block × 2] → [iFFT Predict μ,σ] → [Score]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ========== Improved SSM Block ==========
class SSMBlockV2(nn.Module):
    """SSM with HiPPO+MultiRes+InputDelta and exponential discretization."""
    def __init__(self, d_model, d_state=32, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # HiPPO-LegS + MultiRes: fast/mid/slow groups
        g = d_state // 3
        fast = -0.5 * (2 * torch.arange(1, g+1) - 1) / d_state
        mid = -0.05 * torch.ones(d_state - 2*g) if d_state > 2*g else -0.05 * torch.ones(max(1, d_state-g))
        slow = -0.005 * torch.ones(g)
        mixed = torch.cat([fast, mid, slow])[:d_state]
        self.A_log = nn.Parameter(torch.log((-mixed).clamp(min=1e-4)).unsqueeze(0).repeat(d_model, 1))

        self.B = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model))

        # Input-dependent Δ (per-channel)
        self.delta_proj = nn.Linear(d_model, d_model)

        # Norms + FFN
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model*4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model), nn.Dropout(dropout))

    def forward(self, x):
        B, L, D = x.shape; H = self.d_state
        residual = x; x_n = self.norm1(x)

        # Exponential discretization + per-channel InputDelta
        a_cont = -torch.exp(self.A_log)
        delta = F.softplus(self.delta_proj(x_n)).mean(dim=1)  # (B, D)
        A = torch.exp(delta.unsqueeze(-1) * a_cont.unsqueeze(0))  # (B, D, H)
        A = A.clamp(-0.999, 0.999)

        # FFT convolution kernel
        k_range = torch.arange(L, device=x.device, dtype=x.dtype).view(1,1,1,L)
        A_pow = torch.pow(A.unsqueeze(-1), k_range)  # (B, D, H, L)
        K = (self.C.unsqueeze(0).unsqueeze(-1) * A_pow * self.B.unsqueeze(0).unsqueeze(-1)).sum(dim=2)  # (B, D, L)

        x_f = torch.fft.rfft(x_n, n=2*L, dim=1)
        K_f = torch.fft.rfft(K, n=2*L, dim=2)
        y = torch.fft.irfft(x_f * K_f.permute(0,2,1), n=2*L, dim=1)[:,:L,:]
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_n
        x = y + residual

        residual = x; x = self.norm2(x); x = self.ffn(x)
        return x + residual


class SSMEncoderV2(nn.Module):
    def __init__(self, d_model, d_state=64, num_blocks=2, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([SSMBlockV2(d_model, d_state, dropout) for _ in range(num_blocks)])

    def forward(self, x):
        for blk in self.blocks: x = blk(x)
        return x


# ========== CWT Feature ==========
class CWTFeature(nn.Module):
    """GPU CWT via Conv1d."""
    def __init__(self, window_size, n_scales=16):
        super().__init__()
        self.window_size = window_size; self.n_scales = n_scales
        t = torch.linspace(-window_size//2, window_size//2, window_size)
        filters = []
        for i in range(1, n_scales+1):
            s = i * window_size / n_scales * 0.25 + 0.5
            w = torch.exp(-t**2/(2*s**2)) * torch.cos(5.0*t/s)
            filters.append(w/(w.norm(p=2)+1e-8))
        self.register_buffer('filters', torch.stack(filters).unsqueeze(1))

    def forward(self, x):
        B, N, W = x.shape
        x_f = x.reshape(B*N, 1, W).repeat(1, self.n_scales, 1)
        cwt = F.conv1d(x_f, self.filters, padding=W//2, groups=self.n_scales)
        return cwt.abs().mean(dim=-1).reshape(B, N, self.n_scales)


# ========== Main Model ==========
class S3AD(nn.Module):
    """
    Pure SSM anomaly detector. No decomposition, no noise recycling.
    """
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

        self.win_size = hp.seasonal_window
        self.step = hp.cycle
        self.cwt = CWTFeature(self.win_size, n_scales=16)

        # Input: time values + CWT features
        input_dim = self.win_size + 16
        self.input_proj = nn.Linear(input_dim, hp.d_model)
        self.ssm_encoder = SSMEncoderV2(hp.d_model, d_state=32, num_blocks=1, dropout=hp.dropout_rate)

        # Prediction head (iFFT-based)
        self.pred_head = nn.Sequential(
            nn.Linear(hp.d_model, hp.d_model), nn.Tanh(),
            nn.Linear(hp.d_model, hp.d_model), nn.Tanh(),
            nn.Linear(hp.d_model, self.win_size + 2), nn.Tanh())
        self.mu_layer = nn.Linear(self.win_size, self.win_size)
        self.logvar_layer = nn.Linear(self.win_size, self.win_size)

    def _encode(self, x_flat):
        """x_flat: (B, W) — raw signal"""
        wins = x_flat.unfold(dimension=1, size=self.win_size, step=self.step)  # (B, N, Ws)
        B, N, W = wins.shape
        cwt_f = self.cwt(wins)  # (B, N, 32)
        feat = torch.cat([wins, cwt_f], dim=-1)  # (B, N, Ws+32)
        h = self.input_proj(feat)                # (B, N, d_model)
        h = self.ssm_encoder(h)                  # (B, N, d_model)
        h_last = h[:, -1, :]                     # (B, d_model)

        pred = self.pred_head(h_last).unsqueeze(1)  # (B, 1, Ws+2)
        ws = self.win_size
        real, imag = pred[:,:,:ws//2+1], pred[:,:,ws//2+1:]
        f = torch.view_as_complex(torch.stack([real,imag], dim=-1))
        result = torch.fft.irfft(f)           # (B, 1, Ws)
        mu = self.mu_layer(result)
        log_var = self.logvar_layer(result)
        return mu, log_var

    def forward(self, input, input_normal, mode, mask):
        if mode in ("train","valid"):
            return self._loss(input, input_normal, mask)
        return self._reference(input, input_normal)

    def _loss(self, input, input_normal, mask):
        input_f = input.squeeze(1)
        mu, log_var = self._encode(input_f)
        log_var = log_var.clamp(-10, 10)  # prevent exp(log_var) NaN

        target = input_normal[:,:,-self.win_size:].squeeze(1)
        mask_w = mask[:,-self.win_size:]
        mu, log_var = mu.squeeze(1), log_var.squeeze(1)

        # Hard mask: anomalous positions use predicted μ as target
        target = target * mask_w + mu.detach() * (1 - mask_w)
        loss = torch.mean(0.5*(log_var + (target - mu)**2/torch.exp(log_var)))

        # SpecReg: penalize continuous-time decay magnitudes above 0.99.
        spec = 0.0
        for blk in self.ssm_encoder.blocks:
            a_abs = torch.exp(blk.A_log)
            spec = spec + torch.clamp(a_abs - 0.99, min=0).sum()
        loss = loss + 0.001 * spec

        if torch.isinf(loss): raise ValueError("Loss inf!")
        return loss

    def _reference(self, input, input_normal):
        input_f = input.squeeze(1)
        mu, log_var = self._encode(input_f)
        log_var = log_var.clamp(-10, 10)

        ws = self.win_size
        input_w = input_normal[:,:,-ws:]
        nll = log_var + (input_w - mu)**2/torch.exp(log_var)

        # Variance Discrepancy scoring
        sigma = torch.exp(log_var/2)
        instability = sigma.std(dim=-1)/(sigma.mean(dim=-1)+1e-8)
        score = nll[:,:,-1].unsqueeze(2) * (1.0 + instability.unsqueeze(2))
        recon = mu[:,:,-1].unsqueeze(2)
        return score, recon

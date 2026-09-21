"""
CS-LSTMs model adapter — wraps the original CS-LSTMs (Context & Seasonal LSTMs).

Same interface as S3AD/CWSSM: forward(x, x_normal, mode, mask),
test mode returns (recon_prob, recon_x) — anomaly score + reconstruction.
"""

import sys, os, time
import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from adapters.base_adapter import BaseAnomalyDetector


class WinDS(th.utils.data.Dataset):
    def __init__(self, data, labels, window):
        self.data, self.labels, self.window = data, labels, window
        self.n = max(0, len(data) - window + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (
            th.tensor(self.data[idx:idx + self.window], dtype=th.float32).unsqueeze(0),
            th.tensor(self.labels[idx:idx + self.window], dtype=th.float32),
        )


class CSLSTMsAdapter(BaseAnomalyDetector):
    """Adapter for CS-LSTMs (Context-Seasonal LSTMs, ICLR 2026).

    Wraps the CSLSTMs class directly (same forward signature as S3AD).
    Falls back to MyCSLSTMs Lightning wrapper if needed.
    """

    def __init__(self, window: int = 240, seasonal_window: int = 48,
                 cycle: int = 48, d_model: int = 256, dropout_rate: float = 0.1,
                 learning_rate: float = 1e-3, batch_size: int = 512,
                 max_epoch: int = 30, device: str = 'cuda:0'):
        self.window = window
        self.device = device if th.cuda.is_available() else 'cpu'
        self.hp = SimpleNamespace(
            window=window, seasonal_window=seasonal_window,
            cycle=cycle, contextual_window=min(12, window // 20),
            step=max(2, window // 120), d_model=d_model,
            dropout_rate=dropout_rate, learning_rate=learning_rate,
            batch_size=batch_size, max_epoch=max_epoch,
            use_gpu=('cuda' in self.device), gpu=self.device,
            use_label=True, data_name='experiment', data_dir='.',
            num_workers=0, sliding_window_size=window,
            data_pre_mode=0, only_test=0,
        )
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.model = None
        self._train_mean = 0.0
        self._train_std = 1.0
        self._n_params = 0
        self._n_trainable = 0
        self._train_time_s = 0.0
        self._inf_time_ms = 0.0
        self._gpu_mem_mb = 0.0

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        return (values - self._train_mean) / self._train_std

    def _build_model(self):
        """Try importing CSLSTMs; fall back to LSTM-based simplified version."""
        try:
            # Auto-detect CS-LSTMs package path
            _proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            _cslstms_candidates = [
                os.path.join(_proj_root,
                             'Contextual-and-Seasonal-LSTMs-for-TSAD-main',
                             'Contextual-and-Seasonal-LSTMs-for-TSAD-main',
                             'CS-LSTMs'),
            ]
            # Also search broadly under project root
            for _root, _dirs, _files in os.walk(_proj_root):
                if 'CS-LSTMs' in _dirs:
                    _cslstms_candidates.append(os.path.join(_root, 'CS-LSTMs'))
                    break  # first match only

            _added = False
            for _p in _cslstms_candidates:
                if os.path.isdir(_p) and _p not in sys.path:
                    sys.path.insert(0, _p)
                    _added = True
                    print(f"[CS-LSTMs] Added to path: {_p}")

            from cs_lstms import CSLSTMs
            model = CSLSTMs(self.hp)
            if _added:
                print("[CS-LSTMs] Using original CSLSTMs from cs_lstms package")
            return model
        except ImportError as e:
            print(f"[CS-LSTMs] cs_lstms import failed ({e}) — using PyTorch LSTM fallback")
            return _CSLSTMsFallback(self.hp)

    def fit(self, train_values: np.ndarray, train_labels: np.ndarray,
            valid_values: Optional[np.ndarray] = None,
            valid_labels: Optional[np.ndarray] = None, **kwargs) -> None:
        self._train_mean = train_values.mean()
        self._train_std = train_values.std()
        if self._train_std < 1e-8:
            self._train_std = 1.0

        train_v = self._normalize(train_values)
        v_v = self._normalize(valid_values) if valid_values is not None else None

        train_dl = DataLoader(
            WinDS(train_v, train_labels, self.window),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )
        valid_dl = None
        if v_v is not None:
            valid_dl = DataLoader(
                WinDS(v_v, valid_labels, self.window),
                batch_size=self.batch_size,
            )

        self.model = self._build_model().to(self.device)
        self._n_params = sum(p.numel() for p in self.model.parameters())
        self._n_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"[CSLSTMs] Training {max(1, len(train_dl))} batches x {self.max_epoch} epochs...")
        t0 = time.time()
        best_val, no_imp = float('inf'), 0
        for ep in range(1, self.max_epoch + 1):
            self.model.train()
            tloss = 0
            for x, y in train_dl:
                x, y = x.to(self.device), y.to(self.device)
                loss = self.model(x, x, 'train', th.ones_like(y))
                if th.isnan(loss) or th.isinf(loss):
                    continue
                if isinstance(loss, tuple):
                    loss = loss[0]
                opt.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tloss += loss.item()
            tloss /= max(len(train_dl), 1)

            if valid_dl is not None:
                self.model.eval()
                vloss = 0
                for x, y in valid_dl:
                    out = self.model(x.to(self.device), x.to(self.device),
                                     'valid', th.ones_like(y.to(self.device)))
                    if isinstance(out, tuple):
                        out = out[0]
                    vloss += out.item()
                vloss /= max(len(valid_dl), 1)
                status = '▼' if vloss < best_val else ' '
                print(f"  Ep {ep:2d}/{self.max_epoch} | train={tloss:.4f} | valid={vloss:.4f} {status}")
                if vloss < best_val:
                    best_val = vloss
                    no_imp = 0
                else:
                    no_imp += 1
                if no_imp >= 3:
                    print(f"  Early stop at epoch {ep}")
                    break
            else:
                print(f"  Ep {ep:2d}/{self.max_epoch} | train={tloss:.4f}")
        self._train_time_s = time.time() - t0

    def predict(self, test_values: np.ndarray,
                test_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        test_v = self._normalize(test_values)
        test_dl = DataLoader(
            WinDS(test_v, test_labels, self.window),
            batch_size=self.batch_size,
        )

        self.model.eval()
        scores_list, labels_list = [], []
        total_samples = 0

        t0 = time.time()
        with th.no_grad():
            for x, y in test_dl:
                x = x.to(self.device)
                out = self.model(x, x, 'test', th.ones_like(y))
                if isinstance(out, tuple):
                    # (score, recon) or (recon_prob, recon_x)
                    score = out[0]
                else:
                    score = out
                # score shape: (B, 1, 1) or (B, 1, Ws) — take last
                if score.dim() == 3:
                    score = score[:, :, -1]
                scores_list.append(score.cpu().numpy().flatten())
                labels_list.append(y[:, -1].numpy().flatten())
                total_samples += len(x)
        elapsed = time.time() - t0
        self._inf_time_ms = (elapsed / max(total_samples, 1)) * 1000

        if 'cuda' in str(self.device):
            self._gpu_mem_mb = th.cuda.max_memory_allocated(self.device) / 1024 ** 2
            th.cuda.reset_peak_memory_stats(self.device)

        scores = np.concatenate(scores_list)
        labels = np.concatenate(labels_list)
        valid = ~np.isnan(labels)
        return scores[valid], labels[valid]

    def get_efficiency_stats(self) -> Dict:
        return {
            'n_params': self._n_params,
            'n_trainable': self._n_trainable,
            'train_time_s': self._train_time_s,
            'inf_time_ms_per_sample': self._inf_time_ms,
            'gpu_mem_mb': self._gpu_mem_mb,
        }


class _CSLSTMsFallback(th.nn.Module):
    """Fallback when cs_lstms package is not installed.

    Implements a dual-branch (seasonal + contextual) LSTM encoder that
    mirrors the CS-LSTMs architecture. Not identical, but structurally
    similar for testing the adapter pipeline.
    """

    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.seasonal_lstm = th.nn.LSTM(
            hp.seasonal_window + 32, hp.d_model,
            num_layers=2, batch_first=True, dropout=0.1,
        )
        self.context_lstm = th.nn.LSTM(
            hp.contextual_window + 32, hp.d_model // 4,
            num_layers=2, batch_first=True, dropout=0.1,
        )
        self.seasonal_head = th.nn.Linear(hp.d_model, hp.seasonal_window)
        self.context_head = th.nn.Linear(hp.d_model // 4, hp.contextual_window)
        self._dummy_cwt = th.nn.Sequential(
            th.nn.Conv1d(1, 32, hp.seasonal_window, padding=hp.seasonal_window // 2),
            th.nn.AdaptiveAvgPool1d(1),  # (B, 32, 1) → per-window CWT features
        )

    def forward(self, x, x_normal, mode, mask):
        B = x.shape[0]
        x_sq = x.squeeze(1)  # (B, W)

        # Seasonal branch — unfold + CWT-like features
        N = max(1, x_sq.shape[1] // self.hp.cycle)
        sw = self.hp.seasonal_window
        if x_sq.shape[1] < sw:
            x_sq = th.nn.functional.pad(x_sq, (0, sw - x_sq.shape[1]))
        wins = x_sq.unfold(1, sw, self.hp.cycle)[:, :N, :]  # (B, N, Ws)
        # Dummy CWT: treat each window as independent 1-channel signal
        cwt_f = self._dummy_cwt(wins.reshape(B * N, 1, sw)).reshape(B, N, -1)  # (B, N, 32)
        s_in = th.cat([wins, cwt_f], dim=-1)
        s_out, _ = self.seasonal_lstm(s_in)
        s_pred = self.seasonal_head(s_out[:, -1, :]).unsqueeze(1)  # (B, 1, Ws)

        # Context branch
        cw = self.hp.contextual_window
        c_in = x_sq[:, -cw * 4:].unfold(1, cw, self.hp.step)[:, :, :]
        c_cwt = self._dummy_cwt(c_in.reshape(-1, 1, cw)).reshape(B, -1, 32)
        c_feat = th.cat([c_in, c_cwt], dim=-1)
        c_out, _ = self.context_lstm(c_feat)
        c_pred = self.context_head(c_out[:, -1, :]).unsqueeze(1)

        if mode in ('train', 'valid'):
            # Simple MSE loss as fallback
            target = x_normal[:, :, -sw:].squeeze(1)
            loss = th.nn.functional.mse_loss(s_pred.squeeze(1), target)
            return loss
        else:
            score = th.mean((s_pred[:, :, -1] - x[:, :, -1]) ** 2, dim=1, keepdim=True)
            return score, s_pred[:, :, -1:]

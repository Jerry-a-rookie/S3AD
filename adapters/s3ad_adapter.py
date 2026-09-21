"""
S3AD model adapter — wraps s3ad.S3AD with the unified BaseAnomalyDetector interface.
"""

import sys, os, time
import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s3ad import S3AD
from adapters.base_adapter import BaseAnomalyDetector


class WinDS(th.utils.data.Dataset):
    """Windowed dataset for (B,1,W) models."""

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


class S3ADAdapter(BaseAnomalyDetector):
    """Adapter for S3AD (Selective State Space for Anomaly Detection)."""

    def __init__(self, window: int = 240, seasonal_window: int = 48,
                 cycle: int = 48, d_model: int = 256, dropout_rate: float = 0.1,
                 learning_rate: float = 5e-4, batch_size: int = 512,
                 max_epoch: int = 30, device: str = 'cuda:0'):
        self.window = window
        self.hp = SimpleNamespace(
            window=window, seasonal_window=seasonal_window,
            cycle=cycle, contextual_window=4, step=2,
            d_model=d_model, dropout_rate=dropout_rate,
            learning_rate=learning_rate, batch_size=batch_size,
            max_epoch=max_epoch,
        )
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.learning_rate = learning_rate
        self.device = device if th.cuda.is_available() else 'cpu'
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

        self.model = S3AD(self.hp).to(self.device)
        self._n_params = sum(p.numel() for p in self.model.parameters())
        self._n_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print(f"[S3AD] Training {max(1, len(train_dl))} batches x {self.max_epoch} epochs...")
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
                opt.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                tloss += loss.item()
            tloss /= max(len(train_dl), 1)

            if valid_dl is not None:
                self.model.eval()
                vloss = sum(
                    self.model(x.to(self.device), x.to(self.device),
                               'valid', th.ones_like(y.to(self.device))).item()
                    for x, y in valid_dl
                ) / max(len(valid_dl), 1)
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
                score, _ = self.model(x, x, 'test', th.ones_like(y))
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

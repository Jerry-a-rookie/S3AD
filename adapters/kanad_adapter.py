"""
KAN-AD model adapter — wraps models.kan_ad.KANADModel.

KAN-AD predicts a single next point from a window. For anomaly scoring,
we slide the window across the test sequence, predict at each step,
and score = |prediction - actual|.
"""

import sys, os, time, importlib.util
import numpy as np
import torch as th
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace
from typing import Dict, Optional, Tuple

# Load KANADModel directly — avoids models/__init__.py which drags in CWSSM
_KANAD_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'models', 'kan_ad.py')
spec = importlib.util.spec_from_file_location('kan_ad', _KANAD_PATH)
_kan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_kan)
KANADModel = _kan.KANADModel

from adapters.base_adapter import BaseAnomalyDetector


class KANADAdapter(BaseAnomalyDetector):
    """Adapter for KAN-AD (Fourier Kolmogorov-Arnold Network for Anomaly Detection).

    Only 274 parameters — extremely lightweight baseline.
    """

    def __init__(self, window: int = 96, order: int = 2,
                 learning_rate: float = 1e-3, batch_size: int = 512,
                 max_epoch: int = 30, device: str = 'cuda:0'):
        self.window = window
        self.order = order
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

        # Create window → next-point pairs
        X_train, y_train = self._make_sliding_windows(train_v)

        train_dl = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size, shuffle=True, drop_last=True,
        )

        self.model = KANADModel(window=self.window, order=self.order).to(self.device)
        self._n_params = sum(p.numel() for p in self.model.parameters())
        self._n_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)

        opt = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = th.nn.MSELoss()

        print(f"[KANAD] Training {max(1, len(train_dl))} batches x {self.max_epoch} epochs...")
        t0 = time.time()
        for ep in range(1, self.max_epoch + 1):
            self.model.train()
            tloss = 0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tloss += loss.item()
            tloss /= max(len(train_dl), 1)
            if ep % 5 == 0 or ep == 1:
                print(f"  Ep {ep:2d}/{self.max_epoch} | train={tloss:.6f}")
        self._train_time_s = time.time() - t0

    def predict(self, test_values: np.ndarray,
                test_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        test_v = self._normalize(test_values)
        n = len(test_v)

        # Build sliding windows for entire test sequence
        windows, targets = self._make_sliding_windows(test_v)

        self.model.eval()
        t0 = time.time()
        preds = []
        with th.no_grad():
            for i in range(0, len(windows), self.batch_size):
                batch = windows[i:i + self.batch_size].to(self.device)
                preds.append(self.model(batch).cpu())
        elapsed = time.time() - t0

        preds = th.cat(preds).numpy().flatten()  # ensure 1D
        scores_raw = np.abs(preds - targets.numpy())

        # Align scores to original timeline:
        # window[i] covers positions [i, i+window), predicts at i+window
        scores = np.zeros(n)
        count = np.zeros(n)
        for i in range(len(scores_raw)):
            pos = i + self.window
            if pos < n:
                scores[pos] += scores_raw[i]
                count[pos] += 1

        # Fill positions before first window with 0
        count[count == 0] = 1
        scores = scores / count

        # Align labels to windowed output
        labels_full = test_labels[:len(scores)]
        scores = scores[:len(labels_full)]

        total_samples = max(len(preds), 1)
        self._inf_time_ms = (elapsed / total_samples) * 1000

        if 'cuda' in str(self.device):
            self._gpu_mem_mb = th.cuda.max_memory_allocated(self.device) / 1024 ** 2
            th.cuda.reset_peak_memory_stats(self.device)

        valid = ~np.isnan(labels_full)
        return scores[valid], labels_full[valid]

    def get_efficiency_stats(self) -> Dict:
        return {
            'n_params': self._n_params,
            'n_trainable': self._n_trainable,
            'train_time_s': self._train_time_s,
            'inf_time_ms_per_sample': self._inf_time_ms,
            'gpu_mem_mb': self._gpu_mem_mb,
        }

    def _make_sliding_windows(self, values: np.ndarray):
        """Create windows and next-point targets from 1D values."""
        n = len(values)
        windows_list = []
        targets_list = []
        for i in range(n - self.window):
            windows_list.append(values[i:i + self.window])
            targets_list.append(values[i + self.window])
        if not windows_list:
            return th.zeros(0, self.window), th.zeros(0)
        return (
            th.tensor(np.array(windows_list), dtype=th.float32),
            th.tensor(np.array(targets_list), dtype=th.float32),
        )

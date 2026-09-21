"""
Unified data pipeline for all experiments.

Handles:
- Dataset loading (CSV and NPY formats, auto-detection)
- Train/valid/test splitting (mirrors train_p2.py convention)
- Z-score normalization
- Anomaly injection (4 patterns)
- Point missing and block missing simulation
"""

import os, sys
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Dataset Loading ───────────────────────────────────────────────

def load_dataset(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load values and labels from a dataset directory.

    Auto-detects CSV format (data_dir/*.csv) vs NPY format
    (data_dir/subdirs with train.npy, test.npy, etc.).

    Returns:
        (values, labels) as 1D float64 arrays.
    """
    # NPY format: subdirectories with train.npy / test.npy
    subs = [d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))]
    if subs and os.path.exists(os.path.join(data_dir, subs[0], 'train.npy')):
        return _load_npy(data_dir, subs)

    # CSV format: files with value/label columns
    return _load_csv(data_dir)


def _load_csv(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    vals, lbls = [], []
    for f in sorted(os.listdir(data_dir)):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_dir, f))
            vals.append(df['value'].values.astype(float))
            lbls.append(df['label'].values.astype(float)
                       if 'label' in df.columns
                       else np.zeros(len(df)))
    return np.concatenate(vals), np.concatenate(lbls)


def _load_npy(data_dir: str, subs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    vals, lbls = [], []
    for sd in subs:
        p = os.path.join(data_dir, sd)
        tr = np.load(os.path.join(p, 'train.npy')).flatten()
        te = np.load(os.path.join(p, 'test.npy')).flatten()
        tr_l = np.load(os.path.join(p, 'train_label.npy')).flatten()
        te_l = np.load(os.path.join(p, 'test_label.npy')).flatten()
        m, s = tr.mean(), tr.std()
        if s < 1e-8:
            s = 1.0
        vals.append(np.concatenate([(tr - m) / s, (te - m) / s]))
        lbls.append(np.concatenate([tr_l, te_l]))
    return np.concatenate(vals), np.concatenate(lbls)


# ─── Train/Valid/Test Split ───────────────────────────────────────

def split_data(values: np.ndarray, labels: np.ndarray
               ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                         Tuple[np.ndarray, np.ndarray],
                         Tuple[np.ndarray, np.ndarray]]:
    """Split data 35%/15%/50% — matches train_p2.py convention."""
    n = len(values)
    t_end = int(n * 0.35)
    v_end = int(n * 0.50)
    return (
        (values[:t_end], labels[:t_end]),
        (values[t_end:v_end], labels[t_end:v_end]),
        (values[v_end:], labels[v_end:]),
    )


# ─── Anomaly Injection ────────────────────────────────────────────

class AnomalyInjector:
    """Inject synthetic anomalies into time series data.

    Four anomaly patterns for simulating complex working conditions:
      1. Spike     — sudden amplitude impulse
      2. Trend     — gradual level shift
      3. Variance  — amplified local variance
      4. Frequency — superimposed high-frequency oscillation
    """

    @staticmethod
    def inject_spike(values: np.ndarray, positions: np.ndarray,
                     amplitude_sigma: float = 5.0,
                     duration: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Inject spike anomalies.

        Args:
            values: 1D signal array.
            positions: array of starting indices for injection.
            amplitude_sigma: spike amplitude in units of signal std.
            duration: max spike duration (1..duration steps, random).

        Returns:
            (modified_values, binary_labels) where labels=1 at anomaly positions.
        """
        x = values.copy()
        labels = np.zeros(len(x), dtype=np.float32)
        sigma = np.std(values)
        for pos in positions:
            pos = min(int(pos), len(x) - 2)
            dur = np.random.randint(1, duration + 1)
            end = min(pos + dur, len(x))
            amp = amplitude_sigma * sigma * (1 if np.random.rand() > 0.5 else -1)
            x[pos:end] += amp
            labels[pos:end] = 1
        return x, labels

    @staticmethod
    def inject_trend_shift(values: np.ndarray, positions: np.ndarray,
                           drift_factor: float = 0.5,
                           duration: Tuple[int, int] = (20, 50)
                           ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject trend shift anomalies (linear drift).

        Args:
            drift_factor: max drift in units of signal std.
            duration: (min, max) steps for the shift.
        """
        x = values.copy()
        labels = np.zeros(len(x), dtype=np.float32)
        sigma = np.std(values)
        for pos in positions:
            pos = min(int(pos), len(x) - 5)
            dur = np.random.randint(*duration)
            end = min(pos + dur, len(x))
            drift = np.linspace(0, drift_factor * sigma, end - pos)
            direction = 1 if np.random.rand() > 0.5 else -1
            x[pos:end] += direction * drift
            labels[pos:end] = 1
        return x, labels

    @staticmethod
    def inject_variance_change(values: np.ndarray, positions: np.ndarray,
                               factor: float = 3.0,
                               duration: Tuple[int, int] = (20, 50)
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject variance change anomalies (amplified local variance)."""
        x = values.copy()
        labels = np.zeros(len(x), dtype=np.float32)
        local_mean = np.convolve(values, np.ones(21) / 21, mode='same')
        for pos in positions:
            pos = min(int(pos), len(x) - 5)
            dur = np.random.randint(*duration)
            end = min(pos + dur, len(x))
            mu = local_mean[pos]
            x[pos:end] = mu + (x[pos:end] - mu) * factor
            labels[pos:end] = 1
        return x, labels

    @staticmethod
    def inject_frequency_change(values: np.ndarray, positions: np.ndarray,
                                freq_mult: float = 5.0,
                                duration: Tuple[int, int] = (20, 50)
                                ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject frequency change anomalies (high-freq oscillation)."""
        x = values.copy()
        labels = np.zeros(len(x), dtype=np.float32)
        sigma = np.std(values) * 0.3
        for pos in positions:
            pos = min(int(pos), len(x) - 5)
            dur = np.random.randint(*duration)
            end = min(pos + dur, len(x))
            t = np.arange(end - pos)
            freq = freq_mult * 2 * np.pi / max(dur, 1)
            x[pos:end] += sigma * np.sin(freq * t)
            labels[pos:end] = 1
        return x, labels


# ─── Missing Data Simulation ──────────────────────────────────────

def apply_point_missing(values: np.ndarray, rate: float,
                        seed: int = 42) -> np.ndarray:
    """Randomly mask individual time points and interpolate.

    Args:
        values: 1D signal array.
        rate: fraction of points to mask (0.0 to 1.0).
        seed: random seed for reproducibility.

    Returns:
        Interpolated values with same shape as input.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    mask = rng.rand(n) < rate

    # Create Series, set masked positions to NaN, interpolate
    s = pd.Series(values.astype(float))
    s[mask] = np.nan
    s = s.interpolate(method='linear', limit_direction='both')
    return s.values


def apply_block_missing(values: np.ndarray, block_size: int,
                        coverage: float = 0.10,
                        seed: int = 42) -> np.ndarray:
    """Randomly mask contiguous blocks and interpolate.

    Args:
        values: 1D signal array.
        block_size: number of consecutive points per missing block.
        coverage: target fraction of total data covered by blocks.
        seed: random seed.

    Returns:
        Interpolated values.
    """
    rng = np.random.RandomState(seed)
    n = len(values)
    total_covered = 0
    s = pd.Series(values.astype(float))

    while total_covered / n < coverage:
        start = rng.randint(0, max(1, n - block_size))
        end = min(start + block_size, n)
        s[start:end] = np.nan
        total_covered += (end - start)

    s = s.interpolate(method='linear', limit_direction='both')
    return s.values


def get_missing_mask(values_before: np.ndarray,
                     values_after: np.ndarray) -> np.ndarray:
    """Return boolean mask indicating which points were interpolated."""
    return np.isnan(
        pd.Series(values_before.astype(float)).replace(0, np.nan)
    ) if False else np.abs(values_before - values_after) < 1e-10

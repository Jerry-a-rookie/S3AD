"""
Base class for all anomaly detection model adapters.

Provides a unified interface so that S3AD, CW-SSM, KAN-AD,
Anomaly Transformer, PatchTST, and Informer can be compared
fairly in the same experiment pipeline.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np


class BaseAnomalyDetector(ABC):
    """Unified interface for all anomaly detectors."""

    @abstractmethod
    def fit(self, train_values: np.ndarray, train_labels: np.ndarray,
            valid_values: Optional[np.ndarray] = None,
            valid_labels: Optional[np.ndarray] = None, **kwargs) -> None:
        """Train the model on the given data.

        Args:
            train_values: 1D numpy array of training signal values.
            train_labels: 1D numpy array of binary anomaly labels.
            valid_values: Optional 1D validation signal values.
            valid_labels: Optional 1D validation labels.
        """

    @abstractmethod
    def predict(self, test_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run anomaly detection on test data.

        Args:
            test_values: 1D numpy array of test signal values.

        Returns:
            (scores, labels) tuple:
              - scores: 1D array of per-point anomaly scores.
              - labels: 1D array of binary ground-truth labels (window-aligned).
        """

    @abstractmethod
    def get_efficiency_stats(self) -> Dict:
        """Return computational efficiency statistics.

        Returns:
            dict with keys: n_params, n_trainable, train_time_s,
            inf_time_ms_per_sample, gpu_mem_mb, flops (optional).
        """

    @staticmethod
    def _normalize(train: np.ndarray, *arrays: np.ndarray) -> list:
        """Z-score normalize using train statistics. Returns list of
        normalized arrays (train first, then the *arrays in order)."""
        mean, std = train.mean(), train.std()
        if std < 1e-8:
            std = 1.0
        return [(a - mean) / std for a in (train,) + arrays]

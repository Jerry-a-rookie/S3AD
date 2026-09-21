"""Baseline model adapters — unified interface for all anomaly detectors.

Uses lazy import so that adapters with missing dependencies don't block
the entire module from loading.
"""

from adapters.base_adapter import BaseAnomalyDetector

_ADAPTERS = {}


def _ensure_adapters():
    """Lazy-load adapter classes."""
    if _ADAPTERS:
        return

    # Always available
    from adapters.s3ad_adapter import S3ADAdapter
    _ADAPTERS['S3AD'] = S3ADAdapter

    try:
        from adapters.cslstms_adapter import CSLSTMsAdapter
        _ADAPTERS['CSLSTMs'] = CSLSTMsAdapter
    except Exception as e:
        print(f"[WARN] CWSSM adapter not available: {e}")

    try:
        from adapters.kanad_adapter import KANADAdapter
        _ADAPTERS['KANAD'] = KANADAdapter
    except Exception as e:
        print(f"[WARN] KANAD adapter not available: {e}")

    try:
        from adapters.dlinear_adapter import DLinearAdapter
        _ADAPTERS['DLinear'] = DLinearAdapter
    except Exception as e:
        print(f"[WARN] DLinear adapter not available: {e}")


def get_adapter(name: str, **kwargs) -> BaseAnomalyDetector:
    """Factory: return an adapter instance by name.

    Args:
        name: one of 'S3AD', 'CSLSTMs', 'KANAD', or 'DLinear'.
        **kwargs: passed to the adapter constructor (e.g., window, device).

    Returns:
        A BaseAnomalyDetector instance.
    """
    _ensure_adapters()
    if name not in _ADAPTERS:
        raise ValueError(
            f"Unknown adapter '{name}'. Available: {list(_ADAPTERS.keys())}")
    return _ADAPTERS[name](**kwargs)


def list_adapters() -> list:
    """Return list of available adapter names."""
    _ensure_adapters()
    return list(_ADAPTERS.keys())

"""Math utility functions for SOG conversion."""

import numpy as np


def sigmoid(v: np.ndarray) -> np.ndarray:
    """Apply sigmoid function element-wise."""
    return 1.0 / (1.0 + np.exp(-v))


def log_transform(value: np.ndarray) -> np.ndarray:
    """Apply log transform: sign(x) * log(|x| + 1)."""
    return np.sign(value) * np.log(np.abs(value) + 1)

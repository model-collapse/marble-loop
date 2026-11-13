"""Editable: return a predictor fitted on (x, y). Baseline: ordinary least squares on a line."""
import numpy as np


def fit(x: np.ndarray, y: np.ndarray):
    A = np.stack([x, np.ones_like(x)], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return lambda xs: coef[0] * np.asarray(xs) + coef[1]

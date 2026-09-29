"""OPTIONAL EXTENSION: true elastic-net output layer by coordinate descent.

This is NOT the paper's solver. The paper's beta = (H^T H + lambda1 I + lambda2 I)^-1 H^T T
is ridge regression (see qelm.py). This module exists so a reader can compare a genuine
L1 + L2 penalty against the ridge closed form. Objective per output column c:

    0.5 * || t_c - H beta_c ||^2  +  l1 * || beta_c ||_1  +  0.5 * l2 * || beta_c ||^2

Solved by cyclic coordinate descent with soft thresholding, vectorised across outputs.
Select it with `qelm.solver: elastic_net_cd` in a config.
"""
from __future__ import annotations

import numpy as np


def soft_threshold(z: np.ndarray, t: float) -> np.ndarray:
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def elastic_net_cd(H: np.ndarray, T: np.ndarray, l1: float, l2: float, max_iter: int = 200,
                   tol: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Return beta (d, C) and solver info for the objective in the module docstring."""
    H = np.asarray(H, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    if T.ndim == 1:
        T = T[:, None]
    if l1 < 0 or l2 < 0:
        raise ValueError("l1 and l2 must be non-negative")
    n, d = H.shape
    C = T.shape[1]
    beta = np.zeros((d, C), dtype=np.float64)
    col_sq = (H * H).sum(axis=0)
    residual = T.copy()
    n_iter = 0
    converged = False
    for n_iter in range(1, max_iter + 1):
        max_delta = 0.0
        for j in range(d):
            if col_sq[j] == 0.0:
                continue
            hj = H[:, j]
            old = beta[j].copy()
            partial = residual + np.outer(hj, old)          # residual without coordinate j
            z = hj @ partial
            new = soft_threshold(z, l1) / (col_sq[j] + l2)
            beta[j] = new
            residual = partial - np.outer(hj, new)
            max_delta = max(max_delta, float(np.max(np.abs(new - old))))
        if max_delta < tol:
            converged = True
            break
    info = {"method": "coordinate_descent_elastic_net", "l1": float(l1), "l2": float(l2),
            "iterations": int(n_iter), "converged": converged, "tol": tol}
    return beta, info


def objective(H: np.ndarray, T: np.ndarray, beta: np.ndarray, l1: float, l2: float) -> float:
    r = T - H @ beta
    return float(0.5 * np.sum(r * r) + l1 * np.sum(np.abs(beta)) + 0.5 * l2 * np.sum(beta * beta))

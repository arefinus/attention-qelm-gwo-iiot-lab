"""Runnable baseline adapters (scikit-learn).

Implemented: logistic_regression, random_forest, hist_gradient_boosting. The paper's
deep baselines (CNN-1D, LSTM, GRU, BiLSTM, DenseNet) are `proposed` and NOT implemented
here; see docs/paper-implementation-audit.md.
"""
from __future__ import annotations

from .sklearn_adapters import BASELINES, BaselineResult, run_baseline

__all__ = ["BASELINES", "BaselineResult", "run_baseline"]

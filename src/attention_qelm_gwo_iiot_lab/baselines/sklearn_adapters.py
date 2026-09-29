"""scikit-learn baselines sharing the QELM's preprocessing and split."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

BASELINES = ("logistic_regression", "random_forest", "hist_gradient_boosting")
PROPOSED_NOT_IMPLEMENTED = ("cnn_1d", "lstm", "gru", "bilstm", "densenet")


@dataclass(frozen=True)
class BaselineResult:
    name: str
    y_pred: np.ndarray
    scores: np.ndarray          # (N, C) class scores in ClassSpec order
    seconds: float
    seed: int


PROFILES = {"standard": {"rf_trees": 100, "hgb_iter": 100}, "light": {"rf_trees": 50, "hgb_iter": 50}}


def _make(name: str, seed: int, n_classes: int, profile: str = "standard"):
    if profile not in PROFILES:
        raise ValueError(f"unknown baseline profile {profile!r}; use {list(PROFILES)}")
    p = PROFILES[profile]
    if name == "logistic_regression":
        return LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", random_state=seed)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=p["rf_trees"], class_weight="balanced_subsample",
                                      random_state=seed, n_jobs=1)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=p["hgb_iter"], learning_rate=0.1, random_state=seed,
                                              class_weight="balanced")
    if name in PROPOSED_NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"baseline {name!r} is listed in the paper but is `proposed` in this repository; "
            f"implemented baselines: {BASELINES}")
    raise ValueError(f"unknown baseline {name!r}; implemented: {BASELINES}")


def run_baseline(name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                 n_classes: int, seed: int = 42, profile: str = "standard") -> BaselineResult:
    """Fit on the training partition and score the test partition once."""
    model = _make(name, seed, n_classes, profile)
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)
    scores = np.zeros((len(X_test), n_classes), dtype=np.float64)
    for j, cls in enumerate(model.classes_):
        scores[:, int(cls)] = proba[:, j]
    y_pred = scores.argmax(axis=1)
    return BaselineResult(name=name, y_pred=y_pred, scores=scores,
                          seconds=time.perf_counter() - t0, seed=seed)

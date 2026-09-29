"""Fitness bridge between the GWO and the QELM: k-fold macro-F1 on TRAIN embeddings only.

The fitness callable receives frozen encoder features of the training partition and a
list of inner folds (positions within the training partition). It never receives test
rows: there is no argument through which they could be passed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score

from .gwo import GWOConfig, GWOResult, GreyWolfOptimizer, SearchSpace
from .qelm import QELM, QELMConfig, one_hot

SEARCHED_KEYS = ("n_layers", "n_hidden", "activation", "lambda1", "lambda2", "theta", "rho")


@dataclass
class FitnessLog:
    calls: list[dict] = field(default_factory=list)


def make_qelm_from_params(base: QELMConfig, params: dict) -> QELMConfig:
    """Overlay searched parameters on a base config (keeps angle mode, scaling, seed)."""
    d = {k: getattr(base, k) for k in base.__dataclass_fields__}
    for k in SEARCHED_KEYS:
        if k in params:
            d[k] = params[k]
    return QELMConfig(**d)


def qelm_cv_fitness(H_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                    folds: list[tuple[np.ndarray, np.ndarray]], base: QELMConfig,
                    log: FitnessLog | None = None) -> Callable[[dict], float]:
    """Return fitness(params) = mean macro-F1 across the given inner folds."""
    H_train = np.asarray(H_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.int64)
    n_train = len(y_train)
    for tr, va in folds:
        if tr.max() >= n_train or va.max() >= n_train or tr.min() < 0 or va.min() < 0:
            raise ValueError("fold indices must be positions within the training partition")
    T_all = one_hot(y_train, n_classes)

    def fitness(params: dict) -> float:
        cfg = make_qelm_from_params(base, params)
        scores = []
        for tr, va in folds:
            model = QELM(cfg, H_train.shape[1], n_classes).fit(H_train[tr], T_all[tr])
            pred = model.predict(H_train[va])
            scores.append(f1_score(y_train[va], pred, labels=list(range(n_classes)),
                                   average="macro", zero_division=0))
        val = float(np.mean(scores))
        if log is not None:
            log.calls.append({"params": params, "fold_scores": [float(s) for s in scores], "fitness": val})
        return val

    return fitness


def run_search(H_train: np.ndarray, y_train: np.ndarray, n_classes: int,
               folds: list[tuple[np.ndarray, np.ndarray]], base: QELMConfig,
               gwo_cfg: dict) -> tuple[GWOResult, FitnessLog]:
    space = SearchSpace.from_dict(gwo_cfg["space"])
    log = FitnessLog()
    fitness = qelm_cv_fitness(H_train, y_train, n_classes, folds, base, log)
    optimizer = GreyWolfOptimizer(space, GWOConfig.from_dict(gwo_cfg))
    return optimizer.run(fitness), log

"""Stratified outer split and inner k-fold within the training partition only."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


@dataclass(frozen=True)
class SplitPlan:
    train_idx: np.ndarray
    test_idx: np.ndarray
    seed: int
    test_fraction: float

    def sample_counts(self, y: np.ndarray, class_names: tuple[str, ...]) -> dict:
        def counts(idx: np.ndarray) -> dict[str, int]:
            c = np.bincount(y[idx], minlength=len(class_names))
            return {name: int(n) for name, n in zip(class_names, c)}
        return {"train": int(len(self.train_idx)), "test": int(len(self.test_idx)),
                "train_per_class": counts(self.train_idx), "test_per_class": counts(self.test_idx)}


def outer_split(y: np.ndarray, test_fraction: float = 0.20, seed: int = 42) -> SplitPlan:
    """Stratified 80/20 split over row indices. Every class must have at least 2 rows."""
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be in (0, 1)")
    counts = np.bincount(y)
    too_small = np.where((counts > 0) & (counts < 2))[0]
    if len(too_small):
        raise ValueError(f"classes with fewer than 2 rows cannot be stratified: {too_small.tolist()}")
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_fraction, stratify=y, random_state=seed)
    return SplitPlan(train_idx=np.sort(tr), test_idx=np.sort(te), seed=seed,
                     test_fraction=float(test_fraction))


def inner_folds(y_train: np.ndarray, n_folds: int = 5, seed: int = 42
                ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified k-fold over POSITIONS within the training partition.

    Returned indices index into the training arrays, never into the full table, so a
    caller cannot accidentally reach test rows. The rarest class must have at least
    n_folds rows; otherwise an error is raised rather than silently merging classes.
    """
    counts = np.bincount(y_train)
    present = counts[counts > 0]
    if present.min() < n_folds:
        raise ValueError(
            f"rarest class has {int(present.min())} training rows, fewer than n_folds={n_folds}; "
            "reduce n_folds or supply more rows"
        )
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, va) for tr, va in skf.split(np.zeros(len(y_train)), y_train)]


def split_manifest_hash(plan: SplitPlan) -> str:
    h = hashlib.sha256()
    h.update(plan.train_idx.astype(np.int64).tobytes())
    h.update(b"|")
    h.update(plan.test_idx.astype(np.int64).tobytes())
    return h.hexdigest()

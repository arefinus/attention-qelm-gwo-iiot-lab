"""Preprocessing fitted on the training partition only.

Order of operations (paper Section 3.2 as read by this implementation):
  1. drop columns with more than `drop_missing_above` missing fraction (measured on train)
  2. replace +/- inf by the column maximum of the finite TRAIN values
  3. impute: median (numeric) / mode (categorical), statistics from train
  4. label-encode categorical columns with a train-fit vocabulary (unseen -> -1 then 0..1 scaled)
  5. Min-Max scale to [0, 1] with train min/max; transformed rows outside the train range
     are clipped to [0, 1] (documented choice; the paper does not say)
Constant columns (max == min on train) map to 0.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data.manifest import FeatureManifest


@dataclass(frozen=True)
class FittedStats:
    kept_columns: tuple[str, ...]
    dropped_columns: tuple[str, ...]
    missing_fraction: dict[str, float]
    inf_replacement: dict[str, float]
    impute_value: dict[str, float | str]
    vocab: dict[str, dict[str, int]] = field(default_factory=dict)
    col_min: dict[str, float] = field(default_factory=dict)
    col_max: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "kept_columns": list(self.kept_columns),
            "dropped_columns": list(self.dropped_columns),
            "missing_fraction": self.missing_fraction,
            "inf_replacement": self.inf_replacement,
            "impute_value": self.impute_value,
            "vocab": self.vocab,
            "col_min": self.col_min,
            "col_max": self.col_max,
        }


class Preprocessor:
    def __init__(self, manifest: FeatureManifest, drop_missing_above: float = 0.40) -> None:
        if not 0.0 <= drop_missing_above <= 1.0:
            raise ValueError("drop_missing_above must be in [0, 1]")
        self.manifest = manifest
        self.drop_missing_above = float(drop_missing_above)
        self.stats: FittedStats | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, X_train: pd.DataFrame) -> "Preprocessor":
        missing_fraction: dict[str, float] = {}
        kept: list[str] = []
        dropped: list[str] = []
        for col in self.manifest.names:
            s = X_train[col]
            if self.manifest.dtype_of(col) == "categorical":
                miss = float(s.isna().mean())
            else:
                # inf is a value to be replaced, not a missing entry
                miss = float(pd.to_numeric(s, errors="coerce").isna().mean())
            missing_fraction[col] = miss
            (dropped if miss > self.drop_missing_above else kept).append(col)

        inf_replacement: dict[str, float] = {}
        impute_value: dict[str, float | str] = {}
        vocab: dict[str, dict[str, int]] = {}
        col_min: dict[str, float] = {}
        col_max: dict[str, float] = {}
        for col in kept:
            if self.manifest.dtype_of(col) == "categorical":
                s = X_train[col].astype("string")
                mode = s.mode(dropna=True)
                impute_value[col] = str(mode.iloc[0]) if len(mode) else "missing"
                cats = sorted(s.dropna().unique().tolist())
                vocab[col] = {c: i for i, c in enumerate(cats)}
                encoded = s.fillna(impute_value[col]).map(vocab[col]).astype(float)
                col_min[col], col_max[col] = float(encoded.min()), float(encoded.max())
            else:
                s = pd.to_numeric(X_train[col], errors="coerce").astype(float)
                finite = s[np.isfinite(s)]
                inf_replacement[col] = float(finite.max()) if len(finite) else 0.0
                s = s.replace([np.inf, -np.inf], inf_replacement[col])
                med = float(s.median()) if s.notna().any() else 0.0
                impute_value[col] = med
                s = s.fillna(med)
                col_min[col], col_max[col] = float(s.min()), float(s.max())
        self.stats = FittedStats(kept_columns=tuple(kept), dropped_columns=tuple(dropped),
                                 missing_fraction=missing_fraction, inf_replacement=inf_replacement,
                                 impute_value=impute_value, vocab=vocab,
                                 col_min=col_min, col_max=col_max)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.stats is None:
            raise RuntimeError("Preprocessor.transform called before fit")
        st = self.stats
        out = np.empty((len(X), len(st.kept_columns)), dtype=np.float32)
        for j, col in enumerate(st.kept_columns):
            if col in st.vocab:
                s = X[col].astype("string").fillna(str(st.impute_value[col]))
                enc = s.map(st.vocab[col])
                enc = pd.to_numeric(enc, errors="coerce").fillna(st.vocab[col].get(str(st.impute_value[col]), 0)).astype(float)
                v = enc.to_numpy()
            else:
                s = pd.to_numeric(X[col], errors="coerce").astype(float)
                s = s.replace([np.inf, -np.inf], st.inf_replacement[col]).fillna(float(st.impute_value[col]))
                v = s.to_numpy()
            lo, hi = st.col_min[col], st.col_max[col]
            if hi > lo:
                v = (v - lo) / (hi - lo)
            else:
                v = np.zeros_like(v)
            out[:, j] = np.clip(v, 0.0, 1.0)
        return out

    def fit_transform(self, X_train: pd.DataFrame) -> np.ndarray:
        return self.fit(X_train).transform(X_train)

    @property
    def n_output_features(self) -> int:
        if self.stats is None:
            raise RuntimeError("Preprocessor not fitted")
        return len(self.stats.kept_columns)

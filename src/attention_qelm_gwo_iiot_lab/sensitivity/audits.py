"""Three cheap audits that a tabular IDS result should pass before it is believed.

1. identifier_like_columns: columns whose values behave like row identifiers (near-unique,
   or monotone in row order), which a model can memorise instead of learning traffic.
2. placeholder_encoding_audit: columns where a single placeholder value (0, -1, empty)
   dominates AND its presence mask alone predicts the label (normalised mutual
   information), a sign that missingness encodes the protocol or the attack tool.
3. duplicate_flow_check: exact duplicate feature rows within and across train/test.
Thresholds are configurable and reported; nothing here is a verdict on real data.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

PLACEHOLDERS = (0, 0.0, -1, -1.0, "", "0", "-1", "nan", "None")


def _is_discrete(num: pd.Series, raw: pd.Series) -> bool:
    """Integer-valued numeric columns and non-numeric columns are 'discrete'.

    A continuous float column is near-unique by nature, so uniqueness alone says nothing
    about it; uniqueness is only suspicious for discrete columns (counters, ids, strings).
    """
    if num.notna().sum() < len(raw.dropna()):
        return True                                  # strings present
    vals = num.dropna().to_numpy()
    return bool(len(vals)) and bool(np.all(np.isfinite(vals))) and bool(np.allclose(vals, np.round(vals)))


def identifier_like_columns(X: pd.DataFrame, unique_ratio: float = 0.95,
                            monotone_ratio: float = 0.98) -> list[dict]:
    flagged = []
    n = len(X)
    for col in X.columns:
        s = X[col]
        nun = int(s.nunique(dropna=True))
        ratio = nun / max(n, 1)
        reasons = []
        num = pd.to_numeric(s, errors="coerce")
        if ratio >= unique_ratio and n >= 20 and _is_discrete(num, s):
            reasons.append(f"unique_ratio={ratio:.3f}")
        if num.notna().sum() >= 20:
            d = np.diff(num.dropna().to_numpy())
            if len(d) and (np.mean(d > 0) >= monotone_ratio or np.mean(d < 0) >= monotone_ratio):
                reasons.append("monotone_in_row_order")
        if reasons:
            flagged.append({"column": str(col), "n_unique": nun, "unique_ratio": float(ratio),
                            "reasons": reasons})
    return flagged


def placeholder_encoding_audit(X: pd.DataFrame, y: np.ndarray, min_share: float = 0.05,
                               nmi_threshold: float = 0.30) -> list[dict]:
    flagged = []
    y = np.asarray(y)
    for col in X.columns:
        s = X[col]
        mask = s.isna() | s.isin(PLACEHOLDERS)
        share = float(mask.mean())
        if share < min_share or share > 1.0 - 1e-9:
            continue
        nmi = float(normalized_mutual_info_score(y, mask.astype(int)))
        if nmi >= nmi_threshold:
            flagged.append({"column": str(col), "placeholder_share": share, "nmi_mask_vs_label": nmi,
                            "interpretation": "placeholder presence predicts the label; check whether the "
                                              "value is structurally undefined for some protocols"})
    return flagged


def _row_hashes(X: np.ndarray) -> np.ndarray:
    Xr = np.ascontiguousarray(np.round(np.asarray(X, dtype=np.float64), 6))
    return np.array([hashlib.sha1(row.tobytes()).hexdigest() for row in Xr])


def duplicate_flow_check(X_train: np.ndarray, X_test: np.ndarray) -> dict:
    h_tr = _row_hashes(X_train)
    h_te = _row_hashes(X_test)
    set_tr = set(h_tr.tolist())
    within_train = int(len(h_tr) - len(set_tr))
    within_test = int(len(h_te) - len(set(h_te.tolist())))
    cross = int(sum(1 for h in h_te if h in set_tr))
    return {"n_train": int(len(h_tr)), "n_test": int(len(h_te)),
            "duplicates_within_train": within_train, "duplicates_within_test": within_test,
            "test_rows_also_in_train": cross,
            "test_rows_also_in_train_share": float(cross / max(len(h_te), 1)),
            "interpretation": "test rows identical to a training row are memorisable, not generalisation"}


def run_all_audits(X_df: pd.DataFrame, y: np.ndarray, X_train: np.ndarray, X_test: np.ndarray,
                   evidence_status: str = "demo") -> dict:
    return {
        "evidence_status": evidence_status,
        "identifier_like_columns": identifier_like_columns(X_df),
        "placeholder_encoding": placeholder_encoding_audit(X_df, y),
        "duplicate_flows": duplicate_flow_check(X_train, X_test),
        "note": "Audits are heuristics with stated thresholds. On the authored fixture they are a "
                "demonstration; on real data they are proposed checks, not a verdict.",
    }

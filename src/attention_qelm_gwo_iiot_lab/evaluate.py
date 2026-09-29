"""Metrics with explicit class order.

FPR: for the binary task FP / (FP + TN) with Attack as the positive class. For the
multiclass task the macro average of one-vs-rest FPRs over the class order (the paper does
not define its multiclass FPR; audit item 10 in docs/paper-implementation-audit.md).
ROC: per-class one-vs-rest curves computed from actual scores, never from hard labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (confusion_matrix, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


@dataclass(frozen=True)
class RocCurve:
    class_name: str
    fpr: list[float]
    tpr: list[float]
    auc: float | None
    n_positive: int


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def one_vs_rest_fpr(cm: np.ndarray) -> list[float]:
    total = cm.sum()
    out = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        tn = total - tp - fp - fn
        out.append(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0)
    return out


def per_class_roc(y_true: np.ndarray, scores: np.ndarray, class_names: tuple[str, ...],
                  max_points: int = 200) -> list[RocCurve]:
    curves = []
    for c, name in enumerate(class_names):
        pos = (y_true == c).astype(int)
        n_pos = int(pos.sum())
        if n_pos == 0 or n_pos == len(pos):
            curves.append(RocCurve(name, [], [], None, n_pos))
            continue
        fpr, tpr, _ = roc_curve(pos, scores[:, c])
        if len(fpr) > max_points:
            keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).round().astype(int))
            fpr, tpr = fpr[keep], tpr[keep]
        auc = float(roc_auc_score(pos, scores[:, c]))
        curves.append(RocCurve(name, [float(v) for v in fpr], [float(v) for v in tpr], auc, n_pos))
    return curves


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None,
                    class_names: tuple[str, ...], task: str) -> dict:
    """Return a JSON-serialisable metrics dict. `scores` may be None (no ROC then)."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(class_names)
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (len(y_true), n):
            raise ValueError(f"scores must be (N, {n}) in class order, got {scores.shape}")
    labels = list(range(n))
    cm = confusion(y_true, y_pred, n)
    ovr_fpr = one_vs_rest_fpr(cm)
    if task == "binary":
        if n != 2:
            raise ValueError("binary task requires exactly two classes [Normal, Attack]")
        pos = 1
        fpr = ovr_fpr[pos]
        precision = float(precision_score(y_true, y_pred, pos_label=pos, zero_division=0))
        recall = float(recall_score(y_true, y_pred, pos_label=pos, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, pos_label=pos, zero_division=0))
    else:
        fpr = float(np.mean(ovr_fpr))
        precision = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        recall = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out: dict = {
        "task": task,
        "class_order": list(class_names),
        "n_samples": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "averaging": "binary (positive=Attack)" if task == "binary" else "macro",
        "fpr": float(fpr),
        "fpr_definition": "FP/(FP+TN), positive=Attack" if task == "binary" else "macro mean of one-vs-rest FP/(FP+TN)",
        "per_class_f1": {name: float(v) for name, v in zip(class_names, per_class_f1)},
        "per_class_support": {name: int(v) for name, v in zip(class_names, np.bincount(y_true, minlength=n))},
        "confusion_matrix": cm.tolist(),
    }
    if scores is not None:
        curves = per_class_roc(y_true, scores, class_names)
        out["per_class_auc"] = {c.class_name: c.auc for c in curves}
        present = [c.auc for c in curves if c.auc is not None]
        out["macro_auc"] = float(np.mean(present)) if present else None
        if task == "binary" and curves[1].auc is not None:
            out["auc"] = curves[1].auc
        out["roc_curves"] = [c.__dict__ for c in curves]
    return out

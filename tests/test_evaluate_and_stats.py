from __future__ import annotations

import numpy as np
import pytest

from attention_qelm_gwo_iiot_lab.baselines import run_baseline
from attention_qelm_gwo_iiot_lab.evaluate import compute_metrics, one_vs_rest_fpr, per_class_roc
from attention_qelm_gwo_iiot_lab.run_manifest import REQUIRED_KEYS, build_manifest, validate_manifest_file, write_manifest
from attention_qelm_gwo_iiot_lab.stats import aggregate_seeds, friedman_holm, holm_adjust


def test_multiclass_metrics_on_toy_with_fixed_class_order():
    names = ("Normal", "A", "B")
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 2, 2, 2])
    m = compute_metrics(y, p, None, names, "multiclass")
    assert m["class_order"] == list(names)
    assert m["accuracy"] == pytest.approx(5 / 6)
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 1, 1], [0, 0, 2]]
    assert m["per_class_f1"]["Normal"] == 1.0
    assert m["per_class_f1"]["A"] == pytest.approx(2 / 3)
    assert m["averaging"] == "macro"


def test_binary_fpr_is_fp_over_fp_plus_tn():
    y = np.array([0, 0, 0, 0, 1, 1])
    p = np.array([0, 1, 0, 0, 1, 0])
    m = compute_metrics(y, p, None, ("Normal", "Attack"), "binary")
    assert m["fpr"] == pytest.approx(1 / 4)
    assert m["recall"] == pytest.approx(1 / 2)
    assert m["fpr_definition"].startswith("FP/(FP+TN)")


def test_binary_task_rejects_multiclass_class_order():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1]), np.array([0, 1]), None, ("a", "b", "c"), "binary")


def test_one_vs_rest_fpr_from_confusion():
    cm = np.array([[8, 2], [1, 9]])
    assert one_vs_rest_fpr(cm) == pytest.approx([1 / 10, 2 / 10])


def test_per_class_roc_uses_scores_and_reports_auc():
    y = np.array([0, 0, 1, 1, 2, 2])
    scores = np.eye(3)[y] * 0.9 + 0.05                              # perfect ranking
    curves = per_class_roc(y, scores, ("x", "y", "z"))
    assert all(c.auc == pytest.approx(1.0) for c in curves)
    m = compute_metrics(y, y, scores, ("x", "y", "z"), "multiclass")
    assert m["macro_auc"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        compute_metrics(y, y, scores[:, :2], ("x", "y", "z"), "multiclass")


def test_roc_for_absent_class_is_none_not_invented():
    y = np.array([0, 0, 1, 1])
    scores = np.random.default_rng(0).uniform(size=(4, 3))
    curves = per_class_roc(y, scores, ("a", "b", "c"))
    assert curves[2].auc is None and curves[2].n_positive == 0


def test_aggregate_seeds_uses_sample_sd():
    agg = aggregate_seeds([0.8, 0.9, 1.0])
    assert agg["mean"] == pytest.approx(0.9)
    assert agg["sd"] == pytest.approx(0.1)                          # ddof=1
    assert aggregate_seeds([0.5])["sd"] is None


def test_holm_adjustment_monotone_and_capped():
    adj = holm_adjust([0.01, 0.04, 0.03])
    assert adj[0] == pytest.approx(0.03) and adj[2] == pytest.approx(0.06) and adj[1] == pytest.approx(0.06)
    assert all(a <= 1.0 for a in adj)


def test_friedman_refuses_fewer_than_two_methods():
    with pytest.raises(ValueError):
        friedman_holm({"only": [0.1, 0.2, 0.3]})


def test_friedman_refuses_fewer_than_three_seeds():
    with pytest.raises(ValueError):
        friedman_holm({"a": [0.1, 0.2], "b": [0.2, 0.3]})


def test_friedman_runs_with_three_methods_three_seeds():
    r = friedman_holm({"a": [0.90, 0.91, 0.92], "b": [0.80, 0.81, 0.79], "c": [0.85, 0.86, 0.84]})
    assert r["omnibus"]["test"] == "friedman" and r["n_seeds"] == 3
    assert len(r["pairwise"]) == 3 and all(0 <= p["p_holm"] <= 1 for p in r["pairwise"])
    assert r["mean_rank"]["a"] < r["mean_rank"]["c"] < r["mean_rank"]["b"]


def test_baselines_return_scores_in_class_order_and_reject_proposed():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    y = (X[:, 0] > 0).astype(int) + (X[:, 1] > 1).astype(int)       # classes 0,1,2
    for name in ("logistic_regression", "random_forest", "hist_gradient_boosting"):
        r = run_baseline(name, X[:150], y[:150], X[150:], 3, seed=0)
        assert r.scores.shape == (50, 3) and r.y_pred.shape == (50,)
        assert np.allclose(r.scores.sum(axis=1), 1.0)
    with pytest.raises(NotImplementedError):
        run_baseline("bilstm", X, y, X, 3)
    with pytest.raises(ValueError):
        run_baseline("svm", X, y, X, 3)


def test_run_manifest_schema_roundtrip(tmp_path):
    m = build_manifest("r1", "completed", "smoke", "demo", {"source_id": "synthetic-fixture-v1"}, "abc", 42,
                       None, None, "2026-09-27T00:00:00+00:00", "2026-09-27T00:00:01+00:00")
    assert set(REQUIRED_KEYS) <= set(m) and m["git_commit"] is None
    p = write_manifest(tmp_path / "run_manifest.yaml", m)
    assert validate_manifest_file(p)["project_id"] == "attention-qelm-gwo-iiot-lab"
    with pytest.raises(ValueError):
        build_manifest("r", "completed", "smoke", "reproduced-ish", {}, "h", 1, None, None, "a", "b")

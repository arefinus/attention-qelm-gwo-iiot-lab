from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attention_qelm_gwo_iiot_lab.cli import main
from attention_qelm_gwo_iiot_lab.config import load_run_config
from attention_qelm_gwo_iiot_lab.run_manifest import validate_manifest_file
from attention_qelm_gwo_iiot_lab.sensitivity import (duplicate_flow_check, identifier_like_columns,
                                                     placeholder_encoding_audit)
from attention_qelm_gwo_iiot_lab.train import reload_for_inference, run_training

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


def _tiny_cfg(repo_root: Path, task: str) -> dict:
    cfg = load_run_config(repo_root / "configs" / "smoke.yaml")
    cfg = json.loads(json.dumps(cfg))
    cfg["task"] = task
    cfg["encoder"].update({"max_epochs": 1, "batch_size": 1024})
    cfg["qelm"].update({"n_layers": 1, "n_hidden": 32})
    return cfg


def test_smoke_cli_runs_on_15_class_fixture_and_writes_valid_outputs(tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    out = tmp_path / "smoke"
    assert main(["smoke", "--out", str(out)]) == 0
    metrics = json.loads((out / "metrics.json").read_text())
    assert len(metrics["class_order"]) == 15 and metrics["class_order"][-1] == "MITM"
    assert len(metrics["confusion_matrix"]) == 15
    assert metrics["evidence_status"] == "demo"
    manifest = validate_manifest_file(out / "run_manifest.yaml")
    assert manifest["mode"] == "smoke" and manifest["data"]["source_id"] == "synthetic-fixture-v1"
    assert manifest["data"]["sample_counts"]["train_per_class"]["MITM"] >= 9
    trace = json.loads((out / "demo_trace.json").read_text())
    blocks = {s["block"] for s in trace["stages"]}
    assert blocks == {"attention_encoder", "qelm"}
    preds = pd.read_csv(out / "predictions.csv")
    assert {"y_true", "y_pred", "score_MITM"} <= set(preds.columns)


def test_binary_mode_maps_normal_to_zero_everything_else_to_one(tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    cfg = _tiny_cfg(repo_root, "binary")
    art = run_training(cfg, tmp_path / "bin", gwo_cfg=None, save_artifacts=False)
    assert list(art.bundle.class_spec.classes) == ["Normal", "Attack"]
    assert set(np.unique(art.bundle.y)) == {0, 1}
    assert (art.bundle.y[art.bundle.labels == "Normal"] == 0).all()
    assert (art.bundle.y[art.bundle.labels != "Normal"] == 1).all()
    assert art.qelm.n_outputs == 2 and art.encoder.n_classes == 2


def test_binary_and_multiclass_predictions_are_not_interchangeable(tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    p = tmp_path / "binary_predictions.csv"
    pd.DataFrame({"y_true": [0, 1], "y_pred": [0, 1], "score_Normal": [0.9, 0.1], "score_Attack": [0.1, 0.9]}).to_csv(p, index=False)
    assert main(["evaluate", "--predictions", str(p), "--config", "configs/demo_binary.yaml"]) == 0
    assert main(["evaluate", "--predictions", str(p), "--config", "configs/demo_multiclass.yaml"]) == 2


def test_evaluate_exits_2_when_predictions_missing(tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    assert main(["evaluate", "--predictions", str(tmp_path / "nope.csv"), "--config", "configs/demo_multiclass.yaml"]) == 2


def test_checkpoint_reload_reproduces_predictions(tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    cfg = _tiny_cfg(repo_root, "multiclass")
    art = run_training(cfg, tmp_path / "mc", gwo_cfg=None, save_artifacts=True)
    enc, qelm = reload_for_inference(tmp_path / "mc")
    H = enc.embed(art.X_test)
    assert np.allclose(H, art.H_test, atol=1e-5)
    assert np.array_equal(qelm.predict(H), art.qelm.predict(art.H_test))


def test_identifier_detector_flags_monotone_and_unique_columns():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"frame.number": np.arange(1, 501), "noise": rng.normal(size=500),
                       "uid": rng.permutation(500) * 7.0, "proto": rng.integers(0, 4, 500)})
    flagged = {f["column"]: f["reasons"] for f in identifier_like_columns(df)}
    assert "frame.number" in flagged and "monotone_in_row_order" in flagged["frame.number"]
    assert "uid" in flagged                                          # integer-valued and near-unique
    assert "noise" not in flagged                                    # continuous floats are not ids
    assert "proto" not in flagged


def test_placeholder_audit_flags_mask_that_predicts_label():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 900)
    leaky = np.where(y == 0, 0.0, rng.uniform(1, 5, 900))         # zero exactly for class 0
    clean = rng.uniform(0, 1, 900)
    clean[rng.uniform(size=900) < 0.3] = 0.0                       # zeros unrelated to label
    df = pd.DataFrame({"leaky": leaky, "clean": clean})
    flagged = {f["column"] for f in placeholder_encoding_audit(df, y)}
    assert flagged == {"leaky"}


def test_duplicate_flow_check_counts_cross_split_duplicates():
    rng = np.random.default_rng(0)
    tr = rng.normal(size=(100, 5))
    te = np.vstack([rng.normal(size=(20, 5)), tr[:7]])
    r = duplicate_flow_check(tr, te)
    assert r["test_rows_also_in_train"] == 7 and r["duplicates_within_train"] == 0


def test_audit_cli_on_fixture_finds_frame_number_and_structural_zeros(tmp_path, repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    assert main(["audit", "--out", str(tmp_path)]) == 0
    audit = json.loads((tmp_path / "audit.json").read_text())
    assert audit["evidence_status"] == "demo"
    assert "frame.number" in {c["column"] for c in audit["identifier_like_columns"]}
    assert "feature_40" in {c["column"] for c in audit["placeholder_encoding"]}
    assert audit["duplicate_flows"]["test_rows_also_in_train"] >= 1


def test_paper_reported_evidence_file_is_labelled_and_located(repo_root):
    ev = json.loads((repo_root / "evidence" / "paper_reported" / "qelm.json").read_text(encoding="utf-8"))
    assert ev["evidence_status"] == "paper-reported"
    assert ev["doi"] == "10.1007/s44163-026-01704-3"
    for section in ("binary_table6", "multiclass_table7", "mitm_table8", "ablations_table11", "cost_table12"):
        block = ev["results"][section]
        assert block["source_locator"] and block["evidence_status"] == "paper-reported"
        assert block["values"]
    assert "50.00" in ev["notes"]["friedman_inconsistency"] and "54.98" in ev["notes"]["friedman_inconsistency"]


def test_docs_have_no_em_dashes_or_marketing_adjectives(repo_root):
    files = [repo_root / "README.md", repo_root / "DATA_CARD.md", repo_root / "MODEL_CARD.md",
             repo_root / "REPRODUCIBILITY.md", *sorted((repo_root / "docs").glob("*.md"))]
    banned = re.compile(r"state[- ]of[- ]the[- ]art|\brobust\b|\bnovel\b|\bpowerful\b|cutting-edge|world-leading|production-ready|\bguaranteed\b", re.I)
    for f in files:
        text = f.read_text(encoding="utf-8")
        assert "2014" not in text, f"em dash in {f.name}"
        assert not banned.search(text), f"marketing adjective in {f.name}: {banned.search(text).group(0)}"


def test_readme_states_paper_relationship_and_contribution_wording(repo_root):
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "Contribution: conceptualization, methodology design, implementation, and initial framework development." in text
    assert "10.1007/s44163-026-01704-3" in text
    assert "Md Sultanul Arefin Sourav" in text
    assert "demo_ready" in text

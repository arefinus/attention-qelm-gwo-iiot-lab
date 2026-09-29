from __future__ import annotations

import hashlib
import json
import runpy
import sys

import numpy as np
import pandas as pd
import pytest

from attention_qelm_gwo_iiot_lab.config import ConfigError, config_hash
from attention_qelm_gwo_iiot_lab.data import load_classes, load_edge_iiotset, load_manifest, validate_frame
from attention_qelm_gwo_iiot_lab.data.adapter import SchemaError

TABLE1_ORDER = ["Normal", "DDoS_UDP", "DDoS_ICMP", "SQL_Injection", "DDoS_TCP", "DDoS_HTTP", "Password",
                "Vulnerability_Scanner", "Port_Scanning", "XSS", "Uploading", "Backdoor", "OS_Fingerprinting",
                "Ransomware", "MITM"]


def test_class_order_is_table1_normal_first_mitm_last(classes):
    assert list(classes.classes) == TABLE1_ORDER
    assert classes.classes[0] == "Normal" and classes.classes[-1] == "MITM"
    assert classes.n_classes == 15


def test_binary_spec_is_separate_two_class_order(binary_classes):
    assert binary_classes.task == "binary"
    assert list(binary_classes.classes) == ["Normal", "Attack"]


def test_manifest_has_48_unique_named_slots(manifest):
    assert manifest.n_features == 48
    assert len(set(manifest.names)) == 48
    assert manifest.names[0] == "feature_01" and manifest.names[-1] == "feature_48"
    assert "frame.number" in manifest.drop_columns


def test_manifest_rejects_declared_count_mismatch(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text("n_features: 3\nfeatures:\n  - name: a\n  - name: b\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_manifest(p)


def test_validate_rejects_47_columns_and_names_the_missing_one(toy_frame, manifest, classes):
    df = toy_frame.drop(columns=["feature_17"])
    with pytest.raises(SchemaError) as e:
        validate_frame(df, manifest, classes)
    assert "feature_17" in str(e.value)
    assert "never truncates" in str(e.value)


def test_validate_rejects_extra_undeclared_column_no_silent_truncation(toy_frame, manifest, classes):
    df = toy_frame.copy()
    df["feature_49"] = 1.0
    with pytest.raises(SchemaError) as e:
        validate_frame(df, manifest, classes)
    assert "feature_49" in str(e.value)


def test_validate_drops_declared_metadata_columns(toy_frame, manifest, classes):
    df = toy_frame.copy()
    df.insert(0, "frame.number", range(len(df)))
    out = validate_frame(df, manifest, classes)
    assert "frame.number" not in out.columns and out.shape[1] == 49


def test_validate_rejects_unknown_label(toy_frame, manifest, classes):
    df = toy_frame.copy()
    df.loc[0, "label"] = "Not_A_Class"
    with pytest.raises(SchemaError) as e:
        validate_frame(df, manifest, classes)
    assert "Not_A_Class" in str(e.value)


def test_edge_iiotset_adapter_fails_actionably_without_data(manifest, classes, monkeypatch, tmp_path):
    monkeypatch.delenv("DATA_ROOT", raising=False)
    with pytest.raises(FileNotFoundError) as e:
        load_edge_iiotset(None, manifest, classes)
    assert "Kaggle" in str(e.value) or "IEEE Dataport" in str(e.value)
    with pytest.raises(FileNotFoundError):
        load_edge_iiotset(tmp_path, manifest, classes)


def test_fixture_shape_imbalance_and_rare_class(fixture_bundle, classes):
    y = fixture_bundle.y
    assert fixture_bundle.X.shape == (len(y), 48)
    counts = np.bincount(y, minlength=15)
    assert (counts > 0).all()
    assert counts[0] / len(y) > 0.65                       # Normal dominates as in Table 1
    assert counts[classes.index_of("MITM")] >= 12          # rarest class floor
    assert counts[classes.index_of("MITM")] / len(y) < 0.005


def test_fixture_labels_map_to_class_indices(fixture_bundle, classes):
    for i, name in enumerate(classes.classes):
        mask = fixture_bundle.labels == name
        assert (fixture_bundle.y[mask] == i).all()


def test_fixture_generator_is_deterministic(tmp_path, repo_root):
    script = repo_root / "tools" / "make_fixture.py"
    saved = sys.argv
    try:
        sys.argv = [str(script), "--out", str(tmp_path), "--seed", "42"]
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as e:
            assert int(e.code or 0) == 0
    finally:
        sys.argv = saved
    new = hashlib.sha256((tmp_path / "fixture.csv").read_bytes()).hexdigest()
    committed = json.loads((repo_root / "examples" / "fixtures" / "v1" / "fixture_manifest.json").read_text())
    assert new == committed["sha256_fixture_csv"]
    assert (tmp_path / "FIXTURE.md").read_text(encoding="utf-8").startswith("# Fixture v1")


def test_fixture_has_exact_duplicates_for_the_audit(fixture_bundle):
    dup = fixture_bundle.X.round(6).duplicated().sum()
    assert dup >= 25


def test_config_hash_is_deterministic_and_order_insensitive():
    a = {"x": 1, "y": {"b": 2, "a": [1, 2]}}
    b = {"y": {"a": [1, 2], "b": 2}, "x": 1}
    assert config_hash(a) == config_hash(b)
    assert config_hash({"x": 2}) != config_hash(a)


def test_load_classes_rejects_bad_binary_block(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("classes: [Normal, MITM]\nbinary: {normal_class: Normal, classes: [Attack, Normal]}\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_classes(p, task="binary")

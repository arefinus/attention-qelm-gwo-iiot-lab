from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from attention_qelm_gwo_iiot_lab.config import set_cpu_threads  # noqa: E402
from attention_qelm_gwo_iiot_lab.data import load_classes, load_fixture, load_manifest  # noqa: E402

set_cpu_threads(2)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO


@pytest.fixture(scope="session")
def manifest():
    return load_manifest(REPO / "configs" / "feature_manifest_paper.yaml")


@pytest.fixture(scope="session")
def classes():
    return load_classes(REPO / "configs" / "classes.yaml")


@pytest.fixture(scope="session")
def binary_classes():
    return load_classes(REPO / "configs" / "classes.yaml", task="binary")


@pytest.fixture(scope="session")
def fixture_bundle(manifest, classes):
    return load_fixture(REPO / "examples" / "fixtures" / "v1", manifest, classes)


@pytest.fixture(scope="session")
def small_bundle(fixture_bundle):
    """Stratified 1,500-row subsample of the fixture for fast model tests (every class kept)."""
    rng = np.random.default_rng(0)
    y = fixture_bundle.y
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        n = max(12, int(round(len(idx) * 1500 / len(y))))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return fixture_bundle.X.iloc[keep].reset_index(drop=True), y[keep]


@pytest.fixture
def toy_frame(manifest) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(rng.normal(size=(60, 48)), columns=list(manifest.names))
    df["label"] = ["Normal"] * 40 + ["MITM"] * 20
    return df

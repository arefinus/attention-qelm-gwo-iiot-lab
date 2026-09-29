"""Edge-IIoTset adapter and fixture loader.

The adapter never guesses. A CSV is accepted only when its columns are exactly the
manifest's 48 feature slots plus the label column (declared metadata columns are dropped
first). Any other column, or any missing column, is an error naming the offending
columns. The 48 slots in configs/feature_manifest_paper.yaml are placeholders until the
user fills them from their own preprocessing; see the header of that file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .manifest import ClassSpec, FeatureManifest

OBTAIN_MESSAGE = (
    "Edge-IIoTset is not shipped with this repository. Obtain it from the Kaggle page "
    "'mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot' or IEEE Dataport "
    "(doi 10.21227/mbc1-1h68), apply your own preprocessing to produce a 48-feature table, "
    "fill configs/feature_manifest_paper.yaml with the resulting column names, and point "
    "--data-root (or DATA_ROOT) at the directory containing the processed CSV."
)


class SchemaError(ValueError):
    """Raised when a table does not match the feature manifest or class list."""


@dataclass(frozen=True)
class DataFrameBundle:
    X: pd.DataFrame
    y: np.ndarray            # integer class indices in ClassSpec order
    labels: np.ndarray       # original label strings
    feature_names: tuple[str, ...]
    class_spec: ClassSpec
    source_id: str


def validate_frame(df: pd.DataFrame, manifest: FeatureManifest, spec: ClassSpec,
                   label_column: str | None = None) -> pd.DataFrame:
    """Validate columns and labels; return the frame with metadata columns removed.

    Raises SchemaError listing missing and unexpected columns. Never truncates.
    """
    label_col = label_column or manifest.label_column
    present = list(df.columns)
    dropped = [c for c in manifest.drop_columns if c in present]
    df = df.drop(columns=dropped)
    cols = set(df.columns)
    if label_col not in cols:
        raise SchemaError(f"label column {label_col!r} not found; columns present: {sorted(cols)[:10]}...")
    feature_cols = cols - {label_col}
    expected = set(manifest.names)
    missing = sorted(expected - feature_cols)
    unexpected = sorted(feature_cols - expected)
    if missing or unexpected:
        raise SchemaError(
            "CSV feature columns do not match the 48-slot manifest. "
            f"Missing {len(missing)}: {missing[:8]}{'...' if len(missing) > 8 else ''}. "
            f"Unexpected {len(unexpected)}: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}. "
            "This adapter never truncates or reorders columns by position; fill "
            "configs/feature_manifest_paper.yaml with your processed column names."
        )
    unknown_labels = sorted(set(df[label_col].astype(str).unique()) - set(spec.classes))
    if unknown_labels:
        raise SchemaError(
            f"labels not in configs/classes.yaml: {unknown_labels[:8]}. "
            f"Allowed ({spec.task}): {list(spec.classes)}"
        )
    return df


def _bundle(df: pd.DataFrame, manifest: FeatureManifest, spec: ClassSpec,
            source_id: str, label_column: str | None = None) -> DataFrameBundle:
    label_col = label_column or manifest.label_column
    labels = df[label_col].astype(str).to_numpy()
    if spec.task == "binary":
        y = np.where(labels == spec.normal_class, 0, 1).astype(np.int64)
    else:
        lut = {c: i for i, c in enumerate(spec.classes)}
        y = np.asarray([lut[v] for v in labels], dtype=np.int64)
    X = df.loc[:, list(manifest.names)].reset_index(drop=True)
    return DataFrameBundle(X=X, y=y, labels=labels, feature_names=manifest.names,
                           class_spec=spec, source_id=source_id)


def load_edge_iiotset(data_root: str | Path | None, manifest: FeatureManifest, spec: ClassSpec,
                      filename: str = "edge_iiotset_processed.csv") -> DataFrameBundle:
    """Read a processed Edge-IIoTset CSV from `data_root` and validate it.

    The binary task validates labels against the FULL 15-class list (the raw label column
    holds attack names) and then maps Normal -> 0, everything else -> 1.
    """
    root = data_root or os.environ.get("DATA_ROOT")
    if not root:
        raise FileNotFoundError("No --data-root given and DATA_ROOT is unset. " + OBTAIN_MESSAGE)
    path = Path(root) / filename
    if not path.exists():
        raise FileNotFoundError(f"Expected {path} to exist. " + OBTAIN_MESSAGE)
    df = pd.read_csv(path, low_memory=False)
    full_spec = spec if spec.task == "multiclass" else _full_spec_for(spec)
    df = validate_frame(df, manifest, full_spec)
    return _bundle(df, manifest, spec, source_id="edge-iiotset-processed")


def _full_spec_for(binary_spec: ClassSpec) -> ClassSpec:
    from ..config import resolve_path
    from .manifest import load_classes
    return load_classes(resolve_path("configs/classes.yaml"), task="multiclass")


def load_fixture(fixture_dir: str | Path, manifest: FeatureManifest, spec: ClassSpec,
                 filename: str = "fixture.csv") -> DataFrameBundle:
    """Load the authored synthetic fixture with the same validation as real data."""
    path = Path(fixture_dir) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Fixture not found at {path}. Generate it with "
            "`python tools/make_fixture.py --out examples/fixtures/v1 --seed 42`."
        )
    df = pd.read_csv(path)
    full_spec = spec if spec.task == "multiclass" else _full_spec_for(spec)
    df = validate_frame(df, manifest, full_spec)
    return _bundle(df, manifest, spec, source_id="synthetic-fixture-v1")


def load_dataset(cfg: dict, manifest: FeatureManifest, spec: ClassSpec,
                 data_root: str | Path | None = None) -> DataFrameBundle:
    """Dispatch on cfg['data']['source']: 'fixture' or 'data-root'."""
    from ..config import resolve_path
    source = cfg["data"].get("source", "fixture")
    if source == "fixture":
        return load_fixture(resolve_path(cfg["data"]["fixture_dir"]), manifest, spec)
    if source == "data-root":
        return load_edge_iiotset(data_root, manifest, spec)
    raise ValueError(f"unknown data.source {source!r}; use 'fixture' or 'data-root'")

"""Feature manifest and class-order specifications (schema constants)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..config import ConfigError, load_yaml

VALID_DTYPES = ("numeric", "categorical")


@dataclass(frozen=True)
class FeatureManifest:
    names: tuple[str, ...]
    dtypes: tuple[str, ...]
    label_column: str
    drop_columns: tuple[str, ...]
    source: str

    @property
    def n_features(self) -> int:
        return len(self.names)

    def dtype_of(self, name: str) -> str:
        return self.dtypes[self.names.index(name)]

    @property
    def categorical(self) -> tuple[str, ...]:
        return tuple(n for n, d in zip(self.names, self.dtypes) if d == "categorical")

    @property
    def numeric(self) -> tuple[str, ...]:
        return tuple(n for n, d in zip(self.names, self.dtypes) if d == "numeric")


@dataclass(frozen=True)
class ClassSpec:
    task: str
    classes: tuple[str, ...]
    label_column: str
    normal_class: str

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    def index_of(self, name: str) -> int:
        return self.classes.index(name)

    def to_binary(self) -> "ClassSpec":
        """Return the separate binary specification (Normal vs Attack)."""
        return ClassSpec(task="binary", classes=("Normal", "Attack"),
                         label_column=self.label_column, normal_class=self.normal_class)


def load_manifest(path: str | Path) -> FeatureManifest:
    raw = load_yaml(path)
    feats = raw.get("features")
    if not isinstance(feats, list) or not feats:
        raise ConfigError("feature manifest must contain a non-empty `features` list")
    names: list[str] = []
    dtypes: list[str] = []
    for i, item in enumerate(feats):
        if not isinstance(item, dict) or "name" not in item:
            raise ConfigError(f"feature manifest entry {i} must be a mapping with a `name`")
        dtype = str(item.get("dtype", "numeric"))
        if dtype not in VALID_DTYPES:
            raise ConfigError(f"feature {item['name']!r} has dtype {dtype!r}; allowed {VALID_DTYPES}")
        names.append(str(item["name"]))
        dtypes.append(dtype)
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ConfigError(f"feature manifest has duplicate names: {dupes}")
    declared = int(raw.get("n_features", len(names)))
    if declared != len(names):
        raise ConfigError(f"manifest declares n_features={declared} but lists {len(names)} features")
    return FeatureManifest(
        names=tuple(names),
        dtypes=tuple(dtypes),
        label_column=str(raw.get("label_column", "label")),
        drop_columns=tuple(str(c) for c in raw.get("drop_columns", []) or []),
        source=str(raw.get("source", "unknown")),
    )


def load_classes(path: str | Path, task: str = "multiclass") -> ClassSpec:
    raw = load_yaml(path)
    classes = raw.get("classes")
    if not isinstance(classes, list) or len(classes) < 2:
        raise ConfigError("classes.yaml must list at least two classes")
    if len(set(classes)) != len(classes):
        raise ConfigError("classes.yaml contains duplicate class names")
    binary = raw.get("binary", {}) or {}
    normal = str(binary.get("normal_class", "Normal"))
    if normal not in classes:
        raise ConfigError(f"binary.normal_class {normal!r} is not in the class list")
    spec = ClassSpec(task="multiclass", classes=tuple(str(c) for c in classes),
                     label_column=str(raw.get("label_column", "label")), normal_class=normal)
    if task == "binary":
        bin_classes = tuple(str(c) for c in binary.get("classes", ["Normal", "Attack"]))
        if bin_classes != ("Normal", "Attack"):
            raise ConfigError("binary.classes must be exactly [Normal, Attack]")
        return spec.to_binary()
    if task != "multiclass":
        raise ConfigError(f"unknown task {task!r}")
    return spec

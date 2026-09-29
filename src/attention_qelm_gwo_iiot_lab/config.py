"""Configuration loading, path resolution and content hashing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised when a configuration file is missing or malformed."""


def resolve_path(path: str | Path) -> Path:
    """Resolve a config-relative path.

    Absolute paths are returned unchanged. Relative paths are tried against the current
    working directory first and then against the repository root, so the CLI works from
    the repo checkout and from any other directory after `pip install -e .`.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return REPO_ROOT / p


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    if not p.exists():
        raise ConfigError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ConfigError(f"Config file {p} must contain a mapping at top level")
    return data


def load_run_config(path: str | Path) -> dict[str, Any]:
    """Load a training config and validate the sections every stage relies on."""
    cfg = load_yaml(path)
    required = ("name", "task", "seed", "classes_file", "manifest_file", "data",
                "preprocess", "encoder", "qelm", "search", "evaluation")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ConfigError(f"Config {path} is missing sections: {missing}")
    if cfg["task"] not in ("multiclass", "binary"):
        raise ConfigError(f"task must be 'multiclass' or 'binary', got {cfg['task']!r}")
    return cfg


def set_cpu_threads(n: int | None = None) -> int:
    """Cap torch and BLAS thread pools.

    The models here are tiny (tens of thousands of parameters, 8 tokens), so per-op
    overhead dominates and wide OpenMP pools slow them down, badly so on a loaded machine.
    Default is min(2, cpu_count); override with the AQG_THREADS environment variable.
    """
    import os
    if n is None:
        env = os.environ.get("AQG_THREADS")
        n = int(env) if env else min(2, os.cpu_count() or 1)
    n = max(1, int(n))
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:  # torch absent or already configured elsewhere
        pass
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=n)
    except Exception:
        pass
    return n


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serialisable")


def config_hash(cfg: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON form of a configuration mapping."""
    return hashlib.sha256(canonical_json(dict(cfg)).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, indent=2, sort_keys=False, default=_json_default)
        fh.write("\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)

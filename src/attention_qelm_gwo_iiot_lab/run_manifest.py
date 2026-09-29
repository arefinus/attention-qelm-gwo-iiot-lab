"""Run manifest writer (repository contract, section 5)."""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import PROJECT_ID

REQUIRED_KEYS = ("schema_version", "project_id", "run_id", "status", "mode", "evidence_status",
                 "git_commit", "data", "configuration_hash", "seed", "environment", "metrics_file",
                 "predictions_file", "started_at", "finished_at")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(slug: str) -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + slug


def build_manifest(run_id: str, status: str, mode: str, evidence_status: str, data: dict,
                   configuration_hash: str, seed: int, metrics_file: str | None,
                   predictions_file: str | None, started_at: str, finished_at: str) -> dict:
    if mode not in ("smoke", "demo", "experiment"):
        raise ValueError("mode must be smoke, demo or experiment")
    if evidence_status not in ("demo", "reproduced"):
        raise ValueError("evidence_status must be demo or reproduced")
    if status not in ("completed", "failed"):
        raise ValueError("status must be completed or failed")
    return {
        "schema_version": 1,
        "project_id": PROJECT_ID,
        "run_id": run_id,
        "status": status,
        "mode": mode,
        "evidence_status": evidence_status,
        "git_commit": None,
        "data": data,
        "configuration_hash": configuration_hash,
        "seed": int(seed),
        "environment": {"python": platform.python_version(), "device": "cpu"},
        "metrics_file": metrics_file,
        "predictions_file": predictions_file,
        "started_at": started_at,
        "finished_at": finished_at,
    }


def write_manifest(path: str | Path, manifest: dict) -> Path:
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"run manifest missing keys: {missing}")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, default_flow_style=False)
    return p


def validate_manifest_file(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        m = yaml.safe_load(fh)
    missing = [k for k in REQUIRED_KEYS if k not in m]
    if missing:
        raise ValueError(f"run manifest at {path} missing keys: {missing}")
    return m

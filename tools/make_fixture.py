"""Generate the authored synthetic fixture (examples/fixtures/v1).

Authored synthetic demonstration data. Fictional identifiers. Not derived from any person,
company, account or dataset. The class PROPORTIONS mirror the imbalance reported in the
source paper's Table 1 (Normal about 72%, MITM about 0.1%), but every value is drawn from
simple parametric distributions chosen here; nothing is sampled from Edge-IIoTset.

Structure (48 features, seed 42, about 9,000 rows):
  feature_01..16  class-shifted Gaussians (learnable signal, deliberately overlapping)
  feature_17..24  integer-coded protocol-like fields, class-dependent categorical mix
  feature_25..32  heavy-tailed positive magnitudes (lognormal), class-dependent scale
  feature_33..40  pure noise, one column (feature_40) structurally zero for DDoS classes
                  so the placeholder audit has something to find
  feature_41..48  interactions of earlier columns plus noise
Also: about 1.5% NaN in six columns, a handful of +inf in one column, roughly 30 exact
duplicate rows, and a synthetic `frame.number` metadata column the adapter drops.

Usage: python tools/make_fixture.py --out examples/fixtures/v1 --seed 42
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from attention_qelm_gwo_iiot_lab.data.manifest import load_classes  # noqa: E402

# Paper Table 1 processed-row counts, used ONLY to derive proportions.
PAPER_COUNTS = {
    "Normal": 1_367_959, "DDoS_UDP": 120_532, "DDoS_ICMP": 67_391, "SQL_Injection": 50_421,
    "DDoS_TCP": 49_651, "DDoS_HTTP": 48_693, "Password": 38_217, "Vulnerability_Scanner": 31_482,
    "Port_Scanning": 28_904, "XSS": 22_734, "Uploading": 19_856, "Backdoor": 17_423,
    "OS_Fingerprinting": 14_381, "Ransomware": 11_293, "MITM": 1_734,
}
N_ROWS = 9_000
MIN_RARE_ROWS = 12
N_FEATURES = 48
FIXTURE_VERSION = "v1"


def class_counts(classes: tuple[str, ...], n_rows: int) -> dict[str, int]:
    total = sum(PAPER_COUNTS.values())
    counts = {c: int(np.floor(PAPER_COUNTS[c] / total * n_rows)) for c in classes}
    counts["MITM"] = max(counts["MITM"], MIN_RARE_ROWS)
    counts["Normal"] += n_rows - sum(counts.values())
    assert sum(counts.values()) == n_rows
    return counts


def generate(seed: int, classes: tuple[str, ...], n_rows: int = N_ROWS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    counts = class_counts(classes, n_rows)
    C = len(classes)
    y = np.concatenate([np.full(counts[c], i, dtype=np.int64) for i, c in enumerate(classes)])
    rng.shuffle(y)
    n = len(y)
    X = np.zeros((n, N_FEATURES), dtype=np.float64)

    # 1..16 class-shifted Gaussians
    proto = rng.normal(0.0, 1.2, size=(C, 16))
    X[:, 0:16] = proto[y] + rng.normal(0.0, 1.0, size=(n, 16))

    # 17..24 integer-coded protocol-like fields
    for j in range(16, 24):
        base = rng.dirichlet(np.ones(6) * 0.7, size=C)
        for c in range(C):
            idx = np.where(y == c)[0]
            X[idx, j] = rng.choice(6, size=len(idx), p=base[c])

    # 25..32 heavy-tailed magnitudes
    scale = rng.uniform(0.3, 1.5, size=(C, 8))
    X[:, 24:32] = np.exp(rng.normal(np.log(scale[y] * 100.0), 0.6))

    # 33..40 noise; feature_40 structurally zero for DDoS classes
    X[:, 32:40] = rng.normal(0.0, 1.0, size=(n, 8))
    X[:, 39] = np.abs(X[:, 39]) * 50.0
    ddos = np.isin(y, [classes.index(c) for c in classes if c.startswith("DDoS")])
    X[ddos, 39] = 0.0

    # 41..48 interactions
    X[:, 40] = X[:, 0] * X[:, 1] + rng.normal(0, 0.5, n)
    X[:, 41] = np.sin(X[:, 2]) + X[:, 3] ** 2 * 0.3 + rng.normal(0, 0.5, n)
    X[:, 42] = np.log1p(X[:, 24]) - 0.2 * X[:, 4] + rng.normal(0, 0.5, n)
    X[:, 43] = (X[:, 16] == X[:, 17]).astype(float) + rng.normal(0, 0.3, n)
    X[:, 44] = X[:, 5] - X[:, 6] + rng.normal(0, 0.5, n)
    X[:, 45] = np.maximum(X[:, 7], X[:, 8]) + rng.normal(0, 0.5, n)
    X[:, 46] = X[:, 25] / (X[:, 26] + 1.0) + rng.normal(0, 0.2, n)
    X[:, 47] = rng.normal(0, 1, n) + 0.5 * X[:, 9]

    cols = [f"feature_{i:02d}" for i in range(1, N_FEATURES + 1)]
    df = pd.DataFrame(X, columns=cols)
    for j in range(16, 24):
        df[cols[j]] = df[cols[j]].astype(int)

    # missingness and infinities (exercise imputation and the inf policy)
    for j in (3, 11, 27, 34, 41, 46):
        mask = rng.uniform(size=n) < 0.015
        df.loc[mask, cols[j]] = np.nan
    inf_idx = rng.choice(n, size=8, replace=False)
    df.loc[inf_idx, cols[29]] = np.inf

    # exact duplicate rows
    dup_src = rng.choice(n, size=30, replace=False)
    df = pd.concat([df, df.iloc[dup_src]], ignore_index=True)
    y = np.concatenate([y, y[dup_src]])

    df.insert(0, "frame.number", np.arange(1, len(df) + 1))   # synthetic metadata, dropped by adapter
    df["label"] = [classes[i] for i in y]
    return df


def write_fixture(out_dir: Path, seed: int) -> dict:
    spec = load_classes(REPO_ROOT / "configs" / "classes.yaml")
    df = generate(seed, spec.classes)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "fixture.csv"
    df.to_csv(csv_path, index=False, float_format="%.6g", lineterminator="\n")
    sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    counts = df["label"].value_counts().reindex(list(spec.classes)).fillna(0).astype(int)
    meta = {
        "fixture_version": FIXTURE_VERSION,
        "generator": "tools/make_fixture.py",
        "seed": seed,
        "n_rows": int(len(df)),
        "n_features": N_FEATURES,
        "class_counts": {k: int(v) for k, v in counts.items()},
        "sha256_fixture_csv": sha,
        "statement": "Authored synthetic demonstration data. Fictional identifiers. Not derived from "
                     "any person, company, account or dataset.",
    }
    (out_dir / "fixture_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    fixture_md = (
        "# Fixture v1\n\n"
        "Authored synthetic demonstration data. Fictional identifiers. Not derived from any person, "
        "company, account or dataset. Generated by `tools/make_fixture.py` seed "
        f"{seed}, version v1.\n\n"
        f"Rows: {len(df)} (including 30 deliberate exact duplicates). Features: {N_FEATURES} placeholder "
        "slots `feature_01..feature_48` plus a synthetic `frame.number` metadata column that the "
        "adapter drops, and a `label` column with the 15 class names from `configs/classes.yaml`.\n\n"
        "Class proportions mirror the imbalance of the source paper's Table 1 (rarest class MITM held at "
        f"{MIN_RARE_ROWS} rows minimum so stratified folds are possible). The VALUES are parametric noise "
        "chosen in the generator; nothing is sampled from Edge-IIoTset. Metrics computed on this fixture "
        "are demonstrations of the pipeline, never benchmark results.\n\n"
        "Regenerate: `python tools/make_fixture.py --out examples/fixtures/v1 --seed 42`\n\n"
        f"SHA-256 of fixture.csv: `{sha}`\n"
    )
    (out_dir / "FIXTURE.md").write_text(fixture_md, encoding="utf-8")
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(REPO_ROOT / "examples" / "fixtures" / "v1"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    meta = write_fixture(Path(args.out), args.seed)
    print(json.dumps({k: meta[k] for k in ("n_rows", "n_features", "seed", "sha256_fixture_csv")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

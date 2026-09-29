# Reproducibility

## What this repository reproduces
Nothing from the source paper is reproduced here. Every number this repository produces
comes from the authored synthetic fixture (`examples/fixtures/v1/`, seed 42) and carries the
evidence label `demo`. Paper-reported values live only in `evidence/paper_reported/` with a
source locator and are never used as a test oracle.

## Framework port
The paper reports TensorFlow 2.11. This repository is a PyTorch (CPU) port of the encoder
and a NumPy/SciPy implementation of the fixed-weight ELM and the grey wolf optimizer.
Differences that can change numbers: optimizer default epsilon, weight initialisation of
the projection layers, the GELU implementation (exact vs tanh approximation), and thread
scheduling. None of these is claimed to be equivalent to the original code.

## Exact commands
```
pip install -e . && pip install pytest      # or prefix commands with PYTHONPATH=src
python -m attention_qelm_gwo_iiot_lab make-fixture
python -m attention_qelm_gwo_iiot_lab smoke
python -m attention_qelm_gwo_iiot_lab demo
python -m attention_qelm_gwo_iiot_lab train --config configs/demo_multiclass.yaml --out examples/output/train_multiclass
python -m attention_qelm_gwo_iiot_lab search --config configs/demo_multiclass.yaml --gwo configs/gwo_demo.yaml --out examples/output/search
python -m attention_qelm_gwo_iiot_lab evaluate --predictions examples/output/predictions.csv --config configs/demo_multiclass.yaml
python -m attention_qelm_gwo_iiot_lab audit --out examples/output/audit
python -m pytest -q
```
Set `OMP_NUM_THREADS=1` for bit-identical reruns on CPU.

## Seeds and determinism
Fixture seed 42. The multi-seed runner uses {7, 13, 21, 42, 99}, the paper's seed set, on
the fixture only. GWO demo settings are small (see `configs/gwo_demo.yaml`); the paper's
settings (population 20, 50 iterations, five folds) are recorded in the configs as
settings, not as something executed here.

## Run manifest
Every run writes `run_manifest.yaml` with mode, evidence status, fixture hash,
configuration hash, seed, sample counts and tensor shapes. `git_commit` is null until the
publishing tooling fills it.

## Running on real data
`--data-root` must point at a processed Edge-IIoTset table whose 48 columns match
`configs/feature_manifest_paper.yaml`; the manifest names are placeholders that the user
fills from their own preprocessing. The loader refuses any other column count. A full
paper-scale run needs the paper configs and many CPU hours or a GPU; results are
`reproduced` only with the manifest, data hash and config hash retained.

## Environment
Python 3.10 locally; CI runs 3.10, 3.11 and 3.12 on Ubuntu with torch CPU wheels.
TensorFlow is not a dependency.

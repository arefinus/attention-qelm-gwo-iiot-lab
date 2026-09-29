# Contributing

Thank you for looking at this repository.

## Ground rules

- Every metric printed or written by this code must come from a script in this repository
  running in the same session, or be a paper-reported value stored under
  `evidence/paper_reported/` with a `source_locator`. Do not add numbers to prose.
- No dataset rows are committed. Real-data adapters read `--data-root` and validate the
  schema against `configs/feature_manifest_paper.yaml`.
- Preprocessing is fit on the training partition only. Model selection uses inner
  validation folds only. The test partition is evaluated once.
- Keep modules under 300 lines, typed, with `from __future__ import annotations`.
- No `eval`, `exec`, or pickle of untrusted input. Artifacts are `.npz`, JSON, YAML, or
  `torch.save` of a state dict loaded with `weights_only=True`.

## Workflow

1. `pip install -e . && pip install pytest`
2. `python -m attention_qelm_gwo_iiot_lab smoke`
3. `python -m pytest -q`
4. Open a pull request with a short description of the behaviour change and the test
   that covers it.

## Reporting a fidelity concern

If you believe this implementation departs from the source paper in a way not listed in
`docs/paper-implementation-audit.md`, open an issue quoting the paper section and the
code location. Fidelity items are tracked as numbered audit entries.

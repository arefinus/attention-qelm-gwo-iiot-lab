# Evaluation

## Protocol
Stratified 80/20 outer split of the fixture; five-fold stratified inner cross-validation
inside the training partition for grey wolf fitness (macro-F1) and for early stopping of
the encoder. The outer test partition is touched once by `evaluate`. Binary and
multiclass runs use separate configs and are never collapsed into each other. Metrics:
accuracy, macro precision, recall and F1, per-class F1 in the fixed class order, false
positive rate, confusion matrix, one-vs-rest ROC from actual scores, and a multi-seed
runner over {7, 13, 21, 42, 99} reporting mean and sample standard deviation.

## Where the numbers are
`examples/output/metrics.json`, `predictions.csv`, `roc_curves.json`, `learning_curves.json`,
`gwo_trace.json` and `audit.json` after `demo`; `examples/output/train_binary/` and
`train_multiclass/` after `train`. Every value is `demo`, computed on synthetic fixture v1
(seed 42). Nothing here is a benchmark result or a reproduction of the source paper.
Paper-reported values are only in `evidence/paper_reported/qelm.json` with source locators.

## Local test run (2026-09-27)
```
python -m pytest -q
82 passed, 1 warning in 35.22s
```
The warning is a SciPy precision notice raised inside the Friedman helper test when three
methods produce nearly identical seed results on the tiny fixture.

## Behaviours the tests lock
- input schema and class order; the loader refuses a 47-column table in paper mode;
- tensor shapes at every encoder stage, including the explicit 64 to 256 expansion;
- angle and density boundaries, seeded initialisation, hidden weights fixed after init;
- the encoder is frozen before the ELM solve and receives no update;
- the regularised solve matches a trusted ridge solution on the same hidden matrix;
- grey wolf bounds, rounding, categorical decoding, training-only fitness;
- rare class kept in every inner fold; binary mapping; checkpoint reload parity;
- README carries the paper relationship and the exact contribution wording.

# Data handling

## Adapter contract

`attention_qelm_gwo_iiot_lab.data.load_edge_iiotset(data_root, manifest, spec)` reads
`<data_root>/edge_iiotset_processed.csv` and applies `validate_frame`:

1. columns listed under `drop_columns` in the manifest (`frame.number`, `frame.time`,
   `ip.src_host`, `ip.dst_host`) are removed if present;
2. the remaining feature columns must equal the manifest's 48 names as a set. Missing and
   unexpected columns are both listed in the error. Nothing is truncated or matched by
   position (a CSV with 47 or 49 feature columns is rejected; a test covers this);
3. every label must be one of the 15 names in `configs/classes.yaml`; for the binary task
   the labels are still validated against the 15-class list and then mapped
   Normal -> 0, everything else -> 1.

If `--data-root` is absent and `DATA_ROOT` is unset, the adapter exits with a message
naming the Kaggle and IEEE Dataport sources and the manifest that must be filled.

## Feature manifest

`configs/feature_manifest_paper.yaml` declares `n_features: 48` and one entry per slot with
`name` and `dtype` (`numeric` or `categorical`). The shipped names are placeholders because
the paper does not list the 48 retained columns. Filling the manifest is a user step; the
adapter is deliberately strict so that a mismatch fails loudly rather than silently training
on the wrong columns.

## Preprocessing (train-fit only)

Implemented in `preprocess/pipeline.py`:

| Step | Statistic | Fitted on |
|---|---|---|
| drop columns with missing fraction above 0.40 | missing fraction | train |
| replace +/- inf | maximum of finite train values | train |
| impute numeric | median | train |
| impute categorical | mode; label-encode with train vocabulary | train |
| Min-Max to [0, 1] | column min and max; constant columns map to 0; transformed rows outside the train range are clipped | train |

`tests/test_preprocess_and_splits.py::test_fitted_statistics_unchanged_when_test_rows_perturbed`
multiplies the test rows by 1000 and checks that no fitted statistic moves.

## Splits

- Outer: `sklearn.model_selection.train_test_split(stratify=y, test_size=0.20)` with the run
  seed. Every class must have at least two rows.
- Inner: `StratifiedKFold(n_splits=k, shuffle=True)` over positions within the training
  partition. The rarest class must have at least k rows; otherwise the code raises instead
  of merging classes. The demo uses k = 2 so the 12-row MITM-like class has rows in both
  folds; the paper used k = 5 on 1.5 M rows.
- The run manifest records a SHA-256 of the sorted train and test index arrays
  (`split_manifest_hash`) and per-class counts on each side.

## Fixture

See `examples/fixtures/v1/FIXTURE.md` and DATA_CARD.md. The generator lives in
`tools/make_fixture.py`; `python -m attention_qelm_gwo_iiot_lab make-fixture` calls it.

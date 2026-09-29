from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attention_qelm_gwo_iiot_lab.preprocess import Preprocessor, inner_folds, outer_split, split_manifest_hash


def _frame(manifest, n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(n, 48)), columns=list(manifest.names))


def test_drops_column_with_more_than_40pct_missing(manifest):
    df = _frame(manifest)
    df.loc[: int(0.5 * len(df)), "feature_05"] = np.nan          # about 50% missing
    df.loc[: int(0.3 * len(df)), "feature_06"] = np.nan          # about 30% missing, kept
    pre = Preprocessor(manifest, 0.40).fit(df)
    assert "feature_05" in pre.stats.dropped_columns
    assert "feature_06" in pre.stats.kept_columns
    assert pre.transform(df).shape[1] == 47


def test_inf_replaced_by_train_column_max(manifest):
    df = _frame(manifest)
    df["feature_10"] = np.linspace(0, 10, len(df))
    df.loc[3, "feature_10"] = np.inf
    pre = Preprocessor(manifest).fit(df)
    assert pre.stats.inf_replacement["feature_10"] == pytest.approx(10.0)
    X = pre.transform(df)
    j = pre.stats.kept_columns.index("feature_10")
    assert X[3, j] == pytest.approx(1.0)                          # inf -> max -> scaled 1


def test_median_imputation_uses_train_statistics_only(manifest):
    train = _frame(manifest, seed=1)
    train["feature_02"] = np.arange(len(train), dtype=float)      # median 99.5
    test = _frame(manifest, seed=2)
    test["feature_02"] = np.nan                                    # everything missing in test
    pre = Preprocessor(manifest).fit(train)
    assert pre.stats.impute_value["feature_02"] == pytest.approx(np.median(np.arange(len(train))))
    X = pre.transform(test)
    j = pre.stats.kept_columns.index("feature_02")
    assert np.allclose(X[:, j], 99.5 / 199.0)


def test_fitted_statistics_unchanged_when_test_rows_perturbed(manifest):
    """Contract rule 10: fit on train only; perturbing test rows must not move any statistic."""
    train = _frame(manifest, seed=3)
    test_a = _frame(manifest, seed=4)
    test_b = test_a * 1000.0 + 7.0
    pre = Preprocessor(manifest).fit(train)
    before = pre.stats.to_dict()
    Xa = pre.transform(test_a)
    Xb = pre.transform(test_b)
    assert pre.stats.to_dict() == before
    assert not np.allclose(Xa, Xb)                                 # transform sees the perturbation
    assert np.allclose(pre.transform(train), Preprocessor(manifest).fit(train).transform(train))


def test_minmax_train_output_in_unit_interval_and_test_clipped(manifest):
    train = _frame(manifest, seed=5)
    pre = Preprocessor(manifest).fit(train)
    Xt = pre.transform(train)
    assert Xt.min() >= 0.0 and Xt.max() <= 1.0
    far = train.copy()
    far["feature_01"] = 1e6
    assert pre.transform(far)[:, 0].max() == pytest.approx(1.0)


def test_categorical_mode_imputation_and_vocab(manifest):
    from attention_qelm_gwo_iiot_lab.data.manifest import FeatureManifest
    dtypes = list(manifest.dtypes)
    dtypes[0] = "categorical"
    cat_manifest = FeatureManifest(manifest.names, tuple(dtypes), manifest.label_column,
                                   manifest.drop_columns, manifest.source)
    df = _frame(manifest, n=100)
    df["feature_01"] = ["tcp"] * 60 + ["udp"] * 30 + [None] * 10
    pre = Preprocessor(cat_manifest).fit(df)
    assert pre.stats.impute_value["feature_01"] == "tcp"
    assert pre.stats.vocab["feature_01"] == {"tcp": 0, "udp": 1}
    X = pre.transform(df)
    assert set(np.unique(X[:, 0])) == {0.0, 1.0}


def test_outer_split_is_stratified_80_20_with_every_class_in_both(fixture_bundle):
    plan = outer_split(fixture_bundle.y, 0.20, seed=42)
    n = len(fixture_bundle.y)
    assert len(plan.train_idx) + len(plan.test_idx) == n
    assert abs(len(plan.test_idx) / n - 0.20) < 0.005
    assert len(np.intersect1d(plan.train_idx, plan.test_idx)) == 0
    tr = np.bincount(fixture_bundle.y[plan.train_idx], minlength=15)
    te = np.bincount(fixture_bundle.y[plan.test_idx], minlength=15)
    assert (tr > 0).all() and (te > 0).all()
    assert te[14] >= 2                                             # MITM-like class reaches the test side


def test_outer_split_refuses_singleton_class():
    y = np.array([0] * 10 + [1])
    with pytest.raises(ValueError):
        outer_split(y, 0.2, seed=0)


def test_inner_folds_keep_rare_class_in_every_fold_and_stay_inside_train():
    y = np.array([0] * 400 + [1] * 12)                             # MITM-like: 12 rows
    folds = inner_folds(y, n_folds=2, seed=42)
    assert len(folds) == 2
    for tr, va in folds:
        assert (y[tr] == 1).sum() >= 5 and (y[va] == 1).sum() >= 5
        assert tr.max() < len(y) and va.max() < len(y)
        assert len(np.intersect1d(tr, va)) == 0


def test_inner_folds_refuse_when_rare_class_smaller_than_k():
    y = np.array([0] * 50 + [1] * 3)
    with pytest.raises(ValueError):
        inner_folds(y, n_folds=5, seed=0)


def test_split_manifest_hash_changes_with_seed(fixture_bundle):
    a = split_manifest_hash(outer_split(fixture_bundle.y, 0.2, 1))
    b = split_manifest_hash(outer_split(fixture_bundle.y, 0.2, 2))
    assert a != b and len(a) == 64

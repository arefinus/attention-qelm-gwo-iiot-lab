from __future__ import annotations

import inspect

import numpy as np
import pytest

from attention_qelm_gwo_iiot_lab.config import load_yaml
from attention_qelm_gwo_iiot_lab.gwo import Dimension, GWOConfig, GreyWolfOptimizer, SearchSpace, a_schedule
from attention_qelm_gwo_iiot_lab.qelm import QELMConfig
from attention_qelm_gwo_iiot_lab.search import qelm_cv_fitness, run_search


@pytest.fixture(scope="module")
def space(repo_root):
    return SearchSpace.from_dict(load_yaml(repo_root / "configs" / "gwo_paper.yaml")["space"])


def test_space_is_seven_dimensional_table3(space):
    assert space.n_dims == 7
    assert [d.name for d in space.dims] == ["n_layers", "n_hidden", "activation", "lambda1", "lambda2", "theta", "rho"]
    assert [d.kind for d in space.dims] == ["int", "int", "cat", "log", "log", "float", "float"]


def test_decode_rounds_ints_decodes_categorical_and_exponentiates_log(space):
    x = np.array([2.6, 511.4, 2.7, -3.0, -1.0, 7.0, 1.3])         # theta and rho out of bounds
    p = space.decode(x)
    assert p["n_layers"] == 3 and isinstance(p["n_layers"], int)
    assert p["n_hidden"] == 511
    assert p["activation"] == "gelu"                                  # index 2.7 -> 3
    assert p["lambda1"] == pytest.approx(1e-3) and p["lambda2"] == pytest.approx(1e-1)
    assert p["theta"] == pytest.approx(2 * np.pi) and p["rho"] == 1.0


def test_clip_keeps_positions_inside_internal_bounds(space):
    rng = np.random.default_rng(0)
    x = rng.uniform(-100, 2000, size=(50, 7))
    c = space.clip(x)
    assert (c >= space.lower).all() and (c <= space.upper).all()
    for row in c:
        p = space.decode(row)
        assert 1 <= p["n_layers"] <= 5 and 64 <= p["n_hidden"] <= 1024
        assert 1e-6 <= p["lambda1"] <= 1e-1 and 0 <= p["rho"] <= 1


def test_dimension_validation():
    with pytest.raises(ValueError):
        Dimension("x", "cat", choices=("only",))
    with pytest.raises(ValueError):
        Dimension("x", "log", low=0.0, high=1.0)
    with pytest.raises(ValueError):
        Dimension("x", "int", low=5, high=1)


def test_a_schedule_is_linear_from_two_to_zero():
    assert a_schedule(0, 50) == 2.0
    assert a_schedule(25, 50) == pytest.approx(1.0)
    assert a_schedule(50, 50) == 0.0


def test_gwo_caches_identical_decoded_configs_and_is_seeded():
    sp = SearchSpace([Dimension("k", "int", 1, 3), Dimension("f", "cat", choices=("a", "b"))])
    calls = []

    def fitness(p):
        calls.append(p)
        return float(p["k"]) + (0.5 if p["f"] == "b" else 0.0)

    r1 = GreyWolfOptimizer(sp, GWOConfig(n_wolves=6, max_iter=8, seed=1)).run(fitness)
    assert r1.n_evaluations <= 6                                    # only 6 distinct configs exist
    assert r1.n_cache_hits > 0 and r1.n_evaluations == len(calls)
    assert r1.best_params == {"k": 3, "f": "b"} and r1.best_fitness == 3.5
    r2 = GreyWolfOptimizer(sp, GWOConfig(n_wolves=6, max_iter=8, seed=1)).run(fitness)
    assert [t["best_fitness"] for t in r1.trace] == [t["best_fitness"] for t in r2.trace]


def test_gwo_early_stops_on_flat_fitness_and_positions_stay_bounded():
    sp = SearchSpace([Dimension("x", "float", -5, 5), Dimension("y", "float", -5, 5)])
    seen = []

    def flat(p):
        seen.append(p)
        return 1.0

    r = GreyWolfOptimizer(sp, GWOConfig(n_wolves=5, max_iter=50, early_stop_tol=1e-4, early_stop_patience=3, seed=0)).run(flat)
    assert r.converged_iter is not None and r.converged_iter <= 5
    assert all(-5 <= p["x"] <= 5 and -5 <= p["y"] <= 5 for p in seen)


def test_gwo_improves_on_a_smooth_objective():
    sp = SearchSpace([Dimension("x", "float", -5, 5), Dimension("y", "float", -5, 5)])
    r = GreyWolfOptimizer(sp, GWOConfig(n_wolves=10, max_iter=30, early_stop_patience=30, seed=3)).run(
        lambda p: -(p["x"] - 1.0) ** 2 - (p["y"] + 2.0) ** 2)
    assert r.best_fitness > -0.05
    assert r.trace[-1]["best_fitness"] >= r.trace[0]["best_fitness"]
    assert r.trace[0]["a"] == 2.0


def test_gwo_requires_three_wolves():
    sp = SearchSpace([Dimension("x", "float", 0, 1)])
    with pytest.raises(ValueError):
        GreyWolfOptimizer(sp, GWOConfig(n_wolves=2))


def test_fitness_uses_training_rows_only_and_has_no_test_argument():
    sig = inspect.signature(qelm_cv_fitness)
    assert not any("test" in name for name in sig.parameters)
    rng = np.random.default_rng(0)
    H = rng.normal(size=(120, 16))
    y = rng.integers(0, 3, 120)
    folds = [(np.arange(0, 60), np.arange(60, 120)), (np.arange(60, 120), np.arange(0, 60))]
    fit = qelm_cv_fitness(H, y, 3, folds, QELMConfig(n_layers=1, n_hidden=8))
    val = fit({"n_layers": 1, "n_hidden": 8, "activation": "relu", "lambda1": 1e-3, "lambda2": 1e-3,
               "theta": 0.0, "rho": 0.5})
    assert 0.0 <= val <= 1.0
    with pytest.raises(ValueError):
        qelm_cv_fitness(H, y, 3, [(np.arange(60), np.arange(60, 130))], QELMConfig())   # index beyond train


def test_run_search_returns_best_params_within_bounds(repo_root):
    rng = np.random.default_rng(1)
    H = rng.normal(size=(150, 12))
    y = (H[:, 0] + 0.3 * rng.normal(size=150) > 0).astype(int)
    folds = [(np.arange(0, 75), np.arange(75, 150)), (np.arange(75, 150), np.arange(0, 75))]
    gwo_cfg = load_yaml(repo_root / "configs" / "gwo_demo.yaml")
    gwo_cfg = {**gwo_cfg, "n_wolves": 3, "max_iter": 2}
    result, log = run_search(H, y, 2, folds, QELMConfig(n_hidden=16), gwo_cfg)
    p = result.best_params
    assert set(p) == {"n_layers", "n_hidden", "activation", "lambda1", "lambda2", "theta", "rho"}
    assert p["activation"] in ("sigmoid", "tanh", "relu", "gelu")
    assert len(log.calls) == result.n_evaluations
    assert all(len(c["fold_scores"]) == 2 for c in log.calls)

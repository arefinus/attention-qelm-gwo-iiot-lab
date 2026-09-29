from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import ElasticNet, Ridge

from attention_qelm_gwo_iiot_lab.elastic_net import elastic_net_cd, objective
from attention_qelm_gwo_iiot_lab.qelm import (QELM, QELMConfig, activate, one_hot, quantum_inspired_init,
                                              sample_angles)


def _data(n=300, d=20, c=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, c, n)
    return X, y


def test_seeded_init_is_reproducible_and_hidden_weights_fixed():
    X, y = _data()
    a = QELM(QELMConfig(seed=5, n_layers=2, n_hidden=16), 20, 4)
    b = QELM(QELMConfig(seed=5, n_layers=2, n_hidden=16), 20, 4)
    for (Wa, ba), (Wb, bb) in zip(a.layers, b.layers):
        assert np.array_equal(Wa, Wb) and np.array_equal(ba, bb)
    snapshot = [(W.copy(), b.copy()) for W, b in a.layers]
    a.fit(X, one_hot(y, 4))
    for (W, b), (W0, b0) in zip(a.layers, snapshot):
        assert np.array_equal(W, W0) and np.array_equal(b, b0)
    assert QELM(QELMConfig(seed=6, n_layers=2, n_hidden=16), 20, 4).layers[0][0].sum() != a.layers[0][0].sum()


def test_angle_window_shift_and_none_modes():
    rng = np.random.default_rng(0)
    shifted = sample_angles(rng, (2000,), theta=1.84, mode="shift")
    assert shifted.min() >= 1.84 and shifted.max() < 1.84 + 2 * np.pi
    plain = sample_angles(np.random.default_rng(0), (2000,), theta=1.84, mode="none")
    assert plain.min() >= 0.0 and plain.max() < 2 * np.pi


def test_density_boundaries_rho_one_is_cos_rho_zero_is_sin():
    rng = np.random.default_rng(0)
    W1, _ = quantum_inspired_init(np.random.default_rng(0), 5, 7, theta=0.0, rho=1.0, mode="none", scale="none")
    ang = sample_angles(np.random.default_rng(0), (5, 7), 0.0, "none")
    assert np.allclose(W1, np.cos(ang))
    W0, _ = quantum_inspired_init(np.random.default_rng(0), 5, 7, theta=0.0, rho=0.0, mode="none", scale="none")
    assert np.allclose(W0, np.sin(ang))
    Wm, _ = quantum_inspired_init(rng, 50, 50, theta=1.0, rho=0.63, mode="shift", scale="none")
    assert np.abs(Wm).max() <= 1.0 + 1e-12


def test_fan_in_scaling_shrinks_weights():
    Wn, _ = quantum_inspired_init(np.random.default_rng(0), 100, 10, 0.0, 0.5, scale="none")
    Wf, _ = quantum_inspired_init(np.random.default_rng(0), 100, 10, 0.0, 0.5, scale="fan_in")
    assert np.allclose(Wf * 10.0, Wn)


def test_config_validation_rejects_bad_values():
    for bad in (dict(n_layers=0), dict(rho=1.5), dict(activation="swish"), dict(lambda1=-1.0),
                dict(global_angle_mode="rotate"), dict(weight_scale="xavier")):
        with pytest.raises(ValueError):
            QELMConfig(**bad)
    with pytest.raises(ValueError):
        activate("swish", np.zeros(3))


def test_activations_propagate_through_all_layers_with_shapes():
    X, _ = _data(n=17)
    q = QELM(QELMConfig(n_layers=3, n_hidden=9), 20, 4)
    trace = q.hidden_trace(X)
    stages = [t["stage"] for t in trace]
    assert stages[0] == "qelm_input" and stages.count("fixed_hidden_1_gelu") == 1
    assert sum(s.startswith("fixed_hidden_") for s in stages) == 3
    assert q.hidden(X).shape == (17, 10)                            # 9 hidden + bias column
    assert all(t["trainable"] is False for t in trace if t["stage"].startswith("fixed_hidden"))


def test_ridge_solution_matches_sklearn_on_same_hidden_matrix():
    X, y = _data(n=400, d=20, c=3)
    T = one_hot(y, 3)
    q = QELM(QELMConfig(n_layers=1, n_hidden=64, lambda1=0.3, lambda2=0.7), 20, 3).fit(X, T)
    H = q.hidden(X)
    ref = Ridge(alpha=1.0, fit_intercept=False, solver="cholesky").fit(H, T)
    assert np.allclose(q.beta, ref.coef_.T, atol=1e-6)
    assert np.linalg.norm(T - H @ q.beta) == pytest.approx(np.linalg.norm(T - ref.predict(H)), rel=1e-6)
    assert q.solver_info["method"].startswith("scipy.linalg")


def test_streaming_accumulation_equals_full_batch_solve():
    X, y = _data(n=500)
    T = one_hot(y, 4)
    full = QELM(QELMConfig(n_layers=1, n_hidden=32), 20, 4).fit(X, T, batch_size=10_000)
    streamed = QELM(QELMConfig(n_layers=1, n_hidden=32), 20, 4).fit(X, T, batch_size=64)
    assert np.allclose(full.beta, streamed.beta, atol=1e-8)


def test_npz_save_load_parity_without_pickle(tmp_path):
    X, y = _data()
    q = QELM(QELMConfig(n_layers=2, n_hidden=24), 20, 4).fit(X, one_hot(y, 4))
    q.save(tmp_path / "q.npz")
    q2 = QELM.load(tmp_path / "q.npz")
    assert np.array_equal(q.predict(X), q2.predict(X))
    assert np.allclose(q.decision_function(X), q2.decision_function(X))
    assert q2.cfg == q.cfg


def test_predict_proba_rows_sum_to_one_and_unsolved_model_refuses():
    X, y = _data()
    q = QELM(QELMConfig(n_layers=1, n_hidden=8), 20, 4)
    with pytest.raises(RuntimeError):
        q.predict(X)
    q.fit(X, one_hot(y, 4))
    assert np.allclose(q.predict_proba(X).sum(axis=1), 1.0)


def test_elastic_net_extension_matches_sklearn():
    rng = np.random.default_rng(0)
    H = rng.normal(size=(200, 12))
    T = H @ rng.normal(size=(12, 2)) + 0.1 * rng.normal(size=(200, 2))
    l1, l2 = 2.0, 1.0
    beta, info = elastic_net_cd(H, T, l1, l2, max_iter=2000, tol=1e-10)
    n = H.shape[0]
    ref = ElasticNet(alpha=(l1 + l2) / n, l1_ratio=l1 / (l1 + l2), fit_intercept=False, tol=1e-10,
                     max_iter=50_000).fit(H, T)
    assert np.allclose(beta, ref.coef_.T, atol=1e-4)
    assert info["converged"] and info["method"] == "coordinate_descent_elastic_net"
    assert objective(H, T, beta, l1, l2) <= objective(H, T, np.zeros_like(beta), l1, l2)

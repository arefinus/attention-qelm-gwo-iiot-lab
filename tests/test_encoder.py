from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from attention_qelm_gwo_iiot_lab.encoder import (AttentionEncoder, EncoderConfig, TrainConfig, focal_loss,
                                                 inverse_frequency_weights, load_encoder, save_encoder,
                                                 train_encoder)
from attention_qelm_gwo_iiot_lab.qelm import QELM, QELMConfig, one_hot


def _state_hash(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def test_tensor_shapes_at_every_stage_expand64to256():
    torch.manual_seed(0)
    m = AttentionEncoder(EncoderConfig(), 15)
    feats, trace = m.forward_trace(torch.randn(5, 48))
    shapes = {t["stage"]: t["shape"] for t in trace}
    assert shapes["input"] == [5, 48]
    assert shapes["fc1_gelu"] == [5, 128]
    assert shapes["bottleneck_gelu"] == [5, 64]
    assert shapes["expand_64_to_256"] == [5, 256]
    assert shapes["tokens_plus_feature_index_embedding"] == [5, 8, 32]
    assert shapes["attention_scores"] == [5, 8, 8, 8]
    assert shapes["heads_concat"] == [5, 8, 256]
    assert shapes["w_o"] == [5, 8, 32]
    assert shapes["residual_layernorm_1"] == [5, 8, 32]
    assert shapes["ffn_residual_layernorm_2"] == [5, 8, 32]
    assert shapes["flatten"] == [5, 256]
    assert feats.shape == (5, 256)
    assert m.logits(torch.randn(3, 48)).shape == (3, 15)


def test_direct48to256_tokenisation_shapes():
    m = AttentionEncoder(EncoderConfig(tokenisation="direct48to256"), 2)
    feats, trace = m.forward_trace(torch.randn(4, 48))
    stages = [t["stage"] for t in trace]
    assert "direct_48_to_256" in stages and "bottleneck_gelu" not in stages
    assert feats.shape == (4, 256)


def test_invalid_tokenisation_rejected():
    with pytest.raises(ValueError):
        AttentionEncoder(EncoderConfig(tokenisation="reshape64to256"), 15)


def test_parameter_count_is_positive_and_seed_reproducible():
    torch.manual_seed(7)
    a = AttentionEncoder(EncoderConfig(), 15)
    torch.manual_seed(7)
    b = AttentionEncoder(EncoderConfig(), 15)
    assert a.n_parameters() > 0 and a.n_parameters(include_head=False) < a.n_parameters()
    assert _state_hash(a) == _state_hash(b)


def test_focal_loss_reduces_to_weighted_ce_at_gamma_zero():
    torch.manual_seed(0)
    logits = torch.randn(16, 5)
    y = torch.randint(0, 5, (16,))
    fl = focal_loss(logits, y, gamma=0.0, alpha=1.0, class_weights=None)
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(fl, ce, atol=1e-6)


def test_inverse_frequency_weights_favour_rare_classes():
    y = np.array([0] * 90 + [1] * 10)
    w = inverse_frequency_weights(y, 3)
    assert w[1] > w[0] and w[2] == 0.0
    assert np.isclose(w[[0, 1]].mean(), 1.0)
    assert np.isclose(w[1] / w[0], 9.0)


def test_training_logs_only_inner_validation_and_stops_early():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 48)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    m = AttentionEncoder(EncoderConfig(dropout=0.0), 2)
    log = train_encoder(m, X[:200], y[:200], X[200:], y[200:],
                        TrainConfig(batch_size=50, max_epochs=40, patience=2), seed=0)
    assert set(log["history"][0]) == {"epoch", "train_loss", "val_loss", "val_macro_f1", "elapsed_s"}
    assert log["monitor"] == "val_loss"
    assert log["epochs_run"] <= 40 and log["best_epoch"] <= log["epochs_run"]
    assert "test" not in " ".join(log["history"][0].keys())
    assert log["history"][-1]["train_loss"] < log["history"][0]["train_loss"]


def test_frozen_encoder_unchanged_and_gradient_free_after_qelm_solve():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 48)).astype(np.float32)
    y = rng.integers(0, 3, 120)
    m = AttentionEncoder(EncoderConfig(dropout=0.0), 3)
    train_encoder(m, X[:90], y[:90], X[90:], y[90:], TrainConfig(batch_size=30, max_epochs=2, patience=5), seed=0)
    m.freeze()
    before = _state_hash(m)
    assert all(not p.requires_grad for p in m.parameters())
    H = m.embed(X)
    q = QELM(QELMConfig(n_layers=2, n_hidden=32), H.shape[1], 3).fit(H, one_hot(y, 3))
    q.predict(H)
    assert _state_hash(m) == before
    assert all(p.grad is None for p in m.parameters())
    assert m.is_frozen and not m.training


def test_checkpoint_reload_parity(tmp_path):
    torch.manual_seed(3)
    m = AttentionEncoder(EncoderConfig(dropout=0.0), 15).freeze()
    X = np.random.default_rng(2).normal(size=(10, 48)).astype(np.float32)
    save_encoder(m, tmp_path / "enc.pt")
    m2 = load_encoder(tmp_path / "enc.pt")
    assert m2.cfg == m.cfg and m2.n_classes == 15
    assert np.allclose(m.embed(X), m2.embed(X), atol=1e-6)

"""Two-phase training procedure.

Phase 1: train the attention encoder with a temporary softmax head (focal loss with
inverse-frequency class weights, Adam), early-stopped on an inner validation fold carved
from the TRAINING partition. Then freeze the encoder.
Phase 2: embed the training partition with the frozen encoder, optionally run the GWO over
QELM hyperparameters using inner folds of the training partition, then solve the QELM
output weights in closed form on the whole training partition.

Binary and multiclass are separate tasks with separate configs. The binary model is trained
on Normal/Attack labels from scratch; it is never derived from multiclass predictions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import resolve_path, write_json
from .data import DataFrameBundle, load_classes, load_dataset, load_manifest
from .encoder import (AttentionEncoder, EncoderConfig, TrainConfig, load_encoder, save_encoder,
                      train_encoder)
from .preprocess import Preprocessor, SplitPlan, inner_folds, outer_split
from .qelm import QELM, QELMConfig, one_hot
from .search import make_qelm_from_params, run_search


@dataclass
class TrainedArtifacts:
    bundle: DataFrameBundle
    plan: SplitPlan
    preprocessor: Preprocessor
    X_train: np.ndarray
    X_test: np.ndarray
    encoder: AttentionEncoder
    encoder_log: dict
    H_train: np.ndarray
    H_test: np.ndarray
    qelm: QELM
    qelm_config: QELMConfig
    search_result: dict | None
    seconds: float


def prepare_data(cfg: dict, data_root: str | None = None, seed: int | None = None
                 ) -> tuple[DataFrameBundle, SplitPlan, Preprocessor, np.ndarray, np.ndarray]:
    seed = cfg["seed"] if seed is None else seed
    manifest = load_manifest(resolve_path(cfg["manifest_file"]))
    spec = load_classes(resolve_path(cfg["classes_file"]), task=cfg["task"])
    bundle = load_dataset(cfg, manifest, spec, data_root=data_root)
    plan = outer_split(bundle.y, cfg["data"]["test_fraction"], seed=seed)
    pre = Preprocessor(manifest, cfg["preprocess"]["drop_missing_above"])
    X_train = pre.fit_transform(bundle.X.iloc[plan.train_idx])
    X_test = pre.transform(bundle.X.iloc[plan.test_idx])
    return bundle, plan, pre, X_train, X_test


def phase1_encoder(cfg: dict, X_train: np.ndarray, y_train: np.ndarray, n_classes: int,
                   seed: int) -> tuple[AttentionEncoder, dict]:
    enc_cfg = EncoderConfig.from_dict({**cfg["encoder"], "input_dim": X_train.shape[1]})
    train_cfg = TrainConfig.from_dict(cfg["encoder"])
    import torch
    torch.manual_seed(seed)
    model = AttentionEncoder(enc_cfg, n_classes)
    # inner validation fold for early stopping: first fold of the inner k-fold, train only
    folds = inner_folds(y_train, n_folds=max(2, int(cfg["data"]["inner_folds"])), seed=seed)
    fit_idx, val_idx = folds[0]
    log = train_encoder(model, X_train[fit_idx], y_train[fit_idx], X_train[val_idx], y_train[val_idx],
                        train_cfg, seed=seed)
    model.freeze()
    log["n_parameters_total"] = model.n_parameters(include_head=True)
    log["n_parameters_without_head"] = model.n_parameters(include_head=False)
    log["inner_validation_rows"] = int(len(val_idx))
    return model, log


def phase2_qelm(cfg: dict, H_train: np.ndarray, y_train: np.ndarray, n_classes: int, seed: int,
                gwo_cfg: dict | None) -> tuple[QELM, QELMConfig, dict | None]:
    base = QELMConfig.from_dict({**cfg["qelm"], "seed": seed})
    search_out: dict | None = None
    if gwo_cfg is not None:
        folds = inner_folds(y_train, n_folds=max(2, int(cfg["data"]["inner_folds"])), seed=seed)
        result, log = run_search(H_train, y_train, n_classes, folds, base, gwo_cfg)
        base = make_qelm_from_params(base, result.best_params)
        search_out = {**result.to_dict(), "fitness_calls": log.calls,
                      "fitness_definition": f"{len(folds)}-fold macro-F1 on the training partition only",
                      "evidence_status": "demo"}
    solver = cfg["qelm"].get("solver", "ridge")
    model = QELM(base, H_train.shape[1], n_classes)
    T = one_hot(y_train, n_classes)
    if solver == "ridge":
        model.fit(H_train, T)
    elif solver == "elastic_net_cd":
        from .elastic_net import elastic_net_cd
        H = model.hidden(H_train)
        beta, info = elastic_net_cd(H, T, base.lambda1, base.lambda2)
        model.beta, model.solver_info = beta, info
    else:
        raise ValueError(f"unknown qelm.solver {solver!r}; use ridge or elastic_net_cd")
    return model, base, search_out


def run_training(cfg: dict, out_dir: str | Path, data_root: str | None = None, seed: int | None = None,
                 gwo_cfg: dict | None = None, save_artifacts: bool = True) -> TrainedArtifacts:
    t0 = time.perf_counter()
    seed = cfg["seed"] if seed is None else int(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle, plan, pre, X_train, X_test = prepare_data(cfg, data_root, seed)
    y_train = bundle.y[plan.train_idx]
    n_classes = bundle.class_spec.n_classes
    encoder, enc_log = phase1_encoder(cfg, X_train, y_train, n_classes, seed)
    H_train = encoder.embed(X_train)
    H_test = encoder.embed(X_test)
    qelm, qcfg, search_out = phase2_qelm(cfg, H_train, y_train, n_classes, seed, gwo_cfg)
    if save_artifacts:
        save_encoder(encoder, out / "encoder.pt")
        qelm.save(out / "qelm.npz")
        write_json(out / "encoder_log.json", enc_log)
        write_json(out / "preprocess_stats.json", pre.stats.to_dict() if pre.stats else {})
        if search_out is not None:
            write_json(out / "gwo_trace.json", search_out)
    return TrainedArtifacts(bundle, plan, pre, X_train, X_test, encoder, enc_log, H_train, H_test,
                            qelm, qcfg, search_out, time.perf_counter() - t0)


def reload_for_inference(out_dir: str | Path) -> tuple[AttentionEncoder, QELM]:
    out = Path(out_dir)
    return load_encoder(out / "encoder.pt").freeze(), QELM.load(out / "qelm.npz")

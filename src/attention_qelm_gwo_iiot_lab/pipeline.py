"""Smoke and demo orchestration. Every number written here is computed in-session on the
authored synthetic fixture and labelled `demo`; nothing is a benchmark result."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import PROJECT_ID
from .baselines import run_baseline
from .config import config_hash, load_run_config, load_yaml, resolve_path, write_json
from .evaluate import compute_metrics
from .preprocess import split_manifest_hash
from .run_manifest import build_manifest, make_run_id, now_iso, write_manifest
from .sensitivity import run_all_audits
from .stats import aggregate_seeds, friedman_holm
from .train import TrainedArtifacts, run_training

DEMO_NOTE = "computed by `make demo` on synthetic fixture v1, not a benchmark result"


def _predict(art: TrainedArtifacts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = art.bundle.y[art.plan.test_idx]
    scores = art.qelm.decision_function(art.H_test)
    return y_true, scores.argmax(axis=1), scores


def _write_predictions(path: Path, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray,
                       class_names: tuple[str, ...]) -> None:
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    for j, name in enumerate(class_names):
        df[f"score_{name}"] = np.round(scores[:, j], 4)
    df.to_csv(path, index=False, lineterminator="\n")


def _trace(art: TrainedArtifacts, n_rows: int = 4) -> dict:
    import torch
    xb = torch.as_tensor(art.X_train[:n_rows].astype(np.float32))
    with torch.no_grad():
        feats, enc_trace = art.encoder.forward_trace(xb)
    q_trace = art.qelm.hidden_trace(feats.numpy())
    return {
        "evidence_status": "demo",
        "note": "per-stage tensor shapes for a batch of 4 fixture rows; batch dimension first",
        "tokenisation": art.encoder.cfg.tokenisation,
        "encoder_trainable_parameters_without_head": art.encoder.n_parameters(include_head=False),
        "encoder_parameters_with_phase1_head": art.encoder.n_parameters(include_head=True),
        "qelm_fixed_hidden_parameters": int(sum(W.size + b.size for W, b in art.qelm.layers)),
        "qelm_solved_output_parameters": int(art.qelm.beta.size) if art.qelm.beta is not None else None,
        "stages": [{"block": "attention_encoder", **s} for s in enc_trace]
                  + [{"block": "qelm", **s} for s in q_trace],
        "qelm_config": art.qelm_config.__dict__,
        "solver_info": art.qelm.solver_info,
    }


def _manifest_data(art: TrainedArtifacts, source_id: str) -> dict:
    counts = art.plan.sample_counts(art.bundle.y, art.bundle.class_spec.classes)
    return {"source_id": source_id, "version": "v1", "split_manifest_hash": split_manifest_hash(art.plan),
            "sample_counts": counts}


# ------------------------------------------------------------------ smoke
def run_smoke(out_dir: str | Path = "examples/output/smoke", config: str = "configs/smoke.yaml") -> dict:
    started = now_iso()
    t0 = time.perf_counter()
    cfg = load_run_config(config)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    art = run_training(cfg, out, gwo_cfg=None, save_artifacts=True)
    y_true, y_pred, scores = _predict(art)
    metrics = compute_metrics(y_true, y_pred, scores, art.bundle.class_spec.classes, cfg["task"])
    metrics["evidence_status"] = "demo"
    metrics["note"] = "smoke run on synthetic fixture v1: schema check only, not a result"
    write_json(out / "metrics.json", metrics)
    _write_predictions(out / "predictions.csv", y_true, y_pred, scores, art.bundle.class_spec.classes)
    write_json(out / "demo_trace.json", _trace(art))
    manifest = build_manifest(make_run_id("smoke"), "completed", "smoke", "demo",
                              _manifest_data(art, art.bundle.source_id), config_hash(cfg), cfg["seed"],
                              str(out / "metrics.json"), str(out / "predictions.csv"), started, now_iso())
    write_manifest(out / "run_manifest.yaml", manifest)
    _check_smoke_outputs(out, metrics, art.bundle.class_spec.n_classes)
    return {"out_dir": str(out), "seconds": time.perf_counter() - t0, "n_test": int(len(y_true)),
            "n_classes": art.bundle.class_spec.n_classes, "epochs_run": art.encoder_log["epochs_run"]}


def _check_smoke_outputs(out: Path, metrics: dict, n_classes: int) -> None:
    for name in ("metrics.json", "predictions.csv", "run_manifest.yaml", "encoder.pt", "qelm.npz", "demo_trace.json"):
        if not (out / name).exists():
            raise RuntimeError(f"smoke output missing: {out / name}")
    if len(metrics["class_order"]) != n_classes or len(metrics["confusion_matrix"]) != n_classes:
        raise RuntimeError("smoke metrics schema mismatch: class order or confusion matrix size")
    for key in ("accuracy", "precision", "recall", "f1", "fpr", "per_class_f1"):
        if key not in metrics:
            raise RuntimeError(f"smoke metrics missing key {key}")


# ------------------------------------------------------------------- demo
def run_demo(out_dir: str | Path = "examples/output", config: str = "configs/demo_multiclass.yaml",
             binary_config: str = "configs/demo_binary.yaml", data_root: str | None = None) -> dict:
    started = now_iso()
    t0 = time.perf_counter()
    cfg = load_run_config(config)
    bcfg = load_run_config(binary_config)
    gwo_cfg = load_yaml(resolve_path(cfg["search"]["gwo_file"])) if cfg["search"].get("enabled") else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    timing: dict[str, float] = {}

    # main multiclass run with GWO at the config seed
    t = time.perf_counter()
    art = run_training(cfg, out / "train_multiclass", data_root, gwo_cfg=gwo_cfg, save_artifacts=True)
    timing["multiclass_train_with_search_s"] = time.perf_counter() - t
    classes = art.bundle.class_spec.classes
    y_true, y_pred, scores = _predict(art)
    mc = compute_metrics(y_true, y_pred, scores, classes, "multiclass")
    _write_predictions(out / "predictions.csv", y_true, y_pred, scores, classes)
    write_json(out / "demo_trace.json", _trace(art))
    write_json(out / "learning_curves.json", {"evidence_status": "demo", "seed": cfg["seed"],
                                              "monitor": art.encoder_log["monitor"],
                                              "best_epoch": art.encoder_log["best_epoch"],
                                              "history": art.encoder_log["history"]})
    if art.search_result is not None:
        write_json(out / "gwo_trace.json", art.search_result)
    roc = mc.pop("roc_curves")
    write_json(out / "roc_curves.json", {"evidence_status": "demo", "class_order": list(classes), "curves": roc})

    # baselines at the config seed, same split and preprocessing
    t = time.perf_counter()
    y_tr = art.bundle.y[art.plan.train_idx]
    profile = cfg["evaluation"].get("baseline_profile", "standard")
    baselines: dict[str, dict] = {}
    for name in cfg["evaluation"].get("baselines", []):
        res = run_baseline(name, art.X_train, y_tr, art.X_test, len(classes), seed=cfg["seed"], profile=profile)
        m = compute_metrics(y_true, res.y_pred, res.scores, classes, "multiclass")
        m.pop("roc_curves", None)
        baselines[name] = {**m, "fit_seconds": res.seconds, "profile": profile}
    timing["baselines_s"] = time.perf_counter() - t

    # multi-seed: retrain encoder + QELM (best searched params fixed) and baselines per seed
    t = time.perf_counter()
    seeds = [int(s) for s in cfg["evaluation"]["seeds"]]
    per_seed: dict[str, dict[str, list[float]]] = {"qelm": {"macro_f1": [], "accuracy": []}}
    for name in baselines:
        per_seed[name] = {"macro_f1": [], "accuracy": []}
    fixed_cfg = {**cfg, "qelm": {**cfg["qelm"], **{k: v for k, v in art.qelm_config.__dict__.items() if k != "seed"}}}
    for s in seeds:
        a = run_training(fixed_cfg, out / "train_multiclass", data_root, seed=s, gwo_cfg=None, save_artifacts=False)
        yt, yp, _ = _predict(a)
        m = compute_metrics(yt, yp, None, classes, "multiclass")
        per_seed["qelm"]["macro_f1"].append(m["f1"])
        per_seed["qelm"]["accuracy"].append(m["accuracy"])
        for name in baselines:
            r = run_baseline(name, a.X_train, a.bundle.y[a.plan.train_idx], a.X_test, len(classes), seed=s,
                             profile=profile)
            mb = compute_metrics(yt, r.y_pred, None, classes, "multiclass")
            per_seed[name]["macro_f1"].append(mb["f1"])
            per_seed[name]["accuracy"].append(mb["accuracy"])
    timing["multiseed_s"] = time.perf_counter() - t
    multiseed = {"seeds": seeds, "note": "outer split reseeded per seed; encoder retrained; QELM hyperparameters "
                                         "fixed to the seed-42 search result; " + DEMO_NOTE,
                 "methods": {m: {k: aggregate_seeds(v) for k, v in d.items()} for m, d in per_seed.items()}}
    friedman = friedman_holm({m: d["macro_f1"] for m, d in per_seed.items()}) if len(per_seed) >= 2 and len(seeds) >= 3 else None

    # binary task: separate training, separate config
    t = time.perf_counter()
    bart = run_training(bcfg, out / "train_binary", data_root, gwo_cfg=None, save_artifacts=True)
    by, bp, bs = _predict(bart)
    bm = compute_metrics(by, bp, bs, bart.bundle.class_spec.classes, "binary")
    bm.pop("roc_curves", None)
    timing["binary_train_s"] = time.perf_counter() - t

    # sensitivity audits on the fixture (raw frame incl. metadata column) and the split
    raw = pd.read_csv(resolve_path(cfg["data"]["fixture_dir"]) / "fixture.csv")
    audits = run_all_audits(raw.drop(columns=["label"]), art.bundle.y, art.X_train, art.X_test, "demo")
    write_json(out / "audit.json", audits)

    timing["total_s"] = time.perf_counter() - t0
    metrics = {
        "project_id": PROJECT_ID, "evidence_status": "demo", "note": DEMO_NOTE,
        "fixture": {"source_id": art.bundle.source_id, "n_rows": int(len(art.bundle.y)),
                    "n_train": int(len(art.plan.train_idx)), "n_test": int(len(art.plan.test_idx))},
        "multiclass": {**mc, "seed": cfg["seed"], "encoder_epochs_run": art.encoder_log["epochs_run"],
                       "encoder_best_epoch": art.encoder_log["best_epoch"],
                       "qelm_config_after_search": art.qelm_config.__dict__,
                       "gwo": None if art.search_result is None else {k: art.search_result[k] for k in
                                                                       ("best_fitness", "n_evaluations", "n_cache_hits", "converged_iter")}},
        "baselines_seed42": baselines,
        "multiseed": multiseed,
        "friedman_holm_macro_f1": friedman,
        "binary": {**bm, "seed": bcfg["seed"], "encoder_epochs_run": bart.encoder_log["epochs_run"]},
        "timing_seconds": timing,
        "proposed_not_implemented_baselines": ["cnn_1d", "lstm", "gru", "bilstm", "densenet"],
    }
    write_json(out / "metrics.json", metrics)
    manifest = build_manifest(make_run_id("demo"), "completed", "demo", "demo",
                              _manifest_data(art, art.bundle.source_id), config_hash(cfg), cfg["seed"],
                              str(out / "metrics.json"), str(out / "predictions.csv"), started, now_iso())
    write_manifest(out / "run_manifest.yaml", manifest)
    return {"out_dir": str(out), "timing_seconds": timing, "multiclass_macro_f1": mc["f1"],
            "binary_f1": bm["f1"], "n_seeds": len(seeds)}

"""Command line interface: python -m attention_qelm_gwo_iiot_lab <subcommand>."""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from .config import (REPO_ROOT, config_hash, load_run_config, load_yaml, resolve_path, set_cpu_threads,
                     write_json)

EXIT_OK, EXIT_FAIL, EXIT_MISSING_INPUT = 0, 1, 2


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, indent=2, default=str) + "\n")


def cmd_smoke(args: argparse.Namespace) -> int:
    from .pipeline import run_smoke
    _emit({"status": "completed", **run_smoke(args.out, args.config)})
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    from .pipeline import run_demo
    _emit({"status": "completed", **run_demo(args.out, args.config, args.binary_config, args.data_root)})
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    from .evaluate import compute_metrics
    from .train import run_training
    cfg = load_run_config(args.config)
    gwo = None
    if cfg["search"].get("enabled") and not args.no_search:
        gwo = load_yaml(resolve_path(args.gwo or cfg["search"]["gwo_file"]))
    art = run_training(cfg, args.out, data_root=args.data_root, seed=args.seed, gwo_cfg=gwo)
    y_true = art.bundle.y[art.plan.test_idx]
    scores = art.qelm.decision_function(art.H_test)
    m = compute_metrics(y_true, scores.argmax(1), scores, art.bundle.class_spec.classes, cfg["task"])
    m.pop("roc_curves", None)
    m["evidence_status"] = "demo" if art.bundle.source_id.startswith("synthetic") else "experiment-unverified"
    write_json(Path(args.out) / "metrics.json", m)
    _emit({"status": "completed", "task": cfg["task"], "out": args.out, "seconds": art.seconds,
           "class_order": list(art.bundle.class_spec.classes), "f1": m["f1"], "accuracy": m["accuracy"],
           "evidence_status": m["evidence_status"]})
    return EXIT_OK


def cmd_search(args: argparse.Namespace) -> int:
    from .train import phase1_encoder, phase2_qelm, prepare_data
    cfg = load_run_config(args.config)
    gwo = load_yaml(resolve_path(args.gwo or cfg["search"]["gwo_file"]))
    bundle, plan, _, X_train, _ = prepare_data(cfg, args.data_root)
    y_train = bundle.y[plan.train_idx]
    enc, _ = phase1_encoder(cfg, X_train, y_train, bundle.class_spec.n_classes, cfg["seed"])
    _, qcfg, search_out = phase2_qelm(cfg, enc.embed(X_train), y_train, bundle.class_spec.n_classes, cfg["seed"], gwo)
    out = Path(args.out)
    write_json(out / "gwo_trace.json", search_out)
    _emit({"status": "completed", "out": str(out / "gwo_trace.json"), "best_params": search_out["best_params"],
           "best_fitness_train_cv": search_out["best_fitness"], "n_evaluations": search_out["n_evaluations"]})
    return EXIT_OK


def cmd_evaluate(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd
    from .data import load_classes
    from .evaluate import compute_metrics
    p = Path(args.predictions)
    if not p.exists():
        sys.stderr.write(f"predictions file not found: {p}. Run `make demo` or `train` first.\n")
        return EXIT_MISSING_INPUT
    cfg = load_run_config(args.config)
    spec = load_classes(resolve_path(cfg["classes_file"]), task=cfg["task"])
    df = pd.read_csv(p)
    score_cols = [f"score_{c}" for c in spec.classes]
    missing = [c for c in ["y_true", "y_pred", *score_cols] if c not in df.columns]
    if missing:
        sys.stderr.write(f"predictions file lacks columns for task {cfg['task']!r}: {missing}. "
                         "Binary and multiclass predictions are not interchangeable.\n")
        return EXIT_MISSING_INPUT
    m = compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), df[score_cols].to_numpy(dtype=np.float64),
                        spec.classes, cfg["task"])
    m.pop("roc_curves", None)
    m["evidence_status"] = "demo"
    m["source_predictions"] = str(p)
    if args.out:
        write_json(args.out, m)
    _emit({"status": "completed", "task": cfg["task"], "accuracy": m["accuracy"], "f1": m["f1"], "fpr": m["fpr"],
           "class_order": m["class_order"], "out": args.out})
    return EXIT_OK


def cmd_audit(args: argparse.Namespace) -> int:
    import pandas as pd
    from .sensitivity import run_all_audits
    from .train import prepare_data
    cfg = load_run_config(args.config)
    bundle, plan, _, X_train, X_test = prepare_data(cfg, args.data_root)
    if cfg["data"].get("source", "fixture") == "fixture":
        raw = pd.read_csv(resolve_path(cfg["data"]["fixture_dir"]) / "fixture.csv").drop(columns=["label"])
        status = "demo"
    else:
        raw = bundle.X
        status = "proposed"
    audits = run_all_audits(raw, bundle.y, X_train, X_test, status)
    out = Path(args.out)
    write_json(out / "audit.json", audits)
    _emit({"status": "completed", "out": str(out / "audit.json"), "evidence_status": status,
           "identifier_like_columns": [c["column"] for c in audits["identifier_like_columns"]],
           "placeholder_flags": [c["column"] for c in audits["placeholder_encoding"]],
           "test_rows_also_in_train": audits["duplicate_flows"]["test_rows_also_in_train"]})
    return EXIT_OK


def _run_tool(name: str, argv: list[str]) -> int:
    script = REPO_ROOT / "tools" / name
    if not script.exists():
        sys.stderr.write(f"tool script not found: {script}\n")
        return EXIT_MISSING_INPUT
    saved = sys.argv
    try:
        sys.argv = [str(script), *argv]
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as e:  # tools call SystemExit(main())
        return int(e.code or 0)
    finally:
        sys.argv = saved
    return EXIT_OK


def cmd_make_fixture(args: argparse.Namespace) -> int:
    return _run_tool("make_fixture.py", ["--out", args.out, "--seed", str(args.seed)])


def cmd_figures(args: argparse.Namespace) -> int:
    return _run_tool("render_figures.py", ["--output-dir", args.output_dir, "--figures-dir", args.figures_dir])


def cmd_config_hash(args: argparse.Namespace) -> int:
    _emit({"config": args.config, "sha256": config_hash(load_yaml(args.config))})
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="attention_qelm_gwo_iiot_lab",
                                 description="Attention encoder + QELM + GWO for IIoT intrusion detection (offline, CPU, synthetic fixtures)")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("smoke", help="under-30-second end-to-end check on the 15-class fixture")
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--out", default="examples/output/smoke")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("demo", help="documented demonstration; writes examples/output/")
    p.add_argument("--config", default="configs/demo_multiclass.yaml")
    p.add_argument("--binary-config", default="configs/demo_binary.yaml")
    p.add_argument("--out", default="examples/output")
    p.add_argument("--data-root", default=None)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("train", help="two-phase training for one config (binary or multiclass)")
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gwo", default=None)
    p.add_argument("--no-search", action="store_true")
    p.add_argument("--data-root", default=None)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("search", help="GWO over QELM hyperparameters on the training partition")
    p.add_argument("--config", required=True)
    p.add_argument("--gwo", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default=None)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("evaluate", help="metrics from a saved predictions.csv; exit 2 if missing")
    p.add_argument("--predictions", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("audit", help="identifier / placeholder / duplicate-flow audits")
    p.add_argument("--config", default="configs/demo_multiclass.yaml")
    p.add_argument("--out", default="examples/output/audit")
    p.add_argument("--data-root", default=None)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("make-fixture", help="regenerate the authored synthetic fixture")
    p.add_argument("--out", default="examples/fixtures/v1")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_make_fixture)

    p = sub.add_parser("figures", help="render docs/figures from examples/output")
    p.add_argument("--output-dir", default="examples/output")
    p.add_argument("--figures-dir", default="docs/figures")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("config-hash", help="print the SHA-256 of a config file")
    p.add_argument("--config", required=True)
    p.set_defaults(func=cmd_config_hash)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    set_cpu_threads()
    try:
        return int(args.func(args))
    except FileNotFoundError as e:
        sys.stderr.write(f"missing input: {e}\n")
        return EXIT_MISSING_INPUT
    except (ValueError, RuntimeError, NotImplementedError) as e:
        sys.stderr.write(f"error: {e}\n")
        return EXIT_FAIL

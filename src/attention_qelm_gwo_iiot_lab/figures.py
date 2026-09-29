"""Charts rendered from ACTUAL demo outputs in examples/output (matplotlib, SVG).

Each function reads a JSON file written by `run_demo`, never a hard-coded number, and
refuses to draw if the input is missing. Titles carry the evidence label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import read_json  # noqa: E402

LABEL = "demo on synthetic fixture v1 (not a benchmark result)"
COLORS = ["#2f5d8a", "#c0642f", "#4b8b5f", "#8a5aa8", "#b5952c", "#5aa0b8", "#a34d6b", "#6b6b6b",
          "#3b7a9e", "#d08a4b", "#79a86e", "#a37bc4", "#c9b04e", "#7bbccf", "#c07a94"]


def _require(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `python -m attention_qelm_gwo_iiot_lab demo` first")
    return read_json(path)


def plot_roc(output_dir: Path, figures_dir: Path) -> Path:
    data = _require(output_dir / "roc_curves.json")
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for i, c in enumerate(data["curves"]):
        if c["auc"] is None or not c["fpr"]:
            continue
        ax.plot(c["fpr"], c["tpr"], color=COLORS[i % len(COLORS)], lw=1.4,
                label=f"{c['class_name']} (AUC {c['auc']:.3f}, n={c['n_positive']})")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Per-class one-vs-rest ROC, QELM test scores\n" + LABEL, fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2, frameon=False)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    out = figures_dir / "roc_per_class_demo.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_confusion(output_dir: Path, figures_dir: Path) -> Path:
    m = _require(output_dir / "metrics.json")["multiclass"]
    cm = np.asarray(m["confusion_matrix"], dtype=np.float64)
    names = m["class_order"]
    row_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            v = int(cm[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6,
                        color="white" if row_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (counts; colour = row-normalised), class order fixed\n" + LABEL, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    out = figures_dir / "confusion_matrix_demo.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_learning_curves(output_dir: Path, figures_dir: Path) -> Path:
    data = _require(output_dir / "learning_curves.json")
    hist = data["history"]
    ep = [h["epoch"] for h in hist]
    fig, ax1 = plt.subplots(figsize=(6.4, 4.2))
    ax1.plot(ep, [h["train_loss"] for h in hist], color=COLORS[0], label="train focal loss")
    ax1.plot(ep, [h["val_loss"] for h in hist], color=COLORS[1], label="inner-validation focal loss")
    ax1.axvline(data["best_epoch"], color="#999999", ls=":", lw=1, label=f"best epoch ({data['best_epoch']})")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_macro_f1"] for h in hist], color=COLORS[2], ls="--", label="inner-validation macro-F1")
    ax2.set_ylabel("macro-F1")
    ax2.set_ylim(0, 1.02)
    lines = ax1.get_legend_handles_labels()
    lines2 = ax2.get_legend_handles_labels()
    ax1.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=7, loc="center right", frameon=False)
    ax1.set_title("Phase-1 encoder training; early stopping monitors inner validation only\n" + LABEL, fontsize=9)
    out = figures_dir / "learning_curves_demo.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def plot_gwo(output_dir: Path, figures_dir: Path) -> Path:
    data = _require(output_dir / "gwo_trace.json")
    tr = data["trace"]
    it = [t["iteration"] for t in tr]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(it, [t["best_fitness"] for t in tr], marker="o", color=COLORS[0], label="best so far")
    ax.plot(it, [t["iter_best_fitness"] for t in tr], marker="s", ms=4, color=COLORS[1], ls="--", label="iteration best")
    ax.plot(it, [t["mean_fitness"] for t in tr], marker="^", ms=4, color=COLORS[2], ls=":", label="pack mean")
    ax.set_xlabel("GWO iteration")
    ax.set_ylabel(data.get("fitness_definition", "fitness"), fontsize=8)
    ax.set_title(f"GWO convergence (N={data.get('n_wolves', '?')} wolves, T={data.get('max_iter', '?')}, "
                 f"{data['n_evaluations']} unique evaluations, {data['n_cache_hits']} cache hits)\n" + LABEL, fontsize=8.5)
    ax.legend(fontsize=7, frameon=False)
    ax.set_xticks(it)
    out = figures_dir / "gwo_convergence_demo.svg"
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)
    return out


def render_all_charts(output_dir: str | Path, figures_dir: str | Path) -> list[Path]:
    output_dir, figures_dir = Path(output_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return [plot_roc(output_dir, figures_dir), plot_confusion(output_dir, figures_dir),
            plot_learning_curves(output_dir, figures_dir), plot_gwo(output_dir, figures_dir)]

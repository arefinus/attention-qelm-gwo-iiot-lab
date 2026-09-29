"""Render docs/figures: the architecture SVG (hand-written here) and the demo charts
(matplotlib, from examples/output JSON). No publisher figure is reproduced.

Usage: python tools/render_figures.py [--output-dir examples/output] [--figures-dir docs/figures]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

ATTRIBUTION = (
    "Original drawing for this repository. The block sequence follows the textual description in "
    "the source article (Discover Artificial Intelligence, 2026, doi 10.1007/s44163-026-01704-3, "
    "CC BY 4.0); the 64 to 256 expansion is this repository's documented repair (audit item 1). "
    "No publisher figure is copied."
)


def _box(x: int, y: int, w: int, h: int, title: str, sub: str, fill: str, stroke: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
        f'<text x="{x + w / 2}" y="{y + 22}" text-anchor="middle" font-size="12" font-weight="600" fill="#1f2933">{title}</text>'
        f'<text x="{x + w / 2}" y="{y + 40}" text-anchor="middle" font-size="10" fill="#3e4c59">{sub}</text>'
    )


def _arrow(x1: int, y1: int, x2: int, y2: int, dashed: bool = False, color: str = "#1f2933") -> str:
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="1.5"{dash} marker-end="url(#arrow)"/>'


def architecture_svg() -> str:
    W, H = 1180, 470
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">'
        '<path d="M0,0 L0,6 L9,3 z" fill="#1f2933"/></marker></defs>',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        '<text x="20" y="28" font-size="15" font-weight="700" fill="#1f2933">Attention encoder + quantum-inspired ELM head + GWO selection (this repository\'s implementation)</text>',
        '<text x="20" y="46" font-size="10" fill="#52606d">Tensor shapes for batch B. Solid arrows: inference path (CXR of features flows left to right). Dashed: training-time only.</text>',
    ]
    enc_fill, enc_stroke = "#e3ecf7", "#2f5d8a"
    elm_fill, elm_stroke = "#f5e9dd", "#c0642f"
    gwo_fill, gwo_stroke = "#e6f2e8", "#4b8b5f"
    y = 90
    boxes = [
        (20, "input", "x : (B, 48)", enc_fill, enc_stroke),
        (150, "W_e + GELU", "(B, 128) -> bottleneck (B, 64)", enc_fill, enc_stroke),
        (330, "expand 64 -> 256", "learned; repair of the paper's reshape", enc_fill, enc_stroke),
        (520, "tokens + index emb.", "(B, 8, 32) + (1, 8, 32)", enc_fill, enc_stroke),
        (700, "8-head attention", "d_k 32, concat 256, W_O -> 32", enc_fill, enc_stroke),
        (880, "LN, FFN x4, LN", "residuals; flatten (B, 256)", enc_fill, enc_stroke),
    ]
    xs = []
    for x, t, s, f, st in boxes:
        w = 110 if x == 20 else 160
        parts.append(_box(x, y, w, 56, t, s, f, st))
        xs.append((x, w))
    for (x, w), (nx, _) in zip(xs[:-1], xs[1:]):
        parts.append(_arrow(x + w, y + 28, nx, y + 28))
    parts.append('<rect x="14" y="78" width="1036" height="82" rx="10" fill="none" stroke="#2f5d8a" stroke-dasharray="3 3"/>')
    parts.append('<text x="24" y="172" font-size="10" fill="#2f5d8a">Phase 1: trained with Adam on focal loss via a temporary softmax head (inner-validation early stopping), then FROZEN</text>')

    y2 = 230
    elm = [
        (330, "fixed hidden 1", "GELU; W = rho cos t + (1 - rho) sin t", elm_fill, elm_stroke),
        (520, "fixed hidden 2", "(B, n_h); weights never trained", elm_fill, elm_stroke),
        (700, "fixed hidden 3", "(B, n_h) + bias column", elm_fill, elm_stroke),
        (880, "solved output", "beta = (H'H + (l1+l2) I)^-1 H'T", elm_fill, elm_stroke),
    ]
    for x, t, s, f, st in elm:
        parts.append(_box(x, y2, 160, 56, t, s, f, st))
    for x in (330, 520, 700):
        parts.append(_arrow(x + 160, y2 + 28, x + 190, y2 + 28))
    parts.append(_arrow(960, y + 56, 960, y2 - 40, False))
    parts.append(_arrow(960, y2 - 40, 410, y2 - 40, False))
    parts.append(_arrow(410, y2 - 40, 410, y2, False))
    parts.append('<text x="430" y="200" font-size="10" fill="#c0642f">frozen features (B, 256), numpy from here on</text>')
    parts.append('<rect x="320" y="218" width="730" height="82" rx="10" fill="none" stroke="#c0642f" stroke-dasharray="3 3"/>')
    parts.append('<text x="330" y="312" font-size="10" fill="#c0642f">Phase 2: quantum-inspired ELM = classical trigonometric random features; closed-form ridge solve via scipy.linalg.solve, streaming H\'H and H\'T</text>')
    parts.append(_box(1060, y2, 100, 56, "scores", "(B, C) argmax", "#f2f2f2", "#6b6b6b"))
    parts.append(_arrow(1040, y2 + 28, 1060, y2 + 28))

    y3 = 350
    parts.append('<rect x="20" y="335" width="1140" height="92" rx="10" fill="' + gwo_fill + '" stroke="' + gwo_stroke + '" stroke-width="1.5"/>')
    parts.append('<text x="34" y="356" font-size="12" font-weight="600" fill="#1f2933">Offline loop beside training: Grey Wolf Optimizer (training-time selection only, separate module from the encoder optimizer)</text>')
    parts.append('<text x="34" y="376" font-size="10" fill="#3e4c59">7-dim space (Table 3): L, n_h, activation, lambda1, lambda2, theta, rho. N wolves, T iterations, a(t) = 2(1 - t/T). Fitness = k-fold macro-F1 on the TRAINING partition of the frozen features.</text>')
    parts.append('<text x="34" y="394" font-size="10" fill="#3e4c59">Each fitness call re-initialises fixed hidden layers with the candidate (theta, rho, L, n_h, activation), solves beta with (lambda1 + lambda2), scores the held-out inner fold. Best candidate is then solved once on all training rows.</text>')
    parts.append('<text x="34" y="412" font-size="10" fill="#3e4c59">The test partition is never seen by GWO, by early stopping, or by preprocessing statistics.</text>')
    parts.append(_arrow(600, 335, 600, y2 + 56 + 2, True, "#4b8b5f"))
    parts.append(f'<text x="20" y="{H - 14}" font-size="8.5" fill="#7b8794">{ATTRIBUTION}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Render docs/figures")
    ap.add_argument("--output-dir", default=str(REPO_ROOT / "examples" / "output"))
    ap.add_argument("--figures-dir", default=str(REPO_ROOT / "docs" / "figures"))
    ap.add_argument("--architecture-only", action="store_true")
    args = ap.parse_args(argv)
    figures = Path(args.figures_dir)
    figures.mkdir(parents=True, exist_ok=True)
    arch = figures / "architecture.svg"
    arch.write_text(architecture_svg(), encoding="utf-8")
    written = [arch]
    if not args.architecture_only:
        from attention_qelm_gwo_iiot_lab.figures import render_all_charts
        written += render_all_charts(args.output_dir, figures)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

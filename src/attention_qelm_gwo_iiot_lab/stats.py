"""Multi-seed aggregation and a Friedman / Holm helper that refuses underpowered input."""
from __future__ import annotations

import numpy as np
from scipy import stats

MIN_METHODS = 2
MIN_SEEDS = 3


def aggregate_seeds(values: list[float]) -> dict:
    """Mean and SAMPLE standard deviation (ddof=1); SD is None with a single value."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no values to aggregate")
    return {"n": int(arr.size), "mean": float(arr.mean()),
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "min": float(arr.min()), "max": float(arr.max()), "values": [float(v) for v in arr]}


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjustment, returned in the original order."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return [float(v) for v in adjusted]


def friedman_holm(results: dict[str, list[float]]) -> dict:
    """Friedman test across methods (blocks = seeds) with Holm-adjusted paired t-tests.

    `results` maps method name -> per-seed metric list, all of equal length. Refuses to run
    with fewer than 2 methods or fewer than 3 seeds, because the test would be meaningless.
    """
    if len(results) < MIN_METHODS:
        raise ValueError(f"Friedman test needs at least {MIN_METHODS} methods, got {len(results)}")
    lengths = {len(v) for v in results.values()}
    if len(lengths) != 1:
        raise ValueError(f"all methods must have the same number of seeds, got {sorted(lengths)}")
    n_seeds = lengths.pop()
    if n_seeds < MIN_SEEDS:
        raise ValueError(f"Friedman test needs at least {MIN_SEEDS} seeds per method, got {n_seeds}")
    names = list(results)
    matrix = np.array([results[n] for n in names], dtype=np.float64)   # methods x seeds
    if len(names) == 2:
        # Friedman with two groups degenerates; report the paired test as the omnibus.
        t, p = stats.ttest_rel(matrix[0], matrix[1])
        omnibus = {"test": "paired_t (two methods)", "statistic": float(t), "p_value": float(p), "df": n_seeds - 1}
    else:
        chi2, p = stats.friedmanchisquare(*matrix)
        omnibus = {"test": "friedman", "statistic": float(chi2), "p_value": float(p), "df": len(names) - 1}
    ranks = np.mean([stats.rankdata(-matrix[:, s]) for s in range(n_seeds)], axis=0)
    pairs, raw = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if np.allclose(matrix[i], matrix[j]):
                pv = 1.0
            else:
                pv = float(stats.ttest_rel(matrix[i], matrix[j]).pvalue)
            pairs.append((names[i], names[j]))
            raw.append(pv if np.isfinite(pv) else 1.0)
    adjusted = holm_adjust(raw)
    return {
        "n_methods": len(names), "n_seeds": n_seeds, "omnibus": omnibus,
        "mean_rank": {n: float(r) for n, r in zip(names, ranks)},
        "pairwise": [{"a": a, "b": b, "p_raw": pr, "p_holm": ph}
                     for (a, b), pr, ph in zip(pairs, raw, adjusted)],
        "note": "computed on synthetic fixture runs; not a benchmark result",
    }

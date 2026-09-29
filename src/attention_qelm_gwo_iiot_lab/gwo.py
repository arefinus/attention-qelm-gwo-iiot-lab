"""Grey Wolf Optimizer over a mixed 7-dimensional hyperparameter space (paper Table 3).

Positions live in an internal continuous representation: integers and floats in their
native range, log-scale dims as log10, categoricals as a float index. Decoding clips,
rounds integers and categorical indices, and exponentiates log dims. Fitness is any
callable of the decoded parameter dict; in this repository it is the k-fold macro-F1 on
the TRAINING partition only (see search.py). The optimizer is seeded, caches evaluations
of identical decoded configurations, and stops early when the best fitness improves by
less than `early_stop_tol` for `early_stop_patience` consecutive iterations.

GWO is a training-time selection procedure. It is kept separate from the encoder's Adam
optimizer both in code and in the architecture figure.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

KINDS = ("int", "float", "log", "cat")


@dataclass(frozen=True)
class Dimension:
    name: str
    kind: str
    low: float = 0.0
    high: float = 1.0
    choices: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"dimension {self.name!r}: kind must be one of {KINDS}")
        if self.kind == "cat" and len(self.choices) < 2:
            raise ValueError(f"dimension {self.name!r}: cat needs at least two choices")
        if self.kind != "cat" and not self.high > self.low:
            raise ValueError(f"dimension {self.name!r}: high must exceed low")
        if self.kind == "log" and self.low <= 0:
            raise ValueError(f"dimension {self.name!r}: log dims need low > 0")

    @property
    def internal_bounds(self) -> tuple[float, float]:
        if self.kind == "cat":
            return 0.0, float(len(self.choices) - 1)
        if self.kind == "log":
            return float(np.log10(self.low)), float(np.log10(self.high))
        return float(self.low), float(self.high)

    def decode(self, v: float) -> int | float | str:
        lo, hi = self.internal_bounds
        v = float(np.clip(v, lo, hi))
        if self.kind == "int":
            return int(round(v))
        if self.kind == "cat":
            return self.choices[int(round(v))]
        if self.kind == "log":
            return float(10.0 ** v)
        return v


class SearchSpace:
    def __init__(self, dims: list[Dimension]) -> None:
        if not dims:
            raise ValueError("search space needs at least one dimension")
        self.dims = list(dims)
        b = np.array([d.internal_bounds for d in self.dims], dtype=np.float64)
        self.lower, self.upper = b[:, 0], b[:, 1]

    @classmethod
    def from_dict(cls, space: dict) -> "SearchSpace":
        dims = []
        for name, spec in space.items():
            kind = spec["type"]
            if kind == "cat":
                dims.append(Dimension(name, kind, choices=tuple(str(c) for c in spec["choices"])))
            else:
                dims.append(Dimension(name, kind, low=float(spec["low"]), high=float(spec["high"])))
        return cls(dims)

    @property
    def n_dims(self) -> int:
        return len(self.dims)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(self.lower, self.upper, size=(n, self.n_dims))

    def decode(self, x: np.ndarray) -> dict:
        return {d.name: d.decode(v) for d, v in zip(self.dims, x)}


@dataclass(frozen=True)
class GWOConfig:
    n_wolves: int = 20
    max_iter: int = 50
    early_stop_tol: float = 1e-4
    early_stop_patience: int = 10
    seed: int = 42

    @classmethod
    def from_dict(cls, d: dict) -> "GWOConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})


def a_schedule(t: int, t_max: int) -> float:
    """a(t) = 2 (1 - t / T_max), decreasing linearly from 2 to 0."""
    return 2.0 * (1.0 - t / float(t_max))


@dataclass
class GWOResult:
    best_params: dict
    best_fitness: float
    trace: list[dict] = field(default_factory=list)
    n_evaluations: int = 0
    n_cache_hits: int = 0
    converged_iter: int | None = None
    seconds: float = 0.0

    n_wolves: int = 0
    max_iter: int = 0

    def to_dict(self) -> dict:
        return {"best_params": self.best_params, "best_fitness": self.best_fitness,
                "n_wolves": self.n_wolves, "max_iter": self.max_iter,
                "n_evaluations": self.n_evaluations, "n_cache_hits": self.n_cache_hits,
                "converged_iter": self.converged_iter, "seconds": self.seconds, "trace": self.trace}


class GreyWolfOptimizer:
    """Maximises `fitness(params)`; internally tracks -fitness as the prey distance."""

    def __init__(self, space: SearchSpace, cfg: GWOConfig) -> None:
        if cfg.n_wolves < 3:
            raise ValueError("GWO needs at least 3 wolves (alpha, beta, delta)")
        if cfg.max_iter < 1:
            raise ValueError("max_iter must be >= 1")
        self.space = space
        self.cfg = cfg
        self._cache: dict[str, float] = {}
        self.n_evaluations = 0
        self.n_cache_hits = 0

    def _evaluate(self, fitness: Callable[[dict], float], x: np.ndarray) -> float:
        params = self.space.decode(x)
        key = json.dumps(params, sort_keys=True)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        val = float(fitness(params))
        if not np.isfinite(val):
            val = -np.inf
        self._cache[key] = val
        self.n_evaluations += 1
        return val

    def run(self, fitness: Callable[[dict], float]) -> GWOResult:
        rng = np.random.default_rng(self.cfg.seed)
        t0 = time.perf_counter()
        X = self.space.sample(rng, self.cfg.n_wolves)
        F = np.array([self._evaluate(fitness, x) for x in X])
        trace: list[dict] = []
        best_hist: list[float] = []
        global_best_x = X[int(np.argmax(F))].copy()
        global_best_f = float(F.max())
        converged: int | None = None
        for t in range(self.cfg.max_iter):
            order = np.argsort(-F)
            alpha, beta, delta = X[order[0]].copy(), X[order[1]].copy(), X[order[2]].copy()
            a = a_schedule(t, self.cfg.max_iter)
            new_X = np.empty_like(X)
            for i in range(self.cfg.n_wolves):
                parts = []
                for leader in (alpha, beta, delta):
                    r1 = rng.uniform(size=self.space.n_dims)
                    r2 = rng.uniform(size=self.space.n_dims)
                    A = 2.0 * a * r1 - a
                    C = 2.0 * r2
                    D = np.abs(C * leader - X[i])
                    parts.append(leader - A * D)
                new_X[i] = self.space.clip(np.mean(parts, axis=0))
            X = new_X
            F = np.array([self._evaluate(fitness, x) for x in X])
            # elitism: the best position ever seen is never lost from the pack
            if float(F.max()) < global_best_f:
                worst = int(np.argmin(F))
                X[worst], F[worst] = global_best_x, global_best_f
            best_i = int(np.argmax(F))
            if float(F[best_i]) > global_best_f:
                global_best_f, global_best_x = float(F[best_i]), X[best_i].copy()
            best_hist.append(global_best_f)
            finite = F[np.isfinite(F)]
            trace.append({"iteration": t + 1, "a": a, "best_fitness": global_best_f,
                          "iter_best_fitness": float(F[best_i]),
                          "mean_fitness": float(np.mean(finite)) if len(finite) else None,
                          "best_params": self.space.decode(global_best_x),
                          "n_evaluations": self.n_evaluations, "n_cache_hits": self.n_cache_hits})
            p = self.cfg.early_stop_patience
            if len(best_hist) > p and best_hist[-1] - best_hist[-1 - p] < self.cfg.early_stop_tol:
                converged = t + 1
                break
        best_key = max(self._cache, key=self._cache.__getitem__)
        return GWOResult(best_params=json.loads(best_key), best_fitness=self._cache[best_key], trace=trace,
                         n_evaluations=self.n_evaluations, n_cache_hits=self.n_cache_hits,
                         converged_iter=converged, seconds=time.perf_counter() - t0,
                         n_wolves=self.cfg.n_wolves, max_iter=self.cfg.max_iter)

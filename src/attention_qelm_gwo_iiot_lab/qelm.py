"""Quantum-inspired extreme learning machine (numpy).

"Quantum-inspired" here means classical trigonometric random features: fixed hidden
weights W_ij = rho * cos(theta_ij) + (1 - rho) * sin(theta_ij) with theta_ij ~ U(0, 2 pi).
No quantum hardware, no quantum circuit, no quantum advantage is involved or claimed.

Output weights are solved in closed form as a ridge problem
    beta = (H^T H + (lambda1 + lambda2) I)^-1 H^T T
using a stable linear solve (never an explicit inverse). The paper names the first term
"L1"; it is a second diagonal quadratic term, so it is identical to ridge with
lambda = lambda1 + lambda2 (audit item 3). A true L1 penalty is available as a clearly
separate extension in elastic_net.py.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.special import erf

ACTIVATIONS = ("sigmoid", "tanh", "relu", "gelu")
GLOBAL_ANGLE_MODES = ("shift", "none")
WEIGHT_SCALES = ("fan_in", "none")


def activate(name: str, z: np.ndarray) -> np.ndarray:
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-z))
    if name == "tanh":
        return np.tanh(z)
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "gelu":
        return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))
    raise ValueError(f"unknown activation {name!r}; allowed {ACTIVATIONS}")


@dataclass(frozen=True)
class QELMConfig:
    n_layers: int = 3
    n_hidden: int = 512
    activation: str = "gelu"
    lambda1: float = 3.2e-4
    lambda2: float = 1.7e-3
    theta: float = 1.84
    rho: float = 0.63
    global_angle_mode: str = "shift"
    weight_scale: str = "fan_in"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        if self.n_hidden < 1:
            raise ValueError("n_hidden must be >= 1")
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if not 0.0 <= self.rho <= 1.0:
            raise ValueError("rho must be in [0, 1]")
        if self.lambda1 < 0 or self.lambda2 < 0:
            raise ValueError("lambda1 and lambda2 must be non-negative")
        if self.global_angle_mode not in GLOBAL_ANGLE_MODES:
            raise ValueError(f"global_angle_mode must be one of {GLOBAL_ANGLE_MODES}")
        if self.weight_scale not in WEIGHT_SCALES:
            raise ValueError(f"weight_scale must be one of {WEIGHT_SCALES}")

    @classmethod
    def from_dict(cls, d: dict) -> "QELMConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})

    @property
    def ridge_lambda(self) -> float:
        return float(self.lambda1 + self.lambda2)


def sample_angles(rng: np.random.Generator, shape: tuple[int, ...], theta: float, mode: str) -> np.ndarray:
    """theta_ij ~ U(0, 2 pi), shifted by the global angle when mode == 'shift'.

    Assumption (audit item 5): the paper gives a global theta as a searched hyperparameter
    without stating its role. In 'shift' mode it offsets the sampling window to
    [theta, theta + 2 pi); in 'none' mode it is ignored.
    """
    base = rng.uniform(0.0, 2.0 * np.pi, size=shape)
    return base + float(theta) if mode == "shift" else base


def quantum_inspired_init(rng: np.random.Generator, fan_in: int, fan_out: int, theta: float,
                          rho: float, mode: str = "shift", scale: str = "fan_in"
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Fixed weights W (fan_in, fan_out) and biases b (fan_out,)."""
    angles = sample_angles(rng, (fan_in, fan_out), theta, mode)
    W = rho * np.cos(angles) + (1.0 - rho) * np.sin(angles)
    b_angles = sample_angles(rng, (fan_out,), theta, mode)
    b = rho * np.cos(b_angles) + (1.0 - rho) * np.sin(b_angles)
    if scale == "fan_in":
        W = W / np.sqrt(fan_in)
    return W.astype(np.float64), b.astype(np.float64)


class QELM:
    """Fixed random hidden layers plus a closed-form ridge output layer."""

    def __init__(self, cfg: QELMConfig, input_dim: int, n_outputs: int) -> None:
        self.cfg = cfg
        self.input_dim = int(input_dim)
        self.n_outputs = int(n_outputs)
        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        self.beta: np.ndarray | None = None
        self._HtH: np.ndarray | None = None
        self._HtT: np.ndarray | None = None
        self._n_seen = 0
        self.solver_info: dict = {}
        self._init_hidden()

    # ---------------------------------------------------------- hidden
    def _init_hidden(self) -> None:
        rng = np.random.default_rng(self.cfg.seed)
        fan_in = self.input_dim
        self.layers = []
        for _ in range(self.cfg.n_layers):
            W, b = quantum_inspired_init(rng, fan_in, self.cfg.n_hidden, self.cfg.theta, self.cfg.rho,
                                         self.cfg.global_angle_mode, self.cfg.weight_scale)
            self.layers.append((W, b))
            fan_in = self.cfg.n_hidden

    @property
    def hidden_dim(self) -> int:
        return self.cfg.n_hidden + 1  # plus bias column for the output layer

    def hidden(self, X: np.ndarray) -> np.ndarray:
        h = np.asarray(X, dtype=np.float64)
        if h.ndim != 2 or h.shape[1] != self.input_dim:
            raise ValueError(f"expected (N, {self.input_dim}) input, got {h.shape}")
        for W, b in self.layers:
            h = activate(self.cfg.activation, h @ W + b)
        return np.hstack([h, np.ones((h.shape[0], 1))])

    def hidden_trace(self, X: np.ndarray) -> list[dict]:
        h = np.asarray(X, dtype=np.float64)
        trace = [{"stage": "qelm_input", "shape": list(h.shape)}]
        for i, (W, b) in enumerate(self.layers, start=1):
            h = activate(self.cfg.activation, h @ W + b)
            trace.append({"stage": f"fixed_hidden_{i}_{self.cfg.activation}", "shape": list(h.shape),
                          "trainable": False})
        trace.append({"stage": "hidden_plus_bias", "shape": [h.shape[0], h.shape[1] + 1]})
        trace.append({"stage": "solved_output", "shape": [h.shape[0], self.n_outputs], "trainable": "closed-form"})
        return trace

    # ------------------------------------------------------------- fit
    def reset_accumulators(self) -> None:
        d = self.hidden_dim
        self._HtH = np.zeros((d, d), dtype=np.float64)
        self._HtT = np.zeros((d, self.n_outputs), dtype=np.float64)
        self._n_seen = 0

    def accumulate(self, X_batch: np.ndarray, T_batch: np.ndarray) -> None:
        if self._HtH is None or self._HtT is None:
            self.reset_accumulators()
        H = self.hidden(X_batch)
        T = np.asarray(T_batch, dtype=np.float64)
        if T.shape != (H.shape[0], self.n_outputs):
            raise ValueError(f"targets must be one-hot (N, {self.n_outputs}), got {T.shape}")
        self._HtH += H.T @ H
        self._HtT += H.T @ T
        self._n_seen += H.shape[0]

    def solve(self) -> np.ndarray:
        if self._HtH is None or self._HtT is None or self._n_seen == 0:
            raise RuntimeError("no data accumulated; call accumulate() or fit() first")
        A = self._HtH + self.cfg.ridge_lambda * np.eye(self.hidden_dim)
        try:
            beta = linalg.solve(A, self._HtT, assume_a="pos")
            method = "scipy.linalg.solve(assume_a=pos)"
        except (linalg.LinAlgError, ValueError):
            beta, *_ = linalg.lstsq(A, self._HtT)
            method = "scipy.linalg.lstsq"
        self.beta = np.asarray(beta, dtype=np.float64)
        self.solver_info = {"method": method, "ridge_lambda": self.cfg.ridge_lambda,
                            "n_rows": int(self._n_seen), "hidden_dim": self.hidden_dim}
        return self.beta

    def fit(self, X: np.ndarray, T: np.ndarray, batch_size: int = 4096) -> "QELM":
        self.reset_accumulators()
        for i in range(0, len(X), batch_size):
            self.accumulate(X[i:i + batch_size], T[i:i + batch_size])
        self.solve()
        return self

    # --------------------------------------------------------- predict
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.beta is None:
            raise RuntimeError("model not solved; call fit() first")
        return self.hidden(X) @ self.beta

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax of the linear scores: a monotone squashing for ranking, not calibrated."""
        s = self.decision_function(X)
        s = s - s.max(axis=1, keepdims=True)
        e = np.exp(s)
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1)

    # --------------------------------------------------------- persist
    def save(self, path: str | Path) -> None:
        if self.beta is None:
            raise RuntimeError("cannot save an unsolved model")
        arrays = {f"W{i}": W for i, (W, _) in enumerate(self.layers)}
        arrays.update({f"b{i}": b for i, (_, b) in enumerate(self.layers)})
        arrays["beta"] = self.beta
        arrays["meta"] = np.array(json.dumps({"config": asdict(self.cfg), "input_dim": self.input_dim,
                                              "n_outputs": self.n_outputs, "solver_info": self.solver_info}))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(path), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "QELM":
        with np.load(str(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            model = cls(QELMConfig(**meta["config"]), meta["input_dim"], meta["n_outputs"])
            model.layers = [(z[f"W{i}"], z[f"b{i}"]) for i in range(model.cfg.n_layers)]
            model.beta = z["beta"]
            model.solver_info = meta.get("solver_info", {})
        return model


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    T = np.zeros((len(y), n_classes), dtype=np.float64)
    T[np.arange(len(y)), y] = 1.0
    return T

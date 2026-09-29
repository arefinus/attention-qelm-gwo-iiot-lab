"""Attention encoder (PyTorch port of the paper's TensorFlow 2.11 description).

Tensor path for the default `expand64to256` tokenisation, batch B:
  x            (B, 48)
  fc1 + GELU   (B, 128)
  fc2 + GELU   (B, 64)       bottleneck
  expand       (B, 256)      EXPLICIT learned 64 -> 256 expansion (audit item 1)
  reshape      (B, 8, 32)    8 tokens x 32 dims
  + feature-index embedding (1, 8, 32), learned
  Q, K, V      (B, 8, 8*32)  8 heads, d_k = 32 per head
  attention    (B, 8, 8, 8)  scores per head (B, heads, tokens, tokens)
  concat       (B, 8, 256)
  W_O          (B, 8, 32)
  residual + LayerNorm        (B, 8, 32)
  FFN x4 GELU  (B, 8, 128) -> (B, 8, 32)
  residual + LayerNorm        (B, 8, 32)
  flatten      (B, 256)      frozen feature vector consumed by the QELM
  head         (B, C)        temporary softmax head used ONLY in phase 1

`direct48to256` projects the 48 inputs straight to 256 and skips the 128/64 layers for
tokenisation; it exists to make the ambiguity testable, not as the paper's design.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

TOKENISATIONS = ("expand64to256", "direct48to256")


@dataclass(frozen=True)
class EncoderConfig:
    input_dim: int = 48
    hidden_dim: int = 128
    bottleneck_dim: int = 64
    tokenisation: str = "expand64to256"
    n_tokens: int = 8
    token_dim: int = 32
    n_heads: int = 8
    head_dim: int = 32
    ffn_multiplier: int = 4
    dropout: float = 0.10

    @classmethod
    def from_dict(cls, d: dict) -> "EncoderConfig":
        keys = cls.__dataclass_fields__.keys()
        return cls(**{k: d[k] for k in keys if k in d})


class AttentionEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig, n_classes: int) -> None:
        super().__init__()
        if cfg.tokenisation not in TOKENISATIONS:
            raise ValueError(f"tokenisation must be one of {TOKENISATIONS}, got {cfg.tokenisation!r}")
        self.cfg = cfg
        self.n_classes = int(n_classes)
        seq = cfg.n_tokens * cfg.token_dim
        self.fc1 = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.bottleneck_dim)
        src = cfg.bottleneck_dim if cfg.tokenisation == "expand64to256" else cfg.input_dim
        self.expand = nn.Linear(src, seq)
        self.feature_index_embedding = nn.Parameter(torch.zeros(1, cfg.n_tokens, cfg.token_dim))
        nn.init.normal_(self.feature_index_embedding, std=0.02)
        inner = cfg.n_heads * cfg.head_dim
        self.q = nn.Linear(cfg.token_dim, inner)
        self.k = nn.Linear(cfg.token_dim, inner)
        self.v = nn.Linear(cfg.token_dim, inner)
        self.w_o = nn.Linear(inner, cfg.token_dim)
        self.ln1 = nn.LayerNorm(cfg.token_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.token_dim, cfg.ffn_multiplier * cfg.token_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_multiplier * cfg.token_dim, cfg.token_dim),
        )
        self.ln2 = nn.LayerNorm(cfg.token_dim)
        self.drop = nn.Dropout(cfg.dropout)
        self.act = nn.GELU()
        self.head = nn.Linear(seq, self.n_classes)
        self._frozen = False

    @property
    def feature_dim(self) -> int:
        return self.cfg.n_tokens * self.cfg.token_dim

    # ------------------------------------------------------------ forward
    def forward_trace(self, x: torch.Tensor) -> tuple[torch.Tensor, list[dict]]:
        cfg = self.cfg
        trace: list[dict] = [{"stage": "input", "shape": list(x.shape)}]
        if cfg.tokenisation == "expand64to256":
            h = self.act(self.fc1(x))
            trace.append({"stage": "fc1_gelu", "shape": list(h.shape)})
            h = self.act(self.fc2(h))
            trace.append({"stage": "bottleneck_gelu", "shape": list(h.shape)})
            h = self.expand(h)
            trace.append({"stage": "expand_64_to_256", "shape": list(h.shape)})
        else:
            h = self.expand(x)
            trace.append({"stage": "direct_48_to_256", "shape": list(h.shape)})
        tokens = h.view(x.shape[0], cfg.n_tokens, cfg.token_dim) + self.feature_index_embedding
        trace.append({"stage": "tokens_plus_feature_index_embedding", "shape": list(tokens.shape)})
        B, T, _ = tokens.shape
        q = self.q(tokens).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k(tokens).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        v = self.v(tokens).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(cfg.head_dim)
        trace.append({"stage": "attention_scores", "shape": list(scores.shape)})
        attn = self.drop(torch.softmax(scores, dim=-1))
        ctx = (attn @ v).transpose(1, 2).reshape(B, T, cfg.n_heads * cfg.head_dim)
        trace.append({"stage": "heads_concat", "shape": list(ctx.shape)})
        out = self.w_o(ctx)
        trace.append({"stage": "w_o", "shape": list(out.shape)})
        tokens = self.ln1(tokens + self.drop(out))
        trace.append({"stage": "residual_layernorm_1", "shape": list(tokens.shape)})
        tokens = self.ln2(tokens + self.drop(self.ffn(tokens)))
        trace.append({"stage": "ffn_residual_layernorm_2", "shape": list(tokens.shape)})
        feats = tokens.reshape(B, T * cfg.token_dim)
        trace.append({"stage": "flatten", "shape": list(feats.shape)})
        return feats, trace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats, _ = self.forward_trace(x)
        return feats

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward(x))

    # ------------------------------------------------------------- freeze
    def freeze(self) -> "AttentionEncoder":
        for p in self.parameters():
            p.requires_grad_(False)
            p.grad = None
        self.eval()
        self._frozen = True
        return self

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def embed(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Frozen feature extraction to numpy (no gradient, eval mode)."""
        was_training = self.training
        self.eval()
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                xb = torch.as_tensor(np.asarray(X[i:i + batch_size], dtype=np.float32))
                chunks.append(self.forward(xb).cpu().numpy())
        if was_training and not self._frozen:
            self.train()
        return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, self.feature_dim), np.float32)

    def n_parameters(self, include_head: bool = True) -> int:
        total = sum(p.numel() for p in self.parameters())
        if not include_head:
            total -= sum(p.numel() for p in self.head.parameters())
        return int(total)


# ---------------------------------------------------------------- loss
def inverse_frequency_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    """w_c = N / (C * n_c), rescaled to mean 1 over the classes present."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    w = np.zeros(n_classes, dtype=np.float64)
    present = counts > 0
    w[present] = len(y) / (present.sum() * counts[present])
    w[present] /= w[present].mean()
    return w


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0,
               alpha: float = 0.25, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    """Multiclass focal loss: -alpha * w_c * (1 - p_t)^gamma * log p_t, averaged."""
    logp = torch.log_softmax(logits, dim=-1)
    logp_t = logp.gather(1, targets.view(-1, 1)).squeeze(1)
    p_t = logp_t.exp()
    w = torch.full_like(p_t, float(alpha))
    if class_weights is not None:
        w = w * class_weights.to(logits.dtype)[targets]
    return (-(w * (1.0 - p_t).pow(gamma) * logp_t)).mean()


# -------------------------------------------------------------- training
@dataclass(frozen=True)
class TrainConfig:
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    batch_size: int = 32
    max_epochs: int = 100
    patience: int = 15
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    class_weighting: str = "inverse_frequency"

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        keys = cls.__dataclass_fields__.keys()
        vals = {k: d[k] for k in keys if k in d}
        if "betas" in vals:
            vals["betas"] = tuple(float(b) for b in vals["betas"])
        return cls(**vals)


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0))


def train_encoder(model: AttentionEncoder, X_tr: np.ndarray, y_tr: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray, cfg: TrainConfig, seed: int = 42
                  ) -> dict:
    """Phase 1: supervised training with the temporary head.

    Early stopping monitors the INNER validation loss only (min, patience `cfg.patience`).
    The function has no test argument by design. Returns a history dict with per-epoch
    train_loss, val_loss and val_macro_f1, the best epoch, and the wall time.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_classes = model.n_classes
    weights = None
    if cfg.class_weighting == "inverse_frequency":
        weights = torch.as_tensor(inverse_frequency_weights(y_tr, n_classes), dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, betas=cfg.betas)
    Xt = torch.as_tensor(np.asarray(X_tr, dtype=np.float32))
    yt = torch.as_tensor(np.asarray(y_tr, dtype=np.int64))
    Xv = torch.as_tensor(np.asarray(X_val, dtype=np.float32))
    yv = torch.as_tensor(np.asarray(y_val, dtype=np.int64))
    gen = torch.Generator().manual_seed(seed)
    history: list[dict] = []
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_epoch = 0
    bad = 0
    t0 = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt), generator=gen)
        total, n = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            loss = focal_loss(model.logits(Xt[idx]), yt[idx], cfg.focal_gamma, cfg.focal_alpha, weights)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(idx)
            n += len(idx)
        model.eval()
        with torch.no_grad():
            vlog = model.logits(Xv)
            vloss = float(focal_loss(vlog, yv, cfg.focal_gamma, cfg.focal_alpha, weights).item())
            vf1 = _macro_f1(yv.numpy(), vlog.argmax(1).numpy(), n_classes)
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": vloss,
                        "val_macro_f1": vf1, "elapsed_s": time.perf_counter() - t0})
        if vloss < best_val - 1e-12:
            best_val, best_epoch, bad = vloss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return {"history": history, "best_epoch": best_epoch, "best_val_loss": best_val,
            "epochs_run": len(history), "monitor": "val_loss", "seconds": time.perf_counter() - t0,
            "train_config": asdict(cfg)}


# ------------------------------------------------------------ checkpoints
def save_encoder(model: AttentionEncoder, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": asdict(model.cfg),
                "n_classes": model.n_classes}, str(path))


def load_encoder(path: str | Path) -> AttentionEncoder:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    model = AttentionEncoder(EncoderConfig(**payload["config"]), int(payload["n_classes"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model

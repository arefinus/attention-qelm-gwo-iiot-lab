# Method

## Overview

Two-phase training. Phase 1 fits an attention encoder end to end with a temporary softmax
head and freezes it. Phase 2 treats the frozen 256-dimensional feature as the input of an
extreme learning machine whose hidden layers are fixed random trigonometric features and
whose output layer is solved in closed form. A Grey Wolf Optimizer chooses the ELM
hyperparameters using k-fold macro-F1 on the training partition. The architecture drawing
is `docs/figures/architecture.svg`.

## Encoder (`encoder.py`)

```
x (B,48) -> Linear(48,128) GELU -> Linear(128,64) GELU -> Linear(64,256)
  -> view (B,8,32) + feature_index_embedding (1,8,32)
  -> Q,K,V: Linear(32, 8*32) each -> scores (B,8,8,8) -> softmax -> concat (B,8,256)
  -> W_O: Linear(256,32) -> residual + LayerNorm(32)
  -> FFN: Linear(32,128) GELU Linear(128,32) -> residual + LayerNorm(32)
  -> flatten (B,256) -> [phase 1 only] Linear(256,C)
```

Dropout 0.10 on attention weights, after W_O, inside the FFN and after the FFN. The
`tokenisation` option selects `expand64to256` (default, above) or `direct48to256`, which
projects the 48 inputs straight to 256 and skips the 128/64 layers for tokenisation.

Loss: `-alpha * w_c * (1 - p_t)^gamma * log p_t`, gamma 2, alpha 0.25, `w_c` the
inverse-frequency class weight rescaled to mean 1. Optimizer Adam lr 1e-3, betas (0.9, 0.999).
Early stopping monitors the inner-validation focal loss (min) with the configured patience
and restores the best weights. `train_encoder` has no test argument.

After phase 1 `freeze()` sets `requires_grad=False` on every parameter, clears gradients and
switches to eval mode. `tests/test_encoder.py::test_frozen_encoder_unchanged_and_gradient_free_after_qelm_solve`
hashes the state dict before and after the ELM solve.

## QELM (`qelm.py`)

Hidden layer l has weights `W^(l)` of shape (fan_in, n_h) and bias `b^(l)` with

```
theta_ij ~ U(0, 2 pi) (+ theta_global in 'shift' mode)
W_ij = rho * cos(theta_ij) + (1 - rho) * sin(theta_ij)    [divided by sqrt(fan_in) when weight_scale = fan_in]
h^(l) = f(h^(l-1) W^(l) + b^(l)),   h^(0) = frozen encoder feature
H = [h^(L), 1]
```

Output weights: `beta = (H'H + (lambda1 + lambda2) I)^-1 H'T` for one-hot targets T, computed
by `scipy.linalg.solve(..., assume_a="pos")` with an `lstsq` fallback, never an explicit
inverse. `H'H` and `H'T` are accumulated in batches so the full H is never materialised.
Scores are `H beta`; prediction is the argmax; `predict_proba` is a softmax of the scores for
ranking only. The model persists to `.npz` (weights, biases, beta, JSON metadata), loaded with
`allow_pickle=False`.

The paper calls the first penalty term "L1". In the formula it is a second diagonal
quadratic term, so `lambda1` and `lambda2` are simply summed into one ridge parameter.
A true L1 + L2 output layer is provided in `elastic_net.py` (cyclic coordinate descent,
tested against `sklearn.linear_model.ElasticNet`) and selected with `qelm.solver: elastic_net_cd`.
It is an extension, not the paper's method.

## Grey Wolf Optimizer (`gwo.py`, `search.py`)

Seven dimensions (Table 3): `n_layers` int [1,5], `n_hidden` int [64,1024], `activation` in
{sigmoid, tanh, relu, gelu}, `lambda1` and `lambda2` log-uniform [1e-6, 1e-1], `theta`
[0, 2 pi], `rho` [0, 1]. Positions are continuous internally (log10 for log dims, float index
for the categorical); decoding clips, rounds integers and the categorical index, and
exponentiates. Update: for each wolf and each leader in (alpha, beta, delta),
`A = 2 a r1 - a`, `C = 2 r2`, `X_k = leader - A |C leader - X|`, new position = mean of the
three, clipped; `a(t) = 2 (1 - t / T_max)`. Elitism keeps the best position ever seen in the
pack. Identical decoded configurations are cached. Early stop when the best fitness improves
by less than 1e-4 over 10 consecutive iterations.

Fitness: for each inner fold, build a QELM with the candidate parameters on the training
embeddings of the fold, solve, and score macro-F1 on the held-out inner fold; average.
`qelm_cv_fitness` validates that all fold indices lie inside the training partition and has
no parameter through which test rows could be passed.

GWO is an offline, training-time selection loop. It is a separate module from the encoder's
Adam optimizer and is drawn as a side loop in the architecture figure.

## Binary and multiclass tasks

`configs/*_binary.yaml` and `configs/*_multiclass.yaml` are separate configs. The binary
model is trained from scratch on Normal/Attack labels (two output units, two-class focal
loss). The `evaluate` subcommand refuses a predictions file whose score columns do not match
the task's class order, so the two cannot be silently swapped.

## Settings taken from the source paper

Stored in `configs/paper_multiclass.yaml`, `configs/paper_binary.yaml` and
`configs/gwo_paper.yaml`; also listed under `settings_from_paper` in
`evidence/paper_reported/qelm.json`:

- preprocessing: drop constant/duplicate/metadata columns, hash-dedup rows, median/mode
  imputation, drop above 40% missing, inf to column max, label encoding, Min-Max, 48 features
- split: stratified 80/20; 5-fold CV inside the training partition for the GWO fitness
- encoder: 48 -> 128 -> 64 -> 8 tokens x 32, 8 heads with d_k 32, FFN x4 GELU, dropout 0.1,
  learned feature-index embedding; Adam lr 1e-3, focal gamma 2 alpha 0.25 with
  inverse-frequency weights, batch 32, 100 epochs, patience 15
- QELM final: L 3, n_h 512, GELU, lambda1 3.2e-4, lambda2 1.7e-3, theta 1.84, rho 0.63
- GWO: N 20, T_max 50, a(t) = 2(1 - t/T_max), early stop 1e-4 over 10 iterations
- seeds: 7, 13, 21, 42, 99

The demo configs shorten these (batch 128, 20 epochs, patience 5, 2 inner folds, N 6, T 4,
n_hidden capped at 256, lighter sklearn baselines) so the whole demo runs on a CPU in a
couple of minutes. The demo values are therefore not a run of the paper's configuration.

## Baselines (`baselines/`)

Logistic regression (balanced class weights), random forest (balanced subsample) and
`HistGradientBoostingClassifier` (balanced), all on the same preprocessed features and split.
Profile `light` (demo) uses 50 trees / 50 boosting iterations, `standard` 100 / 100. The
paper's CNN-1D, LSTM, GRU, BiLSTM and DenseNet baselines are `proposed` and not implemented.

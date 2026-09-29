# Model card

## Model

Attention encoder (PyTorch) followed by a quantum-inspired extreme learning machine (numpy)
with a closed-form ridge output layer; hyperparameters of the ELM selected by a Grey Wolf
Optimizer on the training partition. Two separately trained variants: multiclass (15
classes, Table 1 order) and binary (Normal vs Attack).

"Quantum-inspired" is a naming convention inherited from the source paper. It denotes
classical trigonometric random features, `W_ij = rho cos(theta_ij) + (1 - rho) sin(theta_ij)`
with `theta_ij ~ U(0, 2 pi)` optionally offset by a global angle. No quantum hardware,
quantum circuit or quantum advantage is involved or claimed.

## Intended use

Laboratory study of the design on a user's own processed Edge-IIoTset table, and reading
of a shape-annotated implementation. Not a deployed intrusion detector. No claim is made
about real-world detection performance.

## Training data

Only the authored synthetic fixture was used in this repository. Real Edge-IIoTset runs
require the user to supply the data and fill the feature manifest (DATA_CARD.md).

## Architecture and shapes (batch B)

| Stage | Shape | Trainable |
|---|---|---|
| input | (B, 48) | |
| Linear 48 -> 128, GELU | (B, 128) | phase 1 |
| Linear 128 -> 64, GELU (bottleneck) | (B, 64) | phase 1 |
| Linear 64 -> 256 (explicit expansion, audit item 1) | (B, 256) | phase 1 |
| reshape + learned feature-index embedding | (B, 8, 32) | phase 1 |
| 8-head attention, d_k 32, concat 256, W_O -> 32 | (B, 8, 32) | phase 1 |
| residual + LayerNorm, FFN x4 GELU, residual + LayerNorm | (B, 8, 32) | phase 1 |
| flatten | (B, 256) | frozen after phase 1 |
| fixed hidden layers x L, n_h units, activation f | (B, n_h) | never |
| bias column, solved output beta | (B, C) | closed form |

Parameter counts are written by the demo to `examples/output/demo_trace.json`
(`encoder_trainable_parameters_without_head`, `qelm_fixed_hidden_parameters`,
`qelm_solved_output_parameters`) and depend on the searched `n_layers` and `n_hidden`. The
paper's Table 12 figure (0.21 M parameters) is `paper-reported` and is not reproduced here.

## Training procedure

Phase 1: Adam (lr 1e-3, betas 0.9/0.999), focal loss gamma 2, alpha 0.25 multiplied by
inverse-frequency class weights, batch 32 in the paper config (128 in the demo), early
stopping on the inner validation focal loss (patience 15 paper, 5 demo), best weights
restored, then frozen. Phase 2: embeddings of the training rows, GWO fitness = k-fold
macro-F1 on training embeddings, then one ridge solve on all training rows with
`scipy.linalg.solve(H'H + (lambda1 + lambda2) I, H'T)`, falling back to `lstsq`.

## PyTorch port of a TensorFlow 2.11 design: known differences

The paper's environment was TensorFlow 2.11 / Keras. This repository is a PyTorch port.
Even with identical hyperparameters, results would differ because:

- Optimizer defaults: Keras Adam uses epsilon 1e-7; `torch.optim.Adam` uses 1e-8. Weight
  decay defaults are zero in both, but the update ordering of bias correction differs slightly.
- Initialisation: Keras `Dense` defaults to Glorot uniform weights and zero biases;
  `torch.nn.Linear` uses Kaiming-uniform weights with a `1/sqrt(fan_in)` bound and
  uniform non-zero biases. The learned feature-index embedding here is normal(0, 0.02).
- GELU: `tf.keras.activations.gelu` defaults to the exact erf form but accepts
  `approximate=True` (tanh form), and the paper does not state which was used;
  `torch.nn.GELU()` here uses the exact erf form, as does the numpy QELM activation.
- LayerNorm epsilon: Keras 1e-3, PyTorch 1e-5.
- Dropout masks, shuffling and random number streams are not comparable across frameworks,
  so seeds {7, 13, 21, 42, 99} do not reproduce the paper's draws.

## Evaluation

Metrics with explicit class order; binary FPR = FP/(FP+TN) with Attack positive;
multiclass FPR = macro mean of one-vs-rest FPR (the paper does not define its multiclass
FPR). Per-class one-vs-rest ROC from the raw ELM scores. Five seeds with mean and sample SD.
Demo values in `examples/output/metrics.json` are labelled `demo`; paper values in
`evidence/paper_reported/qelm.json` are labelled `paper-reported`. No `reproduced` values exist.

## Limitations and risks

Tabular IDS results on Edge-IIoTset are known to be sensitive to identifier leakage,
placeholder encodings that reveal the protocol, and duplicate flows across splits. The
`audit` subcommand implements three checks for these; they are heuristics with stated
thresholds, not a verdict. The model's ranking scores are not calibrated probabilities.

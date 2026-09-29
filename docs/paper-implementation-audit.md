# Paper implementation audit

Source: "Attention-guided quantum-inspired extreme learning machine with grey wolf
optimization for rare-attack detection in IIoT networks", Discover Artificial Intelligence,
2026 (article in press), doi 10.1007/s44163-026-01704-3. This repository was written from
the article's text. The article's cited code repository
(https://github.com/drmohammedadnan110-ai/Attention-QELM-GWO-IIoT-IDS) was not audited or
reused, so none of the assumptions below were checked against it.

Every row states what the paper says, what this repository does, and the assumption made.
Items are numbered for cross-reference from code comments and config keys.

## Ambiguities and assumptions

| # | Topic | What the paper says | What this repository does | Assumption / status |
|---|---|---|---|---|
| 1 | Reshape 64 -> 8 x 32 | The encoder maps 48 inputs to 128, then to a 64-unit GELU bottleneck, which is "reshaped" to T = 8 tokens of d_k = 32 (Section 4.2). 8 x 32 = 256, so a 64-vector cannot be reshaped into it. | Inserts an explicit learned `Linear(64, 256)` before the reshape (`tokenisation: expand64to256`). An alternative `direct48to256` projects the 48 inputs straight to 256. Both record every tensor shape (`demo_trace.json`). | Implementation repair. Which projection the authors used is unknown; the extra layer adds 16,640 parameters that the paper's Table 12 count may or may not include. |
| 2 | Attention dimensions | "8-head attention with d_k = 32", followed by concatenation and W_O, residual LayerNorm, FFN x4. Token dimension after reshape is 32. | Per-token model dimension 32; each head projects 32 -> 32 (d_k), so Q/K/V are Linear(32, 256), concat is 256, W_O is Linear(256, 32). FFN is 32 -> 128 -> 32. Flatten gives 8 x 32 = 256. | Assumes d_k is per head and the residual stream stays at 32. If the authors meant d_model = 256 per token, the parameter count and the flatten size would differ. |
| 3 | "L1" regularisation term | beta = (H^T H + lambda1 I + lambda2 I)^-1 H^T T, with lambda1 described as an L1 term. | Treats lambda1 as a second diagonal quadratic term and solves one ridge problem with lambda = lambda1 + lambda2 (`qelm.py`). Provides a separate coordinate-descent elastic-net solver (`elastic_net.py`, `qelm.solver: elastic_net_cd`) as a clearly labelled extension. | The closed form in the paper is ridge; a true L1 penalty has no closed form. The paper's searched lambda1 and lambda2 are therefore redundant in the closed-form model. |
| 4 | Single vs multilayer ELM | The prose describes a single hidden layer; the final configuration reports L = 3 hidden layers with n_h = 512 and the GWO searches L in [1, 5]. | Implements L fixed random layers, activations propagated through all of them, bias column appended to the last, output solved on the last layer only (`n_layers` config, default 3). | Assumes the intermediate layers are also fixed and not solved layer-wise. |
| 5 | Global angle theta and weight init | W_ij = rho cos(theta_ij) + (1 - rho) sin(theta_ij), theta_ij ~ U(0, 2 pi); a global theta = 1.84 is also a searched hyperparameter with no stated role. No weight scaling is mentioned. | `global_angle_mode: shift` offsets the sampling window to [theta, theta + 2 pi); `none` ignores it. Biases use the same construction. `weight_scale: fan_in` divides weights by sqrt(fan_in) so three stacked layers do not saturate; `none` reproduces the raw formula. | Two assumptions: the meaning of the global angle, and the fan-in scaling, which the paper does not mention but without which GELU layers with 256 to 1024 inputs explode in scale. Both are config options. |
| 6 | 48-feature manifest | 48 features retained after dropping constant, duplicate and metadata columns; names not listed. | Ships `configs/feature_manifest_paper.yaml` with 48 placeholder slots and a strict adapter that rejects any other column set (never truncates). | The user must fill the manifest from their own preprocessing; raw release columns are not assumed equal. |
| 7 | Fitting scope across folds | Preprocessing is described once (Section 3.2); GWO fitness is 5-fold CV on the training partition; it is not stated whether imputation and scaling are refit inside each fold. | Preprocessing is fit once on the outer training partition; inner folds reuse it. Encoder early stopping uses inner fold 1; GWO uses all inner folds on the frozen embeddings. | Mild optimism inside the inner CV (scaling statistics see the inner validation rows). The outer test partition is untouched by any statistic. |
| 8 | Early stopping on test curves | Figures 5 and 6 plot a per-epoch TEST curve; early stopping with patience 15 is described without naming the monitored split. | Early stopping monitors only the inner validation loss; the test partition is evaluated once after training. `train_encoder` has no test argument, and a test asserts the history contains no test key. | If the paper stopped on the test curve, its reported numbers would be optimistically biased; this repository does not replicate that. |
| 9 | Binary task semantics | Binary results (Table 6) are reported alongside 15-class results; it is not stated whether the binary model was trained separately or derived from the multiclass output. | Separate configs and separate training runs (`*_binary.yaml`, two output units). `evaluate` refuses mismatched prediction files. | Assumes separate training; collapsing the multiclass output is not offered. |
| 10 | Shortcut / identifier checks and FPR definition | No identifier, placeholder-encoding or duplicate-flow check is reported. A multiclass FPR of 0.0079 is reported without a definition. | `sensitivity/` implements three audits (labelled demo on the fixture, proposed on real data). Multiclass FPR is the macro mean of one-vs-rest FP/(FP+TN); binary FPR is FP/(FP+TN) with Attack positive. | The audits are heuristics. The FPR definition is an assumption; a micro-averaged or Normal-class definition would give different values. |

Additional recorded item: the article reports the Friedman statistic as chi-square 50.00
(df 10, p 2.67e-7) in the abstract and Section 5.7 but as 54.98 in the contributions list.
Both cannot be right; `evidence/paper_reported/qelm.json` records the inconsistency without
choosing.

## Port from TensorFlow 2.11 to PyTorch

The paper's environment was Python 3.9, TensorFlow 2.11, scikit-learn 1.2. This repository
uses PyTorch (CPU) and numpy. Differences in optimizer epsilon, layer initialisation, GELU
form and LayerNorm epsilon are listed in MODEL_CARD.md. Seeds are not comparable across
frameworks.

## Inventory of the article's figures and tables

"Redraw" means this repository produces an original figure of the same kind from its own
outputs; "regenerate" means the numbers are recomputed by this code on the fixture;
"transcribe" means values are stored as `paper-reported` only; "no" means nothing here
corresponds to it.

| Item | What the article shows | This repository |
|---|---|---|
| Figure 1 | Overall framework: preprocessing, attention encoder, QELM, GWO | Redraw: `docs/figures/architecture.svg`, original SVG with GWO drawn as an offline side loop and explicit shapes |
| Figure 2 | Encoder block diagram (tokenisation, attention, FFN) | Redraw within `architecture.svg`; shapes also in `examples/output/demo_trace.json` |
| Figure 3 | QELM hidden layer / quantum-inspired initialisation | Redraw within `architecture.svg`; formula in docs/method.md |
| Figure 4 | GWO convergence over iterations (fitness vs iteration) | Regenerate on the fixture: `docs/figures/gwo_convergence_demo.svg` from `gwo_trace.json` (N 6, T 4 demo budget, not the paper's N 20, T 50) |
| Figure 5 | Training curves (loss/accuracy per epoch), plotted for the test split | Regenerate with the inner validation split instead: `docs/figures/learning_curves_demo.svg` from `learning_curves.json` (audit item 8) |
| Figure 6 | Second training-curve figure (binary or second metric), also per-epoch test | Not redrawn separately; the binary run's encoder log is written to `examples/output/train_binary/encoder_log.json` |
| Figure 7 | Confusion matrix and/or ROC curves for the 15-class task | Regenerate on the fixture: `docs/figures/confusion_matrix_demo.svg`, `docs/figures/roc_per_class_demo.svg` |
| Table 1 | Class counts of the processed Edge-IIoTset (15 classes, 1,909,671 rows) | Class ORDER fixed in `configs/classes.yaml`; counts transcribed in `evidence/paper_reported/qelm.json` and used only as proportions by `tools/make_fixture.py` |
| Table 2 | Preprocessing steps | Implemented in `preprocess/pipeline.py`; listed in docs/data.md |
| Table 3 | GWO search space (7 dimensions and bounds) | Implemented in `configs/gwo_paper.yaml` and `gwo.py`; a test checks the seven dimensions and bounds |
| Table 4 | GWO / training hyperparameters (N, T_max, optimizer, loss, batch, epochs) | Stored in `configs/paper_*.yaml`, `configs/gwo_paper.yaml` |
| Table 5 | Encoder layer table with the learned feature-index embedding | Implemented in `encoder.py`; shapes in MODEL_CARD.md and `demo_trace.json` |
| Table 6 | Binary results (accuracy, precision, recall, F1, FPR, AUC) | Transcribe: `evidence/paper_reported/qelm.json` -> `binary_table6`. Demo binary metrics regenerated on the fixture in `metrics.json` -> `binary` |
| Table 7 | 15-class results (accuracy, macro P/R/F1, FPR, training time) | Transcribe -> `multiclass_table7`. Demo multiclass metrics regenerated -> `metrics.json` -> `multiclass` |
| Table 8 | Per-class results incl. MITM F1 vs best baseline (DenseNet) | Transcribe -> `mitm_table8`. Per-class F1 regenerated on the fixture (`per_class_f1`), where the MITM-like class has only a handful of test rows |
| Table 9 | Baseline comparison (CNN-1D, LSTM, GRU, BiLSTM, DenseNet, ...) | Not implemented (`proposed`); this repository's baselines are logistic regression, random forest, hist gradient boosting |
| Table 10 | Statistical tests (Friedman / Holm across methods and seeds) | Regenerate on the fixture: `metrics.json` -> `friedman_holm_macro_f1` over 4 methods x 5 seeds; the article's own chi-square values are transcribed with the inconsistency note |
| Table 11 | Ablations (without attention, QELM, quantum init, GWO, focal loss; encoder + MLP) | Transcribe -> `ablations_table11`. Not regenerated; the config options (`tokenisation`, `global_angle_mode`, `weight_scale`, `search.enabled`, `class_weighting`) make single ablations runnable but no ablation table is produced by the demo |
| Table 12 | Computational cost (0.21 M params, 2.83 M ops, 0.31 ms GPU, 4.12 ms CPU) | Transcribe -> `cost_table12`. Parameter counts of THIS implementation are written to `demo_trace.json`; no latency is measured or claimed |

## Verdict on fidelity

The pipeline order, loss, optimizer, search space, split protocol and closed-form solve
follow the paper. The tokenisation (item 1), the meaning of the global angle and the weight
scaling (item 5), the fitting scope inside inner folds (item 7) and the FPR definition
(item 10) are assumptions that could each move real-data numbers. No paper number is
reproduced or contradicted by this repository.

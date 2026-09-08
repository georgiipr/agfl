# Pipeline audit and comparability

This is a source audit, not an empirical accuracy study. No new training,
evaluation, or tests were executed locally. Numerical and real-data acceptance
checks are supplied for the intended experiment environment. No accuracy gain,
behavioral parity result, or statistical significance is claimed.

## Original execution path

`main.py` seeded Python/NumPy/Torch once, enabled cuDNN benchmarking, then
selected `keys_handling/eeg_key.py` or `ecg_key.py`. EEG selected one of three
hard-coded factories and ran nine separate subjects for 250 epochs each. ECG
ignored the requested model name and always created a Conformer. Dataset paths
were assembled from the current working directory. Separate trainers duplicated
optimizer/loss/scheduler/metric logic. Results were printed and plotted; complete
run configurations, checkpoints, split manifests, and per-seed machine-readable
records were not stored. The original README's numbers remain in Git history at
the commit recorded in `../tests/references/ORIGIN.md`.

## Findings and dispositions

| Original behavior | Why it matters | Current disposition | Historical comparability |
|---|---|---|---|
| Both trainers selected the best checkpoint using validation accuracy and then reported metrics on that same partition. | Those metrics estimate selected validation performance, not independent test performance. This can inflate reports, rather than explain poor accuracy. | A persisted third partition is evaluated once after restoring the best validation checkpoint. | New test scores cannot be compared directly with old validation scores. |
| Weighted focal loss computed `p_t = exp(-weighted_cross_entropy)`. | Class weights altered the focal probability itself, giving class-dependent focusing beyond the intended weight multiplier. | Compute unweighted target CE/probability, then multiply focal loss by training-derived class weights. | Loss gradients change; this is an explicit implementation correction. |
| Class weights were inferred by iterating the shuffled training loader; missing classes could produce division by zero or the wrong weight-vector length. | Consumes loader RNG and can generate invalid loss weights. | Count labels directly from stored train indices with the declared class count; reject missing training classes. | Same inverse-frequency formula where all classes are present. |
| Unseeded worker randomness and cuDNN benchmarking were enabled. | Single-run metrics cannot separate initialization or sampling variation from mechanism effects. | Seed Python, NumPy, CPU/CUDA Torch, worker Python/NumPy, loader generators, and split generators. Disable benchmarking and TF32; request deterministic operations. | New seeds are reproducible on a recorded compatible stack; cross-platform bit equality is not promised. |
| Best score started at zero, so a run with zero validation accuracy could restore the initialization. | No trained checkpoint would be selected. | The first finite validation score always establishes a best checkpoint; ties retain the first epoch. | Changes this failure case only. |
| Warmup was fixed at ten epochs and cosine period was `epochs-10`. | Short validation runs could produce zero/negative periods. | Bound warmup to at most `epochs-1`; record the LR actually used each epoch. | Original 250/50-epoch schedules retain the warmup/cosine shape; short-run behavior is corrected. |
| Epoch losses averaged batch means, giving small final batches excessive influence. | Distorts reported losses and loss-based checkpoint selection. | Sum per-example mean losses weighted by batch size; divide by sample count. | Affects reported loss and loss-selected checkpoints. |
| F1/AUC assumed every held-out class was present. | Group splits can lack a class; a crash or silently changed class set is misleading. | Macro F1 uses every declared class with zero division set to zero; undefined AUC is JSON null with its reason and class counts. | Explicit class semantics; no fabricated AUC or dropped class. |
| Active models imported deprecated AGFL while the top-level file looked current. | Editing the top-level file did not change experiments. | One documented current AGFL implementation; the active original is retained for parity checks and the unused dense original remains in Git history. | Default active polynomial mathematics preserved. |
| AGFL had all-zero signed coefficients and threshold Top-k ties. | Initial graph/filter gradients vanish, and ties can exceed k; these are relevant hypotheses for ablation. | Preserve both defaults. Expose initialization, exact-k, normalization, and formulation alternatives by name. | No silent change of mathematical assumptions. |
| EEG models collapsed physical channels before attention. | Existing adjacency plots represented temporal/latent tokens, not electrode connectivity. | Each model's EEG variant keeps electrode tokens through its attention; ECG variants use model-specific temporal tokens. | A new surrounding architecture, not a parity claim for old EEG models. |
| EEGNet inferred classifier size with BatchNorm/dropout active on a dummy sample. | Construction changed running statistics and consumed RNG. | Current backbones know their token/feature sizes and need no dummy forward; the archived original remains available for parity. | New architecture initialization differs; archived semantics are preserved. |
| EEGEncoder's TCN recomputed earlier blocks on residual branches and mishandled nondefault feature dimensions. | Repeated dropout/BatchNorm and invalid projections make its nondefault architecture questionable. | EEGEncoder now evaluates each causal residual block once and projects differing feature dimensions explicitly. Its variants retain convolution, TCN and attention fusion. | This is a disclosed correction and adaptation; full-original-backbone parity is not claimed. |
| DSTS guessed axes by comparing dimensions and passed layer index zero into every AGFL block. | Axis interpretation and the advertised sparsity schedule could be wrong. | DSTS variants require exact `[B,C,T]` shape and pass actual block indices. Original independent sources remain only in tests/references. | Axis and schedule corrections change original behavior and need separate experimental validation. |
| ECG no-attention fell through to standard attention; Conformer did not implement a missing mixer. | The requested option did not match the network that ran. | No-attention is removed from current model selection, presets, and statistical baselines. | Old no-attention labels are not reusable as current experiment results. |
| Original dependency pins paired Torch 2.2.2 with NumPy 2.x and newer Torch APIs. | Environment reproducibility was inconsistent before considering model quality. | Canonical dependencies now require Torch >=2.5; the historical lock remains in Git history. Resolve a new target-specific lock on the experiment machine. | Environment changes must be recorded; no local installation or validation is part of acceptance. |

Dataset-specific preprocessing, label, subject, and lead findings are in
[data_audit.md](data_audit.md). Exact graph formulas and manuscript differences
are in [mathematics.md](mathematics.md).

## Current training policy

Domain budgets remain EEG 250 epochs / LR 0.005 / focal gamma 3 and ECG 50 epochs
/ LR 0.0003 / focal gamma 1. AdamW, weight decay 0.001, batch size 64, ten-epoch
warmup, and validation accuracy selection remain starting defaults. Every value
is configurable. The ECG preset retains shift/scale/noise augmentation on the
training partition only. Each selected backbone has declared architecture defaults;
these are not a tuned reproduction of original full-training results.

The loss is the per-example mean of `class_weight[y] * focal_factor * CE`.
For cross-entropy mode the focal factor is one. Weights are inverse training
frequency with expected training mean one. This fixed per-example reduction is
explicit; it does not divide each mini-batch by its random weight sum.

Models emit logits. Probability softmax is applied only for metric computation;
ROC-AUC uses class probabilities, not argmax labels. Training uses train mode;
validation/test use eval mode and no gradients. The optimizer zeroes gradients
before each batch. Optional gradient clipping occurs after AMP unscaling. With
AMP enabled, GradScaler handles overflow by skipping the optimizer step and the
history records the skip count. AMP is opt-in; validation/test use float32.
Checkpoints store CPU copies of all model parameters and buffers, including
Performer features, as well as normalization statistics and resolved settings.
They are best-model inference checkpoints, not mid-epoch optimizer-resume files.

A saved run records source hash, source revision, package versions, environment,
hardware descriptors, data fingerprint, split identity, and configuration.
Replaying its configuration requires the same source and relevant package
versions; a different implementation should start a new experiment instead.
Default reruns replace the selected seed's generated artifacts and restart from
epoch 1. `--skip-completed` retains matching completed seeds and restarts failed,
interrupted or incomplete ones. Completion markers are removed before replacement
so a second interruption cannot expose mixed old/new outputs as a completed run.
An OS lock prevents simultaneous writers and releases on process exit. Unrelated
files and other seed directories are preserved.

## Acceptance still required on the experiment machine

### Target-machine EEG pooling failure

The first reported CUDA EEG quick run failed on its initial backward pass:
`adaptive_avg_pool2d_backward_cuda` rejected strict deterministic mode. The
earlier generic EEG tokenizer used `nn.AdaptiveAvgPool1d`, which PyTorch implements
through adaptive 2D pooling internally. Deterministic slice pooling is now shared
by the Conformer EEG stem and the Signal Transformer EEG tokenizer.
The upstream restriction is documented in
[PyTorch's determinism reference](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html),
and the 1D-to-2D delegation is visible in
[the PyTorch source](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/Pooling.cpp).

The tokenizer now takes slice means using the same adaptive boundaries:
`start = floor(i*T/bins)`, `stop = ceil((i+1)*T/bins)`. This preserves overlapping
windows for nondivisible lengths and keeps all model parameter names and shapes.
It changes the reduction implementation, so floating-point roundoff may differ;
it does not change the AGFL equations, window definitions, or determinism policy.
The GradScaler call also uses the supported `torch.amp.GradScaler('cuda', ...)`
API to remove the independent deprecation warning.

`tests/test_pooling.py` supplies CPU forward/gradient comparisons against native
pooling, repeated strict CUDA backward checks, and two-epoch CUDA EEG engine
checks with AMP both disabled and enabled. These checks have not been run locally.
After transferring the changed source to the experiment machine, run them there
and repeat the EEG quick run. The same configuration/output directory can be
used: its failed run artifacts are automatically replaced.

### Remaining acceptance checks

1. Run the source-archive check and numerical model/loss/split/analysis tests.
2. Enable real-data frozen-weight checks for EEGNet temporal and Conformer ECG.
   Check AGFL's original/default forward and gradients independently. Other
   adapted backbones do not claim numerical identity with original sources.
3. Run each actual backbone in both modalities using `--model` and choose its
   attention independently using `--attention`, with a declared three-way split.
4. Run the five-seed baseline matrices and declared ablations. Examine saved
   failure records, class counts, selected epochs, and paired comparisons.
5. Only then assess accuracy and its causes. Numerical parity tests cannot prove
   historical full-training accuracy, and a new test protocol cannot reproduce
   old validation-based headline numbers by definition.

# EEGNet/A03 token-routing results and diagnostic failure

Source: `/Users/egor/Downloads/eegnet-A03-token-agfl-v1`. Reviewed 24 September 2026.

**AGFL accuracy increased from 84.44% to 85.19%, a gain of 0.74 percentage points. It remains below MHA (85.56%) and Linformer rank 4 (85.93%), and below the 90% target.**

## Completed training and verified metrics

There are 20 distinct completed fits, each with 250 recorded epochs, a checkpoint and predictions. The other 20 result files are identical report mirrors, not additional experiments. All methods share per-seed split manifests, data fingerprint, comparison ID, numerical package versions, backbone and training settings. Accuracy, macro F1, confusion matrices and one-vs-rest macro ROC-AUC were independently recalculated from saved probabilities. No project code or checkpoint inference was run locally.

| Method | Test accuracy mean ± seed SD | Validation accuracy mean | Macro F1 | Macro ROC-AUC | Parameters |
|---|---:|---:|---:|---:|---:|
| Original AGFL | 84.44% ± 5.94 pp | 92.22% | 0.8395 | 0.9721 | 8,052 |
| AGFL with token routing | 85.19% ± 5.71 pp | 91.48% | 0.8496 | 0.9762 | 8,244 |
| MHA | 85.56% ± 6.60 pp | 92.96% | 0.8525 | 0.9625 | 8,036 |
| Linformer rank 4 | 85.93% ± 9.13 pp | 90.74% | 0.8535 | 0.9652 | 8,260 |

The original AGFL control reproduces its previous seed accuracies and selected epochs exactly. The restored backbone therefore recovered the 84.44% reference result. Revised AGFL validation accuracy decreased from 92.22% to 91.48%, so the small test gain is not a consistent improvement across both partitions.

## All five seeds

| Seed | Original AGFL | Revised AGFL | MHA | Linformer rank 4 | Revised − original |
|---|---:|---:|---:|---:|---:|
| 0 | 88.89% | 90.74% | 90.74% | 94.44% | +1.85 pp |
| 1 | 90.74% | 85.19% | 83.33% | 92.59% | -5.56 pp |
| 2 | 75.93% | 75.93% | 75.93% | 88.89% | +0.00 pp |
| 3 | 85.19% | 85.19% | 92.59% | 72.22% | +0.00 pp |
| 4 | 81.48% | 88.89% | 85.19% | 81.48% | +7.41 pp |

Revised AGFL has 230 correct predictions out of 270 repeated test occurrences, versus 228 for original AGFL, 231 for MHA and 232 for Linformer. Against original AGFL, it wins two seeds, ties two and loses one. At the trial level, 14 errors were corrected while 12 correct predictions became errors; four other changed predictions remained wrong.

One revised seed reaches 90.74%, but its five-seed mean is 85.19%. These overlapping splits contain only 177 unique test trials, not 270 independent observations. A03 has repeatedly informed development. These results do not establish an AGFL advantage over other attentions or a confirmed article-level generalization result.

## Which classes changed

| Class | Original AGFL test recall | Revised AGFL test recall | Change | Net correct occurrences |
|---|---:|---:|---:|---:|
| Left hand | 87.14% | 82.86% | -4.29 pp | -3 |
| Right hand | 92.86% | 94.29% | +1.43 pp | +1 |
| Feet | 66.15% | 83.08% | +16.92 pp | +11 |
| Tongue | 90.77% | 80.00% | -10.77 pp | -7 |

Feet recognition improved substantially, including on validation (83.08% to 90.77% recall). The test gain was offset mainly by tongue and left-hand errors. Feet→tongue mistakes fell from 15 to 6, while tongue→feet mistakes rose from 5 to 10. This is an incomplete class tradeoff, not a solved four-class task.

## Why plotting failed, and the fix

The crash occurred while writing `coefficient_conditioning.json`, after training had completed. `zero_sum_check_tolerance` was computed using a NumPy float32 epsilon; mixed scalar arithmetic on the cluster NumPy 2.4.2 kept a NumPy float32 result. The standard JSON encoder rejects that type. The convolution padding warning and MNE covariance messages were not the exception.

`agfl/visualization/diagnostics.py` now converts epsilon to a native Python float before constructing the tolerance. Strict finite JSON output is retained. A regression check now covers float32 checkpoint diagnostics as well as float64. The tests are prepared but were not run locally. The local NumPy version is 1.26.4; its older promotion behavior did not reproduce the cluster float32 serialization error during standalone saved-array inspection.

`agfl/cli.py` now attempts result analysis and plots from saved predictions even if checkpoint diagnostics raise an ordinary exception; it still reports and re-raises the original diagnostic failure. KeyboardInterrupt/SystemExit are not swallowed. No model, training or prediction changes are part of this fix.

Only 14 PNG/PDF figure pairs from revised-AGFL seed 0 are downloaded. Its numeric token-coefficient file was saved before the JSON failure, but there is no completed diagnostic manifest/index, and automatic aggregate analysis never ran in this download. That partial file shows finite coefficients, a nonzero router weight norm (1.8611), and token variation; only one checkpoint has these diagnostics, so it does not establish behavior for all seeds.

## Recover the report without training again

Upload these two ordinary source files from the Mac:

```bash
scp /Users/egor/Downloads/AGFL/agfl/visualization/diagnostics.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
scp /Users/egor/Downloads/AGFL/agfl/cli.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
```

In the allocated cluster shell, using the existing environment and original `../ml` dataset:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg
```

Checkpoint diagnostics:

```bash
python main.py diagnose-session results/eegnet-A03-token-agfl-v1 \
  --device cuda --partition validation --embedding tsne
```

Result plots and tables (this command can also run independently of checkpoint diagnostics):

```bash
python main.py analyze results/eegnet-A03-token-agfl-v1 --plots
```

Download `results/eegnet-A03-token-agfl-v1/report/`. Do not relaunch the training sweep to repair plotting; a source update can deliberately invalidate training reuse checks.

## Provenance

- Training source SHA256: `40d9b4b88270d4875fb4857d431b5a32ce7a2ba2337f85728bd516380ef88a66`.
- Dataset fingerprint: `ce4f9758ab17f5ea5cdb8bd730d4daec20e4870cfda03ef7b110bf4df20f5f66`.
- Comparison ID: `2309dd92e91f553f3bc3`.
- Model: original EEGNet, 32-sample kernel, seven time tokens, no power branch.
- Training: AdamW 0.005, balanced focal loss gamma 3, no augmentation, no EMA, original first-best checkpoint tie rule.
- Subject/session: A03T, four classes; each seed has 162 training, 54 validation and 54 test trials.
- Model inference and corrected diagnostics still need execution on the cluster; this review uses saved artifacts only.

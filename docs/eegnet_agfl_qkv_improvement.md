> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# Improve AGFL on A03 before expanding the study

The Q/K/V and three-neighbor studies below are complete. The current launch
is the [EEGNet power/EMA comparison](eegnet_power_ema.md), following the
completed attention comparison, graph/filter refinement and capacity study. Historical commands below
are retained for reproduction with the corresponding saved provenance.

The objective is to make EEGNet with AGFL outperform standard attention,
starting with MHA on A03. The completed three-candidate Q/K/V search reached
**85.56% mean test accuracy (SD 2.75 percentage points)** over seeds 0–4,
equal to the earlier MHA mean. A03 already exceeds 75%; this result does not
establish an AGFL advantage or the nine-subject target. The nine-subject
expansion remains deferred.

## Completed experiment: three neighbors against the existing sparse control

Compare `spatial_qkv_sparse` with the opt-in `spatial_qkv_top3` candidate.
The only configuration change is `attention_options.top_k=3`: the control
uses the existing scheduled Top-k, normally five of the seven temporal
tokens. Both retain the threshold tie policy, so exact score ties can retain
more than the requested number of neighbors. This is a request for three
neighbors, not a guarantee of exactly three nonzero entries in every row.

Keep learned Q/K/V, one-hop initialization, polynomial order `K=2`, score
scaling, EEGNet, preprocessing, loss, learning rate, augmentation and the
250-epoch budget unchanged. The original candidates and default search
remain unchanged. This experiment tests one graph-construction choice; it
does not assume that a sparser graph improves accuracy.

### Historical cluster commands

Update the cluster checkout first. Request one GPU, four CPUs and 10 GB of
memory for up to four hours, then wait for the interactive shell to open:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

Use the existing Python environment and run from the cluster `AGFL` directory.
The dataset is `../ml/A03T.gdf`. No installation or development test run is
part of this launch:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg
```

Run **10 validation-only fits**: two candidates, A03 only, seeds 0–4.
Validation chooses a candidate independently within each seed; only the five
selected checkpoints are evaluated on test. Use the new folder to preserve
the completed `eegnet-A03-agfl-qkv-v3` experiment:

```bash
python main.py tune-eegnet \
  --data-dir ../ml \
  --subjects 3 \
  --seeds 0 1 2 3 4 \
  --epochs 250 \
  --candidate spatial_qkv_sparse \
  --candidate spatial_qkv_top3 \
  --output-dir results/eegnet-A03-agfl-top3-v1
```

Generate signal, embedding, coefficient and graph plots for the selected
checkpoints:

```bash
python main.py diagnose-session results/eegnet-A03-agfl-top3-v1 \
  --device cuda --partition validation --embedding tsne
```

Generate result plots and tables:

```bash
python main.py analyze results/eegnet-A03-agfl-top3-v1 --plots
```

Download **`results/eegnet-A03-agfl-top3-v1/report/`**. Candidate validation
scores and choices are in `selection_report.json`; `search_report.md` and
`search_result.json` describe the validation-selected procedure. Its test
mean is **not** the fixed top-three recipe's test mean, because different
seeds can select different candidates. Result figures are indexed in
`analysis/figures/index.md`, with checkpoint figures below `diagnostics/`.
Keep `artifacts/` on the cluster for checkpoint diagnostics. An identical
relaunch reuses matching completed candidates and restarts incomplete ones;
keep the checkout stable throughout the search.

## Completed Q/K/V study: rationale and historical commands

The following sections document the completed `eegnet-A03-agfl-qkv-v3`
study. They are retained for interpretation and reproduction, not as the
next launch. It contained 15 completed candidate fits and five selected test
evaluations. Sparse won two seeds, renormalized won two and dense won one.
Mean candidate validation accuracy was 92.22%, 91.85% and 91.11%, respectively.
No single fixed candidate was tested on all five seeds by this search.

### Limitation in the earlier spatial-control recipe

The earlier `spatial_control` uses `projection=split_input`: each head's
unprojected features serve as Q, K and V. The MHA baseline learns distinct
Q/K/V projections. The submission PDF also describes learned Q/K/V before
graph construction (page 3). The tested AGFL therefore lacks that explicit
learned graph projection, even though its EEGNet feature extractor is learned.
It instead learns separate feature matrices for each polynomial tap.

The original recipe also starts all signed filter coefficients at zero and
does not normalize propagated feature norms. With zero coefficients, the
graph/filter matrices receive no gradient through the weighted tap sum on
the first backward pass. The residual still trains EEGNet and coefficients
can learn; this is not proof of a sustained optimization failure or the
cause of the measured score difference.
The saved A03 coefficient plots contain nonzero learned taps, so the
previous AGFL branch was not permanently inactive.

### Q/K/V initialization and the three completed candidates

`coefficient_init=one_hop` initializes signed, learnable coefficients to
`[0, 1, 0]` for maximum hop order `K=2`. It requires `K >= 1` and identity
coefficient activation. Zero- and two-hop taps start at zero but receive
gradients and can contribute as training proceeds.

All three candidates use learned Q/K/V with bias, value-only graph-filter
taps, square-root head-dimension score scaling and the new initialization.
They keep the existing spatial-control EEGNet, data, loss, learning rate,
batch size, epoch budget, residual path and four attention heads unchanged.

| Candidate | Graph | Propagation |
|---|---|---|
| `spatial_qkv_dense` | All seven tokens | Polynomial: powers of A applied to V |
| `spatial_qkv_sparse` | Existing scheduled Top-k, normally five of seven tokens | Same polynomial |
| `spatial_qkv_renorm` | Same scheduled Top-k | Preserve each token's feature norm after each hop |

The dense candidate starts with `softmax(QKᵀ / sqrt(d)) V W_out`, the MHA
computation. Q/K/V and output projections consume the same initialization
RNG draws as the existing MHA module. Its extra graph-filter coefficients
then give it learnable zero- and two-hop contributions. The sparse and
renormalized candidates are not initially equivalent to MHA.

This restores the learned projections and tests the paper's polynomial and
norm-preserving filtering forms. The one-hop initialization is an explicit
new choice, not a claim to reproduce the paper's lower-order initialization
verbatim. Adding representational flexibility does not guarantee improved
held-out accuracy. The MHA baseline stays fixed; its settings are not weakened.

The original AGFL defaults and original five-candidate `tune-eegnet` search
remain unchanged. These follow-ups are opt-in.

### Historical Q/K/V commands (completed; not the next experiment)

The commands below record the completed launch from `AGFL`, with the existing
environment active and `../ml/A03T.gdf` available. They are not required for
the top-three follow-up. Any reproduction should use the saved source and
environment provenance and a separate output folder.

Use the existing environment inside an interactive `salloc` allocation, as
described in the README. Training and plotting do not require `pytest` or a
development test run. No additional installation is part of this launch.

Run **15 validation-only candidate fits**: three AGFL
recipes, A03 only, seeds 0–4. Validation chooses one candidate within each
seed; only the five selected checkpoints are evaluated on test.

```bash
python main.py tune-eegnet \
  --data-dir ../ml \
  --subjects 3 \
  --seeds 0 1 2 3 4 \
  --epochs 250 \
  --candidate spatial_qkv_dense \
  --candidate spatial_qkv_sparse \
  --candidate spatial_qkv_renorm \
  --output-dir results/eegnet-A03-agfl-qkv-v3
```

The `v3` and `top3-v1` directories contain completed studies. Preserve them;
use the new power/EMA session for the current follow-up. An identical relaunch with
the original training checkout reuses matching completed candidates and
restarts incomplete candidates.
If the previously launched `results/eegnet-A03-agfl-qkv-v2` search is already
running or finished, do not retrain it merely to reorganize the files. Finish
training and checkpoint plots with its training checkout, then follow the
[existing-session organization commands](result_sessions.md#existing-completed-a03-search).
Keep source stable throughout each search; changing it affects provenance and
completion matching.

Generate selected-checkpoint signal, embedding, coefficient and graph plots:

```bash
python main.py diagnose-session results/eegnet-A03-agfl-qkv-v3 \
  --device cuda --partition validation --embedding tsne
```

Then generate result plots and analysis tables:

```bash
python main.py analyze results/eegnet-A03-agfl-qkv-v3 --plots
```

Download only **`results/eegnet-A03-agfl-qkv-v3/report/`**. Keep `artifacts/`
on the cluster: it contains the candidate and selected checkpoints needed to
resume or diagnose runs. The report contains candidate histories, configuration
and split records, validation selection evidence, selected predictions, metrics,
and generated plots. The mean for the validation-selected procedure is in
`report/search_report.md` and `report/search_result.json`; generic analysis
separates distinct selected configurations. Figure indexes are in
`report/analysis/figures/index.md` and below `report/diagnostics`.

The frozen A03 MHA result (85.56% test mean, 92.96% selected validation mean)
is a development reference. Do not choose the AGFL candidate using test
scores or compare only favorable seeds. Previously inspected, overlapping
seed splits are not fresh confirmation data. After improvement is demonstrated,
fix the chosen method or selection procedure before the broader matched
comparison, including other attention baselines.

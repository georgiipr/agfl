# Improve AGFL on A03 before expanding the study

The objective is to make EEGNet with AGFL outperform standard attention,
starting with MHA on A03. The completed fixed comparison measured 84.81%
AGFL versus 85.56% MHA mean test accuracy over seeds 0–4. That result does
not meet the objective. The next work changes AGFL's graph/filter
parameterization; the nine-subject expansion is deferred.

## Concrete limitation in the tested recipe

The current `spatial_control` uses `projection=split_input`: each head's
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

## New initialization and three bounded candidates

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

## Cluster commands

Update the cluster checkout first. Run from `AGFL` with the environment active
and `../ml/A03T.gdf` available. No project code or training has been run locally.

First run the focused checks on the cluster. They verify initial dense
MHA equivalence, shared gradients and RNG state, learnability of extra taps,
sparse/renormalized gradients, invalid settings, and existing reference behavior:

```bash
python -m pytest tests/test_agfl_initialization.py tests/test_agfl_backbone_initialization.py \
  tests/test_agfl_safeguards.py tests/test_gradient_checks.py tests/test_models.py \
  tests/test_reference.py -m 'not real_data' -q
```

After the checks pass, run **15 validation-only candidate fits**: three AGFL
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

The `v3` directory is for a **fresh experiment with this checkout**. An identical
relaunch reuses matching completed candidates and restarts incomplete candidates.
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

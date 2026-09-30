> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# Restore the retained token-routing AGFL

25 September 2026. Restored target: **original seven-token EEGNet +
`coefficient_conditioning=token_contrast` + ordinary AdamW**, the fixed
configuration with 85.19% observed mean test accuracy on A03, seeds 0–4.
This is the version before energy routing and SAM, not the much earlier
split-input or static-coefficient AGFL.

The energy descriptor, extra gate, diagnostic extensions, analysis label and
pairing, preset and three energy-specific development-test files are removed.
The existing token router, older optional ablations, diagnostic JSON fix and
report recovery remain. Saved results, splits, datasets and checkpoints are
untouched. The retired energy source is preserved as an ordinary text
[reproduction patch](archive/eegnet_energy_reproduction.patch) outside active
package/preset/test discovery. Apply it only in a separate rollback checkout
when deliberately reproducing or diagnosing that historical experiment.

The retained preset is `eegnet-a03-token-agfl`; its four arms are static AGFL,
token AGFL, MHA and Linformer rank 4. Its data, backbone and training settings
are unchanged. The latest completed control already reproduces the retained
per-seed scores and epochs; no rerun is required just to establish the rollback.
The entire checkout is not asserted to be byte-identical to a historical
cluster checkout: earlier feature-routing/report fixes are preserved, and new
launches will record their current source provenance.

## Upload the four changed runtime files

On the Mac:

```bash
cd /Users/egor/Downloads/AGFL
scp agfl/attention/agfl/layer.py agfl/attention/agfl/config.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/attention/agfl/
scp agfl/analysis.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
scp agfl/visualization/diagnostics.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
```

On the cluster, remove only the retired preset/test files:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
rm -f agfl/presets/eegnet-a03-token-energy.json \
  tests/test_agfl_token_energy.py tests/test_agfl_token_energy_diagnostics.py \
  tests/test_eegnet_energy_comparison.py
```

The source update assumes the completed energy study was the last cluster
version, so its ordinary-AdamW engine and previous SAM removal are already
present. No new library, archive extraction or test-suite launch is required.

## Optional reproduction, not a required new study

If an allocation is needed, run on the login node and wait for it:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=4:00:00
```

After allocation:

```bash
srun --pty bash -l
```

Inside the GPU shell, use the existing environment. Dataset:
`../ml/A03T.gdf`. Output: `results/eegnet-A03-token-restored-v2`.

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source /beegfs/home/georgii.promyslov/.venv/bin/activate
export MPLBACKEND=Agg
```

Training plus automatic reporting (20 fits, A03 only, five seeds):

```bash
python main.py sweep --preset eegnet-a03-token-agfl \
  --output-dir results/eegnet-A03-token-restored-v2 --report
```

Separate checkpoint diagnostics:

```bash
python main.py diagnose-session results/eegnet-A03-token-restored-v2 \
  --device cuda --partition validation --embedding tsne
```

Separate result plots:

```bash
python main.py analyze results/eegnet-A03-token-restored-v2 --plots
```

Download only that session's `report/`; retain `artifacts/` on the cluster.
Use `--skip-completed` only to resume matching runs with unchanged source,
configuration and environment. Never point a new iteration at the old energy
session to substitute results. The existing downloaded energy report needs
no regeneration and remains an unsuccessful-ablation record.

## Verification

Python AST and JSON parsing, source comparisons, `git diff --check`, shell
syntax and reproduction-patch applicability were checked. No project code,
training, inference or development tests ran locally. Cluster files are not
changed until the upload/removal commands above are executed by the user.

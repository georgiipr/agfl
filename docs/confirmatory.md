# Confirmatory experiment: are our EEGNet and attentions better than the dataset authors'?

Plan `si-hom-confirmatory-v1`, written on 2026-10-02 **before any confirmatory
run existed**. Everything below — arms, seeds, hypotheses, tests and the wording
of the possible conclusions — is fixed. A change after results have been seen is
a deviation and must be recorded in the last section.

## The question

On the SI_Hom imagined-speech dataset:

1. Is our EEGNet better than the EEGNet settings in the authors' code?
2. Are our attentions (AGFL, MHA) better than the authors' attention (HCANN)?

## What was known when this was written

One exploratory run (preset `si-hom-attentions`, seeds 0–4, 200 epochs) and
analyses chosen after looking at it. Test accuracy, 8 classes, chance 12.5%:

| | AGFL | MHA | HCANN |
|---|---|---|---|
| `eegnet` | 17.0 | 15.7 | 15.3 |
| `eegnet_original` | 15.4 | 15.5 | 14.3 |

The same contrasts as below gave, on those five seeds: H1 +0.9 points, H2 +1.4,
H3 +0.8, S1 +2.7. Only H2 and S1 were consistent across seeds, and the
validation partition did not show the attention differences. A linear
classifier reaches about 17% on this data, so every model is near the ceiling.
These numbers motivated the hypotheses; they are not evidence for them, and
seeds 0–4 are not used again.

## Design

Preset `si-hom-confirmatory`: 8 arms × 20 new seeds (5–24) = 160 fits.

| Arm | Backbone | Attention | Training recipe |
|---|---|---|---|
| `eegnet+agfl` | `eegnet`, `pre_spatial`, kernel 250 | AGFL | project |
| `eegnet+mha` | same | MHA | project |
| `eegnet+hcann` | same | HCANN | project |
| `original+agfl` | `eegnet_original`, kernel 32, max-norm once | AGFL | project |
| `original+mha` | same | MHA | project |
| `original+hcann` | same | HCANN | project |
| `original+mha@authors` | same | MHA | authors |
| `original+hcann@authors` | same | HCANN | authors |

- **Data and split:** pooled cohort, all 7 subjects, 8 classes, 500 samples,
  `train_channel` normalization; `stratified` 60/20/20 by subject and class. A
  seed fixes the split and the initialization; all arms share both.
- **Project recipe:** exactly the first run's (AdamW, learning rate 1e-3, weight
  decay 1e-3, batch 64, class-balanced cross-entropy, warm-up cosine), with
  **60 epochs instead of 200**. In the first run every checkpoint was selected
  before epoch 60. The cosine schedule is therefore compressed; this is a
  declared difference from the first run.
- **Authors' recipe:** Adam, learning rate 1e-3 constant, batch 16, 50 epochs,
  no weight decay, unweighted cross-entropy (their `train.py`).
- **Checkpoint:** lowest validation loss, for every arm. The authors' script
  kept the epoch with the best test accuracy; that is not reproduced.
- **All backbone, attention and model options** are those of the first run.

## Hypotheses

The outcome is **test accuracy**. Each contrast is computed per seed, so the
unit of analysis is the seed (one split and one initialization), n = 20.

Primary (Holm correction over these three):

| | Contrast (per seed) |
|---|---|
| **H1** | mean of the three `eegnet` arms − mean of the three `original` arms (project recipe) |
| **H2** | AGFL − HCANN, averaged over the two EEGNets (project recipe) |
| **H3** | MHA − HCANN, averaged over the two EEGNets (project recipe) |

Secondary (Holm correction within this family; not decisive on their own):

| | Contrast |
|---|---|
| S1 | `eegnet+agfl` − `original+hcann` |
| S2 | `eegnet+agfl` − `original+mha@authors` (closest stand-in for their model as they trained it) |
| S3 | `eegnet+agfl` − `original+hcann@authors` |
| S4 | AGFL − MHA, averaged over the two EEGNets |
| S5 | project recipe − authors' recipe on `eegnet_original` (MHA and HCANN) |

## Tests and conclusions

For each contrast, two-sided, α = 0.05:

- **Decision test:** the Nadeau–Bengio corrected resampled t-test. Random splits
  of one finite dataset overlap, so the variance of the mean difference is
  multiplied by `1/n + n_test/n_train` (here 1/20 + 0.334) instead of `1/n`.
- **Dataset-only test:** the plain paired t-test over seeds. It asks whether the
  difference is consistent over splits and initializations of these 2640 trials.
- Also reported: Wilcoxon signed-rank test, number of positive and negative
  seeds, the same contrast on the validation partition, and the difference with
  each unique test trial as the unit.

The conclusion for each hypothesis is one of these, decided by the script:

| Label | Condition |
|---|---|
| **CONFIRMED** | Holm-adjusted corrected p < 0.05 and the difference favours ours |
| **DIFFERENCE ON THIS DATASET ONLY** | corrected test not significant, Holm-adjusted plain paired p < 0.05 |
| **NOT ESTABLISHED** | neither |
| **CONTRADICTED** | Holm-adjusted corrected p < 0.05 and the difference is against ours |

"NOT ROBUST" is appended when the validation partition shows the opposite sign.
A result is reported whichever label it gets.

Rules:

- All 160 fits must be present. With fewer, the script reports `INCOMPLETE` and
  no conclusion is drawn; resubmitting the launcher completes missing fits.
- A failed fit is rerun with the same settings, never dropped or replaced by
  another seed.
- No seeds are added and no arms removed after seeing results.

## How large a difference this can detect

From the spread of the per-seed contrasts in the first run, the corrected test
with 20 seeds would need an observed difference of roughly **2.0 points for H1,
1.1 for H2 and 1.0 for H3** to pass after Holm correction. The first run showed
0.9, 1.4 and 0.8. So H2 can be confirmed if its effect holds; H1 and H3 as seen
so far are too small for the decision test and would at best reach "difference
on this dataset only". More seeds do not remove this limit: with overlapping
splits the standard error cannot fall below about 0.58 × the between-seed
spread. This is a property of a 2640-trial dataset near its ceiling, not of the
models.

## What a positive result would and would not mean

- It would hold for this dataset, these seven subjects and a pooled random split
  in which every subject appears in train and test.
- `eegnet_original` is the authors' EEGNet settings **with an attention
  inserted**; their code has no attention in EEGNet. HCANN here is their
  attention module inside our backbones, not their full HCANN model.
- Their published figure (about 22%) was selected on test accuracy and is not
  comparable with any number produced here.

## Running it

In the project folder on the cluster, environment active:

```bash
python -m agfl_speech plan --preset si-hom-confirmatory
```

```bash
python -m agfl_speech sweep --preset si-hom-confirmatory \
  --output-dir results/si-hom-confirmatory-v1 --skip-completed
```

```bash
python -m agfl_speech analyze results/si-hom-confirmatory-v1
python scripts/confirmatory_analysis.py results/si-hom-confirmatory-v1
```

The second command writes `report/confirmatory/confirmatory.md`, `.json` and
`confirmatory_per_seed.csv`. It needs only numpy and scipy and also runs on a
downloaded `report/` folder. `--exploratory` applies the same computations to
other seeds or epoch budgets and labels the output accordingly.

Estimated cost: about 4–5 hours on one V100, from the first run's timings
scaled to 60 epochs. The authors'-recipe arms (batch 16) are not yet timed.

## Frozen files

| File | SHA-256 |
|---|---|
| `agfl_speech/presets/si-hom-confirmatory.json` | `4b2e88ef67606d2535bbd94b011085cd67452842df1a4f63732449f34dab98f1` |
| `scripts/confirmatory_analysis.py` | `8340e276763387b1e552785c522c9ce7203ed5aaf7c7cfa2e225502b38e9a4fa` |

The analysis script was exercised on the first run's report in exploratory mode
(seeds 0–4, 200 epochs) and reproduced the numbers quoted above. The package
source is unchanged from the first run, so the recorded source hash of the new
fits should equal the first run's.

## Deviations

None.

## Outcome (run completed 2026-10-02, report read 2026-10-03)

All 160 fits completed with the frozen preset and script, on the same data
fingerprint and source hash as the exploratory run (3.6 GPU hours). The analysis
in `report/confirmatory/confirmatory.md` gave:

| | Mean difference | 95% CI (corrected) | Seeds + / − | Verdict |
|---|---|---|---|---|
| H1 our EEGNet − authors' settings | −0.22 pts | [−2.47, +2.04] | 8 / 12 | not established |
| H2 AGFL − HCANN | +0.20 pts | [−0.85, +1.25] | 11 / 9 | not established |
| H3 MHA − HCANN | +0.01 pts | [−1.27, +1.29] | 10 / 10 | not established |

No secondary contrast was established either (S1 −0.07, S2 −0.13, S3 −0.35,
S4 +0.18, S5 −0.16 points). All eight arms sit at 15.4–15.9% test accuracy
(chance 12.5%; every arm above chance, p < 1e-6 over seeds). The exploratory
differences of +0.9 to +2.7 points did not reappear.

Conclusion under this plan: on the SI_Hom dataset neither our EEGNet nor our
attentions (AGFL, MHA) are better than the dataset authors' EEGNet settings and
attention, and neither are they worse; the differences are within ±1 point.


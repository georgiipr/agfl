# NeurIPS manuscript reference and next EEGNet comparison

Reference checked on 15 September 2026:
`/Users/egor/Downloads/NEU_art_submission.pdf`, 20 pages, titled
*Adaptive Graph Filtering Networks as an Alternative to Self-Attention*.
Its PDF creation/modification timestamp is 4 May 2026, 10:02:45 UTC.
This is the newest submission PDF found in Downloads, and is byte-identical
to `NEU_article-7.pdf`. The earlier `NEU_article_submission.pdf` and its
`(1)` copy date from 2 May. Use the newer submission PDF as the reference;
do not infer the current manuscript from `main.tex`.

Reference SHA256:
`603827ac3cdc3a6f9e8e2e43975662abd20ff8358f2fb85627ca79eefc7061c5`.

## Relevant claims in the PDF

- Page 6, Table 2: AGFL EEG accuracy 76.63%, standard attention 72.99%.
  The claimed difference is 3.64 percentage points. Macro F1 is 0.7626
  versus 0.7204; ROC-AUC is 0.8707 versus 0.8650.
- Page 4, Section 2.3: standard attention is multi-head self-attention.
- Page 11, Appendix A: EEG results are described as subject-specific
  training, averaged across all nine subjects.
- The reference also contains historical no-attention comparisons
  (pages 4 and 6) and binary EEG descriptions (pages 12–13). These are
  manuscript content, not instructions to add experimental variants or
  change the current task.

## Current user requirements take precedence

The EEGNet comparison uses **standard attention (MHA) versus AGFL**.
Do not add a no-attention option. Retain the current four-class BCI IV 2a
task, with each subject trained independently. The manuscript revision
must describe the experiments actually performed; historical scores are
not direct confirmations of the current four-class protocol.

## Fastest useful next pilot

Use **A03** for a matched MHA-versus-AGFL comparison, with the same fixed
`spatial_control` backbone, preprocessing, training settings and seeds 0–4.
This is ten fits. Avoid mixing augmentation choices or batch-size
changes into the attention comparison. Parameter counts may differ between
attention mechanisms and must be reported.

The `eegnet-a03-attention` preset now defines this comparison. The
[README workflow](../README.md#a03-eegnet-mha-versus-agfl) includes training,
checkpoint diagnostics for both methods, and aggregate plot commands.

**Pilot completed:** the subsequent [A03 attention review](eegnet_a03_attention_review.html)
finds MHA 85.56% and AGFL 84.81% mean test accuracy over five matched seeds.
This does not establish the claimed AGFL advantage. The user clarified that
improving AGFL comes before the full-study expansion. The
[current Q/K/V graph-filter workflow](eegnet_agfl_qkv_improvement.md)
provides the focused A03 search and plotting commands; the nine-subject
expansion is deferred.

The already completed spatial-control validation accuracies average
91.48% on A03 and 45.12% on A06 over seeds 0–4. The preceding
control/augmentation selection procedure yielded mean test accuracies
of 86.67% and 31.63%, respectively. Those selected scores are not results
for a single fixed configuration or a comparison against MHA.

A03 supplies a useful pilot because the existing EEGNet/AGFL pipeline
learns that subject reliably. The completed matched MHA results do not
establish an AGFL advantage on A03. The choice of A03 was informed
by existing results, so it cannot independently establish the paper's
across-subject claim. Preserve A06 in the subsequent all-nine-subject
evaluation, after fixing the comparison procedure.

Only document extraction and saved-artifact inspection were performed
during this reference check. No training, inference or project tests ran.

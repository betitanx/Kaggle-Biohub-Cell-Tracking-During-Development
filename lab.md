# Experimental notebook — public recovery edition

This notebook summarizes the scientific record recovered from the project history. Historical work spans August–September 2026. Recovery and software checks were performed on 19 October 2026. Kaggle scores below are preserved observations, not fresh API queries performed during recovery.

## Task and experimental method

The competition concerns cell detection, linking, and division tracking in 3D microscopy over time. Inputs use OME-Zarr and sparse GEFF annotations. Candidate outputs must satisfy the competition CSV and graph contract before they can be evaluated or considered for upload.

The loop was: formulate a hypothesis, construct a candidate, evaluate on explicit dataset/time splits, inspect the generalization gap and topology, record runtime and failures, then compare a small set of validated candidates. A public-score control and metric probes were kept distinct from improvements to the biological tracking pipeline.

The controller ranks evidence across candidate families and maintains a finite adaptive proposal queue. Uploads require explicit activation and a remote budget check. The paired-candidate workflow uses `require_all_candidates=true` so an incomplete pair does not consume the remaining slot. Public example configurations have real-data execution and live eligibility disabled.

## Experiments and outcomes

| Family | What changed or was tested | Historical outcome and scope |
|---|---|---|
| H1 classical | Intensity peaks and physical nearest-neighbor linking | Useful executable baseline and contract checks; weak cross-block generalization. The recomputing runner obtained public score 0.054. |
| Deep baseline | Organizer U-Net/transformer prediction with ILP | Own corrected submissions obtained 0.875 and 0.876. |
| Community-inspired clean variants | TTA, edge rescue, physical gating, legal gap closing | Clean public variants obtained 0.884 and 0.885. Metric probes were recorded separately and were not promoted as biological improvements. |
| H9 division and ILP ablations | Top-k children, edge floors, disappearance and division penalties | Best own recorded result: H9 v11 disappearance penalty 1.8, public score 0.898, reference 55950639. Later tested variants did not improve that public score. |
| H13 fine-tuning | Public initialization and controlled adaptation | Coverage failures and empty-edge outputs prevented promotion in the tested configuration. |
| H14–H16 exploratory work | Feature-space TTA, track consensus, DeepCenter detector rescue | Implementations and structural/smoke checks exist; these are not evidence of a better public result. |
| H17/H18 physical-radius work | Shared inference with controlled physical edge-radius filtering | Interrupted attempts were excluded. Later completed local diagnostic evaluations covered three embryos; the final reviewed figure records them. Sparse division labels and differing local scopes limit the conclusions. No new public-score improvement was established. |

## Resolving the submission failures

Early notebooks passed local CSV checks but received generic scoring-format errors. The operational correction was to use a notebook that discovers mounted test data and recomputes its output within Kaggle's evaluation environment, followed by validation of the actual generated artifact. A precomputed CSV copied into a notebook was not a reliable substitute for that execution contract.

The resulting preflight checks include expected dataset/timepoint coverage, contiguous nonnegative IDs, absence of NaN/empty values, legal temporal edges, maximum outgoing degree, division parents, connected components, runtime, and error-free logs. The H9 gate requires maximum out-degree at most two and no nonadjacent edges.

## Recorded results and effort

- Best public score in the account: **0.900**, reference **55308697**, a replay of a public baseline used as a control.
- Best own pipeline variant: **0.898**, reference **55950639**. It is not the best absolute score in the account.
- **28 local evaluations + 26 accepted submission references = 54 records**. The same candidate can appear at more than one evaluation stage; these are not 54 distinct models.
- The submission references include **17 numeric public scores** and **nine references without a numeric score** in the historical ledger. Missing values are not interpreted as zero.
- Estimated long-run execution time: **146 hours**, from `4 × 27 h + 38 h`, based on the owner's reports. Setup and analysis were not added.

[Public-score records](results/public-results.json) preserve the observed statuses, submission dates, references, and control/probe classifications. [The progress summary](results/progress-summary.json) preserves the daily aggregation. The [historical image](docs/assets/project-progress-snapshot.png) combines local and public observations by color and highlights the 0.900 account control with a pink star.

Adjusted local replay scores can exceed 1.0 and depend on split, annotation sparsity, and aggregation. They are not directly comparable with the public leaderboard. A small local improvement alone did not authorize promotion.

## Public leads and external material

The source baseline, Ultrack, Cellpose, StarDist, TracksData, OpenResearch, and public Kaggle notebooks were examined as implementation references or possible experiment families. The public catalog in `config/public_solutions.json` is historical discovery evidence; listed scores are not reproduced results of this repository.

Weights and Kaggle assets remain external. The best-own runner expects the support pack `pilkwang/biohub-tracking-support-pack-50ep-v1` and gap-closing input `yoikoarmor/biohub-last-call-postprocess-v1`. Their individual terms and access rules must be checked at acquisition time. No competition data, checkpoints, or third-party Kaggle notebook contents are redistributed in this export.

## Recovery boundary

The raw account ledger, private kernel paths, cloud operation identifiers, machine paths, and logs are omitted. Some raw predictions, local metric JSON files, environment lockfiles, and the original filesystem state were unavailable. The release provides recovered code, a curated record, historical evidence, and software checks; it does not claim a fresh end-to-end GPU reproduction. See [docs/RECOVERY.md](docs/RECOVERY.md).

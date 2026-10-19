# Experimental notebook

This notebook summarizes the scientific record of the project. Experimental work spans August–September 2026. Software checks and the signed-in Kaggle review were performed on 19 October 2026; the latter verified existing results without a new API query or inference rerun.

## Kaggle reconciliation — 19 October 2026

The submission details and the competition's “Your Work” notebook list are accessible again. There are 17 successful scored submissions and nine scoring errors; the earlier ledger's `COMPLETE` status without a score must not be read as a successful scored result.

| Variant | Public | Final/private | Selected for final evaluation |
|---|---:|---:|---|
| Public-baseline replay control, ref 55308697 | 0.90096 | 0.89586 | Yes |
| Own H9 v11 disappearance 1.8, ref 55950639 | 0.89856 | 0.89217 | Yes |
| Own H9 v13 edge floor 0.30, ref 55980843 | 0.89796 | 0.89259 | No |

H9 v11 remains the best own public result. H9 v13 has the best own final/private score among submitted variants, but it was not selected for the official final evaluation. The selected control still has the highest final score in the account. This post-competition evidence changes neither the historical selection nor the old progress figure.

Downloaded the original Version 1 notebooks `346635318` and `346992236`. Their code cells are preserved verbatim in `notebooks/`; outputs, execution state, and metadata are stripped. The only code difference is `edge_min_threshold=0.35` versus `0.30`. Preview runtimes were 1008.9 and 1203.7 seconds on T4 ×2. Both completed despite Gurobi-to-SCIP fallback warnings. No cloud run or new submission was started.

The input description also clarifies a key contract: the four visible preview datasets are not the full hidden test set. Reruns replace them with approximately training-sized hidden data. Coverage validation must use the datasets actually discovered during each execution, not a fixed list of four names. See [notebook instructions](notebooks/README.md) and [verified evidence](results/kaggle-review.json).

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

Weights and Kaggle assets remain external. The best-own runner expects the support pack `pilkwang/biohub-tracking-support-pack-50ep-v1` and gap-closing input `yoikoarmor/biohub-last-call-postprocess-v1`. Their terms and access rules must be checked at acquisition time. No competition data, checkpoints, or third-party Kaggle notebook downloads are redistributed. The included original notebooks belong to this account and load the external inputs at execution time.

## Validation scope

The repository includes experiment code, submitted inference notebooks, historical results, and software checks. Competition data, weights, credentials, operational account state, and raw predictions are not bundled. Lightweight validation covers tests, notebook syntax, synthetic runs, ranking, and graph topology; it does not establish a fresh end-to-end GPU reproduction of the leaderboard scores. Optional ML dependencies are not fully locked.

Filesystem safety updates on 19 October 2026: cache preparation never removes existing paths; CSV conversion validates every name/destination and defaults to no overwrite; subset downloads reject unsafe names and symlinks before writes. Regression tests were observed failing before the fixes. The complete lightweight check passed with 362 tests, two optional-data skips, and five subtests. These changes do not modify the original submitted notebooks or establish new competition scores. See [safe usage](docs/SAFE_USAGE.md) and [changelog](CHANGELOG.md).

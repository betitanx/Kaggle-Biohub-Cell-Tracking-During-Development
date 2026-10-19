# Inference notebooks

| File | Purpose | Kaggle Version 1 | Public / final score |
|---|---|---|---|
| [best-own-h9-v11.ipynb](best-own-h9-v11.ipynb) | Best own public result; selected for final evaluation | [346635318](https://www.kaggle.com/code/betitanx/biohub-cellmot-community-h9-bcd08e-92d6acc4?scriptVersionId=346635318) | 0.89856 / 0.89217 |
| [best-final-h9-v13.ipynb](best-final-h9-v13.ipynb) | Best own final/private score among submitted variants; not selected | [346992236](https://www.kaggle.com/code/betitanx/biohub-cellmot-community-h9-c2f668-2ad019de?scriptVersionId=346992236) | 0.89796 / 0.89259 |

These are the original downloaded code cells, not regenerated approximations. Outputs, execution state, and notebook metadata were removed; source and original download hashes are in [kaggle-review.json](../results/kaggle-review.json). The source pages were private to the owner at verification time, so those links may require permission. The copies here do not require access to the original notebook pages.

## Kaggle setup

1. Import either `.ipynb` into a new Kaggle notebook.
2. Attach the [competition](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development), the [support pack](https://www.kaggle.com/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1), and the [gap-closing input](https://www.kaggle.com/datasets/yoikoarmor/biohub-last-call-postprocess-v1).
3. Choose a GPU accelerator (historical preview: T4 ×2), disable Internet, then run. The support pack supplies offline wheels, weights, and predictor code.
4. Inspect the timing/gap-closing report and validate `/kaggle/working/submission.csv` before considering a late submission. Running this repository's build or tests never uploads automatically.

Both downloads record competition ID `136605` and dataset-version IDs `17804310` and `18189029`. The version IDs are an unordered pair because the download does not map each to a named dataset. Reattaching current versions may not reproduce the original environment.

The preview ran on four visible example datasets. The official data description says these are copies from training; submission reruns replace them with a hidden test set approximately as large as training. **Do not hard-code the four preview names or assume the hidden set has four datasets.** The original notebooks discover all mounted `.zarr` inputs.

The H9 v13 ablation changes only `edge_min_threshold` from 0.35 to 0.30. Both use XY dihedral-eight detection TTA, top-two parent/child edge rescue, a 10 µm edge gate, ILP, and legal gap closing. `metric_probe` is false; two inert probe parameters remain to preserve the original code.

Historical logs show Gurobi falling back to SCIP because a Gurobi license was unavailable. The fallback completed successfully; these warnings were not scoring failures.

## Regenerated versus submitted source

`python scripts/build_best_own_notebook.py --output outputs/regenerated-h9-v11.ipynb` builds the currently recovered runner with the v11 parameters. It does not submit anything or prove equivalence with the historical submission. Use the included original to study the actual submitted execution contract. No fresh GPU reproduction was performed during this review.

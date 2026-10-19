# Kaggle Biohub Cell Tracking During Development

A research laboratory for detecting and tracking cells in 3D time-lapse microscopy: reproducible candidate runs, graph validation, held-out evaluation, ablations, ranking, and explicit Kaggle submission controls.

This is a curated recovery of the completed project. The original working directory was unavailable; code and historical evidence were reconstructed from preserved development records. See [recovery notes](docs/RECOVERY.md) for the boundaries of that reconstruction.

## Results and scope

| Historical result | Public score | Submission reference |
|---|---:|---|
| Best score in the account: replay of a public baseline, used as a control | 0.900 | 55308697 |
| Best own pipeline variant: H9 v11, disappearance penalty 1.8 | 0.898 | 55950639 |

The control is a public-baseline replay and is not an original algorithm developed in this project. The historical record contains **28 local evaluations and 26 accepted Kaggle submission references**, for **54 evaluation/submission records**. These are not 54 unique architectures. Of the 26 references, 17 have numeric public scores and nine have no numeric score in the record. See [public results](results/public-results.json) and [the experimental notebook](lab.md).

The runtime estimate is **146 hours**, calculated conservatively from the owner's reported long runs: four runs of about 27 hours plus one of 38 hours. It is an estimate of execution time, not a measurement of hands-on effort or billed GPU time.

![Historical project progress](docs/assets/project-progress-snapshot.png)

Local OOF/holdout/replay scores use different scopes and an adjusted local metric that can exceed 1.0. They should not be interpreted as Kaggle public scores. The image was recovered from the final reviewed historical preview; its source data and raw predictions were not all recovered.

## Quick start

Python 3.11 or later is required. The recovery checks use Python 3.12.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pip install --no-deps -e .
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Run a small synthetic demonstration without competition data, credentials, GPU use, or network access:

```bash
python -m biohub_lab.cli run --config config/experiments.json \
  --duration-seconds 2 --max-experiments 2 --output-dir runs/demo
python -m biohub_lab.cli rank --runs-dir runs/demo --output-dir outputs/demo
python -m biohub_lab.cli validate --submission tests/fixtures/submission.csv
```

Only the synthetic smoke candidates are enabled in the public default configuration. Real-data candidates and live eligibility are disabled. The synthetic scores exercise the software contract and are not biological results.

## Reproduce the best own configuration

The historical H9 v11 parameters are preserved in [config/best-own-h9-v11.json](config/best-own-h9-v11.json). The generated [best-own notebook](notebooks/best-own-h9-v11.ipynb) is included, or you can regenerate it locally:

```bash
python scripts/build_best_own_notebook.py --output outputs/best-own-h9-v11.ipynb
```

This command writes a notebook; it does not upload or run it. To run it on Kaggle, attach the competition and the public inputs listed in that configuration using your own account and access permissions. Dataset licenses and competition rules apply separately. Models, images, labels, checkpoints, private kernels, and credentials are not bundled here. A rerun's score is not guaranteed to match the historical score.

The notebook runner recomputes predictions from mounted competition inputs. The validation layer checks IDs, coverage, finite values, timepoints, edge direction, division degree, and graph topology. Historical scoring failures and their resolution are described in [lab.md](lab.md).

Run the complete lightweight release check with `python scripts/verify_public_release.py`. The recorded [validation report](results/validation.json) covers 221 passing tests, two skipped optional-data tests, notebook syntax, a synthetic two-candidate run, ranking, and topology validation. This check does not run the historical competition workload.

## Repository guide

- `src/biohub_lab/`: control plane, validation, metrics bridges, classical and deep-model adapters, notebook builders, and experimental postprocessing.
- `tests/`: unit/regression tests and a small synthetic CSV fixture.
- `config/`: sanitized, disabled real-data examples and the recorded best-own parameters.
- `scripts/`: local utilities and optional explicit Kaggle/holdout launchers.
- `external/kaggle-cell-tracking-competition/`: the BSD-licensed source files needed by the baseline wrappers, with license and attribution preserved.
- `results/`: curated historical public-score evidence and recovery validation summaries.
- `docs/`: recovery notes, architecture decisions, and the restored progress image.

The H13 fine-tuning, H15 motion-consensus, H16 DeepCenter, and H17/H18 radius experiments are research code with different validation scopes; their inclusion does not establish public-score improvements. Optional ML integrations load externally supplied Python and model checkpoints. Use only reviewed, trusted sources and verified files. The historical Modal integration is included for reference and has not been revalidated against a live Modal service during recovery.

## Provenance and attribution

The organizer's [baseline](https://github.com/royerlab/kaggle-cell-tracking-competition) combines a 3D U-Net, transformer-based linking, and ILP. Its source is BSD-3-Clause. The laboratory adds experiment orchestration, validation gates, ablations, and adapters; original and third-party code remain distinguished in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The hypothesis-selection workflow was informed by [arXiv:2301.12987v4](https://arxiv.org/pdf/2301.12987v4), operationalized here through held-out generalization, controlled ablations, and complexity checks. Public competition discussions and notebooks were treated as leads to investigate, not as independently verified performance evidence.

No blanket license has been selected for the original laboratory code in this recovery export. The bundled upstream files retain their BSD license. See [LICENSE_INFO.md](LICENSE_INFO.md).

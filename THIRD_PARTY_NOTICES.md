# Third-party notices

## Organizer baseline

Source: https://github.com/royerlab/kaggle-cell-tracking-competition

Upstream source commit: `075fc5f5a52d11077f9dc2b074644618f26939e2`.

The relevant Python files under `external/kaggle-cell-tracking-competition/` are third-party source under the **BSD 3-Clause License**, copyright (c) 2026 Thibaut Goldsborough. Its full license is preserved at `external/kaggle-cell-tracking-competition/LICENSE`. The bundled source includes compatibility modifications and is not presented as pristine upstream code. The notebook builder includes that license when packaging this source.

## Optional integrations and research references

The laboratory references Ultrack, TracksData, Cellpose, StarDist, and OpenResearch through adapters or proposals. Their packages are not redistributed here. Their respective upstream licenses and terms govern separate installation and use:

- https://github.com/royerlab/ultrack
- https://github.com/royerlab/tracksdata
- https://github.com/MouseLand/cellpose
- https://github.com/stardist/stardist
- https://github.com/alphaXiv/OpenResearch

## Kaggle inputs

Competition data, pretrained weights, and public Kaggle support/postprocessing inputs are not included. References to them do not grant redistribution rights. Acquire them through their original pages and comply with the terms of each input and the competition.

The submitted notebooks load these inputs at runtime. Their references, import statements, and configuration parameters do not relicense the external source, wheels, weights, or data. Dataset-version IDs identify historical inputs but are not permission grants.

The [competition rules](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/rules) restrict redistribution of Competition Data and specify conditions for public code sharing through the competition's forum/notebooks. A public GitHub repository alone must not be assumed to satisfy every competition-specific sharing condition.

## Project visualization

`docs/assets/project-progress-snapshot.png` is the project's results visualization. It does not include competition image volumes, annotations, or model checkpoints.

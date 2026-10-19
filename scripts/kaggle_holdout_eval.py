#!/usr/bin/env python
"""Private Kaggle evaluator for complete train pairs.

This notebook source is intentionally separate from the live-submission
runner.  It evaluates the current H9 lineage on complete competition train
image/GEFF pairs, writes candidate CSVs and official local metric rows, and
never uploads a submission.  In particular, this evaluator intentionally
never invokes competition_submit_code.
"""

from __future__ import annotations

import contextlib
import csv
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


COMPETITION_ROOT_CANDIDATES = (
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
)
WORKING_DIR = Path(os.environ.get("BIOHUB_WORKING_DIR", "/kaggle/working"))
EXPECTED_TIMEPOINTS = int(os.environ.get("BIOHUB_EXPECTED_TIMEPOINTS", "100"))
MAX_DATASETS = int(os.environ.get("BIOHUB_HOLDOUT_MAX_DATASETS", "1"))
SEED = os.environ.get("BIOHUB_HOLDOUT_SEED", "42")
RUNNER_CONFIG = {
    "det_threshold": 0.965,
    "edge_threshold": 0.48,
    "edge_min_threshold": 0.35,
    "edge_topk_parents": 2,
    "edge_topk_children": 2,
    "edge_max_distance_um": 10.0,
    "det_tta_mode": "dihedral8",
    "feature_tta": False,
    "feature_tta_blend": 1.0,
    "motion_consensus": False,
    "motion_consensus_max_distance_um": 7.0,
    "motion_consensus_max_prediction_error_um": 3.0,
    "motion_consensus_require_unique_parent": True,
    "motion_consensus_max_out_degree": 2,
    "deepcenter": False,
    "deepcenter_max_endpoint_distance_um": 7.0,
    "deepcenter_min_existing_distance_um": 2.0,
    "deepcenter_require_unique_parent": True,
    "deepcenter_max_out_degree": 2,
    "pool_kernel_um": 3.0,
    "ilp_appearance_weight": 0.0,
    "ilp_disappearance_weight": 1.4,
    "ilp_division_weight": 0.5,
    "gap_closing": True,
    "gap_closing_max_total_distance_um": 10.0,
    "gap_closing_reuse_existing_um": 2.5,
    "gap_closing_max_added_fraction": 0.02,
    "gap_closing_max_added_absolute": 1000,
    "metric_probe": False,
}


def _parse_disappearance_weights(raw: str | None) -> tuple[float, ...]:
    """Parse an optional, explicit ILP disappearance-weight ablation list."""

    if raw is None or not raw.strip():
        return ()
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError("disappearance weights cannot contain empty values")
        value = float(token)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("disappearance weights must be finite and non-negative")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("disappearance weights must be distinct")
    return tuple(values)


def _disappearance_variant_specs(
    weights: tuple[float, ...],
) -> tuple[tuple[str, float], ...]:
    """Return stable output labels for a disappearance-weight ablation."""

    specs = tuple(
        ("disappear{:03d}".format(int(round(weight * 100))), float(weight))
        for weight in weights
    )
    labels = [label for label, _weight in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("disappearance weights are too close to label distinctly")
    return specs


def _parse_division_weights(raw: str | None) -> tuple[float, ...]:
    """Parse an optional, explicit ILP division-weight ablation list."""

    if raw is None or not raw.strip():
        return ()
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError("division weights cannot contain empty values")
        value = float(token)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("division weights must be finite and non-negative")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("division weights must be distinct")
    return tuple(values)


def _division_variant_specs(
    weights: tuple[float, ...],
) -> tuple[tuple[str, float], ...]:
    """Return stable output labels for a division-weight ablation."""

    specs = tuple(
        ("division{:03d}".format(int(round(weight * 100))), float(weight))
        for weight in weights
    )
    labels = [label for label, _weight in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("division weights are too close to label distinctly")
    return specs


def _parse_edge_floors(raw: str | None) -> tuple[float, ...]:
    """Parse an optional, explicit edge-candidate floor ablation list."""

    if raw is None or not raw.strip():
        return ()
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError("edge floors cannot contain empty values")
        value = float(token)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("edge floors must be finite values between 0 and 1")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("edge floors must be distinct")
    return tuple(values)


def _edge_floor_variant_specs(
    floors: tuple[float, ...],
) -> tuple[tuple[str, float], ...]:
    """Return stable output labels for an edge-floor ablation."""

    specs = tuple(
        ("edge{:03d}".format(int(round(floor * 100))), float(floor))
        for floor in floors
    )
    labels = [label for label, _floor in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("edge floors are too close to label distinctly")
    return specs


def _parse_edge_radius_variants(raw: str | None) -> tuple[float, ...]:
    """Parse the two physical edge-radius values for a linking ablation."""

    if raw is None or not raw.strip():
        return ()
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError("edge radius variants cannot contain empty values")
        value = float(token)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("edge radius variants must be finite and positive")
        values.append(value)
    if len(values) != 2:
        raise ValueError("edge radius holdout requires exactly two values")
    if len(set(values)) != len(values):
        raise ValueError("edge radius variants must be distinct")
    return tuple(values)


def _edge_radius_variant_specs(
    radii: tuple[float, ...],
) -> tuple[tuple[str, float], ...]:
    """Return stable labels for a physical edge-radius ablation."""

    specs = tuple(
        ("radius{:03d}".format(int(round(radius * 10))), float(radius))
        for radius in radii
    )
    labels = [label for label, _radius in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("edge radius variants are too close to label distinctly")
    return specs


def _parse_center_thresholds(raw: str | None) -> tuple[float, ...]:
    """Parse the two explicit DeepCenter heatmap thresholds."""

    if raw is None or not raw.strip():
        return ()
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            raise ValueError("DeepCenter thresholds cannot contain empty values")
        value = float(token)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("DeepCenter thresholds must be finite values between 0 and 1")
        values.append(value)
    if len(values) != 2:
        raise ValueError("DeepCenter holdout requires exactly two thresholds")
    if len(set(values)) != len(values):
        raise ValueError("DeepCenter thresholds must be distinct")
    return tuple(values)


def _center_variant_specs(
    thresholds: tuple[float, ...],
) -> tuple[tuple[str, float], ...]:
    """Return stable labels for a strict/balanced DeepCenter pair."""

    specs = tuple(
        ("deepcenter{:03d}".format(int(round(threshold * 100))), float(threshold))
        for threshold in thresholds
    )
    labels = [label for label, _threshold in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("DeepCenter thresholds are too close to label distinctly")
    return specs


DISAPPEARANCE_WEIGHTS = _parse_disappearance_weights(
    os.environ.get("BIOHUB_HOLDOUT_DISAPPEARANCE_WEIGHTS")
)
DIVISION_WEIGHTS = _parse_division_weights(
    os.environ.get("BIOHUB_HOLDOUT_DIVISION_WEIGHTS")
)
EDGE_FLOORS = _parse_edge_floors(
    os.environ.get("BIOHUB_HOLDOUT_EDGE_FLOORS")
)
EDGE_RADIUS_VARIANTS = _parse_edge_radius_variants(
    os.environ.get("BIOHUB_HOLDOUT_EDGE_RADIUS_VARIANTS")
)
EDGE_MAX_DISTANCE_RAW = os.environ.get("BIOHUB_HOLDOUT_EDGE_MAX_DISTANCE_UM")
if EDGE_MAX_DISTANCE_RAW is not None:
    RUNNER_CONFIG["edge_max_distance_um"] = float(EDGE_MAX_DISTANCE_RAW)
if (
    not math.isfinite(float(RUNNER_CONFIG["edge_max_distance_um"]))
    or float(RUNNER_CONFIG["edge_max_distance_um"]) <= 0.0
):
    raise ValueError("BIOHUB_HOLDOUT_EDGE_MAX_DISTANCE_UM must be finite and positive")
if EDGE_RADIUS_VARIANTS:
    RUNNER_CONFIG["edge_max_distance_um"] = max(EDGE_RADIUS_VARIANTS)
DET_TTA_MODE_RAW = os.environ.get("BIOHUB_HOLDOUT_DET_TTA_MODE")
if DET_TTA_MODE_RAW is not None:
    det_tta_mode = DET_TTA_MODE_RAW.strip().lower()
    if det_tta_mode not in {"flip4", "dihedral8"}:
        raise ValueError("BIOHUB_HOLDOUT_DET_TTA_MODE must be flip4 or dihedral8")
    RUNNER_CONFIG["det_tta_mode"] = det_tta_mode
BASE_DISAPPEARANCE_WEIGHTS = _parse_disappearance_weights(
    os.environ.get("BIOHUB_HOLDOUT_BASE_DISAPPEARANCE_WEIGHT")
)
if len(BASE_DISAPPEARANCE_WEIGHTS) > 1:
    raise ValueError("base disappearance weight must contain exactly one value")
if EDGE_RADIUS_VARIANTS and not BASE_DISAPPEARANCE_WEIGHTS:
    raise ValueError(
        "edge-radius variants require BIOHUB_HOLDOUT_BASE_DISAPPEARANCE_WEIGHT"
    )
if BASE_DISAPPEARANCE_WEIGHTS:
    RUNNER_CONFIG["ilp_disappearance_weight"] = BASE_DISAPPEARANCE_WEIGHTS[0]
FEATURE_TTA_RAW = os.environ.get("BIOHUB_HOLDOUT_FEATURE_TTA")
if FEATURE_TTA_RAW is not None:
    feature_tta_value = FEATURE_TTA_RAW.strip().lower()
    if feature_tta_value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError("BIOHUB_HOLDOUT_FEATURE_TTA must be a boolean value")
    RUNNER_CONFIG["feature_tta"] = feature_tta_value in {"1", "true", "yes", "on"}
FEATURE_TTA_BLEND_RAW = os.environ.get("BIOHUB_HOLDOUT_FEATURE_TTA_BLEND")
if FEATURE_TTA_BLEND_RAW is not None:
    RUNNER_CONFIG["feature_tta_blend"] = float(FEATURE_TTA_BLEND_RAW)
if not math.isfinite(float(RUNNER_CONFIG["feature_tta_blend"])):
    raise ValueError("BIOHUB_HOLDOUT_FEATURE_TTA_BLEND must be finite")
if not 0.0 <= float(RUNNER_CONFIG["feature_tta_blend"]) <= 1.0:
    raise ValueError("BIOHUB_HOLDOUT_FEATURE_TTA_BLEND must be in [0, 1]")


def _parse_boolean_env(name: str, raw: str | None, default: bool) -> bool:
    if raw is None:
        return default
    value = raw.strip().lower()
    if value not in {"0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(name + " must be a boolean value")
    return value in {"1", "true", "yes", "on"}


RUNNER_CONFIG["motion_consensus"] = _parse_boolean_env(
    "BIOHUB_HOLDOUT_MOTION_CONSENSUS",
    os.environ.get("BIOHUB_HOLDOUT_MOTION_CONSENSUS"),
    bool(RUNNER_CONFIG["motion_consensus"]),
)
for _name, _env_name in (
    ("motion_consensus_max_distance_um", "BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_DISTANCE_UM"),
    ("motion_consensus_max_prediction_error_um", "BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_PREDICTION_ERROR_UM"),
):
    _raw = os.environ.get(_env_name)
    if _raw is not None:
        RUNNER_CONFIG[_name] = float(_raw)
    if not math.isfinite(float(RUNNER_CONFIG[_name])) or float(RUNNER_CONFIG[_name]) < 0.0:
        raise ValueError(_env_name + " must be finite and non-negative")
RUNNER_CONFIG["motion_consensus_require_unique_parent"] = _parse_boolean_env(
    "BIOHUB_HOLDOUT_MOTION_CONSENSUS_REQUIRE_UNIQUE_PARENT",
    os.environ.get("BIOHUB_HOLDOUT_MOTION_CONSENSUS_REQUIRE_UNIQUE_PARENT"),
    bool(RUNNER_CONFIG["motion_consensus_require_unique_parent"]),
)
_motion_degree_raw = os.environ.get("BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_OUT_DEGREE")
if _motion_degree_raw is not None:
    RUNNER_CONFIG["motion_consensus_max_out_degree"] = int(_motion_degree_raw)
if int(RUNNER_CONFIG["motion_consensus_max_out_degree"]) not in {1, 2}:
    raise ValueError("BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_OUT_DEGREE must be 1 or 2")
DEEPCENTER_THRESHOLDS_RAW = os.environ.get("BIOHUB_HOLDOUT_DEEPCENTER_THRESHOLDS")
DEEPCENTER_THRESHOLDS = _parse_center_thresholds(DEEPCENTER_THRESHOLDS_RAW)
DEEPCENTER_ENABLED = _parse_boolean_env(
    "BIOHUB_HOLDOUT_DEEPCENTER",
    os.environ.get("BIOHUB_HOLDOUT_DEEPCENTER"),
    bool(DEEPCENTER_THRESHOLDS),
)
if DEEPCENTER_ENABLED and not DEEPCENTER_THRESHOLDS:
    DEEPCENTER_THRESHOLDS = (0.4, 0.35)
if not DEEPCENTER_ENABLED and DEEPCENTER_THRESHOLDS:
    raise ValueError("DeepCenter thresholds require BIOHUB_HOLDOUT_DEEPCENTER=true")
RUNNER_CONFIG["deepcenter"] = DEEPCENTER_ENABLED
for _name, _env_name in (
    ("deepcenter_max_endpoint_distance_um", "BIOHUB_HOLDOUT_DEEPCENTER_MAX_ENDPOINT_DISTANCE_UM"),
    ("deepcenter_min_existing_distance_um", "BIOHUB_HOLDOUT_DEEPCENTER_MIN_EXISTING_DISTANCE_UM"),
):
    _raw = os.environ.get(_env_name)
    if _raw is not None:
        RUNNER_CONFIG[_name] = float(_raw)
    if not math.isfinite(float(RUNNER_CONFIG[_name])) or float(RUNNER_CONFIG[_name]) < 0.0:
        raise ValueError(_env_name + " must be finite and non-negative")
RUNNER_CONFIG["deepcenter_require_unique_parent"] = _parse_boolean_env(
    "BIOHUB_HOLDOUT_DEEPCENTER_REQUIRE_UNIQUE_PARENT",
    os.environ.get("BIOHUB_HOLDOUT_DEEPCENTER_REQUIRE_UNIQUE_PARENT"),
    bool(RUNNER_CONFIG["deepcenter_require_unique_parent"]),
)
_deepcenter_degree_raw = os.environ.get("BIOHUB_HOLDOUT_DEEPCENTER_MAX_OUT_DEGREE")
if _deepcenter_degree_raw is not None:
    RUNNER_CONFIG["deepcenter_max_out_degree"] = int(_deepcenter_degree_raw)
if int(RUNNER_CONFIG["deepcenter_max_out_degree"]) not in {1, 2}:
    raise ValueError("BIOHUB_HOLDOUT_DEEPCENTER_MAX_OUT_DEGREE must be 1 or 2")
if DEEPCENTER_ENABLED and RUNNER_CONFIG["motion_consensus"]:
    raise ValueError("configure DeepCenter and motion consensus in separate holdout cycles")
if sum(bool(value) for value in (DISAPPEARANCE_WEIGHTS, DIVISION_WEIGHTS, EDGE_FLOORS, EDGE_RADIUS_VARIANTS, DEEPCENTER_THRESHOLDS)) > 1:
    raise ValueError(
        "configure only one holdout ablation family at a time"
    )
REQUIRED_IMPORTS = (
    "tracksdata",
    "zarr",
    "numcodecs",
    "geff",
    "pyscipopt",
    "ilpy",
    "rustworkx",
    "polars",
    "dask",
)
OFFLINE_PACKAGES = (
    "tracksdata",
    "zarr==3.2.1",
    "numcodecs==0.15.1",
    "donfig==0.8.1.post1",
    "geff==1.2.0.1.1",
    "geff-spec==1.1.1",
    "pyscipopt==6.2.1",
    "ilpy==0.6.0",
    "rustworkx==0.18.0",
    "polars==1.42.0",
    "polars-runtime-32==1.42.0",
    "bidict==0.23.1",
    "imagecodecs==2026.6.26",
    "dask==2026.6.0",
    "blosc2==4.7.0",
    "pyarrow==24.0.0",
    "sqlalchemy==2.0.51",
)
COLUMNS = [
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
]


def _find_support_pack() -> Path:
    override = os.environ.get("BIOHUB_SUPPORT_PACK_DIR", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if (
            (candidate / "repo" / "src" / "biohub_tracking").is_dir()
            and (candidate / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth").is_file()
            and (candidate / "wheels").is_dir()
        ):
            return candidate
        raise FileNotFoundError(
            "BIOHUB_SUPPORT_PACK_DIR is not a valid public support pack: "
            + str(candidate)
        )
    for manifest in sorted(Path("/kaggle/input").rglob("ARTIFACT_MANIFEST.json")):
        root = manifest.parent
        if (
            (root / "repo" / "src" / "biohub_tracking").is_dir()
            and (root / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth").is_file()
            and (root / "wheels").is_dir()
        ):
            return root
    raise FileNotFoundError("public Biohub tracking support pack was not attached")


def _find_deepcenter_artifact() -> Path:
    """Locate the audited public DeepCenter source/checkpoint pack."""

    for manifest in sorted(Path("/kaggle/input").rglob("ARTIFACT_MANIFEST.json")):
        root = manifest.parent
        if (
            (root / "weights" / "full_frame_center" / "best.pt").is_file()
            and (root / "source_scripts" / "train_full_frame_center_detector.py").is_file()
        ):
            return root
    raise FileNotFoundError("public DeepCenter detector pack was not attached")


def _require_offline_dependencies(support_pack: Path) -> None:
    missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
    if not missing:
        return
    wheels = support_pack / "wheels"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--no-index",
        "--no-deps",
        "--find-links",
        str(wheels),
        *OFFLINE_PACKAGES,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stdout + "\n" + completed.stderr).strip()[-4000:]
        raise RuntimeError("offline dependency installation failed: " + detail)
    importlib.invalidate_caches()


def _resolve_competition_root() -> Path:
    override = os.environ.get("BIOHUB_COMPETITION_DIR", "").strip()
    candidates = (Path(override),) if override else COMPETITION_ROOT_CANDIDATES
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    raise FileNotFoundError(
        "competition train/test directories not found: "
        + ", ".join(str(item) for item in candidates)
    )


def _patch_support_predictor(predict_script: Path) -> None:
    """Apply the audited H9 rescue and XY-dihedral TTA patch."""

    source = predict_script.read_text(encoding="utf-8")
    if "edge_topk_parents" in source:
        return
    config_marker = "    threshold: float = 0.5\n"
    if config_marker not in source:
        raise RuntimeError("support-pack predictor config marker not found")
    source = source.replace(
        config_marker,
        config_marker
        + "    edge_min_threshold: float = 0.0\n"
        + "    edge_topk_parents: int = 0\n"
        + "    edge_topk_children: int = 0\n"
        + "    edge_max_distance_um: float = 0.0\n",
        1,
    )
    candidate_block = """            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )
"""
    candidate_replacement = """            candidate_pairs = {
                (i, j)
                for i in range(n_src)
                for j in range(n_tgt)
                if probs[i, j] > cfg.threshold
            }
            if cfg.edge_topk_parents > 0 and cfg.edge_min_threshold > 0.0:
                for j in range(n_tgt):
                    parents = np.argsort(probs[:, j])[::-1][:cfg.edge_topk_parents]
                    for i in parents:
                        if probs[i, j] >= cfg.edge_min_threshold:
                            candidate_pairs.add((int(i), j))
            if cfg.edge_topk_children > 0 and cfg.edge_min_threshold > 0.0:
                for i in range(n_src):
                    children = np.argsort(probs[i, :])[::-1][:cfg.edge_topk_children]
                    for j in children:
                        if probs[i, j] >= cfg.edge_min_threshold:
                            candidate_pairs.add((i, int(j)))
            candidates = sorted(
                [(probs[i, j], i, j) for i, j in candidate_pairs],
                reverse=True,
            )
"""
    if candidate_block not in source:
        raise RuntimeError("support-pack predictor edge candidate block not found")
    source = source.replace(candidate_block, candidate_replacement, 1)
    distance_block = """                dist = float(np.linalg.norm(
                    coords_so_far[gi, 1:].astype(np.float32)
                    - coords_so_far[gj, 1:].astype(np.float32)
                ))
"""
    distance_replacement = """                dist = float(np.linalg.norm(
                    (coords_so_far[gi, 1:].astype(np.float32)
                     - coords_so_far[gj, 1:].astype(np.float32))
                    * np.asarray(voxel_size, dtype=np.float32)
                ))
                if cfg.edge_max_distance_um > 0.0 and dist > cfg.edge_max_distance_um:
                    continue
"""
    if distance_block not in source:
        raise RuntimeError("support-pack predictor distance block not found")
    source = source.replace(distance_block, distance_replacement, 1)
    tta_config_marker = "    edge_max_distance_um: float = 0.0\n"
    if tta_config_marker not in source:
        raise RuntimeError("support-pack predictor TTA config marker not found")
    source = source.replace(
        tta_config_marker,
        tta_config_marker
        + '    det_tta_mode: str = "flip4"\n'
        + '    feature_tta: bool = False\n'
        + '    feature_tta_blend: float = 1.0\n',
        1,
    )
    tta_block = """        # Detection TTA: original + flip-x + flip-y + flip-xy, average logits.
        # TTA: flip along Y (-2) and X (-1) only.  Z is excluded because
        # the data is highly anisotropic (Z resolution ~4x coarser than XY),
        # so Z-flips would produce out-of-distribution inputs.
        if cfg.det_tta:
            tta_flips = [(-1,), (-2,), (-2, -1)]
            for dims in tta_flips:
                imgs_flip = imgs.flip(dims)
                _, det_flip = model.encode(imgs_flip)
                for f in range(W):
                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
                del imgs_flip, det_flip
            for f in range(W):
                det_logits[f] = det_logits[f] / 4
"""
    tta_replacement = """        # Reuse the same XY transform family for detection TTA and for
        # optional feature-space TTA.  The inverse transform restores the
        # encoded feature map before node/edge feature pooling.
        if cfg.det_tta or cfg.feature_tta:
            if cfg.det_tta_mode == "dihedral8":
                tta_transforms = [
                    (lambda value: value.flip((-1,)), lambda value: value.flip((-1,))),
                    (lambda value: value.flip((-2,)), lambda value: value.flip((-2,))),
                    (lambda value: value.flip((-2, -1)), lambda value: value.flip((-2, -1))),
                    (lambda value: torch.rot90(value, 1, dims=(-2, -1)),
                     lambda value: torch.rot90(value, -1, dims=(-2, -1))),
                    (lambda value: torch.rot90(value, -1, dims=(-2, -1)),
                     lambda value: torch.rot90(value, 1, dims=(-2, -1))),
                    (lambda value: value.transpose(-1, -2),
                     lambda value: value.transpose(-1, -2)),
                    (lambda value: torch.rot90(value, 1, dims=(-2, -1)).transpose(-1, -2),
                     lambda value: torch.rot90(value.transpose(-1, -2), -1, dims=(-2, -1))),
                ]
            else:
                tta_transforms = [
                    (lambda value: value.flip((-1,)), lambda value: value.flip((-1,))),
                    (lambda value: value.flip((-2,)), lambda value: value.flip((-2,))),
                    (lambda value: value.flip((-2, -1)), lambda value: value.flip((-2, -1))),
                ]
            feature_sum = unet_out.clone() if cfg.feature_tta else None
            for forward, inverse in tta_transforms:
                imgs_flip = forward(imgs)
                unet_flip, det_flip = model.encode(imgs_flip)
                if cfg.feature_tta:
                    feature_sum = feature_sum + inverse(unet_flip)
                if cfg.det_tta:
                    for f in range(W):
                        det_logits[f] = det_logits[f] + inverse(det_flip[f])
                del imgs_flip, unet_flip, det_flip
            if cfg.det_tta:
                for f in range(W):
                    det_logits[f] = det_logits[f] / (len(tta_transforms) + 1)
            if cfg.feature_tta:
                tta_mean = feature_sum / (len(tta_transforms) + 1)
                blend = float(cfg.feature_tta_blend)
                unet_out = unet_out * (1.0 - blend) + tta_mean * blend
                del feature_sum, tta_mean
"""
    if tta_block not in source:
        raise RuntimeError("support-pack predictor TTA block not found")
    source = source.replace(tta_block, tta_replacement, 1)
    progress_marker = "        del unet_out\n"
    progress_replacement = """        del unet_out
        if len(seen_pairs) % 10 == 0 or ws == window_starts[-1]:
            print(
                "HOLDOUT_WINDOW_PROGRESS:"
                + json.dumps({"window_end": int(ws + W), "pairs": len(seen_pairs)}),
                flush=True,
            )
"""
    if progress_marker not in source:
        raise RuntimeError("support-pack predictor progress marker not found")
    source = source.replace(progress_marker, progress_replacement, 1)
    predict_script.write_text(source, encoding="utf-8")


def _load_graph(path: Path, td: Any) -> Any:
    result = td.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def _scope_audit(train_dir: Path, td: Any, open_dataset: Any, geff_metadata: Any) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for image_path in sorted(train_dir.glob("*.zarr")):
        name = image_path.stem
        gt_path = train_dir / (name + ".geff")
        record: dict[str, Any] = {
            "dataset": name,
            "image": str(image_path),
            "ground_truth": str(gt_path),
            "frame_count": None,
            "image_timepoints": [],
            "gt_timepoints": [],
            "gt_timepoints_within_image_scope": False,
            "gt_nodes": None,
            "gt_edges": None,
            "estimated_number_of_nodes": None,
            "actual_to_metadata_node_ratio": None,
            "complete_time_scope": False,
            "metadata_scope_consistent": False,
            "eligible": False,
            "reason": None,
        }
        if not gt_path.exists():
            record["reason"] = "matching GEFF directory is missing"
            audits.append(record)
            continue
        try:
            dataset = open_dataset(
                image_path,
                normalize=False,
                require_tracks=True,
                load_image=False,
            )
            gt_graph = _load_graph(gt_path, td)
            frame_count = int(dataset.image_shape[0])
            image_times = list(range(frame_count))
            gt_times = sorted({int(row["t"]) for row in gt_graph.node_attrs().iter_rows(named=True)})
            estimated = (geff_metadata.read(gt_path).extra or {}).get("estimated_number_of_nodes")
            estimated_value = float(estimated) if estimated is not None else float("nan")
            actual_nodes = int(gt_graph.num_nodes())
            ratio = actual_nodes / estimated_value if math.isfinite(estimated_value) and estimated_value > 0 else float("nan")
            gt_timepoints_within_image_scope = all(
                0 <= timepoint < frame_count for timepoint in gt_times
            )
            # Ground truth is intentionally sparse: it need not contain a
            # node at every frame. Completeness here means a full 100-frame
            # image paired with annotations that stay inside that image.
            complete_time_scope = (
                frame_count == EXPECTED_TIMEPOINTS
                and gt_timepoints_within_image_scope
            )
            # estimated_number_of_nodes is the coarse total-node denominator
            # used by the official adjusted metric, not the sparse GEFF node
            # count. A sparse annotation must therefore be no larger than it.
            metadata_scope_consistent = (
                math.isfinite(estimated_value)
                and estimated_value > 0
                and 0 < actual_nodes <= estimated_value
            )
            record.update(
                {
                    "frame_count": frame_count,
                    "image_timepoints": image_times,
                    "gt_timepoints": gt_times,
                    "gt_timepoints_within_image_scope": gt_timepoints_within_image_scope,
                    "gt_nodes": actual_nodes,
                    "gt_edges": int(gt_graph.num_edges()),
                    "estimated_number_of_nodes": estimated_value if math.isfinite(estimated_value) else None,
                    "actual_to_metadata_node_ratio": ratio if math.isfinite(ratio) else None,
                    "complete_time_scope": complete_time_scope,
                    "metadata_scope_consistent": metadata_scope_consistent,
                    "eligible": complete_time_scope and metadata_scope_consistent,
                    "reason": (
                        None
                        if complete_time_scope and metadata_scope_consistent
                        else "image/annotation scope or node-denominator check failed"
                    ),
                }
            )
        except Exception as exc:
            record["reason"] = type(exc).__name__ + ": " + str(exc)
        audits.append(record)
    return audits


def _select_datasets(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible = [item for item in audits if item["eligible"]]
    if MAX_DATASETS <= 0:
        raise ValueError("BIOHUB_HOLDOUT_MAX_DATASETS must be positive")
    if not eligible:
        raise RuntimeError(
            "no complete train/GEFF pairs are eligible: "
            + json.dumps(audits, sort_keys=True)
        )
    return sorted(
        eligible,
        key=lambda item: hashlib.sha256((SEED + ":" + item["dataset"]).encode()).hexdigest(),
    )[:MAX_DATASETS]


def _find_gap_closing_package() -> Path:
    override = os.environ.get("BIOHUB_GAP_CLOSING_DIR", "").strip()
    if override:
        candidate = Path(override).expanduser().resolve()
        if (candidate / "postprocess" / "last_call.py").is_file():
            return candidate
        raise FileNotFoundError(
            "BIOHUB_GAP_CLOSING_DIR is not a valid postprocess package: "
            + str(candidate)
        )
    matches = sorted(Path("/kaggle/input").rglob("postprocess/last_call.py"))
    if not matches:
        raise FileNotFoundError("legal gap-closing package was not attached")
    return matches[0].parent.parent


def _preflight_gap_closing_package() -> Path | None:
    """Resolve the optional postprocessor before starting expensive inference."""

    if not RUNNER_CONFIG["gap_closing"]:
        return None
    return _find_gap_closing_package()


def _apply_gap_closing(
    rows: list[dict[str, Any]],
    data_root: Path,
    package_root: Path | None = None,
) -> list[dict[str, Any]]:
    if not RUNNER_CONFIG["gap_closing"]:
        return rows
    package_root = package_root or _find_gap_closing_package()
    sys.path.insert(0, str(package_root))
    import pandas as pd
    from postprocess.last_call import GapClosingConfig, postprocess_submission

    frame = pd.DataFrame(rows, columns=COLUMNS)
    config = GapClosingConfig(
        max_total_distance_um=RUNNER_CONFIG["gap_closing_max_total_distance_um"],
        reuse_existing_um=RUNNER_CONFIG["gap_closing_reuse_existing_um"],
        max_added_fraction=RUNNER_CONFIG["gap_closing_max_added_fraction"],
        max_added_absolute=RUNNER_CONFIG["gap_closing_max_added_absolute"],
        synthetic_mode="intensity_centroid",
        reuse_existing=True,
        min_track_length=0,
    )
    clean, report = postprocess_submission(frame, data_root, config)
    print("GAP_CLOSING_REPORT:" + report.to_json(orient="records"), flush=True)
    return clean.to_dict("records")


def _rows_from_graph(graph: Any, dataset: str) -> list[dict[str, Any]]:
    nodes = list(graph.node_attrs().iter_rows(named=True))
    edges = list(graph.edge_attrs().iter_rows(named=True))
    node_ids = {int(row["node_id"]) for row in nodes}
    if not node_ids:
        raise ValueError(dataset + ": learned model produced no nodes")
    rows: list[dict[str, Any]] = []
    for row in sorted(nodes, key=lambda item: int(item["node_id"])):
        rows.append(
            {
                "id": -1,
                "dataset": dataset,
                "row_type": "node",
                "node_id": int(row["node_id"]),
                "t": int(row["t"]),
                "z": max(0, int(round(float(row["z"])))),
                "y": max(0, int(round(float(row["y"])))),
                "x": max(0, int(round(float(row["x"])))),
                "source_id": -1,
                "target_id": -1,
            }
        )
    for row in edges:
        source_id, target_id = int(row["source_id"]), int(row["target_id"])
        if source_id not in node_ids or target_id not in node_ids:
            raise ValueError(dataset + ": dangling edge")
        rows.append(
            {
                "id": -1,
                "dataset": dataset,
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": source_id,
                "target_id": target_id,
            }
        )
    return rows


def _normalise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key in ("id", "node_id", "t", "z", "y", "x", "source_id", "target_id"):
            value = item.get(key, -1)
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                raise ValueError("submission row contains an empty numeric value")
            item[key] = int(value)
        item["dataset"] = str(item["dataset"])
        item["row_type"] = str(item["row_type"])
        normalised.append(item)
    return normalised


def _validate_rows(
    rows: list[dict[str, Any]],
    dataset: str,
    frame_count: int,
    *,
    legal_gap_closing: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    ids = [int(row["id"]) for row in rows]
    if ids != list(range(len(rows))):
        errors.append("id values are not contiguous from zero")
    if any(row["dataset"] != dataset for row in rows):
        errors.append("unexpected dataset value")
    nodes = [row for row in rows if row["row_type"] == "node"]
    edges = [row for row in rows if row["row_type"] == "edge"]
    node_ids = {int(row["node_id"]) for row in nodes}
    time_by_id = {int(row["node_id"]): int(row["t"]) for row in nodes}
    times = sorted(set(time_by_id.values()))
    if not nodes:
        errors.append("no node rows")
    if times != list(range(frame_count)):
        errors.append("node time scope does not cover all image timepoints")
    dangling = 0
    nonadjacent = 0
    out_degree: dict[int, int] = {}
    for row in edges:
        source, target = int(row["source_id"]), int(row["target_id"])
        if source not in node_ids or target not in node_ids:
            dangling += 1
            continue
        out_degree[source] = out_degree.get(source, 0) + 1
        if time_by_id[target] != time_by_id[source] + 1:
            nonadjacent += 1
    if dangling:
        errors.append("dangling edges")
    if nonadjacent and not legal_gap_closing:
        errors.append("non-adjacent edges without legal gap-closing")
    max_out_degree = max(out_degree.values(), default=0)
    if max_out_degree > 2:
        errors.append("out-degree exceeds H9 limit of two")
    if nonadjacent:
        errors.append("H9 requires zero non-adjacent edges")
    return {
        "valid": not errors,
        "errors": errors,
        "rows": len(rows),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "timepoints": times,
        "division_parent_count": sum(value > 1 for value in out_degree.values()),
        "max_out_degree": max_out_degree,
        "nonadjacent_edge_count": nonadjacent,
        "dangling_edge_count": dangling,
        "legal_gap_closing": legal_gap_closing,
    }


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_submission_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """Write a combined CSV with row IDs contiguous across all datasets."""

    submission_rows = [dict(row) for row in rows]
    for row_id, row in enumerate(submission_rows):
        row["id"] = row_id
    _write_rows(submission_rows, path)


def _graph_from_rows(rows: list[dict[str, Any]], td: Any, pl: Any) -> Any:
    """Rebuild the same spatial graph represented by one candidate CSV."""

    node_rows = [row for row in rows if row["row_type"] == "node"]
    edge_rows = [row for row in rows if row["row_type"] == "edge"]
    graph = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        graph.add_node_attr_key(key, pl.Float64, -999999.0)
    assigned = graph.bulk_add_nodes(
        [
            {
                "t": int(row["t"]),
                "z": float(row["z"]),
                "y": float(row["y"]),
                "x": float(row["x"]),
            }
            for row in node_rows
        ]
    )
    id_map = {
        int(row["node_id"]): assigned[index]
        for index, row in enumerate(node_rows)
    }
    if edge_rows:
        graph.bulk_add_edges(
            [
                {
                    "source_id": id_map[int(row["source_id"])],
                    "target_id": id_map[int(row["target_id"])],
                }
                for row in edge_rows
            ]
        )
    return graph


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> None:
    cycle_started = time.monotonic()
    WORKING_DIR.mkdir(parents=True, exist_ok=True)
    gap_closing_package_root = _preflight_gap_closing_package()
    support_pack = _find_support_pack()
    _require_offline_dependencies(support_pack)
    competition_root = _resolve_competition_root()
    train_dir = competition_root / "train"
    runtime_repo = WORKING_DIR / "holdout_support_repo"
    if runtime_repo.exists():
        shutil.rmtree(runtime_repo)
    shutil.copytree(support_pack / "repo", runtime_repo)
    predictor_script = runtime_repo / "scripts" / "predict_unet_transformer.py"
    _patch_support_predictor(predictor_script)
    sys.path.insert(0, str(runtime_repo / "src"))
    sys.path.insert(0, str(runtime_repo / "scripts"))
    os.environ["USER"] = "biohub_kaggle_holdout"

    import polars as pl
    import torch
    import tracksdata as td
    import zarr
    from geff import GeffMetadata
    from predict_unet_transformer import PredictConfig, build_graph, load_model, predict_video
    from biohub_tracking.io import open_dataset, save_graph
    from biohub_tracking.metrics import (
        evaluate as compute_metric,
        node_recall,
        per_sample_metrics,
        summarise,
    )

    audits = _scope_audit(train_dir, td, open_dataset, GeffMetadata)
    selected = _select_datasets(audits)
    selected_names = [item["dataset"] for item in selected]
    selected_lookup = {item["dataset"]: item for item in selected}
    print("HOLDOUT_SCOPE_AUDIT:" + json.dumps(_json_safe(audits), sort_keys=True), flush=True)
    print("HOLDOUT_SELECTED:" + json.dumps(selected_names), flush=True)

    weights = support_pack / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    config = PredictConfig(
        det_threshold=RUNNER_CONFIG["det_threshold"],
        threshold=RUNNER_CONFIG["edge_threshold"],
        edge_min_threshold=RUNNER_CONFIG["edge_min_threshold"],
        edge_topk_parents=RUNNER_CONFIG["edge_topk_parents"],
        edge_topk_children=RUNNER_CONFIG["edge_topk_children"],
        edge_max_distance_um=RUNNER_CONFIG["edge_max_distance_um"],
        det_tta_mode=RUNNER_CONFIG["det_tta_mode"],
        feature_tta=RUNNER_CONFIG["feature_tta"],
        feature_tta_blend=RUNNER_CONFIG["feature_tta_blend"],
        pool_kernel_um=RUNNER_CONFIG["pool_kernel_um"],
        use_ilp=True,
        ilp_edge_weight=-1.0,
        ilp_appearance_weight=RUNNER_CONFIG["ilp_appearance_weight"],
        ilp_disappearance_weight=RUNNER_CONFIG["ilp_disappearance_weight"],
        ilp_division_weight=RUNNER_CONFIG["ilp_division_weight"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, window_size, downsample = load_model(weights, device)
    raw_by_dataset: dict[str, tuple[np.ndarray, list[tuple[int, int, float, float]]]] = {}
    deepcenter_peaks_by_dataset: dict[str, list[dict[str, Any]]] = {}
    deepcenter_inference_seconds = 0.0
    deepcenter_pack = None
    deepcenter_model = None
    deepcenter_config = None
    deepcenter_spatial_shape_by_dataset: dict[str, tuple[int, int, int]] = {}
    if DEEPCENTER_ENABLED:
        deepcenter_pack = _find_deepcenter_artifact()
        deepcenter_model, deepcenter_config, _deepcenter_source_module = load_deepcenter_model(
            deepcenter_pack / "weights" / "full_frame_center" / "best.pt",
            deepcenter_pack / "source_scripts" / "train_full_frame_center_detector.py",
            device=device,
        )
        deepcenter_started = time.monotonic()
        for name in selected_names:
            image_array = zarr.open_group(str(train_dir / (name + ".zarr")), mode="r")["0"]
            deepcenter_spatial_shape_by_dataset[name] = tuple(
                int(value) for value in image_array.shape[1:]
            )
            peaks: list[dict[str, Any]] = []
            for timepoint in range(int(selected_lookup[name]["frame_count"])):
                frame = np.asarray(image_array[timepoint])
                peaks.extend(
                    predict_center_peaks(
                        deepcenter_model,
                        frame,
                        config=deepcenter_config,
                        device=device,
                        threshold=float(min(DEEPCENTER_THRESHOLDS)),
                        min_distance=1,
                        time_index=timepoint,
                    )
                )
            deepcenter_peaks_by_dataset[name] = peaks
            print(
                "DEEPCENTER_INFERENCE:"
                + json.dumps({"dataset": name, "peaks": len(peaks)}),
                flush=True,
            )
        deepcenter_inference_seconds = time.monotonic() - deepcenter_started
    inference_started = time.monotonic()
    for name in selected_names:
        coords, raw_edges = predict_video(
            model,
            train_dir / (name + ".zarr"),
            device,
            cfg=config,
            window_size=window_size,
            unet_batch_size=4,
            downsample=downsample,
        )
        raw_by_dataset[name] = (coords, raw_edges)
        print(
            "HOLDOUT_INFERENCE:"
            + json.dumps({"dataset": name, "nodes": len(coords), "edges": len(raw_edges)}),
            flush=True,
        )
    inference_seconds = time.monotonic() - inference_started

    deepcenter_threshold_by_label: dict[str, float] = {}
    edge_radius_by_label: dict[str, float] = {}
    if EDGE_RADIUS_VARIANTS:
        radius_specs = _edge_radius_variant_specs(EDGE_RADIUS_VARIANTS)
        variant_specs = tuple(
            (
                label,
                float(RUNNER_CONFIG["edge_min_threshold"]),
                float(RUNNER_CONFIG["ilp_disappearance_weight"]),
            )
            for label, _radius in radius_specs
        )
        edge_radius_by_label = {
            label: radius for label, radius in radius_specs
        }
        division_weights = {
            label: float(RUNNER_CONFIG["ilp_division_weight"])
            for label, _radius in radius_specs
        }
    elif DEEPCENTER_ENABLED:
        center_specs = _center_variant_specs(DEEPCENTER_THRESHOLDS)
        variant_specs = tuple(
            (label, float(RUNNER_CONFIG["edge_min_threshold"]), float(RUNNER_CONFIG["ilp_disappearance_weight"]))
            for label, _threshold in center_specs
        )
        deepcenter_threshold_by_label = {
            label: threshold for label, threshold in center_specs
        }
        division_weights = {
            label: float(RUNNER_CONFIG["ilp_division_weight"])
            for label, _threshold in center_specs
        }
    elif DISAPPEARANCE_WEIGHTS:
        variant_specs = tuple(
            (label, float(RUNNER_CONFIG["edge_min_threshold"]), weight)
            for label, weight in _disappearance_variant_specs(DISAPPEARANCE_WEIGHTS)
        )
        division_weights = {
            label: float(RUNNER_CONFIG["ilp_division_weight"])
            for label, _weight in _disappearance_variant_specs(DISAPPEARANCE_WEIGHTS)
        }
    elif DIVISION_WEIGHTS:
        variant_specs = tuple(
            (label, float(RUNNER_CONFIG["edge_min_threshold"]), float(RUNNER_CONFIG["ilp_disappearance_weight"]))
            for label, _weight in _division_variant_specs(DIVISION_WEIGHTS)
        )
        division_weights = {
            label: weight for label, weight in _division_variant_specs(DIVISION_WEIGHTS)
        }
    elif EDGE_FLOORS:
        variant_specs = tuple(
            (label, floor, float(RUNNER_CONFIG["ilp_disappearance_weight"]))
            for label, floor in _edge_floor_variant_specs(EDGE_FLOORS)
        )
        division_weights = {
            label: float(RUNNER_CONFIG["ilp_division_weight"])
            for label, _floor in _edge_floor_variant_specs(EDGE_FLOORS)
        }
    else:
        variant_specs = (
            ("edge035", 0.35, float(RUNNER_CONFIG["ilp_disappearance_weight"])),
            ("edge045", 0.45, float(RUNNER_CONFIG["ilp_disappearance_weight"])),
            ("edge055", 0.55, float(RUNNER_CONFIG["ilp_disappearance_weight"])),
        )
        division_weights = {
            label: float(RUNNER_CONFIG["ilp_division_weight"])
            for label, _floor, _weight in variant_specs
        }
    variant_reports: dict[str, Any] = {}
    for label, floor, disappearance_weight in variant_specs:
        division_weight = division_weights[label]
        variant_started = time.monotonic()
        variant_dir = WORKING_DIR / ("predictions_" + label)
        if variant_dir.exists():
            shutil.rmtree(variant_dir)
        variant_dir.mkdir(parents=True, exist_ok=True)
        all_rows: list[dict[str, Any]] = []
        per_dataset: list[dict[str, Any]] = []
        motion_consensus_added_total = 0
        deepcenter_rescue_added_total = 0
        for name in selected_names:
            coords, raw_edges = raw_by_dataset[name]
            edge_radius = edge_radius_by_label.get(
                label,
                float(RUNNER_CONFIG["edge_max_distance_um"]),
            )
            filtered_edges = [
                edge
                for edge in raw_edges
                if float(edge[2]) >= float(floor)
                and float(edge[3]) <= edge_radius
            ]
            graph = build_graph(coords, filtered_edges)
            if config.use_ilp and graph.num_edges() > 0:
                solver = td.solvers.ILPSolver(
                    edge_weight=config.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                    appearance_weight=config.ilp_appearance_weight,
                    disappearance_weight=disappearance_weight,
                    division_weight=division_weight,
                )
                with open(os.devnull, "w") as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        graph = solver.solve(graph)
            save_graph(graph, variant_dir / (name + ".geff"))
            rows = _rows_from_graph(graph, name)
            rows = _normalise_rows(
                _apply_gap_closing(rows, train_dir, gap_closing_package_root)
            )
            deepcenter_rescue_report: dict[str, Any] = {
                "endpoint_count": 0,
                "ranked_candidate_count": 0,
                "accepted_candidate_count": 0,
                "added_nodes": 0,
                "added_edges": 0,
                "stage": "disabled",
            }
            deepcenter_threshold = deepcenter_threshold_by_label.get(label)
            if deepcenter_threshold is not None:
                center_peaks = [
                    peak
                    for peak in deepcenter_peaks_by_dataset[name]
                    if float(peak["score"]) >= float(deepcenter_threshold)
                ]
                rows, deepcenter_rescue_report = add_center_rescue_nodes(
                    rows,
                    center_peaks,
                    frame_count=int(selected_lookup[name]["frame_count"]),
                    spatial_shape=deepcenter_spatial_shape_by_dataset[name],
                    max_endpoint_distance_um=RUNNER_CONFIG["deepcenter_max_endpoint_distance_um"],
                    min_existing_distance_um=RUNNER_CONFIG["deepcenter_min_existing_distance_um"],
                    scale_um=open_dataset(train_dir / (name + ".zarr"), load_image=False).scale,
                    max_out_degree=RUNNER_CONFIG["deepcenter_max_out_degree"],
                    require_unique_parent=RUNNER_CONFIG["deepcenter_require_unique_parent"],
                )
                deepcenter_rescue_report["stage"] = "post_ilp_serialization"
                deepcenter_rescue_report["threshold"] = float(deepcenter_threshold)
                deepcenter_rescue_added_total += int(deepcenter_rescue_report["added_nodes"])
            motion_consensus_added = 0
            if RUNNER_CONFIG["motion_consensus"]:
                before_edges = sum(row["row_type"] == "edge" for row in rows)
                rows = add_motion_consensus_links(
                    rows,
                    max_distance_um=RUNNER_CONFIG["motion_consensus_max_distance_um"],
                    max_prediction_error_um=RUNNER_CONFIG["motion_consensus_max_prediction_error_um"],
                    require_unique_parent=RUNNER_CONFIG["motion_consensus_require_unique_parent"],
                    max_out_degree=RUNNER_CONFIG["motion_consensus_max_out_degree"],
                )
                motion_consensus_added = sum(row["row_type"] == "edge" for row in rows) - before_edges
                motion_consensus_added_total += motion_consensus_added
            for row_id, row in enumerate(rows):
                row["id"] = row_id
            all_rows.extend(rows)

            node_frame_count = int(selected_lookup[name]["frame_count"])
            validation = _validate_rows(
                rows,
                name,
                node_frame_count,
                legal_gap_closing=bool(RUNNER_CONFIG["gap_closing"]),
            )
            pred_graph = _graph_from_rows(rows, td, pl)
            gt_graph = _load_graph(train_dir / (name + ".geff"), td)
            scale = open_dataset(train_dir / (name + ".zarr"), load_image=False).scale
            er = compute_metric(pred_graph, gt_graph, scale=scale, max_distance=7.0)
            recall = (
                node_recall(pred_graph, gt_graph)
                if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0
                else 0.0
            )
            n_total = float(selected_lookup[name]["estimated_number_of_nodes"])
            metrics = per_sample_metrics(er, n_total, recall)
            per_dataset.append(
                {
                    "dataset": name,
                    "validation": validation,
                    "metrics": metrics,
                    "pred_nodes": pred_graph.num_nodes(),
                    "pred_edges": pred_graph.num_edges(),
                    "gt_nodes": gt_graph.num_nodes(),
                    "gt_edges": gt_graph.num_edges(),
                    "motion_consensus_added_edges": motion_consensus_added,
                    "deepcenter_rescue": deepcenter_rescue_report,
                }
            )
        output_path = WORKING_DIR / ("holdout_submission_" + label + ".csv")
        _write_submission_rows(all_rows, output_path)
        aggregate = summarise([item["metrics"] for item in per_dataset])
        variant_reports[label] = {
            "edge_floor": floor,
            "edge_max_distance_um": edge_radius_by_label.get(
                label,
                float(RUNNER_CONFIG["edge_max_distance_um"]),
            ),
            "ilp_disappearance_weight": disappearance_weight,
            "ilp_division_weight": division_weight,
            "motion_consensus_added_edges": motion_consensus_added_total,
            "motion_consensus_stage": (
                "post_ilp_serialization"
                if RUNNER_CONFIG["motion_consensus"]
                else "disabled"
            ),
            "deepcenter_threshold": deepcenter_threshold_by_label.get(label),
            "deepcenter_rescue_added_nodes": deepcenter_rescue_added_total,
            "deepcenter_stage": (
                "post_ilp_serialization"
                if deepcenter_threshold_by_label.get(label) is not None
                else "disabled"
            ),
            "submission": str(output_path),
            "prediction_dir": str(variant_dir),
            "per_dataset": per_dataset,
            "aggregate": aggregate,
            "timings_seconds": {
                "variant_total": round(time.monotonic() - variant_started, 3),
            },
        }
        print(
            "HOLDOUT_VARIANT:"
            + json.dumps(_json_safe({"label": label, "aggregate": aggregate}), sort_keys=True),
            flush=True,
        )

    metrics_path = WORKING_DIR / "holdout_metrics.json"
    report = {
        "status": "COMPLETE",
        "evaluator": "kaggle_holdout_eval",
        "competition": "biohub-cell-tracking-during-development",
        "scope_policy": {
            "complete_time_scope": True,
            "gt_is_sparse": True,
            "gt_timepoints_must_be_within_image": True,
            "metadata_scope_consistent": True,
            "expected_timepoints": EXPECTED_TIMEPOINTS,
            "max_datasets": MAX_DATASETS,
            "seed": SEED,
        },
        "competition_root": str(competition_root),
        "scope_audit": audits,
        "selected_datasets": selected_names,
        "config": RUNNER_CONFIG,
        "edge_radius_variants": list(EDGE_RADIUS_VARIANTS),
        "deepcenter": {
            "enabled": DEEPCENTER_ENABLED,
            "thresholds": list(DEEPCENTER_THRESHOLDS),
            "artifact_root": str(deepcenter_pack) if deepcenter_pack is not None else None,
            "checkpoint": (
                str(deepcenter_pack / "weights" / "full_frame_center" / "best.pt")
                if deepcenter_pack is not None
                else None
            ),
            "inference_config": (
                {
                    "pool_factor": int(deepcenter_config.pool_factor),
                    "base_channels": int(deepcenter_config.base_channels),
                    "norm_lo_pct": float(deepcenter_config.norm_lo_pct),
                    "norm_hi_pct": float(deepcenter_config.norm_hi_pct),
                    "norm_clip_lo": float(deepcenter_config.norm_clip_lo),
                    "norm_clip_hi": float(deepcenter_config.norm_clip_hi),
                }
                if deepcenter_config is not None
                else None
            ),
        },
        "device": str(device),
        "variants": variant_reports,
        "timings_seconds": {
            "inference_shared_seconds": round(inference_seconds, 3),
            "deepcenter_inference_seconds": round(deepcenter_inference_seconds, 3),
            "total_seconds": round(time.monotonic() - cycle_started, 3),
        },
    }
    metrics_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True), encoding="utf-8")
    print("HOLDOUT_METRICS:" + str(metrics_path), flush=True)
    print(json.dumps(_json_safe(report), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

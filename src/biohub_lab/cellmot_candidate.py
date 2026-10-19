"""Adapter for the public Cellmot UNet + transformer baseline.

The upstream predictor writes GEFF files into a username-based directory and
does not emit the Kaggle CSV contract.  This adapter gives each laboratory run
an isolated username, converts its GEFF output with the upstream converter,
validates ``submission.csv`` through the local control plane, and optionally
calls the upstream evaluator for in-domain/holdout scores.
"""

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .evaluation import evaluate_geff_directory
from .validation import validate_submission


def prediction_directory(
    baseline_root: Path,
    user: str,
    method: str,
    split: int,
) -> Path:
    """Mirror ``dataspec.PREDICTIONS_PATH / USERNAME / method / split_N``."""

    return Path(baseline_root) / "predictions" / user / method / "split_{}".format(int(split))


def build_predict_command(
    baseline_root: Path,
    python_executable: str,
    data_dir: Path,
    split: int,
    method: str,
    weights: Optional[Path] = None,
    det_threshold: Optional[float] = None,
    use_ilp: bool = False,
    debug_video: Optional[Path] = None,
    pool_kernel_um: Optional[float] = None,
    edge_threshold: Optional[float] = None,
    edge_min_threshold: Optional[float] = None,
    edge_topk_parents: Optional[int] = None,
    edge_topk_children: Optional[int] = None,
    edge_max_distance_um: Optional[float] = None,
    det_tta_mode: Optional[str] = None,
    ilp_appearance_weight: Optional[float] = None,
    ilp_disappearance_weight: Optional[float] = None,
    ilp_division_weight: Optional[float] = None,
) -> List[str]:
    """Build the upstream command without invoking a shell."""

    script = Path(baseline_root) / "scripts" / "predict_unet_transformer.py"
    command = [
        str(python_executable),
        str(script),
        "--data-dir",
        str(data_dir),
        "--method",
        str(method),
        "--split",
        str(int(split)),
    ]
    if weights is not None:
        command.extend(["--weights", str(weights)])
    if det_threshold is not None:
        command.extend(["--det-threshold", str(float(det_threshold))])
    if edge_threshold is not None:
        command.extend(["--edge-threshold", str(float(edge_threshold))])
    if edge_min_threshold is not None:
        command.extend(["--edge-min-threshold", str(float(edge_min_threshold))])
    if edge_topk_parents is not None:
        command.extend(["--edge-topk-parents", str(int(edge_topk_parents))])
    if edge_topk_children is not None:
        command.extend(["--edge-topk-children", str(int(edge_topk_children))])
    if edge_max_distance_um is not None:
        command.extend(["--edge-max-distance-um", str(float(edge_max_distance_um))])
    if det_tta_mode is not None:
        command.extend(["--det-tta-mode", str(det_tta_mode)])
    if pool_kernel_um is not None:
        command.extend(["--pool-kernel-um", str(float(pool_kernel_um))])
    if use_ilp:
        if ilp_appearance_weight is not None:
            command.extend(["--ilp-appearance-weight", str(float(ilp_appearance_weight))])
        if ilp_disappearance_weight is not None:
            command.extend(["--ilp-disappearance-weight", str(float(ilp_disappearance_weight))])
        if ilp_division_weight is not None:
            command.extend(["--ilp-division-weight", str(float(ilp_division_weight))])
        command.append("--use-ilp")
    if debug_video is not None:
        command.extend(["--debug-video", str(debug_video)])
    return command


def summarise_geff_directory(path: Path) -> Dict[str, int]:
    entries = sorted(Path(path).glob("*.geff"))
    return {"dataset_count": len(entries), "geff_count": len(entries)}


def _safe_user(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return result or "biohub_run"


def _copy_geff_entries(source_dir: Path, destination_dir: Path) -> List[Path]:
    source_entries = sorted(Path(source_dir).glob("*.geff"))
    if not source_entries:
        raise FileNotFoundError("upstream predictor produced no .geff files: {}".format(source_dir))
    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for source in source_entries:
        destination = destination_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        copied.append(destination)
    return copied


def _patch_predict_script_cli(runtime_root: Path) -> None:
    """Add compatibility CLI options to an older public predictor checkout."""

    script = Path(runtime_root) / "scripts" / "predict_unet_transformer.py"
    if not script.is_file():
        raise FileNotFoundError("predictor script not found under {}".format(runtime_root))
    source = script.read_text(encoding="utf-8")
    config_marker = "        det_threshold=args.det_threshold,\n"
    if config_marker not in source:
        raise RuntimeError("unsupported predictor CLI: PredictConfig marker not found")
    changed = False
    if "    edge_topk_parents: int = 0\n" not in source:
        field_marker = "    threshold: float = 0.5\n"
        if field_marker not in source:
            raise RuntimeError("unsupported predictor config: edge threshold field marker not found")
        source = source.replace(
            field_marker,
            field_marker
            + "    edge_min_threshold: float = 0.0\n"
            + "    edge_topk_parents: int = 0\n"
            + "    edge_topk_children: int = 0\n"
            + "    edge_max_distance_um: float = 0.0\n",
            1,
        )
        changed = True
    if "    det_tta_mode: str = \"flip4\"\n" not in source:
        field_marker = "    edge_max_distance_um: float = 0.0\n"
        if field_marker not in source:
            raise RuntimeError("unsupported predictor config: TTA field marker not found")
        source = source.replace(
            field_marker,
            field_marker + "    det_tta_mode: str = \"flip4\"\n",
            1,
        )
        changed = True
    if "--pool-kernel-um" not in source:
        help_marker = '''                             "threshold keeps precision up. Sweep it for your model.")\n'''
        if help_marker not in source:
            raise RuntimeError("unsupported predictor CLI: det-threshold help marker not found")
        source = source.replace(
            help_marker,
            help_marker
            + '''    parser.add_argument("--pool-kernel-um", type=float, default=3.0,\n'''
            + '''                        help="Physical max-pool kernel size for detection peaks (default 3.0).")\n''',
            1,
        )
        source = source.replace(
            config_marker,
            config_marker + "        pool_kernel_um=args.pool_kernel_um,\n",
            1,
        )
        changed = True

    if "--edge-threshold" not in source:
        ilp_marker = '''    parser.add_argument("--use-ilp", action="store_true",\n'''
        if ilp_marker not in source:
            raise RuntimeError("unsupported predictor CLI: use-ilp argument not found")
        source = source.replace(
            ilp_marker,
            '''    parser.add_argument("--edge-threshold", type=float, default=0.5,\n'''
            + '''                        help="Minimum edge probability to consider a link (default 0.5).")\n'''
            + ilp_marker,
            1,
        )
        source = source.replace(
            config_marker,
            config_marker + "        threshold=args.edge_threshold,\n",
            1,
        )
        changed = True

    if "--det-tta-mode" not in source:
        ilp_marker = '''    parser.add_argument("--use-ilp", action="store_true",\n'''
        if ilp_marker not in source:
            raise RuntimeError("unsupported predictor CLI: use-ilp argument not found")
        source = source.replace(
            ilp_marker,
            '''    parser.add_argument("--det-tta-mode", choices=("flip4", "dihedral8"), default="flip4",\n'''
            + '''                        help="Detection TTA group: flip4 or XY dihedral8 (default flip4).")\n'''
            + ilp_marker,
            1,
        )
        source = source.replace(
            config_marker,
            config_marker + "        det_tta_mode=args.det_tta_mode,\n",
            1,
        )
        changed = True

    if "--edge-topk-parents" not in source:
        ilp_marker = '''    parser.add_argument("--use-ilp", action="store_true",\n'''
        if ilp_marker not in source:
            raise RuntimeError("unsupported predictor CLI: use-ilp argument not found")
        source = source.replace(
            ilp_marker,
            '''    parser.add_argument("--edge-min-threshold", type=float, default=0.0,\n'''
            + '''                        help="Minimum probability for top-k edge rescue (default 0).")\n'''
            + '''    parser.add_argument("--edge-topk-parents", type=int, default=0,\n'''
            + '''                        help="Rescue this many best parents per target (default 0).")\n'''
            + '''    parser.add_argument("--edge-topk-children", type=int, default=0,\n'''
            + '''                        help="Rescue this many best children per source (default 0).")\n'''
            + '''    parser.add_argument("--edge-max-distance-um", type=float, default=0.0,\n'''
            + '''                        help="Reject links longer than this physical distance; 0 disables it.")\n'''
            + ilp_marker,
            1,
        )
        source = source.replace(
            config_marker,
            config_marker
            + "        edge_min_threshold=args.edge_min_threshold,\n"
            + "        edge_topk_parents=args.edge_topk_parents,\n"
            + "        edge_topk_children=args.edge_topk_children,\n"
            + "        edge_max_distance_um=args.edge_max_distance_um,\n",
            1,
        )
        candidate_marker = '''            candidates = sorted(\n'''
        candidate_block = '''            candidates = sorted(\n                [\n                    (probs[i, j], i, j)\n                    for i in range(n_src)\n                    for j in range(n_tgt)\n                    if probs[i, j] > cfg.threshold\n                ],\n                reverse=True,\n            )\n'''
        candidate_replacement = '''            candidate_pairs = {\n                (i, j)\n                for i in range(n_src)\n                for j in range(n_tgt)\n                if probs[i, j] > cfg.threshold\n            }\n            if cfg.edge_topk_parents > 0 and cfg.edge_min_threshold > 0.0:\n                for j in range(n_tgt):\n                    parents = np.argsort(probs[:, j])[::-1][:cfg.edge_topk_parents]\n                    for i in parents:\n                        if probs[i, j] >= cfg.edge_min_threshold:\n                            candidate_pairs.add((int(i), j))\n            if cfg.edge_topk_children > 0 and cfg.edge_min_threshold > 0.0:\n                for i in range(n_src):\n                    children = np.argsort(probs[i, :])[::-1][:cfg.edge_topk_children]\n                    for j in children:\n                        if probs[i, j] >= cfg.edge_min_threshold:\n                            candidate_pairs.add((i, int(j)))\n            candidates = sorted(\n                [(probs[i, j], i, j) for i, j in candidate_pairs],\n                reverse=True,\n            )\n'''
        if candidate_block not in source:
            raise RuntimeError("unsupported predictor graph code: edge candidate block not found")
        source = source.replace(candidate_block, candidate_replacement, 1)
        distance_marker = '''                dist = float(np.linalg.norm(\n                    coords_so_far[gi, 1:].astype(np.float32)\n                    - coords_so_far[gj, 1:].astype(np.float32)\n                ))\n'''
        distance_replacement = '''                dist = float(np.linalg.norm(\n                    (coords_so_far[gi, 1:].astype(np.float32)\n                     - coords_so_far[gj, 1:].astype(np.float32))\n                    * np.asarray(voxel_size, dtype=np.float32)\n                ))\n                if cfg.edge_max_distance_um > 0.0 and dist > cfg.edge_max_distance_um:\n                    continue\n'''
        if distance_marker not in source:
            raise RuntimeError("unsupported predictor graph code: edge distance block not found")
        source = source.replace(distance_marker, distance_replacement, 1)
        changed = True

    tta_block = '''        # Detection TTA: original + flip-x + flip-y + flip-xy, average logits.\n        # TTA: flip along Y (-2) and X (-1) only.  Z is excluded because\n        # the data is highly anisotropic (Z resolution ~4x coarser than XY),\n        # so Z-flips would produce out-of-distribution inputs.\n        if cfg.det_tta:\n            tta_flips = [(-1,), (-2,), (-2, -1)]\n            for dims in tta_flips:\n                imgs_flip = imgs.flip(dims)\n                _, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n                del imgs_flip, det_flip\n            for f in range(W):\n                det_logits[f] = det_logits[f] / 4\n'''
    tta_replacement = '''        # Detection TTA: use the public lead's XY dihedral group when requested.\n        if cfg.det_tta:\n            if cfg.det_tta_mode == "dihedral8":\n                tta_transforms = [\n                    (lambda value: value.flip((-1,)), lambda value: value.flip((-1,))),\n                    (lambda value: value.flip((-2,)), lambda value: value.flip((-2,))),\n                    (lambda value: value.flip((-2, -1)), lambda value: value.flip((-2, -1))),\n                    (lambda value: torch.rot90(value, 1, dims=(-2, -1)),\n                     lambda value: torch.rot90(value, -1, dims=(-2, -1))),\n                    (lambda value: torch.rot90(value, -1, dims=(-2, -1)),\n                     lambda value: torch.rot90(value, 1, dims=(-2, -1))),\n                    (lambda value: value.transpose(-1, -2),\n                     lambda value: value.transpose(-1, -2)),\n                    (lambda value: torch.rot90(value, 1, dims=(-2, -1)).transpose(-1, -2),\n                     lambda value: torch.rot90(value.transpose(-1, -2), -1, dims=(-2, -1))),\n                ]\n            else:\n                tta_transforms = [\n                    (lambda value: value.flip((-1,)), lambda value: value.flip((-1,))),\n                    (lambda value: value.flip((-2,)), lambda value: value.flip((-2,))),\n                    (lambda value: value.flip((-2, -1)), lambda value: value.flip((-2, -1))),\n                ]\n            for forward, inverse in tta_transforms:\n                imgs_flip = forward(imgs)\n                _, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + inverse(det_flip[f])\n                del imgs_flip, det_flip\n            for f in range(W):\n                det_logits[f] = det_logits[f] / (len(tta_transforms) + 1)\n'''
    if tta_block in source:
        source = source.replace(tta_block, tta_replacement, 1)
        changed = True

    if changed:
        script.write_text(source, encoding="utf-8")


def _execution_root(
    baseline_root: Path,
    support_pack_dir: Optional[Path],
    runtime_dir: Optional[Path] = None,
) -> Path:
    """Resolve the repository that owns the predictor being executed."""

    if support_pack_dir is None:
        return Path(baseline_root).resolve()
    pack = Path(support_pack_dir).resolve()
    candidate = pack / "repo"
    if candidate.is_dir():
        if runtime_dir is None:
            return candidate
        runtime_root = Path(runtime_dir) / "support_pack_repo"
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        shutil.copytree(candidate, runtime_root)
        _patch_predict_script_cli(runtime_root)
        return runtime_root
    if (pack / "scripts" / "predict_unet_transformer.py").is_file():
        if runtime_dir is None:
            return pack
        runtime_root = Path(runtime_dir) / "support_pack_repo"
        if runtime_root.exists():
            shutil.rmtree(runtime_root)
        shutil.copytree(pack, runtime_root)
        _patch_predict_script_cli(runtime_root)
        return runtime_root
    raise FileNotFoundError("support pack repo not found under {}".format(pack))


def _prepare_batch_data_dir(data_dir: Path, run_dir: Path, label: str) -> Path:
    """Create a cheap symlink view with the split manifest expected upstream."""

    data_dir = Path(data_dir).resolve()
    if (data_dir / "dataset_splits.json").is_file():
        return data_dir
    dataset_paths = sorted(data_dir.glob("*.zarr"))
    if not dataset_paths:
        raise FileNotFoundError("no .zarr datasets found in {}".format(data_dir))
    batch_dir = Path(run_dir) / (label + "_batch_data")
    if batch_dir.exists():
        shutil.rmtree(batch_dir)
    batch_dir.mkdir(parents=True, exist_ok=True)
    names: List[str] = []
    for source in dataset_paths:
        name = source.stem
        (batch_dir / (name + ".zarr")).symlink_to(source, target_is_directory=source.is_dir())
        names.append(name)
    (batch_dir / "dataset_splits.json").write_text(
        json.dumps([{"test": names}], sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return batch_dir


def _convert_geffs_locally(geff_dir: Path, csv_path: Path) -> Path:
    """Convert GEFF graphs without assuming the external baseline checkout."""

    try:
        import tracksdata as td  # type: ignore
    except ImportError as exc:
        raise RuntimeError("tracksdata is required to convert support-pack GEFF output") from exc

    columns = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row_id = 0
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for geff_path in sorted(Path(geff_dir).glob("*.geff")):
            loaded = td.graph.IndexedRXGraph.from_geff(geff_path)
            graph = loaded[0] if isinstance(loaded, tuple) else loaded
            nodes = list(graph.node_attrs().iter_rows(named=True))
            edges = list(graph.edge_attrs().iter_rows(named=True))
            node_ids = {int(row["node_id"]) for row in nodes}
            for row in sorted(nodes, key=lambda item: int(item["node_id"])):
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": geff_path.stem,
                        "row_type": "node",
                        "node_id": int(row["node_id"]),
                        "t": int(row["t"]),
                        "z": max(0, int(round(float(row["z"]))),),
                        "y": max(0, int(round(float(row["y"]))),),
                        "x": max(0, int(round(float(row["x"]))),),
                        "source_id": -1,
                        "target_id": -1,
                    }
                )
                row_id += 1
            for row in edges:
                source_id = int(row["source_id"])
                target_id = int(row["target_id"])
                if source_id not in node_ids or target_id not in node_ids:
                    raise ValueError("{} contains a dangling edge".format(geff_path.name))
                writer.writerow(
                    {
                        "id": row_id,
                        "dataset": geff_path.stem,
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
                row_id += 1
    return csv_path


def _run_command(command: Sequence[str], cwd: Path, env: Dict[str, str], stdout_path: Path, stderr_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            list(command),
            cwd=str(cwd),
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        detail = ""
        try:
            detail = stderr_path.read_text(encoding="utf-8")[-2000:]
        except OSError:
            pass
        raise RuntimeError(
            "upstream command failed with return code {}: {}".format(completed.returncode, detail.strip())
        )


def _run_prediction(
    baseline_root: Path,
    python_executable: str,
    data_dir: Path,
    run_dir: Path,
    label: str,
    split: int,
    method: str,
    weights: Optional[Path],
    det_threshold: Optional[float],
    use_ilp: bool,
    support_pack_dir: Optional[Path] = None,
    convert_csv: bool = True,
    pool_kernel_um: Optional[float] = None,
    edge_threshold: Optional[float] = None,
    edge_min_threshold: Optional[float] = None,
    edge_topk_parents: Optional[int] = None,
    edge_max_distance_um: Optional[float] = None,
    det_tta_mode: Optional[str] = None,
    ilp_appearance_weight: Optional[float] = None,
    ilp_disappearance_weight: Optional[float] = None,
    ilp_division_weight: Optional[float] = None,
) -> Dict[str, Any]:
    baseline_root = Path(baseline_root).resolve()
    run_dir = Path(run_dir).resolve()
    execution_root = _execution_root(baseline_root, support_pack_dir, runtime_dir=run_dir / label)
    geff_dir = run_dir / (label + "_geffs")
    data_dir = Path(data_dir).resolve()
    dataset_paths = sorted(data_dir.glob("*.zarr"))
    if not dataset_paths:
        raise FileNotFoundError("no .zarr datasets found in {}".format(data_dir))
    execution_data_dir = _prepare_batch_data_dir(data_dir, run_dir, label)
    batch_mode = (execution_data_dir / "dataset_splits.json").is_file()
    runs = [None] if batch_mode else dataset_paths
    for dataset_path in runs:
        suffix = dataset_path.stem if dataset_path is not None else "batch"
        run_user = _safe_user("biohub_{}_{}_{}".format(run_dir.name, label, suffix))
        env = dict(os.environ)
        env["USER"] = run_user
        env["USERNAME"] = run_user
        env["CELLMOT_DATA_DIR"] = str(data_dir)
        source_path = execution_root / "src"
        if source_path.is_dir():
            old_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = os.pathsep.join(
                part for part in (str(source_path), old_pythonpath) if part
            )
        command = build_predict_command(
            baseline_root=execution_root,
            python_executable=python_executable,
            data_dir=execution_data_dir,
            split=split,
            method=method,
            weights=weights,
            det_threshold=det_threshold,
            use_ilp=use_ilp,
            debug_video=dataset_path,
            pool_kernel_um=pool_kernel_um,
            edge_threshold=edge_threshold,
            edge_min_threshold=edge_min_threshold,
            edge_topk_parents=edge_topk_parents,
            edge_max_distance_um=edge_max_distance_um,
            det_tta_mode=det_tta_mode,
            ilp_appearance_weight=ilp_appearance_weight,
            ilp_disappearance_weight=ilp_disappearance_weight,
            ilp_division_weight=ilp_division_weight,
        )
        _run_command(
            command,
            cwd=execution_root,
            env=env,
            stdout_path=run_dir / (label + "_" + suffix + "_predict.stdout.log"),
            stderr_path=run_dir / (label + "_" + suffix + "_predict.stderr.log"),
        )
        generated_dir = prediction_directory(execution_root, run_user, method, split)
        _copy_geff_entries(generated_dir, geff_dir)
    expected_names = {path.stem for path in dataset_paths}
    actual_names = {path.stem for path in geff_dir.glob("*.geff")}
    if actual_names != expected_names:
        raise RuntimeError(
            "upstream predictor produced an incomplete dataset set: expected={}, actual={}".format(
                sorted(expected_names), sorted(actual_names)
            )
        )
    csv_path = run_dir / ("submission.csv" if label == "final" else label + "_submission.csv")
    report = None
    if convert_csv:
        converter = baseline_root / "scripts" / "geffs_to_csv.py"
        if converter.is_file():
            _run_command(
                [python_executable, str(converter), "--in-dir", str(geff_dir), "--csv", str(csv_path)],
                cwd=baseline_root,
                env=dict(os.environ),
                stdout_path=run_dir / (label + "_convert.stdout.log"),
                stderr_path=run_dir / (label + "_convert.stderr.log"),
            )
        else:
            _convert_geffs_locally(geff_dir, csv_path)
        report = validate_submission(csv_path, expected_datasets=expected_names)
        if not report.valid:
            raise ValueError("upstream CSV failed local validation: {}".format("; ".join(report.errors[:3])))
    return {
        "csv": csv_path,
        "geff_dir": geff_dir,
        "validation": report.as_dict() if report is not None else None,
        "candidate_metrics": summarise_geff_directory(geff_dir),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-python", default=sys.executable)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--method", default="baseline")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--det-threshold", type=float)
    parser.add_argument("--pool-kernel-um", type=float, default=3.0)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--edge-min-threshold", type=float, default=0.0)
    parser.add_argument("--edge-topk-parents", type=int, default=0)
    parser.add_argument("--edge-max-distance-um", type=float, default=0.0)
    parser.add_argument("--det-tta-mode", choices=("flip4", "dihedral8"), default="flip4")
    parser.add_argument("--ilp-appearance-weight", type=float, default=0.0)
    parser.add_argument("--ilp-disappearance-weight", type=float, default=1.4)
    parser.add_argument("--ilp-division-weight", type=float, default=1.0)
    parser.add_argument("--use-ilp", action="store_true")
    parser.add_argument(
        "--support-pack-dir",
        type=Path,
        help="Public support-pack directory containing repo/ and weights/.",
    )
    parser.add_argument("--validation-data-dir", type=Path)
    parser.add_argument("--evaluate-gt-dir", type=Path)
    parser.add_argument("--holdout-data-dir", type=Path)
    parser.add_argument("--holdout-gt-dir", type=Path)
    parser.add_argument("--max-eval-distance", type=float, default=7.0)
    return parser


def _run(args: argparse.Namespace) -> Dict[str, Any]:
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if args.support_pack_dir is not None and args.weights is None:
        args.weights = (
            args.support_pack_dir
            / "weights"
            / "unet_transformer"
            / "split_0"
            / "edge_predictor_best.pth"
        )
    final = _run_prediction(
        args.baseline_root,
        args.baseline_python,
        args.data_dir,
        args.run_dir,
        "final",
        args.split,
        args.method,
        args.weights,
        args.det_threshold,
        args.use_ilp,
        args.support_pack_dir,
        pool_kernel_um=args.pool_kernel_um if args.support_pack_dir is not None else None,
        edge_threshold=args.edge_threshold if args.support_pack_dir is not None else None,
        edge_min_threshold=args.edge_min_threshold if args.support_pack_dir is not None else None,
        edge_topk_parents=args.edge_topk_parents if args.support_pack_dir is not None else None,
        edge_max_distance_um=args.edge_max_distance_um if args.support_pack_dir is not None else None,
        det_tta_mode=args.det_tta_mode if args.support_pack_dir is not None else None,
        ilp_appearance_weight=args.ilp_appearance_weight if args.support_pack_dir is not None else None,
        ilp_disappearance_weight=args.ilp_disappearance_weight if args.support_pack_dir is not None else None,
        ilp_division_weight=args.ilp_division_weight if args.support_pack_dir is not None else None,
    )
    metrics: Dict[str, Any] = {
        "artifact": "submission.csv",
        "candidate_metrics": final["candidate_metrics"],
        "validation": final["validation"],
        "evaluation": {},
    }

    evaluation = final
    if args.validation_data_dir is not None:
        evaluation = _run_prediction(
            args.baseline_root,
            args.baseline_python,
            args.validation_data_dir,
            args.run_dir,
            "validation",
            args.split,
            args.method,
            args.weights,
            args.det_threshold,
            args.use_ilp,
            args.support_pack_dir,
            convert_csv=False,
            pool_kernel_um=args.pool_kernel_um if args.support_pack_dir is not None else None,
            edge_threshold=args.edge_threshold if args.support_pack_dir is not None else None,
            edge_min_threshold=args.edge_min_threshold if args.support_pack_dir is not None else None,
            edge_topk_parents=args.edge_topk_parents if args.support_pack_dir is not None else None,
            edge_max_distance_um=args.edge_max_distance_um if args.support_pack_dir is not None else None,
            det_tta_mode=args.det_tta_mode if args.support_pack_dir is not None else None,
            ilp_appearance_weight=args.ilp_appearance_weight if args.support_pack_dir is not None else None,
            ilp_disappearance_weight=args.ilp_disappearance_weight if args.support_pack_dir is not None else None,
            ilp_division_weight=args.ilp_division_weight if args.support_pack_dir is not None else None,
        )
    if args.evaluate_gt_dir is not None:
        evaluation_result = evaluate_geff_directory(
            evaluation["geff_dir"],
            args.evaluate_gt_dir,
            baseline_root=args.baseline_root,
            max_distance=args.max_eval_distance,
            expected_datasets={path.stem for path in Path(args.evaluate_gt_dir).glob("*.geff")},
            require_complete=True,
        )
        metrics["local_score"] = evaluation_result.get("score")
        metrics["in_domain_score"] = evaluation_result.get("score")
        metrics["evaluation"]["in_domain"] = evaluation_result

    if args.holdout_data_dir is not None:
        holdout = _run_prediction(
            args.baseline_root,
            args.baseline_python,
            args.holdout_data_dir,
            args.run_dir,
            "holdout",
            args.split,
            args.method,
            args.weights,
            args.det_threshold,
            args.use_ilp,
            args.support_pack_dir,
            convert_csv=False,
            pool_kernel_um=args.pool_kernel_um if args.support_pack_dir is not None else None,
            edge_threshold=args.edge_threshold if args.support_pack_dir is not None else None,
            edge_min_threshold=args.edge_min_threshold if args.support_pack_dir is not None else None,
            edge_topk_parents=args.edge_topk_parents if args.support_pack_dir is not None else None,
            edge_max_distance_um=args.edge_max_distance_um if args.support_pack_dir is not None else None,
            det_tta_mode=args.det_tta_mode if args.support_pack_dir is not None else None,
            ilp_appearance_weight=args.ilp_appearance_weight if args.support_pack_dir is not None else None,
            ilp_disappearance_weight=args.ilp_disappearance_weight if args.support_pack_dir is not None else None,
            ilp_division_weight=args.ilp_division_weight if args.support_pack_dir is not None else None,
        )
        holdout_gt = args.holdout_gt_dir or args.holdout_data_dir
        holdout_result = evaluate_geff_directory(
            holdout["geff_dir"],
            holdout_gt,
            baseline_root=args.baseline_root,
            max_distance=args.max_eval_distance,
            expected_datasets={path.stem for path in Path(holdout_gt).glob("*.geff")},
            require_complete=True,
        )
        metrics["holdout_score"] = holdout_result.get("score")
        metrics["evaluation"]["holdout"] = holdout_result

    with (args.run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True, allow_nan=True, default=str)
        handle.write("\n")
    return metrics


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    _run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

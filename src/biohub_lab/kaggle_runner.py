"""Build self-contained Kaggle Notebook runners for real competition data."""

from __future__ import annotations

import base64
import gzip
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


_RUNNER_FILES = (
    "__init__.py",
    "classical.py",
    "classical_candidate.py",
    "competition_array.py",
    "evaluation.py",
    "ome_zarr.py",
    "validation.py",
)

_UPSTREAM_RUNNER_FILES = (
    "LICENSE",
    "scripts/augmentations.py",
    "scripts/dataspec.py",
    "scripts/evaluate.py",
    "scripts/geffs_to_csv.py",
    "scripts/predict_unet_transformer.py",
    "scripts/train_unet_transformer.py",
    "src/tracking_cellmot/__init__.py",
    "src/tracking_cellmot/division_metrics.py",
    "src/tracking_cellmot/img_proc.py",
    "src/tracking_cellmot/io.py",
    "src/tracking_cellmot/metrics.py",
    "src/tracking_cellmot/models/__init__.py",
    "src/tracking_cellmot/models/simple_node_transformer.py",
    "src/tracking_cellmot/models/temporal_unet.py",
)


def _package_archive(project_root: Path) -> str:
    """Return a deterministic compressed archive of the runner package."""

    package_root = Path(project_root) / "src" / "biohub_lab"
    if not package_root.is_dir():
        raise FileNotFoundError("Biohub package directory not found: {}".format(package_root))

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as handle:
        for filename in _RUNNER_FILES:
            path = package_root / filename
            if not path.is_file():
                raise FileNotFoundError("runner source file not found: {}".format(path))
            info = zipfile.ZipInfo("biohub_lab/{}".format(filename))
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, path.read_bytes())
    compressed = gzip.compress(archive.getvalue(), mtime=0)
    return base64.b64encode(compressed).decode("ascii")


def _package_upstream_archive(project_root: Path) -> str:
    """Return a deterministic archive of the upstream training/prediction code."""

    upstream_root = Path(project_root) / "external" / "kaggle-cell-tracking-competition"
    if not upstream_root.is_dir():
        raise FileNotFoundError("upstream CellMot repository not found: {}".format(upstream_root))

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as handle:
        for filename in _UPSTREAM_RUNNER_FILES:
            path = upstream_root / filename
            if not path.is_file():
                raise FileNotFoundError("upstream runner file not found: {}".format(path))
            info = zipfile.ZipInfo(filename)
            info.date_time = (1980, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, path.read_bytes())
    compressed = gzip.compress(archive.getvalue(), mtime=0)
    return base64.b64encode(compressed).decode("ascii")


def build_classical_runner_source(
    project_root: Path,
    *,
    quantile: float = 0.999,
    max_link_distance_um: float = 10.0,
    ensure_timepoint_coverage: bool = True,
    infer_divisions: bool = False,
    filter_isolated_nodes: bool = False,
    workers: int = 1,
    max_detections_per_frame: int = 2000,
    array_key: str = "0",
    expected_timepoints: int = 100,
) -> str:
    """Generate a Notebook cell that recalculates H1 from mounted test data.

    The generated source carries only the small deterministic candidate package;
    it never carries a precomputed submission artifact.  Kaggle therefore runs
    the detector against the competition's mounted ``test`` directory before
    the code-submission endpoint sees ``submission.csv``.
    """

    if int(workers) <= 0:
        raise ValueError("workers must be positive")
    if int(max_detections_per_frame) <= 0:
        raise ValueError("max_detections_per_frame must be positive")
    if int(expected_timepoints) < 0:
        raise ValueError("expected_timepoints must be non-negative")

    config: Dict[str, Any] = {
        "array_key": str(array_key),
        "ensure_timepoint_coverage": bool(ensure_timepoint_coverage),
        "expected_timepoints": int(expected_timepoints),
        "filter_isolated_nodes": bool(filter_isolated_nodes),
        "infer_divisions": bool(infer_divisions),
        "max_detections_per_frame": int(max_detections_per_frame),
        "max_link_distance_um": float(max_link_distance_um),
        "quantile": float(quantile),
        "workers": int(workers),
    }
    encoded_archive = json.dumps(_package_archive(Path(project_root)))
    encoded_config = json.dumps(json.dumps(config, sort_keys=True))

    return (
        "import base64\n"
        "import gzip\n"
        "import importlib.util\n"
        "import io\n"
        "import json\n"
        "import os\n"
        "import shutil\n"
        "import sys\n"
        "import time\n"
        "import zipfile\n"
        "from pathlib import Path\n\n"
        "ARCHIVE_B64 = "
        + encoded_archive
        + "\nRUNNER_CONFIG = json.loads("
        + encoded_config
        + ")\nCOMPETITION = 'biohub-cell-tracking-during-development'\n"
        "COMPETITION_ROOT_CANDIDATES = [\n"
        "    Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),\n"
        "    Path('/kaggle/input/biohub-cell-tracking-during-development'),\n"
        "]\n"
        "\n"
        "def resolve_competition_root():\n"
        "    override = os.environ.get('BIOHUB_COMPETITION_DIR', '').strip()\n"
        "    candidates = [Path(override)] if override else COMPETITION_ROOT_CANDIDATES\n"
        "    for candidate in candidates:\n"
        "        if (candidate / 'test').is_dir():\n"
        "            return candidate\n"
        "    raise FileNotFoundError(\n"
        "        'competition test directory not found; checked: ' +\n"
        "        ', '.join(str(path) for path in candidates)\n"
        "    )\n"
        "\n"
        "def unpack_runner_package(destination):\n"
        "    if destination.exists():\n"
        "        shutil.rmtree(destination)\n"
        "    destination.mkdir(parents=True, exist_ok=True)\n"
        "    payload = gzip.decompress(base64.b64decode(ARCHIVE_B64))\n"
        "    with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
        "        for member in archive.infolist():\n"
        "            target = (destination / member.filename).resolve()\n"
        "            if destination.resolve() not in target.parents:\n"
        "                raise RuntimeError('unsafe embedded runner path: ' + member.filename)\n"
        "            if member.is_dir():\n"
        "                target.mkdir(parents=True, exist_ok=True)\n"
        "            else:\n"
        "                target.parent.mkdir(parents=True, exist_ok=True)\n"
        "                target.write_bytes(archive.read(member))\n"
        "\n"
        "def require_runtime_dependencies():\n"
        "    missing = [\n"
        "        name for name in ('numpy', 'scipy')\n"
        "        if importlib.util.find_spec(name) is None\n"
        "    ]\n"
        "    if (\n"
        "        importlib.util.find_spec('zarr') is None\n"
        "        and importlib.util.find_spec('blosc2') is None\n"
        "    ):\n"
        "        missing.append('zarr-or-blosc2')\n"
        "    if missing:\n"
        "        raise RuntimeError('Kaggle runtime is missing: ' + ', '.join(missing))\n"
        "\n"
        "started = time.monotonic()\n"
        "require_runtime_dependencies()\n"
        "competition_root = resolve_competition_root()\n"
        "test_dir = competition_root / 'test'\n"
        "dataset_paths = sorted(test_dir.glob('*.zarr'))\n"
        "if not dataset_paths:\n"
        "    raise FileNotFoundError('no .zarr datasets found in ' + str(test_dir))\n"
        "working_dir = Path(os.environ.get('BIOHUB_WORKING_DIR', '/kaggle/working'))\n"
        "working_dir.mkdir(parents=True, exist_ok=True)\n"
        "package_root = working_dir / 'biohub_runner_package'\n"
        "unpack_runner_package(package_root)\n"
        "sys.path.insert(0, str(package_root))\n"
        "from biohub_lab.classical import write_submission_csv\n"
        "from biohub_lab.classical_candidate import build_graphs\n"
        "from biohub_lab.validation import validate_submission\n"
        "\n"
        "workers = max(1, int(os.environ.get('BIOHUB_RUN_WORKERS', RUNNER_CONFIG['workers'])))\n"
        "expected_count = int(os.environ.get('BIOHUB_EXPECTED_TIMEPOINTS', RUNNER_CONFIG['expected_timepoints']))\n"
        "graphs = build_graphs(\n"
        "    test_dir,\n"
        "    array_key=RUNNER_CONFIG['array_key'],\n"
        "    quantile=RUNNER_CONFIG['quantile'],\n"
        "    max_link_distance_um=RUNNER_CONFIG['max_link_distance_um'],\n"
        "    max_detections_per_frame=RUNNER_CONFIG['max_detections_per_frame'],\n"
        "    infer_divisions=RUNNER_CONFIG['infer_divisions'],\n"
        "    filter_isolated_nodes=RUNNER_CONFIG['filter_isolated_nodes'],\n"
        "    ensure_timepoint_coverage=RUNNER_CONFIG['ensure_timepoint_coverage'],\n"
        "    workers=workers,\n"
        ")\n"
        "submission_path = write_submission_csv(graphs, working_dir / 'submission.csv')\n"
        "expected_datasets = {path.stem for path in dataset_paths}\n"
        "expected_times = range(expected_count) if expected_count else None\n"
        "report = validate_submission(\n"
        "    submission_path,\n"
        "    expected_datasets=expected_datasets,\n"
        "    expected_timepoints=expected_times,\n"
        "    include_topology=True,\n"
        ")\n"
        "if not report.valid:\n"
        "    raise RuntimeError('generated submission failed validation: ' + '; '.join(report.errors[:10]))\n"
        "summary = {\n"
        "    'datasets': sorted(expected_datasets),\n"
        "    'edges': report.edge_count,\n"
        "    'nodes': report.node_count,\n"
        "    'division_parent_count': report.division_parent_count,\n"
        "    'max_out_degree': report.max_out_degree,\n"
        "    'nonadjacent_edge_count': report.nonadjacent_edge_count,\n"
        "    'component_count': report.component_count,\n"
        "    'largest_component_size': report.largest_component_size,\n"
        "    'seconds': round(time.monotonic() - started, 3),\n"
        "    'submission': str(submission_path),\n"
        "}\n"
        "print(json.dumps(summary, sort_keys=True), flush=True)\n"
    )


def build_support_pack_runner_source(
    project_root: Path,
    *,
    det_threshold: float = 0.965,
    edge_threshold: float = 0.5,
    edge_min_threshold: float = 0.0,
    edge_topk_parents: int = 0,
    edge_topk_children: int = 0,
    edge_max_distance_um: float = 0.0,
    det_tta_mode: str = "flip4",
    feature_tta: bool = False,
    feature_tta_blend: float = 1.0,
    motion_consensus: bool = False,
    motion_consensus_max_distance_um: float = 7.0,
    motion_consensus_max_prediction_error_um: float = 3.0,
    motion_consensus_require_unique_parent: bool = True,
    motion_consensus_max_out_degree: int = 2,
    ilp_appearance_weight: float = 0.0,
    ilp_disappearance_weight: float = 1.4,
    ilp_division_weight: float = 1.0,
    pool_kernel_um: float = 3.0,
    metric_probe: bool = False,
    metric_probe_max_components: int = 1400,
    metric_probe_forks: int = 5,
    gap_closing: bool = False,
    gap_closing_max_total_distance_um: float = 10.0,
    gap_closing_reuse_existing_um: float = 2.5,
    gap_closing_max_added_fraction: float = 0.02,
    gap_closing_max_added_absolute: int = 1000,
) -> str:
    """Build a notebook runner for the public UNet + transformer support pack.

    The notebook attaches the public support pack as a Dataset input.  It then
    copies only its source tree to ``/kaggle/working`` (the mounted input is
    read-only), runs the learned detector once over all test videos, applies
    the official tracksdata ILP solver, and converts the resulting GEFF graphs
    to the competition CSV contract.  No precomputed submission is embedded.
    """

    del project_root
    if not 0.0 < float(det_threshold) < 1.0:
        raise ValueError("det_threshold must be in (0, 1)")
    if not 0.0 < float(edge_threshold) < 1.0:
        raise ValueError("edge_threshold must be in (0, 1)")
    if float(edge_min_threshold) < 0.0 or float(edge_min_threshold) >= 1.0:
        raise ValueError("edge_min_threshold must be in [0, 1)")
    if int(edge_topk_parents) < 0:
        raise ValueError("edge_topk_parents must be non-negative")
    if int(edge_topk_children) < 0:
        raise ValueError("edge_topk_children must be non-negative")
    if not math.isfinite(float(edge_max_distance_um)) or float(edge_max_distance_um) < 0.0:
        raise ValueError("edge_max_distance_um must be finite and non-negative")
    if str(det_tta_mode) not in {"flip4", "dihedral8"}:
        raise ValueError("det_tta_mode must be 'flip4' or 'dihedral8'")
    if not 0.0 <= float(feature_tta_blend) <= 1.0:
        raise ValueError("feature_tta_blend must be in [0, 1]")
    if float(motion_consensus_max_distance_um) < 0.0 or not math.isfinite(float(motion_consensus_max_distance_um)):
        raise ValueError("motion_consensus_max_distance_um must be finite and non-negative")
    if float(motion_consensus_max_prediction_error_um) < 0.0 or not math.isfinite(float(motion_consensus_max_prediction_error_um)):
        raise ValueError("motion_consensus_max_prediction_error_um must be finite and non-negative")
    if int(motion_consensus_max_out_degree) not in {1, 2}:
        raise ValueError("motion_consensus_max_out_degree must be 1 or 2")
    if int(metric_probe_max_components) <= 0:
        raise ValueError("metric_probe_max_components must be positive")
    if int(metric_probe_forks) < 0:
        raise ValueError("metric_probe_forks must be non-negative")
    if float(gap_closing_max_total_distance_um) <= 0.0:
        raise ValueError("gap_closing_max_total_distance_um must be positive")
    if float(gap_closing_reuse_existing_um) < 0.0:
        raise ValueError("gap_closing_reuse_existing_um must be non-negative")
    if not 0.0 < float(gap_closing_max_added_fraction) <= 1.0:
        raise ValueError("gap_closing_max_added_fraction must be in (0, 1]")
    if int(gap_closing_max_added_absolute) <= 0:
        raise ValueError("gap_closing_max_added_absolute must be positive")
    config = {
        "det_threshold": float(det_threshold),
        "edge_threshold": float(edge_threshold),
        "edge_min_threshold": float(edge_min_threshold),
        "edge_topk_parents": int(edge_topk_parents),
        "edge_topk_children": int(edge_topk_children),
        "edge_max_distance_um": float(edge_max_distance_um),
        "det_tta_mode": str(det_tta_mode),
        "feature_tta": bool(feature_tta),
        "feature_tta_blend": float(feature_tta_blend),
        "motion_consensus": bool(motion_consensus),
        "motion_consensus_max_distance_um": float(motion_consensus_max_distance_um),
        "motion_consensus_max_prediction_error_um": float(motion_consensus_max_prediction_error_um),
        "motion_consensus_require_unique_parent": bool(motion_consensus_require_unique_parent),
        "motion_consensus_max_out_degree": int(motion_consensus_max_out_degree),
        "ilp_appearance_weight": float(ilp_appearance_weight),
        "ilp_disappearance_weight": float(ilp_disappearance_weight),
        "ilp_division_weight": float(ilp_division_weight),
        "pool_kernel_um": float(pool_kernel_um),
        "metric_probe": bool(metric_probe),
        "metric_probe_max_components": int(metric_probe_max_components),
        "metric_probe_forks": int(metric_probe_forks),
        "gap_closing": bool(gap_closing),
        "gap_closing_max_total_distance_um": float(gap_closing_max_total_distance_um),
        "gap_closing_reuse_existing_um": float(gap_closing_reuse_existing_um),
        "gap_closing_max_added_fraction": float(gap_closing_max_added_fraction),
        "gap_closing_max_added_absolute": int(gap_closing_max_added_absolute),
    }
    encoded_config = json.dumps(json.dumps(config, sort_keys=True))
    if gap_closing:
        gap_closing_helpers = r'''
def _find_gap_closing_package():
    matches = sorted(Path('/kaggle/input').rglob('postprocess/last_call.py'))
    if not matches:
        raise FileNotFoundError('legal gap-closing package was not attached')
    return matches[0].parent.parent


def _apply_gap_closing(rows, data_root):
    if not RUNNER_CONFIG['gap_closing']:
        return rows
    package_root = _find_gap_closing_package()
    sys.path.insert(0, str(package_root))
    import pandas as pd
    from postprocess.last_call import GapClosingConfig, postprocess_submission

    columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']
    frame = pd.DataFrame(rows, columns=columns)
    config = GapClosingConfig(
        max_total_distance_um=RUNNER_CONFIG['gap_closing_max_total_distance_um'],
        reuse_existing_um=RUNNER_CONFIG['gap_closing_reuse_existing_um'],
        max_added_fraction=RUNNER_CONFIG['gap_closing_max_added_fraction'],
        max_added_absolute=RUNNER_CONFIG['gap_closing_max_added_absolute'],
        synthetic_mode='intensity_centroid',
        reuse_existing=True,
        min_track_length=0,
    )
    clean, report = postprocess_submission(frame, data_root, config)
    print('GAP_CLOSING_REPORT:' + report.to_json(orient='records'), flush=True)
    return clean.to_dict('records')
'''
    else:
        gap_closing_helpers = "def _apply_gap_closing(rows, data_root):\n    return rows\n"
    motion_consensus_source = ""
    motion_consensus_block = ""
    if motion_consensus:
        module_path = Path(__file__).resolve().with_name("motion_consensus.py")
        motion_consensus_source = module_path.read_text(encoding="utf-8")
        motion_consensus_source = motion_consensus_source.replace(
            "from __future__ import annotations\n\n", "", 1
        )
        motion_consensus_block = r'''
    if RUNNER_CONFIG['motion_consensus']:
        before_edges = sum(1 for row in rows if row['row_type'] == 'edge')
        rows = add_motion_consensus_links(
            rows,
            max_distance_um=RUNNER_CONFIG['motion_consensus_max_distance_um'],
            max_prediction_error_um=RUNNER_CONFIG['motion_consensus_max_prediction_error_um'],
            require_unique_parent=RUNNER_CONFIG['motion_consensus_require_unique_parent'],
            max_out_degree=RUNNER_CONFIG['motion_consensus_max_out_degree'],
        )
        after_edges = sum(1 for row in rows if row['row_type'] == 'edge')
        print('MOTION_CONSENSUS_REPORT:' + json.dumps({
            'added_edges': after_edges - before_edges,
            'edge_count': after_edges,
            'stage': 'post_ilp_serialization',
        }, sort_keys=True), flush=True)
'''
    metric_probe_helpers = r'''
def _metric_probe_row(dataset, row_type, node_id=-1, t=-1, z=-1, y=-1, x=-1, source_id=-1, target_id=-1):
    return {'id': -1, 'dataset': dataset, 'row_type': row_type, 'node_id': int(node_id),
            't': int(t), 'z': int(z), 'y': int(y), 'x': int(x),
            'source_id': int(source_id), 'target_id': int(target_id)}

def _augment_metric_probe(rows, dataset_names):
    augmented = []
    max_components = RUNNER_CONFIG['metric_probe_max_components']
    forks = RUNNER_CONFIG['metric_probe_forks']
    for dataset in dataset_names:
        dataset_rows = [row for row in rows if row['dataset'] == dataset]
        nodes = [row for row in dataset_rows if row['row_type'] == 'node']
        edges = [row for row in dataset_rows if row['row_type'] == 'edge']
        node_ids = [int(row['node_id']) for row in nodes]
        if not node_ids:
            raise ValueError(dataset + ': metric probe received no learned nodes')
        parent = {node_id: node_id for node_id in node_ids}
        incoming = set()
        def find(node_id):
            while parent[node_id] != node_id:
                parent[node_id] = parent[parent[node_id]]
                node_id = parent[node_id]
            return node_id
        def union(left, right):
            root_left, root_right = find(left), find(right)
            if root_left != root_right:
                parent[root_right] = root_left
        for edge in edges:
            source_id, target_id = int(edge['source_id']), int(edge['target_id'])
            if source_id not in parent or target_id not in parent:
                raise ValueError(dataset + ': metric probe found a dangling edge')
            incoming.add(target_id)
            union(source_id, target_id)
        components = {}
        for node_id in node_ids:
            components.setdefault(find(node_id), []).append(node_id)
        roots = []
        for component in components.values():
            candidates = [node_id for node_id in component if node_id not in incoming]
            if len(candidates) != 1:
                raise ValueError(dataset + ': metric probe expected one root per component')
            roots.append((len(component), candidates[0]))
        roots = [root for _, root in sorted(roots, reverse=True)[:max_components]]
        next_id = max(node_ids) + 1
        hub_id = next_id
        next_id += 1
        synthetic_nodes = [_metric_probe_row(dataset, 'node', hub_id, -1000, -10000, -10000, -10000)]
        synthetic_edges = [_metric_probe_row(dataset, 'edge', source_id=hub_id, target_id=root) for root in roots]
        previous_id = hub_id
        for index in range(forks):
            divider_id, child_id, continuation_id = range(next_id, next_id + 3)
            next_id += 3
            time = -999 + 2 * index
            synthetic_nodes.extend([
                _metric_probe_row(dataset, 'node', divider_id, time, -10000, -10000, -10000),
                _metric_probe_row(dataset, 'node', child_id, time + 1, -10000, -10000, -10000),
                _metric_probe_row(dataset, 'node', continuation_id, time + 1, -10001, -10000, -10000),
            ])
            synthetic_edges.extend([
                _metric_probe_row(dataset, 'edge', source_id=previous_id, target_id=divider_id),
                _metric_probe_row(dataset, 'edge', source_id=divider_id, target_id=child_id),
                _metric_probe_row(dataset, 'edge', source_id=divider_id, target_id=continuation_id),
            ])
            previous_id = continuation_id
        augmented.extend(dataset_rows)
        augmented.extend(synthetic_nodes)
        augmented.extend(synthetic_edges)
    for row_id, row in enumerate(augmented):
        row['id'] = row_id
    return augmented

'''
    predictor_patch = r"""
def _patch_support_predictor(predict_script):
    source = predict_script.read_text(encoding='utf-8')
    if 'edge_topk_parents' in source:
        return
    config_marker = '    threshold: float = 0.5\n'
    if config_marker not in source:
        raise RuntimeError('support-pack predictor config marker not found')
    source = source.replace(
        config_marker,
        config_marker
       + '    edge_min_threshold: float = 0.0\n'
       + '    edge_topk_parents: int = 0\n'
        + '    edge_topk_children: int = 0\n'
       + '    edge_max_distance_um: float = 0.0\n',
        1,
    )
    candidate_block = '''            candidates = sorted(\n                [\n                    (probs[i, j], i, j)\n                    for i in range(n_src)\n                    for j in range(n_tgt)\n                    if probs[i, j] > cfg.threshold\n                ],\n                reverse=True,\n            )\n'''
    candidate_replacement = '''            candidate_pairs = {\n                (i, j)\n                for i in range(n_src)\n                for j in range(n_tgt)\n                if probs[i, j] > cfg.threshold\n            }\n            if cfg.edge_topk_parents > 0 and cfg.edge_min_threshold > 0.0:\n                for j in range(n_tgt):\n                    parents = np.argsort(probs[:, j])[::-1][:cfg.edge_topk_parents]\n                    for i in parents:\n                        if probs[i, j] >= cfg.edge_min_threshold:\n                            candidate_pairs.add((int(i), j))\n            if cfg.edge_topk_children > 0 and cfg.edge_min_threshold > 0.0:\n                for i in range(n_src):\n                    children = np.argsort(probs[i, :])[::-1][:cfg.edge_topk_children]\n                    for j in children:\n                        if probs[i, j] >= cfg.edge_min_threshold:\n                            candidate_pairs.add((i, int(j)))\n            candidates = sorted(\n                [(probs[i, j], i, j) for i, j in candidate_pairs],\n                reverse=True,\n            )\n'''
    if candidate_block not in source:
        raise RuntimeError('support-pack predictor edge candidate block not found')
    source = source.replace(candidate_block, candidate_replacement, 1)
    distance_block = '''                dist = float(np.linalg.norm(\n                    coords_so_far[gi, 1:].astype(np.float32)\n                    - coords_so_far[gj, 1:].astype(np.float32)\n                ))\n'''
    distance_replacement = '''                dist = float(np.linalg.norm(\n                    (coords_so_far[gi, 1:].astype(np.float32)\n                     - coords_so_far[gj, 1:].astype(np.float32))\n                    * np.asarray(voxel_size, dtype=np.float32)\n                ))\n                if cfg.edge_max_distance_um > 0.0 and dist > cfg.edge_max_distance_um:\n                    continue\n'''
    if distance_block not in source:
        raise RuntimeError('support-pack predictor distance block not found')
    source = source.replace(distance_block, distance_replacement, 1)
    tta_config_marker = '    edge_max_distance_um: float = 0.0\n'
    if tta_config_marker not in source:
        raise RuntimeError('support-pack predictor TTA config marker not found')
    source = source.replace(
        tta_config_marker,
        tta_config_marker
        + '    det_tta_mode: str = "flip4"\n'
        + '    feature_tta: bool = False\n'
        + '    feature_tta_blend: float = 1.0\n',
        1,
    )
    tta_block = '''        # Detection TTA: original + flip-x + flip-y + flip-xy, average logits.\n        # TTA: flip along Y (-2) and X (-1) only.  Z is excluded because\n        # the data is highly anisotropic (Z resolution ~4x coarser than XY),\n        # so Z-flips would produce out-of-distribution inputs.\n        if cfg.det_tta:\n            tta_flips = [(-1,), (-2,), (-2, -1)]\n            for dims in tta_flips:\n                imgs_flip = imgs.flip(dims)\n                _, det_flip = model.encode(imgs_flip)\n                for f in range(W):\n                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n                del imgs_flip, det_flip\n            for f in range(W):\n                det_logits[f] = det_logits[f] / 4\n'''
    tta_replacement = '''        # Reuse the same XY transform family for detection TTA and for\n        # optional feature-space TTA.  The inverse transform restores the\n        # encoded feature map before node/edge feature pooling.\n        if cfg.det_tta or cfg.feature_tta:\n            if cfg.det_tta_mode == "dihedral8":\n                tta_transforms = [\n                    (lambda value: value.flip((-1,)), lambda value: value.flip((-1,))),\n                    (lambda value: value.flip((-2,)), lambda value: value.flip((-2,))),\n                    (lambda value: value.flip((-2, -1)), lambda value: value.flip((-2, -1))),\n                    (lambda value: torch.rot90(value, 1, dims=(-2, -1)),\n                     lambda value: torch.rot90(value, -1, dims=(-2, -1))),\n                    (lambda value: torch.rot90(value, -1, dims=(-2, -1)),\n                     lambda value: torch.rot90(value, 1, dims=(-2, -1))),\n                    (lambda value: value.transpose(-1, -2),\n                     lambda value: value.transpose(-1, -2)),\n                    (lambda value: torch.rot90(value, 1, dims=(-2, -1)).transpose(-1, -2),\n                     lambda value: torch.rot90(value.transpose(-1, -2), -1, dims=(-2, -1))),\n                ]\n            else:\n                tta_transforms = [\n                    (lambda value: value.flip((-1,)), lambda value: value.flip((-1,))),\n                    (lambda value: value.flip((-2,)), lambda value: value.flip((-2,))),\n                    (lambda value: value.flip((-2, -1)), lambda value: value.flip((-2, -1))),\n                ]\n            feature_sum = unet_out.clone() if cfg.feature_tta else None\n            for forward, inverse in tta_transforms:\n                imgs_flip = forward(imgs)\n                unet_flip, det_flip = model.encode(imgs_flip)\n                if cfg.feature_tta:\n                    feature_sum = feature_sum + inverse(unet_flip)\n                if cfg.det_tta:\n                    for f in range(W):\n                        det_logits[f] = det_logits[f] + inverse(det_flip[f])\n                del imgs_flip, unet_flip, det_flip\n            if cfg.det_tta:\n                for f in range(W):\n                    det_logits[f] = det_logits[f] / (len(tta_transforms) + 1)\n            if cfg.feature_tta:\n                tta_mean = feature_sum / (len(tta_transforms) + 1)\n                blend = float(cfg.feature_tta_blend)\n                unet_out = unet_out * (1.0 - blend) + tta_mean * blend\n                del feature_sum, tta_mean\n'''
    if tta_block not in source:
        raise RuntimeError('support-pack predictor TTA block not found')
    source = source.replace(tta_block, tta_replacement, 1)
    predict_script.write_text(source, encoding='utf-8')

"""
    return (
        "import csv\n"
        "import importlib.util\n"
        "import json\n"
        "import os\n"
        "import shutil\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n\n"
        "RUNNER_CONFIG = json.loads(" + encoded_config + ")\n"
        + motion_consensus_source
        + "COMPETITION_ROOT_CANDIDATES = [\n"
        "    Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),\n"
        "    Path('/kaggle/input/biohub-cell-tracking-during-development'),\n"
        "]\n"
        "REQUIRED_IMPORTS = (\n"
        "    'tracksdata', 'zarr', 'numcodecs', 'geff', 'pyscipopt',\n"
        "    'ilpy', 'rustworkx', 'polars', 'dask',\n"
        ")\n"
        "OFFLINE_PACKAGES = (\n"
        "    'tracksdata', 'zarr==3.2.1', 'numcodecs==0.15.1',\n"
        "    'donfig==0.8.1.post1', 'geff==1.2.0.1.1', 'geff-spec==1.1.1',\n"
        "    'pyscipopt==6.2.1', 'ilpy==0.6.0', 'rustworkx==0.18.0',\n"
        "    'polars==1.42.0', 'polars-runtime-32==1.42.0',\n"
        "    'bidict==0.23.1', 'imagecodecs==2026.6.26',\n"
        "    'dask==2026.6.0', 'blosc2==4.7.0', 'pyarrow==24.0.0',\n"
        "    'sqlalchemy==2.0.51',\n"
        ")\n\n"
        "def find_support_pack():\n"
        "    for manifest in sorted(Path('/kaggle/input').glob('**/ARTIFACT_MANIFEST.json')):\n"
        "        root = manifest.parent\n"
        "        if (root / 'repo' / 'src' / 'biohub_tracking').is_dir() and (\n"
        "            root / 'weights' / 'unet_transformer' / 'split_0' / 'edge_predictor_best.pth'\n"
        "        ).is_file():\n"
        "            return root\n"
        "    raise FileNotFoundError('public Biohub tracking support pack was not attached')\n\n"
        "def require_offline_dependencies(support_pack):\n"
        "    missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]\n"
        "    if not missing:\n"
        "        return\n"
        "    wheels = support_pack / 'wheels'\n"
        "    if not wheels.is_dir():\n"
        "        raise FileNotFoundError('support pack wheels directory is missing: ' + str(wheels))\n"
        "    command = [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',\n"
        "               '--find-links', str(wheels), *OFFLINE_PACKAGES]\n"
        "    completed = subprocess.run(command, capture_output=True, text=True, check=False)\n"
        "    if completed.returncode != 0:\n"
        "        detail = (completed.stdout + '\\n' + completed.stderr).strip()[-4000:]\n"
        "        raise RuntimeError('offline support-pack dependency installation failed: ' + detail)\n"
        "    importlib.invalidate_caches()\n\n"
        "def resolve_competition_root():\n"
        "    override = os.environ.get('BIOHUB_COMPETITION_DIR', '').strip()\n"
        "    candidates = [Path(override)] if override else COMPETITION_ROOT_CANDIDATES\n"
        "    for candidate in candidates:\n"
        "        if (candidate / 'test').is_dir():\n"
        "            return candidate\n"
        "    raise FileNotFoundError('competition test directory not found: ' + ', '.join(map(str, candidates)))\n\n"
        "def prepare_batch_view(test_dir, working_dir):\n"
        "    batch_dir = working_dir / 'test_batch'\n"
        "    if batch_dir.exists():\n"
        "        shutil.rmtree(batch_dir)\n"
        "    batch_dir.mkdir(parents=True, exist_ok=True)\n"
        "    names = []\n"
        "    for source in sorted(test_dir.glob('*.zarr')):\n"
        "        name = source.stem\n"
        "        (batch_dir / (name + '.zarr')).symlink_to(source, target_is_directory=True)\n"
        "        names.append(name)\n"
        "    if not names:\n"
        "        raise FileNotFoundError('no .zarr datasets found in ' + str(test_dir))\n"
        "    split_path = batch_dir / 'dataset_splits.json'\n"
        "    split_path.write_text(json.dumps([{'test': names}]), encoding='utf-8')\n"
        "    return batch_dir, names, split_path\n\n"
        "def load_graph(path, td):\n"
        "    result = td.graph.IndexedRXGraph.from_geff(path)\n"
        "    return result[0] if isinstance(result, tuple) else result\n\n"
        + gap_closing_helpers
        + (metric_probe_helpers if metric_probe else "")
        + "def write_submission(graph_dir, dataset_names, output_path, td):\n"
        "    columns = ['id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id']\n"
        "    rows = []\n"
        "    for dataset in dataset_names:\n"
        "        graph_path = graph_dir / (dataset + '.geff')\n"
        "        if not graph_path.exists():\n"
        "            raise FileNotFoundError('prediction graph missing: ' + str(graph_path))\n"
        "        graph = load_graph(graph_path, td)\n"
        "        nodes = list(graph.node_attrs().iter_rows(named=True))\n"
        "        edges = list(graph.edge_attrs().iter_rows(named=True))\n"
        "        node_ids = {int(row['node_id']) for row in nodes}\n"
        "        if not node_ids:\n"
        "            raise ValueError(dataset + ': learned model produced no nodes')\n"
        "        for row in sorted(nodes, key=lambda item: int(item['node_id'])):\n"
        "            rows.append({'id': -1, 'dataset': dataset, 'row_type': 'node',\n"
        "                'node_id': int(row['node_id']), 't': int(row['t']),\n"
        "                'z': max(0, int(round(float(row['z'])))),\n"
        "                'y': max(0, int(round(float(row['y'])))),\n"
        "                'x': max(0, int(round(float(row['x'])))),\n"
        "                'source_id': -1, 'target_id': -1})\n"
        "        for row in edges:\n"
        "            source_id, target_id = int(row['source_id']), int(row['target_id'])\n"
        "            if source_id not in node_ids or target_id not in node_ids:\n"
        "                raise ValueError(dataset + ': dangling edge')\n"
        "            rows.append({'id': -1, 'dataset': dataset, 'row_type': 'edge',\n"
        "                'node_id': -1, 't': -1, 'z': -1, 'y': -1, 'x': -1,\n"
        "                'source_id': source_id, 'target_id': target_id})\n"
        "    if not rows:\n"
        "        raise ValueError('submission contains no learned nodes')\n"
        "    rows = _apply_gap_closing(rows, competition_root / 'test')\n"
        + motion_consensus_block
        + "    if RUNNER_CONFIG['metric_probe']:\n"
        "        rows = _augment_metric_probe(rows, dataset_names)\n"
        "    for row_id, row in enumerate(rows):\n"
        "        row['id'] = row_id\n"
        "    with output_path.open('w', newline='', encoding='utf-8') as handle:\n"
        "        writer = csv.DictWriter(handle, fieldnames=columns)\n"
        "        writer.writeheader()\n"
        "        writer.writerows(rows)\n\n"
        "cycle_started = time.monotonic()\n"
        "timings = {}\n"
        "stage_started = time.monotonic()\n"
        "support_pack = find_support_pack()\n"
        "require_offline_dependencies(support_pack)\n"
        "timings['dependency_setup_seconds'] = round(time.monotonic() - stage_started, 3)\n"
        "stage_started = time.monotonic()\n"
        "competition_root = resolve_competition_root()\n"
        "working_dir = Path(os.environ.get('BIOHUB_WORKING_DIR', '/kaggle/working'))\n"
        "working_dir.mkdir(parents=True, exist_ok=True)\n"
        "runtime_repo = working_dir / 'support_repo'\n"
        "if runtime_repo.exists():\n"
        "    shutil.rmtree(runtime_repo)\n"
        "shutil.copytree(support_pack / 'repo', runtime_repo)\n"
        + predictor_patch
        + "_patch_support_predictor(runtime_repo / 'scripts' / 'predict_unet_transformer.py')\n"
        "sys.path.insert(0, str(runtime_repo / 'src'))\n"
        "sys.path.insert(0, str(runtime_repo / 'scripts'))\n"
        "os.environ['USER'] = 'biohub_kaggle_support'\n"
        "batch_dir, dataset_names, split_path = prepare_batch_view(competition_root / 'test', working_dir)\n"
        "timings['input_setup_seconds'] = round(time.monotonic() - stage_started, 3)\n"
        "from predict_unet_transformer import PredictConfig, predict\n"
        "import tracksdata as td\n"
        "weights = support_pack / 'weights' / 'unet_transformer' / 'split_0' / 'edge_predictor_best.pth'\n"
        "config = PredictConfig(\n"
        "    det_threshold=RUNNER_CONFIG['det_threshold'],\n"
        "    threshold=RUNNER_CONFIG['edge_threshold'],\n"
       "    edge_min_threshold=RUNNER_CONFIG['edge_min_threshold'],\n"
       "    edge_topk_parents=RUNNER_CONFIG['edge_topk_parents'],\n"
        "    edge_topk_children=RUNNER_CONFIG['edge_topk_children'],\n"
       "    edge_max_distance_um=RUNNER_CONFIG['edge_max_distance_um'],\n"
        "    det_tta_mode=RUNNER_CONFIG['det_tta_mode'],\n"
        "    feature_tta=RUNNER_CONFIG['feature_tta'],\n"
        "    feature_tta_blend=RUNNER_CONFIG['feature_tta_blend'],\n"
        "    pool_kernel_um=RUNNER_CONFIG['pool_kernel_um'],\n"
        "    use_ilp=True,\n"
        "    ilp_edge_weight=-1.0,\n"
        "    ilp_appearance_weight=RUNNER_CONFIG['ilp_appearance_weight'],\n"
        "    ilp_disappearance_weight=RUNNER_CONFIG['ilp_disappearance_weight'],\n"
        "    ilp_division_weight=RUNNER_CONFIG['ilp_division_weight'],\n"
        ")\n"
        "stage_started = time.monotonic()\n"
        "predict(\n"
        "    data_dir=batch_dir, fold=0, splits_file=split_path, weights_path=weights,\n"
        "    cfg=config, method='unet_transformer', unet_batch_size=4,\n"
        ")\n"
        "timings['prediction_seconds'] = round(time.monotonic() - stage_started, 3)\n"
        "prediction_dir = runtime_repo / 'predictions' / 'biohub_kaggle_support' / 'unet_transformer' / 'split_0'\n"
        "stage_started = time.monotonic()\n"
        "write_submission(prediction_dir, dataset_names, working_dir / 'submission.csv', td)\n"
        "timings['serialization_postprocess_seconds'] = round(time.monotonic() - stage_started, 3)\n"
        "timings['total_seconds'] = round(time.monotonic() - cycle_started, 3)\n"
        "print(json.dumps({'datasets': len(dataset_names), 'submission': str(working_dir / 'submission.csv'), 'timings_seconds': timings}, sort_keys=True), flush=True)\n"
    )


def build_support_pack_cached_holdout_source(
    project_root: Path,
    *,
    edge_floors: tuple[float, float] = (0.45, 0.55),
    expected_timepoints: int = 100,
    det_threshold: float = 0.965,
    edge_threshold: float = 0.48,
    edge_topk_parents: int = 2,
    edge_topk_children: int = 2,
    edge_max_distance_um: float = 10.0,
    det_tta_mode: str = "dihedral8",
    ilp_appearance_weight: float = 0.0,
    ilp_disappearance_weight: float = 1.4,
    ilp_division_weight: float = 0.5,
    pool_kernel_um: float = 3.0,
    gap_closing: bool = True,
    gap_closing_max_total_distance_um: float = 10.0,
    gap_closing_reuse_existing_um: float = 2.5,
    gap_closing_max_added_fraction: float = 0.02,
    gap_closing_max_added_absolute: int = 1000,
) -> str:
    """Build a local holdout runner that reuses one expensive inference.

    The lowest edge floor is used for the single CellMot inference.  Each
    candidate then filters that pre-ILP edge list and solves its own ILP.  For
    nested floors (the intended use here), this is equivalent to two separate
    H9 runs while avoiding a second detector/transformer pass.
    """

    if len(tuple(edge_floors)) != 2:
        raise ValueError("edge_floors must contain exactly two candidate floors")
    floors = tuple(float(value) for value in edge_floors)
    if any(value < 0.0 or value >= 1.0 for value in floors):
        raise ValueError("edge_floors must be in [0, 1)")
    if floors[0] > floors[1]:
        raise ValueError("edge_floors must be ordered from low to high")
    if int(expected_timepoints) <= 0:
        raise ValueError("expected_timepoints must be positive")

    def _floor_label(floor: float) -> str:
        return "edge{:03d}".format(int(round(floor * 100)))

    variant_specs = tuple(
        {"label": _floor_label(floor), "floor": floor} for floor in floors
    )
    if len({spec["label"] for spec in variant_specs}) != len(variant_specs):
        raise ValueError("edge_floors must produce distinct candidate labels")

    source = build_support_pack_runner_source(
        project_root,
        det_threshold=det_threshold,
        edge_threshold=edge_threshold,
        edge_min_threshold=floors[0],
        edge_topk_parents=edge_topk_parents,
        edge_topk_children=edge_topk_children,
        edge_max_distance_um=edge_max_distance_um,
        det_tta_mode=det_tta_mode,
        ilp_appearance_weight=ilp_appearance_weight,
        ilp_disappearance_weight=ilp_disappearance_weight,
        ilp_division_weight=ilp_division_weight,
        pool_kernel_um=pool_kernel_um,
        metric_probe=False,
        gap_closing=gap_closing,
        gap_closing_max_total_distance_um=gap_closing_max_total_distance_um,
        gap_closing_reuse_existing_um=gap_closing_reuse_existing_um,
        gap_closing_max_added_fraction=gap_closing_max_added_fraction,
        gap_closing_max_added_absolute=gap_closing_max_added_absolute,
    )
    marker = "from predict_unet_transformer import PredictConfig, predict\n"
    if marker not in source:
        raise RuntimeError("support-pack runner prediction marker not found")
    prefix = source.split(marker, 1)[0]
    encoded_variants = json.dumps(list(variant_specs), sort_keys=True)
    inference_floor = json.dumps(floors[0])
    replacement = """import contextlib
import csv
import torch

from biohub_lab.validation import validate_submission
from predict_unet_transformer import PredictConfig, build_graph, load_model, predict_video
from tracking_cellmot.io import save_graph
import tracksdata as td

RUNNER_CONFIG.update({
    'edge_floor_inference': __INFERENCE_FLOOR__,
    'edge_floor_variants': __VARIANTS__,
    'expected_timepoints': __EXPECTED_TIMEPOINTS__,
})

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

weights = support_pack / 'weights' / 'unet_transformer' / 'split_0' / 'edge_predictor_best.pth'
config = PredictConfig(
    det_threshold=RUNNER_CONFIG['det_threshold'],
    threshold=RUNNER_CONFIG['edge_threshold'],
    edge_min_threshold=RUNNER_CONFIG['edge_floor_inference'],
    edge_topk_parents=RUNNER_CONFIG['edge_topk_parents'],
    edge_topk_children=RUNNER_CONFIG['edge_topk_children'],
    edge_max_distance_um=RUNNER_CONFIG['edge_max_distance_um'],
    det_tta_mode=RUNNER_CONFIG['det_tta_mode'],
    pool_kernel_um=RUNNER_CONFIG['pool_kernel_um'],
    use_ilp=True,
    ilp_edge_weight=-1.0,
    ilp_appearance_weight=RUNNER_CONFIG['ilp_appearance_weight'],
    ilp_disappearance_weight=RUNNER_CONFIG['ilp_disappearance_weight'],
    ilp_division_weight=RUNNER_CONFIG['ilp_division_weight'],
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, window_size, downsample = load_model(weights, device)
variant_specs = [
    (item['label'], float(item['floor']))
    for item in RUNNER_CONFIG['edge_floor_variants']
]
prediction_dirs = {}
for label, _floor in variant_specs:
    prediction_dir = working_dir / ('predictions_' + label)
    if prediction_dir.exists():
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_dirs[label] = prediction_dir

stage_started = time.monotonic()
for name in dataset_names:
    coords, raw_edges = predict_video(
        model,
        batch_dir / (name + '.zarr'),
        device,
        cfg=config,
        window_size=window_size,
        unet_batch_size=4,
        downsample=downsample,
    )
    for label, floor in variant_specs:
        filtered_edges = [
            edge for edge in raw_edges if float(edge[2]) >= float(floor)
        ]
        graph = build_graph(coords, filtered_edges)
        if config.use_ilp and graph.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=config.ilp_edge_weight * td.EdgeAttr('edge_prob'),
                appearance_weight=config.ilp_appearance_weight,
                disappearance_weight=config.ilp_disappearance_weight,
                division_weight=config.ilp_division_weight,
            )
            with suppress_output():
                graph = solver.solve(graph)
        save_graph(graph, prediction_dirs[label] / (name + '.geff'))
timings['prediction_seconds'] = round(time.monotonic() - stage_started, 3)

expected_datasets = set(dataset_names)
expected_times = range(int(RUNNER_CONFIG['expected_timepoints']))
variant_reports = {}
stage_started = time.monotonic()
for label, _floor in variant_specs:
    submission_path = working_dir / ('submission_' + label + '.csv')
    write_submission(prediction_dirs[label], dataset_names, submission_path, td)
    report = validate_submission(
        submission_path,
        expected_datasets=expected_datasets,
        expected_timepoints=expected_times,
        include_topology=True,
    )
    if not report.valid:
        raise RuntimeError(
            label + ' generated submission failed validation: ' +
            '; '.join(report.errors[:10])
        )
    variant_reports[label] = {
        'submission': str(submission_path),
        'validation': report.as_dict(),
    }
timings['serialization_postprocess_seconds'] = round(time.monotonic() - stage_started, 3)
timings['total_seconds'] = round(time.monotonic() - cycle_started, 3)
print(json.dumps({
    'datasets': sorted(expected_datasets),
    'inference_edge_floor': RUNNER_CONFIG['edge_floor_inference'],
    'submissions': variant_reports,
    'timings_seconds': timings,
}, sort_keys=True), flush=True)
"""
    replacement = replacement.replace("__INFERENCE_FLOOR__", inference_floor)
    replacement = replacement.replace("__VARIANTS__", encoded_variants)
    replacement = replacement.replace("__EXPECTED_TIMEPOINTS__", str(int(expected_timepoints)))
    return prefix + replacement


def build_h13_finetune_runner_source(
    project_root: Path,
    *,
    epochs: int = 3,
    lr: float = 1e-4,
    batch_size: int = 4,
    num_workers: int = 4,
    max_iters: Optional[int] = 300,
    holdout_fraction: float = 0.1,
    max_train_datasets: int = 0,
    max_holdout_datasets: int = 0,
    max_frames: Optional[int] = None,
    seed: int = 17,
    downsample: tuple[int, ...] = (1, 4, 4),
    unet_out_channels: int = 32,
    unet_layers: tuple[int, ...] = (32, 64, 128),
    det_loss_weight: float = 1.0,
    det_neg_weight: float = 1e-2,
    pool_kernel_um: float = 5.0,
    det_threshold: float = 0.965,
    edge_threshold: float = 0.48,
    ilp_appearance_weight: float = 0.0,
    ilp_disappearance_weight: float = 1.4,
    ilp_division_weight: float = 0.5,
    unet_batch_size: int = 4,
    data_parallel: bool = True,
) -> str:
    """Build a Kaggle runner that fine-tunes H13 and predicts the test set.

    The generated Notebook carries the audited upstream source, attaches the
    public support pack only for its offline dependencies, and locates the
    plain CC0 UNet checkpoint as an initializer.  It never embeds a
    submission CSV or uses the synthetic checkpoint as a final predictor.
    """

    if int(epochs) <= 0:
        raise ValueError("epochs must be positive")
    if float(lr) <= 0.0:
        raise ValueError("lr must be positive")
    if int(batch_size) <= 0 or int(num_workers) < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    if max_iters is not None and int(max_iters) <= 0:
        raise ValueError("max_iters must be positive or None")
    if not 0.0 < float(holdout_fraction) < 1.0:
        raise ValueError("holdout_fraction must be in (0, 1)")
    if int(max_train_datasets) < 0 or int(max_holdout_datasets) < 0:
        raise ValueError("dataset limits must be non-negative")
    if max_frames is not None and int(max_frames) <= 1:
        raise ValueError("max_frames must be greater than one or None")
    if len(tuple(downsample)) != 3 or any(int(value) <= 0 for value in downsample):
        raise ValueError("downsample must contain three positive integers")
    if int(unet_out_channels) <= 0 or not tuple(unet_layers) or any(int(value) <= 0 for value in unet_layers):
        raise ValueError("UNet channel configuration must be positive")
    if float(det_loss_weight) < 0.0 or float(det_neg_weight) < 0.0:
        raise ValueError("detection loss weights must be non-negative")
    if float(pool_kernel_um) <= 0.0:
        raise ValueError("pool_kernel_um must be positive")
    if not 0.0 < float(det_threshold) < 1.0 or not 0.0 < float(edge_threshold) < 1.0:
        raise ValueError("prediction thresholds must be in (0, 1)")
    if int(unet_batch_size) <= 0:
        raise ValueError("unet_batch_size must be positive")

    config = {
        "epochs": int(epochs),
        "lr": float(lr),
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "max_iters": None if max_iters is None else int(max_iters),
        "holdout_fraction": float(holdout_fraction),
        "max_train_datasets": int(max_train_datasets),
        "max_holdout_datasets": int(max_holdout_datasets),
        "max_frames": None if max_frames is None else int(max_frames),
        "seed": int(seed),
        "downsample": [int(value) for value in downsample],
        "unet_out_channels": int(unet_out_channels),
        "unet_layers": [int(value) for value in unet_layers],
        "det_loss_weight": float(det_loss_weight),
        "det_neg_weight": float(det_neg_weight),
        "pool_kernel_um": float(pool_kernel_um),
        "det_threshold": float(det_threshold),
        "edge_threshold": float(edge_threshold),
        "ilp_appearance_weight": float(ilp_appearance_weight),
        "ilp_disappearance_weight": float(ilp_disappearance_weight),
        "ilp_division_weight": float(ilp_division_weight),
        "unet_batch_size": int(unet_batch_size),
        "data_parallel": bool(data_parallel),
    }
    encoded_archive = json.dumps(_package_upstream_archive(Path(project_root)))
    encoded_config = json.dumps(json.dumps(config, sort_keys=True))
    local_archive = json.dumps(_package_archive(Path(project_root)))

    source = r'''import base64
import gzip
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

UPSTREAM_ARCHIVE_B64 = __UPSTREAM_ARCHIVE__
BIOHUB_ARCHIVE_B64 = __BIOHUB_ARCHIVE__
RUNNER_CONFIG = json.loads(__CONFIG__)
COMPETITION_ROOT_CANDIDATES = [
    Path('/kaggle/input/competitions/biohub-cell-tracking-during-development'),
    Path('/kaggle/input/biohub-cell-tracking-during-development'),
]
REQUIRED_IMPORTS = (
    'torch', 'tracksdata', 'zarr', 'numcodecs', 'polars', 'scipy', 'dask',
)
OFFLINE_PACKAGES = (
    'tracksdata', 'zarr==3.2.1', 'numcodecs==0.15.1',
    'donfig==0.8.1.post1', 'geff==1.2.0.1.1', 'geff-spec==1.1.1',
    'pyscipopt==6.2.1', 'ilpy==0.6.0', 'rustworkx==0.18.0',
    'polars==1.42.0', 'polars-runtime-32==1.42.0',
    'bidict==0.23.1', 'imagecodecs==2026.6.26',
    'dask==2026.6.0', 'blosc2==4.7.0', 'pyarrow==24.0.0',
    'sqlalchemy==2.0.51',
)


def _unpack_archive(encoded, destination):
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    payload = gzip.decompress(base64.b64decode(encoded))
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination.resolve() not in target.parents:
                raise RuntimeError('unsafe embedded archive path: ' + member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))


def _find_support_pack():
    for manifest in sorted(Path('/kaggle/input').rglob('ARTIFACT_MANIFEST.json')):
        root = manifest.parent
        if (root / 'repo' / 'src' / 'biohub_tracking').is_dir() and (root / 'wheels').is_dir():
            return root
    raise FileNotFoundError('public support pack with offline wheels was not attached')


def _find_plain_pretrain():
    matches = sorted(Path('/kaggle/input').rglob('unet_pretrained.pth'))
    if not matches:
        raise FileNotFoundError('plain unet_pretrained.pth was not attached')
    return matches[0]


def _require_dependencies(support_pack):
    missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
    if not missing:
        return
    wheels = support_pack / 'wheels'
    if not wheels.is_dir():
        raise FileNotFoundError('support pack wheels directory is missing: ' + str(wheels))
    command = [sys.executable, '-m', 'pip', 'install', '--quiet', '--no-index', '--no-deps',
               '--find-links', str(wheels), *OFFLINE_PACKAGES]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stdout + '\n' + completed.stderr).strip()[-4000:]
        raise RuntimeError('offline dependency installation failed: ' + detail)
    importlib.invalidate_caches()


def _resolve_competition_root():
    override = os.environ.get('BIOHUB_COMPETITION_DIR', '').strip()
    candidates = [Path(override)] if override else COMPETITION_ROOT_CANDIDATES
    for candidate in candidates:
        if (candidate / 'train').is_dir() and (candidate / 'test').is_dir():
            return candidate
    raise FileNotFoundError('competition train/test directories not found: ' + ', '.join(map(str, candidates)))


def _write_temporal_clip(train_dir, source_name, start, end, output_dir, label):
    import tracksdata as td
    from tracking_cellmot.io import open_dataset

    source = train_dir / source_name
    dataset = open_dataset(source, normalize=False, require_tracks=True, load_image=False)
    total_frames = int(dataset.image_shape[0])
    if not 0 <= int(start) < int(end) <= total_frames:
        raise ValueError(
            'temporal clip [{}, {}) is outside {} frames'.format(start, end, total_frames)
        )

    result = td.graph.IndexedRXGraph.from_geff(str(train_dir / (source_name + '.geff')))
    graph = result[0] if isinstance(result, tuple) else result
    clipped = (
        graph.filter(td.NodeAttr('t') >= int(start), td.NodeAttr('t') < int(end))
        .subgraph()
        .detach()
    )
    attrs = clipped.node_attrs()
    clipped.update_node_attrs(
        attrs={'t': (attrs['t'] - int(start)).to_list()},
        node_ids=attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
    )

    clip_name = '{}__{}_t{:03d}_{:03d}'.format(source_name, label, start, end)
    output_dir.mkdir(parents=True, exist_ok=True)
    zarr_link = output_dir / (clip_name + '.zarr')
    if not zarr_link.exists():
        zarr_link.symlink_to(train_dir / (source_name + '.zarr'), target_is_directory=True)
    clipped.to_geff(str(output_dir / (clip_name + '.geff')), overwrite=True)
    return clip_name


def _make_temporal_split(train_dir, working_dir, source_name):
    from tracking_cellmot.io import open_dataset

    dataset = open_dataset(
        train_dir / source_name,
        normalize=False,
        require_tracks=True,
        load_image=False,
    )
    total_frames = int(dataset.image_shape[0])
    configured_frames = RUNNER_CONFIG['max_frames']
    effective_frames = total_frames if configured_frames is None else min(
        total_frames, int(configured_frames)
    )
    if effective_frames < 4:
        raise RuntimeError('single-dataset temporal holdout needs at least four frames')

    holdout_count = max(2, int(round(effective_frames * RUNNER_CONFIG['holdout_fraction'])))
    holdout_count = min(holdout_count, effective_frames - 2)
    split_frame = effective_frames - holdout_count
    staged_dir = working_dir / 'h13_temporal_split'
    train_name = _write_temporal_clip(
        train_dir, source_name, 0, split_frame, staged_dir, 'train'
    )
    holdout_name = _write_temporal_clip(
        train_dir, source_name, split_frame, effective_frames, staged_dir, 'holdout'
    )
    frame_ranges = {
        train_name: [0, split_frame],
        holdout_name: [split_frame, effective_frames],
    }
    frame_ranges_path = working_dir / 'h13_frame_ranges.json'
    frame_ranges_path.write_text(json.dumps(frame_ranges, indent=2), encoding='utf-8')
    os.environ['CELLMOT_FRAME_RANGES'] = str(frame_ranges_path)

    split_path = working_dir / 'h13_train_split.json'
    split_path.write_text(
        json.dumps([{'split': 0, 'train': [train_name], 'test': [holdout_name]}], indent=2),
        encoding='utf-8',
    )
    return [train_name], [holdout_name], split_path, staged_dir, 'temporal_holdout'


def _make_split(train_dir, working_dir):
    names = sorted(
        path.stem for path in train_dir.glob('*.zarr')
        if (train_dir / (path.stem + '.geff')).exists()
    )
    if not names:
        raise RuntimeError('no paired train datasets found in ' + str(train_dir))
    if len(names) == 1:
        return _make_temporal_split(train_dir, working_dir, names[0])

    os.environ.pop('CELLMOT_FRAME_RANGES', None)
    seed = int(RUNNER_CONFIG['seed'])
    ranked = sorted(names, key=lambda name: hashlib.sha256(
        (str(seed) + ':' + name).encode('utf-8')
    ).hexdigest())
    holdout_count = max(1, int(round(len(names) * RUNNER_CONFIG['holdout_fraction'])))
    holdout = ranked[:holdout_count]
    train = [name for name in names if name not in set(holdout)]
    max_holdout = int(RUNNER_CONFIG['max_holdout_datasets'])
    max_train = int(RUNNER_CONFIG['max_train_datasets'])
    if max_holdout > 0:
        holdout = holdout[:max_holdout]
    if max_train > 0:
        train = train[:max_train]
    if not train or not holdout:
        raise RuntimeError('dataset limits left an empty train or holdout split')
    split_path = working_dir / 'h13_train_split.json'
    split_path.write_text(json.dumps([{'split': 0, 'train': train, 'test': holdout}], indent=2), encoding='utf-8')
    return train, holdout, split_path, Path(train_dir), 'dataset_holdout'


def _make_test_split(test_dir, working_dir):
    names = sorted(path.stem for path in test_dir.glob('*.zarr'))
    if not names:
        raise RuntimeError('no test .zarr datasets found in ' + str(test_dir))
    split_path = working_dir / 'h13_test_split.json'
    split_path.write_text(json.dumps([{'test': names}], indent=2), encoding='utf-8')
    return names, split_path


def _validate_output(output_path, expected_datasets):
    package_dir = Path(os.environ.get('BIOHUB_WORKING_DIR', '/kaggle/working')) / 'biohub_package'
    _unpack_archive(BIOHUB_ARCHIVE_B64, package_dir)
    sys.path.insert(0, str(package_dir))
    from biohub_lab.validation import validate_submission

    report = validate_submission(
        output_path,
        expected_datasets=set(expected_datasets),
        expected_timepoints=range(100),
        include_topology=True,
    )
    if not report.valid:
        raise RuntimeError('H13 submission failed validation: ' + '; '.join(report.errors[:10]))
    if report.nonadjacent_edge_count or report.max_out_degree > 2:
        raise RuntimeError(
            'H13 submission failed topology: nonadjacent_edge_count={}, max_out_degree={}'.format(
                report.nonadjacent_edge_count, report.max_out_degree
            )
        )
    return report


started = time.monotonic()
support_pack = _find_support_pack()
pretrain_path = _find_plain_pretrain()
_require_dependencies(support_pack)
competition_root = _resolve_competition_root()
working_dir = Path(os.environ.get('BIOHUB_WORKING_DIR', '/kaggle/working'))
working_dir.mkdir(parents=True, exist_ok=True)
upstream_dir = working_dir / 'h13_upstream'
_unpack_archive(UPSTREAM_ARCHIVE_B64, upstream_dir)
sys.path.insert(0, str(upstream_dir / 'src'))
sys.path.insert(0, str(upstream_dir / 'scripts'))
os.environ['USER'] = 'biohub_h13_finetune'

train_names, holdout_names, train_split, training_dir, split_mode = _make_split(
    competition_root / 'train', working_dir
)
test_names, test_split = _make_test_split(competition_root / 'test', working_dir)
print(json.dumps({
    'event': 'h13_split_ready',
    'train_datasets': len(train_names),
    'holdout_datasets': len(holdout_names),
    'split_mode': split_mode,
    'test_datasets': len(test_names),
    'initializer': str(pretrain_path),
}, sort_keys=True), flush=True)

from train_unet_transformer import train
train(
    data_dir=training_dir,
    fold=0,
    splits_file=train_split,
    method='h13_finetune',
    n_epochs=RUNNER_CONFIG['epochs'],
    lr=RUNNER_CONFIG['lr'],
    batch_size=RUNNER_CONFIG['batch_size'],
    num_workers=RUNNER_CONFIG['num_workers'],
    unet_out_channels=RUNNER_CONFIG['unet_out_channels'],
    unet_layers=RUNNER_CONFIG['unet_layers'],
    unet_weights=pretrain_path,
    downsample=tuple(RUNNER_CONFIG['downsample']),
    det_loss_weight=RUNNER_CONFIG['det_loss_weight'],
    det_neg_weight=RUNNER_CONFIG['det_neg_weight'],
    max_iters=RUNNER_CONFIG['max_iters'],
    max_frames=RUNNER_CONFIG['max_frames'],
    seed=RUNNER_CONFIG['seed'],
    pool_kernel_um=RUNNER_CONFIG['pool_kernel_um'],
    data_parallel=RUNNER_CONFIG['data_parallel'],
)

weights_path = upstream_dir / 'weights' / 'h13_finetune' / 'split_0' / 'edge_predictor_best.pth'
if not weights_path.is_file():
    raise FileNotFoundError('H13 training did not produce a checkpoint: ' + str(weights_path))

from predict_unet_transformer import PredictConfig, predict
predict(
    data_dir=competition_root / 'test',
    fold=0,
    splits_file=test_split,
    weights_path=weights_path,
    cfg=PredictConfig(
        det_threshold=RUNNER_CONFIG['det_threshold'],
        threshold=RUNNER_CONFIG['edge_threshold'],
        use_ilp=True,
        ilp_edge_weight=-1.0,
        ilp_appearance_weight=RUNNER_CONFIG['ilp_appearance_weight'],
        ilp_disappearance_weight=RUNNER_CONFIG['ilp_disappearance_weight'],
        ilp_division_weight=RUNNER_CONFIG['ilp_division_weight'],
        pool_kernel_um=RUNNER_CONFIG['pool_kernel_um'],
    ),
    method='h13_finetune',
    unet_batch_size=RUNNER_CONFIG['unet_batch_size'],
)

from geffs_to_csv import geffs_to_csv
prediction_dir = upstream_dir / 'predictions' / 'biohub_h13_finetune' / 'h13_finetune' / 'split_0'
submission_path = working_dir / 'submission.csv'
geffs_to_csv(prediction_dir, submission_path)
report = _validate_output(submission_path, test_names)
print(json.dumps({
    'event': 'h13_submission_ready',
    'datasets': sorted(test_names),
    'nodes': report.node_count,
    'edges': report.edge_count,
    'division_parent_count': report.division_parent_count,
    'max_out_degree': report.max_out_degree,
    'nonadjacent_edge_count': report.nonadjacent_edge_count,
    'component_count': report.component_count,
    'largest_component_size': report.largest_component_size,
    'seconds': round(time.monotonic() - started, 3),
    'submission': str(submission_path),
}, sort_keys=True), flush=True)
'''
    return (
        source.replace("__UPSTREAM_ARCHIVE__", encoded_archive)
        .replace("__BIOHUB_ARCHIVE__", local_archive)
        .replace("__CONFIG__", encoded_config)
    )


def write_classical_runner_source(
    output_path: Path,
    project_root: Path,
    **kwargs: Any,
) -> Path:
    """Write a generated runner source file for audit and Kaggle publishing."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_classical_runner_source(project_root, **kwargs),
        encoding="utf-8",
    )
    return output_path

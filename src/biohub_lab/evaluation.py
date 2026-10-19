"""Optional adapter to the public Cellmot evaluator.

The evaluator is imported only when a real validation run requests it.  This
keeps the control plane usable in a minimal Python environment and avoids
reimplementing the competition metric with an approximate local score.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _load_graph(path: Path, tracksdata: Any) -> Any:
    result = tracksdata.graph.IndexedRXGraph.from_geff(path)
    return result[0] if isinstance(result, tuple) else result


def _estimated_nodes(path: Path) -> float:
    try:
        from geff import GeffMetadata  # type: ignore

        metadata = GeffMetadata.read(path)
        value = (metadata.extra or {}).get("estimated_number_of_nodes")
        return float(value) if value is not None else float("nan")
    except Exception:
        return float("nan")


def evaluate_geff_directory(
    pred_dir: Path,
    gt_dir: Path,
    baseline_root: Optional[Path] = None,
    max_distance: float = 7.0,
    expected_datasets: Optional[Iterable[str]] = None,
    require_complete: bool = False,
) -> Dict[str, Any]:
    """Evaluate matching ``.geff`` files with the public baseline metric."""

    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    if baseline_root is not None:
        root = Path(baseline_root).resolve()
        src = root / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    try:
        import tracksdata as td  # type: ignore
        from tracking_cellmot.io import open_dataset  # type: ignore
        from tracking_cellmot.metrics import (  # type: ignore
            evaluate as compute_metric,
            node_recall,
            per_sample_metrics,
            summarise,
        )
    except ImportError as exc:
        raise RuntimeError(
            "official evaluation requires the public tracking-cellmot environment "
            "(tracksdata, geff, polars, scipy and tracking_cellmot)"
        ) from exc

    pred_names = {path.stem for path in pred_dir.glob("*.geff")}
    gt_names = {path.stem for path in gt_dir.glob("*.geff")}
    expected_names = {str(name) for name in expected_datasets} if expected_datasets is not None else None
    if require_complete and expected_names is not None:
        missing_predictions = sorted(expected_names - pred_names)
        missing_ground_truth = sorted(expected_names - gt_names)
        extra_predictions = sorted(pred_names - expected_names)
        if missing_predictions or missing_ground_truth or extra_predictions:
            raise ValueError(
                "incomplete GEFF evaluation: missing_predictions={}, missing_ground_truth={}, extra_predictions={}".format(
                    missing_predictions,
                    missing_ground_truth,
                    extra_predictions,
                )
            )
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for name in sorted(pred_names & gt_names):
        try:
            pred = _load_graph(pred_dir / (name + ".geff"), td)
            gt = _load_graph(gt_dir / (name + ".geff"), td)
            try:
                scale = open_dataset(gt_dir / name, load_image=False).scale
            except FileNotFoundError:
                scale = (1.625, 0.40625, 0.40625)
            result = compute_metric(pred, gt, scale=scale, max_distance=max_distance)
            recall = node_recall(pred, gt) if pred.num_edges() > 0 and pred.num_nodes() > 0 else 0.0
            rows.append(per_sample_metrics(result, _estimated_nodes(gt_dir / (name + ".geff")), recall))
        except Exception as exc:
            skipped.append("{}: {}: {}".format(name, type(exc).__name__, exc))
    summary = summarise(rows)
    if require_complete and expected_names is not None and (skipped or len(rows) != len(expected_names)):
        raise ValueError(
            "incomplete GEFF evaluation after scoring: expected={}, scored={}, skipped={}".format(
                len(expected_names), len(rows), skipped
            )
        )
    return {
        "n": summary.get("n", 0),
        "score": summary.get("score"),
        "edge_jaccard": summary.get("edge_jaccard"),
        "adj_edge_jaccard": summary.get("adj_edge_jaccard"),
        "division_jaccard": summary.get("division_jaccard"),
        "node_recall": summary.get("node_recall"),
        "skipped": skipped,
    }


def convert_csv_to_geff(
    submission_csv: Path,
    output_dir: Path,
    baseline_root: Path,
    python_executable: Optional[str] = None,
) -> Path:
    """Use the public baseline converter to create evaluator input graphs."""

    baseline_root = Path(baseline_root).resolve()
    converter = baseline_root / "scripts" / "csv_to_geffs.py"
    if not converter.is_file():
        raise FileNotFoundError("public baseline converter not found: {}".format(converter))
    # The converter runs with ``cwd=baseline_root``.  Resolve both paths so a
    # caller can safely pass run-relative paths without making the adapter
    # depend on the converter's working directory.
    submission_csv = Path(submission_csv).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [python_executable or sys.executable, str(converter), "--csv", str(submission_csv), "--out-dir", str(output_dir)]
    completed = subprocess.run(
        command,
        cwd=str(baseline_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError("csv_to_geffs.py failed ({}): {}".format(completed.returncode, detail))
    return output_dir

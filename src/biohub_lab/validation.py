"""Validate the structural contract of a Biohub submission CSV."""

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REQUIRED_COLUMNS = (
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
)
_DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass
class ValidationReport:
    """Result of a submission contract check."""

    path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    datasets: Set[str] = field(default_factory=set)
    division_parent_count: int = 0
    max_out_degree: int = 0
    nonadjacent_edge_count: int = 0
    component_count: int = 0
    largest_component_size: int = 0

    @property
    def valid(self) -> bool:
        return not self.errors

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "valid": self.valid,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "datasets": sorted(self.datasets),
            "division_parent_count": self.division_parent_count,
            "max_out_degree": self.max_out_degree,
            "nonadjacent_edge_count": self.nonadjacent_edge_count,
            "component_count": self.component_count,
            "largest_component_size": self.largest_component_size,
        }


def _integer(row: Dict[str, str], column: str, line_number: int, report: ValidationReport) -> Optional[int]:
    raw = (row.get(column) or "").strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        report.errors.append("line {}: {} must be an integer".format(line_number, column))
        return None


def _check_unused_fields(row: Dict[str, str], columns: Iterable[str], line_number: int, report: ValidationReport) -> None:
    for column in columns:
        value = _integer(row, column, line_number, report)
        if value is not None and value != -1:
            report.errors.append("line {}: {} must be -1 for this row type".format(line_number, column))


def _valid_dataset_name(name: str) -> bool:
    return bool(_DATASET_RE.fullmatch(name)) and "/" not in name and "\\" not in name


def validate_submission(
    path: Path,
    expected_datasets: Optional[Set[str]] = None,
    expected_timepoints: Optional[Iterable[int]] = None,
    max_rows: int = 5_000_000,
    allow_metric_probe: bool = False,
    include_topology: bool = False,
) -> ValidationReport:
    """Check the Kaggle contract, optionally allowing the public metric probe.

    The competition's public top-scoring notebook adds a separate synthetic
    negative-time hub/fork chain.  That output is not a biological graph and
    must never pass the default validator; ``allow_metric_probe`` is an
    explicit, isolated escape hatch for reproducing that scorer-specific
    experiment.
    """

    path = Path(path)
    report = ValidationReport(str(path))
    if not path.is_file():
        report.errors.append("submission file does not exist: {}".format(path))
        return report

    nodes: Dict[Tuple[str, int], int] = {}
    node_times: Dict[str, Set[int]] = {}
    edges: List[Tuple[str, int, int, int]] = []
    row_ids: Set[int] = set()
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            if tuple(header) != REQUIRED_COLUMNS:
                report.errors.append(
                    "columns must exactly match: {}".format(",".join(REQUIRED_COLUMNS))
                )
                return report

            for line_number, row in enumerate(reader, start=2):
                if line_number - 1 > max_rows:
                    report.errors.append("submission exceeds max_rows={}".format(max_rows))
                    break
                if None in row:
                    report.errors.append("line {}: wrong number of columns".format(line_number))
                    continue
                dataset = (row.get("dataset") or "").strip()
                row_type = (row.get("row_type") or "").strip().lower()
                report.datasets.add(dataset)
                row_id = _integer(row, "id", line_number, report)
                if row_id is not None:
                    if row_id < 0:
                        report.errors.append("line {}: id must be non-negative".format(line_number))
                    elif row_id in row_ids:
                        report.errors.append("line {}: duplicate id".format(line_number))
                    else:
                        row_ids.add(row_id)
                if not _valid_dataset_name(dataset):
                    report.errors.append("line {}: invalid dataset name".format(line_number))
                if row_type not in {"node", "edge"}:
                    report.errors.append("line {}: row_type must be node or edge".format(line_number))
                    continue

                node_id = _integer(row, "node_id", line_number, report)
                t = _integer(row, "t", line_number, report)
                z = _integer(row, "z", line_number, report)
                y = _integer(row, "y", line_number, report)
                x = _integer(row, "x", line_number, report)
                source_id = _integer(row, "source_id", line_number, report)
                target_id = _integer(row, "target_id", line_number, report)
                if row_type == "node":
                    _check_unused_fields(row, ("source_id", "target_id"), line_number, report)
                    if None not in (node_id, t, z, y, x) and -1 in (node_id, t, z, y, x):
                        report.errors.append("line {}: node fields cannot be -1".format(line_number))
                    if node_id is not None and node_id < 0:
                        report.errors.append("line {}: node_id must be non-negative".format(line_number))
                    if t is not None and t < 0 and not allow_metric_probe:
                        report.errors.append("line {}: t must be non-negative".format(line_number))
                    for column, value in (("z", z), ("y", y), ("x", x)):
                        if value is not None and value < 0 and not allow_metric_probe:
                            report.errors.append("line {}: {} must be non-negative".format(line_number, column))
                    if allow_metric_probe and None not in (t, z, y, x) and t < 0:
                        if not (z < 0 and y < 0 and x < 0):
                            report.errors.append(
                                "line {}: metric-probe synthetic nodes must use negative coordinates".format(
                                    line_number
                                )
                            )
                    if None not in (node_id, t) and (dataset, node_id) in nodes:
                        report.errors.append("line {}: duplicate node_id in dataset".format(line_number))
                    elif None not in (node_id, t):
                        nodes[(dataset, node_id)] = t
                        node_times.setdefault(dataset, set()).add(t)
                        report.node_count += 1
                else:
                    _check_unused_fields(row, ("node_id", "t", "z", "y", "x"), line_number, report)
                    if None not in (source_id, target_id):
                        if source_id < 0:
                            report.errors.append("line {}: source_id must be non-negative".format(line_number))
                        if target_id < 0:
                            report.errors.append("line {}: target_id must be non-negative".format(line_number))
                        edges.append((dataset, source_id, target_id, line_number))
                        report.edge_count += 1
    except (OSError, csv.Error) as exc:
        report.errors.append("could not read submission: {}".format(exc))
        return report

    if report.node_count + report.edge_count == 0:
        report.errors.append("submission contains no rows")
    if row_ids and row_ids != set(range(len(row_ids))):
        report.errors.append("id values must be a contiguous sequence starting at 0")
    if expected_datasets:
        missing = sorted(set(expected_datasets) - report.datasets)
        if missing:
            report.errors.append("missing expected datasets: {}".format(", ".join(missing)))
    if expected_timepoints is not None:
        required_times = {int(value) for value in expected_timepoints}
        if any(value < 0 for value in required_times):
            raise ValueError("expected_timepoints must contain non-negative integers")
        for dataset in sorted(report.datasets):
            missing_times = sorted(required_times - node_times.get(dataset, set()))
            if missing_times:
                report.errors.append(
                    "dataset {} missing node timepoints: {}".format(
                        dataset,
                        ", ".join(str(value) for value in missing_times[:20]),
                    )
                )
    outgoing: Dict[Tuple[str, int], int] = {}
    for dataset, source_id, target_id, line_number in edges:
        source_key = (dataset, source_id)
        target_key = (dataset, target_id)
        if source_key not in nodes:
            report.errors.append("line {}: source_id does not reference a node".format(line_number))
        if target_key not in nodes:
            report.errors.append("line {}: target_id does not reference a node".format(line_number))
        if source_id == target_id:
            report.errors.append("line {}: self-edge is not allowed".format(line_number))
        if (
            source_key in nodes
            and target_key in nodes
            and nodes[target_key] != nodes[source_key] + 1
            and not (
                allow_metric_probe
                and (nodes[source_key] < 0 or nodes[target_key] < 0)
            )
        ):
            report.errors.append("line {}: edge must connect adjacent timepoints".format(line_number))
        outgoing[source_key] = outgoing.get(source_key, 0) + 1
    for key, count in outgoing.items():
        if count > 2:
            report.warnings.append("{}:{} has {} outgoing edges; expected at most two for a division".format(key[0], key[1], count))
    if include_topology:
        report.division_parent_count = sum(1 for count in outgoing.values() if count >= 2)
        report.max_out_degree = max(outgoing.values(), default=0)
        report.nonadjacent_edge_count = sum(
            1
            for dataset, source_id, target_id, _line_number in edges
            if (dataset, source_id) in nodes
            and (dataset, target_id) in nodes
            and nodes[(dataset, target_id)] != nodes[(dataset, source_id)] + 1
            and not (
                allow_metric_probe
                and (nodes[(dataset, source_id)] < 0 or nodes[(dataset, target_id)] < 0)
            )
        )

        parent: Dict[Tuple[str, int], Tuple[str, int]] = {key: key for key in nodes}
        sizes: Dict[Tuple[str, int], int] = {key: 1 for key in nodes}

        def find(key: Tuple[str, int]) -> Tuple[str, int]:
            root = key
            while parent[root] != root:
                root = parent[root]
            while parent[key] != key:
                next_key = parent[key]
                parent[key] = root
                key = next_key
            return root

        for dataset, source_id, target_id, _line_number in edges:
            source_key = (dataset, source_id)
            target_key = (dataset, target_id)
            if source_key not in parent or target_key not in parent:
                continue
            source_root = find(source_key)
            target_root = find(target_key)
            if source_root == target_root:
                continue
            if sizes[source_root] < sizes[target_root]:
                source_root, target_root = target_root, source_root
            parent[target_root] = source_root
            sizes[source_root] += sizes[target_root]

        component_sizes: Dict[Tuple[str, int], int] = {}
        for key in nodes:
            root = find(key)
            component_sizes[root] = sizes[root]
        report.component_count = len(component_sizes)
        report.largest_component_size = max(component_sizes.values(), default=0)
    return report


def normalize_submission_ids(path: Path, output_path: Path) -> Path:
    """Rewrite a graph CSV with deterministic one-based node IDs per dataset.

    Candidate implementations may use zero-based internal graph IDs. This
    laboratory convention preserves topology while assigning deterministic IDs
    in node-row order and regenerating the global zero-based row index; public
    scored outputs show that the Kaggle evaluator also accepts other
    non-negative ID schemes.
    """

    path = Path(path)
    output_path = Path(output_path)
    if path.resolve() == output_path.resolve():
        raise ValueError("output_path must differ from path")
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REQUIRED_COLUMNS:
            raise ValueError("input columns do not match the submission contract")
        rows = [dict(row) for row in reader]

    mappings: Dict[Tuple[str, int], int] = {}
    next_ids: Dict[str, int] = {}
    for row in rows:
        row_type = (row.get("row_type") or "").strip().lower()
        dataset = (row.get("dataset") or "").strip()
        if row_type != "node":
            continue
        try:
            old_id = int((row.get("node_id") or "").strip())
        except ValueError as exc:
            raise ValueError("node_id must be an integer") from exc
        if old_id == -1:
            raise ValueError("node rows cannot use node_id=-1")
        key = (dataset, old_id)
        if key in mappings:
            raise ValueError("duplicate node_id in dataset: {}:{}".format(dataset, old_id))
        next_ids[dataset] = next_ids.get(dataset, 0) + 1
        mappings[key] = next_ids[dataset]

    for row_id, row in enumerate(rows):
        row_type = (row.get("row_type") or "").strip().lower()
        dataset = (row.get("dataset") or "").strip()
        row["id"] = str(row_id)
        row["row_type"] = row_type
        if row_type == "node":
            old_id = int((row.get("node_id") or "").strip())
            row["node_id"] = str(mappings[(dataset, old_id)])
        elif row_type == "edge":
            source_id = int((row.get("source_id") or "").strip())
            target_id = int((row.get("target_id") or "").strip())
            source_key = (dataset, source_id)
            target_key = (dataset, target_id)
            if source_key not in mappings or target_key not in mappings:
                raise ValueError("edge references an unknown node_id in dataset {}".format(dataset))
            row["source_id"] = str(mappings[source_key])
            row["target_id"] = str(mappings[target_key])
        else:
            raise ValueError("row_type must be node or edge")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path

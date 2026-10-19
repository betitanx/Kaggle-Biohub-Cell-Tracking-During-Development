"""Exercise the real converter without installing optional ML dependencies."""

import csv
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "external/kaggle-cell-tracking-competition/scripts/csv_to_geffs.py"
)
HEADER = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


class Frame:
    """Only the grouping/filtering boundary used by csv_to_geffs."""

    def __init__(self, rows):
        self.rows = rows
        self.height = len(rows)

    def group_by(self, column):
        groups = {}
        for row in self.rows:
            groups.setdefault(row[column], []).append(row)
        for name, rows in groups.items():
            yield (name,), Frame(rows)

    def filter(self, predicate):
        column, value = predicate
        return Frame([row for row in self.rows if row[column] == value])


class Column:
    def __init__(self, name):
        self.name = name

    def __eq__(self, value):
        return self.name, value


@pytest.fixture
def converter(monkeypatch):
    polars = ModuleType("polars")

    def read_csv(path, columns, schema_overrides=None):
        with Path(path).open(newline="", encoding="utf-8") as handle:
            rows = [{key: row[key] for key in columns} for row in csv.DictReader(handle)]
        # Model numeric-only inference so leading-zero dataset IDs need an
        # explicit string schema, just as they do with the optional CSV reader.
        string_dataset = schema_overrides and schema_overrides.get("dataset") is polars.String
        if rows and not string_dataset and all(row["dataset"].isdigit() for row in rows):
            for row in rows:
                row["dataset"] = int(row["dataset"])
        return Frame(rows)

    polars.read_csv = read_csv
    polars.col = Column
    polars.String = str
    io = ModuleType("tracking_cellmot.io")
    calls = []

    def save_graph(graph, path, *, overwrite):
        calls.append((graph, Path(path), overwrite))

    io.save_graph = save_graph
    stubs = {
        "polars": polars,
        "tracksdata": ModuleType("tracksdata"),
        "tracking_cellmot": ModuleType("tracking_cellmot"),
        "tracking_cellmot.io": io,
    }
    spec = importlib.util.spec_from_file_location("csv_to_geffs_security_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    monkeypatch.setattr(
        module, "build_graph_from_rows",
        lambda nodes, edges: SimpleNamespace(nodes=nodes.rows, edges=edges.rows),
    )
    return SimpleNamespace(module=module, calls=calls)


def submission(tmp_path, names):
    path = tmp_path / "submission.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for index, name in enumerate(names):
            writer.writerow({"dataset": name, "row_type": "node", "node_id": 10 + index,
                             "t": index, "z": 2, "y": 3, "x": 4,
                             "source_id": -1, "target_id": -1})
    return path


@pytest.mark.parametrize("name", [
    "", " ", ".", "..", "../victim", "nested/../../victim", "nested/sample",
    "nested\\sample", "..\\victim", "/absolute", "\\absolute",
    "C:relative", "C:/absolute", "C:\\absolute", "\\\\server\\share",
    "sample:stream", ".hidden", "_sample", "-sample", "with space", "sample\n",
    "sample\x00", "sample*", "sample?", "sample|", "sample<", 'sample"', "émbryo",
])
def test_invalid_later_group_never_reaches_writer(converter, tmp_path, name):
    csv_path = submission(tmp_path, ["safe", name])
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    marker = out_dir / "safe.geff"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="dataset"):
        converter.module.csv_to_geffs(csv_path, out_dir, overwrite=True)

    assert converter.calls == []
    assert marker.read_text(encoding="utf-8") == "keep"


def test_absolute_dataset_cannot_select_external_victim(converter, tmp_path):
    victim = tmp_path / "victim.geff"
    victim.mkdir()
    marker = victim / "keep"
    marker.write_text("outside", encoding="utf-8")
    csv_path = submission(tmp_path, ["safe", str(tmp_path / "victim")])
    out_dir = tmp_path / "outputs"

    with pytest.raises(ValueError, match="dataset"):
        converter.module.csv_to_geffs(csv_path, out_dir, overwrite=True)

    assert converter.calls == []
    assert marker.read_text(encoding="utf-8") == "outside"
    assert not out_dir.exists()


@pytest.mark.parametrize("name", [None, 123, True])
def test_non_string_dataset_is_not_coerced_into_a_filename(converter, tmp_path, name, monkeypatch):
    monkeypatch.setattr(converter.module.pl, "read_csv", lambda *a, **k: Frame([
        {"dataset": "safe", "row_type": "node"}, {"dataset": name, "row_type": "node"},
    ]))

    with pytest.raises(ValueError, match="dataset"):
        converter.module.csv_to_geffs("unused.csv", tmp_path / "outputs", overwrite=True)

    assert converter.calls == []


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("kind", ["directory", "file", "dangling"])
def test_later_symlink_target_is_rejected_even_with_overwrite(
    converter, tmp_path, overwrite, kind,
):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    victim = tmp_path / "victim"
    if kind == "directory":
        victim.mkdir()
        (victim / "keep").write_text("outside", encoding="utf-8")
    elif kind == "file":
        victim.write_text("outside", encoding="utf-8")
    link = out_dir / "linked.geff"
    link.symlink_to(victim, target_is_directory=kind != "file")
    csv_path = submission(tmp_path, ["safe", "linked"])

    with pytest.raises(ValueError, match="[Ss]ymlink"):
        converter.module.csv_to_geffs(csv_path, out_dir, overwrite=overwrite)

    assert converter.calls == []
    assert link.is_symlink()
    if kind != "dangling":
        marker = victim / "keep" if kind == "directory" else victim
        assert marker.read_text(encoding="utf-8") == "outside"


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("suffix", ["", "missing/outputs", "../outputs"])
def test_symlink_output_directory_or_ancestor_is_rejected(
    converter, tmp_path, overwrite, suffix,
):
    victim = tmp_path / "victim"
    victim.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(victim, target_is_directory=True)
    out_dir = link / suffix
    csv_path = submission(tmp_path, ["safe"])

    with pytest.raises(ValueError, match="[Ss]ymlink"):
        converter.module.csv_to_geffs(csv_path, out_dir, overwrite=overwrite)

    assert converter.calls == []
    assert list(victim.iterdir()) == []


def test_dangling_output_directory_is_rejected(converter, tmp_path):
    out_dir = tmp_path / "outputs"
    out_dir.symlink_to(tmp_path / "missing", target_is_directory=True)
    csv_path = submission(tmp_path, ["safe"])

    with pytest.raises(ValueError, match="[Ss]ymlink"):
        converter.module.csv_to_geffs(csv_path, out_dir, overwrite=True)

    assert converter.calls == []
    assert not (tmp_path / "missing").exists()


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_existing_later_output_preflight_prevents_partial_writes(converter, tmp_path, kind):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    existing = out_dir / "existing.geff"
    if kind == "directory":
        existing.mkdir()
        marker = existing / "keep"
    else:
        marker = existing
    marker.write_text("keep", encoding="utf-8")
    csv_path = submission(tmp_path, ["safe", "existing"])

    with pytest.raises(FileExistsError):
        converter.module.csv_to_geffs(csv_path, out_dir)

    assert converter.calls == []
    assert marker.read_text(encoding="utf-8") == "keep"
    assert not (out_dir / "safe.geff").exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_valid_names_preserve_output_paths_and_overwrite_policy(converter, tmp_path, overwrite):
    names = ["6bba_76db78c1", "Sample-2.v1", "123"]
    csv_path = submission(tmp_path, names)
    out_dir = tmp_path / "missing" / "outputs"

    written = converter.module.csv_to_geffs(csv_path, out_dir, overwrite=overwrite)

    assert written == [out_dir / f"{name}.geff" for name in names]
    assert [call[1] for call in converter.calls] == written
    assert [call[2] for call in converter.calls] == [overwrite] * len(names)
    assert converter.calls[0][0].nodes[0]["node_id"] == "10"
    assert converter.calls[0][0].edges == []


def test_default_writer_policy_is_non_destructive(converter, tmp_path):
    csv_path = submission(tmp_path, ["safe"])
    converter.module.csv_to_geffs(csv_path, tmp_path / "outputs")
    assert converter.calls[0][2] is False


def test_numeric_only_dataset_names_keep_leading_zeros(converter, tmp_path):
    csv_path = submission(tmp_path, ["00123"])
    out_dir = tmp_path / "outputs"

    assert converter.module.csv_to_geffs(csv_path, out_dir) == [out_dir / "00123.geff"]
    assert converter.calls[0][0].nodes[0]["dataset"] == "00123"


def test_valid_node_and_edge_rows_reach_graph_builder_unchanged(converter, tmp_path):
    csv_path = submission(tmp_path, ["safe", "safe"])
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=HEADER).writerow({
            "dataset": "safe", "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1, "source_id": 10, "target_id": 11,
        })

    converter.module.csv_to_geffs(csv_path, tmp_path / "outputs")

    graph = converter.calls[0][0]
    assert len(graph.nodes) == 2
    assert all(row["row_type"] == "node" for row in graph.nodes)
    assert len(graph.edges) == 1
    assert graph.edges[0]["source_id"] == "10"
    assert graph.edges[0]["target_id"] == "11"


@pytest.mark.parametrize("overwrite", [False, True])
def test_symlink_introduced_during_graph_build_is_rechecked(
    converter, tmp_path, monkeypatch, overwrite,
):
    csv_path = submission(tmp_path, ["safe"])
    out_dir = tmp_path / "outputs"
    victim = tmp_path / "victim"
    victim.mkdir()

    def build_graph(nodes, edges):
        (out_dir / "safe.geff").symlink_to(victim, target_is_directory=True)
        return object()

    monkeypatch.setattr(converter.module, "build_graph_from_rows", build_graph)
    with pytest.raises(ValueError, match="[Ss]ymlink"):
        converter.module.csv_to_geffs(csv_path, out_dir, overwrite=overwrite)

    assert converter.calls == []
    assert list(victim.iterdir()) == []


def test_output_directory_cannot_be_an_existing_file(converter, tmp_path):
    csv_path = submission(tmp_path, ["safe"])
    out_dir = tmp_path / "outputs"
    out_dir.write_text("keep", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        converter.module.csv_to_geffs(csv_path, out_dir, overwrite=True)

    assert converter.calls == []
    assert out_dir.read_text(encoding="utf-8") == "keep"


def test_relative_output_paths_remain_relative(converter, tmp_path, monkeypatch):
    csv_path = submission(tmp_path, ["safe"])
    monkeypatch.chdir(tmp_path)

    assert converter.module.csv_to_geffs(csv_path, "outputs") == [Path("outputs/safe.geff")]
    assert converter.calls[0][1] == Path("outputs/safe.geff")


@pytest.mark.parametrize("kind", ["directory", "file"])
def test_explicit_overwrite_allows_existing_non_symlink_target(converter, tmp_path, kind):
    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    target = out_dir / "safe.geff"
    target.mkdir() if kind == "directory" else target.touch()
    csv_path = submission(tmp_path, ["safe"])

    assert converter.module.csv_to_geffs(csv_path, out_dir, overwrite=True) == [target]
    assert converter.calls[0][1:] == (target, True)


@pytest.mark.parametrize("flags, expected", [([], False), (["--no-overwrite"], False),
                                            (["--overwrite"], True)])
def test_cli_overwrite_is_explicit(converter, tmp_path, monkeypatch, flags, expected):
    csv_path = submission(tmp_path, ["safe"])
    out_dir = tmp_path / "outputs"
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--csv", str(csv_path),
                                      "--out-dir", str(out_dir), *flags])

    converter.module.main()

    assert converter.calls[0][1:] == (out_dir / "safe.geff", expected)


def test_conflicting_cli_overwrite_flags_are_rejected(converter, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--csv", "input.csv",
                                      "--out-dir", "outputs", "--overwrite", "--no-overwrite"])
    with pytest.raises(SystemExit) as error:
        converter.module.main()
    assert error.value.code == 2
    assert converter.calls == []

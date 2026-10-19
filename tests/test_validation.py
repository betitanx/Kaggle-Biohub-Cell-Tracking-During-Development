import csv
import tempfile
import unittest
from pathlib import Path

from biohub_lab.validation import normalize_submission_ids, validate_submission


HEADER = [
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


def write_rows(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)


class SubmissionValidationTests(unittest.TestCase):
    def test_accepts_nodes_edges_and_temporal_adjacency(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_rows(
                path,
                [
                    {
                        "id": "0",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "10",
                        "t": "0",
                        "z": "2",
                        "y": "3",
                        "x": "4",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                    {
                        "id": "1",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "11",
                        "t": "1",
                        "z": "2",
                        "y": "3",
                        "x": "5",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                    {
                        "id": "2",
                        "dataset": "sample_1",
                        "row_type": "edge",
                        "node_id": "-1",
                        "t": "-1",
                        "z": "-1",
                        "y": "-1",
                        "x": "-1",
                        "source_id": "10",
                        "target_id": "11",
                    },
                ],
            )
            report = validate_submission(path, expected_datasets={"sample_1"})

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.node_count, 2)
        self.assertEqual(report.edge_count, 1)

    def test_optional_topology_summary_exposes_divisions_and_components(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "topology.csv"
            rows = []
            for row_id, node_id, timepoint in (
                (0, 10, 0),
                (1, 11, 1),
                (2, 12, 1),
                (3, 20, 0),
            ):
                rows.append(
                    {
                        "id": str(row_id),
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": str(node_id),
                        "t": str(timepoint),
                        "z": "2",
                        "y": "3",
                        "x": str(4 + node_id),
                        "source_id": "-1",
                        "target_id": "-1",
                    }
                )
            for row_id, target_id in ((4, 11), (5, 12)):
                rows.append(
                    {
                        "id": str(row_id),
                        "dataset": "sample_1",
                        "row_type": "edge",
                        "node_id": "-1",
                        "t": "-1",
                        "z": "-1",
                        "y": "-1",
                        "x": "-1",
                        "source_id": "10",
                        "target_id": str(target_id),
                    }
                )
            write_rows(path, rows)
            report = validate_submission(path, include_topology=True)

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.division_parent_count, 1)
        self.assertEqual(report.max_out_degree, 2)
        self.assertEqual(report.nonadjacent_edge_count, 0)
        self.assertEqual(report.component_count, 2)
        self.assertEqual(report.largest_component_size, 3)

    def test_metric_probe_requires_explicit_opt_in_for_negative_hub_rows(self):
        rows = [
            {
                "id": "0", "dataset": "sample_1", "row_type": "node",
                "node_id": "10", "t": "0", "z": "2", "y": "3", "x": "4",
                "source_id": "-1", "target_id": "-1",
            },
            {
                "id": "1", "dataset": "sample_1", "row_type": "node",
                "node_id": "11", "t": "-1000", "z": "-10000", "y": "-10000", "x": "-10000",
                "source_id": "-1", "target_id": "-1",
            },
            {
                "id": "2", "dataset": "sample_1", "row_type": "edge",
                "node_id": "-1", "t": "-1", "z": "-1", "y": "-1", "x": "-1",
                "source_id": "11", "target_id": "10",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.csv"
            write_rows(path, rows)
            strict = validate_submission(path, expected_timepoints={0})
            probe = validate_submission(path, expected_timepoints={0}, allow_metric_probe=True)

        self.assertFalse(strict.valid)
        self.assertTrue(probe.valid, probe.errors)

    def test_rejects_edge_to_missing_node_and_non_adjacent_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_rows(
                path,
                [
                    {
                        "id": "0",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "10",
                        "t": "0",
                        "z": "2",
                        "y": "3",
                        "x": "4",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                    {
                        "id": "1",
                        "dataset": "sample_1",
                        "row_type": "edge",
                        "node_id": "-1",
                        "t": "-1",
                        "z": "-1",
                        "y": "-1",
                        "x": "-1",
                        "source_id": "10",
                        "target_id": "99",
                    },
                ],
            )
            report = validate_submission(path)

        self.assertFalse(report.valid)
        self.assertTrue(any("target_id" in error for error in report.errors))

    def test_rejects_path_like_dataset_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_rows(
                path,
                [
                    {
                        "id": "0",
                        "dataset": "../private",
                        "row_type": "node",
                        "node_id": "1",
                        "t": "0",
                        "z": "0",
                        "y": "0",
                        "x": "0",
                        "source_id": "-1",
                        "target_id": "-1",
                    }
                ],
            )
            report = validate_submission(path)

        self.assertFalse(report.valid)
        self.assertTrue(any("dataset" in error for error in report.errors))

    def test_rejects_duplicate_row_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_rows(
                path,
                [
                    {
                        "id": "0",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "1",
                        "t": "0",
                        "z": "0",
                        "y": "0",
                        "x": "0",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                    {
                        "id": "0",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "2",
                        "t": "1",
                        "z": "0",
                        "y": "0",
                        "x": "0",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                ],
            )
            report = validate_submission(path)

        self.assertFalse(report.valid)
        self.assertTrue(any("duplicate id" in error for error in report.errors))

    def test_accepts_zero_based_graph_ids_used_by_scored_notebooks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_rows(
                path,
                [
                    {
                        "id": "0",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "0",
                        "t": "0",
                        "z": "0",
                        "y": "0",
                        "x": "0",
                        "source_id": "-1",
                        "target_id": "-1",
                    }
                ],
            )
            report = validate_submission(path)

        self.assertTrue(report.valid, report.errors)

    def test_rejects_missing_expected_timepoints(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_rows(
                path,
                [
                    {
                        "id": "0",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "0",
                        "t": "0",
                        "z": "0",
                        "y": "0",
                        "x": "0",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                    {
                        "id": "1",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "1",
                        "t": "2",
                        "z": "0",
                        "y": "0",
                        "x": "0",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                ],
            )
            report = validate_submission(path, expected_timepoints=range(3))

        self.assertFalse(report.valid)
        self.assertTrue(any("missing node timepoints" in error for error in report.errors))

    def test_normalize_submission_ids_preserves_edges_and_uses_one_based_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.csv"
            target = Path(directory) / "normalized.csv"
            write_rows(
                source,
                [
                    {
                        "id": "88",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "0",
                        "t": "0",
                        "z": "0",
                        "y": "0",
                        "x": "0",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                    {
                        "id": "99",
                        "dataset": "sample_1",
                        "row_type": "node",
                        "node_id": "7",
                        "t": "1",
                        "z": "0",
                        "y": "0",
                        "x": "1",
                        "source_id": "-1",
                        "target_id": "-1",
                    },
                    {
                        "id": "100",
                        "dataset": "sample_1",
                        "row_type": "edge",
                        "node_id": "-1",
                        "t": "-1",
                        "z": "-1",
                        "y": "-1",
                        "x": "-1",
                        "source_id": "0",
                        "target_id": "7",
                    },
                ],
            )
            normalize_submission_ids(source, target)
            report = validate_submission(target, expected_datasets={"sample_1"})
            with target.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            normalized_bytes = target.read_bytes()

        self.assertTrue(report.valid, report.errors)
        self.assertEqual([row["id"] for row in rows], ["0", "1", "2"])
        self.assertEqual([row["node_id"] for row in rows[:2]], ["1", "2"])
        self.assertEqual((rows[2]["source_id"], rows[2]["target_id"]), ("1", "2"))
        self.assertNotIn(b"\r\n", normalized_bytes)


if __name__ == "__main__":
    unittest.main()

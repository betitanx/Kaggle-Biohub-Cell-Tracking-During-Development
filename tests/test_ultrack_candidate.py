import tempfile
import unittest
from pathlib import Path

from biohub_lab.ultrack_candidate import (
    graph_from_tracks_rows,
    track_labels_with_ultrack,
)
from biohub_lab.classical import write_submission_csv
from biohub_lab.validation import validate_submission


class UltrackCandidateTests(unittest.TestCase):
    def test_tracks_rows_preserve_parent_division_and_csv_contract(self):
        rows = [
            {"id": 10, "track_id": 1, "t": 0, "z": 2.0, "y": 3.0, "x": 4.0, "parent_id": -1},
            {"id": 20, "track_id": 1, "t": 1, "z": 2.0, "y": 3.0, "x": 5.0, "parent_id": 10},
            {"id": 21, "track_id": 2, "t": 1, "z": 2.0, "y": 4.0, "x": 5.0, "parent_id": 10},
        ]

        graph = graph_from_tracks_rows("sample", rows)

        self.assertEqual([(node.t, node.z, node.y, node.x) for node in graph.nodes], [(0, 2, 3, 4), (1, 2, 3, 5), (1, 2, 4, 5)])
        self.assertEqual(graph.edges, [(0, 1), (0, 2)])
        with tempfile.TemporaryDirectory() as directory:
            artifact = write_submission_csv([graph], Path(directory) / "submission.csv")
            report = validate_submission(artifact, expected_datasets={"sample"})
        self.assertTrue(report.valid, report.errors)

    def test_tracks_rows_reject_non_adjacent_parent_edges(self):
        rows = [
            {"id": 10, "t": 0, "z": 1, "y": 1, "x": 1, "parent_id": -1},
            {"id": 20, "t": 2, "z": 1, "y": 1, "x": 2, "parent_id": 10},
        ]

        with self.assertRaises(ValueError):
            graph_from_tracks_rows("sample", rows)

    def test_tracks_rows_without_parent_metadata_remain_valid_root_tracks(self):
        graph = graph_from_tracks_rows(
            "sample",
            [{"id": 1, "t": 0, "z": 1, "y": 1, "x": 1}],
        )

        self.assertEqual(len(graph.nodes), 1)
        self.assertEqual(graph.edges, [])

    def test_lineage_mapping_supplies_parent_edges_when_rows_omit_parent_column(self):
        graph = graph_from_tracks_rows(
            "sample",
            [
                {"id": 10, "t": 0, "z": 1, "y": 1, "x": 1},
                {"id": 20, "t": 1, "z": 1, "y": 1, "x": 2},
            ],
            lineage={20: 10},
        )

        self.assertEqual(graph.edges, [(0, 1)])

    def test_tracks_rows_reject_negative_coordinates(self):
        with self.assertRaises(ValueError):
            graph_from_tracks_rows(
                "sample",
                [{"id": 1, "t": 0, "z": -1, "y": 1, "x": 1}],
            )

    def test_ultrack_bridge_uses_documented_labels_scale_and_isolated_config(self):
        class ConfigPart:
            def __init__(self):
                self.working_dir = None
                self.database_file_name = None
                self.min_frontier = None
                self.max_distance = None

        class FakeConfig:
            def __init__(self):
                self.data_config = ConfigPart()
                self.segmentation_config = ConfigPart()
                self.linking_config = ConfigPart()

        class FakeTracker:
            instance = None

            def __init__(self, config):
                self.config = config
                self.calls = []
                FakeTracker.instance = self

            def track(self, **kwargs):
                self.calls.append(kwargs)

            def to_tracks_layer(self):
                return ([{"id": 1, "t": 0, "z": 1, "y": 1, "x": 1, "parent_id": -1}], {})

        class FakeUltrack:
            MainConfig = FakeConfig
            Tracker = FakeTracker

        with tempfile.TemporaryDirectory() as directory:
            graph = track_labels_with_ultrack(
                labels=[[[[1]]]],
                dataset="sample",
                working_dir=Path(directory),
                scale_um=(1.625, 0.40625, 0.40625),
                ultrack_module=FakeUltrack,
            )

            tracker = FakeTracker.instance
            self.assertEqual(len(graph.nodes), 1)
            self.assertEqual(tracker.calls[0]["labels"], [[[[1]]]])
            self.assertEqual(tracker.calls[0]["scale"], (1.625, 0.40625, 0.40625))
            self.assertEqual(tracker.calls[0]["overwrite"], "all")
            self.assertEqual(tracker.config.data_config.database_file_name, "data.db")
            self.assertEqual(tracker.config.segmentation_config.min_frontier, 0.0)


if __name__ == "__main__":
    unittest.main()

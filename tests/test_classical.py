import csv
import tempfile
import unittest
from pathlib import Path

from biohub_lab.classical import (
    Detection,
    TrackingGraph,
    detect_local_maxima,
    detect_local_maxima_fast,
    infer_division_links,
    link_detections,
    remove_isolated_nodes,
    track_frames,
    write_submission_csv,
)
from biohub_lab.validation import validate_submission


def frame(shape=(3, 5, 5), values=None):
    z_size, y_size, x_size = shape
    result = [
        [[0.0 for _ in range(x_size)] for _ in range(y_size)]
        for _ in range(z_size)
    ]
    for (z, y, x), value in (values or {}).items():
        result[z][y][x] = value
    return result


class ClassicalCandidateTests(unittest.TestCase):
    def test_detects_deterministic_3d_local_maxima(self):
        image = frame(
            values={
                (1, 1, 1): 10.0,
                (1, 2, 3): 8.0,
                (1, 1, 2): 2.0,
            }
        )

        detections = detect_local_maxima(image, threshold=3.0)

        self.assertEqual([(item.z, item.y, item.x) for item in detections], [(1, 1, 1), (1, 2, 3)])
        self.assertEqual([item.intensity for item in detections], [10.0, 8.0])

    def test_fast_detector_preserves_the_same_contract(self):
        image = frame(values={(1, 1, 1): 10.0, (1, 2, 3): 8.0})

        detections = detect_local_maxima_fast(image, threshold=3.0)

        self.assertEqual([(item.z, item.y, item.x) for item in detections], [(1, 1, 1), (1, 2, 3)])

    def test_plateau_has_one_lexicographically_stable_detection_and_respects_cap(self):
        image = frame(
            values={
                (1, 1, 1): 9.0,
                (1, 1, 2): 9.0,
                (1, 3, 3): 8.0,
            }
        )

        detections = detect_local_maxima(image, threshold=3.0, max_detections=1)

        self.assertEqual(len(detections), 1)
        self.assertEqual((detections[0].z, detections[0].y, detections[0].x), (1, 1, 1))

    def test_links_nearest_detections_once_in_physical_space(self):
        previous = [
            Detection(t=0, z=0, y=0, x=0, intensity=5.0),
            Detection(t=0, z=0, y=0, x=10, intensity=4.0),
        ]
        current = [
            Detection(t=1, z=0, y=0, x=1, intensity=6.0),
            Detection(t=1, z=0, y=0, x=11, intensity=3.0),
        ]

        links = link_detections(
            previous,
            current,
            scale_um=(1.0, 1.0, 1.0),
            max_distance_um=2.0,
        )

        self.assertEqual(links, [(0, 0), (1, 1)])

    def test_spatial_index_matches_dependency_free_linking(self):
        previous = [
            Detection(t=0, z=0, y=index // 4, x=index % 4, intensity=5.0 - index / 100.0)
            for index in range(12)
        ]
        current = [
            Detection(t=1, z=0, y=index // 4, x=index % 4, intensity=6.0 - index / 100.0)
            for index in range(12)
        ]

        indexed = link_detections(
            previous,
            current,
            scale_um=(1.0, 1.0, 1.0),
            max_distance_um=1.5,
            use_spatial_index=True,
        )
        reference = link_detections(
            previous,
            current,
            scale_um=(1.0, 1.0, 1.0),
            max_distance_um=1.5,
            use_spatial_index=False,
        )

        self.assertEqual(indexed, reference)

    def test_track_and_write_csv_satisfy_submission_contract(self):
        first = frame(values={(1, 1, 1): 10.0})
        second = frame(values={(1, 1, 2): 10.0})

        graph = track_frames(
            [first, second],
            dataset="sample",
            threshold=3.0,
            scale_um=(1.0, 1.0, 1.0),
            max_link_distance_um=2.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_submission_csv([graph], path)
            report = validate_submission(path, expected_datasets={"sample"})
            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            serialized_bytes = path.read_bytes()

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.node_count, 2)
        self.assertEqual(report.edge_count, 1)
        self.assertEqual([row["row_type"] for row in rows], ["node", "node", "edge"])
        self.assertEqual([row["node_id"] for row in rows[:2]], ["1", "2"])
        self.assertEqual((rows[2]["source_id"], rows[2]["target_id"]), ("1", "2"))
        self.assertNotIn(b"\r\n", serialized_bytes)

    def test_optional_timepoint_coverage_adds_a_deterministic_fallback_node(self):
        empty_above_threshold = frame(values={(0, 0, 0): 1.0})

        graph = track_frames(
            [empty_above_threshold, empty_above_threshold],
            dataset="sample",
            threshold=2.0,
            ensure_timepoint_coverage=True,
        )

        self.assertEqual([node.t for node in graph.nodes], [0, 1])
        self.assertEqual([(node.z, node.y, node.x) for node in graph.nodes], [(0, 0, 0), (0, 0, 0)])

    def test_conservative_division_inference_adds_second_daughter(self):
        from biohub_lab.synthetic_benchmark import evaluate_graph, make_case

        case = make_case(seed=13)
        one_to_one = track_frames(
            case.frames,
            dataset=case.truth.dataset,
            threshold=case.threshold,
            scale_um=case.scale_um,
            max_link_distance_um=7.0,
            fast=False,
        )
        division_aware = infer_division_links(
            one_to_one,
            scale_um=case.scale_um,
            max_link_distance_um=7.0,
            sister_max_distance_um=5.0,
            midpoint_tolerance_um=3.0,
        )
        metrics = evaluate_graph(division_aware, case.truth, scale_um=case.scale_um)

        self.assertEqual(metrics["division_recall"], 1.0)
        self.assertEqual(metrics["edge_recall"], 1.0)

    def test_isolated_node_filter_remaps_ids_and_preserves_edges(self):
        graph = TrackingGraph(
            dataset="sample",
            nodes=[
                Detection(t=0, z=1, y=1, x=1, intensity=10.0, node_id=10),
                Detection(t=1, z=1, y=1, x=2, intensity=9.0, node_id=20),
                Detection(t=1, z=5, y=5, x=5, intensity=8.0, node_id=30),
            ],
            edges=[(10, 20)],
        )

        filtered = remove_isolated_nodes(graph)

        self.assertEqual([node.node_id for node in filtered.nodes], [0, 1])
        self.assertEqual(filtered.edges, [(0, 1)])


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from biohub_lab.synthetic_benchmark import evaluate_graph, make_case, run_benchmark
from biohub_lab.validation import validate_submission


class SyntheticBenchmarkTests(unittest.TestCase):
    def test_h1_detects_nodes_and_exposes_division_limitation(self):
        case = make_case(seed=7)
        from biohub_lab.classical import track_frames

        predicted = track_frames(
            case.frames,
            dataset=case.truth.dataset,
            threshold=case.threshold,
            scale_um=case.scale_um,
            max_link_distance_um=7.0,
            fast=False,
        )
        metrics = evaluate_graph(predicted, case.truth, scale_um=case.scale_um)

        self.assertGreaterEqual(metrics["node_recall"], 0.95)
        self.assertGreaterEqual(metrics["edge_recall"], 0.85)
        self.assertEqual(metrics["division_recall"], 0.0)

    def test_benchmark_writes_valid_artifact_and_diagnostic_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            metrics = run_benchmark(run_dir, seed=11)
            artifact = Path(str(metrics["artifact"]))
            artifact_valid = validate_submission(artifact).valid
            persisted = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertTrue(metrics["quality_gate_passed"])
        self.assertEqual(metrics["evaluation_scope"], "synthetic_diagnostic_only")
        self.assertTrue(artifact_valid)
        self.assertEqual(persisted["candidate"], "H1_peak_link")

    def test_division_aware_benchmark_closes_the_known_h1_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_benchmark(Path(directory), seed=17, infer_divisions=True)

        self.assertEqual(metrics["candidate"], "H7_division_aware")
        self.assertEqual(metrics["division_recall"], 1.0)
        self.assertTrue(metrics["quality_gate_passed"])

    def test_isolated_node_postprocessing_removes_the_known_false_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics = run_benchmark(Path(directory), seed=19, filter_isolated_nodes=True)

        self.assertEqual(metrics["candidate"], "H8_clean_graph")
        self.assertEqual(metrics["node_precision"], 1.0)
        self.assertTrue(metrics["quality_gate_passed"])


if __name__ == "__main__":
    unittest.main()

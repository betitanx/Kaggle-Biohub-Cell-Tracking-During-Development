import unittest
import tempfile
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from biohub_lab.classical import Detection, TrackingGraph
from biohub_lab.classical_candidate import (
    _run,
    quantile_linear,
    summarise_graphs,
    threshold_for_dataset,
)
from biohub_lab.ome_zarr import OpenedOmeZarr
from biohub_lab.validation import validate_submission


class FakeArray:
    def __init__(self, frames):
        self.frames = frames
        self.shape = (len(frames), len(frames[0]), len(frames[0][0]), len(frames[0][0][0]))

    def __getitem__(self, index):
        return self.frames[index]


def _frame(z, y, x, peak):
    image = [[[0.0 for _ in range(x)] for _ in range(y)] for _ in range(z)]
    image[1][1][1] = peak
    return image


class ClassicalCandidateAdapterTests(unittest.TestCase):
    def test_quantile_is_deterministic_without_numpy(self):
        self.assertEqual(quantile_linear([0.0, 1.0, 2.0, 3.0], 0.75), 2.25)

    def test_explicit_threshold_wins_over_quantile(self):
        self.assertEqual(threshold_for_dataset(12.5, [1.0, 2.0], 0.99), 12.5)
        self.assertEqual(threshold_for_dataset(None, [1.0, 2.0], 0.5), 1.5)

    def test_summarise_graphs_reports_structure_not_a_fake_score(self):
        graph = TrackingGraph(
            dataset="sample",
            nodes=[Detection(t=0, z=1, y=2, x=3, intensity=10.0, node_id=0)],
            edges=[],
        )

        summary = summarise_graphs([graph], thresholds={"sample": 4.0})

        self.assertEqual(summary["node_count"], 1)
        self.assertEqual(summary["edge_count"], 0)
        self.assertEqual(summary["thresholds"], {"sample": 4.0})
        self.assertNotIn("local_score", summary)

    def test_run_writes_valid_artifact_without_fabricating_a_score(self):
        fake = OpenedOmeZarr(
            path=Path("sample.zarr"),
            array=FakeArray([_frame(3, 5, 5, 10.0), _frame(3, 5, 5, 10.0)]),
            attrs={},
            scale=(1.0, 1.0, 1.0),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "sample.zarr").mkdir()
            run_dir = root / "run"
            args = Namespace(
                data_dir=data_dir,
                run_dir=run_dir,
                array_key="0",
                threshold=3.0,
                quantile=0.995,
                max_frames=None,
                neighborhood=(1, 1, 1),
                max_link_distance_um=2.0,
                max_detections_per_frame=20,
                infer_divisions=True,
                sister_max_distance_um=5.0,
                midpoint_tolerance_um=3.0,
                max_parent_displacement_delta_um=3.0,
                min_sibling_intensity_ratio=0.9,
                evaluate_gt_dir=None,
                holdout_data_dir=None,
                holdout_gt_dir=None,
                baseline_root=None,
                max_eval_distance=7.0,
            )
            with patch("biohub_lab.classical_candidate.open_ome_zarr", return_value=fake):
                metrics = _run(args)
            report = validate_submission(run_dir / "submission.csv", expected_datasets={"sample"})

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(metrics["candidate_metrics"]["node_count"], 2)
        self.assertEqual(metrics["candidate_metrics"]["thresholds"], {"sample": 3.0})
        self.assertTrue(metrics["division_inference"]["enabled"])
        self.assertNotIn("local_score", metrics)


if __name__ == "__main__":
    unittest.main()

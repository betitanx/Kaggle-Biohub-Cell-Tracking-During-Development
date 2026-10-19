import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import ModuleType
import sys
from unittest.mock import patch

from biohub_lab.classical import write_submission_csv
from biohub_lab.external_candidate import (
    centroids_from_labels,
    load_detector,
    normalise_detector_name,
    predict_labels,
    track_label_frames,
    _run,
)
from biohub_lab.ome_zarr import OpenedOmeZarr
from biohub_lab.validation import validate_submission


def labels_frame(values):
    result = [[[0 for _ in range(5)] for _ in range(5)] for _ in range(3)]
    for (z, y, x), label in values.items():
        result[z][y][x] = label
    return result


class ExternalCandidateCoreTests(unittest.TestCase):
    def test_centroids_are_deterministic_and_intensity_weighted(self):
        labels = labels_frame({(1, 1, 1): 1, (1, 1, 2): 1, (1, 3, 3): 2})
        image = labels_frame({(1, 1, 1): 4, (1, 1, 2): 8, (1, 3, 3): 10})

        detections = centroids_from_labels(labels, image=image, time_index=2)

        self.assertEqual(
            [(item.t, item.z, item.y, item.x, item.intensity) for item in detections],
            [(2, 1, 3, 3, 10.0), (2, 1, 1, 2, 6.0)],
        )

    def test_label_frames_link_adjacent_centroids_and_validate_csv(self):
        frames = [
            (labels_frame({(1, 1, 1): 1}), labels_frame({(1, 1, 1): 1})),
            (labels_frame({(1, 1, 2): 1}), labels_frame({(1, 1, 2): 1})),
        ]

        graph = track_label_frames(
            frames,
            dataset="sample",
            scale_um=(1.0, 1.0, 1.0),
            max_link_distance_um=2.0,
        )

        self.assertEqual(len(graph.nodes), 2)
        self.assertEqual(graph.edges, [(0, 1)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            write_submission_csv([graph], path)
            report = validate_submission(path, expected_datasets={"sample"})
        self.assertTrue(report.valid, report.errors)

    def test_detector_name_is_restricted_to_supported_adapters(self):
        self.assertEqual(normalise_detector_name("CellPose"), "cellpose")
        with self.assertRaises(ValueError):
            normalise_detector_name("unknown")

    def test_cellpose_prediction_uses_documented_3d_parameters(self):
        class FakeCellpose:
            def __init__(self):
                self.kwargs = None

            def eval(self, image, **kwargs):
                self.kwargs = kwargs
                return labels_frame({(1, 1, 1): 1}), [], []

        model = FakeCellpose()
        labels = predict_labels(
            "cellpose",
            model,
            labels_frame({(1, 1, 1): 3}),
            cellprob_threshold=0.2,
            flow_threshold=0.3,
            anisotropy=4.0,
        )

        self.assertEqual(labels[1][1][1], 1)
        self.assertEqual(
            model.kwargs,
            {
                "do_3D": True,
                "cellprob_threshold": 0.2,
                "flow_threshold": 0.3,
                "anisotropy": 4.0,
            },
        )

    def test_stardist_prediction_extracts_instance_labels(self):
        class FakeStarDist:
            def predict_instances(self, image, **kwargs):
                return labels_frame({(1, 1, 1): 7}), {"prob_thresh": kwargs["prob_thresh"]}

        labels = predict_labels(
            "stardist",
            FakeStarDist(),
            labels_frame({(1, 1, 1): 3}),
            probability_threshold=0.4,
            nms_threshold=0.2,
        )

        self.assertEqual(labels[1][1][1], 7)

    def test_load_cellpose_uses_current_model_class_and_options(self):
        fake_models = ModuleType("cellpose.models")

        class FakeModel:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        fake_models.CellposeModel = FakeModel
        fake_cellpose = ModuleType("cellpose")
        fake_cellpose.models = fake_models
        with patch.dict(sys.modules, {"cellpose": fake_cellpose, "cellpose.models": fake_models}):
            model = load_detector("cellpose", model_path="weights", use_gpu=True)

        self.assertEqual(model.kwargs, {"gpu": True, "pretrained_model": "weights"})

    def test_load_stardist_requires_and_opens_a_model_directory(self):
        fake_models = ModuleType("stardist.models")
        captured = {}

        class FakeStarDist3D:
            def __init__(self, config, **kwargs):
                captured["config"] = config
                captured.update(kwargs)

        fake_models.StarDist3D = FakeStarDist3D
        fake_stardist = ModuleType("stardist")
        fake_stardist.models = fake_models
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory) / "model"
            model_dir.mkdir()
            with patch.dict(sys.modules, {"stardist": fake_stardist, "stardist.models": fake_models}):
                load_detector("stardist", model_path=model_dir)

        self.assertIsNone(captured["config"])
        self.assertEqual(captured["name"], "model")
        self.assertEqual(captured["basedir"], str(model_dir.parent))

    def test_run_adapter_writes_valid_csv_without_fabricating_score(self):
        class FakeArray:
            shape = (2, 3, 5, 5)

            def __getitem__(self, index):
                return labels_frame({(1, 1, 1 + index): 3})

        class FakeCellpose:
            def eval(self, image, **_kwargs):
                return labels_frame({(1, 1, 1): 1}), [], []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "sample.zarr").mkdir()
            run_dir = root / "run"
            args = Namespace(
                detector="cellpose",
                model_path=None,
                gpu=False,
                data_dir=data_dir,
                run_dir=run_dir,
                array_key="0",
                max_frames=None,
                max_link_distance_um=2.0,
                max_detections_per_frame=20,
                cellprob_threshold=0.0,
                flow_threshold=0.4,
                probability_threshold=0.5,
                nms_threshold=0.3,
                validation_data_dir=None,
                evaluate_gt_dir=None,
                holdout_data_dir=None,
                holdout_gt_dir=None,
                baseline_root=None,
                max_eval_distance=7.0,
            )
            opened = OpenedOmeZarr(data_dir / "sample.zarr", FakeArray(), {}, (1.0, 1.0, 1.0))
            with patch("biohub_lab.external_candidate.load_detector", return_value=FakeCellpose()), patch(
                "biohub_lab.external_candidate.open_ome_zarr", return_value=opened
            ):
                metrics = _run(args)
            report = validate_submission(run_dir / "submission.csv", expected_datasets={"sample"})

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(metrics["candidate_metrics"]["node_count"], 2)
        self.assertNotIn("local_score", metrics)


if __name__ == "__main__":
    unittest.main()

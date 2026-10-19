import tempfile
import unittest
from pathlib import Path

from biohub_lab.cellmot_candidate import (
    _run_prediction,
    _patch_predict_script_cli,
    build_predict_command,
    prediction_directory,
    summarise_geff_directory,
)
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class CellmotCandidateTests(unittest.TestCase):
    def test_prediction_directory_isolated_by_run_user_method_and_split(self):
        root = Path("/tmp/cellmot")

        result = prediction_directory(root, "biohub_run_1", "baseline", 2)

        self.assertEqual(result, root / "predictions" / "biohub_run_1" / "baseline" / "split_2")

    def test_predict_command_uses_external_script_without_shell(self):
        command = build_predict_command(
            baseline_root=Path("/tmp/cellmot"),
            python_executable="/tmp/cellmot/.venv/bin/python",
            data_dir=Path("/tmp/data/test"),
            split=0,
            method="baseline",
            weights=Path("/tmp/cellmot/weights/model.pth"),
        )

        self.assertEqual(command[0], "/tmp/cellmot/.venv/bin/python")
        self.assertEqual(command[1], "/tmp/cellmot/scripts/predict_unet_transformer.py")
        self.assertIn("--data-dir", command)
        self.assertIn("/tmp/data/test", command)
        self.assertIn("--weights", command)
        self.assertNotIn("sh", command)

        command_with_pool = build_predict_command(
            baseline_root=Path("/tmp/cellmot"),
            python_executable="python",
            data_dir=Path("/tmp/data/test"),
            split=0,
            method="baseline",
            pool_kernel_um=5.0,
        )
        self.assertEqual(command_with_pool[-2:], ["--pool-kernel-um", "5.0"])

        command_with_edge_threshold = build_predict_command(
            baseline_root=Path("/tmp/cellmot"),
            python_executable="python",
            data_dir=Path("/tmp/data/test"),
            split=0,
            method="baseline",
            edge_threshold=0.48,
        )
        self.assertEqual(command_with_edge_threshold[-2:], ["--edge-threshold", "0.48"])

        command_with_edge_children = build_predict_command(
            baseline_root=Path("/tmp/cellmot"),
            python_executable="python",
            data_dir=Path("/tmp/data/test"),
            split=0,
            method="baseline",
            edge_topk_children=2,
        )
        self.assertEqual(command_with_edge_children[-2:], ["--edge-topk-children", "2"])

        command_with_dihedral_tta = build_predict_command(
            baseline_root=Path("/tmp/cellmot"),
            python_executable="python",
            data_dir=Path("/tmp/data/test"),
            split=0,
            method="baseline",
            det_tta_mode="dihedral8",
        )
        self.assertEqual(command_with_dihedral_tta[-2:], ["--det-tta-mode", "dihedral8"])

        command_with_ilp = build_predict_command(
            baseline_root=Path("/tmp/cellmot"),
            python_executable="python",
            data_dir=Path("/tmp/data/test"),
            split=0,
            method="baseline",
            use_ilp=True,
            ilp_appearance_weight=0.0,
            ilp_disappearance_weight=1.4,
            ilp_division_weight=1.0,
        )
        self.assertIn("--ilp-appearance-weight", command_with_ilp)
        self.assertIn("--ilp-disappearance-weight", command_with_ilp)
        self.assertIn("--ilp-division-weight", command_with_ilp)

        debug_command = build_predict_command(
            baseline_root=Path("/tmp/cellmot"),
            python_executable="python",
            data_dir=Path("/tmp/data/test"),
            split=0,
            method="baseline",
            debug_video=Path("/tmp/data/test/sample.zarr"),
        )
        self.assertIn("--debug-video", debug_command)
        self.assertIn("/tmp/data/test/sample.zarr", debug_command)

    def test_summary_counts_geff_entries_without_claiming_a_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a.geff").write_text("placeholder", encoding="utf-8")
            (root / "b.geff").mkdir()

            summary = summarise_geff_directory(root)

        self.assertEqual(summary, {"dataset_count": 2, "geff_count": 2})
        self.assertNotIn("local_score", summary)

    def test_prediction_adapter_converts_and_validates_upstream_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            (baseline / "scripts").mkdir(parents=True)
            (baseline / "scripts" / "predict_unet_transformer.py").write_text("", encoding="utf-8")
            (baseline / "scripts" / "geffs_to_csv.py").write_text("", encoding="utf-8")
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "sample_1.zarr").mkdir()
            run_dir = root / "run"
            run_dir.mkdir()

            def fake_command(command, cwd, env, stdout_path, stderr_path):
                if "predict_unet_transformer.py" in command[1]:
                    generated = prediction_directory(cwd, env["USER"], "baseline", 0)
                    generated.mkdir(parents=True)
                    (generated / "sample_1.geff").write_text("placeholder", encoding="utf-8")
                else:
                    Path(command[-1]).write_text(
                        (ROOT / "tests" / "fixtures" / "submission.csv").read_text(encoding="utf-8"),
                        encoding="utf-8",
                    )

            with patch("biohub_lab.cellmot_candidate._run_command", side_effect=fake_command):
                result = _run_prediction(
                    baseline,
                    "python",
                    data_dir,
                    run_dir,
                    "final",
                    0,
                    "baseline",
                    None,
                    None,
                    False,
                )

        self.assertEqual(result["candidate_metrics"], {"dataset_count": 1, "geff_count": 1})
        self.assertTrue(result["validation"]["valid"])

    def test_prediction_adapter_can_skip_intermediate_csv_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline"
            (baseline / "scripts").mkdir(parents=True)
            (baseline / "scripts" / "predict_unet_transformer.py").write_text("", encoding="utf-8")
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "sample.zarr").mkdir()
            run_dir = root / "run"
            run_dir.mkdir()

            def fake_command(command, cwd, env, stdout_path, stderr_path):
                generated = prediction_directory(cwd, env["USER"], "baseline", 0)
                generated.mkdir(parents=True)
                (generated / "sample.geff").write_text("placeholder", encoding="utf-8")

            with patch("biohub_lab.cellmot_candidate._run_command", side_effect=fake_command):
                result = _run_prediction(
                    baseline,
                    "python",
                    data_dir,
                    run_dir,
                    "validation",
                    0,
                    "baseline",
                    None,
                    None,
                    False,
                    convert_csv=False,
                )

        self.assertIsNone(result["validation"])
        self.assertFalse((run_dir / "validation_submission.csv").exists())

    def test_support_predictor_copy_receives_pool_kernel_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            (scripts / "predict_unet_transformer.py").write_text(
                '''    threshold: float = 0.5\n'''
                '''parser.add_argument("--det-threshold", type=float, default=0.99,\n'''
                '''                             "threshold keeps precision up. Sweep it for your model.")\n'''
                '''    parser.add_argument("--use-ilp", action="store_true",\n'''
                '''    cfg = PredictConfig(\n'''
                '''        det_threshold=args.det_threshold,\n'''
                '''            candidates = sorted(\n'''
                '''                [\n'''
                '''                    (probs[i, j], i, j)\n'''
                '''                    for i in range(n_src)\n'''
                '''                    for j in range(n_tgt)\n'''
                '''                    if probs[i, j] > cfg.threshold\n'''
                '''                ],\n'''
                '''                reverse=True,\n'''
                '''            )\n'''
                '''                dist = float(np.linalg.norm(\n'''
                '''                    coords_so_far[gi, 1:].astype(np.float32)\n'''
                '''                    - coords_so_far[gj, 1:].astype(np.float32)\n'''
                '''                ))\n'''
                '''        # Detection TTA: original + flip-x + flip-y + flip-xy, average logits.\n'''
                '''        # TTA: flip along Y (-2) and X (-1) only.  Z is excluded because\n'''
                '''        # the data is highly anisotropic (Z resolution ~4x coarser than XY),\n'''
                '''        # so Z-flips would produce out-of-distribution inputs.\n'''
                '''        if cfg.det_tta:\n'''
                '''            tta_flips = [(-1,), (-2,), (-2, -1)]\n'''
                '''            for dims in tta_flips:\n'''
                '''                imgs_flip = imgs.flip(dims)\n'''
                '''                _, det_flip = model.encode(imgs_flip)\n'''
                '''                for f in range(W):\n'''
                '''                    det_logits[f] = det_logits[f] + det_flip[f].flip(dims)\n'''
                '''                del imgs_flip, det_flip\n'''
                '''            for f in range(W):\n'''
                '''                det_logits[f] = det_logits[f] / 4\n''',

                encoding="utf-8",
            )

            _patch_predict_script_cli(root)
            source = (scripts / "predict_unet_transformer.py").read_text(encoding="utf-8")

        self.assertIn("--pool-kernel-um", source)
        self.assertIn("pool_kernel_um=args.pool_kernel_um", source)
        self.assertIn("--edge-threshold", source)
        self.assertIn("threshold=args.edge_threshold", source)
        self.assertIn("--edge-topk-parents", source)
        self.assertIn("--edge-topk-children", source)
        self.assertIn("edge_min_threshold=args.edge_min_threshold", source)
        self.assertIn("cfg.edge_max_distance_um", source)
        self.assertIn("--det-tta-mode", source)
        self.assertIn("det_tta_mode=args.det_tta_mode", source)
        self.assertIn("dihedral8", source)


if __name__ == "__main__":
    unittest.main()

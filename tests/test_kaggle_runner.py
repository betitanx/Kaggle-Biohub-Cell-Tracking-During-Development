import os
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from biohub_lab.kaggle_runner import (
    build_classical_runner_source,
    build_h13_finetune_runner_source,
    build_support_pack_cached_holdout_source,
    build_support_pack_runner_source,
)
from biohub_lab.h13_postprocess import rescue_physical_links
from biohub_lab.validation import validate_submission

import scripts.kaggle_holdout_eval as holdout_eval
from scripts.kaggle_holdout_eval import (
    _center_variant_specs,
    _division_variant_specs,
    _disappearance_variant_specs,
    _edge_floor_variant_specs,
    _edge_radius_variant_specs,
    _parse_center_thresholds,
    _parse_division_weights,
    _parse_disappearance_weights,
    _parse_edge_floors,
    _parse_edge_radius_variants,
)
from scripts.run_private_holdout import _configured_source


ROOT = Path(__file__).resolve().parents[1]


class KaggleRunnerTests(unittest.TestCase):
    def test_h13_source_finetunes_from_public_unet_and_predicts_competition_test(self):
        source = build_h13_finetune_runner_source(
            ROOT,
            epochs=3,
            max_iters=200,
            holdout_fraction=0.1,
            seed=17,
        )

        self.assertIn("unet_pretrained.pth", source)
        self.assertIn("train_unet_transformer", source)
        self.assertIn("unet_weights=pretrain_path", source)
        self.assertIn("holdout_fraction", source)
        self.assertIn("competition_root / 'train'", source)
        self.assertIn("competition_root / 'test'", source)
        self.assertIn("geffs_to_csv", source)
        self.assertIn("validate_submission", source)
        self.assertIn("include_topology=True", source)
        self.assertIn('\\"max_iters\\": 200', source)
        self.assertIn('\\"seed\\": 17', source)
        compile(source, "<h13-finetune-runner>", "exec")

    def test_h13_source_supports_single_dataset_temporal_holdout(self):
        source = build_h13_finetune_runner_source(ROOT, holdout_fraction=0.5, max_frames=6)

        self.assertIn("CELLMOT_FRAME_RANGES", source)
        self.assertIn("h13_temporal_split", source)
        self.assertIn("frame_range", source)
        self.assertIn("training_dir", source)
        self.assertIn("temporal_holdout", source)
        self.assertIn("path.stem + '.geff')).exists()", source)
        compile(source, "<h13-temporal-holdout-runner>", "exec")

    def test_h13_upstream_converter_handles_empty_prediction_graphs(self):
        converter = (ROOT / "external" / "kaggle-cell-tracking-competition" / "scripts" / "geffs_to_csv.py").read_text(encoding="utf-8")

        self.assertIn("if graph.num_nodes() == 0", converter)
        self.assertIn("COLUMNS", converter)

    def test_h13_physical_link_rescue_adds_only_adjacent_one_to_one_edges(self):
        rows = [
            {"id": 0, "dataset": "sample", "row_type": "node", "node_id": 10, "t": 0, "z": 1, "y": 10, "x": 10, "source_id": -1, "target_id": -1},
            {"id": 1, "dataset": "sample", "row_type": "node", "node_id": 11, "t": 0, "z": 1, "y": 30, "x": 30, "source_id": -1, "target_id": -1},
            {"id": 2, "dataset": "sample", "row_type": "node", "node_id": 20, "t": 1, "z": 1, "y": 10, "x": 11, "source_id": -1, "target_id": -1},
            {"id": 3, "dataset": "sample", "row_type": "node", "node_id": 21, "t": 1, "z": 1, "y": 30, "x": 31, "source_id": -1, "target_id": -1},
        ]

        rescued = rescue_physical_links(rows, max_distance_um=2.0)
        edges = [row for row in rescued if row["row_type"] == "edge"]

        self.assertEqual([(row["source_id"], row["target_id"]) for row in edges], [(10, 20), (11, 21)])
        self.assertEqual([row["id"] for row in rescued], list(range(len(rescued))))

    def test_classical_source_recomputes_submission_from_competition_input(self):
        source = build_classical_runner_source(ROOT)

        self.assertIn("build_graphs", source)
        self.assertIn("/kaggle/input/competitions/biohub-cell-tracking-during-development", source)
        self.assertIn("submission.csv", source)
        self.assertIn("validate_submission", source)
        self.assertIn("include_topology=True", source)
        self.assertIn("largest_component_size", source)
        self.assertNotIn("Path('/kaggle/input').rglob", source)

    def test_support_pack_source_recomputes_with_public_model_and_ilp(self):
        source = build_support_pack_runner_source(ROOT, det_threshold=0.965, ilp_disappearance_weight=1.4)

        self.assertIn("biohub_tracking", source)
        self.assertIn("predict_unet_transformer", source)
        self.assertIn("pyscipopt", source)
        self.assertIn("edge_predictor_best.pth", source)
        self.assertIn("0.965", source)
        self.assertIn("1.4", source)
        self.assertIn("pool_kernel_um", source)
        self.assertIn("3.0", source)
        self.assertIn("ilp_appearance_weight=RUNNER_CONFIG['ilp_appearance_weight']", source)
        self.assertIn("disappearance_weight=RUNNER_CONFIG['ilp_disappearance_weight']", source)
        self.assertIn("for row_id, row in enumerate(rows):", source)
        self.assertIn("row['id'] = row_id", source)
        self.assertIn("timings_seconds", source)
        self.assertIn("prediction_seconds", source)
        self.assertNotIn("-10000", source)

    def test_support_pack_source_records_all_runtime_ablation_parameters(self):
        source = build_support_pack_runner_source(
            ROOT,
            det_threshold=0.99,
            edge_threshold=0.6,
            edge_min_threshold=0.25,
            edge_topk_parents=2,
            edge_topk_children=2,
            edge_max_distance_um=10.0,
            det_tta_mode="dihedral8",
            ilp_appearance_weight=0.2,
            ilp_disappearance_weight=1.8,
            ilp_division_weight=1.3,
            pool_kernel_um=4.5,
        )

        self.assertIn('\\"det_threshold\\": 0.99', source)
        self.assertIn('\\"edge_threshold\\": 0.6', source)
        self.assertIn('\\"edge_min_threshold\\": 0.25', source)
        self.assertIn('\\"edge_topk_parents\\": 2', source)
        self.assertIn('\\"edge_max_distance_um\\": 10.0', source)
        self.assertIn('\\"det_tta_mode\\": \\"dihedral8\\"', source)
        self.assertIn('\\"ilp_appearance_weight\\": 0.2', source)
        self.assertIn('\\"ilp_disappearance_weight\\": 1.8', source)
        self.assertIn('\\"ilp_division_weight\\": 1.3', source)
        self.assertIn('\\"pool_kernel_um\\": 4.5', source)
        self.assertIn("edge_topk_parents", source)
        self.assertIn("edge_topk_children", source)
        self.assertIn("candidate_pairs", source)
        self.assertIn("np.asarray(voxel_size", source)
        self.assertIn("tta_transforms", source)

    def test_support_pack_source_accepts_h17_physical_radius_for_live_candidate(self):
        source = build_support_pack_runner_source(
            ROOT,
            edge_min_threshold=0.35,
            edge_topk_parents=2,
            edge_topk_children=2,
            edge_max_distance_um=8.4,
            det_tta_mode="dihedral8",
            ilp_disappearance_weight=1.8,
            ilp_division_weight=0.5,
            gap_closing=True,
        )

        self.assertIn('\\"edge_max_distance_um\\": 8.4', source)
        self.assertIn("edge_max_distance_um=RUNNER_CONFIG['edge_max_distance_um']", source)
        with self.assertRaises(ValueError):
            build_support_pack_runner_source(ROOT, edge_max_distance_um=float("nan"))

    def test_support_pack_source_supports_feature_space_d4_tta(self):
        source = build_support_pack_runner_source(
            ROOT,
            feature_tta=True,
            feature_tta_blend=0.75,
        )

        self.assertIn('\\"feature_tta\\": true', source.lower())
        self.assertIn('\\"feature_tta_blend\\": 0.75', source)
        self.assertIn("feature_tta: bool = False", source)
        self.assertIn("feature_tta_blend: float = 1.0", source)
        self.assertIn("RUNNER_CONFIG['feature_tta']", source)
        self.assertIn("RUNNER_CONFIG['feature_tta_blend']", source)
        self.assertIn("feature_sum", source)
        self.assertIn("unet_out = unet_out * (1.0 - blend) + tta_mean * blend", source)
        compile(source, "<feature-tta-runner>", "exec")

    def test_support_pack_source_defaults_feature_space_tta_off(self):
        source = build_support_pack_runner_source(ROOT)

        self.assertIn('\\"feature_tta\\": false', source.lower())
        self.assertIn('\\"feature_tta_blend\\": 1.0', source)
        self.assertNotIn('\\"metric_probe\\": true', source.lower())
        compile(source, "<default-feature-tta-runner>", "exec")

    def test_support_pack_source_rejects_invalid_feature_tta_blend(self):
        with self.assertRaises(ValueError):
            build_support_pack_runner_source(ROOT, feature_tta_blend=-0.01)
        with self.assertRaises(ValueError):
            build_support_pack_runner_source(ROOT, feature_tta_blend=1.01)

    def test_support_pack_source_can_enable_motion_consensus(self):
        source = build_support_pack_runner_source(
            ROOT,
            motion_consensus=True,
            motion_consensus_max_distance_um=7.0,
            motion_consensus_max_prediction_error_um=3.0,
            motion_consensus_require_unique_parent=True,
            motion_consensus_max_out_degree=2,
        )

        self.assertIn('\\"motion_consensus\\": true', source.lower())
        self.assertIn('\\"motion_consensus_max_distance_um\\": 7.0', source)
        self.assertIn('\\"motion_consensus_max_prediction_error_um\\": 3.0', source)
        self.assertIn("RUNNER_CONFIG['motion_consensus']", source)
        self.assertIn("MOTION_CONSENSUS_REPORT", source)
        self.assertIn("def add_motion_consensus_links(", source)
        compile(source, "<motion-consensus-runner>", "exec")

    def test_support_pack_source_defaults_motion_consensus_off(self):
        source = build_support_pack_runner_source(ROOT)

        self.assertIn('\\"motion_consensus\\": false', source.lower())
        self.assertNotIn("MOTION_CONSENSUS_REPORT", source)
        compile(source, "<default-motion-consensus-runner>", "exec")

    def test_support_pack_source_rejects_invalid_motion_consensus_parameters(self):
        with self.assertRaises(ValueError):
            build_support_pack_runner_source(ROOT, motion_consensus_max_distance_um=-0.01)
        with self.assertRaises(ValueError):
            build_support_pack_runner_source(ROOT, motion_consensus_max_prediction_error_um=float("nan"))
        with self.assertRaises(ValueError):
            build_support_pack_runner_source(ROOT, motion_consensus_max_out_degree=3)

    def test_metric_probe_source_is_explicitly_separate_from_strict_runner(self):
        strict = build_support_pack_runner_source(ROOT)
        probe = build_support_pack_runner_source(
            ROOT,
            metric_probe=True,
            metric_probe_max_components=1200,
            metric_probe_forks=5,
        )

        self.assertNotIn("'metric_probe': True", strict)
        self.assertIn('\\"metric_probe\\": true', probe.lower())
        self.assertIn('\\"metric_probe_max_components\\": 1200', probe)
        self.assertIn('\\"metric_probe_forks\\": 5', probe)
        self.assertIn("RUNNER_CONFIG['metric_probe']", probe)
        self.assertIn("-10000", probe)

    def test_support_pack_source_can_attach_legal_gap_closing_without_metric_probe(self):
        source = build_support_pack_runner_source(ROOT, gap_closing=True)

        self.assertIn("GapClosingConfig", source)
        self.assertIn("postprocess_submission", source)
        self.assertIn("RUNNER_CONFIG['gap_closing']", source)
        self.assertNotIn("-10000", source)

    def test_cached_holdout_source_reuses_one_inference_for_edge_floor_variants(self):
        source = build_support_pack_cached_holdout_source(
            ROOT,
            edge_floors=(0.45, 0.55),
            expected_timepoints=100,
        )

        self.assertEqual(source.count("coords, raw_edges = predict_video("), 1)
        self.assertIn("float(edge[2]) >= float(floor)", source)
        self.assertIn('"label": "edge045"', source)
        self.assertIn('"label": "edge055"', source)
        self.assertIn("'submission_' + label + '.csv'", source)
        self.assertIn("validate_submission", source)
        self.assertIn("include_topology=True", source)
        self.assertIn("'edge_floor_inference': 0.45", source)
        compile(source, "<cached-holdout-runner>", "exec")

    def test_holdout_disappearance_variants_are_explicit_and_deterministic(self):
        weights = _parse_disappearance_weights("1.0,1.8")

        self.assertEqual(weights, (1.0, 1.8))
        self.assertEqual(
            _disappearance_variant_specs(weights),
            (("disappear100", 1.0), ("disappear180", 1.8)),
        )

    def test_holdout_division_variants_are_explicit_and_deterministic(self):
        weights = _parse_division_weights("0.25,1.0")

        self.assertEqual(weights, (0.25, 1.0))
        self.assertEqual(
            _division_variant_specs(weights),
            (("division025", 0.25), ("division100", 1.0)),
        )

    def test_holdout_division_weights_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            _parse_division_weights("0.25,0.25")
        with self.assertRaises(ValueError):
            _parse_division_weights("-0.1,1.0")

    def test_holdout_edge_floor_variants_are_explicit_and_deterministic(self):
        floors = _parse_edge_floors("0.30,0.40")

        self.assertEqual(floors, (0.30, 0.40))
        self.assertEqual(
            _edge_floor_variant_specs(floors),
            (("edge030", 0.30), ("edge040", 0.40)),
        )

    def test_holdout_edge_floors_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            _parse_edge_floors("0.30,0.30")
        with self.assertRaises(ValueError):
            _parse_edge_floors("-0.1,0.4")
        with self.assertRaises(ValueError):
            _parse_edge_floors("0.4,1.1")

    def test_holdout_edge_radius_variants_are_physical_and_deterministic(self):
        radii = _parse_edge_radius_variants("8.4,9.0")

        self.assertEqual(radii, (8.4, 9.0))
        self.assertEqual(
            _edge_radius_variant_specs(radii),
            (("radius084", 8.4), ("radius090", 9.0)),
        )

    def test_holdout_edge_radius_variants_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            _parse_edge_radius_variants("8.4")
        with self.assertRaises(ValueError):
            _parse_edge_radius_variants("8.4,8.4")
        with self.assertRaises(ValueError):
            _parse_edge_radius_variants("0.0,9.0")

    def test_main_fails_missing_gap_closing_package_before_support_pack_lookup(self):
        with tempfile.TemporaryDirectory() as directory:
            working_dir = Path(directory) / "holdout"
            with (
                patch.object(holdout_eval, "WORKING_DIR", working_dir),
                patch.dict(holdout_eval.RUNNER_CONFIG, {"gap_closing": True}),
                patch.object(
                    holdout_eval,
                    "_find_gap_closing_package",
                    side_effect=FileNotFoundError("gap-closing package missing"),
                ),
                patch.object(
                    holdout_eval,
                    "_find_support_pack",
                    side_effect=AssertionError("inference setup started before preflight"),
                ),
            ):
                with self.assertRaisesRegex(FileNotFoundError, "gap-closing package missing"):
                    holdout_eval.main()

    def test_gap_closing_preflight_is_skipped_when_gap_closing_is_disabled(self):
        with (
            patch.dict(holdout_eval.RUNNER_CONFIG, {"gap_closing": False}),
            patch.object(holdout_eval, "_find_gap_closing_package") as package_lookup,
        ):
            self.assertIsNone(holdout_eval._preflight_gap_closing_package())
        package_lookup.assert_not_called()

    def test_combined_holdout_submission_uses_globally_contiguous_row_ids(self):
        rows = []
        for dataset in ("embryo_a", "embryo_b"):
            rows.extend(
                [
                    {
                        "id": 0,
                        "dataset": dataset,
                        "row_type": "node",
                        "node_id": 0,
                        "t": 0,
                        "z": 1,
                        "y": 2,
                        "x": 3,
                        "source_id": -1,
                        "target_id": -1,
                    },
                    {
                        "id": 1,
                        "dataset": dataset,
                        "row_type": "node",
                        "node_id": 1,
                        "t": 1,
                        "z": 1,
                        "y": 2,
                        "x": 4,
                        "source_id": -1,
                        "target_id": -1,
                    },
                    {
                        "id": 2,
                        "dataset": dataset,
                        "row_type": "edge",
                        "node_id": -1,
                        "t": -1,
                        "z": -1,
                        "y": -1,
                        "x": -1,
                        "source_id": 0,
                        "target_id": 1,
                    },
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            submission = Path(directory) / "submission.csv"
            holdout_eval._write_submission_rows(rows, submission)
            report = validate_submission(
                submission,
                expected_datasets={"embryo_a", "embryo_b"},
                expected_timepoints=range(2),
                include_topology=True,
            )

        self.assertTrue(report.valid, report.errors)
        self.assertEqual(report.node_count, 4)
        self.assertEqual(report.edge_count, 2)
        self.assertEqual(report.max_out_degree, 1)
        self.assertEqual(report.nonadjacent_edge_count, 0)

    def test_holdout_deepcenter_variants_are_explicit_and_deterministic(self):
        thresholds = _parse_center_thresholds("0.40,0.35")

        self.assertEqual(thresholds, (0.40, 0.35))
        self.assertEqual(
            _center_variant_specs(thresholds),
            (("deepcenter040", 0.40), ("deepcenter035", 0.35)),
        )

    def test_holdout_deepcenter_thresholds_reject_invalid_values(self):
        with self.assertRaises(ValueError):
            _parse_center_thresholds("0.40")
        with self.assertRaises(ValueError):
            _parse_center_thresholds("0.40,0.40")
        with self.assertRaises(ValueError):
            _parse_center_thresholds("-0.1,0.4")

    def test_holdout_launcher_embeds_edge_floor_ablation(self):
        output_root = ROOT / "outputs" / "test-holdout-edge-floor"
        configured = _configured_source(
            ROOT / "scripts" / "kaggle_holdout_eval.py",
            output_root,
            None,
            base_disappearance_weight=1.8,
            edge_floors="0.30,0.40",
        )

        configured_source = configured.read_text(encoding="utf-8")
        self.assertIn("BIOHUB_HOLDOUT_EDGE_FLOORS", configured_source)
        self.assertIn("0.30,0.40", configured_source)
        compile(configured_source, str(configured), "exec")

    def test_holdout_launcher_embeds_edge_radius_ablation(self):
        output_root = ROOT / "outputs" / "test-holdout-edge-radius"
        configured = _configured_source(
            ROOT / "scripts" / "kaggle_holdout_eval.py",
            output_root,
            None,
            base_disappearance_weight=1.8,
            edge_radius_variants="8.4,9.0",
        )

        configured_source = configured.read_text(encoding="utf-8")
        self.assertIn("BIOHUB_HOLDOUT_EDGE_RADIUS_VARIANTS", configured_source)
        self.assertIn("BIOHUB_HOLDOUT_BASE_DISAPPEARANCE_WEIGHT", configured_source)
        self.assertIn("1.8", configured_source)
        self.assertIn("8.4,9.0", configured_source)
        self.assertIn("edge_max_distance_um", configured_source)
        self.assertIn("float(edge[3]) <= edge_radius", configured_source)
        compile(configured_source, str(configured), "exec")

    def test_holdout_launcher_rejects_edge_radius_without_parent_disappearance_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluator.py"
            source.write_text("print('evaluator')\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()

            with self.assertRaisesRegex(ValueError, "base disappearance weight"):
                _configured_source(
                    source,
                    output,
                    None,
                    edge_radius_variants="8.4,9.0",
                )

    def test_holdout_evaluator_rejects_edge_radius_env_without_parent_weight(self):
        environment = os.environ.copy()
        environment.pop("BIOHUB_HOLDOUT_BASE_DISAPPEARANCE_WEIGHT", None)
        environment["BIOHUB_HOLDOUT_EDGE_RADIUS_VARIANTS"] = "8.4,9.0"

        completed = subprocess.run(
            [sys.executable, "-c", "import scripts.kaggle_holdout_eval"],
            cwd=str(ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "edge-radius variants require BIOHUB_HOLDOUT_BASE_DISAPPEARANCE_WEIGHT",
            completed.stderr,
        )

    def test_holdout_launcher_embeds_detection_tta_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluator.py"
            source.write_text("print('evaluator')\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()

            configured = _configured_source(
                source,
                output,
                None,
                det_tta_mode="flip4",
            )
            generated = configured.read_text(encoding="utf-8")

        self.assertIn("BIOHUB_HOLDOUT_DET_TTA_MODE", generated)
        self.assertIn("flip4", generated)
        compile(generated, str(configured), "exec")

    def test_holdout_launcher_embeds_division_weight_ablation(self):
        output_root = ROOT / "outputs" / "test-holdout-division"
        configured = _configured_source(
            ROOT / "scripts" / "kaggle_holdout_eval.py",
            output_root,
            None,
            "0.25,1.0",
            base_disappearance_weight=1.8,
        )

        self.assertEqual(configured, output_root / "kaggle_holdout_eval_configured.py")
        configured_source = configured.read_text(encoding="utf-8")
        self.assertIn("BIOHUB_HOLDOUT_DIVISION_WEIGHTS", configured_source)
        self.assertIn("BIOHUB_HOLDOUT_BASE_DISAPPEARANCE_WEIGHT", configured_source)
        compile(configured_source, str(configured), "exec")

    def test_holdout_disappearance_variants_reject_invalid_weights(self):
        with self.assertRaises(ValueError):
            _parse_disappearance_weights("1.0,1.0")
        with self.assertRaises(ValueError):
            _parse_disappearance_weights("-0.1,1.8")

    def test_private_holdout_launcher_embeds_variant_config_in_kernel_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluator.py"
            source.write_text("print('evaluator')\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()

            configured = _configured_source(source, output, "1.0,1.8")

            generated = configured.read_text(encoding="utf-8")

        self.assertIn("BIOHUB_HOLDOUT_DISAPPEARANCE_WEIGHTS", generated)
        self.assertIn("1.0,1.8", generated)
        self.assertTrue(generated.endswith("print('evaluator')\n"))

    def test_private_holdout_launcher_embeds_feature_space_tta_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluator.py"
            source.write_text("print('evaluator')\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()

            configured = _configured_source(
                source,
                output,
                None,
                feature_tta=True,
                feature_tta_blend=0.75,
            )
            generated = configured.read_text(encoding="utf-8")

        self.assertIn("BIOHUB_HOLDOUT_FEATURE_TTA", generated)
        self.assertIn("BIOHUB_HOLDOUT_FEATURE_TTA_BLEND", generated)
        self.assertIn("0.75", generated)
        compile(generated, str(configured), "exec")

    def test_private_holdout_launcher_embeds_motion_consensus_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluator.py"
            source.write_text("print('evaluator')\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()

            configured = _configured_source(
                source,
                output,
                None,
                motion_consensus=True,
                motion_consensus_max_distance_um=7.0,
                motion_consensus_max_prediction_error_um=3.0,
                motion_consensus_require_unique_parent=True,
                motion_consensus_max_out_degree=2,
            )
            generated = configured.read_text(encoding="utf-8")

        self.assertIn("BIOHUB_HOLDOUT_MOTION_CONSENSUS", generated)
        self.assertIn("BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_DISTANCE_UM", generated)
        self.assertIn("BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_PREDICTION_ERROR_UM", generated)
        self.assertIn("BIOHUB_HOLDOUT_MOTION_CONSENSUS_REQUIRE_UNIQUE_PARENT", generated)
        self.assertIn("BIOHUB_HOLDOUT_MOTION_CONSENSUS_MAX_OUT_DEGREE", generated)
        self.assertIn("7.0", generated)
        self.assertIn("3.0", generated)
        compile(generated, str(configured), "exec")

    def test_private_holdout_launcher_embeds_deepcenter_config_and_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "evaluator.py"
            source.write_text("print('evaluator')\n", encoding="utf-8")
            output = root / "output"
            output.mkdir()

            configured = _configured_source(
                source,
                output,
                None,
                deepcenter=True,
                deepcenter_thresholds="0.40,0.35",
                deepcenter_max_endpoint_distance_um=7.0,
                deepcenter_min_existing_distance_um=2.0,
                deepcenter_require_unique_parent=True,
                deepcenter_max_out_degree=2,
            )
            generated = configured.read_text(encoding="utf-8")

        self.assertIn("BIOHUB_HOLDOUT_DEEPCENTER", generated)
        self.assertIn("BIOHUB_HOLDOUT_DEEPCENTER_THRESHOLDS", generated)
        self.assertIn("0.40,0.35", generated)
        self.assertIn("def add_center_rescue_nodes(", generated)
        compile(generated, str(configured), "exec")

    def test_kaggle_train_holdout_runner_rejects_incomplete_ground_truth_scope(self):
        source = (ROOT / "scripts" / "kaggle_holdout_eval.py").read_text(encoding="utf-8")

        self.assertIn("complete_time_scope", source)
        self.assertIn("gt_timepoints_within_image_scope", source)
        self.assertIn("metadata_scope_consistent", source)
        self.assertIn("frame_count == EXPECTED_TIMEPOINTS", source)
        self.assertIn("holdout_metrics.json", source)
        self.assertIn("competition_submit_code", source)
        self.assertNotIn("api.competition_submit_code", source)
        self.assertIn("BIOHUB_HOLDOUT_DET_TTA_MODE", source)
        self.assertIn("HOLDOUT_WINDOW_PROGRESS", source)
        compile(source, "<kaggle-train-holdout-runner>", "exec")

    def test_classical_runner_executes_against_a_local_ome_zarr(self):
        try:
            import zarr  # type: ignore
        except ImportError as exc:  # pragma: no cover - the project ML env has zarr.
            self.skipTest("zarr is required for the runner integration smoke test: {}".format(exc))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            competition_dir = root / "competition"
            test_dir = competition_dir / "test"
            test_dir.mkdir(parents=True)
            array_path = test_dir / "sample.zarr"
            array = zarr.open(
                str(array_path),
                mode="w",
                shape=(3, 3, 7, 7),
                dtype="f4",
            )
            array[:] = 0.0
            array[0, 1, 3, 3] = 100.0
            array[1, 1, 3, 4] = 100.0
            array[2, 1, 3, 5] = 100.0
            working_dir = root / "working"
            source_path = root / "runner.py"
            source_path.write_text(build_classical_runner_source(ROOT), encoding="utf-8")

            environment = dict(os.environ)
            environment.update(
                {
                    "BIOHUB_COMPETITION_DIR": str(competition_dir),
                    "BIOHUB_WORKING_DIR": str(working_dir),
                    "BIOHUB_EXPECTED_TIMEPOINTS": "3",
                    "BIOHUB_RUN_WORKERS": "1",
                }
            )
            completed = subprocess.run(
                [sys.executable, str(source_path)],
                cwd=str(root),
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = validate_submission(
                working_dir / "submission.csv",
                expected_datasets={"sample"},
                expected_timepoints={0, 1, 2},
            )
            self.assertTrue(report.valid, report.errors)
            self.assertGreaterEqual(report.node_count, 3)


if __name__ == "__main__":
    unittest.main()

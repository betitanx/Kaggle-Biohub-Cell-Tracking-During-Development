import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from biohub_lab.runner import load_config, run_experiment
from biohub_lab.proposals import expand_experiment_specs


ROOT = Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_config_loads_candidates_and_disabled_candidates_are_kept(self):
        config = load_config(ROOT / "config" / "experiments.json")

        self.assertEqual(config["competition"], "biohub-cell-tracking-during-development")
        self.assertTrue(any(item["enabled"] for item in config["experiments"]))
        self.assertTrue(any(not item["enabled"] for item in config["experiments"]))
        synthetic = next(item for item in config["experiments"] if item["name"] == "synthetic_3d_quality_gate")
        self.assertFalse(synthetic["live_eligible"])

    def test_catalog_expands_scientific_proposals_and_uses_adapters(self):
        config = load_config(ROOT / "config" / "experiments.json")
        expanded = expand_experiment_specs(config["experiments"])

        classical = [item for item in expanded if item["family"] == "classical"]
        cellmot = [item for item in expanded if item["family"] == "deep_baseline"]
        self.assertEqual(len(classical), 4)
        self.assertEqual(len(cellmot), 2)
        self.assertTrue(all("biohub_lab.classical_candidate" in item["command"] for item in classical))
        self.assertTrue(all("biohub_lab.cellmot_candidate" in item["command"] for item in cellmot))
        self.assertTrue(all(item["notebook_runner_kind"] == "support_pack" for item in cellmot))
        self.assertTrue(
            all(
                item["notebook_additional_dataset_sources"]
                == ["pilkwang/biohub-tracking-support-pack-50ep-v1"]
                for item in cellmot
            )
        )
        self.assertEqual(cellmot[0]["notebook_runner_options"]["pool_kernel_um"], 3.0)
        det099 = next(item for item in cellmot if item["variant"]["name"] == "det099_ilp")
        self.assertEqual(det099["notebook_runner_options"]["det_threshold"], 0.99)
        topology = [item for item in expanded if item["family"] == "topology_aware"]
        self.assertEqual(len(topology), 2)
        self.assertTrue(all("--infer-divisions" in item["command"] for item in topology))
        postprocess = [item for item in expanded if item["family"] == "postprocess"]
        self.assertEqual(len(postprocess), 2)
        self.assertTrue(all("--filter-isolated-nodes" in item["command"] for item in postprocess))

    def test_community_cycle_keeps_inference_and_metric_probe_parameters_explicit(self):
        config = load_config(ROOT / "config" / "community-cycle.json")
        expanded = expand_experiment_specs(config["experiments"])

        self.assertEqual(len(expanded), 2)
        self.assertTrue(all(item["notebook_runner_kind"] == "support_pack_metric_probe" for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"]["edge_topk_parents"] == 2 for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"]["edge_min_threshold"] == 0.25 for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"]["edge_max_distance_um"] == 10.0 for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"]["det_tta_mode"] == "dihedral8" for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"]["gap_closing"] is True for item in expanded))
        self.assertTrue(
            all(
                item["notebook_additional_dataset_sources"]
                == [
                    "pilkwang/biohub-tracking-support-pack-50ep-v1",
                    "yoikoarmor/biohub-last-call-postprocess-v1",
                ]
                for item in expanded
            )
        )
        edge_variant = next(item for item in expanded if item["variant"]["name"] == "v7_edge_048_tta_rescue")
        self.assertEqual(edge_variant["notebook_runner_options"]["edge_threshold"], 0.48)
        self.assertIn("--edge-topk-parents", edge_variant["command"])

    def test_community_clean_cycle_excludes_metric_probe_and_keeps_legal_gap_closing(self):
        config = load_config(ROOT / "config" / "community-clean-cycle.json")
        expanded = expand_experiment_specs(config["experiments"])

        self.assertEqual(len(expanded), 2)
        self.assertTrue(all(item["notebook_runner_kind"] == "support_pack" for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"].get("metric_probe") is False for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"]["gap_closing"] is True for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"]["edge_topk_parents"] == 2 for item in expanded))
        self.assertTrue(all(item["notebook_runner_options"]["edge_max_distance_um"] == 10.0 for item in expanded))
        self.assertTrue(all("no-metric-probe" in item["tags"] for item in expanded))

    def test_runner_persists_logs_metrics_and_validated_artifact(self):
        spec = {
            "name": "runner_test",
            "family": "smoke",
            "command": ["${PYTHON}", "-m", "biohub_lab.demo_candidate", "--run-dir", "${RUN_DIR}", "--variant", "baseline"],
            "timeout_seconds": 30,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(spec, Path(directory), ROOT)
            run_dir = Path(result["run_dir"])

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["artifact_valid"])
            self.assertTrue((run_dir / "stdout.log").is_file())
            self.assertTrue((run_dir / "events.jsonl").is_file())
            self.assertTrue((run_dir / "result.json").is_file())
            self.assertTrue((run_dir / "metrics.json").is_file())
            self.assertEqual(json.loads((run_dir / "result.json").read_text())["experiment"], "runner_test")

    def test_runner_persists_live_eligibility_for_safety(self):
        spec = {
            "name": "non_live_runner_test",
            "family": "synthetic_quality",
            "live_eligible": False,
            "command": ["${PYTHON}", "-m", "biohub_lab.synthetic_benchmark", "--run-dir", "${RUN_DIR}"],
            "timeout_seconds": 30,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(spec, Path(directory), ROOT)

        self.assertFalse(result["live_eligible"])

    def test_subset_scope_cannot_be_live_eligible(self):
        spec = {
            "name": "subset_runner_test",
            "family": "classical",
            "live_eligible": True,
            "command": ["${PYTHON}", "-m", "biohub_lab.demo_candidate", "--run-dir", "${RUN_DIR}", "--variant", "baseline"],
            "timeout_seconds": 30,
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_experiment(spec, Path(directory), ROOT, env={"BIOHUB_DATA_SCOPE": "subset"})

        self.assertFalse(result["live_eligible"])
        self.assertEqual(result["data_scope"], "subset")

    def test_relative_output_dir_does_not_duplicate_run_path(self):
        spec = {
            "name": "relative_runner_test",
            "family": "smoke",
            "command": ["${PYTHON}", "-m", "biohub_lab.demo_candidate", "--run-dir", "${RUN_DIR}", "--variant", "baseline"],
            "timeout_seconds": 30,
        }
        relative_dir = Path(".runner-relative-test")
        previous_cwd = os.getcwd()
        try:
            os.chdir(ROOT)
            result = run_experiment(spec, relative_dir, ROOT)
            self.assertTrue(result["artifact_valid"])
            self.assertTrue(Path(result["artifact"]).is_file())
        finally:
            os.chdir(previous_cwd)
            shutil.rmtree(ROOT / relative_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

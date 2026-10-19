import json
import tempfile
import unittest
from pathlib import Path

from biohub_lab.openresearch import (
    OpenResearchBridgeError,
    build_openresearch_proposals,
    parse_fixed_run_command,
    write_openresearch_proposals,
)


FIXED = ["/project/.venv-biohub/bin/python", "-m", "biohub_lab.kaggle_runner"]


class OpenResearchBridgeTests(unittest.TestCase):
    def test_converts_planned_node_without_making_it_live(self):
        payload = {
            "project": {"id": "orx-project", "name": "Biohub research", "runCommand": FIXED},
            "experiments": [
                {
                    "id": "orx-exp-1",
                    "parentId": "55950639",
                    "title": "D4 feature TTA",
                    "status": "planned",
                    "description": "Test feature-space D4 TTA on top of the best clean H9 base.",
                    "biohub": {
                        "runnerKind": "support_pack",
                        "notebookRunnerOptions": {
                            "det_tta_mode": "dihedral8",
                            "edge_topk_children": 2,
                            "metric_probe": False,
                        },
                    },
                }
            ],
        }

        result = build_openresearch_proposals(payload, fixed_run_command=FIXED, imported_at="2026-09-14T00:00:00Z")

        self.assertEqual(result["counts"], {"input_nodes": 1, "proposals": 1, "blocked": 0})
        proposal = result["proposals"][0]
        self.assertFalse(proposal["enabled"])
        self.assertFalse(proposal["live_eligible"])
        self.assertTrue(proposal["proposed_only"])
        self.assertTrue(proposal["execution_ready"])
        self.assertEqual(proposal["command"], FIXED)
        self.assertEqual(proposal["notebook_runner_options"]["det_tta_mode"], "dihedral8")
        self.assertEqual(proposal["openresearch"]["parent_experiment_id"], "55950639")
        self.assertEqual(result["contract"]["metric_probe"], False)

    def test_blocks_run_contract_overrides_probe_and_answered_nodes(self):
        payload = {
            "experiments": [
                {
                    "id": "bad-command",
                    "title": "Command override",
                    "status": "planned",
                    "runCommand": ["python", "other_runner.py"],
                    "environment": {"CUDA_VISIBLE_DEVICES": "0"},
                },
                {
                    "id": "probe",
                    "title": "Probe",
                    "status": "planned",
                    "changes": {"metric_probe": True},
                },
                {
                    "id": "answered",
                    "title": "Answered",
                    "status": "completed",
                },
            ]
        }

        result = build_openresearch_proposals(payload, fixed_run_command=FIXED)

        self.assertEqual(result["counts"]["proposals"], 0)
        self.assertEqual(result["counts"]["blocked"], 3)
        reasons = " ".join(" ".join(item["reasons"]) for item in result["blocked"])
        self.assertIn("fixed run command", reasons)
        self.assertIn("fixed execution environment", reasons)
        self.assertIn("metric_probe=true", reasons)
        self.assertIn("answered/frozen", reasons)

    def test_unknown_public_lead_options_remain_research_only(self):
        payload = {
            "experiments": [
                {
                    "id": "lead",
                    "title": "Harmonic association",
                    "status": "draft",
                    "changes": {
                        "harmonic_bidirectional_weight": 0.15,
                        "det_tta_mode": "dihedral8",
                    },
                }
            ]
        }

        result = build_openresearch_proposals(payload, fixed_run_command=FIXED)

        proposal = result["proposals"][0]
        self.assertFalse(proposal["execution_ready"])
        self.assertEqual(proposal["notebook_runner_options"], {"det_tta_mode": "dihedral8"})
        self.assertEqual(proposal["openresearch"]["unsupported_option_keys"], ["harmonic_bidirectional_weight"])
        self.assertTrue(any("harmonic_bidirectional_weight" in warning for warning in result["warnings"]))

    def test_duplicate_names_are_blocked_and_write_is_json(self):
        payload = {"experiments": [{"id": "one", "title": "Same idea", "status": "planned"}]}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "orx.json"
            output = Path(directory) / "outputs" / "bridge.json"
            source.write_text(json.dumps(payload), encoding="utf-8")
            result = write_openresearch_proposals(
                source,
                output,
                fixed_run_command=FIXED,
                existing_names=["openresearch__Same-idea"],
            )

            self.assertTrue(output.is_file())
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["counts"], result["counts"])
            self.assertEqual(result["counts"]["blocked"], 1)

    def test_fixed_command_rejects_shell_execution(self):
        with self.assertRaises(OpenResearchBridgeError):
            parse_fixed_run_command("python runner.py && rm -rf outputs")


if __name__ == "__main__":
    unittest.main()

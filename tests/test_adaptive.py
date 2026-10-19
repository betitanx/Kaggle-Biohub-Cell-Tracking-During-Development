import unittest
import json
import tempfile
from pathlib import Path

from biohub_lab.adaptive import propose_next_experiments
from biohub_lab.controller import write_proposal_queue


BASE = {
    "name": "classical_peak_link",
    "family": "classical",
    "enabled": True,
    "command": ["python", "candidate"],
    "hypothesis": "base",
    "tags": ["classical"],
    "adaptive_variants": [
        {"name": "weaker_wide", "direction": "weaker", "args": ["--quantile", "0.98"]},
        {"name": "precision_tight", "direction": "precision", "args": ["--quantile", "0.999"]},
    ],
}


class AdaptiveProposalTests(unittest.TestCase):
    def test_promising_holdout_prioritises_weaker_hypothesis(self):
        proposals = propose_next_experiments(
            [BASE],
            [
                {
                    "experiment": "classical_peak_link__q995_d7",
                    "family": "classical",
                    "status": "completed",
                    "artifact_valid": True,
                    "local_score": 0.8,
                    "in_domain_score": 0.8,
                    "holdout_score": 0.76,
                }
            ],
            max_proposals=1,
        )

        self.assertEqual(proposals[0]["name"], "classical_peak_link__adaptive__weaker_wide")
        self.assertEqual(proposals[0]["proposal_reason"], "promising_holdout_test_weaker_hypothesis")

    def test_non_promising_holdout_prioritises_precision_and_skips_completed(self):
        base = dict(BASE)
        base["adaptive_variants"] = [
            {"name": "precision_tight", "direction": "precision", "args": ["--quantile", "0.999"]},
            {"name": "precision_tighter", "direction": "precision", "args": ["--quantile", "0.9995"]},
            BASE["adaptive_variants"][0],
        ]
        proposals = propose_next_experiments(
            [base],
            [
                {
                    "experiment": "classical_peak_link__adaptive__precision_tight",
                    "family": "classical",
                    "status": "completed",
                    "artifact_valid": True,
                    "local_score": 0.9,
                    "in_domain_score": 0.95,
                    "holdout_score": 0.2,
                }
            ],
            max_proposals=2,
        )

        self.assertEqual(proposals[0]["name"], "classical_peak_link__adaptive__precision_tighter")
        self.assertEqual(proposals[1]["name"], "classical_peak_link__adaptive__weaker_wide")
        self.assertEqual(proposals[0]["proposal_reason"], "overfit_proxy_test_precision_hypothesis")

    def test_proposal_queue_is_persisted_as_machine_readable_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            payload = write_proposal_queue(
                {"adaptive": {"enabled": True, "max_proposals_per_cycle": 1}, "experiments": [BASE]},
                [],
                Path(directory),
            )
            saved = json.loads((Path(directory) / "proposals.json").read_text(encoding="utf-8"))

        self.assertEqual(payload["proposals"], saved["proposals"])
        self.assertEqual(saved["proposals"][0]["proposal_reason"], "bootstrap_test_weaker_hypothesis")


if __name__ == "__main__":
    unittest.main()

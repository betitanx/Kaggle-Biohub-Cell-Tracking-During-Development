import unittest

from biohub_lab.proposals import expand_experiment_specs


class ProposalTests(unittest.TestCase):
    def test_expands_small_variant_grid_without_mutating_base_spec(self):
        base = {
            "name": "classical",
            "family": "classical",
            "enabled": True,
            "command": ["python", "candidate.py"],
            "variants": [
                {"name": "q99", "args": ["--quantile", "0.99"]},
                {"name": "q995", "args": ["--quantile", "0.995"]},
            ],
        }

        expanded = expand_experiment_specs([base])

        self.assertEqual([item["name"] for item in expanded], ["classical__q99", "classical__q995"])
        self.assertEqual(expanded[0]["command"][-2:], ["--quantile", "0.99"])
        self.assertEqual(base["name"], "classical")

    def test_specs_without_variants_pass_through_as_copies(self):
        base = {"name": "single", "command": ["python"]}

        expanded = expand_experiment_specs([base])

        self.assertEqual(expanded, [base])
        self.assertIsNot(expanded[0], base)


if __name__ == "__main__":
    unittest.main()

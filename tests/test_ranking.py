import unittest
import math

from biohub_lab.ranking import rank_records, select_submission_candidates


class RankingTests(unittest.TestCase):
    def test_holdout_generalization_beats_train_only_score(self):
        records = [
            {
                "experiment": "overfit",
                "family": "deep",
                "status": "completed",
                "local_score": 0.94,
                "in_domain_score": 0.98,
                "holdout_score": 0.51,
                "artifact": "overfit.csv",
            },
            {
                "experiment": "weak",
                "family": "global",
                "status": "completed",
                "local_score": 0.83,
                "in_domain_score": 0.84,
                "holdout_score": 0.80,
                "artifact": "weak.csv",
            },
        ]

        ranked = rank_records(records)

        self.assertEqual(ranked[0]["experiment"], "weak")
        self.assertGreater(ranked[0]["weakness_proxy"], ranked[1]["weakness_proxy"])
        self.assertEqual(ranked[0]["decision"], "promising_proxy")
        self.assertEqual(ranked[1]["decision"], "not_promising_proxy")

    def test_failed_and_invalid_records_are_not_selected(self):
        records = [
            {
                "experiment": "failed",
                "family": "deep",
                "status": "failed",
                "local_score": 1.0,
                "artifact": "failed.csv",
            },
            {
                "experiment": "invalid",
                "family": "classic",
                "status": "completed",
                "artifact_valid": False,
                "local_score": 0.99,
                "artifact": "invalid.csv",
            },
        ]

        self.assertEqual(rank_records(records), [])

    def test_no_holdout_result_is_visible_but_not_submission_eligible(self):
        ranked = rank_records(
            [
                {
                    "experiment": "train_only",
                    "family": "deep",
                    "status": "completed",
                    "local_score": 0.99,
                    "in_domain_score": 0.99,
                    "artifact": "train_only.csv",
                }
            ]
        )

        self.assertEqual(ranked[0]["decision"], "inconclusive")
        self.assertEqual(select_submission_candidates(ranked, limit=1), [])

    def test_selection_prefers_family_diversity_before_fill(self):
        records = [
            {"experiment": "a1", "family": "deep", "selection_score": 0.95, "artifact": "a1.csv"},
            {"experiment": "a2", "family": "deep", "selection_score": 0.94, "artifact": "a2.csv"},
            {"experiment": "b1", "family": "global", "selection_score": 0.80, "artifact": "b1.csv"},
        ]

        selected = select_submission_candidates(records, limit=2)

        self.assertEqual([item["experiment"] for item in selected], ["a1", "b1"])

    def test_selection_excludes_candidates_marked_non_live(self):
        records = [
            {
                "experiment": "synthetic",
                "family": "synthetic_quality",
                "status": "completed",
                "artifact_valid": True,
                "live_eligible": False,
                "artifact": "/tmp/synthetic.csv",
                "local_score": 0.99,
                "in_domain_score": 0.99,
                "holdout_score": 0.99,
            },
            {
                "experiment": "real",
                "family": "classical",
                "status": "completed",
                "artifact_valid": True,
                "live_eligible": True,
                "artifact": "/tmp/real.csv",
                "local_score": 0.70,
                "in_domain_score": 0.70,
                "holdout_score": 0.68,
            },
        ]

        selected = select_submission_candidates(rank_records(records), limit=5)

        self.assertEqual([item["experiment"] for item in selected], ["real"])

    def test_subset_real_candidates_beat_synthetic_scores_when_no_live_result_exists(self):
        ranked = rank_records(
            [
                {
                    "experiment": "smoke",
                    "family": "smoke",
                    "status": "completed",
                    "live_eligible": False,
                    "local_score": 0.99,
                    "holdout_score": 0.99,
                    "artifact": "smoke.csv",
                },
                {
                    "experiment": "real_subset",
                    "family": "classical",
                    "status": "completed",
                    "live_eligible": False,
                    "local_score": 0.20,
                    "in_domain_score": 0.20,
                    "holdout_score": 0.10,
                    "artifact": "real.csv",
                },
            ]
        )

        selected = select_submission_candidates(ranked, limit=1)

        self.assertEqual([item["experiment"] for item in selected], ["real_subset"])

    def test_missing_live_flag_does_not_promote_smoke_to_live_order(self):
        ranked = rank_records(
            [
                {
                    "experiment": "smoke_without_flag",
                    "family": "smoke",
                    "status": "completed",
                    "local_score": 0.99,
                    "holdout_score": 0.99,
                    "artifact": "smoke.csv",
                },
                {
                    "experiment": "real_non_live",
                    "family": "classical",
                    "status": "completed",
                    "live_eligible": False,
                    "local_score": 0.20,
                    "in_domain_score": 0.20,
                    "holdout_score": 0.10,
                    "artifact": "real.csv",
                },
            ]
        )

        selected = select_submission_candidates(ranked, limit=1)

        self.assertEqual([item["experiment"] for item in selected], ["real_non_live"])

    def test_selection_score_records_a_rarity_bonus_for_variant_families(self):
        ranked = rank_records(
            [
                {
                    "experiment": "common-1",
                    "family": "classical",
                    "status": "completed",
                    "local_score": 0.8,
                    "holdout_score": 0.8,
                    "artifact": "common-1.csv",
                },
                {
                    "experiment": "common-2",
                    "family": "classical",
                    "status": "completed",
                    "local_score": 0.8,
                    "holdout_score": 0.8,
                    "artifact": "common-2.csv",
                },
                {
                    "experiment": "rare",
                    "family": "ultrack",
                    "status": "completed",
                    "local_score": 0.8,
                    "holdout_score": 0.8,
                    "artifact": "rare.csv",
                },
            ]
        )

        self.assertEqual(ranked[0]["experiment"], "rare")
        self.assertGreater(ranked[0]["diversity_bonus"], ranked[1]["diversity_bonus"])

    def test_nan_scores_are_not_treated_as_available_validation(self):
        ranked = rank_records(
            [
                {
                    "experiment": "nan_holdout",
                    "family": "classical",
                    "status": "completed",
                    "local_score": 0.8,
                    "in_domain_score": 0.8,
                    "holdout_score": math.nan,
                    "artifact": "nan.csv",
                },
                {
                    "experiment": "nan_local",
                    "family": "deep",
                    "status": "completed",
                    "local_score": math.nan,
                    "holdout_score": 0.7,
                    "artifact": "nan-local.csv",
                },
            ]
        )

        self.assertEqual(len(ranked), 1)
        self.assertFalse(ranked[0]["holdout_available"])
        self.assertEqual(select_submission_candidates(ranked, limit=1), [])


if __name__ == "__main__":
    unittest.main()

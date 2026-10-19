import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from biohub_lab.budget import SubmissionBudget


class SubmissionBudgetTests(unittest.TestCase):
    def test_limit_is_counted_in_utc_day(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = SubmissionBudget(Path(directory) / "budget.json", daily_limit=2)
            first = datetime(2026, 8, 4, 1, 0, tzinfo=timezone.utc)
            second = datetime(2026, 8, 4, 2, 0, tzinfo=timezone.utc)

            budget.record("old", first)
            budget.record("new", second)

            self.assertEqual(budget.remaining(second), 0)
            self.assertEqual(budget.remaining(datetime(2026, 8, 5, 0, 1, tzinfo=timezone.utc)), 2)

    def test_reserve_does_not_exceed_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            budget = SubmissionBudget(Path(directory) / "budget.json", daily_limit=2)
            now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)

            reserved = budget.reserve(["a", "b", "c"], now)

            self.assertEqual(reserved, ["a", "b"])
            self.assertEqual(budget.remaining(now), 0)

    def test_state_is_persisted_between_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
            SubmissionBudget(path, daily_limit=5).record("one", now)

            restored = SubmissionBudget(path, daily_limit=5)

            self.assertEqual(restored.remaining(now), 4)

    def test_reserve_skips_duplicate_and_already_attempted_experiments(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
            budget = SubmissionBudget(path, daily_limit=5)
            budget.record("already-sent", now, status="submitted")

            reserved = budget.reserve(["already-sent", "new", "new"], now)

        self.assertEqual(reserved, ["new"])

    def test_release_failed_makes_explicitly_verified_failures_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            now = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
            budget = SubmissionBudget(path, daily_limit=2)
            budget.record("failed-before-upload", now, status="failed")

            self.assertTrue(budget.release_failed(["failed-before-upload"]))
            self.assertEqual(budget.remaining(now), 2)
            self.assertEqual(budget.reserve(["failed-before-upload"], now), ["failed-before-upload"])

    def test_release_reserved_makes_pre_upload_stops_retryable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "budget.json"
            now = datetime(2026, 8, 8, 1, 0, tzinfo=timezone.utc)
            budget = SubmissionBudget(path, daily_limit=5)
            budget.reserve(["preempted"], now)

            self.assertTrue(budget.release_reserved(["preempted"]))
            self.assertEqual(budget.remaining(now), 5)
            self.assertEqual(budget.reserve(["preempted"], now), ["preempted"])


if __name__ == "__main__":
    unittest.main()

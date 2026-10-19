"""Reconcile an externally accepted Kaggle submission in the remote ledger."""

import sys
from pathlib import Path

from biohub_lab.budget import SubmissionBudget


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: reconcile_submission.py EXPERIMENT REFERENCE")
    experiment, reference = sys.argv[1:]
    path = Path("/mnt/biohub/state/submissions.json")
    budget = SubmissionBudget(path, daily_limit=5)
    budget.release_failed([experiment])
    if experiment not in budget.attempted_experiments([experiment]):
        budget.record(experiment, status="submitted", reference=reference)
    print("reconciled {} {}".format(experiment, reference))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

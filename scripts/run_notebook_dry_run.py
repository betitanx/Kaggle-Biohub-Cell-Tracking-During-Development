"""Execute a Kaggle Notebook runner without calling the competition scorer."""

import argparse
import json
from pathlib import Path

from biohub_lab.kaggle import NotebookSubmissionClient, _kaggle_api


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=float, default=14400.0)
    args = parser.parse_args()

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    candidate = next(
        item for item in plan.get("candidates", [])
        if str(item.get("experiment")) == args.experiment
    )
    client = NotebookSubmissionClient(
        str(plan["competition"]),
        str(plan["notebook_dataset"]),
        api=_kaggle_api(),
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
        dataset_timeout_seconds=180.0,
        expected_timepoints=plan.get("expected_timepoints"),
        expected_datasets=plan.get("expected_datasets"),
        payload_mode=str(plan.get("notebook_payload_mode", "runner")),
        enable_gpu=bool(plan.get("notebook_enable_gpu", False)),
        machine_shape=plan.get("notebook_machine_shape"),
        additional_dataset_sources=plan.get("notebook_additional_dataset_sources", ()),
    )
    results = client.dry_run_many(
        [candidate],
        plan.get("cycle_id"),
        args.output_root,
    )
    print(json.dumps(results, indent=2, sort_keys=True, default=str))
    return 0 if results and results[0].get("status") == "dry_run_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())

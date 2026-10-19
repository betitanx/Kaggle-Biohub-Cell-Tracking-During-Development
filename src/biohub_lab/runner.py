"""Run catalogued candidates and persist evidence for later ranking."""

import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .events import emit_event
from .requirements import missing_requirements
from .validation import validate_submission


def load_config(path: Path) -> Dict[str, Any]:
    """Load and minimally validate the JSON experiment catalog."""

    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config.get("experiments"), list):
        raise ValueError("config must contain an experiments list")
    for spec in config["experiments"]:
        if not spec.get("name") or not isinstance(spec.get("command"), list):
            raise ValueError("each experiment needs name and command list")
    return config


def _resolve_token(token: str, run_dir: Path, project_root: Path, env: Mapping[str, str]) -> str:
    replacements = {
        "${PYTHON}": sys.executable,
        "${RUN_DIR}": str(run_dir),
        "${PROJECT_ROOT}": str(project_root),
    }
    for key, value in replacements.items():
        token = token.replace(key, value)
    if "${" in token:
        start = token.find("${")
        end = token.find("}", start + 2)
        if end != -1:
            name = token[start + 2 : end]
            value = env.get(name)
            if value is None:
                raise ValueError("environment variable {} is required by experiment".format(name))
            token = token[:start] + value + token[end + 1 :]
    return token


def render_command(command: Iterable[str], run_dir: Path, project_root: Path, env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Resolve only known placeholders; never invoke a shell."""

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    return [_resolve_token(str(token), run_dir, project_root, merged_env) for token in command]


def _read_metrics(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("metrics.json must contain a JSON object")
    return payload


def run_experiment(spec: Mapping[str, Any], output_dir: Path, project_root: Path, env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Execute one experiment, validate its standard artifact and save result.json."""

    project_root = Path(project_root).resolve()
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = output_dir.resolve()
    run_id = "{}-{}".format(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"), uuid.uuid4().hex[:8])
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.jsonl"
    result: Dict[str, Any] = {
        "run_id": run_id,
        "cycle_id": spec.get("cycle_id"),
        "experiment": str(spec["name"]),
        "family": str(spec.get("family", "unknown")),
        "hypothesis": str(spec.get("hypothesis", "")),
        "tags": list(spec.get("tags", [])),
        "variant": spec.get("variant"),
        "proposal_reason": spec.get("proposal_reason"),
        "notebook_runner_kind": spec.get("notebook_runner_kind"),
        "notebook_runner_source": spec.get("notebook_runner_source"),
        "notebook_runner_det_threshold": spec.get("notebook_runner_det_threshold"),
        "notebook_runner_options": spec.get("notebook_runner_options"),
        "notebook_expected_timepoints": spec.get("notebook_expected_timepoints"),
        "notebook_additional_dataset_sources": spec.get("notebook_additional_dataset_sources"),
        "live_eligible": bool(spec.get("live_eligible", True)),
        "run_dir": str(run_dir),
        "status": "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "artifact": str(run_dir / "submission.csv"),
        "artifact_valid": False,
    }
    # A downloaded subset is useful for local diagnostics but cannot produce a
    # valid competition submission.  Keep this guard in the runner so a
    # future caller cannot accidentally promote a partial-data result merely
    # by passing ``--live-submit`` to the outer controller.
    data_scope = str((env or {}).get("BIOHUB_DATA_SCOPE", os.environ.get("BIOHUB_DATA_SCOPE", "full")))
    result["data_scope"] = data_scope
    if data_scope != "full":
        result["live_eligible"] = False
    emit_event(events_path, run_id, "experiment_started", experiment=result["experiment"], family=result["family"])
    child_env = dict(os.environ)
    if env:
        child_env.update(env)
    blocked = missing_requirements(spec, child_env, project_root)
    if blocked:
        result["status"] = "blocked"
        result["blocked_requirements"] = blocked
        result["error"] = "missing experiment requirements: {}".format(", ".join(blocked))
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        emit_event(events_path, run_id, "experiment_finished", status=result["status"], artifact_valid=result["artifact_valid"])
        with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        return result
    try:
        command = render_command(spec["command"], run_dir, Path(project_root), env)
        timeout = int(spec.get("timeout_seconds", 3600))
        with (run_dir / "stdout.log").open("w", encoding="utf-8") as stdout, (run_dir / "stderr.log").open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(command, cwd=str(project_root), env=child_env, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
        result["returncode"] = completed.returncode
        if completed.returncode != 0:
            result["error"] = "candidate exited with return code {}".format(completed.returncode)
        else:
            metrics = _read_metrics(run_dir / "metrics.json")
            result.update(metrics)
            raw_artifact = metrics.get("artifact")
            artifact = Path(str(raw_artifact)) if raw_artifact else run_dir / "submission.csv"
            if raw_artifact and not artifact.is_absolute():
                artifact = run_dir / artifact
            result["artifact"] = str(artifact)
            report = validate_submission(artifact)
            result["validation"] = report.as_dict()
            result["artifact_valid"] = report.valid
            result["status"] = "completed" if report.valid else "invalid"
    except subprocess.TimeoutExpired:
        result["error"] = "candidate timed out"
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        result["error"] = "{}: {}".format(type(exc).__name__, exc)
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    emit_event(events_path, run_id, "experiment_finished", status=result["status"], artifact_valid=result["artifact_valid"])
    with (run_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return result

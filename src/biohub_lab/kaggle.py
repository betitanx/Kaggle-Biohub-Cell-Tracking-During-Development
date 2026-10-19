"""Opt-in wrapper around the official Kaggle competitions CLI."""

import base64
import csv
import gzip
import json
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .budget import SubmissionBudget
from .validation import validate_submission


_REFERENCE_RE = re.compile(r"Submission ref:\s*(\d+)", re.IGNORECASE)
_SCORE_RE = re.compile(r"Public Score:\s*([0-9.eE+-]+)", re.IGNORECASE)
_DEFAULT_EXPECTED_TIMEPOINTS = object()


def _remaining_submission_window(plan: Mapping[str, Any]) -> Optional[float]:
    """Return remaining persisted cycle time, or None for manual plans."""

    started_raw = plan.get("window_started_at")
    duration_raw = plan.get("window_seconds")
    if started_raw is None and duration_raw is None:
        return None
    if started_raw is None or duration_raw is None:
        raise ValueError("submission plan must contain window_started_at and window_seconds together")
    try:
        started = datetime.fromisoformat(str(started_raw).replace("Z", "+00:00"))
        duration = float(duration_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("submission plan contains an invalid submission window") from exc
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    if not duration >= 0.0:
        raise ValueError("submission window duration must be non-negative")
    elapsed = (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
    return max(0.0, duration - elapsed)


def _redact(text: str, env: Mapping[str, str]) -> str:
    for name in ("KAGGLE_API_TOKEN", "KAGGLE_TOKEN"):
        value = env.get(name)
        if value:
            text = text.replace(value, "***REDACTED***")
    return text


def _resolve_executable(executable: str) -> Optional[str]:
    """Resolve a CLI from PATH or the active virtual environment."""

    resolved = shutil.which(executable)
    if resolved is not None:
        return resolved
    if os.sep not in executable:
        candidate_directories = (
            Path(sys.executable).parent,
            Path(sys.executable).resolve().parent,
            Path(sys.prefix) / "bin",
        )
        seen: set[Path] = set()
        for directory in candidate_directories:
            key = directory.resolve()
            if key in seen:
                continue
            seen.add(key)
            interpreter_cli = directory / executable
            if interpreter_cli.is_file():
                return str(interpreter_cli)
    return None


def _response_field(response: Any, *names: str) -> Any:
    """Read a field from either a Kaggle SDK object or a test double."""

    if response is None:
        return None
    if isinstance(response, Mapping):
        for name in names:
            if name in response:
                return response[name]
    for name in names:
        value = getattr(response, name, None)
        if value is not None:
            return value
    to_dict = getattr(response, "to_dict", None)
    if callable(to_dict):
        try:
            payload = to_dict()
        except Exception:
            payload = None
        if isinstance(payload, Mapping):
            for name in names:
                if name in payload:
                    return payload[name]
    return None


def _status_text(status: Any) -> str:
    value = _response_field(status, "status", "state")
    return str(value if value is not None else status)


def _exception_detail(exc: Exception) -> str:
    """Include safe HTTP response text when an external SDK rejects a request."""

    detail = "{}: {}".format(type(exc).__name__, exc)
    response = getattr(exc, "response", None)
    body = getattr(response, "text", None)
    if body:
        detail = "{}; response={}".format(detail, str(body)[-2000:])
    return detail


def _safe_slug(value: str, maximum: int = 64) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value).lower()).strip("-")
    return (slug or "candidate")[:maximum].rstrip("-")


def _normalise_expected_timepoints(value: Optional[Iterable[int] | int]) -> Optional[tuple[int, ...]]:
    if value is None:
        return None
    if isinstance(value, int):
        if value < 0:
            raise ValueError("expected_timepoints count must be non-negative")
        values = range(value)
    else:
        values = value
    try:
        result = tuple(sorted({int(item) for item in values}))
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_timepoints must be an integer count or iterable of integers") from exc
    if any(item < 0 for item in result):
        raise ValueError("expected_timepoints must contain non-negative integers")
    return result


def _normalise_expected_datasets(value: Optional[Iterable[str] | str]) -> Optional[tuple[str, ...]]:
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    try:
        result = tuple(sorted({str(item).strip() for item in values if str(item).strip()}))
    except TypeError as exc:
        raise ValueError("expected_datasets must be an iterable of dataset names") from exc
    if not result:
        raise ValueError("expected_datasets must contain at least one dataset name")
    return result


def _normalise_dataset_sources(value: Optional[Iterable[str]]) -> tuple[str, ...]:
    """Validate and deduplicate Kaggle Dataset input references."""

    if value is None:
        return ()
    result: list[str] = []
    seen: set[str] = set()
    for raw in value:
        reference = str(raw).strip()
        pieces = reference.split("/")
        if len(pieces) not in {2, 3} or not all(pieces):
            raise ValueError("Kaggle Dataset input must be <owner>/<dataset-slug>[/version]")
        if reference not in seen:
            seen.add(reference)
            result.append(reference)
    return tuple(result)


def _kaggle_api() -> Any:
    """Authenticate the optional Kaggle SDK without exposing credentials."""

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Notebook submission requires the optional 'kaggle' dependency") from exc
    api = KaggleApi()
    api.authenticate()
    return api


class NotebookSubmissionClient:
    """Publish validated artifacts through Kaggle's notebook-only contract.

    The competition accepts code submissions only.  A candidate can either
    copy a validated artifact from a Dataset, embed a diagnostic artifact, or
    execute a self-contained runner against the mounted competition input.
    Each candidate gets a fresh private notebook so stale Dataset attachments
    or kernel versions cannot change the submitted source.
    """

    def __init__(
        self,
        competition: str,
        dataset_ref: str,
        api: Any = None,
        poll_seconds: float = 5.0,
        timeout_seconds: float = 1800.0,
        output_timeout_seconds: float = 180.0,
        dataset_timeout_seconds: float = 180.0,
        expected_timepoints: Optional[Iterable[int] | int] = None,
        expected_datasets: Optional[Iterable[str] | str] = None,
        line_ending: str = "lf",
        payload_mode: str = "dataset",
        enable_gpu: bool = False,
        machine_shape: Optional[str] = None,
        runner_source: Optional[Path | str] = None,
        additional_dataset_sources: Optional[Iterable[str]] = None,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        pieces = str(dataset_ref).split("/")
        if len(pieces) != 2 or not all(pieces):
            raise ValueError("notebook dataset must be <owner>/<dataset-slug>")
        self.competition = str(competition)
        self.dataset_ref = str(dataset_ref)
        self.dataset_owner = pieces[0]
        self.api = api if api is not None else _kaggle_api()
        self.poll_seconds = max(0.0, float(poll_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.output_timeout_seconds = max(0.01, float(output_timeout_seconds))
        self.dataset_timeout_seconds = max(1.0, float(dataset_timeout_seconds))
        self.expected_timepoints = _normalise_expected_timepoints(expected_timepoints)
        self.expected_datasets = _normalise_expected_datasets(expected_datasets)
        self.enable_gpu = bool(enable_gpu)
        normalized_machine_shape = str(machine_shape or "").strip()
        self.machine_shape = normalized_machine_shape or None
        normalized_line_ending = str(line_ending).strip().lower()
        if normalized_line_ending not in {"lf", "crlf"}:
            raise ValueError("line_ending must be 'lf' or 'crlf'")
        self.line_ending = normalized_line_ending
        normalized_payload_mode = str(payload_mode).strip().lower()
        if normalized_payload_mode not in {"dataset", "embedded", "runner"}:
            raise ValueError("payload_mode must be 'dataset', 'embedded' or 'runner'")
        self.payload_mode = normalized_payload_mode
        self.runner_source = Path(runner_source) if runner_source else None
        self.additional_dataset_sources = _normalise_dataset_sources(additional_dataset_sources)
        self.sleep = sleep
        self.monotonic = monotonic
        self._output_timeout_triggered = False

    def _dataset_exists(self) -> bool:
        try:
            status = self.api.dataset_status(self.dataset_ref)
        except Exception:
            return False
        text = str(status).lower()
        return text not in {"", "not found", "error", "failed"}

    def _current_dataset_version(self) -> Optional[int]:
        if not hasattr(self.api, "dataset_status"):
            return None
        try:
            payload = self.api.dataset_status(self.dataset_ref, format="json")
        except TypeError:
            return None
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, Mapping):
            return None
        value = payload.get("current_version_number", payload.get("currentVersionNumber"))
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def _wait_for_dataset(self, minimum_version: Optional[int] = None) -> None:
        if not hasattr(self.api, "dataset_status"):
            return
        deadline = self.monotonic() + self.dataset_timeout_seconds
        while True:
            status = _status_text(self.api.dataset_status(self.dataset_ref)).lower()
            current_version = self._current_dataset_version()
            version_ready = minimum_version is None or current_version is not None and current_version >= minimum_version
            if ("ready" in status or "complete" in status) and version_ready:
                return
            if any(word in status for word in ("error", "failed", "cancel")):
                raise RuntimeError("Kaggle Dataset is not ready: {}".format(status))
            if self.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Kaggle Dataset {}".format(self.dataset_ref))
            self.sleep(self.poll_seconds)

    def publish_dataset(self, candidates: List[Mapping[str, Any]], cycle_id: Optional[str]) -> Dict[str, Any]:
        """Upload the candidate files as one private Dataset version."""

        with tempfile.TemporaryDirectory(prefix="biohub-kaggle-dataset-") as directory:
            folder = Path(directory)
            metadata = {
                "title": "Biohub Cell Tracking Candidates",
                "id": self.dataset_ref,
                "licenses": [{"name": "CC0-1.0"}],
                "description": "Generated candidate submissions for the Biohub experiment loop.",
            }
            (folder / "dataset-metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            names: Dict[str, str] = {}
            for candidate in candidates:
                experiment = str(candidate.get("experiment", "candidate"))
                artifact = Path(str(candidate.get("artifact", "")))
                report = validate_submission(
                    artifact,
                    expected_datasets=set(self.expected_datasets) if self.expected_datasets else None,
                    expected_timepoints=self.expected_timepoints,
                )
                if not report.valid:
                    raise ValueError("submission is invalid for {}: {}".format(experiment, "; ".join(report.errors[:3])))
                filename = "{}.csv".format(_safe_slug(experiment))
                # Normalize at this boundary so legacy artifacts and freshly
                # generated files share one deterministic representation. LF
                # remains the default; CRLF is available only for controlled
                # diagnostics through the explicit line_ending option.
                normalized = artifact.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
                if self.line_ending == "crlf":
                    normalized = normalized.replace(b"\n", b"\r\n")
                (folder / filename).write_bytes(normalized)
                names[experiment] = filename
            notes = "biohub-lab cycle {}".format(cycle_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
            dataset_exists = self._dataset_exists()
            previous_version = self._current_dataset_version() if dataset_exists else None
            if dataset_exists:
                response = self.api.dataset_create_version(
                    str(folder), version_notes=notes, quiet=True, convert_to_csv=False
                )
            else:
                response = self.api.dataset_create_new(
                    str(folder), public=False, quiet=True, convert_to_csv=False
                )
            version = _response_field(response, "version_number", "versionNumber")
            target_version = int(version) if version is not None else None
            if target_version is None and previous_version is not None:
                target_version = previous_version + 1
            self._wait_for_dataset(target_version)
            current_version = self._current_dataset_version()
            dataset_version = current_version if current_version is not None else target_version
            return {
                "dataset": self.dataset_ref,
                "dataset_version": dataset_version,
                "files": names,
            }

    @staticmethod
    def _notebook_source(filename: str) -> str:
        encoded = json.dumps(filename)
        return (
            "from pathlib import Path\n"
            "matches = sorted(Path('/kaggle/input').rglob({}))\n"
            "if not matches:\n"
            "    raise FileNotFoundError('candidate file not mounted: ' + {} )\n"
            "Path('/kaggle/working/submission.csv').write_bytes(matches[0].read_bytes())\n"
        ).format(encoded, encoded)

    @staticmethod
    def _notebook_source_embedded(payload: bytes) -> str:
        """Return a self-contained Notebook source for a validated artifact."""

        compressed = gzip.compress(payload, mtime=0)
        encoded = json.dumps(base64.b64encode(compressed).decode("ascii"))
        return (
            "import base64\n"
            "import gzip\n"
            "from pathlib import Path\n"
            "payload = base64.b64decode({})\n"
            "Path('/kaggle/working/submission.csv').write_bytes(gzip.decompress(payload))\n"
        ).format(encoded)

    def _push_notebook(
        self,
        experiment: str,
        filename: str,
        cycle_id: Optional[str],
        dataset_version: Optional[int] = None,
        embedded_payload: Optional[bytes] = None,
        runner_source: Optional[Path | str] = None,
        runner_kind: Optional[str] = None,
        runner_det_threshold: Optional[float] = None,
        runner_options: Optional[Mapping[str, Any]] = None,
        additional_dataset_sources: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        del cycle_id
        suffix = uuid.uuid4().hex[:8]
        experiment_slug = _safe_slug(experiment, maximum=20)
        digest = hashlib.sha1(experiment.encode("utf-8")).hexdigest()[:6]
        slug = _safe_slug("biohub-{}-{}-{}".format(experiment_slug, digest, suffix), maximum=48)
        title = slug.replace("-", " ")
        kernel_ref = "{}/{}".format(self.dataset_owner, slug)
        if self.payload_mode == "embedded":
            if embedded_payload is None:
                raise ValueError("embedded payload is required in embedded payload mode")
            notebook_source = self._notebook_source_embedded(embedded_payload)
            dataset_sources = list(self.additional_dataset_sources)
        elif self.payload_mode == "runner":
            source_path = Path(runner_source) if runner_source else self.runner_source
            if runner_kind in {"support_pack", "support_pack_metric_probe"}:
                from .kaggle_runner import build_support_pack_runner_source

                package_root = Path(__file__).resolve().parents[2]
                if not (package_root / "src" / "biohub_lab").is_dir():
                    package_root = Path.cwd()
                configured_options = dict(runner_options or {})
                if runner_det_threshold is not None:
                    configured_options["det_threshold"] = float(runner_det_threshold)
                supported_options = {
                    "det_threshold",
                    "edge_threshold",
                    "edge_min_threshold",
                    "edge_topk_parents",
                    "edge_topk_children",
                    "edge_max_distance_um",
        "det_tta_mode",
        "feature_tta",
        "feature_tta_blend",
        "motion_consensus",
        "motion_consensus_max_distance_um",
        "motion_consensus_max_prediction_error_um",
        "motion_consensus_require_unique_parent",
        "motion_consensus_max_out_degree",
        "ilp_appearance_weight",
                    "ilp_disappearance_weight",
                    "ilp_division_weight",
                    "pool_kernel_um",
                    "metric_probe",
                    "metric_probe_max_components",
                    "metric_probe_forks",
                    "gap_closing",
                    "gap_closing_max_total_distance_um",
                    "gap_closing_reuse_existing_um",
                    "gap_closing_max_added_fraction",
                    "gap_closing_max_added_absolute",
                }
                configured_options = {
                    key: value
                    for key, value in configured_options.items()
                    if key in supported_options
                }
                if runner_kind == "support_pack_metric_probe":
                    configured_options["metric_probe"] = True
                notebook_source = build_support_pack_runner_source(package_root, **configured_options)
            elif runner_kind == "h13_finetune":
                from .kaggle_runner import build_h13_finetune_runner_source

                package_root = Path(__file__).resolve().parents[2]
                if not (package_root / "external" / "kaggle-cell-tracking-competition").is_dir():
                    package_root = Path.cwd()
                configured_options = dict(runner_options or {})
                if runner_det_threshold is not None:
                    configured_options["det_threshold"] = float(runner_det_threshold)
                supported_options = {
                    "epochs",
                    "lr",
                    "batch_size",
                    "num_workers",
                    "max_iters",
                    "holdout_fraction",
                    "max_train_datasets",
                    "max_holdout_datasets",
                    "max_frames",
                    "seed",
                    "downsample",
                    "unet_out_channels",
                    "unet_layers",
                    "det_loss_weight",
                    "det_neg_weight",
                    "pool_kernel_um",
                    "det_threshold",
                    "edge_threshold",
                    "ilp_appearance_weight",
                    "ilp_disappearance_weight",
                    "ilp_division_weight",
                    "unet_batch_size",
                    "data_parallel",
                }
                configured_options = {
                    key: value
                    for key, value in configured_options.items()
                    if key in supported_options
                }
                notebook_source = build_h13_finetune_runner_source(package_root, **configured_options)
            elif source_path is None:
                from .kaggle_runner import build_classical_runner_source

                package_root = Path(__file__).resolve().parents[2]
                if not (package_root / "src" / "biohub_lab").is_dir():
                    package_root = Path.cwd()
                notebook_source = build_classical_runner_source(package_root)
            else:
                if not source_path.is_file():
                    raise FileNotFoundError("runner source not found: {}".format(source_path))
                notebook_source = source_path.read_text(encoding="utf-8")
                if not notebook_source.strip():
                    raise ValueError("runner source is empty: {}".format(source_path))
            dataset_sources = list(self.additional_dataset_sources)
        else:
            notebook_source = self._notebook_source(filename)
            dataset_sources = [
                self.dataset_ref
                if dataset_version is None
                else "{}/{}".format(self.dataset_ref, dataset_version)
            ]
            dataset_sources.extend(self.additional_dataset_sources)
        dataset_sources.extend(_normalise_dataset_sources(additional_dataset_sources))
        dataset_sources = list(dict.fromkeys(dataset_sources))
        with tempfile.TemporaryDirectory(prefix="biohub-kaggle-kernel-") as directory:
            folder = Path(directory)
            notebook_name = "{}.ipynb".format(slug)
            notebook = {
                "cells": [
                    {
                        "cell_type": "code",
                        "id": hashlib.sha1(kernel_ref.encode("utf-8")).hexdigest()[:16],
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": notebook_source,
                    }
                ],
                "metadata": {
                    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python", "version": "3.11"},
                },
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            (folder / notebook_name).write_text(json.dumps(notebook), encoding="utf-8")
            metadata = {
                "id": kernel_ref,
                "title": title,
                "code_file": notebook_name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": self.enable_gpu,
                "enable_tpu": False,
                "enable_internet": False,
                "dataset_sources": dataset_sources,
                "competition_sources": [self.competition],
            }
            if self.machine_shape:
                metadata["machine_shape"] = self.machine_shape
            (folder / "kernel-metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            response = self.api.kernels_push(str(folder))
        response_ref = _response_field(response, "ref")
        response_text = str(response_ref).strip() if response_ref is not None else ""
        response_parts = response_text.split("/")
        valid_response_ref = (
            len(response_parts) == 2
            and all(response_parts[:2])
        )
        # Kaggle's SDK returns refs such as ``/code/owner/slug``. That is a
        # notebook URL path, not the kernel identifier accepted by status,
        # output and competition submission endpoints.
        # The metadata id is the kernel we actually pushed.  Under concurrent
        # pushes Kaggle can return another valid kernel ref from a stale
        # response, which would make us wait for and download the wrong
        # notebook output.  Keep the deterministic requested id authoritative
        # and retain the inconsistent response only as audit evidence.
        response_ref_mismatch = valid_response_ref and response_text != kernel_ref
        actual_kernel = kernel_ref
        version = _response_field(response, "version_number", "versionNumber")
        if version is None:
            raise RuntimeError("Kaggle did not return a Notebook version for {}".format(actual_kernel))
        result = {"kernel": str(actual_kernel), "kernel_version": int(version)}
        if response_ref_mismatch:
            result["push_response_ref"] = response_text
        return result

    def _wait_for_kernel(self, kernel: str) -> None:
        deadline = self.monotonic() + self.timeout_seconds
        while True:
            try:
                status = _status_text(self.api.kernels_status(kernel))
            except Exception as exc:
                # Kaggle may create the Notebook before its execution session
                # is visible to GetKernelSessionStatus.  Treat that narrow
                # 404 as launch latency, but surface all other failures.
                detail = _exception_detail(exc).lower()
                if "no runs found for this kernel" not in detail:
                    raise
                if self.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for Kaggle Notebook {}".format(kernel)) from exc
                self.sleep(self.poll_seconds)
                continue
            normalized = status.lower()
            if "complete" in normalized:
                return
            if any(word in normalized for word in ("error", "failed", "cancel")):
                raise RuntimeError("Kaggle Notebook {} ended with status {}".format(kernel, status))
            if self.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Kaggle Notebook {}".format(kernel))
            self.sleep(self.poll_seconds)

    def _verify_kernel_output(
        self,
        kernel: str,
        expected_timepoints: Optional[Iterable[int] | int] | object = _DEFAULT_EXPECTED_TIMEPOINTS,
        allow_metric_probe: bool = False,
    ) -> None:
        """Download and validate the exact file that Kaggle will score."""

        # The official SDK exposes kernels_output. Keeping the capability check
        # makes small offline test doubles and older SDKs usable, while real
        # Kaggle submissions always take this post-execution validation path.
        if not hasattr(self.api, "kernels_output"):
            return
        with tempfile.TemporaryDirectory(prefix="biohub-kaggle-kernel-output-") as directory:
            download_errors: list[BaseException] = []

            def download_output() -> None:
                try:
                    self.api.kernels_output(
                        kernel,
                        directory,
                        file_pattern="submission.csv",
                        force=True,
                        quiet=True,
                    )
                except BaseException as exc:  # pragma: no cover - exercised through the caller
                    download_errors.append(exc)

            worker = threading.Thread(
                target=download_output,
                name="biohub-kernel-output-download",
                daemon=True,
            )
            worker.start()
            worker.join(self.output_timeout_seconds)
            if worker.is_alive():
                self._output_timeout_triggered = True
                raise TimeoutError(
                    "timed out downloading Kaggle Notebook output for {} after {:.1f}s".format(
                        kernel, self.output_timeout_seconds
                    )
                )
            if download_errors:
                raise download_errors[0]
            output = Path(directory) / "submission.csv"
            if expected_timepoints is _DEFAULT_EXPECTED_TIMEPOINTS:
                required_timepoints = self.expected_timepoints
            elif expected_timepoints is None:
                required_timepoints = None
            else:
                required_timepoints = _normalise_expected_timepoints(expected_timepoints)
            report = validate_submission(
                output,
                expected_datasets=set(self.expected_datasets) if self.expected_datasets else None,
                expected_timepoints=required_timepoints,
                allow_metric_probe=allow_metric_probe,
            )
            if not report.valid:
                raise ValueError(
                    "Kaggle Notebook output is invalid: {}".format("; ".join(report.errors[:5]))
                )

    def submit_many(self, candidates: List[Mapping[str, Any]], cycle_id: Optional[str]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        dataset: Dict[str, Any] = {"dataset": None, "dataset_version": None, "files": {}}
        if self.payload_mode == "dataset":
            try:
                dataset = self.publish_dataset(candidates, cycle_id)
            except Exception as exc:
                detail = _exception_detail(exc)
                return [
                    {"experiment": str(candidate.get("experiment", "")), "status": "failed", "error": detail}
                    for candidate in candidates
                ]
        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            experiment = str(candidate.get("experiment", ""))
            try:
                artifact = Path(str(candidate.get("artifact", "")))
                if self.payload_mode == "embedded":
                    report = validate_submission(
                        artifact,
                        expected_datasets=set(self.expected_datasets) if self.expected_datasets else None,
                        expected_timepoints=self.expected_timepoints,
                    )
                    if not report.valid:
                        raise ValueError(
                            "submission is invalid for {}: {}".format(experiment, "; ".join(report.errors[:3]))
                        )
                    filename = "submission.csv"
                    embedded_payload = artifact.read_bytes()
                    runner_source = None
                    runner_kind = None
                    runner_det_threshold = None
                    runner_options = None
                elif self.payload_mode == "runner":
                    filename = "submission.csv"
                    embedded_payload = None
                    runner_source = candidate.get("notebook_runner_source") or self.runner_source
                    runner_kind = candidate.get("notebook_runner_kind")
                    runner_det_threshold = candidate.get("notebook_runner_det_threshold")
                    runner_options = candidate.get("notebook_runner_options")
                else:
                    filename = str(dataset["files"][experiment])
                    embedded_payload = None
                    runner_source = None
                    runner_kind = None
                    runner_det_threshold = None
                    runner_options = None
                candidate_sources = candidate.get("notebook_additional_dataset_sources", ())
                kernel = self._push_notebook(
                    experiment,
                    filename,
                    cycle_id,
                    dataset_version=dataset.get("dataset_version"),
                    embedded_payload=embedded_payload,
                    runner_source=runner_source,
                    runner_kind=runner_kind,
                    runner_det_threshold=runner_det_threshold,
                    runner_options=runner_options,
                    additional_dataset_sources=candidate_sources,
                )
                self._wait_for_kernel(kernel["kernel"])
                candidate_expected_timepoints = (
                    _normalise_expected_timepoints(candidate.get("notebook_expected_timepoints"))
                    if "notebook_expected_timepoints" in candidate
                    else self.expected_timepoints
                )
                self._verify_kernel_output(
                    kernel["kernel"],
                    expected_timepoints=candidate_expected_timepoints,
                    allow_metric_probe=runner_kind == "support_pack_metric_probe",
                )
                response = self.api.competition_submit_code(
                    "submission.csv",
                    "biohub-notebook:{}".format(experiment),
                    competition=self.competition,
                    kernel=kernel["kernel"],
                    kernel_version=kernel["kernel_version"],
                    quiet=True,
                )
                reference = _response_field(response, "ref")
                result: Dict[str, Any] = {
                    "experiment": experiment,
                    "status": "submitted",
                    "dataset": dataset["dataset"],
                    "dataset_version": dataset.get("dataset_version"),
                    **kernel,
                }
                if reference is not None:
                    result["reference"] = str(reference)
                results.append(result)
            except Exception as exc:
                results.append(
                    {
                        "experiment": experiment,
                        "status": "failed",
                        "dataset": dataset["dataset"],
                        "dataset_version": dataset.get("dataset_version"),
                        "error": _exception_detail(exc),
                    }
                )
        return results

    def dry_run_many(
        self,
        candidates: List[Mapping[str, Any]],
        cycle_id: Optional[str],
        output_root: Path,
    ) -> List[Dict[str, Any]]:
        """Execute runner candidates and persist validated outputs without scoring them."""

        if not candidates:
            return []
        if self.payload_mode == "dataset":
            raise ValueError("dry-run kernel execution requires payload_mode='runner' or 'embedded'")
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        results: List[Dict[str, Any]] = []
        for candidate in candidates:
            experiment = str(candidate.get("experiment", ""))
            started = self.monotonic()
            try:
                artifact = Path(str(candidate.get("artifact", "")))
                if self.payload_mode == "embedded":
                    report = validate_submission(
                        artifact,
                        expected_datasets=set(self.expected_datasets) if self.expected_datasets else None,
                        expected_timepoints=self.expected_timepoints,
                    )
                    if not report.valid:
                        raise ValueError(
                            "submission is invalid for {}: {}".format(experiment, "; ".join(report.errors[:3]))
                        )
                    embedded_payload = artifact.read_bytes()
                    runner_source = None
                    runner_kind = None
                    runner_det_threshold = None
                    runner_options = None
                else:
                    embedded_payload = None
                    runner_source = candidate.get("notebook_runner_source") or self.runner_source
                    runner_kind = candidate.get("notebook_runner_kind")
                    runner_det_threshold = candidate.get("notebook_runner_det_threshold")
                    runner_options = candidate.get("notebook_runner_options")
                kernel = self._push_notebook(
                    experiment,
                    "submission.csv",
                    cycle_id,
                    embedded_payload=embedded_payload,
                    runner_source=runner_source,
                    runner_kind=runner_kind,
                    runner_det_threshold=runner_det_threshold,
                    runner_options=runner_options,
                    additional_dataset_sources=candidate.get("notebook_additional_dataset_sources", ()),
                )
                self._wait_for_kernel(kernel["kernel"])
                candidate_dir = output_root / _safe_slug(experiment)
                candidate_dir.mkdir(parents=True, exist_ok=True)
                self.api.kernels_output(
                    kernel["kernel"],
                    str(candidate_dir),
                    file_pattern="submission.csv",
                    force=True,
                    quiet=True,
                )
                output = candidate_dir / "submission.csv"
                report = validate_submission(
                    output,
                    expected_datasets=set(self.expected_datasets) if self.expected_datasets else None,
                    expected_timepoints=self.expected_timepoints,
                    include_topology=True,
                )
                if not report.valid:
                    raise ValueError("Kaggle Notebook dry-run output is invalid: {}".format("; ".join(report.errors[:5])))
                if report.nonadjacent_edge_count or report.max_out_degree > 2:
                    raise ValueError(
                        "Kaggle Notebook dry-run topology failed: nonadjacent_edge_count={}, max_out_degree={}".format(
                            report.nonadjacent_edge_count, report.max_out_degree
                        )
                    )
                results.append(
                    {
                        "experiment": experiment,
                        "status": "dry_run_complete",
                        "artifact": str(output),
                        "generation_seconds": self.monotonic() - started,
                        "validation": report.as_dict(),
                        **kernel,
                    }
                )
            except Exception as exc:
                results.append(
                    {
                        "experiment": experiment,
                        "status": "failed",
                        "generation_seconds": self.monotonic() - started,
                        "error": _exception_detail(exc),
                    }
                )
        return results


class KaggleClient:
    """Submit a validated CSV through `kaggle competitions submit` when enabled."""

    def __init__(self, competition: str, executable: str = "kaggle"):
        self.competition = competition
        self.executable = executable

    def remaining_submission_limit(self, env: Optional[Mapping[str, str]] = None) -> int:
        """Read Kaggle's current allowance before reserving live slots."""

        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        executable = _resolve_executable(self.executable)
        if executable is None:
            raise FileNotFoundError("Kaggle CLI not found: {}".format(self.executable))
        command = [executable, "competitions", "submission-limits", self.competition, "--json"]
        completed = subprocess.run(command, env=merged_env, capture_output=True, text=True, check=False)
        output = _redact("{}\n{}".format(completed.stdout, completed.stderr).strip(), merged_env)
        if completed.returncode != 0:
            raise RuntimeError("Kaggle submission limit query failed: {}".format(output[-1000:]))
        try:
            payload = json.loads(completed.stdout)
            allowed = int(payload["numAllowedNow"])
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError("Kaggle submission limit query returned invalid JSON") from exc
        if allowed < 0:
            raise RuntimeError("Kaggle submission limit query returned a negative allowance")
        return allowed

    def remote_submission_records(self, env: Optional[Mapping[str, str]] = None) -> List[Dict[str, Any]]:
        """Read the official submission list for duplicate-name protection."""

        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        executable = _resolve_executable(self.executable)
        if executable is None:
            raise FileNotFoundError("Kaggle CLI not found: {}".format(self.executable))
        command = [executable, "competitions", "submissions", self.competition, "--csv"]
        completed = subprocess.run(
            command,
            env=merged_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        output = _redact("{}\n{}".format(completed.stdout, completed.stderr).strip(), merged_env)
        if completed.returncode != 0:
            raise RuntimeError("Kaggle submission list query failed: {}".format(output[-1000:]))
        records: List[Dict[str, Any]] = []
        for row in csv.DictReader(io.StringIO(completed.stdout or "")):
            description = str(row.get("description", "")).strip()
            if ":" not in description:
                continue
            prefix, experiment = description.split(":", 1)
            if prefix not in {"biohub-notebook", "biohub-lab"} or not experiment.strip():
                continue
            records.append(
                {
                    "experiment": experiment.strip(),
                    "reference": str(row.get("ref", "")).strip() or None,
                    "status": str(row.get("status", "")).strip() or None,
                    "public_score": str(row.get("publicScore", "")).strip() or None,
                    "private_score": str(row.get("privateScore", "")).strip() or None,
                }
            )
        return records

    def remote_attempted_experiments(self, env: Optional[Mapping[str, str]] = None) -> set[str]:
        return {str(record["experiment"]) for record in self.remote_submission_records(env) if record.get("experiment")}

    def submit(
        self,
        artifact: Path,
        message: str,
        wait_seconds: int = 600,
        live: bool = False,
        env: Optional[Mapping[str, str]] = None,
        expected_timepoints: Optional[Iterable[int] | int] = None,
        expected_datasets: Optional[Iterable[str] | str] = None,
    ) -> Dict[str, Any]:
        merged_env = dict(os.environ)
        if env:
            merged_env.update(env)
        if not live or merged_env.get("BIOHUB_ALLOW_LIVE_SUBMISSIONS") != "1":
            raise PermissionError("live submission requires --live-submit and BIOHUB_ALLOW_LIVE_SUBMISSIONS=1")
        report = validate_submission(
            Path(artifact),
            expected_datasets=set(_normalise_expected_datasets(expected_datasets) or ()),
            expected_timepoints=_normalise_expected_timepoints(expected_timepoints),
        )
        if not report.valid:
            raise ValueError("submission is invalid: {}".format("; ".join(report.errors[:3])))
        executable = _resolve_executable(self.executable)
        if executable is None:
            raise FileNotFoundError("Kaggle CLI not found: {}".format(self.executable))
        # Kaggle CLI 2.x performs the upload synchronously and does not expose
        # the legacy ``--wait`` option.  Keep wait_seconds for API compatibility
        # with existing callers, but never pass it to the installed CLI.
        command = [executable, "competitions", "submit", self.competition, "-f", str(artifact), "-m", message]
        completed = subprocess.run(command, env=merged_env, capture_output=True, text=True, check=False)
        output = _redact("{}\n{}".format(completed.stdout, completed.stderr).strip(), merged_env)
        result: Dict[str, Any] = {"returncode": completed.returncode, "output": output}
        reference = _REFERENCE_RE.search(output)
        score = _SCORE_RE.search(output)
        if reference:
            result["reference"] = reference.group(1)
        if score:
            result["public_score"] = float(score.group(1))
        if completed.returncode != 0:
            raise RuntimeError("Kaggle submission failed: {}".format(output[-1000:]))
        return result


def submit_plan(
    plan_path: Path,
    state_path: Path,
    live: bool = False,
    wait_seconds: int = 600,
    env: Optional[Mapping[str, str]] = None,
    retry_failed: bool = False,
) -> List[Dict[str, Any]]:
    with Path(plan_path).open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    limit = int(merged_env.get("KAGGLE_DAILY_SUBMISSION_LIMIT", plan.get("limit", 5)))
    candidates = list(plan.get("candidates", []))
    expected_timepoints = _normalise_expected_timepoints(plan.get("expected_timepoints"))
    expected_datasets = _normalise_expected_datasets(plan.get("expected_datasets"))
    if not live:
        return [{"experiment": item.get("experiment"), "status": "dry_run", "artifact": item.get("artifact")} for item in candidates]
    if merged_env.get("BIOHUB_ALLOW_LIVE_SUBMISSIONS") != "1":
        raise PermissionError("live submission requires BIOHUB_ALLOW_LIVE_SUBMISSIONS=1")
    remaining_window = _remaining_submission_window(plan)
    if remaining_window is not None and remaining_window > 0.0:
        raise PermissionError(
            "submission window has not elapsed; {:.0f} seconds remain".format(remaining_window)
        )
    eligible = [
        item
        for item in candidates
        if str(item.get("family", "")) != "smoke" and item.get("live_eligible", True) is not False
    ]
    names = [str(item.get("experiment")) for item in eligible]
    client = KaggleClient(str(plan["competition"]))
    remote_allowance: Optional[int] = None
    if names:
        remote_allowance = client.remaining_submission_limit(merged_env)
    budget = SubmissionBudget(state_path, daily_limit=limit)
    if retry_failed:
        budget.release_failed(names)
    if remote_allowance is not None:
        # The remote value is the number of new uploads still allowed.  The
        # local ledger may already contain accepted uploads from this day, so
        # do not treat the remote remainder as the entire daily limit.
        budget.daily_limit = min(limit, budget.used_today() + remote_allowance)
    already_attempted = budget.attempted_experiments(names)
    remote_attempted: set[str] = set()
    if (
        plan.get("require_all_candidates", False)
        and names
        and remote_allowance is not None
        and remote_allowance >= len(names)
    ):
        remote_attempted = client.remote_attempted_experiments(merged_env)
        already_attempted |= remote_attempted
    if plan.get("require_all_candidates", False) and remote_allowance is not None:
        fresh_names = [name for name in names if name not in already_attempted]
        if len(fresh_names) != len(names) or remote_allowance < len(fresh_names):
            required = len(fresh_names)
            results = []
            for item in candidates:
                name = str(item.get("experiment"))
                if name in already_attempted:
                    status = "already_attempted"
                elif name in fresh_names:
                    status = "blocked_remote_duplicate" if len(fresh_names) != len(names) else "blocked_remote_budget"
                else:
                    status = "blocked_non_live_candidate"
                results.append({
                    "experiment": name,
                    "status": status,
                    "artifact": item.get("artifact"),
                    "error": (
                        "plan requires all {} candidates to be fresh, but a local or remote attempt already exists"
                        .format(len(names))
                        if status == "blocked_remote_duplicate"
                        else "plan requires {} fresh slots, Kaggle allows {}".format(required, remote_allowance)
                    )
                    if status in {"blocked_remote_budget", "blocked_remote_duplicate"}
                    else None,
                })
            return results
    accepted = budget.reserve(names)
    submission_mode = str(plan.get("submission_mode", merged_env.get("BIOHUB_SUBMISSION_MODE", "file"))).lower()
    if submission_mode not in {"file", "notebook"}:
        raise ValueError("unsupported submission_mode: {}".format(submission_mode))
    notebook_results: Dict[str, Dict[str, Any]] = {}
    if submission_mode == "notebook" and accepted:
        dataset_ref = str(plan.get("notebook_dataset", merged_env.get("BIOHUB_KAGGLE_DATASET", ""))).strip()
        payload_mode = str(
            plan.get("notebook_payload_mode", merged_env.get("BIOHUB_NOTEBOOK_PAYLOAD_MODE", "dataset"))
        )
        if not dataset_ref:
            raise ValueError("notebook submission requires notebook_dataset in the plan or BIOHUB_KAGGLE_DATASET")
        notebook_candidates = [
            item
            for item in candidates
            if str(item.get("experiment")) in accepted
        ]
        client_options = {
            "poll_seconds": float(merged_env.get("BIOHUB_NOTEBOOK_POLL_SECONDS", "5")),
            "timeout_seconds": float(merged_env.get("BIOHUB_NOTEBOOK_TIMEOUT_SECONDS", "1800")),
            "output_timeout_seconds": float(merged_env.get("BIOHUB_NOTEBOOK_OUTPUT_TIMEOUT_SECONDS", "180")),
            "dataset_timeout_seconds": float(merged_env.get("BIOHUB_DATASET_TIMEOUT_SECONDS", "180")),
            "expected_timepoints": expected_timepoints,
            "expected_datasets": expected_datasets,
            "line_ending": str(plan.get("notebook_line_ending", merged_env.get("BIOHUB_NOTEBOOK_LINE_ENDING", "lf"))),
            "payload_mode": payload_mode,
            "enable_gpu": str(plan.get("notebook_enable_gpu", merged_env.get("BIOHUB_NOTEBOOK_ENABLE_GPU", "0"))).lower()
            in {"1", "true", "yes", "on"},
            "machine_shape": str(
                plan.get("notebook_machine_shape", merged_env.get("BIOHUB_NOTEBOOK_MACHINE_SHAPE", ""))
            ).strip()
            or None,
            "additional_dataset_sources": _normalise_dataset_sources(
                plan.get("notebook_additional_dataset_sources", ())
            ),
        }
        runner_source = plan.get(
            "notebook_runner_source",
            merged_env.get("BIOHUB_NOTEBOOK_RUNNER_SOURCE", ""),
        )
        if runner_source:
            client_options["runner_source"] = str(runner_source)
        try:
            notebook_client = NotebookSubmissionClient(
                str(plan["competition"]),
                dataset_ref,
                **client_options,
            )
            notebook_results = {
                str(result.get("experiment")): result
                for result in notebook_client.submit_many(notebook_candidates, plan.get("cycle_id"))
            }
        except Exception as exc:
            detail = _exception_detail(exc)
            notebook_results = {
                str(item.get("experiment")): {
                    "experiment": str(item.get("experiment")),
                    "status": "failed",
                    "error": detail,
                }
                for item in notebook_candidates
            }
    results: List[Dict[str, Any]] = []
    for item in candidates:
        name = str(item.get("experiment"))
        if str(item.get("family", "")) == "smoke":
            results.append({"experiment": name, "status": "blocked_smoke_candidate", "artifact": item.get("artifact")})
            continue
        if item.get("live_eligible", True) is False:
            results.append({"experiment": name, "status": "blocked_non_live_candidate", "artifact": item.get("artifact")})
            continue
        if name in already_attempted:
            results.append({"experiment": name, "status": "already_attempted", "artifact": item.get("artifact")})
            continue
        if name not in accepted:
            if name in budget.attempted_experiments([name]):
                status = "already_attempted"
                status = "already_attempted"
            else:
                status = "budget_exhausted"
            results.append({"experiment": name, "status": status, "artifact": item.get("artifact")})
            continue
        if submission_mode == "notebook":
            notebook_result = dict(
                notebook_results.get(
                    name,
                    {"experiment": name, "status": "failed", "error": "Notebook result missing"},
                )
            )
            notebook_result["artifact"] = item.get("artifact")
            if notebook_result.get("status") == "submitted":
                budget.mark(name, "submitted", str(notebook_result.get("reference", "")) or None)
            else:
                budget.mark(name, "failed")
            results.append(notebook_result)
            continue
        try:
            result = client.submit(
                Path(str(item["artifact"])),
                "biohub-lab:{}".format(name),
                wait_seconds=wait_seconds,
                live=True,
                env=merged_env,
                expected_timepoints=expected_timepoints,
                expected_datasets=expected_datasets,
            )
            budget.mark(name, "submitted", str(result.get("reference", "")) or None)
            results.append({"experiment": name, "status": "submitted", "artifact": item.get("artifact"), **result})
        except Exception as exc:
            budget.mark(name, "failed")
            results.append({"experiment": name, "status": "failed", "artifact": item.get("artifact"), "error": "{}: {}".format(type(exc).__name__, exc)})
    return results

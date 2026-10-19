"""Bridge OpenResearch experiment snapshots into non-live Biohub proposals.

OpenResearch owns research conversations, branches and runs.  This module
accepts a small JSON snapshot exported by an OpenResearch-side helper and
turns only the research metadata into Biohub proposal records.  It deliberately
does not import the ``orx`` binary, launch commands, mutate Git branches or
mark anything live-eligible.

The bridge enforces OpenResearch's fixed run-contract rule: a proposal may
vary code/configuration, but it may not replace the run command or its
environment.  The Biohub runner and its validation gates remain authoritative.
"""

from __future__ import annotations

import json
import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_SOURCE_URL = "https://github.com/alphaXiv/OpenResearch"
BRIDGE_SCHEMA_VERSION = 1

# Keep this list synchronized with the options that the Biohub support-pack
# runner actually materializes in src/biohub_lab/kaggle.py.  Unknown options
# are retained as research notes but cannot become executable automatically.
SUPPORTED_SUPPORT_PACK_OPTIONS = frozenset(
    {
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
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")
_SHELL_OPERATOR = re.compile(r"(?:&&|\|\||[;|<>`]|\$\()")

_RUN_COMMAND_KEYS = (
    "run_command",
    "runCommand",
    "command",
    "execution_command",
    "executionCommand",
)
_ENVIRONMENT_KEYS = (
    "env",
    "environment",
    "env_vars",
    "envVars",
    "backend",
    "compute",
    "provider",
    "gpu",
    "flavor",
)
_TERMINAL_STATUSES = {
    "complete",
    "completed",
    "done",
    "error",
    "failed",
    "failure",
    "cancelled",
    "canceled",
    "frozen",
    "answered",
    "succeeded",
    "success",
}
_IN_FLIGHT_STATUSES = {
    "running",
    "in_progress",
    "in-progress",
    "queued",
    "launching",
}


class OpenResearchBridgeError(ValueError):
    """Raised when an OpenResearch snapshot cannot be safely interpreted."""


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _slug(value: Any) -> str:
    text = _SAFE_NAME.sub("-", str(value)).strip("-.")
    return text[:80] or "experiment"


def _text(value: Any, *, maximum: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()[:maximum]
    return str(value).strip()[:maximum]


def _nonempty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, (str, bytes, bytearray)):
        return bool(value)
    return True


def parse_fixed_run_command(value: str | Sequence[str]) -> tuple[str, list[str]]:
    """Return a shell-free display command and argv for a fixed contract.

    OpenResearch stores a command as text, while Biohub subprocesses use an
    argv list.  Shell operators are rejected because neither side should
    silently turn the bridge into a shell launcher.
    """

    if isinstance(value, str):
        command_text = value.strip()
        if not command_text:
            raise OpenResearchBridgeError("fixed run command cannot be empty")
        if _SHELL_OPERATOR.search(command_text):
            raise OpenResearchBridgeError(
                "fixed run command must be a single shell-free command"
            )
        try:
            argv = shlex.split(command_text, posix=True)
        except ValueError as exc:
            raise OpenResearchBridgeError(
                "fixed run command has invalid quoting: {}".format(exc)
            ) from exc
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        argv = [_text(item, maximum=1000) for item in value]
        command_text = shlex.join(argv)
    else:
        raise OpenResearchBridgeError(
            "fixed run command must be a string or an argv sequence"
        )
    if not argv or any(not item for item in argv):
        raise OpenResearchBridgeError("fixed run command must contain non-empty argv")
    return command_text, argv


def _parse_optional_command(value: Any) -> list[str] | None:
    if not _nonempty(value):
        return None
    _display, argv = parse_fixed_run_command(value)
    return argv


def _extract_nodes(payload: Any) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if isinstance(payload, list):
        nodes = payload
        root: Mapping[str, Any] = {}
    elif isinstance(payload, Mapping):
        root = payload
        nodes_value = _first(
            payload,
            "experiments",
            "experiment_nodes",
            "experimentNodes",
            "nodes",
            "items",
        )
        if isinstance(nodes_value, Mapping):
            nodes_value = _first(nodes_value, "experiments", "nodes", "items")
        if nodes_value is None:
            raise OpenResearchBridgeError(
                "OpenResearch snapshot must contain an experiments or nodes list"
            )
        nodes = nodes_value
    else:
        raise OpenResearchBridgeError("OpenResearch snapshot must be a JSON object or list")

    if not isinstance(nodes, list) or not all(isinstance(node, Mapping) for node in nodes):
        raise OpenResearchBridgeError("OpenResearch experiments/nodes must be a list of objects")
    return root, list(nodes)


def _merge_option_maps(node: Mapping[str, Any]) -> tuple[dict[str, Any], list[str], list[str]]:
    """Extract safe runner options without copying arbitrary input objects."""

    containers: list[Mapping[str, Any]] = []
    for key in (
        "biohub",
        "changes",
        "overrides",
        "config",
        "runner_options",
        "runnerOptions",
        "notebook_runner_options",
        "notebookRunnerOptions",
    ):
        value = node.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
            for nested_key in ("options", "runner_options", "runnerOptions", "notebook_runner_options", "notebookRunnerOptions"):
                nested = value.get(nested_key)
                if isinstance(nested, Mapping):
                    containers.append(nested)

    options: dict[str, Any] = {}
    unknown: list[str] = []
    change_keys: list[str] = []
    for container in containers:
        for key, value in container.items():
            key_text = _text(key, maximum=120)
            if key_text in SUPPORTED_SUPPORT_PACK_OPTIONS:
                if isinstance(value, (Mapping, list, tuple, set)):
                    unknown.append(key_text)
                    continue
                options[key_text] = value
                change_keys.append(key_text)
            elif key_text in {"options", "runner_options", "runnerOptions", "notebook_runner_options", "notebookRunnerOptions"}:
                continue
            elif key_text in {"runner_kind", "runnerKind", "kind"}:
                continue
            elif key_text:
                # Keys are recorded, but their values are intentionally not
                # copied because OpenResearch descriptions are untrusted data.
                unknown.append(key_text)

    return options, sorted(set(unknown)), sorted(set(change_keys))


def _source_id(node: Mapping[str, Any], index: int) -> str:
    value = _first(node, "id", "experiment_id", "experimentId", "node_id", "nodeId", "slug")
    return _text(value, maximum=200) or "node-{}".format(index + 1)


def _parent_id(node: Mapping[str, Any]) -> str | None:
    value = _first(
        node,
        "parent_id",
        "parentId",
        "parent_experiment_id",
        "parentExperimentId",
        "parent",
    )
    text = _text(value, maximum=200)
    return text or None


def _status(node: Mapping[str, Any]) -> str:
    return _text(_first(node, "status", "agent_status", "agentStatus", default=""), maximum=80).lower()


def _contract_issues(node: Mapping[str, Any], expected_argv: Sequence[str]) -> list[str]:
    issues: list[str] = []
    for key in _RUN_COMMAND_KEYS:
        if key not in node or not _nonempty(node[key]):
            continue
        try:
            actual = _parse_optional_command(node[key])
        except OpenResearchBridgeError as exc:
            issues.append("{}: {}".format(key, exc))
            continue
        if actual is not None and list(actual) != list(expected_argv):
            issues.append("{} changes the fixed run command".format(key))
    for key in _ENVIRONMENT_KEYS:
        if key in node and _nonempty(node[key]):
            issues.append("{} changes the fixed execution environment".format(key))
    return issues


def _node_title(node: Mapping[str, Any], source_id: str) -> str:
    return _text(_first(node, "title", "name", "display_name", "displayName", "slug"), maximum=160) or source_id


def _node_hypothesis(node: Mapping[str, Any], title: str) -> str:
    value = _first(node, "hypothesis", "question", "objective", "description", "desc")
    hypothesis = _text(value, maximum=4000)
    return hypothesis or title


def _tags(node: Mapping[str, Any]) -> list[str]:
    raw = _first(node, "tags", "labels", default=[])
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray)):
        raw = []
    values = [_slug(item) for item in raw if _text(item, maximum=80)]
    return list(dict.fromkeys(["openresearch", "research-only", "needs-holdout"] + values))


def _source_metadata(node: Mapping[str, Any], source_id: str, parent_id: str | None) -> dict[str, Any]:
    metadata: dict[str, Any] = {"experiment_id": source_id}
    if parent_id:
        metadata["parent_experiment_id"] = parent_id
    for output_key, aliases in (
        ("branch", ("branch", "branch_name", "branchName")),
        ("commit", ("commit", "commit_sha", "commitSha", "commit_hash", "commitHash")),
        ("source_status", ("status", "agent_status", "agentStatus")),
    ):
        value = _text(_first(node, *aliases), maximum=240)
        if value:
            metadata[output_key] = value
    return metadata


def _unique_name(base: str, used: set[str]) -> str:
    if base not in used:
        return base
    index = 2
    while "{}-{}".format(base, index) in used:
        index += 1
    return "{}-{}".format(base, index)


def build_openresearch_proposals(
    payload: Any,
    *,
    fixed_run_command: str | Sequence[str],
    existing_names: Iterable[str] = (),
    competition: str = "biohub-cell-tracking-during-development",
    source_url: str = DEFAULT_SOURCE_URL,
    imported_at: str | None = None,
) -> dict[str, Any]:
    """Convert an OpenResearch snapshot into a bounded, non-live proposal queue."""

    fixed_display, fixed_argv = parse_fixed_run_command(fixed_run_command)
    root, nodes = _extract_nodes(payload)
    existing = {_text(item, maximum=200) for item in existing_names if _text(item, maximum=200)}
    used_names = set(existing)

    project = root.get("project") if isinstance(root.get("project"), Mapping) else {}
    project_id = _text(_first(root, "project_id", "projectId", default=_first(project, "id", "project_id", "projectId")), maximum=200)
    project_name = _text(_first(root, "project_name", "projectName", default=_first(project, "name", "title")), maximum=200)
    project_repo = _text(_first(root, "repo", "repo_path", "repoPath", default=_first(project, "repo", "repoPath", "path")), maximum=400)
    warnings: list[str] = []
    blocked: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    source_ids = {_source_id(node, index) for index, node in enumerate(nodes)}

    project_command = _first(root, "run_command", "runCommand", default=_first(project, "run_command", "runCommand"))
    if _nonempty(project_command):
        try:
            project_argv = _parse_optional_command(project_command)
        except OpenResearchBridgeError as exc:
            warnings.append("project run command was not imported: {}".format(exc))
        else:
            if project_argv is not None and project_argv != fixed_argv:
                warnings.append("project run command differs from the fixed Biohub command; affected nodes are blocked")

    for index, node in enumerate(nodes):
        source_id = _source_id(node, index)
        title = _node_title(node, source_id)
        base_name = "openresearch__{}".format(_slug(title))
        # An exact name already present in the Biohub ledger is a duplicate,
        # not a new variant that may be silently renamed.  Suffixes are only
        # for two distinct nodes in the same fresh snapshot with the same
        # title.
        name = base_name if base_name in existing else _unique_name(base_name, used_names)
        used_names.add(name)
        parent_id = _parent_id(node)
        status = _status(node)
        options, unknown_options, changed_keys = _merge_option_maps(node)
        issues = _contract_issues(node, fixed_argv)
        if _nonempty(project_command):
            try:
                project_argv = _parse_optional_command(project_command)
            except OpenResearchBridgeError:
                project_argv = None
            if project_argv is not None and project_argv != fixed_argv:
                issues.append("project run command differs from the fixed Biohub command")
        if status in _TERMINAL_STATUSES:
            issues.append("source node is already answered/frozen; repair it in OpenResearch instead of importing it as a new proposal")
        elif status in _IN_FLIGHT_STATUSES:
            issues.append("source node is still in flight; wait for its OpenResearch run before branching")
        if parent_id and parent_id not in source_ids:
            warnings.append("parent {} for {} is not present in this snapshot".format(parent_id, source_id))
        if options.get("metric_probe") is True:
            issues.append("metric_probe=true is never imported into an executable Biohub proposal")
        if unknown_options:
            warnings.append(
                "{} contains unsupported/non-executable option keys: {}".format(
                    source_id, ", ".join(unknown_options)
                )
            )
        if name in existing:
            issues.append("proposal name already exists in the Biohub ledger")
        if issues:
            blocked.append(
                {
                    "name": name,
                    "source_experiment_id": source_id,
                    "title": title,
                    "reasons": list(dict.fromkeys(issues)),
                }
            )
            continue

        runner_kind = _text(
            _first(
                node,
                "runner_kind",
                "runnerKind",
                default=_first(node.get("biohub", {}) if isinstance(node.get("biohub"), Mapping) else {}, "runner_kind", "runnerKind"),
            ),
            maximum=80,
        ) or None
        execution_ready = runner_kind == "support_pack" and not unknown_options
        if runner_kind not in (None, "support_pack"):
            warnings.append("{} requests runner_kind={} which this bridge does not materialize automatically".format(source_id, runner_kind))

        proposal: dict[str, Any] = {
            "name": name,
            "family": "openresearch_lead",
            "enabled": False,
            "live_eligible": False,
            "proposed_only": True,
            "execution_ready": execution_ready,
            "status": "proposed",
            "competition": competition,
            "command": list(fixed_argv),
            "fixed_run_command": fixed_display,
            "hypothesis": _node_hypothesis(node, title),
            "description": _text(_first(node, "description", "desc", default=""), maximum=4000),
            "tags": _tags(node),
            "requires": ["holdout", "topology", "time", "logs"],
            "notebook_runner_kind": runner_kind,
            "notebook_runner_options": options,
            "metric_probe": False,
            "openresearch": {
                **_source_metadata(node, source_id, parent_id),
                "title": title,
                "source_url": source_url,
                "requested_change_keys": changed_keys,
                "unsupported_option_keys": unknown_options,
            },
        }
        if parent_id:
            proposal["openresearch"]["parent_resolution"] = "snapshot" if parent_id in source_ids else "external"
        proposals.append(proposal)

    return {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "adapter": "openresearch-to-biohub",
        "created_at": imported_at or datetime.now(timezone.utc).isoformat(),
        "competition": competition,
        "source": {
            "kind": "openresearch",
            "url": source_url,
            "project_id": project_id or None,
            "project_name": project_name or None,
            "repo": project_repo or None,
        },
        "contract": {
            "fixed_run_command": fixed_display,
            "fixed_run_argv": list(fixed_argv),
            "require_all_candidates": True,
            "live_eligible": False,
            "metric_probe": False,
        },
        "proposals": proposals,
        "blocked": blocked,
        "warnings": list(dict.fromkeys(warnings)),
        "counts": {
            "input_nodes": len(nodes),
            "proposals": len(proposals),
            "blocked": len(blocked),
        },
    }


def write_openresearch_proposals(
    input_path: Path,
    output_path: Path,
    *,
    fixed_run_command: str | Sequence[str],
    existing_names: Iterable[str] = (),
    competition: str = "biohub-cell-tracking-during-development",
    source_url: str = DEFAULT_SOURCE_URL,
) -> dict[str, Any]:
    """Read a JSON snapshot, write the bridge queue, and return its payload."""

    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    result = build_openresearch_proposals(
        payload,
        fixed_run_command=fixed_run_command,
        existing_names=existing_names,
        competition=competition,
        source_url=source_url,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result

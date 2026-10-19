"""Modal entrypoint for the full-data Biohub tracking cycle.

Examples are documented in ``docs/modal-run.md``.  The default entrypoint is
diagnostic: it never submits to Kaggle unless ``--live-submit`` is provided.
"""

import json
import hashlib
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List

import modal

from biohub_lab.cloud import (
    RemoteRunConfig,
    is_transient_remote_error,
    normalise_gpu_request,
    require_kaggle_api_token,
    retry_delay_seconds,
)
from biohub_lab.cloud_data import dataset_prefixes, entries_for_prefix, parse_file_listing_page
from biohub_lab.bulk import extract_competition_archive
from biohub_lab.local_cache import prepare_data_cache
from biohub_lab.validation_subset import prepare_validation_subset


PROJECT_ROOT = Path(__file__).resolve().parent
REMOTE_PROJECT_ROOT = "/root/biohub"
VOLUME_ROOT = "/mnt/biohub"
MANIFEST_ROOT = Path(VOLUME_ROOT) / "manifests"
VOLUME_NAME = os.environ.get("BIOHUB_MODAL_VOLUME", "biohub-cell-tracking-data")
KAGGLE_SECRET_NAME = os.environ.get("BIOHUB_MODAL_KAGGLE_SECRET", "kaggle-api")
GPU_REQUEST = os.environ.get("BIOHUB_MODAL_GPU", "A100").strip()
GPU_TYPE = normalise_gpu_request(GPU_REQUEST)
CPU_CORES = max(0.125, float(os.environ.get("BIOHUB_MODAL_CPU", "8")))
# Reading directly from the persistent Volume avoids copying the full 80+ GB
# corpus before an experiment can start.  The cache remains an opt-in speed
# optimization for runs where the copy cost is known to be worthwhile.
USE_LOCAL_DATA_CACHE = os.environ.get("BIOHUB_USE_LOCAL_DATA_CACHE", "0") != "0"
LOCAL_DATA_CACHE_ROOT = Path(os.environ.get("BIOHUB_LOCAL_DATA_CACHE_ROOT", "/tmp/biohub-data-cache"))
CACHE_COPY_WORKERS = max(1, int(os.environ.get("BIOHUB_CACHE_COPY_WORKERS", "4")))
VALIDATION_SUBSET_SIZE = max(1, int(os.environ.get("BIOHUB_VALIDATION_SUBSET_SIZE", "24")))
DOWNLOAD_WORKERS = max(1, int(os.environ.get("BIOHUB_MODAL_DOWNLOAD_WORKERS", "4")))
DOWNLOAD_TIMEOUT_SECONDS = max(1, int(os.environ.get("BIOHUB_MODAL_DOWNLOAD_TIMEOUT_SECONDS", "300")))
BULK_DOWNLOAD_TIMEOUT_SECONDS = max(60, int(os.environ.get("BIOHUB_MODAL_BULK_DOWNLOAD_TIMEOUT_SECONDS", "600")))
COMPETITION = os.environ.get("KAGGLE_COMPETITION", "biohub-cell-tracking-during-development")
SUPPORT_PACK_DATASET = os.environ.get(
    "BIOHUB_SUPPORT_PACK_DATASET",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
)
SUPPORT_PACK_DIR = Path(VOLUME_ROOT) / "support-pack"
SUPPORT_PACK_WEIGHT_SHA256 = "12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771"


app = modal.App("biohub-cell-tracking-remote")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
kaggle_secret = modal.Secret.from_name(KAGGLE_SECRET_NAME, required_keys=["KAGGLE_API_TOKEN"])

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "unzip")
    .uv_pip_install(
        "numpy==2.4.6",
        "scipy==1.17.1",
        "zarr==3.1.6",
        "torch>=2.9.1,<3",
        "kaggle>=2.2.4",
        "tracksdata",
        "geff",
        "polars",
        "pyscipopt",
        "ilpy",
        "rustworkx",
        "dask",
        "imagecodecs",
        "pyarrow",
        "sqlalchemy",
    )
    .add_local_dir(PROJECT_ROOT / "src", f"{REMOTE_PROJECT_ROOT}/src", copy=True)
    .add_local_dir(PROJECT_ROOT / "config", f"{REMOTE_PROJECT_ROOT}/config", copy=True)
    .add_local_dir(PROJECT_ROOT / "scripts", f"{REMOTE_PROJECT_ROOT}/scripts", copy=True)
    .add_local_dir(
        PROJECT_ROOT / "external/kaggle-cell-tracking-competition",
        f"{REMOTE_PROJECT_ROOT}/external/kaggle-cell-tracking-competition",
        copy=True,
    )
    .add_local_file(PROJECT_ROOT / "lab.md", f"{REMOTE_PROJECT_ROOT}/lab.md", copy=True)
    .env({"PYTHONPATH": f"{REMOTE_PROJECT_ROOT}/src"})
    .run_commands(
        "python -m pip install --no-cache-dir -e "
        f"{REMOTE_PROJECT_ROOT}/external/kaggle-cell-tracking-competition"
    )
)


def _runtime_env() -> Dict[str, str]:
    environment = dict(os.environ)
    require_kaggle_api_token(environment)
    environment["KAGGLE_CONFIG_DIR"] = "/tmp/kaggle"
    environment["PYTHONPATH"] = f"{REMOTE_PROJECT_ROOT}/src"
    Path("/tmp/kaggle").mkdir(parents=True, exist_ok=True)
    return environment


def _support_pack_ready(path: Path) -> bool:
    weight = path / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
    repo = path / "repo" / "src" / "biohub_tracking"
    if not weight.is_file() or not repo.is_dir():
        return False
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    return digest == SUPPORT_PACK_WEIGHT_SHA256


def _config_uses_support_pack(config_path: str) -> bool:
    candidate = Path(config_path)
    if not candidate.is_absolute():
        candidate = Path(REMOTE_PROJECT_ROOT) / candidate
    if not candidate.is_file():
        return False
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    for experiment in payload.get("experiments", []):
        command = experiment.get("command", []) if isinstance(experiment, dict) else []
        if "BIOHUB_SUPPORT_PACK_DIR" in json.dumps(command):
            return True
    return False


def _ensure_support_pack(environment: Dict[str, str]) -> Path:
    """Download the public learned baseline once and verify its checksum."""

    if _support_pack_ready(SUPPORT_PACK_DIR):
        return SUPPORT_PACK_DIR
    download_dir = Path("/tmp/biohub-support-pack-download")
    if download_dir.exists():
        shutil.rmtree(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "kaggle",
            "datasets",
            "download",
            "-d",
            SUPPORT_PACK_DATASET,
            "--path",
            str(download_dir),
            "--unzip",
        ],
        cwd=REMOTE_PROJECT_ROOT,
        env=environment,
        attempts=3,
    )
    if not _support_pack_ready(download_dir):
        raise RuntimeError("downloaded public support pack failed its repository or weight checksum")
    SUPPORT_PACK_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(download_dir, SUPPORT_PACK_DIR, dirs_exist_ok=True)
    if not _support_pack_ready(SUPPORT_PACK_DIR):
        raise RuntimeError("persisted public support pack failed its weight checksum")
    volume.commit()
    return SUPPORT_PACK_DIR


def _run(command: List[str], *, cwd: str, env: Dict[str, str], attempts: int = 4) -> str:
    for attempt in range(attempts):
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode == 0:
            return completed.stdout
        detail = "{}\n{}".format(completed.stdout, completed.stderr).strip()
        if attempt + 1 >= attempts or not is_transient_remote_error(detail):
            raise RuntimeError("remote command failed ({}): {}".format(completed.returncode, detail[-1000:]))
        time.sleep(retry_delay_seconds(attempt))
    raise AssertionError("unreachable")


def _next_page_token(stdout: str) -> str | None:
    matches = re.findall(r"Next Page Token = ([^\r\n]+)", stdout)
    return matches[-1].strip() if matches else None


def _manifest_path(competition: str) -> Path:
    """Return a safe persistent path for one competition file manifest."""

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(competition).strip()).strip("._")
    if not slug:
        raise ValueError("competition must produce a non-empty manifest name")
    return MANIFEST_ROOT / f"{slug}.json"


def _load_manifest_entries(path: Path) -> List[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries", []) if isinstance(payload, dict) else payload
    if not isinstance(entries, list):
        raise ValueError("cached Kaggle manifest entries must be a JSON list")
    return [entry for entry in entries if isinstance(entry, dict)]


def _fetch_manifest_entries(
    competition: str,
    environment: Dict[str, str],
) -> List[dict[str, Any]]:
    token = None
    seen_tokens = set()
    entries: List[dict[str, Any]] = []
    while True:
        command = [
            "kaggle",
            "competitions",
            "files",
            competition,
            "--format",
            "json",
            "--page-size",
            "200",
        ]
        if token:
            command.extend(["--page-token", token])
        stdout = _run(command, cwd=REMOTE_PROJECT_ROOT, env=environment)
        entries.extend(parse_file_listing_page(stdout))
        next_token = _next_page_token(stdout)
        if not next_token or next_token in seen_tokens:
            break
        seen_tokens.add(next_token)
        token = next_token
    return entries


def _cached_or_fetch_manifest(
    competition: str,
    environment: Dict[str, str],
    *,
    refresh: bool = False,
) -> List[dict[str, Any]]:
    path = _manifest_path(competition)
    if path.is_file() and not refresh:
        return _load_manifest_entries(path)
    entries = _fetch_manifest_entries(competition, environment)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"competition": competition, "entries": entries},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    volume.commit()
    return entries


def _safe_prefix(prefix: str) -> str:
    normalized = str(prefix).strip().lstrip("/")
    parts = Path(normalized).parts
    if not normalized.endswith("/") or ".." in parts or not normalized.startswith(("test/", "train/")):
        raise ValueError("prefix must be a relative test/ or train/ directory")
    return normalized


def _train_prefixes(value: str) -> List[str]:
    prefixes: List[str] = []
    for raw in str(value).split(","):
        stem = raw.strip().strip("/")
        if not stem:
            continue
        if stem.startswith("train/"):
            stem = stem[len("train/") :]
        if stem.endswith(".zarr") or stem.endswith(".geff"):
            base = stem.rsplit(".", 1)[0]
        else:
            base = stem
        prefixes.extend([f"train/{base}.zarr/", f"train/{base}.geff/"])
    return sorted(set(_safe_prefix(prefix) for prefix in prefixes))


def _stage_one_prefix(
    prefix: str,
    competition: str,
    environment: Dict[str, str],
    manifest_path: Path,
    output: Path,
) -> Dict[str, Any]:
    prefix = _safe_prefix(prefix)
    command = [
        "/usr/local/bin/python",
        f"{REMOTE_PROJECT_ROOT}/scripts/download_subset.py",
        "--competition",
        competition,
        "--prefix",
        prefix,
        "--output-dir",
        str(output),
        "--executable",
        "kaggle",
        "--manifest",
        str(manifest_path),
        "--workers",
        str(DOWNLOAD_WORKERS),
        "--timeout-seconds",
        str(DOWNLOAD_TIMEOUT_SECONDS),
    ]
    return json.loads(_run(command, cwd=REMOTE_PROJECT_ROOT, env=environment))


@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60,
    max_containers=1,
)
def discover_prefixes(
    split: str,
    competition: str = COMPETITION,
    refresh: bool = False,
) -> List[str]:
    """Discover root prefixes without downloading any competition data."""

    if split not in {"test", "train"}:
        raise ValueError("split must be test or train")
    environment = _runtime_env()
    entries = _cached_or_fetch_manifest(competition, environment, refresh=refresh)
    return dataset_prefixes(entries, split=split)


@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60,
    max_containers=1,
)
def manifest_summary(
    competition: str = COMPETITION,
    refresh: bool = False,
) -> Dict[str, Any]:
    """Report manifest bytes grouped by dataset prefix without downloading."""

    environment = _runtime_env()
    entries = _cached_or_fetch_manifest(competition, environment, refresh=refresh)
    summary: Dict[str, Any] = {"competition": competition, "splits": {}}
    totals: Dict[str, Dict[str, int]] = {}
    for split in ("test", "train"):
        datasets = []
        for prefix in dataset_prefixes(entries, split=split):
            selected = entries_for_prefix(entries, prefix)
            datasets.append(
                {
                    "prefix": prefix,
                    "files": len(selected),
                    "bytes": sum(int(item.get("size", 0)) for item in selected),
                }
            )
        summary["splits"][split] = datasets
        totals[split] = {
            "datasets": len(datasets),
            "files": sum(item["files"] for item in datasets),
            "bytes": sum(item["bytes"] for item in datasets),
        }
    summary["totals"] = totals
    return summary


@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={VOLUME_ROOT: volume},
    timeout=6 * 60 * 60,
    max_containers=1,
)
def stage_prefix(prefix: str, competition: str = COMPETITION) -> Dict[str, Any]:
    """Download one exact OME-Zarr/GEFF prefix into the persistent Volume."""

    prefix = _safe_prefix(prefix)
    environment = _runtime_env()
    manifest_path = _manifest_path(competition)
    _cached_or_fetch_manifest(competition, environment)
    output = Path(VOLUME_ROOT) / "data"
    output.mkdir(parents=True, exist_ok=True)
    result = _stage_one_prefix(prefix, competition, environment, manifest_path, output)
    volume.commit()
    return result


@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={VOLUME_ROOT: volume},
    timeout=24 * 60 * 60,
    max_containers=1,
)
def stage_prefixes(prefixes: List[str], competition: str = COMPETITION) -> Dict[str, Any]:
    """Stage many prefixes in one worker with a commit after each prefix."""

    normalized = sorted(set(_safe_prefix(prefix) for prefix in prefixes))
    if not normalized:
        raise ValueError("at least one prefix is required")
    environment = _runtime_env()
    manifest_path = _manifest_path(competition)
    _cached_or_fetch_manifest(competition, environment)
    output = Path(VOLUME_ROOT) / "data"
    output.mkdir(parents=True, exist_ok=True)
    report_path = MANIFEST_ROOT / "staging-report.json"
    results_by_prefix: Dict[str, Dict[str, Any]] = {}
    if report_path.is_file():
        cached_report = json.loads(report_path.read_text(encoding="utf-8"))
        for cached in cached_report.get("results", []) if isinstance(cached_report, dict) else []:
            if not isinstance(cached, dict) or not cached.get("prefix"):
                continue
            results_by_prefix[_safe_prefix(str(cached["prefix"]))] = cached
    for prefix in normalized:
        result = _stage_one_prefix(prefix, competition, environment, manifest_path, output)
        results_by_prefix[prefix] = result
        results = [results_by_prefix[key] for key in sorted(results_by_prefix)]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"competition": competition, "results": results}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        volume.commit()
    return {
        "prefixes": len(results_by_prefix),
        "files": sum(int(result.get("files", 0)) for result in results_by_prefix.values()),
        "bytes": sum(int(result.get("bytes", 0)) for result in results_by_prefix.values()),
        "downloaded": sum(int(result.get("downloaded", 0)) for result in results_by_prefix.values()),
        "report_path": str(report_path),
    }


def _download_bulk_archive(competition: str, archive_path: Path) -> int:
    """Download the official competition bundle through one authenticated request."""

    from kaggle.api.kaggle_api_extended import ApiDownloadDataFilesRequest, KaggleApi

    api = KaggleApi()
    with api.build_kaggle_client() as client:
        request = ApiDownloadDataFilesRequest()
        request.competition_name = competition
        response = client.competitions.competition_api_client.download_data_files(request)
        api.download_file(
            response,
            str(archive_path),
            client.http_client(),
            quiet=True,
            resume=True,
            timeout=BULK_DOWNLOAD_TIMEOUT_SECONDS,
        )
    return archive_path.stat().st_size


@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={VOLUME_ROOT: volume},
    timeout=24 * 60 * 60,
    max_containers=1,
)
def stage_bulk(competition: str = COMPETITION, refresh_manifest: bool = False) -> Dict[str, Any]:
    """Stage the complete competition bundle and verify it against the manifest."""

    environment = _runtime_env()
    manifest_path = _manifest_path(competition)
    entries = _cached_or_fetch_manifest(competition, environment, refresh=refresh_manifest)
    expected = {
        str(entry["name"]): int(entry.get("size", 0))
        for entry in entries
        if str(entry.get("name", "")).startswith(("test/", "train/"))
    }
    if not expected:
        raise ValueError("bulk staging manifest has no test/train files")

    output = Path(VOLUME_ROOT) / "data"
    output.mkdir(parents=True, exist_ok=True)
    archive_path = Path("/tmp/biohub-competition-bundle.zip")
    archive_path.unlink(missing_ok=True)
    archive_path.with_name(archive_path.name + ".kaggle-partial").unlink(missing_ok=True)
    root_results: List[Dict[str, Any]] = []
    try:
        archive_bytes = _download_bulk_archive(competition, archive_path)
        for root_name in ("test/", "train/"):
            root_expected = {
                name: size for name, size in expected.items() if name.startswith(root_name)
            }
            if not root_expected:
                continue
            result = extract_competition_archive(archive_path, output, root_expected, prefixes=(root_name,))
            root_results.append(
                {
                    "prefix": root_name,
                    "files": result["files"],
                    "bytes": result["bytes"],
                    "downloaded": result["extracted"],
                    "skipped": result["skipped"],
                }
            )
            volume.commit()
    finally:
        archive_path.unlink(missing_ok=True)
        archive_path.with_name(archive_path.name + ".kaggle-partial").unlink(missing_ok=True)

    report_path = MANIFEST_ROOT / "bulk-staging-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "competition": competition,
                "mode": "bulk_archive",
                "manifest_path": str(manifest_path),
                "archive_bytes": archive_bytes,
                "results": root_results,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    volume.commit()
    return {
        "competition": competition,
        "mode": "bulk_archive",
        "manifest_path": str(manifest_path),
        "archive_bytes": archive_bytes,
        "files": sum(result["files"] for result in root_results),
        "bytes": sum(result["bytes"] for result in root_results),
        "downloaded": sum(result["downloaded"] for result in root_results),
        "skipped": sum(result["skipped"] for result in root_results),
        "report_path": str(report_path),
    }


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60,
)
def create_holdout(
    fraction: float = 0.2,
    seed: int = 20260804,
    replace_existing: bool = False,
) -> Dict[str, Any]:
    """Create a dataset-level holdout once the selected train prefixes exist."""

    root = Path(VOLUME_ROOT)
    manifest = root / "data/holdout/manifest.json"
    source_dir = root / "data/train"
    datasets = sorted(path.stem for path in source_dir.glob("*.zarr") if (source_dir / f"{path.stem}.geff").is_dir())
    if len(datasets) < 2:
        raise ValueError("at least two complete train datasets are required for a holdout")
    if manifest.is_file():
        existing = json.loads(manifest.read_text(encoding="utf-8"))
        existing_datasets = set(existing.get("source_datasets", [])) if isinstance(existing, dict) else set()
        same_split = (
            existing_datasets == set(datasets)
            and float(existing.get("fraction", -1.0)) == float(fraction)
            and int(existing.get("seed", -1)) == int(seed)
        )
        if same_split:
            return existing
        if not replace_existing:
            raise RuntimeError(
                "existing holdout manifest does not match the staged source; "
                "rerun with replace_existing=True to rebuild derived split directories"
            )

    def clear_derived_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for child in path.iterdir():
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    (root / "data").mkdir(parents=True, exist_ok=True)
    train_split_dir = root / "data/train_split"
    holdout_dir = root / "data/holdout"
    if replace_existing:
        clear_derived_directory(train_split_dir)
        clear_derived_directory(holdout_dir)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{REMOTE_PROJECT_ROOT}/src"
    command = [
        "/usr/local/bin/python",
        "-m",
        "biohub_lab.cli",
        "split",
        "--source-dir",
        str(source_dir),
        "--train-dir",
        str(train_split_dir),
        "--holdout-dir",
        str(holdout_dir),
        "--fraction",
        str(fraction),
        "--seed",
        str(seed),
    ]
    stdout = _run(command, cwd=str(root), env=environment)
    volume.commit()
    return json.loads(stdout)


def _remap_text_paths(path: Path, source: str, target: str) -> None:
    """Replace ephemeral run paths in persisted text artifacts."""

    if not path.is_file() or path.suffix.lower() not in {".csv", ".json", ".jsonl", ".log", ".md"}:
        return
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return
    updated = content.replace(source, target)
    if updated != content:
        path.write_text(updated, encoding="utf-8")


def _persist_local_runs(local_runs: Path, persistent_runs: Path) -> None:
    """Copy candidate artifacts from local disk to the Volume safely."""

    if not local_runs.is_dir():
        return
    persistent_runs.mkdir(parents=True, exist_ok=True)
    for source in local_runs.iterdir():
        target = persistent_runs / source.name
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
        elif source.is_file():
            shutil.copy2(source, target)
    for path in persistent_runs.rglob("*"):
        _remap_text_paths(path, str(local_runs), str(persistent_runs))


@app.function(
    image=image,
    gpu=GPU_TYPE,
    cpu=CPU_CORES,
    secrets=[kaggle_secret],
    volumes={VOLUME_ROOT: volume},
    timeout=8 * 60 * 60,
    max_containers=1,
)
def run_cycle(
    duration_hours: float = 4.0,
    data_scope: str = "full",
    live_submit: bool = False,
    config_path: str = "config/experiments.json",
) -> Dict[str, Any]:
    """Run the existing controller on the persistent Volume."""

    config = RemoteRunConfig(
        volume_name=VOLUME_NAME,
        gpu=GPU_REQUEST or "cpu",
        data_scope=data_scope,
        duration_hours=duration_hours,
        live_submit=live_submit,
        remote_project_root=Path(REMOTE_PROJECT_ROOT),
        volume_root=Path(VOLUME_ROOT),
        config_path=Path(config_path),
    )
    environment = _runtime_env()
    environment.update(config.environment())
    uses_support_pack = _config_uses_support_pack(config_path)
    if uses_support_pack:
        environment["BIOHUB_SUPPORT_PACK_DIR"] = str(_ensure_support_pack(environment))
    root = Path(VOLUME_ROOT)
    subset_test_dir = config.layout.data_dir / "test_subset"
    test_dir = subset_test_dir if data_scope == "subset" and subset_test_dir.is_dir() else config.layout.test_dir
    environment["TEST_DATA_DIR"] = str(test_dir)
    required = [test_dir, config.layout.train_dir, config.layout.holdout_dir]
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise FileNotFoundError("remote data directories are missing: {}".format(", ".join(missing)))
    validation_data_dir = config.layout.train_dir
    if uses_support_pack and environment.get("BIOHUB_FAST_VALIDATION", "1") != "0":
        validation_subset = prepare_validation_subset(
            config.layout.train_dir,
            config.layout.data_dir / "validation_subset_{}".format(VALIDATION_SUBSET_SIZE),
            limit=VALIDATION_SUBSET_SIZE,
        )
        validation_data_dir = config.layout.data_dir / "validation_subset_{}".format(VALIDATION_SUBSET_SIZE)
        environment["DATA_DIR"] = str(validation_data_dir)
        environment["BIOHUB_VALIDATION_SUBSET_COUNT"] = str(validation_subset["count"])
    local_data_cache = None
    if data_scope == "full" and USE_LOCAL_DATA_CACHE:
        local_data_cache = prepare_data_cache(
            config.layout.data_dir,
            LOCAL_DATA_CACHE_ROOT,
            workers=CACHE_COPY_WORKERS,
        )
        test_dir = local_data_cache["test"]
        environment["TEST_DATA_DIR"] = str(test_dir)
        environment["DATA_DIR"] = str(local_data_cache["train_split"])
        environment["HOLDOUT_DATA_DIR"] = str(local_data_cache["holdout"])
    root.mkdir(parents=True, exist_ok=True)
    lab_path = root / "lab.md"
    if not lab_path.exists():
        shutil.copyfile(f"{REMOTE_PROJECT_ROOT}/lab.md", lab_path)
    local_runs = Path("/tmp/biohub-runs")
    shutil.rmtree(local_runs, ignore_errors=True)
    local_runs.mkdir(parents=True, exist_ok=True)
    command = config.command(output_dir=local_runs)
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    persistent_runs = config.layout.runs_dir
    _persist_local_runs(local_runs, persistent_runs)
    for relative_path in (
        "outputs/submission_plan.json",
        "outputs/ranking.csv",
        "outputs/proposals.json",
        "lab.md",
    ):
        _remap_text_paths(root / relative_path, str(local_runs), str(persistent_runs))
    (root / "remote_cycle.stdout.log").write_text(
        completed.stdout.replace(str(local_runs), str(persistent_runs)), encoding="utf-8"
    )
    (root / "remote_cycle.stderr.log").write_text(
        completed.stderr.replace(str(local_runs), str(persistent_runs)), encoding="utf-8"
    )
    volume.commit()
    plan_path = root / "outputs/submission_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.is_file() else None
    if completed.returncode != 0:
        raise RuntimeError("remote cycle failed with return code {}".format(completed.returncode))
    return {
        "returncode": completed.returncode,
        "gpu": GPU_REQUEST or "cpu",
        "cpu_cores": CPU_CORES,
        "local_data_cache": str(local_data_cache["root"]) if local_data_cache else None,
        "validation_data_dir": str(validation_data_dir),
        "validation_subset_count": VALIDATION_SUBSET_SIZE if validation_data_dir != config.layout.train_dir else None,
        "data_scope": data_scope,
        "live_submit": live_submit,
        "plan_path": str(plan_path),
        "candidate_count": len(plan.get("candidates", [])) if plan else 0,
    }


@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={VOLUME_ROOT: volume},
    timeout=10 * 60,
)
def launch_cycle(
    duration_hours: float = 4.0,
    data_scope: str = "full",
    live_submit: bool = False,
    config_path: str = "config/experiments.json",
) -> Dict[str, Any]:
    """Start a long cycle and return without waiting for its result."""

    call = run_cycle.spawn(duration_hours, data_scope, live_submit, config_path)
    return {
        "status": "spawned",
        "function_call_id": call.object_id,
        "duration_hours": duration_hours,
        "data_scope": data_scope,
        "live_submit": live_submit,
        "config_path": config_path,
    }


@app.function(
    image=image,
    secrets=[kaggle_secret],
    volumes={VOLUME_ROOT: volume},
    timeout=60 * 60,
)
def submit_existing_plan(retry_failed: bool = False, recover_reserved: bool = False) -> Dict[str, Any]:
    """Submit a persisted plan without rerunning the full experiment cycle."""

    environment = _runtime_env()
    environment["BIOHUB_ALLOW_LIVE_SUBMISSIONS"] = "1"
    environment["BIOHUB_SUBMISSION_MODE"] = "notebook"
    environment["BIOHUB_KAGGLE_DATASET"] = os.environ.get(
        "BIOHUB_KAGGLE_DATASET", "your-kaggle-user/biohub-cell-candidates"
    )
    root = Path(VOLUME_ROOT)
    source_plan = root / "outputs/submission_plan.json"
    if not source_plan.is_file():
        raise FileNotFoundError("persisted submission plan is missing: {}".format(source_plan))

    payload = json.loads(source_plan.read_text(encoding="utf-8"))
    if recover_reserved:
        from biohub_lab.budget import SubmissionBudget

        state_path = root / "state" / "submissions.json"
        budget = SubmissionBudget(state_path, daily_limit=5)
        experiments = [str(item.get("experiment")) for item in payload.get("candidates", [])]
        budget.release_reserved(experiments)
        volume.commit()
    changed = False
    if str(payload.get("submission_mode", "")).lower() != "notebook":
        payload["submission_mode"] = "notebook"
        changed = True
    if not payload.get("notebook_dataset"):
        payload["notebook_dataset"] = environment["BIOHUB_KAGGLE_DATASET"]
        changed = True
    for candidate in payload.get("candidates", []):
        raw_artifact = candidate.get("artifact")
        if not raw_artifact:
            continue
        artifact = Path(str(raw_artifact))
        if artifact.is_file():
            continue
        persistent = root / "runs" / artifact.parent.name / artifact.name
        if persistent.is_file():
            candidate["artifact"] = str(persistent)
            changed = True
    plan_path = root / "outputs/submission_plan.recovery.json"
    if changed:
        plan_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        plan_path = source_plan

    command = [
        "/usr/local/bin/python",
        "-m",
        "biohub_lab.cli",
        "submit",
        "--plan",
        str(plan_path),
        "--state",
        str(root / "state/submissions.json"),
        "--live-submit",
    ]
    if retry_failed:
        command.append("--retry-failed")
    completed = subprocess.run(
        command,
        cwd=str(root),
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        if completed.returncode != 0:
            detail = "{}\n{}".format(completed.stdout, completed.stderr).strip()
            raise RuntimeError("persisted submission failed: {}".format(detail[-2000:]))
        results = json.loads(completed.stdout)
        (root / "outputs/submission_results.json").write_text(
            json.dumps(results, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        volume.commit()
        return {"returncode": completed.returncode, "results": results}
    finally:
        if changed:
            plan_path.unlink(missing_ok=True)
            volume.commit()


@app.function(
    image=image,
    volumes={VOLUME_ROOT: volume},
    timeout=10 * 60,
)
def register_external_submission(experiment: str, reference: str) -> Dict[str, Any]:
    """Reconcile a submission accepted outside the persisted controller ledger."""

    from biohub_lab.budget import SubmissionBudget

    state_path = Path(VOLUME_ROOT) / "state" / "submissions.json"
    budget = SubmissionBudget(state_path, daily_limit=5)
    if experiment not in budget.attempted_experiments([experiment]):
        budget.record(experiment, status="submitted", reference=str(reference))
    volume.commit()
    return {"experiment": experiment, "reference": str(reference), "status": "submitted"}


@app.local_entrypoint()
def main(
    mode: str = "discover",
    train_prefixes: str = "",
    data_scope: str = "full",
    live_submit: bool = False,
    duration_hours: float = 4.0,
    refresh_manifest: bool = False,
    all_train: bool = False,
    retry_failed: bool = False,
    recover_reserved: bool = False,
    experiment: str = "",
    reference: str = "",
) -> None:
    """Discover, stage, or run the remote cycle.

    ``train_prefixes`` is a comma-separated list of train stems.  Test data is
    discovered automatically; train data is explicit to avoid an accidental
    multi-terabyte staging job.
    """

    if mode == "discover":
        print(
            json.dumps(
                {
                    "test": discover_prefixes.remote("test", COMPETITION, refresh_manifest),
                    "train": discover_prefixes.remote("train", COMPETITION, False),
                },
                indent=2,
            )
        )
        return
    if mode == "summary":
        print(json.dumps(manifest_summary.remote(COMPETITION, refresh_manifest), indent=2, sort_keys=True))
        return
    if mode == "stage":
        if all_train and train_prefixes.strip():
            raise ValueError("use either --all-train or --train-prefixes, not both")
        if all_train:
            results = stage_bulk.remote(COMPETITION, refresh_manifest)
            holdout = create_holdout.remote(0.2, 20260804, True)
            print(json.dumps({"mode": "bulk_archive", "downloads": results, "holdout": holdout}, indent=2, default=str))
            return
        test = discover_prefixes.remote("test", COMPETITION, refresh_manifest)
        train = discover_prefixes.remote("train", COMPETITION, refresh_manifest) if all_train else _train_prefixes(train_prefixes)
        prefixes = test + train
        results = stage_prefixes.remote(prefixes, COMPETITION)
        holdout = create_holdout.remote() if train else None
        print(json.dumps({"prefixes": len(prefixes), "downloads": results, "holdout": holdout}, indent=2, default=str))
        return
    if mode == "stage-bulk":
        if train_prefixes.strip():
            raise ValueError("stage-bulk downloads the complete competition and does not accept --train-prefixes")
        results = stage_bulk.remote(COMPETITION, refresh_manifest)
        holdout = create_holdout.remote(0.2, 20260804, True)
        print(json.dumps({"mode": "bulk_archive", "downloads": results, "holdout": holdout}, indent=2, default=str))
        return
    if mode == "create-holdout":
        holdout = create_holdout.remote(0.2, 20260804, True)
        print(json.dumps(holdout, indent=2, sort_keys=True, default=str))
        return
    if mode == "run":
        result = run_cycle.remote(
            duration_hours,
            data_scope,
            live_submit,
            os.environ.get("BIOHUB_REMOTE_CONFIG", "config/experiments.json"),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if mode in {"launch", "run-async"}:
        result = launch_cycle.remote(
            duration_hours,
            data_scope,
            live_submit,
            os.environ.get("BIOHUB_REMOTE_CONFIG", "config/experiments.json"),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if mode == "submit":
        if not live_submit:
            raise ValueError("mode submit requires --live-submit")
        result = submit_existing_plan.remote(retry_failed, recover_reserved)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return
    if mode == "register":
        if not experiment.strip() or not reference.strip():
            raise ValueError("mode register requires --experiment and --reference")
        result = register_external_submission.remote(experiment.strip(), reference.strip())
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if mode == "stage-and-run":
        if all_train and train_prefixes.strip():
            raise ValueError("use either --all-train or --train-prefixes, not both")
        if all_train:
            results = stage_bulk.remote(COMPETITION, refresh_manifest)
            holdout = create_holdout.remote(0.2, 20260804, True)
            result = run_cycle.remote(
                duration_hours,
                data_scope,
                live_submit,
                os.environ.get("BIOHUB_REMOTE_CONFIG", "config/experiments.json"),
            )
            print(json.dumps({"mode": "bulk_archive", "downloads": results, "holdout": holdout, "run": result}, indent=2, default=str))
            return
        test = discover_prefixes.remote("test", COMPETITION, refresh_manifest)
        train = discover_prefixes.remote("train", COMPETITION, refresh_manifest) if all_train else _train_prefixes(train_prefixes)
        prefixes = test + train
        results = stage_prefixes.remote(prefixes, COMPETITION)
        holdout = create_holdout.remote() if train else None
        result = run_cycle.remote(duration_hours, data_scope, live_submit)
        print(json.dumps({"prefixes": len(prefixes), "downloads": results, "holdout": holdout, "run": result}, indent=2, default=str))
        return
    raise ValueError("mode must be discover, summary, stage, stage-bulk, create-holdout, run, submit, register, or stage-and-run")

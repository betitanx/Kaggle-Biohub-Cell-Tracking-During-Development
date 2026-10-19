"""Report environment prerequisites without touching credentials or datasets."""

import json
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def _kaggle_auth_status() -> Dict[str, Any]:
    """Report credential presence without reading or returning secret values."""

    configured_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    config_dir = Path(configured_dir).expanduser() if configured_dir else Path.home() / ".kaggle"
    sources: List[str] = []
    if os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_TOKEN"):
        sources.append("environment")
    if (config_dir / "access_token").is_file():
        sources.append("access_token")
    if (config_dir / "kaggle.json").is_file():
        sources.append("kaggle.json")
    # Kaggle CLI 2.x stores OAuth refresh credentials in this fixed user path.
    if (Path.home() / ".kaggle" / "credentials.json").is_file():
        sources.append("oauth_credentials")
    return {
        "config_dir": str(config_dir),
        "configured": bool(sources),
        "sources": sources,
    }


def check_environment(
    packages: Iterable[str] = ("numpy", "scipy", "zarr", "torch", "tracksdata", "ultrack", "cellpose", "stardist", "kaggle"),
    data_dir: Optional[Path] = None,
    extra_paths: Optional[Mapping[str, Path]] = None,
    external_python: Optional[str] = None,
    external_packages: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    package_names = [str(package) for package in packages]
    package_checks = [
        {"name": package, "available": importlib.util.find_spec(package) is not None}
        for package in package_names
    ]
    data_path = Path(data_dir) if data_dir else None
    paths: Dict[str, Dict[str, Any]] = {}
    if data_path is not None:
        paths["data_dir"] = {"path": str(data_path), "exists": data_path.is_dir()}
    for name, raw_path in (extra_paths or {}).items():
        path = Path(raw_path) if raw_path else Path("")
        paths[str(name)] = {"path": str(path) if raw_path else None, "exists": bool(raw_path and path.exists())}
    kaggle_auth = _kaggle_auth_status()
    report: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_supported_by_control": sys.version_info >= (3, 9),
        "kaggle_cli": shutil.which("kaggle") is not None,
        "kaggle_auth_configured": kaggle_auth["configured"],
        "kaggle_auth_sources": kaggle_auth["sources"],
        "kaggle_config_dir": kaggle_auth["config_dir"],
        "live_submission_consent": os.environ.get("BIOHUB_ALLOW_LIVE_SUBMISSIONS") == "1",
        "data_dir": str(data_path) if data_path else None,
        "data_dir_exists": data_path.is_dir() if data_path else False,
        "paths": paths,
        "packages": package_checks,
    }
    if external_python:
        report["external_python"] = _probe_external_python(
            external_python,
            list(external_packages) if external_packages is not None else package_names,
        )
    else:
        report["external_python"] = None
    return report


def _probe_external_python(executable: str, packages: List[str]) -> Dict[str, Any]:
    """Probe package metadata in a candidate interpreter without importing packages."""

    requested = str(executable)
    resolved = shutil.which(requested) or requested
    available = bool(shutil.which(requested) or Path(resolved).is_file())
    result: Dict[str, Any] = {
        "path": str(resolved),
        "available": available,
        "python": None,
        "packages": [{"name": package, "available": False} for package in packages],
    }
    if not available:
        return result
    probe = (
        "import importlib.util,json,sys;"
        "names=json.loads(sys.argv[1]);"
        "print(json.dumps({'python':sys.version.split()[0],"
        "'packages':[{'name':name,'available':importlib.util.find_spec(name) is not None} for name in names]}))"
    )
    try:
        completed = subprocess.run(
            [str(resolved), "-c", probe, json.dumps(packages)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
        if completed.returncode != 0:
            result["error"] = "external interpreter probe failed"
            return result
        payload = json.loads(completed.stdout)
        result["python"] = payload.get("python")
        result["packages"] = payload.get("packages", result["packages"])
    except (OSError, subprocess.TimeoutExpired, TypeError, ValueError, json.JSONDecodeError):
        result["error"] = "external interpreter probe failed"
    return result

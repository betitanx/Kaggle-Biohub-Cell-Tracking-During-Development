"""Preflight checks for catalogued experiment requirements."""

import importlib.util
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


_PACKAGE_ALIASES = {
    "tracking-cellmot": ("tracking_cellmot",),
    "cellpose": ("cellpose",),
    "stardist": ("stardist",),
    "ultrack": ("ultrack",),
    "cellpose-or-stardist": ("cellpose", "stardist"),
}


def _has_import(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _missing_env_path(name: str, value: str, project_root: Path) -> bool:
    if name.endswith("_PYTHON"):
        return not (Path(value).is_file() or shutil.which(value) is not None)
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return not path.exists()


def missing_requirements(
    spec: Mapping[str, Any],
    env: Optional[Mapping[str, str]] = None,
    project_root: Optional[Path] = None,
) -> List[str]:
    """Return unmet ``requires`` entries without installing or downloading anything."""

    merged_env = dict(os.environ)
    if env:
        merged_env.update(env)
    root = Path(project_root or Path.cwd())
    missing: List[str] = []
    for raw_requirement in spec.get("requires", []) or []:
        requirement = str(raw_requirement)
        if requirement.isupper() and "_" in requirement:
            value = merged_env.get(requirement)
            if not value or _missing_env_path(requirement, value, root):
                missing.append(requirement)
            continue
        if "/" in requirement or requirement.startswith("."):
            path = Path(requirement)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                missing.append(requirement)
            continue
        import_names = _PACKAGE_ALIASES.get(requirement, (requirement.replace("-", "_"),))
        if not any(_has_import(import_name) for import_name in import_names):
            missing.append(requirement)
    return missing

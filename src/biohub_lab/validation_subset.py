"""Create deterministic, read-only views for fast in-domain validation."""

import json
from pathlib import Path
from typing import Dict, List


def prepare_validation_subset(source_dir: Path, target_dir: Path, limit: int = 24) -> Dict[str, object]:
    """Create a symlink view of a bounded set of paired train Zarr/GEFF files.

    The full holdout remains untouched. This view is only for ranking a cycle
    quickly and is reproducible because dataset stems are selected in sorted
    order. A manifest prevents silently reusing a view with a different size.
    """

    source_dir = Path(source_dir).resolve()
    target_dir = Path(target_dir)
    limit = int(limit)
    if limit < 1:
        raise ValueError("validation subset limit must be positive")
    datasets = sorted(source_dir.glob("*.zarr"))
    if not datasets:
        raise FileNotFoundError("validation source contains no .zarr datasets: {}".format(source_dir))
    selected = datasets[:limit]
    names = [path.stem for path in selected]
    paired_geffs = {}
    for source in selected:
        geff = source.with_suffix(".geff")
        if not geff.exists():
            raise FileNotFoundError("validation source is missing paired GEFF: {}".format(geff))
        paired_geffs[source.stem] = geff
    manifest_path = target_dir / "manifest.json"
    if target_dir.is_dir() and manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            existing = None
        if isinstance(existing, dict) and existing.get("dataset_names") == names:
            for source in selected:
                target_zarr = target_dir / source.name
                target_geff = target_dir / (source.stem + ".geff")
                if not target_zarr.exists():
                    target_zarr.symlink_to(source, target_is_directory=True)
                if not target_geff.exists():
                    target_geff.symlink_to(paired_geffs[source.stem])
            existing["ground_truth_count"] = len(paired_geffs)
            manifest_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return existing
        raise RuntimeError("validation subset already exists with a different manifest: {}".format(target_dir))
    if target_dir.exists():
        raise RuntimeError("validation subset target is not an empty managed directory: {}".format(target_dir))
    target_dir.mkdir(parents=True, exist_ok=False)
    for source in selected:
        (target_dir / source.name).symlink_to(source, target_is_directory=True)
        geff = paired_geffs[source.stem]
        (target_dir / geff.name).symlink_to(geff)
    manifest = {
        "source_dir": str(source_dir),
        "dataset_names": names,
        "count": len(names),
        "ground_truth_count": len(paired_geffs),
        "selection": "lexicographic_prefix",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest

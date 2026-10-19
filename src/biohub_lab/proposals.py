"""Expand explicit, bounded experiment variants from the catalog."""

import copy
import re
from typing import Any, Dict, Iterable, List


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _slug(value: object) -> str:
    text = _SAFE_NAME.sub("-", str(value)).strip("-.")
    return text or "variant"


def expand_experiment_specs(specs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return catalog entries plus their declared finite proposal variants.

    Variants are intentionally explicit rather than generated from unbounded
    search.  Each variant may append command ``args`` and override scalar
    fields such as timeout or hypothesis metadata.
    """

    expanded: List[Dict[str, Any]] = []
    for original in specs:
        base = copy.deepcopy(dict(original))
        variants = base.pop("variants", None)
        if not variants:
            expanded.append(base)
            continue
        if not isinstance(variants, list):
            raise ValueError("experiment variants must be a list")
        for variant in variants:
            if not isinstance(variant, dict):
                raise ValueError("each experiment variant must be an object")
            name = variant.get("name")
            if not name:
                raise ValueError("each experiment variant needs a name")
            item = copy.deepcopy(base)
            item["name"] = "{}__{}".format(base["name"], _slug(name))
            args = variant.get("args", [])
            if not isinstance(args, list):
                raise ValueError("experiment variant args must be a list")
            item["command"] = list(item["command"]) + [str(arg) for arg in args]
            overrides = variant.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError("experiment variant overrides must be an object")
            item.update(copy.deepcopy(overrides))
            if "notebook_runner_det_threshold" in overrides:
                options = dict(item.get("notebook_runner_options") or {})
                options["det_threshold"] = float(overrides["notebook_runner_det_threshold"])
                item["notebook_runner_options"] = options
            item["variant"] = copy.deepcopy(variant)
            item["tags"] = list(dict.fromkeys(list(item.get("tags", [])) + ["proposal", _slug(name)]))
            expanded.append(item)
    return expanded

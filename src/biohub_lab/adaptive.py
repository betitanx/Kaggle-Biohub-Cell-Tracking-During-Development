"""Bounded, result-aware experiment proposals.

This is a small operationalisation of the paper's weakness principle: a
candidate with a strong holdout/in-domain ratio earns tests of less-specific
neighbours; a candidate with a poor ratio earns precision-oriented neighbours.
It is a proposal mechanism, not an assertion that either direction is
universally better.
"""

import copy
from typing import Any, Dict, Iterable, List

from .ranking import rank_records


def _direction_order(best: Dict[str, Any]) -> List[str]:
    decision = best.get("decision")
    if decision == "promising_proxy":
        return ["weaker", "precision"]
    if decision == "not_promising_proxy":
        return ["precision", "weaker"]
    return ["weaker", "precision"]


def propose_next_experiments(
    specs: Iterable[Dict[str, Any]],
    records: Iterable[Dict[str, Any]],
    max_proposals: int = 3,
) -> List[Dict[str, Any]]:
    """Create a finite next-step queue from ``adaptive_variants`` metadata."""

    if max_proposals <= 0:
        return []
    records = list(records)
    completed_names = {str(record.get("experiment")) for record in records}
    ranked = rank_records(records)
    best_by_family: Dict[str, Dict[str, Any]] = {}
    for record in ranked:
        family = str(record.get("family", "unknown"))
        if family not in best_by_family:
            best_by_family[family] = record

    proposals: List[Dict[str, Any]] = []
    for original in specs:
        if len(proposals) >= max_proposals or not original.get("enabled", False):
            continue
        variants = original.get("adaptive_variants", [])
        if not isinstance(variants, list) or not variants:
            continue
        base = copy.deepcopy(dict(original))
        base.pop("adaptive_variants", None)
        family = str(base.get("family", "unknown"))
        best = best_by_family.get(family, {})
        directions = _direction_order(best)
        ordered = sorted(
            enumerate(variants),
            key=lambda pair: (
                directions.index(str(pair[1].get("direction", "weaker")))
                if str(pair[1].get("direction", "weaker")) in directions
                else len(directions),
                pair[0],
            ),
        )
        for _index, variant in ordered:
            if len(proposals) >= max_proposals:
                break
            if not isinstance(variant, dict) or not variant.get("name"):
                continue
            name = "{}__adaptive__{}".format(base["name"], variant["name"])
            if name in completed_names:
                continue
            direction = str(variant.get("direction", "weaker"))
            if best.get("decision") == "promising_proxy":
                reason = "promising_holdout_test_weaker_hypothesis" if direction == "weaker" else "promising_holdout_precision_ablation"
            elif best.get("decision") == "not_promising_proxy":
                reason = "overfit_proxy_test_precision_hypothesis" if direction == "precision" else "overfit_proxy_weaker_ablation"
            else:
                reason = "bootstrap_test_{}_hypothesis".format(direction)
            item = copy.deepcopy(base)
            item["name"] = name
            args = variant.get("args", [])
            if not isinstance(args, list):
                raise ValueError("adaptive variant args must be a list")
            item["command"] = list(item["command"]) + [str(arg) for arg in args]
            overrides = variant.get("overrides", {})
            if not isinstance(overrides, dict):
                raise ValueError("adaptive variant overrides must be an object")
            item.update(copy.deepcopy(overrides))
            item["variant"] = copy.deepcopy(variant)
            item["variant"]["adaptive"] = True
            item["proposal_reason"] = reason
            item["tags"] = list(dict.fromkeys(list(item.get("tags", [])) + ["adaptive", direction]))
            item["hypothesis"] = "{} [proposal: {}]".format(item.get("hypothesis", ""), reason)
            proposals.append(item)
            completed_names.add(name)
    return proposals

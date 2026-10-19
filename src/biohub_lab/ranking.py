"""Rank experiment evidence while preferring out-of-sample generalisation."""

import math
from typing import Dict, Iterable, List


def _float(record: Dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(record.get(key, default))
    except (TypeError, ValueError):
        return default


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def rank_records(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    """Return completed, valid records ordered by a weakness-aware score."""

    records = [dict(record) for record in records]
    family_counts = {}
    for record in records:
        family = str(record.get("family", "unknown"))
        family_counts[family] = family_counts.get(family, 0) + 1
    ranked: List[Dict[str, object]] = []
    for original in records:
        record = dict(original)
        if record.get("status") not in {"completed", "success"}:
            continue
        if record.get("artifact_valid", True) is not True:
            continue
        if "local_score" not in record:
            continue
        local_raw = _float(record, "local_score")
        if not math.isfinite(local_raw):
            continue
        local_score = _clamp(local_raw)
        in_domain_raw = _float(record, "in_domain_score", local_score)
        in_domain = _clamp(in_domain_raw if math.isfinite(in_domain_raw) else local_score)
        holdout_raw = record.get("holdout_score")
        try:
            holdout_value = float(holdout_raw) if holdout_raw is not None else float("nan")
        except (TypeError, ValueError):
            holdout_value = float("nan")
        if not math.isfinite(holdout_value):
            holdout = 0.0
            weakness_proxy = 0.0
            record["holdout_available"] = False
        else:
            holdout = _clamp(holdout_value)
            weakness_proxy = _clamp(holdout / max(in_domain, 1e-9))
            record["holdout_available"] = True
        specificity_penalty = max(0.0, in_domain - holdout)
        family = str(record.get("family", "unknown"))
        diversity_bonus = 1.0 / max(1, family_counts.get(family, 1))
        selection_score = (
            0.65 * local_score
            + 0.25 * weakness_proxy
            - 0.10 * specificity_penalty
            + 0.05 * diversity_bonus
        )
        if not record["holdout_available"]:
            decision = "inconclusive"
        elif weakness_proxy >= 0.90:
            decision = "promising_proxy"
        elif weakness_proxy < 0.75:
            decision = "not_promising_proxy"
        else:
            decision = "inconclusive"
        record.update(
            {
                "local_score": local_score,
                "in_domain_score": in_domain,
                "holdout_score": holdout,
                "weakness_proxy": weakness_proxy,
                "specificity_penalty": specificity_penalty,
                "diversity_bonus": diversity_bonus,
                "selection_score": selection_score,
                "decision": decision,
            }
        )
        ranked.append(record)
    return sorted(ranked, key=lambda item: (-_float(item, "selection_score"), str(item.get("experiment", ""))))


def select_submission_candidates(records: Iterable[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
    """Select high-ranking artifacts while using distinct method families first."""

    if limit <= 0:
        return []
    ordered = sorted(records, key=lambda item: (-_float(item, "selection_score"), str(item.get("experiment", ""))))
    # Prefer live real candidates whenever possible.  When a diagnostic subset
    # makes every real result non-live, still prefer those real measurements
    # over synthetic smoke scores; smoke is only a fallback when no real
    # candidate exists in the cycle.
    live_ordered = [
        item
        for item in ordered
        if item.get("live_eligible", True) is not False
        and str(item.get("family", "")) not in {"smoke", "synthetic_quality"}
        and str(item.get("evaluation_scope", "")) != "synthetic_diagnostic_only"
    ]
    prefer_live = bool(live_ordered)
    if live_ordered:
        ordered = live_ordered
    else:
        real_ordered = [
            item
            for item in ordered
            if str(item.get("family", "")) not in {"smoke", "synthetic_quality"}
            and str(item.get("evaluation_scope", "")) != "synthetic_diagnostic_only"
        ]
        if real_ordered:
            ordered = real_ordered
    selected: List[Dict[str, object]] = []
    used_families = set()
    for require_new_family in (True, False):
        for record in ordered:
            if len(selected) >= limit or record in selected:
                continue
            if prefer_live and record.get("live_eligible", True) is False:
                continue
            if record.get("holdout_available", True) is False:
                continue
            family = str(record.get("family", "unknown"))
            if require_new_family and family in used_families:
                continue
            if not record.get("artifact"):
                continue
            selected.append(record)
            used_families.add(family)
    return selected

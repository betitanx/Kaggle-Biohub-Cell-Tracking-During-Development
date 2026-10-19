"""Keep the generated execution summary in sync with lab.md."""

from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from .ranking import rank_records


START = "<!-- AUTO-RESULTS-START -->"
END = "<!-- AUTO-RESULTS-END -->"
SUBMISSION_START = "<!-- AUTO-SUBMISSIONS-START -->"
SUBMISSION_END = "<!-- AUTO-SUBMISSIONS-END -->"
PROPOSAL_START = "<!-- AUTO-PROPOSALS-START -->"
PROPOSAL_END = "<!-- AUTO-PROPOSALS-END -->"


def _artifact_for_log(lab_path: Path, raw_artifact: Any) -> str:
    if not raw_artifact:
        return "—"
    candidate = Path(str(raw_artifact))
    if not candidate.is_absolute():
        candidate = lab_path.parent / candidate
    return str(raw_artifact) if candidate.is_file() else "—"


def update_lab_results(lab_path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path = Path(lab_path)
    text = path.read_text(encoding="utf-8")
    records = list(records)
    ranked_by_run = {item.get("run_id"): item for item in rank_records(records)}
    rows: List[str] = []
    for record in records:
        enriched = ranked_by_run.get(record.get("run_id"), record)
        artifact = _artifact_for_log(path, record.get("artifact"))
        data_scope = str(record.get("data_scope", "full"))
        scope = record.get("evaluation_scope", "competition")
        if data_scope != "full" and scope == "competition":
            scope = "{}_diagnostic_only".format(data_scope)
        local = record.get("local_score", "—")
        if local in (None, "—") and record.get("diagnostic_score") is not None:
            try:
                local = "diag:{:.4f}".format(float(record.get("diagnostic_score")))
            except (TypeError, ValueError):
                local = "diag:—"
        rows.append(
            "| {date} | {experiment} | {scope} | {local} | {holdout} | {weakness} | {decision} | {status} | `{artifact}` |".format(
                date=str(record.get("finished_at", record.get("started_at", "")))[:10],
                experiment=record.get("experiment", ""),
                scope=scope,
                local=local,
                holdout=record.get("holdout_score", "—"),
                weakness=enriched.get("weakness_proxy", "—"),
                decision=enriched.get("decision", "—"),
                status=record.get("status", ""),
                artifact=artifact,
            )
        )
    body = "\n".join(rows) or "_Nenhuma execução registrada._"
    generated = "{}\n{}\n{}".format(START, body, END)
    if START in text and END in text:
        before = text.split(START, 1)[0]
        after = text.split(END, 1)[1]
        path.write_text(before + generated + after, encoding="utf-8")
    else:
        path.write_text(text.rstrip() + "\n\n## 12. Registro automático de execuções\n\n" + generated + "\n", encoding="utf-8")


def update_lab_submissions(lab_path: Path, results: Iterable[Dict[str, Any]]) -> None:
    """Keep the live/dry-run Kaggle submission ledger in lab.md."""

    path = Path(lab_path)
    text = path.read_text(encoding="utf-8")
    rows: List[str] = []
    for result in results:
        date_value = result.get("submitted_at", result.get("finished_at"))
        date = str(date_value)[:10] if date_value else datetime.now(timezone.utc).date().isoformat()
        artifact = result.get("artifact", "—")
        rows.append(
            "| {date} | {experiment} | {status} | {reference} | {score} | `{artifact}` |".format(
                date=date,
                experiment=result.get("experiment", ""),
                status=result.get("status", ""),
                reference=result.get("reference", "—"),
                score=result.get("public_score", "—"),
                artifact=artifact,
            )
        )
    existing_rows: List[str] = []
    if SUBMISSION_START in text and SUBMISSION_END in text:
        existing_block = text.split(SUBMISSION_START, 1)[1].split(SUBMISSION_END, 1)[0]
        for line in existing_block.splitlines():
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                existing_rows.append(stripped)

    def row_key(row: str) -> str:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if len(cells) >= 4 and cells[3] not in ("", "—"):
            return "reference:" + cells[3]
        if len(cells) >= 2:
            return "experiment:" + cells[0] + ":" + cells[1]
        return "row:" + row

    merged_rows = list(existing_rows)
    keys = {row_key(row): index for index, row in enumerate(merged_rows)}
    for row in rows:
        key = row_key(row)
        if key in keys:
            merged_rows[keys[key]] = row
        else:
            keys[key] = len(merged_rows)
            merged_rows.append(row)

    body = "\n".join(merged_rows) or "_Nenhuma submissão registrada._"
    generated = "{}\n{}\n{}".format(SUBMISSION_START, body, SUBMISSION_END)
    if SUBMISSION_START in text and SUBMISSION_END in text:
        before = text.split(SUBMISSION_START, 1)[0]
        after = text.split(SUBMISSION_END, 1)[1]
        path.write_text(before + generated + after, encoding="utf-8")
    else:
        path.write_text(
            text.rstrip()
            + "\n\n## 13. Registro automático de submissions Kaggle\n\n"
            + generated
            + "\n",
            encoding="utf-8",
        )


def update_lab_proposals(lab_path: Path, proposals: Iterable[Dict[str, Any]]) -> None:
    """Keep the next adaptive proposal queue visible in lab.md."""

    path = Path(lab_path)
    text = path.read_text(encoding="utf-8")
    rows: List[str] = []
    for proposal in proposals:
        rows.append(
            "| {name} | {family} | {reason} | {hypothesis} |".format(
                name=proposal.get("name", ""),
                family=proposal.get("family", ""),
                reason=proposal.get("proposal_reason", ""),
                hypothesis=proposal.get("hypothesis", ""),
            )
        )
    body = "\n".join(rows) or "_Nenhuma proposta adaptativa pendente._"
    generated = "{}\n{}\n{}".format(PROPOSAL_START, body, PROPOSAL_END)
    if PROPOSAL_START in text and PROPOSAL_END in text:
        before = text.split(PROPOSAL_START, 1)[0]
        after = text.split(PROPOSAL_END, 1)[1]
        path.write_text(before + generated + after, encoding="utf-8")
    else:
        path.write_text(
            text.rstrip()
            + "\n\n## 14. Fila adaptativa gerada\n\n"
            + generated
            + "\n",
            encoding="utf-8",
        )

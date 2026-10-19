"""Persistent UTC submission budget with conservative reservation semantics."""

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set


class SubmissionBudget:
    """Count attempted/reserved submissions and persist the ledger atomically."""

    def __init__(self, path: Path, daily_limit: int = 5):
        if daily_limit < 1:
            raise ValueError("daily_limit must be positive")
        self.path = Path(path)
        self.daily_limit = daily_limit
        self.records: List[Dict[str, object]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.records = list(payload.get("records", []))

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="budget-", suffix=".json", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"daily_limit": self.daily_limit, "records": self.records}, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, str(self.path))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _utc(value: Optional[datetime]) -> datetime:
        if value is None:
            return datetime.now(timezone.utc)
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    def used_today(self, now: Optional[datetime] = None) -> int:
        day = self._utc(now).date().isoformat()
        counted_statuses = {"reserved", "submitted", "failed"}
        return sum(
            1
            for record in self.records
            if record.get("status") in counted_statuses
            and str(record.get("submitted_at", "")).startswith(day)
        )

    def remaining(self, now: Optional[datetime] = None) -> int:
        return max(0, self.daily_limit - self.used_today(now))

    def attempted_experiments(self, experiments: Sequence[str]) -> Set[str]:
        """Return names already reserved or attempted in this ledger."""

        wanted = {str(experiment) for experiment in experiments}
        return {
            str(record.get("experiment"))
            for record in self.records
            if record.get("status") in {"reserved", "submitted", "failed"}
            and str(record.get("experiment")) in wanted
        }

    def record(
        self,
        experiment: str,
        submitted_at: Optional[datetime] = None,
        status: str = "submitted",
        reference: Optional[str] = None,
    ) -> Dict[str, object]:
        timestamp = self._utc(submitted_at).isoformat()
        entry: Dict[str, object] = {"id": uuid.uuid4().hex, "experiment": experiment, "submitted_at": timestamp, "status": status}
        if reference:
            entry["reference"] = reference
        self.records.append(entry)
        self._save()
        return entry

    def mark(self, experiment: str, status: str, reference: Optional[str] = None) -> bool:
        """Update the newest reservation for an experiment without double-counting it."""

        for record in reversed(self.records):
            if record.get("experiment") == experiment and record.get("status") == "reserved":
                record["status"] = status
                if reference:
                    record["reference"] = reference
                self._save()
                return True
        return False

    def release_failed(self, experiments: Sequence[str]) -> bool:
        """Mark selected failed attempts as explicitly retryable.

        This is an operator-controlled recovery path for failures verified to
        occur before Kaggle accepted an upload.  Ordinary failures remain
        counted and block retries, preserving the conservative budget default.
        """

        wanted = {str(experiment) for experiment in experiments}
        changed = False
        for record in self.records:
            if record.get("status") == "failed" and str(record.get("experiment")) in wanted:
                record["status"] = "retryable_failed"
                changed = True
        if changed:
            self._save()
        return changed

    def release_reserved(self, experiments: Sequence[str]) -> bool:
        """Release reservations after an externally verified pre-upload stop."""

        wanted = {str(experiment) for experiment in experiments}
        changed = False
        for record in self.records:
            if record.get("status") == "reserved" and str(record.get("experiment")) in wanted:
                record["status"] = "retryable_failed"
                changed = True
        if changed:
            self._save()
        return changed

    def reserve(self, experiments: Sequence[str], now: Optional[datetime] = None) -> List[str]:
        """Reserve available slots; reservations remain counted after failures."""

        attempted = self.attempted_experiments(experiments)
        unique: List[str] = []
        for experiment in experiments:
            name = str(experiment)
            if not name or name in attempted or name in unique:
                continue
            unique.append(name)
        accepted = unique[: self.remaining(now)]
        for experiment in accepted:
            self.record(experiment, submitted_at=now, status="reserved")
        return accepted

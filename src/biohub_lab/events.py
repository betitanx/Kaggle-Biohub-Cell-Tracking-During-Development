"""Small structured event logger used by experiment runs."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def emit_event(path: Path, run_id: str, event: str, **fields: Any) -> Dict[str, Any]:
    """Append one bounded, machine-readable event and return its payload."""

    payload: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "event": event,
    }
    payload.update(fields)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    print(json.dumps(payload, sort_keys=True, default=str))
    return payload

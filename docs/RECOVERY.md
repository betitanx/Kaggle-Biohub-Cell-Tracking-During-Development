# Recovery and provenance

The original project directory was unavailable on 19 October 2026. Four preserved development-session streams provided text patches, observed file reads, and historical results. The recovery process decoded those records as data and reapplied file edits inside separate recovery directories; it did not execute shell commands or JavaScript taken from the old sessions.

File-read observations helped recover changes that could not be reconstructed from patches alone. Original failed patches and records with incomplete context were tracked separately. Recovery generations and raw logs remain private and are not part of this release.

The released `src/`, tests, and selected utilities come from the recovered source. The `entries_for_prefix` helper was restored verbatim from a complete observed source section. Public-release changes make the tests use the current Python interpreter, sanitize machine/account paths, disable real-data configurations by default, and preserve upstream licensing in generated training archives. These changes do not retune the historical best-own parameters.

The upstream baseline was independently fetched during recovery at commit `075fc5f5a52d11077f9dc2b074644618f26939e2`, then the available recorded compatibility patches were reapplied. That commit identifies the fetched dependency, not a proven bit-for-bit match to the original baseline checkout. Only relevant Python source and its license are bundled.

The progress image was extracted from the last reviewed historical preview, observed at `2026-09-24T02:11:43.762Z`. It preserves the final label layout. Public-score records were extracted from a complete historical JSON observation and reduced to candidate names, references, dates, public status/score, and control/probe classification. No private scores or authentication material are included.

Unavailable or intentionally excluded material includes competition images and labels, model weights, raw predictions, private notebooks, full operational ledgers, cloud state, virtual environments, and session logs. The old plotting utility depends on unavailable raw result files, so the restored image and curated summary are provided instead of claiming that utility is self-contained.

Software validation covers unit/regression tests, generated notebook syntax, the CSV validator, and a synthetic pipeline demonstration. It does not rerun the original competition dataset or reproduce leaderboard scores. Optional ML/service dependencies are not fully locked or revalidated; `requirements-dev.txt` records the lightweight verification environment.

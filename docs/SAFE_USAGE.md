# Safe usage

This is a local research toolkit, not a sandbox for untrusted code or files. Use a disposable environment with only the data and permissions needed for the experiment.

## Filesystem safeguards

- Cache preparation requires a **new, nonexistent cache directory**. It never clears an existing cache and rejects a symlink at the cache path or overlap with the source. Choose a fresh path for every run.
- CSV-to-GEFF conversion accepts simple dataset identifiers only, validates every destination before writing, and does not overwrite by default. Replacing existing output requires explicit `--overwrite`; symlink destinations are refused even then.
- Subset downloads validate relative paths and destination containment before downloads or directory creation. Existing symlink components and symlink destinations are refused; ordinary completed files remain resumable.

Use directories owned by you and not writable by untrusted processes. These checks do not claim protection against another process with the same filesystem privileges replacing directories concurrently. An explicit overwrite still replaces valid output inside the selected destination; it is not a backup mechanism.

Conversion and subset download destinations must use physical paths without symlink ancestors. This also rejects macOS aliases such as `/tmp` or `/var`; use a verified canonical directory (for example, the corresponding `/private/...` path) rather than a symlinked alias. Do not resolve an untrusted user-supplied path merely to bypass these checks.

## External models and code

The support pack, postprocessing inputs, optional model source, and checkpoints are executable trust boundaries. Review their source, select known versions, and verify independently approved hashes before execution. Never attach unrelated account secrets or writable data.

The optional DeepCenter loader imports external Python and uses unrestricted checkpoint deserialization. It must only receive trusted source and weights. Historical notebooks retain their submitted code; their inclusion is not a claim that arbitrary attached datasets are safe to run.

## Remote execution

The default examples enable only local synthetic smoke candidates. Competition submissions require explicit live activation and budget checks. Remote notebook dry runs and holdout launchers **do upload code and consume compute** when invoked with account access; “dry run” there means no competition scoring submission, not no remote execution.

## Sharing and licenses

See [license information](../LICENSE_INFO.md) and [third-party notices](../THIRD_PARTY_NOTICES.md). Software permissions do not grant rights to competition data, external weights, logos, or third-party images. Do not redistribute competition data; obtain inputs through their original access-controlled pages and follow the competition rules.

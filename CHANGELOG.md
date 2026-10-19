# Changelog

## Unreleased

### Security

- Cache preparation no longer recursively deletes existing paths. It requires a new destination and rejects source/cache overlap and a symlink at the cache path.
- CSV-to-GEFF conversion validates all dataset identifiers and output destinations before writing. Existing output is preserved unless `--overwrite` is explicitly requested; output symlinks are rejected.
- Subset downloads reject ambiguous, absolute, traversal, and symlink paths before downloads or destination creation, and recheck destinations before moving files.

### Documentation

- Added safe-usage guidance covering filesystem assumptions, trusted model inputs, and remote compute.
- Separated software permissions from external datasets, weights, and third-party images in the license notices.
- Recorded the owner's all-rights-reserved choice for original contributions, preserving the baseline's BSD license and separate third-party image rights.

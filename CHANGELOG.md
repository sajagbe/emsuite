# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Shared safe config parser (`emsuite.config`) replacing `exec()` for surface inputs.
- Unit test suite (`tests/unit/`) with 17 tests covering config, properties, surf I/O, normalization.
- GitHub Actions workflow for unit tests.
- `docs/ROADMAP.md` documenting three-channel architecture and property backlog.

### Changed
- Removed local dev clutter from repository root and `opt/`.
- Trimmed CCO example point logs to three representative samples on disk.
- Updated `.gitignore` for a cleaner professional layout.
- Synced `docs/tuning/reference/inputs.rst` to current two-stage API.
- Marked potential/combined Sphinx pages as not yet implemented.

### Removed
- `requests` dependency and Office quote Easter egg.
- Unsafe `exec()` parsing in `surface.parse_surface_input()`.

## [1.0.5] - 2026-03-25

### Added
- Documented the two-stage CLI workflow with explicit command coverage for:
  - emsuite -s surface.in
  - emsuite -t tuning.in
- Added complete surface.in reference with required keys, defaults, and optimization behavior.
- Added complete tuning.in reference with required keys, defaults, and execution settings.
- Documented output structure details including normalized MOL2 outputs and resume metadata in logs.

### Changed
- Updated Quick Start to match the implemented API by splitting surface generation and tuning runs.
- Replaced outdated tuning examples that used input_type/input_data with current tuning inputs (molecule or xyz_file plus surface_file).
- Clarified calc_type behavior (separate vs combined) and parallel processing controls.

### Documentation
- Synced README command and input schema details to current runtime behavior in:
  - src/emsuite/cli.py
  - src/emsuite/surface.py
  - src/emsuite/tuning.py

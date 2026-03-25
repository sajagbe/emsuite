# Changelog

All notable changes to this project are documented in this file.

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

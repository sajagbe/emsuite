# Changelog

All notable changes to this project are documented in this file.

## [Unreleased]

### Added
- Public `SurfaceInput` / `PotentialInput` / `TuningInput` / `CoupledInput` and matching `*Result` types; `.in` files load inputs.
- `Geometry.from_xyz` and `PotentialResult.to_surf()`.
- `SurfaceResult.to_xyz()` (pseudo-atom point cloud) and `SurfaceResult.to_mol2()` (pseudo-atoms with `values` in the charge column), plus `surface.io.save_mol2()`, for visualizing a `.surf` in a molecular viewer.
- Protein/ligand APBS occupancy: `ligand_atoms='present'` (ligand in PQR at q=0) vs `'absent'` (ligand omitted). Box always spans protein and ligand coordinates.

### Changed
- CLI and Python code both construct `*Input` objects (`.from_file()` or `.from_config(**kwargs)`) and call `.run()`.
- `emsuite.tuning.runner.main` renamed to `run_tuning_calculation(config)`, matching `run_surface_calculation` / `run_potential_calculation` / `run_coupled_calculation`.
- `run_coupled_calculation` now returns the `CoupledResult` instead of discarding it and returning `None`.
- Coupled pipeline calls potential then tuning in memory (no throwaway `coupled_*.in` files).
- Potential mapping is APBS only. Gasteiger remains the temporary charge helper for PQR writing.

### Removed
- `emsuite.api` (`surface()`, `tune()`, `potential()`, `coupled()`) and the top-level `emsuite.tune` shorthand — redundant with `*Input.from_config(**kwargs).run()`, which already accepted the same kwargs and `config=`.
- Bond-axis scan (`bond_scan_atoms` / `bond_scan.py`).
- MLIP/xTB engines and the `mlip` extra.
- User-facing `method='coulomb'` vacuum ESP (no Coulomb fallback).
- Advanced tuning properties: `freq`, `stark_*`, `eint`, `h2o`, `pa`, fugacity extensions.
- Unused `core.create_mol2_file` and a dead duplicate `core.smiles_to_xyz` (the real one lives in `surface/optimize.py`).
- Unused `potential.run_apbs_potential`, superseded by the inline APBS + quantity dispatch in `potential/runner.py`.
- The `engines/` abstraction (`Engine` protocol, `PySCFEngine`, `get_engine()`) — unreferenced outside its own test; runners call `emsuite.core` directly.

## [1.3.0] - 2026-08-27

### Added
- Keyword-argument API (`emsuite.api`): `surface()`, `tune()`, `potential()`, `coupled()` — call channels with only the params you need; defaults fill the rest.
- `emsuite.tune` shorthand exported at the top level.
- `config=` argument on every channel accepts a `.in` file path or a dict; explicit kwargs override config values.
- `emsuite.config.resolve_config()` and `UNSET` sentinel for layered config resolution (defaults < file/dict < explicit kwargs).
- Channel default constants: `SURFACE_DEFAULTS`, `POTENTIAL_DEFAULTS`, `COUPLED_DEFAULTS`.
- Potential channel `quantity` switch: `'potential'` (interpolated APBS φ) or `'charge'` (Gauss-law ρ → q at surface points).
- APBS dielectric maps (`dielx`/`dely`/`dielz`) and Gauss-law conversion (`emsuite.potential.gauss`).

### Changed
- `run_surface_calculation`, `run_potential_calculation`, `run_coupled_calculation`, and tuning `main()` now accept a parameter dict in addition to a file path (non-breaking).
- Potential default engine is APBS (`method='apbs'`). Coulomb remains a vacuum fallback (potential only).
- Coupled defaults to APBS Gauss-law surface charges (`potential_quantity='charge'`).

### Removed
- Spatial Fukui properties (`fukui_spa_plus`, `fukui_spa_minus`) and the global `ts_barrier` property, along with their modules (`properties/fukui_spatial.py`, `properties/ts.py`, `tuning/surface_maps.py`), surface-map runner wiring, the `ts_xyz`/`fukui_projection` tuning inputs, and their unit/integration tests.

## [1.2.0] - 2026-06-24

### Added
- Advanced tuning properties: spatial Fukui (`fukui_spa_*`), `freq`, Stark (`stark_*`), `eint`, `h2o`, `pa`, fugacity extensions, `ts_barrier`.
- Bond-axis electrostatic scan in potential channel (`bond_scan_atoms` in `potential.in`).
- `TBLiteEngine` / MLIP optional extra (`uv sync --extra mlip`) for GFN-xTB screening.
- Property modules: `fukui_spatial`, `vibrational`, `stark`, `interaction`, `thermo_ext`, `ts`.
- `tuning/surface_maps.py` for surface-projected property maps.
- Traceable integration audit: `scripts/run_slow_integration_audit.py`, `tests/integration/conftest.py`, 14 slow integration tests.
- Unit tests for v1.2 property registry, Fukui projection, and bond scan.
- Session records: `docs/SESSION_CHANGELOG_V1.2.md`, `docs/session_records/2026-06-24T070253Z_*`.

### Changed
- Version 1.2.0 (still v1.x semver — not marketed as v2.0).
- `MLIPEngine` delegates to TBLite when the optional dependency is installed.

### Fixed
- `mulliken_charges()` in Fukui spatial module now uses atomic charges (`pop[1]`), not AO populations.

## [1.1.0] - 2026-06-24

### Added
- **Potential channel** (`emsuite -p`) — Coulomb/Gasteiger ESP maps with optional APBS fallback.
- **Coupled channel** (`emsuite -c`) — potential-derived heterogeneous surfaces feed tuning.
- Shared safe config parser (`emsuite.config`) replacing `exec()` for surface inputs.
- Config schemas (`emsuite.config.schemas`) and `load_config` alias.
- Unit, integration, and regression test suites (32 tests total).
- GitHub Actions CI with Python 3.11/3.12 matrix and `uv` workflow.
- `docs/ROADMAP.md`, `docs/ENGINEERING.md`, `docs/SESSION_HANDOFF.md`.
- Ruff lint/format, pre-commit hooks, `.python-version`, and `py.typed` marker.
- Property modules: `ground_state`, `excited_state`, `thermo`; new properties `spin`, `fukui_plus`, `fukui_minus`.
- Ray workers extracted to `tuning/parallel.py` (CPU `@ray.remote` fix).
- `PySCFEngine` / `MLIPEngine` engine wrappers and `get_engine()`.
- Example templates: `surface.in`, `tuning.in`, `potential.in`, `coupled.in`.
- `[project.optional-dependencies] docs` extra.

### Changed
- Cookie-cutter package restructure: `config/`, `cli/`, `core/`, `surface/`, `tuning/`, `engines/`, `potential/`, `coupled/` subpackages.
- Public API: `generate_surface`, `run_potential_calculation`, `run_coupled_calculation`, `run_tuning`, `load_config`.
- CLI router adds `-p` / `-c` flags alongside `-s` / `-t`.
- Input templates moved to `examples/templates/`; README GIF moved to `docs/_static/`.
- Trimmed CCO example point logs to representative samples on disk.
- Synced Sphinx tuning/potential/coupled pages to shipped API.
- CI runs unit + regression on every push; integration `@pytest.mark.slow` on all pushes.

### Removed
- `requests` dependency and Office quote Easter egg.
- Unsafe `exec()` parsing in surface input files.
- Root dev clutter (`opt/`, stray `*.in` / `*.xyz` patterns gitignored).

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
- Synced README command and input schema details to current runtime behavior.

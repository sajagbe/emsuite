# Session changelog — v1.1.0 + v1.2.0

Exhaustive record of work done in this session (unpushed). See also `CHANGELOG.md`.

---

## Version

- **1.1.0** → refoundation, four channels, CI, core properties
- **1.2.0** → former v2.x backlog (advanced properties, bond scan, MLIP, integration audit)

---

## v1.1.0 — channels and refoundation

### CLI (`src/emsuite/cli/main.py`)

- `-s` surface, `-t` tuning, **`-p` potential**, **`-c` coupled**

### Potential channel (new)

| File | Functions / classes |
|------|---------------------|
| `potential/runner.py` | `run_potential_calculation()` |
| `potential/config_io.py` | `parse_potential_input()` |
| `potential/coulomb.py` | `partial_charges_from_xyz()`, `coulomb_potential_at_points()` |
| `potential/apbs.py` | `_write_apbs_input()`, `run_apbs_potential()` |
| `potential/pqr.py` | `read_xyz()`, PQR helpers for APBS |

### Coupled channel (new)

| File | Functions |
|------|-----------|
| `coupled/runner.py` | `parse_coupled_input()`, `_write_potential_input()`, `_write_tuning_input()`, `run_coupled_calculation()` |

### Config (new / extended)

| File | Functions |
|------|-----------|
| `config/parser.py` | Safe assignment parser (replaces `exec()`) |
| `config/schemas.py` | `validate_surface_params()`, `validate_tuning_params()`, `validate_potential_params()`, `validate_coupled_params()` |

### Tuning properties (v1.1 split)

| File | Role |
|------|------|
| `properties/ground_state.py` | `gse`, `homo`, `lumo`, `gap`, `dm`, `spin` |
| `properties/thermo.py` | `ie`, `ea`, `cp`, `eng`, `hard`, `efl`, `nfl` |
| `properties/excited_state.py` | `exe`, `osc` |
| `properties/registry.py` | `PROPERTY_CONFIG`, `setup_calculation()`, `calculate_all_properties()` |
| `tuning/parallel.py` | `calculate_point_effect_cpu`, `calculate_point_effect_cpu_remote`, `calculate_point_effect_gpu` (Ray) |

### Engines

| File | Role |
|------|------|
| `engines/pyscf_engine.py` | `PySCFEngine` (moved from monolith) |
| `engines/mlip_engine.py` | `MLIPEngine` stub → delegates to TBLite in v1.2 |
| `engines/__init__.py` | `get_engine(name)` |

### Examples / docs

- `examples/templates/{surface,tuning,potential,coupled}.in`
- `docs/ROADMAP.md`, `docs/ENGINEERING.md`, Sphinx updates

---

## v1.2.0 — advanced properties (former v2.x)

### New property modules

| Module | Functions | Properties |
|--------|-----------|------------|
| `properties/fukui_spatial.py` | `mulliken_charges()`, `condensed_fukui_indices()`, `atom_coords_from_mf()`, `project_atom_property_to_point()`, `build_surface_fukui_maps()` | `fukui_spa_plus`, `fukui_spa_minus` |
| `properties/vibrational.py` | `fundamental_frequency_cm1()` | `freq` |
| `properties/stark.py` | `_field_from_probe()`, `stark_orbital_shifts()`, `compute_stark_properties()` | `stark_homo`, `stark_lumo`, `stark_gap` |
| `properties/interaction.py` | `water_probe_coords_and_charges()`, `interaction_energy_kcal()`, `proton_affinity_kcal()` | `eint`, `h2o`, `pa` |
| `properties/thermo_ext.py` | `fugacity_extensions()` | `efl_fug`, `nfl_fug`, `eng_fug` |
| `properties/ts.py` | `ts_barrier_kcal()` | `ts_barrier` (global) |

### Surface maps (tuning)

| File | Functions |
|------|-----------|
| `tuning/surface_maps.py` | `is_surface_map_property()`, `split_property_lists()`, `surface_map_effects_for_point()`, `precompute_surface_maps()` |

### Runner changes (`tuning/runner.py`)

- `_merge_surface_map_effects()` — merges spatial Fukui into per-point effects (serial + Ray)
- Water probe placement for `h2o`; `probe_coord` / `probe_charge` for Stark
- `eint` / `h2o` as interaction-energy differences vs baseline
- `ts_barrier_kcal()` when `ts_xyz` set
- `precompute_surface_maps()` before point loops
- `split_property_lists()` separates QM vs surface-map properties

### Potential bond scan

| File | Functions |
|------|-----------|
| `surface/bond_scan.py` | `bond_scan_coords()` |
| `potential/runner.py` | Uses bond scan when `bond_scan_atoms = [i, j]` |
| `potential/config_io.py` | `bond_scan_atoms`, `bond_scan_steps`, `bond_scan_span` defaults |

### MLIP / xTB engine

| File | Class / methods |
|------|-----------------|
| `engines/tblite_engine.py` | `TBLiteEngine`: `is_available()`, `describe()`, `optimize_geometry()`, `single_point_energy()` |
| `engines/mlip_engine.py` | `MLIPEngine` delegates to `TBLiteEngine` when installed |
| `pyproject.toml` | `[project.optional-dependencies] mlip = ["tblite", "ase"]` |

### Registry additions (`PROPERTY_CONFIG`)

12 new keys with flags where needed:

- `surface_map: True` — `fukui_spa_plus`, `fukui_spa_minus`
- `global: True` — `ts_barrier`

`calculate_all_properties()` extended with `probe_coord`, `probe_charge`; calls new modules.

### Bugfix (integration audit)

- `mulliken_charges()` used `pop[0]` (AO populations) → fixed to `pop[1]` (atomic charges)

---

## Tests added

### Unit (`tests/unit/`)

- `test_v2_properties.py` — registry entries for v1.2 props
- `test_fukui_spatial.py` — projection + Mulliken shape
- `test_bond_scan.py` — bond axis coordinates
- `test_property_registry_extended.py`, `test_schemas.py`, `test_engines.py`

### Integration (`tests/integration/`)

| Test file | Feature |
|-----------|---------|
| `test_smoke_methane.py` | v1.1 end-to-end surface → tuning |
| `test_tuning_separate.py` | Alias for separate mode |
| `test_surface_smiles.py` | SMILES → surf |
| `test_potential_coulomb.py` | Coulomb ESP |
| `test_coupled_smoke.py` | Coupled pipeline |
| `test_potential_bond_scan.py` | Bond-axis ESP |
| `test_tuning_fukui_spatial.py` | Spatial Fukui maps |
| `test_tuning_stark.py` | Stark gap |
| `test_tuning_freq.py` | Hessian frequency |
| `test_tuning_water_probe.py` | H2O probe |
| `test_tuning_ts_barrier.py` | TS barrier |
| `test_tuning_fugacity.py` | Fugacity extensions |
| `test_tuning_pa.py` | Proton affinity |
| `test_mlip_engine.py` | MLIP optional (skips if no tblite) |

### Integration infrastructure

| File | Role |
|------|------|
| `tests/integration/conftest.py` | `INTEGRATION_FEATURE_TRACE`, artifact capture hooks |
| `tests/integration/helpers.py` | Shared methane fixtures, `record_assertions()` |
| `scripts/run_slow_integration_audit.py` | Full audited run + `RUN_MANIFEST.md` |
| `tests/integration_runs/README.md` | Documents artifact layout |

### Regression

- `tests/regression/test_cco_golden.py` — CCO structure golden

---

## Documentation updated

- `README.md` — 29 properties, bond scan, tuning extras
- `CHANGELOG.md` — 1.1.0 and 1.2.0 sections
- `docs/ROADMAP.md` — v1.2 shipped, future ideas only
- `docs/ENGINEERING.md`, Sphinx `properties.rst`, potential/combined index pages

---

## Integration audit result (2026-06-24T070253Z)

- **13 passed, 1 skipped** (~11 min)
- **Skipped:** `test_mlip_engine_optional` (no `tblite`)
- **On disk:** `tests/integration_runs/LATEST/<test>/workspace/` (full QM outputs)
- **Committed summary:** `docs/session_records/2026-06-24T070253Z_*`

---
title: "Streamline EMSuite inputs and ligand occupancy"
type: refactor
date: 2026-08-27
---

# Streamline EMSuite inputs and ligand occupancy

## Summary

Delete code that is not needed for the five scientific runs, then reshape what remains into `*Input` / `*Result` objects. After that, add protein/ligand APBS with two occupancy modes (ligand atoms present at charge 0 vs ligand atoms absent), both for potential and Gauss-law charge, then tuning on those surfaces.

Do not mix deletion, rename, and new science in one pass.

## Problem Frame

Callers think in dicts and `.in` files. The potential channel charges every atom in one XYZ. Tuning cannot start from SMILES. Bond scan, xTB, vacuum Coulomb, and advanced properties sit on the same runners as the work that matters.

The protein-field use case needs a ligand surface, protein charges only, and a comparison of whether the ligand occupies the dielectric cavity.

## Requirements

- R1. Surface generation from SMILES or XYZ, with or without optimization, still works.
- R2. Tuning maps run from XYZ plus a surface, and from SMILES via surface-then-tune.
- R3. Potential mapping uses APBS only. The `.surf` fourth column is either interpolated φ or Gauss-law charge, chosen by `quantity`.
- R4. A ligand/protein pair can be mapped: surface from the ligand; APBS charges from the protein only.
- R5. Ligand occupancy is selectable: **present** (ligand atoms in the PQR at charge 0) or **absent** (ligand atoms omitted from the PQR). Both modes are first-class, for comparison.
- R6. Occupancy applies to both `quantity='potential'` and `quantity='charge'`, and those surfaces feed tuning of the ligand.
- R7. Public types are `SurfaceInput`, `PotentialInput`, `TuningInput`, `CoupledInput` and matching `*Result`. `.in` files load inputs; they are not the live type inside runners.
- R8. Bond-axis scan, MLIP/xTB, user-facing vacuum Coulomb, and advanced properties (`freq`, `stark_*`, `eint`, `h2o`, `pa`, fugacity) are gone from `src/` and tests.

## Key Technical Decisions

- KTD1. Three commit arcs, in order: deletions → input/result reshape with unchanged science → protein/ligand occupancy. A broken occupancy feature must not hide inside a rename.
- KTD2. Names: top-level `*Input` / `*Result`. Nested extras (if any remain) are optional objects, `None` meaning off.
- KTD3. Occupancy is `ligand_atoms='present' | 'absent'`, not a boolean named after charges. Present always means q = 0 on ligand atoms. Absent means those atoms are not in the PQR. `include_ligand_charges` is not a separate knob.
- KTD4. APBS grid extent is the bounding box of **protein and ligand coordinates**, even when ligand atoms are absent from the PQR, so a pocket ligand is not clipped.
- KTD5. Coupled is `PotentialInput` (quantity charge) then `TuningInput` on the ligand. No throwaway `.in` files.
- KTD6. Gasteiger stays only as the temporary protein charge source until a PQR/PDB charge path exists. It is not a user-facing `method='coulomb'`.
- KTD7. GPU, Ray, and resume stay. They are not scientific goals; tuning maps need them.
- KTD8. Closes KTD6: `protein_format='pdb'` converts via `pdb2pqr` (real AMBER/CHARMM/PARSE force-field charges + propka protonation for the protein), selecting the ligand explicitly by `(ligand_resname, ligand_chain, ligand_resseq)` rather than trusting pdb2pqr's own `--ligand` atom-name matching, which has no residue-identity check and can silently mismatch a different HETATM residue with the same atom names. `ligand_atoms` gains a third value, `'charged'`, alongside `'present'`/`'absent'`: pdb2pqr can now give a selected ligand a real (PEOE) nonzero charge, which the old Gasteiger-on-XYZ path never could — extending the existing field keeps KTD3's "one field, one concept" rule rather than adding a second orthogonal knob. The XYZ+Gasteiger path (`protein_format='xyz'`, still the default) is unchanged.

## High-Level Technical Design

```text
SurfaceInput(smiles|xyz, optimize) → SurfaceResult → .to_surf()

PotentialInput(
  ligand, protein,
  ligand_atoms=present|absent,   # cavity vs no cavity
  quantity=potential|charge
) → PotentialResult → .to_surf()

TuningInput(xyz=ligand, surface=PotentialResult|path, properties) → TuningResult

CoupledInput(...) = potential (charge) + tuning on ligand
```

APBS PQR construction:

| `ligand_atoms` | Ligand in PQR | Ligand q | Protein q | Surface |
|----------------|---------------|----------|-----------|---------|
| `present` | yes | 0 | protein charges | ligand VDW |
| `absent` | no | n/a | protein charges | ligand VDW |

Comparison runs: same ligand/protein, both occupancy values, both quantities, then tuning on the charge surfaces.

## Scope Boundaries

In: the twelve-step cleanup/reorg plus occupancy comparison and downstream tuning.

Out: ESP/MEP backends, rewriting PySCF/Ray, dropping `.in` files, charging the ligand in APBS, bond scan, xTB, vacuum Coulomb method, advanced properties listed in R8.

Deferred: reading protein charges from a real PQR/PDB instead of Gasteiger; JSON schema versioning beyond a simple `to_json`/`from_json`.

## Implementation Units

### U1. Freeze the test line

Run `uv run pytest tests/unit tests/regression -q`. Record the pass count. Every later unit must keep unit+regression green. Do not start protein/ligand work in this unit.

- **Tests:** existing `tests/unit/`, `tests/regression/`. No new files.

### U2. Remove bond scan

Delete `src/emsuite/surface/bond_scan.py`. Strip `bond_scan_*` from `src/emsuite/potential/runner.py`, `config_io.py`, `src/emsuite/api.py`, templates, `tests/unit/test_bond_scan.py`, `tests/integration/test_potential_bond_scan.py`.

- **Tests:** remaining potential tests still write a VDW-surface `.surf`.

### U3. Remove MLIP / xTB

Delete `src/emsuite/engines/mlip_engine.py` and `tblite_engine.py`. `get_engine()` returns PySCF only. Drop mlip extra and `tests/integration/test_mlip_engine.py`, `tests/unit/test_engines.py` MLIP cases.

- **Tests:** `tests/unit/test_engines.py` only asserts PySCF.

### U4. Remove user-facing Coulomb method

Delete `coulomb_potential_at_points` and `method='coulomb'`. Move Gasteiger assignment to a charge helper used by APBS PQR writing (keep file or rename; do not export Coulomb as a method). Update `validate_potential_params`, defaults, templates, `tests/integration/test_potential_coulomb.py` (replace with APBS or drop).

- **Tests:** `tests/unit/test_gauss.py` schema tests no longer accept `method='coulomb'`. Potential default is APBS.

### U5. Strip advanced properties

Delete `stark.py`, `vibrational.py`, `interaction.py`, `thermo_ext.py`. Remove registry keys and tests: `test_v2_properties.py` (trim), `test_tuning_stark.py`, `test_tuning_freq.py`, `test_tuning_fugacity.py`, `test_tuning_pa.py`, `test_tuning_water_probe.py`. Core keys stay: gse, homo, lumo, gap, dm, spin, ie, ea, cp, eng, hard, efl, nfl, fukui_plus, fukui_minus, exe, osc.

- **Tests:** `tests/unit/test_property_config.py`, `test_property_registry_extended.py` still pass on the core set.

### U6. Coupled calls in memory

`src/emsuite/coupled/runner.py` calls potential then tuning with parameter dicts/objects. No `coupled_potential.in` / `coupled_tuning.in` on disk.

- **Tests:** `tests/integration/test_coupled_smoke.py` still produces a results directory. Assert no leftover coupled_*.in, or ignore them if tests chdir to tmp.

### U7. Geometry and result objects (science unchanged)

Add `src/emsuite/geometry.py` (`Geometry.from_xyz`). Add `src/emsuite/results.py`: `SurfaceResult`, `PotentialResult` (coords, values, quantity). `.surf` is `to_surf()`. Existing runners may still be called underneath.

- **Tests:** `tests/unit/test_geometry.py` XYZ round-trip; `PotentialResult.to_surf` / load_surf round-trip.

### U8. Inputs replace dicts, channel by channel

Add `src/emsuite/inputs.py`: `SurfaceInput`, `PotentialInput`, `TuningInput`, `CoupledInput` with `.run()`. `config.resolve_config` builds an Input from `.in` or kwargs. Runners take Input, not a dict. Order: surface, potential, tuning, coupled.

- **Tests:** unit tests that kwargs and a temp `.in` produce equal Input objects; `api.surface` / `api.potential` return result paths or result objects consistently (pick one and document).

### U9. Protein, ligand, occupancy

`PotentialInput` fields: `ligand`, `protein`, `ligand_atoms='present'|'absent'`, `quantity='potential'|'charge'`. Surface = ligand VDW. PQR per the occupancy table. Grid box from protein+ligand coords (KTD4). `TuningInput` uses ligand XYZ + that surface. `CoupledInput` sets quantity charge.

- **Tests:** unit tests with tiny synthetic XYZ (2-atom “protein”, 1-atom “ligand”): present PQR has ligand line with q=0; absent PQR has no ligand line; box still spans ligand. Integration (slow): methane-as-ligand + a few extra atoms as “protein”, both occupancies, both quantities, then a short tuning on the charge surface. Skip or mark slow if APBS missing.

### U10. Thin tuning runner and wire CLI/API

`tuning/runner.py` reads `TuningInput`, returns `TuningResult`. `cli/main.py` and `api.py` construct Inputs and call `.run()`. Same four flags.

- **Tests:** existing smoke tests still pass through CLI-equivalent runners.

### U11. Docs and templates

Rewrite `examples/templates/{surface,tuning,potential,coupled}.in` and README sample runs for the five goals plus occupancy comparison. Remove bond scan, Coulomb method, MLIP, Stark/freq/pa/h2o from user docs.

- **Tests:** none beyond link/parse of templates if `test_input_integration.py` still reads examples.

## Test Scenarios

- AE1. SMILES + optimize=True writes a `.surf` and an optimized XYZ.
- AE2. XYZ + optimize=False uses the given coordinates.
- AE3. Tuning from XYZ + surface writes MOL2/CSV for requested core properties.
- AE4. Tuning from SMILES is surface-then-tune (R2).
- AE5. `ligand_atoms='present'`: PQR contains ligand atoms, all ligand q = 0, protein q nonzero.
- AE6. `ligand_atoms='absent'`: PQR contains no ligand atoms; ligand surface points still sampled.
- AE7. Same pair, both occupancies, `quantity='potential'`: two finite φ maps, not required to be equal.
- AE8. Same pair, both occupancies, `quantity='charge'`: two finite Gauss-law maps.
- AE9. Tuning on the charge surface from AE8 uses ligand XYZ only (protein is not the QM molecule).
- AE10. `method='coulomb'`, `bond_scan_atoms`, `properties=['stark_gap']` raise or are rejected.

## Risks

- APBS box when ligand is absent: mitigated by KTD4.
- Gasteiger on a protein-sized XYZ is crude; accepted until PQR charges exist.
- Deleting advanced properties is user-visible; changelog must list removals (R8).
- Slow integration for occupancy needs APBS; keep unit tests as the contract if CI skips APBS.

## Documentation

After U11, README shows only: surface ± optimize; tuning from SMILES/XYZ; potential/charge with `ligand_atoms` present vs absent; coupled tuning on Gauss-law charge. Changelog under Unreleased / next version records deletions and the occupancy flag.

## Assumptions

- Core electronic properties (including ie/ea/fukui/exe) stay; only the four advanced modules in U5 go.
- Occupancy comparison is two runs the caller sets up, not a built-in diff command.
- Ligand and protein files share the same coordinate frame.

## Sources

- `src/emsuite/potential/apbs.py`, `pqr.py`, `gauss.py` — current APBS and Gauss-law path
- `src/emsuite/potential/runner.py` — dict runner, bond scan branch, Coulomb fallback
- `src/emsuite/coupled/runner.py` — writes temp `.in` files
- `src/emsuite/api.py` — kwargs over dicts
- Session agreement: delete extras first, then Input/Result, then protein/ligand; names are Input/Result not Spec

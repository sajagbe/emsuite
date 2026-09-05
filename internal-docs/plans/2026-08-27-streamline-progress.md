# Streamline progress log

Branch: `feat/streamline-inputs-ligand-occupancy`  
Plan: `docs/plans/2026-08-27-streamline-inputs-ligand-occupancy.md`  
Restore tag: `pre-streamline` → `018028d` (v1.3.0)

Do not tick boxes in the plan file. This log is the progress record.

## Freeze line (U1)

- 2026-08-27: `uv run pytest tests/unit tests/regression -q` → **51 passed** (pre-streamline).
- After U2–U11: **62 passed** (unit+regression).

## Units

| Unit | Status | Notes |
|------|--------|-------|
| U1 Freeze tests | done | 51 unit+regression; tagged `pre-streamline` |
| U2 Bond scan | done | Deleted `bond_scan.py` and tests; schema rejects leftover keys |
| U3 MLIP/xTB | done | PySCF-only `get_engine()`; dropped `mlip` extra |
| U4 Coulomb method | done | Gasteiger stays in `charges.py`; no vacuum ESP / fallback |
| U5 Advanced properties | done | Core registry only; AE10 rejects `stark_gap` |
| U6 Coupled in memory | done | No `coupled_*.in`; Input.run() chains potential then tuning |
| U7 Geometry/Result | done | `geometry.py`, `results.py` |
| U8 Input objects | done | Frozen `*Input` with `.from_config` / `.run()` |
| U9 Occupancy | done | `ligand_atoms='present'\|'absent'`; box spans protein+ligand |
| U10 CLI/API | done | CLI and `api` construct Inputs and call `.run()` |
| U11 Docs | done | README, templates, CHANGELOG Unreleased, ENGINEERING layout |

## Session notes

- 2026-08-27: Tagged `pre-streamline` before deletions. Executed U2–U11 on this branch. Occupancy unit tests cover AE5/AE6/KTD4 without APBS. Slow integration for occupancy is still optional.

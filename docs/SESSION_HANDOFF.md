# Session Handoff — v1.2.0

Last updated after v1.2.0 implementation and integration audit (local, **not pushed**).

## Current state

| Item | Status |
|------|--------|
| Version | **1.2.0** (`pyproject.toml`, `emsuite.__version__`) |
| Channels | `-s` surface, `-t` tuning, `-p` potential, `-c` coupled — all shipped |
| Properties | **29** tuning properties (17 core + 12 advanced) |
| Fast tests | 34 unit + regression — passing |
| Slow integration | 14 tests — **13 passed, 1 skipped** (MLIP) |
| Integration audit | `tests/integration_runs/LATEST/` on disk; summary in `docs/session_records/` |
| Git | `main`, ahead of `origin/main`; session work committed locally |

## Quick commands

```bash
cd /data/PHO_WORK/sajagbe2/packages/emsuite
uv sync --extra dev
uv run pytest tests/unit tests/regression -q
uv run pytest tests/integration -m slow -v
python scripts/run_slow_integration_audit.py   # traceable run → tests/integration_runs/
uv sync --extra mlip   # optional xTB engine
```

Templates: `examples/templates/{surface,tuning,potential,coupled}.in`

## Integration audit (saved)

| Location | Contents |
|----------|----------|
| `tests/integration_runs/LATEST/` | Full per-test workspaces, logs, MOL2, CSV (on disk, gitignored) |
| `docs/session_records/2026-06-24T070253Z_RUN_MANIFEST.md` | Committed summary table |
| `docs/session_records/2026-06-24T070253Z_run_meta.json` | Committed run metadata |
| `scripts/run_slow_integration_audit.py` | Re-run auditor |

See `docs/SESSION_CHANGELOG_V1.2.md` for exhaustive change list.

## v1.2 advanced properties

`fukui_spa_plus`, `fukui_spa_minus`, `freq`, `stark_homo`, `stark_lumo`, `stark_gap`, `eint`, `h2o`, `pa`, `efl_fug`, `nfl_fug`, `eng_fug`, `ts_barrier`

Tuning extras: `ts_xyz`, `fukui_projection`. Potential extras: `bond_scan_atoms`, `bond_scan_steps`, `bond_scan_span`.

## Still open (manual)

- FMN protein validation (`dev/tuning.in`)
- GPU + `parallel=True` smoke
- MLIP integration test (needs `uv sync --extra mlip`)
- Full CCO QM golden values (structure-only regression exists)

## Key decisions

- v2.x backlog implemented as **v1.2.0** (not marketed as v2.0)
- `calc_type='combined'` ≠ **coupled** channel (`emsuite -c`)
- Integration run workspaces gitignored; manifests committed under `docs/session_records/`

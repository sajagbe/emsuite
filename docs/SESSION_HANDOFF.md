# EMSuite status — 2026-08-27

Progress snapshot of `emsuite` on `main` at `/data/PHO_WORK/sajagbe2/packages/emsuite`. Version is still **1.2.0**; unreleased work is sitting in the working tree and has not been version-bumped.

## Snapshot

| Item | Status |
|------|--------|
| Version | **1.2.0** (`pyproject.toml`, `emsuite.__version__`) |
| Channels | `-s` surface, `-t` tuning, `-p` potential, `-c` coupled — shipped |
| Python API | **Unreleased:** `emsuite.api` kwargs (`surface`, `tune`, `potential`, `coupled`) |
| Properties | **27** (17 core + 10 advanced). Spatial Fukui and `ts_barrier` removed in working tree |
| Fast tests | 39 unit + 3 regression (includes 11 new API tests; not re-run this write-up) |
| Slow integration | 12 collected tests (2 deleted with Fukui/TS). Last full audit: 2026-06-24, **stale** vs current tree |
| Git | `main` at `880c4fd`, **9 commits ahead of `origin/main`**, **uncommitted** API + property trim |
| Remote | `origin/main` still at `8590549` (2026-03-25 README/CHANGELOG). Nothing since the March snapshot has been pushed |

## Two layers of progress

Work exists in two layers. Do not treat HEAD as “what’s on disk.”

### 1. Committed locally, not pushed (v1.1 → v1.2)

Nine commits on `main` since `origin/main`. The last one is:

`880c4fd` — *Ship v1.2.0: advanced properties, four channels, and traceable integration audit.* (2026-06-24)

That history covers the cookie-cutter refoundation, four-channel CLI, safe config parser, CI, 17 core properties, then the v1.2 advanced properties, xTB/MLIP extra, bond-axis ESP scan, and the 2026-06-24 integration audit.

Authoritative records: [CHANGELOG.md](../CHANGELOG.md) (`[1.1.0]`, `[1.2.0]`), [SESSION_CHANGELOG_V1.2.md](SESSION_CHANGELOG_V1.2.md), [session_records/2026-06-24T070253Z_RUN_MANIFEST.md](session_records/2026-06-24T070253Z_RUN_MANIFEST.md).

### 2. Uncommitted working tree (post-1.2, `[Unreleased]`)

Not staged. ~163 insertions / 462 deletions across 24 tracked files, plus three untracked files.

**Added**

- `src/emsuite/api.py` — kwargs API: `api.surface()`, `api.tune()`, `api.potential()`, `api.coupled()`
- `src/emsuite/config/resolve.py` — `resolve_config()` + `UNSET` (defaults < file/dict < explicit kwargs)
- `tests/unit/test_api.py` — 11 unit tests for resolution and runner wiring
- `emsuite.tune` top-level shorthand
- Channel runners now accept a parameter **dict** as well as a `.in` path (non-breaking)
- Default dicts: `SURFACE_DEFAULTS`, `POTENTIAL_DEFAULTS`, `COUPLED_DEFAULTS`

**Removed**

- Spatial Fukui (`fukui_spa_plus`, `fukui_spa_minus`) — `properties/fukui_spatial.py`, `tuning/surface_maps.py`
- Global `ts_barrier` — `properties/ts.py`
- Inputs `ts_xyz` / `fukui_projection`
- Tests: `test_fukui_spatial.py`, `test_tuning_fukui_spatial.py`, `test_tuning_ts_barrier.py`

Docs already updated in the same tree: `CHANGELOG.md` `[Unreleased]`, `README.md` (kwargs API + 27-property table), `docs/ROADMAP.md`. Remaining drift is listed below.

## What is shipped (v1.2 HEAD)

Four channels, two engines, 27 properties once the working-tree trim is included.

### Channels

| Channel | Purpose | CLI | Python |
|---------|---------|-----|--------|
| Surface | Geometry + VDW envelope | `emsuite -s surface.in` | `api.surface(...)` / `run_surface_calculation` |
| Tuning | Electrostatic maps of molecular properties | `emsuite -t tuning.in` | `api.tune(...)` / `run_tuning` |
| Potential | ESP on a surface (Coulomb, APBS, bond-axis scan) | `emsuite -p potential.in` | `api.potential(...)` / `run_potential_calculation` |
| Coupled | Potential → tuning pipeline | `emsuite -c coupled.in` | `api.coupled(...)` / `run_coupled_calculation` |

`calc_type='combined'` is a **tuning mode**, not the coupled channel.

### Engines

| Engine | Status |
|--------|--------|
| PySCF | Default QM (`PySCFEngine`) |
| MLIP / xTB | `TBLiteEngine` via `uv sync --extra mlip`; last audit **skipped** this extra |

### Tuning properties (working tree)

**Core (17):** `gse`, `homo`, `lumo`, `gap`, `dm`, `spin`, `ie`, `ea`, `cp`, `eng`, `hard`, `efl`, `nfl`, `fukui_plus`, `fukui_minus`, `exe`, `osc`

**Advanced (10):** `freq`, `stark_homo`, `stark_lumo`, `stark_gap`, `eint`, `h2o`, `pa`, `efl_fug`, `nfl_fug`, `eng_fug`

`properties = ['all']` expands to the full registry (`PROPERTY_CONFIG` in `src/emsuite/tuning/properties/registry.py`).

Potential extras: `bond_scan_atoms`, `bond_scan_steps`, `bond_scan_span`.

## Tests

| Suite | Count now | Notes |
|-------|-----------|--------|
| Unit | 39 | +11 API tests; − Fukui spatial unit tests |
| Regression | 3 | CCO golden CSV structure only — not full QM goldens |
| Integration (`@pytest.mark.slow`) | 12 collected | 11 unique + `test_tuning_separate.py` re-exports the methane smoke |

Last **traceable** slow audit (`docs/session_records/2026-06-24T070253Z_*`): 13 passed, 1 skipped (MLIP), ~11 min. That run still lists Fukui spatial and TS barrier as PASS. It does **not** cover the kwargs API or the property removals.

CI (`.github/workflows/tests.yml`): Ruff + unit/regression + slow integration on Python 3.11/3.12. Has never run against the unpushed v1.1/v1.2 history.

### Quick commands

```bash
cd /data/PHO_WORK/sajagbe2/packages/emsuite
uv sync --extra dev
uv run pytest tests/unit tests/regression -q
uv run pytest tests/integration -m slow -v
python scripts/run_slow_integration_audit.py   # → tests/integration_runs/
uv sync --extra mlip   # optional xTB engine
```

Templates: `examples/templates/{surface,tuning,potential,coupled}.in`

## Document drift (still open)

These still describe the **committed** 1.2.0 surface (29 properties, Fukui/TS present):

| File | Issue |
|------|--------|
| `README.md` overview | Still says “29 molecular properties”; the property table below it already has 27 |
| `docs/ENGINEERING.md` | Package map and test counts frozen at v1.1 |
| `docs/SESSION_CHANGELOG_V1.2.md` | Exhaustive 1.2 record; still lists Fukui/TS as current features |
| `docs/session_records/2026-06-24T070253Z_*` | Historical audit; includes deleted tests |

Do not rewrite the 2026-06-24 session record — it is a snapshot of that run.

## Still open (manual / not done)

- **Commit** the unreleased API + property trim (or split: removal vs API)
- **Version bump** if this ships as 1.2.1 (docs/removal) or 1.3.0 (kwargs API)
- **Push** 9 local commits + any new commit; `origin/main` is March 2026
- Re-run slow integration audit after the working tree is committed
- Fix README “29” → “27”
- FMN protein validation (`dev/tuning.in`)
- GPU + `parallel=True` smoke
- MLIP integration test (`uv sync --extra mlip`)
- Full CCO QM golden values (structure-only regression exists)
- Engineering backlog: Codecov, `ty`/mypy in CI (see `docs/ENGINEERING.md`)

## Key decisions already made

- Former v2.x backlog shipped as **v1.2.0**, not marketed as v2.0
- Spatial Fukui and global TS barrier were **removed** after 1.2 rather than kept as optional
- File-path API stays; kwargs API layers on top via `config=` + overrides
- Integration workspaces stay gitignored; manifests live under `docs/session_records/`
- Local research stays in gitignored `dev/`

## Suggested next session

1. Fix the README “29 properties” leftover.
2. Run `uv run pytest tests/unit tests/regression -q` on this tree.
3. Commit the unreleased work (ask before committing), then decide 1.2.1 vs 1.3.0.
4. Push `main` only when ready — remote is nine commits plus this WIP behind.

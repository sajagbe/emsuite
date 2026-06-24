# EMSuite Roadmap

Approved direction for EMSuite. Shipped features are documented in [README.md](../README.md).

## Channels (v1.2)

| Channel | Purpose | CLI |
|---------|---------|-----|
| **Surface** | Geometry + VDW envelope | `emsuite -s surface.in` |
| **Tuning** | Electrostatic tuning of molecular properties | `emsuite -t tuning.in` |
| **Potential** | ESP maps (Coulomb, APBS, bond-axis scan) | `emsuite -p potential.in` |
| **Coupled** | Potential → tuning pipeline | `emsuite -c coupled.in` |

**Naming:** `calc_type='combined'` ≠ the **coupled** channel.

## Engines

| Engine | Status |
|--------|--------|
| **PySCF** | Default QM (`PySCFEngine`) |
| **MLIP/xTB** | `TBLiteEngine` via `uv sync --extra mlip` (`get_engine('mlip')`) |

## Tuning properties (v1.2)

**Core (17):** `gse`, `homo`, `lumo`, `gap`, `dm`, `spin`, `ie`, `ea`, `cp`, `eng`, `hard`, `efl`, `nfl`, `fukui_plus`, `fukui_minus`, `exe`, `osc`

**Advanced (12):** `fukui_spa_plus`, `fukui_spa_minus`, `freq`, `stark_homo`, `stark_lumo`, `stark_gap`, `eint`, `h2o`, `pa`, `efl_fug`, `nfl_fug`, `eng_fug`, `ts_barrier`

Use `properties = ['all']` for the full registry.

### Tuning input extras

| Key | Purpose |
|-----|---------|
| `ts_xyz` | Transition-state XYZ for `ts_barrier` baseline |
| `fukui_projection` | `'nearest'` or `'inverse_distance'` for spatial Fukui maps |

### Potential bond scan

In `potential.in`:

```python
bond_scan_atoms = [0, 1]
bond_scan_steps = 10
bond_scan_span = 3.0
```

## Future ideas (not scheduled)

Higher-accuracy MLIP models (MACE, etc.), explicit solvent MD coupling, automated TS search for `ts_barrier` effects per probe.

## Version milestones

| Version | Scope |
|---------|-------|
| **v1.1** | Refoundation, four channels, tests, CI |
| **v1.2** | Advanced properties, spatial Fukui, xTB engine, bond scan |
| **v1.x+** | Further properties and engine backends as needed |

See [ENGINEERING.md](ENGINEERING.md) and [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

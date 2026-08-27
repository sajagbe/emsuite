# EMSuite Roadmap

Approved direction for EMSuite. Shipped features are documented in [README.md](../README.md).

## Channels (v1.2)

| Channel | Purpose | CLI |
|---------|---------|-----|
| **Surface** | Geometry + VDW envelope | `emsuite -s surface.in` |
| **Tuning** | Electrostatic tuning of molecular properties | `emsuite -t tuning.in` |
| **Potential** | APBS φ or Gauss-law charge on a surface | `emsuite -p potential.in` |
| **Coupled** | Potential → tuning pipeline | `emsuite -c coupled.in` |

**Naming:** `calc_type='combined'` ≠ the **coupled** channel.

## Engines

| Engine | Status |
|--------|--------|
| **PySCF** | Default QM (`PySCFEngine`) |

## Tuning properties

**Core:** `gse`, `homo`, `lumo`, `gap`, `dm`, `spin`, `ie`, `ea`, `cp`, `eng`, `hard`, `efl`, `nfl`, `fukui_plus`, `fukui_minus`, `exe`, `osc`

Use `properties = ['all']` for the full registry.

## Protein/ligand occupancy

`PotentialInput` / `potential.in`: `ligand` (or `molecule`), optional `protein`, `ligand_atoms='present'|'absent'`, `quantity='potential'|'charge'`. APBS box spans protein and ligand coordinates even when the ligand is omitted from the PQR.

## Future ideas (not scheduled)

PySCF ESP/MEP backends for the potential channel (`method='esp'` / `'mep'`).
Protein charges from a real PQR/PDB instead of Gasteiger.

## Version milestones

| Version | Scope |
|---------|-------|
| **v1.1** | Refoundation, four channels, tests, CI |
| **v1.2** | Advanced properties, xTB engine, bond scan |
| **v1.3** | Kwargs API, APBS `quantity` (potential or Gauss-law charge) |
| **v1.x+** | Further properties and engine backends as needed |

See [ENGINEERING.md](ENGINEERING.md) and [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

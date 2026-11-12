# Family examples (combined protein exe)

Organized by protein family. Each domain has geometry + singlet/triplet combined `exe` inputs and lean outputs.

| Family | Domains | Notes |
|--------|---------|-------|
| `AtLOV/` | LOV1, LOV2 | Arabidopsis |
| `AsLOV/` | LOV1, LOV2 | Avena |
| `CrLOV/` | LOV1, LOV2 | Chlamydomonas |
| `miniSOG/` | `wt/` (Step_5), `step0/` | engineered SOG |
| `SOPP3/` | `wt/` (Step_5), `step0/` | engineered SOG |


## Input format (2026-11)

The `coupled.in` files use the PDB-based potential channel: `pdb =` names the complex, and
`off = ['CHR']` keeps the chromophore in place with its charges set to zero. The stored
`output/` folders were produced with the earlier pdb2pqr-based channel, which rebuilt the
hydrogen positions; rerunning the potential step with the current code gives different numbers.
Folders without a stored `coupled.surf` need `emsuite -s surface.in` first.

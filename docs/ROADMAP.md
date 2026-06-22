# EMSuite Roadmap

This document captures the approved direction for EMSuite beyond v1.0.5. Shipped features are documented in [README.md](../README.md).

## Three channels

| Channel | Purpose | CLI | Status |
|---------|---------|-----|--------|
| **Surface** | Geometry + VDW envelope | `emsuite -s surface.in` | Shipped |
| **Tuning** | Electrostatic tuning of molecular properties | `emsuite -t tuning.in` | Shipped |
| **Potential** | Electrostatic potential maps on surfaces | `emsuite -p potential.in` | Planned |
| **Coupled** | Potential-derived charges feed tuning | `emsuite -c coupled.in` (name TBD) | Planned |

**Naming note:** `calc_type='combined'` in tuning applies all surface charges simultaneously. The future **coupled** channel is different: it uses potential-computed heterogeneous charges as the tuning surface.

## Engine abstraction (planned)

- **PySCF** — default QM engine today
- **MLIP** — planned alternate engine for geometry, energies, and fast screening maps

## Tuning property backlog

Existing: `gse`, `homo`, `lumo`, `gap`, `dm`, `ie`, `ea`, `cp`, `eng`, `hard`, `efl`, `nfl`, `exe`, `osc`.

Planned (not prioritized yet):

- Water interaction
- Nucleophilicity / electrophilicity fugacity extensions
- Stark effect
- Bond potential
- Interaction energy
- Spin density
- Vibrational frequencies
- Proton affinity map (with equilibrium H bond distance)
- Fukui indices
- Transition-state tuning

## Refoundation milestones

| Version | Scope |
|---------|-------|
| **v1.1** | Repo cleanup, unified config parser, tests, CI, importable API, package restructure |
| **v2.0** | Potential + coupled channels |
| **v2.x** | New tuning properties, MLIP engine |

## Current refoundation work (in progress)

- Remove local dev clutter from repository root
- Replace `exec()` surface input parsing with safe shared parser
- Add pytest unit tests and GitHub Actions CI
- Remove unused `requests` dependency
- Align documentation with the two-stage surface → tuning workflow

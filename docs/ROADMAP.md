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
| **v1.1** | Repo cleanup, unified config parser, tests, CI, Ruff/pre-commit, importable API, package restructure |
| **v2.0** | Potential + coupled channels |
| **v2.x** | New tuning properties, MLIP engine |

## Engineering standards

Refoundation follows:

- [uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter) — `uv`, `src/` layout, Ruff, pytest, pre-commit, GitHub Actions
- [How to Make a Great Open-Source Scientific Project](https://rowansci.com/blog/how-to-make-a-great-open-source-scientific-project) — minimal focused libraries, clean packaging, tested and documented code

See [ENGINEERING.md](ENGINEERING.md) for the concrete checklist and local dev workflow.

## v1.1 refoundation status

**Done (local, unpushed):** dev workspace isolation, safe config parser, 17 unit tests, GitHub Actions CI, Ruff/pre-commit, cookie-cutter subpackage restructure, importable API. See [SESSION_HANDOFF.md](SESSION_HANDOFF.md).

**Remaining before v1.1 release:**

- End-to-end integration and regression tests
- Extract Ray parallel workers from `tuning/runner.py`
- Push to remote and tag v1.1.0 after testing

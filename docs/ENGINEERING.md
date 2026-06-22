# Engineering Guide

EMSuite refoundation follows two reference guides:

- [uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter) — project layout, `uv` workflow, Ruff, pytest, pre-commit, GitHub Actions
- [How to Make a Great Open-Source Scientific Project](https://rowansci.com/blog/how-to-make-a-great-open-source-scientific-project) — principles for minimal, packaged, clean, tested scientific Python

This document maps those principles to EMSuite.

## Rowan principles → EMSuite

| Principle | Target for EMSuite |
|-----------|-------------------|
| **Minimal** | Three focused channels (surface, tuning, potential) + coupled orchestration. Engines (PySCF, MLIP) behind an interface — not bundled into one monolith. |
| **Packaged** | `pip install emsuite` / `uv sync`. Loose deps in `pyproject.toml`, lockfile for reproducible dev. Test on Python 3.11+. |
| **Clean** | Ruff format + lint on every commit (pre-commit). Type hints on public API; `py.typed` marker. |
| **Tested** | pytest unit tests on every PR; integration/regression marked `@pytest.mark.slow`. Coverage tracked in CI. |
| **Intuitive** | Importable API (`emsuite.tuning.run`, etc.) with keyword args and clear names. CLI is a thin wrapper. |
| **Documented** | README for users, Sphinx for reference, `ROADMAP.md` for direction, docstrings on public functions. |
| **Maintained** | Semver, `CHANGELOG.md`, deprecation warnings before breaking changes. |
| **Open source** | MIT license in `LICENSE`. |

## uv-cookiecutter alignment

| uv-cookiecutter feature | EMSuite status |
|-------------------------|----------------|
| `src/` layout | Done (`src/emsuite/`) |
| `uv` + `uv.lock` | Done |
| `.python-version` | Done |
| Ruff format/lint | Done (see `pyproject.toml`, `.pre-commit-config.yaml`) |
| pytest | Done (`tests/unit/`) |
| pre-commit hooks | Done |
| GitHub Actions CI | Done (`.github/workflows/`) |
| Codecov | Planned |
| `ty` / mypy type checking | Planned (enable in CI after API stabilizes) |
| `direnv` | Optional for local dev |

## Recommended local workflow

```bash
# One-time setup
uv sync --extra dev
uv run pre-commit install

# Daily development
uv run ruff check --fix src tests
uv run ruff format src tests
uv run pytest tests/unit -v

# Before pushing
uv run pre-commit run --all-files
```

## Package structure (target)

Aligned with uv-cookiecutter `src/` layout and Rowan’s “minimal libraries” goal:

```
src/emsuite/
  config/          # safe input parsing
  core/            # shared QM primitives
  engines/         # PySCF, MLIP (protocol)
  surface/         # VDW surface generation
  tuning/          # property tuning maps
  potential/       # ESP maps (planned)
  coupled/         # potential → tuning (planned)
  cli/             # argparse entry points
tests/
  unit/
  integration/
  regression/
docs/
examples/
```

Migrate incrementally; tests must stay green after each move.

## What not to do

- Do not pin runtime deps aggressively in a library (lockfile is for dev/apps only).
- Do not add training/MLIP dev deps to the default install.
- Do not grow `tuning.py` with new properties — use the property registry pattern.
- Do not commit local research work — use the gitignored [`dev/`](../dev/README.md) directory for protein runs, ad-hoc scripts, and local inputs.

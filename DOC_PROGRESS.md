# EMSuite Documentation Migration Progress

This file is the durable implementation and test journal for the migration from
Sphinx/Shibuya to Zensical. Update it after every meaningful step with the
command run, result, evidence, and next action.

Plan source: [`codexdoc.md`](codexdoc.md)

## Status summary

| Step | State | Outcome |
| --- | --- | --- |
| 0. Repository and environment baseline | Complete | Branch and parallel-agent state recorded |
| 1. Existing Sphinx build baseline | Complete | Both documented dependency paths fail; diagnostic build finds 11 warnings |
| 2. Zensical dependency and minimal build | In progress | Isolated strict build passes; project dependency replacement still pending |
| 3. Navigation and foundational pages | Pending | Home, Quick Start, Concepts |
| 4. Guides and reference migration | Pending | Surface, Potential, Tuning, Coupled, Python, reference |
| 5. Internal/public documentation separation | Pending | Keep private records outside the generated site |
| 6. Sphinx retirement | Pending | Remove old build system only after content parity |
| 7. CI and GitHub Pages | Pending | PR build plus `main` deployment |
| 8. Visual and acceptance QA | Pending | Desktop/mobile, themes, search, links, snippets |

## Test protocol

For every step:

1. Record the starting commit and working-tree changes.
2. Make only changes owned by the documentation migration.
3. Run the narrowest relevant test first.
4. Run `uv run zensical build --clean` after Zensical is introduced.
5. Record warnings and failures verbatim enough to reproduce them.
6. Do not remove the previous documentation path until the replacement passes.
7. Preserve changes made by parallel agents.

## Step 0 — Repository and environment baseline

**Timestamp:** 2026-09-01T17:32:14-04:00

### Commands

```bash
git status --short --branch
git log -3 --oneline --decorate
git rev-parse HEAD
python3 --version
uv --version
find docs -type f | sort | wc -l
find docs -name '*.rst' -type f | wc -l
rg -l 'Coming soon' docs -g '*.rst' | sort
```

### Repository state

```text
Branch: feat/streamline-inputs-ligand-occupancy
HEAD: 7620921daafe61d6cd4a508f90a28486fe6b48c7
Tracking: origin/feat/streamline-inputs-ligand-occupancy
Ahead of origin: 3 commits
Existing untracked file: codexdoc.md
```

The three existing commits ahead of the remote belong to ongoing implementation
work and must not be rewritten or reverted by the documentation migration:

```text
7620921 Fix run_apbs_grids relative-workdir path bug
89bd33f Fix APBS run failures and Gasteiger charge bug in potential channel
b71d526 Collapse api.py into Input.from_config().run(), fix coupled/tuning naming
```

### Concurrent-work checkpoint

After the isolated documentation tests completed, additional parallel-agent
changes appeared in the shared worktree. They include modifications to
`pyproject.toml`, `uv.lock`, runtime modules, and tests, plus new potential/PDB
modules and tests. These changes are not part of the documentation work and
were not edited or reverted during this session.

Because Step 2 must eventually modify both `pyproject.toml` and `uv.lock`, the
repository-local dependency replacement is paused until the parallel agent has
finished or its current edits have been integrated. The isolated `/tmp` proof
allowed Zensical feasibility testing to continue without creating a conflict.

### Tool versions

```text
Host Python: 3.14.0
uv: 0.8.3 (7e78f54e7 2025-07-24)
Project build Python selected by uv: 3.11.15
```

### Current documentation inventory

```text
Files under docs/: 38
reStructuredText pages: 24
Public pages containing "Coming soon": 10
```

Placeholder pages:

```text
docs/combined/explanation/index.rst
docs/combined/how-to/index.rst
docs/combined/reference/index.rst
docs/combined/tutorials/index.rst
docs/potential/explanation/index.rst
docs/potential/how-to/index.rst
docs/potential/reference/index.rst
docs/potential/tutorials/index.rst
docs/tuning/explanation/what-is-tuning.rst
docs/tuning/how-to/index.rst
```

### Findings

- `docs/conf.py` reports release `1.4.0`, while `pyproject.toml` and the package
  report `1.3.0`.
- `pyproject.toml`'s `docs` extra contains only Sphinx and Myst Parser.
- `conf.py` additionally imports `sphinx_copybutton` and `sphinx_design`.
- `docs/requirements.txt` contains `sphinx-copybutton` but omits
  `sphinx-design`.
- The current documentation therefore has two dependency declarations and
  neither is complete.

## Step 1 — Existing Sphinx build baseline

Goal: establish whether the current documentation can be reproduced before any
migration work begins.

### Test 1A — Build using the `docs` optional dependency

Command:

```bash
uv run --extra docs sphinx-build -W --keep-going -b html \
  docs /tmp/emsuite-docs-sphinx-baseline
```

Result: **FAIL** (exit code 2)

Primary error:

```text
ExtensionError: Could not import extension sphinx_copybutton
(exception: No module named 'sphinx_copybutton')
```

Interpretation: the package-supported `docs` extra cannot build the existing
site because it does not declare all extensions loaded by `docs/conf.py`.

### Test 1B — Build using `docs/requirements.txt`

Command:

```bash
uv run --with-requirements docs/requirements.txt \
  sphinx-build -W --keep-going -b html \
  docs /tmp/emsuite-docs-sphinx-requirements
```

Result: **FAIL** (exit code 2)

Primary error:

```text
ExtensionError: Could not import extension sphinx_design
(exception: No module named 'sphinx_design')
```

Interpretation: the standalone requirements file is also incomplete.

### Test 1C — Diagnostic build with the missing dependency supplied

This command does not change repository dependencies. It injects the missing
package only to reveal page-level build health.

Command:

```bash
uv run --with-requirements docs/requirements.txt --with sphinx-design \
  sphinx-build -W --keep-going -b html \
  docs /tmp/emsuite-docs-sphinx-complete
```

Result: **FAIL STRICT BUILD** (exit code 1)

Progress before failure:

```text
Sphinx version: 9.0.4
Python: 3.11.15
Sources read: 24/24
HTML output generated: yes
Warnings: 11
```

Warnings:

```text
8 documents are not included in any toctree:
  combined/explanation/index.rst
  combined/how-to/index.rst
  combined/reference/index.rst
  combined/tutorials/index.rst
  potential/explanation/index.rst
  potential/how-to/index.rst
  potential/reference/index.rst
  potential/tutorials/index.rst

3 unknown document references:
  docs/combined/index.rst -> ../ROADMAP
  docs/potential/index.rst -> ../ROADMAP
  docs/tuning/reference/inputs.rst -> ../../ROADMAP
```

Interpretation:

- Existing prose is parseable when the full extension set is installed.
- The current documented installation paths are not reproducible.
- Strict builds fail because placeholder pages are orphaned and ROADMAP links
  use invalid Sphinx document targets.
- These problems should be resolved by the Zensical content/navigation design,
  not patched into the old Sphinx system unless needed for migration tooling.

## Step 2 — Zensical dependency and minimal build

State: **In progress**

### Test 2A — Invoke pinned Zensical through the current project

Command:

```bash
uv run --with zensical==0.0.57 zensical --version
```

Result: **FAIL** (exit code 1)

Dependency conflict:

```text
pdb2pqr==3.7.1 requires docutils<0.18
sphinx>=7.0 requires docutils>=0.18.1
emsuite[docs] currently requires sphinx>=7.0
```

Interpretation:

- The failure occurs before Zensical runs.
- The current Sphinx `docs` extra is incompatible with the project's resolved
  `pdb2pqr` dependency across supported Python versions.
- Zensical does not cause this conflict; invoking a tool through the project
  exposes the already-invalid Sphinx dependency group.
- The migration should replace the Sphinx `docs` extra atomically instead of
  attempting to keep both toolchains in the project dependency graph.

### Test 2B — Verify pinned Zensical in an isolated tool environment

Commands:

```bash
uvx --from zensical==0.0.57 zensical --version
uvx --from zensical==0.0.57 zensical build --help
```

Result: **PASS**

```text
Zensical: 0.0.57
Supported build flags confirmed:
  --config-file
  --clean
  --strict
```

### Test 2C — Disposable strict site build

Created outside the repository:

```text
/tmp/emsuite-zensical-proof/zensical.toml
/tmp/emsuite-zensical-proof/docs/index.md
```

The proof configuration used the planned GitHub Pages URL, Geist fonts,
light/dark palettes, search highlighting, code copy, footer navigation, and a
one-page navigation tree.

Command:

```bash
cd /tmp/emsuite-zensical-proof
uvx --from zensical==0.0.57 zensical build --clean --strict
```

Result: **PASS**

```text
Build started
No issues found
Build finished in 0.21s
```

Generated evidence:

```text
site/index.html
site/404.html
site/search.json
site/sitemap.xml
site/objects.inv
```

HTML checks:

```text
Canonical URL: https://sajagbe.github.io/emsuite/
Stylesheet paths: relative ./assets/... URLs
Search worker: generated
Code-copy feature: configured
```

Interpretation: Zensical 0.0.57 is compatible with the planned configuration,
strict builds, and GitHub Pages project URL. It is safe to proceed with an
atomic replacement of the `docs` dependency group and a repository-local site
skeleton.

### Remaining Step 2 sequence

1. Replace the `docs` optional dependency with `zensical==0.0.57`.
2. Regenerate `uv.lock` without changing runtime dependency versions.
3. Add the repository `zensical.toml` and the first Markdown homepage.
4. Run:

   ```bash
   uv sync --extra docs
   uv run zensical build --clean
   ```

5. Confirm output is generated beneath `site/` and works under the `/emsuite/`
   project path.
6. Record the build output, warnings, and generated
   routes here before proceeding to content conversion.

Acceptance criteria:

- A clean environment can install the locked docs dependency set.
- `uv run zensical build --clean` exits successfully.
- No existing Sphinx file is removed during this proof step.
- Runtime dependencies and tests remain unchanged.

## Step 2 — Repository Zensical implementation (complete)

Date: 2026-09-01

Branch preparation:

- The documentation branch was realigned on runtime commit `b3177b8`.
- Planning commit `468aed3` was retained on `feat/zensical-docs`.
- The pre-realignment branch tip remains available as
  `archive/zensical-docs-pre-realign`.

Implemented:

- Replaced the incompatible Sphinx docs extra with `zensical==0.0.57` and
  regenerated `uv.lock`.
- Added `zensical.toml`, Calcflow-inspired theme configuration, navigation,
  Geist typography, light/dark palettes, search, code-copy controls, and
  repository metadata.
- Added 17 public Markdown pages: Home, Quick Start, Concepts, five Guides,
  six Reference pages, and three Developer pages.
- Reused the existing Water animation at
  `docs/assets/images/water-test.gif` and added focused site CSS.
- Added `site/` to `.gitignore`.
- Moved private engineering/session records to `internal-docs/` and preserved
  the complete former Sphinx source under `internal-docs/legacy-sphinx/`.
- Added `.github/workflows/docs.yml` for strict pull-request builds and GitHub
  Pages deployment from `main`.

Dependency and first strict-build command:

```bash
uv lock
uv run --extra docs zensical build --clean --strict
```

The first repository build correctly failed strict mode with two broken links,
both originating in private `ROADMAP.md` and `SESSION_HANDOFF.md` files that
Zensical discovered under the public source directory. Moving those records to
`internal-docs/` fixed the content boundary without deleting history.

Final strict-build command:

```bash
uv run --extra docs zensical build --clean --strict
```

Result: **PASS**

```text
Build started
No issues found
Build finished in 0.39s
```

Generated-route and asset checks:

```bash
test -f site/index.html
test -f site/quick-start/index.html
test -f site/reference/python-api/index.html
rg -n 'water-test.gif|lucide/atom' site/index.html
```

Result: all routes exist; the Water image resolves as
`assets/images/water-test.gif`. An invalid icon-as-image logo override found
during HTML inspection was removed before the final build.

CLI verification command:

```bash
env -u PYTHONPATH uv run --project \
  /Users/stephenajagbe/orca/emsuite emsuite --help
```

Result: **PASS** — the executable exposes the documented `-s/--surface`,
`-p/--potential`, `-t/--tuning`, and `-c/--coupled` modes.

Runtime regression command:

```bash
env -u PYTHONPATH uv run --extra dev pytest tests/unit tests/regression -q
```

Result: **PASS**

```text
63 passed in 1.31s
```

Locked CI simulation:

```bash
uv sync --extra docs --locked
uv run --extra docs zensical build --clean --strict
```

Result: **PASS** — lock resolution completed without changes and the strict
build reported no issues in 0.42 seconds.

Parallel-work safeguard: edits present in `src/emsuite/config/schemas.py` and
`src/emsuite/potential/runner.py` were not made by this documentation work and
must remain outside the documentation commit.

## Next action

Inspect the deployed GitHub Pages result after this branch is merged to `main`,
then expand scientific examples with additional validated output captures.

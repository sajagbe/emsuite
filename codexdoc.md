# EMSuite Zensical Documentation Plan

## Summary

Replace the existing Sphinx/Shibuya site with a Zensical site modeled on
[Calcflow's documentation](https://calcflow.ischemist.com/).

The first release will be a polished foundation, not a mechanical format
conversion. It will document the current
`feat/streamline-inputs-ligand-occupancy` behavior, including the CLI and the
new immutable `Input`/`Result` Python API.

Initial publishing target:

```text
https://sajagbe.github.io/emsuite/
```

No EMSuite runtime APIs or scientific behavior will change.

## Implementation Changes

### 1. Replace the documentation toolchain

- Add a root `zensical.toml` configured with:
  - `site_name = "EMSuite Docs"`;
  - project description, author, repository URL, and GitHub Pages URL;
  - Geist and Geist Mono fonts;
  - light, dark, and system palettes;
  - search highlighting, code copying, instant navigation, breadcrumbs,
    navigation sections, back-to-top, footnote tooltips, and footer navigation;
  - GitHub repository and social links.
- Pin the current official Zensical release in the `docs` optional dependency
  and regenerate `uv.lock`.
- Remove Sphinx, Shibuya, Myst, `conf.py`, Sphinx Makefiles, and obsolete
  documentation requirements.
- Convert public `.rst` content to Markdown using native Zensical features:
  tabs, admonitions, tables, code annotations, icons, cards, and page metadata.
- Replace the current gradient-on-every-link CSS with restrained EMSuite
  branding: blue primary, red accent, accessible contrast, and optional
  hero/card styling.
- Preserve the existing Water animation as a homepage or Tuning visual.

### 2. Establish the Calcflow-style information architecture

Use this explicit navigation:

```text
Home
Quick Start
Concepts
Guides
  Surface Generation
  Potential Mapping
  Tuning Maps
  Coupled Workflows
  Python Workflows
Reference
  Command-Line Interface
  Input Configuration
  Python API
  Properties
  Output Files
  File Formats
Developers
  Architecture
  Testing and Validation
  Contributing Documentation
```

Content requirements:

- **Home:** concise product statement, badges, feature list, installation tabs,
  four-channel capability table, minimal CLI/Python examples, Water visual, and
  next-step links.
- **Quick Start:** validated small Water workflow covering installation, input
  preparation, execution, expected artifacts, and result inspection without
  promising unrealistic runtime.
- **Concepts:** explain Surface -> Potential/Tuning -> Coupled data flow,
  `.surf` semantics, homogeneous probes versus heterogeneous potential-derived
  values, and immutable `Input`/`Result` objects.
- **Guides:** task-oriented end-to-end pages for Surface, Potential, Tuning,
  Coupled, and Python usage.
- **Reference:** document current configuration keys and defaults directly from
  validators/dataclasses, all four CLI switches, property dependencies,
  generated artifacts, and public Python types/functions.
- **Developers:** document the channel architecture, how configuration becomes
  typed input, test markers, fast versus scientific tests, and the
  documentation contribution workflow.
- Migrate the useful existing Tuning tutorial and reference material, but
  remove inaccurate `.etm` terminology in favor of `.surf`.
- Spell "homogeneous" and "heterogeneous" correctly in prose while explicitly
  documenting the compatibility configuration literals `homogenous` and
  `heterogenous`.
- Eliminate every public "Coming soon" page; unfinished material stays out of
  navigation.

### 3. Separate public and internal documentation

- Keep only user-facing documentation beneath the Zensical public `docs/` tree.
- Move engineering handoffs, session records, implementation plans, and
  historical changelogs into a repository-level internal documentation area
  that Zensical does not publish.
- Repair affected relative links and add a short internal index so parallel
  agents can still discover those records.
- Use `pyproject.toml` and `emsuite.__version__` as version authorities; remove
  the stale Sphinx `1.4.0` value while the package remains `1.3.0`.

### 4. Add documentation CI and deployment

Create a GitHub Actions documentation workflow that:

- runs for documentation-relevant pull requests and pushes to `main`;
- installs the locked `docs` dependency set with `uv`;
- runs `zensical build --clean`;
- fails on build errors, missing pages, or broken internal links;
- uploads the generated `site/` directory as a GitHub Pages artifact only on
  `main`;
- deploys through `actions/deploy-pages`;
- uses Pages concurrency so deployments cannot overlap.

Document local commands:

```bash
uv sync --extra docs
uv run zensical serve
uv run zensical build --clean
```

A custom domain is intentionally deferred.

## Public Interfaces

No runtime interfaces change. Documentation must cover the current public
surface:

- `SurfaceInput`, `PotentialInput`, `TuningInput`, `CoupledInput`;
- `SurfaceResult`, `PotentialResult`, `TuningResult`, `CoupledResult`;
- `run_surface_calculation`, `run_potential_calculation`,
  `run_tuning_calculation`, and `run_coupled_calculation`;
- `emsuite -s`, `-p`, `-t`, and `-c`;
- `.from_file()`, `.from_mapping()`, `.from_config()`, `.to_dict()`, `.run()`,
  and `.to_surf()` where applicable.

The API reference will be hand-authored from these typed interfaces for the
first release, avoiding dependence on unsupported or unstable autodoc plugins.

## Test Plan

- Build from a clean checkout using only the locked `docs` extra.
- Confirm every configured navigation entry resolves and no public placeholder
  pages remain.
- Validate internal and external links, anchors, images, code-copy controls,
  search indexing, and GitHub source links.
- Check desktop and mobile layouts in light, dark, and system modes.
- Verify the homepage, Quick Start, one Guide, and one Reference page visually
  against Calcflow's hierarchy and readability.
- Run existing unit/regression tests to ensure dependency changes do not affect
  EMSuite.
- Validate CLI snippets against `emsuite --help` and current templates.
- Validate Python snippets against the current typed API without running
  expensive quantum calculations.
- Perform one GitHub Pages staging deployment and verify asset paths work under
  `/emsuite/`.

Acceptance criteria:

- `uv run zensical build --clean` succeeds from a clean checkout.
- The deployed site is searchable, responsive, and complete for the
  first-release navigation.
- Public examples match current branch behavior and accepted configuration
  values.
- No Sphinx build path remains in dependencies, CI, or contributor instructions.
- Internal session and planning records are not published.
- The documentation workflow deploys only from `main`.

## Assumptions

- The site should reproduce Calcflow's information density and navigation
  behavior while retaining EMSuite identity; it should not be a pixel-for-pixel
  clone.
- GitHub Pages project hosting is the initial production target.
- English is the only first-release language.
- Documentation changes are delivered independently of runtime refactors.
- The implementation should follow the current official
  [Zensical setup](https://zensical.org/docs/get-started/) and Calcflow's public
  [Zensical configuration](https://github.com/ischemist/project-prometheus/blob/master/zensical.toml).

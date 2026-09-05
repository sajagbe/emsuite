---
icon: lucide/book-open-check
---

# Contributing Documentation

## Local preview

```bash
uv sync --extra docs
uv run zensical serve
```

Open `http://127.0.0.1:8000/emsuite/` when `site_url` uses the GitHub Pages
project path.

## Strict build

```bash
uv run zensical build --clean --strict
```

The strict build must pass before documentation changes are merged.

## Content rules

- Prefer tested commands and concrete outputs.
- Link to the nearest relevant guide or reference page.
- Keep tutorials task-focused and reference pages exhaustive.
- Use correct prose spelling while preserving literal compatibility values.
- Do not publish session handoffs, implementation plans, or run manifests.
- Update `DOC_PROGRESS.md` for migration checkpoints and build evidence.

## Page metadata

Use a Lucide icon where it helps navigation:

```yaml
---
icon: lucide/book-open
---
```

Use tabs for equivalent installation methods, admonitions for actionable notes,
and tables for field-oriented reference material.

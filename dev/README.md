# Local development workspace

This directory is **gitignored**. Put all local research, protein runs, ad-hoc
scripts, and input files here — nothing in this folder is part of the published
package.

## Suggested layout

```
dev/
  proteins/          # Application-Prelims, asL1, atL2, etc.
  molecules/         # XYZ files for local runs
  scripts/           # one-off utilities (STG_LOV.py, etc.)
  inputs/            # surface.in, tuning.in for your runs
  outputs/           # results_*, *.surf, logs
```

Move existing subfolders here as you like; EMSuite does not read from `dev/` unless
you point an input file at a path inside it.

## Package inputs

Use tracked templates under `examples/templates/` or the CCO example under
`examples/tuning/CCO2-exe/`.

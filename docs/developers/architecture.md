---
icon: lucide/blocks
---

# Architecture

EMSuite uses a consistent channel pipeline:

```text
.in file / mapping / kwargs
          │
          ▼
  validation + defaults
          │
          ▼
 immutable Input dataclass
          │ run()
          ▼
      channel runner
          │
          ▼
   typed Result object
```

## Boundaries

- `inputs.py` owns the public immutable configuration objects.
- `config/schemas.py` owns cross-entry-point validation.
- Channel packages own parsing defaults, scientific execution, and file I/O.
- `results.py` owns portable return types and format conversion.
- `cli/main.py` maps one CLI option to one typed input.

## Design rules

- Validate before external programs or quantum calculations start.
- Keep the same semantics for file, mapping, and keyword configuration.
- Use `.surf` as the explicit channel boundary.
- Keep Potential quantity provenance on `PotentialResult`.
- Preserve both intermediate and final results in Coupled workflows.
- Avoid hidden changes to scientific defaults during interface refactors.

## Adding a field

Update the channel defaults, validator, input dataclass, runner handoff, template,
tests, and input reference together. For coupled Potential fields, also forward
the value from `CoupledInput` into `PotentialInput`.

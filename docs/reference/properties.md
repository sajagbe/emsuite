---
icon: lucide/sigma
---

# Properties

| Code | Property | Additional state |
| --- | --- | --- |
| `gse` | ground-state energy | neutral |
| `homo` | highest occupied orbital energy | neutral |
| `lumo` | lowest unoccupied orbital energy | neutral |
| `gap` | HOMO–LUMO gap | neutral |
| `dm` | dipole-moment magnitude | neutral |
| `spin` | spin-related observable | neutral |
| `ie` | ionization energy | cation |
| `ea` | electron affinity | anion |
| `cp` | chemical potential | cation + anion |
| `eng` | electronegativity | cation + anion |
| `hard` | chemical hardness | cation + anion |
| `efl` | electrophilicity | cation + anion |
| `nfl` | nucleophilicity | derived response |
| `fukui_plus` | electrophilic Fukui response | anion/neutral |
| `fukui_minus` | nucleophilic Fukui response | neutral/cation |
| `exe` | excitation energies | excited-state calculation |
| `osc` | oscillator strengths | excited-state calculation |
| `all` | all supported properties | all required states |

```python
properties = ["homo", "lumo", "gap"]
```

`state_of_interest` selects the number of excited states used by `exe` and
`osc`. Set `triplet=True` for triplet-state analysis where supported.

Dependencies are resolved automatically; requesting a derived property causes
the necessary primary calculations to be scheduled.

---
icon: lucide/map
---

# Tuning Maps

Tuning places the charge from each `.surf` point into the molecular Hamiltonian
and records the resulting change in requested properties.

## Input

```python title="tuning.in"
molecule = "water.xyz"
surface_file = "water.surf"
properties = ["gse", "homo", "lumo", "gap", "dm"]
basis_set = "6-31G*"
method = "dft"
functional = "b3lyp"
charge = 0
spin = 0
solvent = None
calc_type = "separate"
parallel = True
num_procs = 4
state_of_interest = 2
triplet = False
```

```bash
emsuite -t tuning.in
```

## Property dependencies

EMSuite schedules the electronic states needed by the selected properties.
For example, ionization energy requires a cation calculation, electron affinity
requires an anion calculation, and derived reactivity descriptors require both.
See the [property reference](../reference/properties.md).

## Parallel execution

With `parallel=True`, Ray distributes separate surface points across available
CPU or GPU workers. `num_procs` is an upper bound; EMSuite limits it to detected
resources. For reproducibility:

- capture the EMSuite commit and dependency versions;
- keep molecule, surface, and input files together;
- write each run to a fresh directory;
- compare scientific values independently of timestamps and worker ordering.

## Separate versus combined

- `separate` evaluates points independently and produces a map.
- `combined` applies all surface charges in one calculation.

The Coupled channel is different: it generates or loads a potential surface and
then passes it into Tuning.

## Results

Each property produces raw and normalized MOL2 maps. The summary CSV preserves
coordinates, baselines, raw effects, and normalized effects for quantitative
analysis. Point logs and resume metadata support audit and recovery.

See [Outputs](../reference/outputs.md) for the exact layout.

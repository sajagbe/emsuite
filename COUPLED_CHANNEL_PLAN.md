# Surface → environment charges → tuning

Implemented in the working tree after checkpoint tag
`potential-channel-checkpoint-2026-11-12`; included in the surface/coupled assembly-line commit.

## Current behavior

`emsuite -p potential.in` runs independently from an input specifying a PDB and
an existing surface, subject to supported chemistry and stated unknown-residue charges.
`emsuite -c coupled.in` generates a surface automatically when none is supplied,
then runs potential-derived charges → tuning. Explicit surface or charge reuse remains available.
Tuning consumes point charges in electrons, not electrostatic potentials in kT/e.

## Intended coupled workflow

One input and one command: `emsuite -c coupled.in`.

1. Validate all inputs before expensive work. Use the same molecular XYZ for
   surface generation and tuning, in the coordinate frame of the environment PDB.
2. Reuse an explicitly supplied valid surface, or generate the tuned molecule's
   surface automatically with the shared surface implementation. Preserve coordinates;
   do not optimize or recenter the molecule implicitly.
3. Assign PDB atomic charges/radii, apply explicit off/removed selections, solve
   APBS, and derive the charge-valued surface with the existing Gauss-law method.
4. Pass that charge-valued surface directly into tuning. Reject potential-valued
   surfaces in the coupled path before starting quantum calculations.
5. Save inspectable surface geometry, charged surface, PQR/source table, tuning
   outputs and stage timings in one run directory. No manual handoff between commands.

The PDB describes the environment; the molecular XYZ defines what is tuned and
where its sampling surface is generated. A potential is not a charge, and must
never be silently interpreted as one merely because both fit a fourth file column.

## Input layout to implement

```python
molecule = 'ligand.xyz'
pdb = 'complex.pdb'

# Omit surface_file to generate automatically; supply it to reuse coordinates.
# surface_file = 'ligand.surf'
surface_density = 0.5
surface_scale = 1.0

off = ['LIG']                  # explicit environment selection, charge off/radius kept
# grid_spacing = 0.3          # default decision still awaits charge/grid review

properties = ['homo', 'lumo', 'gap']
method = 'dft'
functional = 'b3lyp'
basis_set = '6-31g*'
charge = 0
spin = 0
```

The coupled parser accepts surface_density/surface_scale and no longer requires
a surface_file. Potential settings
and tuning settings continue to use their existing separate meanings. Coupled mode
always requests charge; standalone potential mode still supports potential or charge.

## Shared surface service

Use one coordinate-generation implementation across standalone surface, potential,
tuning and coupled orchestration. Keep point coordinates separate from their values
and record whether values are placeholders, assigned charges or potentials, with units.
Preserve existing standalone commands and explicit surface reuse.

Surface geometry generation should not invoke SCF or optimization unless explicitly
requested in a geometry-preparation workflow. Fix executable discovery for vsg, isolate
its temporary files to avoid collisions, and check finite/nonempty coordinates and
coordinate-frame compatibility. Time generation separately from writing/reading.
Reuse requires matching geometry, radii and surface settings, not just an existing filename.
No promise of instantaneous geometry optimization or universal chemical support.

## Implementation order and acceptance

- [x] ~~Make the shared surface call reliable: executable resolution, isolated temporary
  files, no implicit geometry changes, coordinate validation and timing.~~ First changes
  implemented after tag `potential-channel-checkpoint-2026-11-12` and validated before commit.
- [x] ~~Add automatic surface generation to coupled inputs while retaining explicit reuse.~~
- [x] ~~Enforce charge-only coupled handoff; distinguish file value types/units.~~ New
  outputs carry header metadata; untyped legacy charge inputs remain accepted with a warning.
- [x] ~~Use one run directory with inspectable outputs and stage timing.~~
- [x] ~~Verify generate/reuse parity, unchanged coordinates, standalone channel behavior,
  early rejection of potential-as-charge input, and end-to-end coupled results.~~ 221 unit
  tests and 8 CPU integration checks passed across validation and the corrected fixture rerun.
- [ ] Check convergence of final surface charges and tuning results as well as potentials;
  potential-map convergence alone does not establish charge/tuning convergence.

Covalently bonded unknown-residue support remains deferred. No new Slurm scope,
additional implementation or commit is authorized merely by listing this plan.

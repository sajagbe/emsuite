# SOPP3 wt (SOPP3)

Combined `exe` maps (singlet + triplet), `calc_type=combined`, `charge=-2`, B3LYP/6-31G*.

Source: pghosh4 SOPP3 Step_5 Selected_100.gro frame 51/100. Combined jobs 4319915–16.

## Layout
- `ligand.xyz`, `complex.pdb`, `CHR.mol2` — middle Selected_100 frame (SOL/ions stripped)
- `singlet/`, `triplet/` — `coupled.in`, `run.slurm`, `output/` (summary + mol2)

## Run
```bash
cd singlet   # or triplet
# from a GPU allocation with emsuite on PATH:
emsuite -c coupled.in
```

# CrLOV LOV2 (CR2)

Combined `exe` maps (singlet + triplet), `calc_type=combined`, `charge=-2`, B3LYP/6-31G*.

Source: Selected100-L1vL2/CR2Sel.gro middle frame. Combined jobs 4295648–49.

## Layout
- `ligand.xyz`, `complex.pdb`, `CHR.mol2` — middle Selected_100 frame (SOL/ions stripped)
- `singlet/`, `triplet/` — `coupled.in`, `run.slurm`, `output/` (summary + mol2)

## Run
```bash
cd singlet   # or triplet
# from a GPU allocation with emsuite on PATH:
emsuite -c coupled.in
```

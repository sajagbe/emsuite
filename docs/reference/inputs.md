---
icon: lucide/list-checks
---

# Input Configuration

`.in` files use Python-style `key = value` assignments. The same fields are
available through `Input.from_config()` and `Input.from_mapping()`.

## Surface fields

| Field | Default | Notes |
| --- | --- | --- |
| `input_type` | required | `XYZ` or `SMILES` |
| `input_data` | required | XYZ path or SMILES text |
| `output_surf` | `surface.surf` | output path |
| `optimized_xyz` | `None` | optional optimized geometry path |
| `surface_density` | `1.0` | points per square ångström |
| `surface_scale` | `1.0` | van der Waals radius multiplier |
| `surface_type` | `homogenous` | compatibility literal |
| `surface_charge` | `0.10` | uniform fourth-column value |
| `optimize` | input-dependent | enable geometry optimization |
| `optimize_method` | `mmff` | `mmff`, `uff`, or `pyscf` |
| `method` / `basis_set` / `functional` | `dft` / `6-31G*` / `b3lyp` | QM optimization |
| `solvent` / `charge` / `spin` | `None` / `0` / `0` | molecular state |

## Potential fields

| Field | Default | Notes |
| --- | --- | --- |
| `molecule` or `ligand` | required | ligand geometry |
| `surface_file` | `None` | reuse a sampling surface |
| `output_surf` | `potential.surf` | output path |
| `surface_density` / `surface_scale` | `0.5` / `1.0` | generated surface |
| `method` | `apbs` | only implemented backend |
| `quantity` | `potential` | `potential` or `charge` |
| `pdie` / `sdie` | `2.0` / `78.54` | dielectric constants |
| `protein` | `None` | optional environment |
| `ligand_atoms` | `present` | `present`, `absent`, or `charged` |
| `protein_format` | `xyz` | `xyz` or `pdb` |
| `ligand_resname` | `None` | required for PDB mode |
| `ligand_chain` / `ligand_resseq` | `None` | residue disambiguation |
| `ligand_mol2` | `None` | required for present/charged PDB ligand |
| `forcefield` / `ph` | `AMBER` / `7.0` | PDB2PQR settings |

## Tuning fields

| Field | Default | Notes |
| --- | --- | --- |
| `molecule` | required | XYZ path; `xyz_file` remains an alias |
| `surface_file` | required | charged `.surf` path |
| `properties` | `all` | property codes |
| `basis_set` / `method` / `functional` | `6-31G*` / `dft` / `b3lyp` | QM model |
| `charge` / `spin` / `solvent` | `0` / `0` / `None` | molecular state |
| `calc_type` | `separate` | `separate` or `combined` |
| `parallel` / `num_procs` | `True` / `None` | Ray execution |
| `state_of_interest` / `triplet` | `2` / `False` | excited states |

## Coupled fields

Coupled accepts the Potential and Tuning fields under one configuration. Use
`potential_method` and `potential_quantity` for the Potential stage. Set
`potential_surf` to reuse a previously generated surface and skip APBS.

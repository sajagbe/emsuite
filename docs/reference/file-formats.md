---
icon: lucide/file-json-2
---

# File Formats

## XYZ geometry

```text
3
Water
O  0.008935  0.404022  0.000000
H -0.787313 -0.184699  0.000000
H  0.778378 -0.219323  0.000000
```

The first line is the atom count, the second is a comment, and remaining lines
contain element and Cartesian coordinates in ångström.

## SURF point table

```text
x          y          z          q
0.330051   0.258893   1.478588   0.100000
```

The header is optional in some readers. The first three columns are point
coordinates; the fourth is a uniform probe charge, APBS potential, or
potential-derived charge depending on provenance.

## MOL2 visualization map

EMSuite represents surface points as pseudo-atoms. The MOL2 charge column stores
the mapped property or potential value. These files are intended for spatial
visualization rather than chemical topology.

## Input configuration

```python
key = "value"
number = 1.0
enabled = True
items = ["a", "b"]
```

Input files are parsed as restricted configuration assignments. Use the field
reference rather than adding executable Python expressions.

## PDB, PQR, and ligand MOL2

PDB protein workflows use PDB2PQR to assign protein charges and radii. Ligand
selection is based on residue name with optional chain and residue number.
Present or charged PDB ligands require a ligand MOL2 file for atom-level charge
and radius data.

"""CPU (NumPy + PySCF) port of gpu4pyscf.pop.esp, gpu4pyscf 1.4.3.

Line-for-line port of the GPU module so the two can be compared on the same
density matrix. Same grid (Merz-Kollman shells at 1.4/1.6/1.8/2.0 x R_VDW),
same linear system, same RESP hyperbolic restraint. Only the array library
(NumPy instead of CuPy) and the potential integral (PySCF `int1e_grids`
instead of gpu4pyscf's) differ.

Original: gpu4pyscf/pop/esp.py, Copyright 2021-2024 The PySCF Developers,
Apache License 2.0. Reference cited there: https://doi.org/10.1002/qua.26035
"""

from __future__ import annotations

import numpy as np
from pyscf import gto
from pyscf.data import radii
from scipy.spatial import distance_matrix

# Van der Waals radii (in angstrom) are taken from GAMESS (Merz-Kollman set).
# Same table as the GPU code: it stops at Cl (Z = 17).
R_VDW = (
    1.0
    / radii.BOHR
    * np.asarray(
        [
            -1,
            1.20,  # H
            1.20,  # He
            1.37,  # Li
            1.45,  # Be
            1.45,  # B
            1.50,  # C
            1.50,  # N
            1.40,  # O
            1.35,  # F
            1.30,  # Ne
            1.57,  # Na
            1.36,  # Mg
            1.24,  # Al
            1.17,  # Si
            1.80,  # P
            1.75,  # S
            1.70,  # Cl
        ]
    )
)


def unit_surface(n):
    """Points on the unit sphere (fewer than n in general). Identical to the GPU code."""
    ux, uy, uz = [], [], []
    eps = 1e-10
    nequat = int(np.sqrt(np.pi * n))
    nvert = int(nequat / 2)
    for i in range(nvert + 1):
        fi = np.pi * i / nvert
        z = np.cos(fi)
        xy = np.sin(fi)
        nhor = int(nequat * xy + eps)
        if nhor < 1:
            nhor = 1
        fj = 2.0 * np.pi * np.arange(nhor) / nhor
        ux.append(np.cos(fj) * xy)
        uy.append(np.sin(fj) * xy)
        uz.append(z * np.ones(nhor))
    ux, uy, uz = np.concatenate(ux), np.concatenate(uy), np.concatenate(uz)
    return np.array([ux[:n], uy[:n], uz[:n]]).T


def vdw_surface(mol, scales=(1.0,), density=1.0 * radii.BOHR**2, rad=R_VDW):
    """Points on scaled vdW spheres, outside all other spheres (Bohr). Identical to GPU code."""
    coords = mol.atom_coords(unit="B")
    charges = [gto.charge(sym) for sym in mol.elements]
    atom_radii = rad[charges]
    surface_points = []
    for scale in scales:
        scaled_radii = atom_radii * scale
        for i, coord in enumerate(coords):
            r = scaled_radii[i]
            nd = int(density * 4.0 * np.pi * r**2)
            points = coord + r * unit_surface(nd)
            dist = distance_matrix(points, coords) + 1e-10
            included = np.all(dist >= scaled_radii, axis=1)
            surface_points.append(points[included])
    return np.concatenate(surface_points)


def electron_potential(mol, dm, points, blksize=256):
    """sum_uv D_uv (u | 1/|r-C| | v) at each point C (Bohr). CPU replacement for
    gpu4pyscf int1e_grids(mol, points, dm=dm). Points are processed in blocks to
    bound memory (blksize * nao^2 doubles)."""
    dm = np.asarray(dm)
    if dm.ndim == 3:  # unrestricted: total density
        dm = dm[0] + dm[1]
    out = np.empty(len(points))
    for p0 in range(0, len(points), blksize):
        blk = points[p0 : p0 + blksize]
        ints = mol.intor("int1e_grids", grids=blk)
        out[p0 : p0 + len(blk)] = np.einsum("pij,ij->p", ints, dm)
    return out


def build_ab(
    mol,
    dm,
    grid_density=1.0 * radii.BOHR**2,
    rad=R_VDW,
    sum_constraints=(),
    equal_constraints=(),
    return_points=False,
):
    """Normal equations of the ESP fit plus Lagrange constraints. Mirrors GPU build_ab."""
    natm = mol.natm
    nconstraints = 1 + len(sum_constraints) + sum(len(g) - 1 for g in equal_constraints)

    surface_points = vdw_surface(mol, scales=[1.4, 1.6, 1.8, 2.0], density=grid_density, rad=rad)
    charges = mol.atom_charges()

    dim = natm + nconstraints
    coords = mol.atom_coords(unit="B")
    r = distance_matrix(coords, surface_points)
    rinv = 1.0 / r
    A = np.zeros((dim, dim))
    A[:natm, :natm] = rinv.dot(rinv.T)

    v_grids_e = electron_potential(mol, dm, surface_points)
    v_grids_n = np.dot(charges, rinv)

    B = np.empty(dim)
    B[:natm] = rinv.dot(v_grids_n - v_grids_e)

    off = natm
    A[:natm, off] = 1.0  # total charge
    A[off, :natm] = 1.0
    B[off] = mol.charge
    off += 1
    for i, group in enumerate(sum_constraints):
        A[i + off, group[1]] = 1
        A[group[1], i + off] = 1
        B[i + off] = group[0]
    off += len(sum_constraints)
    for group in equal_constraints:
        i = group[0]
        for ic, j in enumerate(group[1:]):
            A[off + ic, i] = 1
            A[off + ic, j] = -1
            A[i, off + ic] = 1
            A[j, off + ic] = -1
            B[off + ic] = 0
        off += len(group) - 1
    if return_points:
        return A, B, surface_points, v_grids_n - v_grids_e
    return A, B


def esp_solve(mol, dm, grid_density=1.0 * radii.BOHR**2, rad=R_VDW):
    A, B = build_ab(mol, dm, grid_density=grid_density, rad=rad)
    return np.linalg.solve(A, B)[: mol.natm]


def resp_solve(
    mol,
    dm,
    grid_density=1.0 * radii.BOHR**2,
    rad=R_VDW,
    maxit=25,
    resp_a=5e-4,
    resp_b=0.1,
    hfree=True,
    tol=1e-5,
    sum_constraints=(),
    equal_constraints=(),
):
    """Single-stage RESP with hyperbolic restraint toward zero. Mirrors GPU resp_solve."""
    charges = np.asarray([gto.charge(sym) for sym in mol.elements])
    natm = mol.natm
    is_restraint = charges > 1
    is_restraint[charges == 1] = not hfree

    A0, B = build_ab(
        mol,
        dm,
        grid_density=grid_density,
        rad=rad,
        sum_constraints=sum_constraints,
        equal_constraints=equal_constraints,
    )
    q = np.linalg.solve(A0, B)[:natm]
    q_prev = q.copy()
    for _ in range(maxit):
        penalty = resp_a / (q**2 + resp_b**2) ** 0.5
        A = A0.copy()
        A[np.diag_indices(natm)] += is_restraint * penalty
        q = np.linalg.solve(A, B)[:natm]
        if np.linalg.norm(q_prev - q) < tol:
            break
        q_prev = q
    return q

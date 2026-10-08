"""CPU-safe oscillator strength extraction for (possibly GPU-backed) TD objects.

gpu4pyscf TDDFT stores ``td.e`` / ``td.xy`` (and SCF MOs) as CuPy arrays.
Calling ``td.oscillator_strength()`` then routes through
``gpu4pyscf.tdscf.rhf._contract_multipole`` → ``pyscf.lib.einsum``, which
fails on NumPy ≥2.4 (``ValueError: not enough values to unpack (expected 4,
got 3)``) and is fragile with mixed CuPy/NumPy operands.

This module converts energies, amplitudes, and SCF MO arrays to NumPy and
evaluates length-gauge oscillator strengths with ``numpy.einsum`` only
(never ``lib.einsum`` / GPU ``_contract_multipole``).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def as_numpy(arr: Any) -> Any:
    """Convert a CuPy (or array-like) object to a NumPy array; leave scalars/None."""
    if arr is None:
        return None
    if isinstance(arr, (int, float, complex, np.generic)):
        return arr
    if isinstance(arr, (tuple, list)):
        return type(arr)(as_numpy(x) for x in arr)
    if hasattr(arr, "get"):
        return np.asarray(arr.get())
    return np.asarray(arr)


def _xy_to_numpy(xy) -> list:
    """Convert TD X/Y amplitude pairs to NumPy (handles TDA ``y=0`` and UHF tuples)."""
    out = []
    for x, y in xy:
        x_np = as_numpy(x)
        if isinstance(y, (int, float)) or y is None:
            y_np = y
        else:
            y_np = as_numpy(y)
        out.append((x_np, y_np))
    return out


def _charge_center(mol) -> np.ndarray:
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    return np.einsum("z,zr->r", charges, coords) / charges.sum()


def _frozen_mask_r(td, nmo: int) -> np.ndarray:
    """Boolean MO mask for restricted TD; default = all orbitals active."""
    if hasattr(td, "get_frozen_mask"):
        try:
            mask = as_numpy(td.get_frozen_mask())
            if isinstance(mask, np.ndarray) and mask.dtype == bool and mask.size == nmo:
                return mask
        except Exception:
            pass
    return np.ones(nmo, dtype=bool)


def _transition_dipole_rks(mol, mo_coeff, mo_occ, xy, singlet: bool, mask: np.ndarray) -> np.ndarray:
    """Length-gauge transition dipoles for closed-shell RKS/RHF TD (nstates, 3)."""
    nstates = len(xy)
    if not singlet:
        return np.zeros((nstates, 3))

    mo_coeff = mo_coeff[:, mask]
    mo_occ = mo_occ[mask]
    orbo = mo_coeff[:, mo_occ == 2]
    orbv = mo_coeff[:, mo_occ == 0]

    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor_symmetric("int1e_r", comp=3)  # (3, nao, nao)

    # AO → OV: use numpy.einsum (not pyscf.lib.einsum) for NumPy 2.4 safety.
    ints_ov = np.einsum("xpq,pi,qj->xij", ints, orbo, orbv.conj(), optimize=True)

    pol = np.empty((nstates, 3), dtype=float)
    for s, (x, y) in enumerate(xy):
        mu = np.einsum("xij,ij->x", ints_ov, x) * 2.0
        if isinstance(y, np.ndarray):
            mu = mu + np.einsum("xij,ij->x", ints_ov, y) * 2.0
        pol[s] = mu.real
    return pol


def _transition_dipole_uks(mol, mo_coeff, mo_occ, xy, mask) -> np.ndarray:
    """Length-gauge transition dipoles for UKS/UHF TD (nstates, 3)."""
    if isinstance(mask, tuple):
        maska, maskb = as_numpy(mask[0]), as_numpy(mask[1])
    else:
        # Fall back: no freeze
        maska = np.ones(mo_occ[0].size, dtype=bool)
        maskb = np.ones(mo_occ[1].size, dtype=bool)

    ca = mo_coeff[0][:, maska]
    cb = mo_coeff[1][:, maskb]
    oa = mo_occ[0][maska]
    ob = mo_occ[1][maskb]
    orbo_a = ca[:, oa == 1]
    orbv_a = ca[:, oa == 0]
    orbo_b = cb[:, ob == 1]
    orbv_b = cb[:, ob == 0]

    with mol.with_common_orig(_charge_center(mol)):
        ints = mol.intor_symmetric("int1e_r", comp=3)

    ints_a = np.einsum("xpq,pi,qj->xij", ints, orbo_a, orbv_a.conj(), optimize=True)
    ints_b = np.einsum("xpq,pi,qj->xij", ints, orbo_b, orbv_b.conj(), optimize=True)

    nstates = len(xy)
    pol = np.empty((nstates, 3), dtype=float)
    for s, (x, y) in enumerate(xy):
        mu = np.einsum("xij,ij->x", ints_a, x[0]) + np.einsum("xij,ij->x", ints_b, x[1])
        if isinstance(y, (tuple, list)) and isinstance(y[0], np.ndarray):
            mu = mu + np.einsum("xij,ij->x", ints_a, y[0]) + np.einsum("xij,ij->x", ints_b, y[1])
        pol[s] = mu.real
    return pol


def oscillator_strength_cpu(td, gauge: str = "length") -> np.ndarray:
    """Return length-gauge oscillator strengths as a 1-D NumPy float array.

    Converts ``td.e``, ``td.xy``, and SCF ``mo_coeff`` / ``mo_occ`` to NumPy,
    then evaluates ``f_s = (2/3) * e_s * |μ_s|^2`` on CPU. Does not mutate
    the TD/SCF objects permanently.
    """
    if gauge != "length":
        raise NotImplementedError("Only length-gauge oscillator strengths are supported")
    if td is None:
        raise ValueError("td object is None")
    if getattr(td, "e", None) is None or getattr(td, "xy", None) is None:
        raise ValueError("td object missing e/xy; run td.kernel() first")

    e = np.asarray(as_numpy(td.e), dtype=float).ravel()
    xy = _xy_to_numpy(td.xy)
    mf = td._scf
    mo_coeff = as_numpy(mf.mo_coeff)
    mo_occ = as_numpy(mf.mo_occ)
    singlet = bool(getattr(td, "singlet", True))

    # Restricted: ndarray; unrestricted: tuple/list of α/β arrays (as_numpy preserves).
    if isinstance(mo_coeff, (tuple, list)):
        mo_coeff = tuple(np.asarray(c) for c in mo_coeff)
        mo_occ = tuple(np.asarray(o).ravel() for o in mo_occ)
        mask = td.get_frozen_mask() if hasattr(td, "get_frozen_mask") else None
        trans_dip = _transition_dipole_uks(td.mol, mo_coeff, mo_occ, xy, mask)
    else:
        mo_coeff = np.asarray(mo_coeff)
        mo_occ = np.asarray(mo_occ).ravel()
        mask = _frozen_mask_r(td, mo_occ.size)
        trans_dip = _transition_dipole_rks(td.mol, mo_coeff, mo_occ, xy, singlet, mask)

    # f_s = (2/3) * E_s * |μ_s|^2
    f = (2.0 / 3.0) * np.einsum("s,sx,sx->s", e, trans_dip, trans_dip, optimize=True)
    return np.asarray(f, dtype=float).ravel()

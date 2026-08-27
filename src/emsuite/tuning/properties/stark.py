"""Stark effect under probe-induced electric fields."""

from __future__ import annotations

import numpy as np

from emsuite.core import find_homo_lumo_and_gap

from ..constants import HARTREE_TO_EV


def _field_from_probe(
    probe_coord: np.ndarray, probe_charge: float, origin: np.ndarray
) -> np.ndarray:
    """Approximate electric field (atomic units) from a point charge at probe_coord."""
    vec = origin - probe_coord
    dist = float(np.linalg.norm(vec))
    if dist < 1e-6:
        return np.zeros(3)
    # E = q * r / |r|^3 in a.u.
    return probe_charge * vec / dist**3


def stark_orbital_shifts(mf, field_vector: np.ndarray) -> dict[str, float]:
    """
    Estimate Stark shifts of HOMO/LUMO/gap under a uniform electric field.

    Uses a dipole coupling estimate: Δε ≈ -μ·E (first-order).
    """
    if mf is None:
        return {}
    dip = np.asarray(mf.dip_moment(), dtype=float)
    shift_hartree = -float(np.dot(dip, field_vector))
    shift_ev = shift_hartree * HARTREE_TO_EV
    homo, lumo, gap = [x * HARTREE_TO_EV for x in find_homo_lumo_and_gap(mf)]
    return {
        "stark_homo": homo + shift_ev,
        "stark_lumo": lumo + shift_ev,
        "stark_gap": gap,
    }


def compute_stark_properties(
    mf,
    probe_coord: np.ndarray | None,
    probe_charge: float = 0.0,
    props_to_calc: list[str] | None = None,
) -> dict[str, float]:
    if probe_coord is None or not props_to_calc:
        return {}
    origin = np.asarray(mf.mol.atom_coords()).mean(axis=0)
    field = _field_from_probe(np.asarray(probe_coord), probe_charge, origin)
    shifts = stark_orbital_shifts(mf, field)
    return {k: v for k, v in shifts.items() if k in props_to_calc}

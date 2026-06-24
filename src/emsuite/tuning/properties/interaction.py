"""Interaction energies and explicit water probes."""

from __future__ import annotations

import numpy as np

from ..constants import HARTREE_TO_KCAL

# TIP3P-like partial charges for a water probe (relative geometry in Å)
_WATER_TEMPLATE = np.array(
    [
        (0.0, 0.0, 0.0, -0.834),  # O
        (0.757, 0.586, 0.0, 0.417),  # H
        (-0.757, 0.586, 0.0, 0.417),  # H
    ]
)


def water_probe_coords_and_charges(anchor: np.ndarray, direction: np.ndarray | None = None):
    """Place a water molecule near anchor; direction sets O-H bisector."""
    anchor = np.asarray(anchor, dtype=float)
    if direction is None or np.linalg.norm(direction) < 1e-6:
        direction = np.array([0.0, 0.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    # Simple rotation: align template z with direction
    coords = []
    charges = []
    for x, y, z, q in _WATER_TEMPLATE:
        local = np.array([x, y, z])
        coords.append(anchor + local + direction * 2.8)  # ~H-bond distance
        charges.append(q)
    return np.asarray(coords), np.asarray(charges)


def interaction_energy_kcal(mf_alone, mf_complex) -> float:
    if mf_alone is None or mf_complex is None:
        return 0.0
    return (mf_complex.e_tot - mf_alone.e_tot) * HARTREE_TO_KCAL


def proton_affinity_kcal(neutral_mf, protonated_mf) -> float:
    """Gas-phase PA ≈ E(H+) + E(A-) - E(HA); use cation as proxy for protonated species."""
    if neutral_mf is None or protonated_mf is None:
        return 0.0
    return (protonated_mf.e_tot - neutral_mf.e_tot) * HARTREE_TO_KCAL

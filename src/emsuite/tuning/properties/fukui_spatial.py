"""Condensed Fukui indices and surface projection."""

from __future__ import annotations

import numpy as np


def mulliken_charges(mf) -> np.ndarray:
    """Return Mulliken atomic charges (one scalar per atom)."""
    if mf is None:
        return np.array([])
    pop = mf.mulliken_pop()
    if isinstance(pop, tuple):
        # PySCF returns (ao_populations, atomic_charges)
        return np.asarray(pop[1], dtype=float)
    return np.asarray(pop, dtype=float)


def condensed_fukui_indices(
    neutral_mf,
    anion_mf,
    cation_mf,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Atom-resolved Fukui indices from finite differences on Mulliken charges.

    f- (electrophilic) = q(N) - q(N+1)  [neutral - cation]
    f+ (nucleophilic) = q(N-1) - q(N)  [anion - neutral]
    """
    q0 = mulliken_charges(neutral_mf)
    qm = mulliken_charges(anion_mf) if anion_mf is not None else q0
    qp = mulliken_charges(cation_mf) if cation_mf is not None else q0
    f_plus = qm - q0
    f_minus = q0 - qp
    return f_plus, f_minus


def atom_coords_from_mf(mf) -> np.ndarray:
    return np.asarray(mf.mol.atom_coords(), dtype=float)


def project_atom_property_to_point(
    atom_coords: np.ndarray,
    atom_values: np.ndarray,
    point: np.ndarray,
    method: str = "nearest",
) -> float:
    if atom_coords.size == 0:
        return 0.0
    if method == "nearest":
        idx = int(np.argmin(np.linalg.norm(atom_coords - point, axis=1)))
        return float(atom_values[idx])
    distances = np.linalg.norm(atom_coords - point, axis=1)
    weights = 1.0 / np.maximum(distances, 1e-3) ** 2
    weights /= weights.sum()
    return float(np.dot(weights, atom_values))


def build_surface_fukui_maps(
    neutral_mf,
    anion_mf,
    cation_mf,
    surface_coords: np.ndarray,
    projection: str = "nearest",
) -> dict[str, np.ndarray]:
    f_plus, f_minus = condensed_fukui_indices(neutral_mf, anion_mf, cation_mf)
    atoms = atom_coords_from_mf(neutral_mf)
    maps: dict[str, np.ndarray] = {}
    for prop, values in (("fukui_spa_plus", f_plus), ("fukui_spa_minus", f_minus)):
        maps[prop] = np.array(
            [project_atom_property_to_point(atoms, values, pt, projection) for pt in surface_coords]
        )
    return maps

"""Bond-axis electrostatic scan utilities."""

from __future__ import annotations

import numpy as np


def bond_scan_coords(
    atom_coords: np.ndarray,
    atom_a: int,
    atom_b: int,
    n_steps: int = 10,
    span_angstrom: float = 3.0,
) -> np.ndarray:
    """
    Generate probe coordinates along the line between two atoms.

    Points are centered on the bond midpoint and span ``span_angstrom`` total.
    """
    if atom_a < 0 or atom_b < 0 or atom_a >= len(atom_coords) or atom_b >= len(atom_coords):
        raise ValueError(f"Invalid bond atom indices: {atom_a}, {atom_b}")
    a = np.asarray(atom_coords[atom_a], dtype=float)
    b = np.asarray(atom_coords[atom_b], dtype=float)
    midpoint = 0.5 * (a + b)
    direction = b - a
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = np.array([0.0, 0.0, 1.0])
    else:
        direction = direction / norm
    if n_steps < 1:
        n_steps = 1
    offsets = np.linspace(-span_angstrom / 2, span_angstrom / 2, n_steps)
    return np.array([midpoint + direction * off for off in offsets])

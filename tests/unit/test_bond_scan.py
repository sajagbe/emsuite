"""Bond scan coordinate generation."""

import numpy as np

from emsuite.surface.bond_scan import bond_scan_coords


def test_bond_scan_produces_n_steps():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points = bond_scan_coords(coords, 0, 1, n_steps=5, span_angstrom=2.0)
    assert points.shape == (5, 3)
    assert np.allclose(points[2], [1.0, 0.0, 0.0])

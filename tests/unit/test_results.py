"""PotentialResult / SurfaceResult .surf round-trip."""

from pathlib import Path

import numpy as np

from emsuite.results import PotentialResult
from emsuite.surface.io import load_surf


def test_potential_result_to_surf_roundtrip(tmp_path: Path):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.array([0.1, -0.2])
    result = PotentialResult(coords=coords, values=values, quantity="potential")
    path = tmp_path / "map.surf"
    written = result.to_surf(path)
    loaded_coords, loaded_values = load_surf(written)
    np.testing.assert_allclose(loaded_coords, coords)
    np.testing.assert_allclose(loaded_values, values)
    again = PotentialResult.from_surf(written, quantity="potential")
    assert again.quantity == "potential"
    np.testing.assert_allclose(again.values, values)

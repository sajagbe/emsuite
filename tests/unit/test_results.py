"""PotentialResult / SurfaceResult .surf round-trip."""

from pathlib import Path

import numpy as np

from emsuite.results import PotentialResult, SurfaceResult
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


def test_potential_result_to_mol2(tmp_path: Path):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.array([0.1, -0.2])
    result = PotentialResult(coords=coords, values=values, quantity="charge")

    written = result.to_mol2(tmp_path / "potential.mol2")
    lines = Path(written).read_text().splitlines()
    assert lines[0] == "@<TRIPOS>MOLECULE"
    assert "0.100000" in lines[6]
    assert "-0.200000" in lines[7]

    result_with_path = PotentialResult(
        coords=coords, values=values, quantity="charge", path=str(tmp_path / "potential.surf")
    )
    default_written = result_with_path.to_mol2()
    assert default_written == str(tmp_path / "potential.mol2")


def test_surface_result_to_xyz(tmp_path: Path):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.array([0.1, 0.2])
    result = SurfaceResult(coords=coords, values=values)

    written = result.to_xyz(tmp_path / "surface.xyz")
    lines = Path(written).read_text().splitlines()
    assert lines[0] == "2"
    assert lines[2] == "H 0.000000 0.000000 0.000000"
    assert lines[3] == "H 1.000000 0.000000 0.000000"

    result_with_path = SurfaceResult(coords=coords, values=values, path=str(tmp_path / "surface.surf"))
    default_written = result_with_path.to_xyz()
    assert default_written == str(tmp_path / "surface.xyz")


def test_surface_result_to_mol2(tmp_path: Path):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.array([0.1, -0.2])
    result = SurfaceResult(coords=coords, values=values)

    written = result.to_mol2(tmp_path / "surface.mol2")
    lines = Path(written).read_text().splitlines()
    assert lines[0] == "@<TRIPOS>MOLECULE"
    assert lines[2] == "    2 0 0 0"
    assert lines[5] == "@<TRIPOS>ATOM"
    assert "0.100000" in lines[6]
    assert "-0.200000" in lines[7]

    result_with_path = SurfaceResult(coords=coords, values=values, path=str(tmp_path / "surface.surf"))
    default_written = result_with_path.to_mol2()
    assert default_written == str(tmp_path / "surface.mol2")

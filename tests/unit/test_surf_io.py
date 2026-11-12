"""Unit tests for surf file I/O."""

import numpy as np
import pytest

from emsuite.surface import load_surf, save_surf


def test_save_and_load_surf_roundtrip(tmp_path):
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    charges = 0.1
    surf_path = tmp_path / "test.surf"
    save_surf(coords, charges, str(surf_path))
    loaded_coords, loaded_charges = load_surf(str(surf_path))
    np.testing.assert_allclose(coords, loaded_coords)
    np.testing.assert_allclose([0.1, 0.1], loaded_charges)


def test_load_surf_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_surf(str(tmp_path / "missing.surf"))


def test_load_surf_invalid_columns(tmp_path):
    bad = tmp_path / "bad.surf"
    bad.write_text("x y z\n1 2 3\n")
    with pytest.raises(ValueError, match="4 columns"):
        load_surf(str(bad))


@pytest.mark.parametrize("row", ["nan 0 0 .1", "0 0 0 inf"])
def test_load_rejects_nonfinite_surface(tmp_path, row):
    path = tmp_path / "bad.surf"
    path.write_text("x y z q\n" + row + "\n")
    with pytest.raises(ValueError, match="finite"):
        load_surf(path)


@pytest.mark.parametrize(
    "coords,charges",
    [(np.empty((0, 3)), 0.1), (np.ones((2, 3)), [0.1]), (np.ones((1, 3)), float("nan"))],
)
def test_invalid_save_preserves_existing_file(tmp_path, coords, charges):
    path = tmp_path / "existing.surf"
    path.write_text("preserve me")
    with pytest.raises(ValueError):
        save_surf(coords, charges, path)
    assert path.read_text() == "preserve me"

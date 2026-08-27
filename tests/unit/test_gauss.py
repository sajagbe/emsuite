"""Gauss-law operator, DX parse, and potential config rules."""

from pathlib import Path

import numpy as np
import pytest

from emsuite.config import ConfigValidationError, validate_potential_params
from emsuite.potential.dx import DxGrid, parse_dx
from emsuite.potential.gauss import (
    charges_at_points,
    div_eps_grad,
    interpolate_scalar,
    potential_at_points,
)


def _write_dx(path: Path, data: np.ndarray, origin=(0.0, 0.0, 0.0), h=1.0) -> Path:
    nx, ny, nz = data.shape
    values = " ".join(f"{v:.6f}" for v in data.reshape(-1))
    path.write_text(
        "\n".join(
            [
                f"object 1 class gridpositions counts {nx} {ny} {nz}",
                f"origin {origin[0]} {origin[1]} {origin[2]}",
                f"delta {h} 0.0 0.0",
                f"delta 0.0 {h} 0.0",
                f"delta 0.0 0.0 {h}",
                f"object 3 class array type double rank 0 items {data.size} data follows",
                values,
            ]
        )
        + "\n"
    )
    return path


def test_parse_dx_roundtrip(tmp_path: Path):
    data = np.arange(27, dtype=float).reshape(3, 3, 3)
    path = _write_dx(tmp_path / "g.dx", data, origin=(1.0, 2.0, 3.0), h=0.5)
    grid = parse_dx(path)
    assert grid.shape == (3, 3, 3)
    assert grid.origin == (1.0, 2.0, 3.0)
    assert grid.spacing == (0.5, 0.5, 0.5)
    assert np.allclose(grid.data, data)


def test_div_eps_grad_zero_for_linear_potential():
    n = 5
    phi = np.zeros((n, n, n))
    for i in range(n):
        phi[i, :, :] = 3.0 * i
    eps = np.ones_like(phi)
    div = div_eps_grad(phi, eps, eps, eps, h=1.0)
    assert np.allclose(div[1:-1, 1:-1, 1:-1], 0.0, atol=1e-12)


def test_div_eps_grad_quadratic_matches_laplacian():
    n = 5
    phi = np.zeros((n, n, n))
    for i in range(n):
        phi[i, :, :] = float(i) ** 2
    eps = np.ones_like(phi)
    div = div_eps_grad(phi, eps, eps, eps, h=1.0)
    assert np.allclose(div[1:-1, 1:-1, 1:-1], 2.0, atol=1e-12)


def test_interpolate_scalar_midpoint():
    grid = np.zeros((2, 2, 2))
    grid[0, 0, 0] = 0.0
    grid[1, 0, 0] = 2.0
    grid[0, 1, 0] = 0.0
    grid[1, 1, 0] = 2.0
    grid[:, :, 1] = grid[:, :, 0]
    value = interpolate_scalar(grid, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), np.array([[0.5, 0.0, 0.0]]))
    assert value.shape == (1,)
    assert value[0] == pytest.approx(1.0)


def test_constant_potential_gives_near_zero_charge():
    n = 5
    phi = np.full((n, n, n), 1.5)
    eps = np.ones((n, n, n))
    pot = DxGrid(phi, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    diel = DxGrid(eps, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    q = charges_at_points(pot, diel, diel, diel, np.array([[2.0, 2.0, 2.0]]))
    assert q[0] == pytest.approx(0.0, abs=1e-18)


def test_potential_at_points_reads_grid_value():
    data = np.zeros((3, 3, 3))
    data[1, 1, 1] = 4.2
    pot = DxGrid(data, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert potential_at_points(pot, np.array([[1.0, 1.0, 1.0]]))[0] == pytest.approx(4.2)


def test_coulomb_method_rejected():
    with pytest.raises(ConfigValidationError, match="coulomb"):
        validate_potential_params({"molecule": "m.xyz", "method": "coulomb"})


def test_future_esp_method_rejected():
    with pytest.raises(ConfigValidationError, match="not implemented yet"):
        validate_potential_params({"molecule": "m.xyz", "method": "esp"})


def test_defaults_apbs_potential():
    params = validate_potential_params({"molecule": "m.xyz"})
    assert params["method"] == "apbs"
    assert params["quantity"] == "potential"

"""Gauss-law operator and DX parse."""

from pathlib import Path

import numpy as np
import pytest

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


def test_parse_dx_multiline_data_and_footer(tmp_path: Path):
    path = _write_dx(tmp_path / "g.dx", np.arange(8.0).reshape(2, 2, 2))
    text = path.read_text().replace(
        "0.000000 1.000000 2.000000 3.000000 4.000000 5.000000 6.000000 7.000000",
        "0e0 1e0 2e0\n3e0 4e0 5e0\n6e0 7e0",
    )
    path.write_text(text + 'attribute "dep" string "positions"\nobject "field" class field\n')
    np.testing.assert_array_equal(parse_dx(path).data.ravel(), np.arange(8.0))


def test_parse_dx_rejects_truncated_data(tmp_path: Path):
    path = _write_dx(tmp_path / "short.dx", np.zeros((2, 2, 2)))
    lines = path.read_text().splitlines()
    lines[-1] = "1 2 3"
    path.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="3 values, expected 8"):
        parse_dx(path)


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

"""Gauss-law conversion of APBS potential + dielectric grids to charge.

Discrete form of ρ = −ε₀ ∇·(ε ∇φ), using APBS face-centered dielectric
maps (dielx/dely/dielz) rather than a cell-centered ε and ∇ε stencil.

Charges are reported in elementary charge units (e).
"""

from __future__ import annotations

import numpy as np

from .dx import DxGrid

EPSILON_0 = 8.854e-12  # C / (V m)
KT_E_TO_V = 0.02568  # 1 kT/e in volts at 298 K
A2_TO_M2 = 1e-20
A3_TO_M3 = 1e-30
ELEMENTARY_CHARGE = 1.602e-19  # C


def _require_interior(shape: tuple[int, int, int]) -> None:
    if min(shape) < 3:
        raise ValueError("Gauss-law stencil needs at least a 3×3×3 grid")


def div_eps_grad(
    phi: np.ndarray,
    dielx: np.ndarray,
    diely: np.ndarray,
    dielz: np.ndarray,
    h: float,
) -> np.ndarray:
    """Return ∇·(ε ∇φ) on interior points; edges are 0.

    ``phi`` must already be in volts. ``h`` is the cubic grid spacing in metres.
    """
    _require_interior(phi.shape)
    out = np.zeros_like(phi, dtype=float)
    flux_x = dielx[1:-1, 1:-1, 1:-1] * (phi[2:, 1:-1, 1:-1] - phi[1:-1, 1:-1, 1:-1]) - dielx[
        :-2, 1:-1, 1:-1
    ] * (phi[1:-1, 1:-1, 1:-1] - phi[:-2, 1:-1, 1:-1])
    flux_y = diely[1:-1, 1:-1, 1:-1] * (phi[1:-1, 2:, 1:-1] - phi[1:-1, 1:-1, 1:-1]) - diely[
        1:-1, :-2, 1:-1
    ] * (phi[1:-1, 1:-1, 1:-1] - phi[1:-1, :-2, 1:-1])
    flux_z = dielz[1:-1, 1:-1, 1:-1] * (phi[1:-1, 1:-1, 2:] - phi[1:-1, 1:-1, 1:-1]) - dielz[
        1:-1, 1:-1, :-2
    ] * (phi[1:-1, 1:-1, 1:-1] - phi[1:-1, 1:-1, :-2])
    out[1:-1, 1:-1, 1:-1] = (flux_x + flux_y + flux_z) / (h * h)
    return out


def charge_density_e_per_a3(
    potential_kte: np.ndarray,
    dielx: np.ndarray,
    diely: np.ndarray,
    dielz: np.ndarray,
    spacing_angstrom: float,
) -> np.ndarray:
    """Charge density (e / Å³) from APBS φ (kT/e) and dielectric maps."""
    h_m = spacing_angstrom * 1e-10
    phi_v = potential_kte * KT_E_TO_V
    div = div_eps_grad(phi_v, dielx, diely, dielz, h_m)
    rho_si = -EPSILON_0 * div  # C / m³
    return rho_si * A3_TO_M3 / ELEMENTARY_CHARGE


def interpolate_scalar(
    grid: np.ndarray,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    points: np.ndarray,
) -> np.ndarray:
    """Trilinear interpolation of a scalar grid at arbitrary points (Å)."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    nx, ny, nz = grid.shape
    ox, oy, oz = origin
    hx, hy, hz = spacing
    out = np.empty(points.shape[0], dtype=float)

    for n, (x, y, z) in enumerate(points):
        i = (x - ox) / hx
        j = (y - oy) / hy
        k = (z - oz) / hz
        i0 = int(np.floor(i))
        j0 = int(np.floor(j))
        k0 = int(np.floor(k))
        i0 = min(max(i0, 0), nx - 2)
        j0 = min(max(j0, 0), ny - 2)
        k0 = min(max(k0, 0), nz - 2)
        xd = np.clip(i - i0, 0.0, 1.0)
        yd = np.clip(j - j0, 0.0, 1.0)
        zd = np.clip(k - k0, 0.0, 1.0)
        c000 = grid[i0, j0, k0]
        c100 = grid[i0 + 1, j0, k0]
        c010 = grid[i0, j0 + 1, k0]
        c110 = grid[i0 + 1, j0 + 1, k0]
        c001 = grid[i0, j0, k0 + 1]
        c101 = grid[i0 + 1, j0, k0 + 1]
        c011 = grid[i0, j0 + 1, k0 + 1]
        c111 = grid[i0 + 1, j0 + 1, k0 + 1]
        out[n] = (
            c000 * (1 - xd) * (1 - yd) * (1 - zd)
            + c100 * xd * (1 - yd) * (1 - zd)
            + c010 * (1 - xd) * yd * (1 - zd)
            + c110 * xd * yd * (1 - zd)
            + c001 * (1 - xd) * (1 - yd) * zd
            + c101 * xd * (1 - yd) * zd
            + c011 * (1 - xd) * yd * zd
            + c111 * xd * yd * zd
        )
    return out


def potential_at_points(pot: DxGrid, points: np.ndarray) -> np.ndarray:
    """Interpolate APBS potential (kT/e) onto surface coordinates."""
    return interpolate_scalar(pot.data, pot.origin, pot.spacing, points)


def charges_at_points(
    pot: DxGrid,
    dielx: DxGrid,
    diely: DxGrid,
    dielz: DxGrid,
    points: np.ndarray,
) -> np.ndarray:
    """Gauss-law voxel charge interpolated onto surface coordinates (units: e)."""
    if pot.shape != dielx.shape or pot.shape != diely.shape or pot.shape != dielz.shape:
        raise ValueError("Potential and dielectric grids must share the same shape")
    h = float(np.mean(pot.spacing))
    rho = charge_density_e_per_a3(pot.data, dielx.data, diely.data, dielz.data, h)
    voxel_charge = rho * (h**3)
    return interpolate_scalar(voxel_charge, pot.origin, pot.spacing, points)

"""Electrostatic potential maps on surfaces."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from emsuite.surface import load_surf, save_surf

from .apbs import run_apbs_grids
from .gauss import charges_at_points, potential_at_points


def _apbs_values(
    coords: np.ndarray,
    atoms,
    charges,
    box_coords: np.ndarray,
    pdie: float,
    sdie: float,
    quantity: str,
    pqr_path: str | None = None,
    grid_spacing: float | None = None,
) -> np.ndarray:
    if pqr_path is not None:
        grids = run_apbs_grids(
            pqr_path=pqr_path,
            box_coords=box_coords,
            pdie=pdie,
            sdie=sdie,
            grid_spacing=grid_spacing,
            include_dielectric=quantity == "charge",
        )
    else:
        grids = run_apbs_grids(
            atoms=atoms,
            charges=charges,
            box_coords=box_coords,
            pdie=pdie,
            sdie=sdie,
            grid_spacing=grid_spacing,
            include_dielectric=quantity == "charge",
        )
    from time import perf_counter

    timings = grids.timings
    if quantity == "charge":
        if grids.dielx is None or grids.diely is None or grids.dielz is None:
            raise RuntimeError("Gauss-law charges require all three APBS dielectric maps")
        values = charges_at_points(
            grids.potential, grids.dielx, grids.diely, grids.dielz, coords, timings=timings
        )
        print("Computed Gauss-law surface charges from APBS potential and dielectric maps")
    else:
        start = perf_counter()
        values = potential_at_points(grids.potential, coords)
        timings["surface_interpolation_s"] = perf_counter() - start
        print("Interpolated APBS potentials at surface points")
    print(
        "Timings (seconds): "
        + ", ".join(f"{name}={seconds:.6f}" for name, seconds in timings.items())
    )
    return values


def _check_surface_frame(points: np.ndarray, atoms: np.ndarray, limit: float = 8.0) -> None:
    """Stop when the surface is not in the coordinate frame of the PDB."""
    from scipy.spatial import cKDTree

    nearest, _ = cKDTree(atoms).query(points)
    if np.median(nearest) > limit:
        raise ValueError(
            f"the surface points are far from every PDB atom (median distance "
            f"{np.median(nearest):.1f} Å): the .surf and the PDB are not in the same frame"
        )


def run_potential_spec(spec, resp=None) -> dict[str, str]:
    """PDB + .surf + input settings -> one heterogeneous .surf (and .mol2) of the chosen quantity.

    Also keeps the PQR, a per-point CSV and a table of where each residue's
    parameters came from. Returns the paths written.
    """
    from emsuite.surface.io import save_mol2

    from .table import build_parameter_table

    print("\n" + "=" * 60)
    print("              Electrostatic Potential Module")
    print("=" * 60 + "\n")
    spec.check_files()
    coords, _ = load_surf(spec.surface_file)
    print(f"Surface: {spec.surface_file} ({len(coords)} points)")

    table = build_parameter_table(spec, resp=resp)
    _check_surface_frame(coords, table.structure.coords)
    print(f"Parameters for {table.structure.n_atoms} atoms of {spec.pdb}:")
    print(table.summary())

    out = Path(spec.output_surf)
    paths = {
        "surf": str(out),
        "mol2": str(out.with_suffix(".mol2")),
        "csv": str(out.with_suffix(".csv")),
        "pqr": str(out.with_suffix(".pqr")),
        "sources": str(out.with_name(out.stem + "_sources.csv")),
    }
    table.write_pqr(paths["pqr"])
    table.write_sources(paths["sources"])

    values = _apbs_values(
        coords,
        None,
        None,
        np.vstack([table.structure.coords, coords]),
        pdie=spec.pdie,
        sdie=spec.sdie,
        quantity=spec.quantity,
        pqr_path=str(Path(paths["pqr"]).resolve()),
        grid_spacing=spec.grid_spacing,
    )
    save_surf(coords, values, paths["surf"], heterogenous=True, quantity=spec.quantity)
    save_mol2(coords, values, paths["mol2"])
    header = "charge_e" if spec.quantity == "charge" else "potential"
    np.savetxt(
        paths["csv"],
        np.column_stack([np.arange(len(coords)), coords, values]),
        header=f"point_index,x,y,z,{header}",
        comments="",
        delimiter=",",
        fmt=["%d", "%.6f", "%.6f", "%.6f", "%.8f"],
    )
    for kind, path in paths.items():
        print(f"  {kind:8s} {path}")
    print("=" * 60)
    return paths

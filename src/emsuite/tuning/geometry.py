"""Geometric consistency check for an XYZ and a VDW surface."""

from pathlib import Path

import numpy as np
from vdw_surfgen.cli import VDW_RADII


def check_surface_frame(xyz_file, surface_coords, tolerance=1e-3):
    """Report scale consistency; warn and continue when geometry does not match.

    On a union of scaled VDW spheres, min_j(distance / radius_j) is the
    same scale at every surface point. This is a consistency check, not proof
    of provenance. Memory usage is proportional to the number of points.
    """
    lines = Path(xyz_file).read_text().splitlines()
    atoms = [line.split() for line in lines[2 : 2 + int(lines[0])]]
    points = np.asarray(surface_coords, dtype=float)
    if not atoms or points.size == 0:
        print(
            "WARNING: Cannot check XYZ/surface alignment: no atoms or surface points. Continuing."
        )
        return None
    scales = np.full(len(points), np.inf)
    for atom in atoms:
        center = np.asarray(atom[1:4], dtype=float)
        radius = VDW_RADII.get(atom[0], 1.5)
        scales = np.minimum(scales, np.linalg.norm(points - center, axis=1) / radius)

    median = float(np.median(scales))
    minimum, maximum = float(np.min(scales)), float(np.max(scales))
    valid = np.isfinite(scales).all() and minimum > 0
    spread = (maximum - minimum) / median if valid else float("inf")
    # Half the allowed full range on each side of the median defines off points.
    off = int(
        np.count_nonzero(
            ~np.isfinite(scales)
            | (scales <= 0)
            | (np.abs(scales - median) > tolerance * median / 2)
        )
    )
    if spread > tolerance:
        print(
            f"WARNING: XYZ/surface geometry is inconsistent for {xyz_file}: "
            f"median scale={median:.6f}, min={minimum:.6f}, max={maximum:.6f}; "
            f"{off}/{len(points)} points off (relative full-range tolerance {tolerance:g}). "
            "The XYZ may differ from the molecule used to make the surface, or may have "
            "been moved or rotated afterwards. Continuing the calculation."
        )
    else:
        print(f"XYZ/surface geometry is consistent; inferred surface scale={median:.6f}.")
    return median

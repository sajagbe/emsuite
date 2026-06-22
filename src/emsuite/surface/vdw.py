"""VDW surface point generation."""

import os
import subprocess

import numpy as np


def get_vdw_surface_coordinates(xyz_file, surface_density=1.0, surface_scale=1.0):
    """
    Generate van der Waals surface coordinates for a molecule.

    This function uses the external 'vsg' tool to generate points on the
    van der Waals surface of a molecule from its XYZ coordinates.

    Args:
        xyz_file (str): Path to the XYZ file containing molecular coordinates
        surface_density (float, optional): Surface point density. Defaults to 1.0.
        surface_scale (float, optional): Scaling factor for van der Waals radii. Defaults to 1.0.

    Returns:
        numpy.ndarray: Array of surface coordinates with shape [N, 3]

    Raises:
        RuntimeError: If the vsg command fails
        FileNotFoundError: If the expected surface file is not created

    Note:
        - Requires the 'vsg' external tool to be installed and in PATH
        - Automatically cleans up temporary surface files after reading
    """
    ret = subprocess.run(
        ["vsg", xyz_file, "-d", str(surface_density), "-s", str(surface_scale), "-t"],
        capture_output=True,
        text=True,
    )
    if ret.returncode != 0:
        raise RuntimeError(f"vsg failed: {ret.stderr}")

    base, _ = os.path.splitext(xyz_file)
    surface_file = f"{base}_vdw_surface.txt"

    if not os.path.isfile(surface_file):
        raise FileNotFoundError(f"Expected surface file not found: {surface_file}")

    coords = np.loadtxt(surface_file, dtype=float)
    if coords.ndim == 1 and coords.size == 3:
        coords = coords.reshape(1, 3)

    # Clean up temporary file
    try:
        os.remove(surface_file)
    except OSError as e:
        print(f"Warning: could not remove {surface_file}: {e}")

    # Also clean up the xyz surface file if created
    xyz_surface_file = f"{base}_vdw_surface.xyz"
    if os.path.exists(xyz_surface_file):
        try:
            os.remove(xyz_surface_file)
        except OSError:
            pass

    return coords


##############################################

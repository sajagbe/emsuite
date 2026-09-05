"""VDW surface point generation."""

import os
import subprocess

import numpy as np


def _vdw_surface_candidates(xyz_file: str) -> list[str]:
    """
    Paths where vsg may write ``{stem}_vdw_surface.txt``.

    ``vsg`` names outputs from the XYZ basename and writes them in the
    process CWD, not necessarily next to a relative XYZ path like
    ``../ligand.xyz``. Prefer CWD, then alongside the XYZ file.
    """
    stem = os.path.splitext(os.path.basename(xyz_file))[0]
    name = f"{stem}_vdw_surface.txt"
    xyz_dir = os.path.dirname(os.path.abspath(xyz_file))
    # Preserve historical relative join (../ligand -> ../ligand_vdw_surface.txt)
    rel_base, _ = os.path.splitext(xyz_file)
    candidates = [
        os.path.join(os.getcwd(), name),
        os.path.join(xyz_dir, name),
        f"{rel_base}_vdw_surface.txt",
    ]
    # Deduplicate while preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for path in candidates:
        key = os.path.normpath(path)
        if key not in seen:
            seen.add(key)
            ordered.append(path)
    return ordered


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

    surface_file = next((p for p in _vdw_surface_candidates(xyz_file) if os.path.isfile(p)), None)
    if surface_file is None:
        tried = ", ".join(_vdw_surface_candidates(xyz_file))
        raise FileNotFoundError(f"Expected surface file not found (tried: {tried})")

    coords = np.loadtxt(surface_file, dtype=float)
    if coords.ndim == 1 and coords.size == 3:
        coords = coords.reshape(1, 3)

    # Clean up temporary files (txt + companion xyz) at all candidate locations
    for txt_path in _vdw_surface_candidates(xyz_file):
        for path in (txt_path, f"{os.path.splitext(txt_path)[0]}.xyz"):
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"Warning: could not remove {path}: {e}")

    return coords

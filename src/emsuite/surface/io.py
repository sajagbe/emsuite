"""Surface file I/O."""

import os

import numpy as np


def load_surf(path):
    """
    Load surface coordinates and charges from a surf file.

    Args:
        path (str): Path to the surf file (must have 4 columns: x, y, z, q)

    Returns:
        tuple: (coords, charges) where:
            - coords: numpy array of shape [N, 3]
            - charges: numpy array of shape [N]

    Raises:
        FileNotFoundError: If the surf file does not exist
        ValueError: If the file does not have exactly 4 columns
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"surf file not found: {path}")

    data = np.loadtxt(path, skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] != 4:
        raise ValueError(
            f"surf file must have 4 columns (x, y, z, q), found {data.shape[1]} columns. "
            f"Please regenerate the surface using 'emsuite -s <surface.in>'"
        )

    coords = data[:, :3]
    charges = data[:, 3]

    return coords, charges


def save_surf(coords, charges, output_path, heterogenous=False):
    """
    Save surface coordinates and charges to a surf file.

    Always writes 4-column format (x, y, z, q).

    Args:
        coords (numpy.ndarray): Surface coordinates with shape [N, 3]
        charges (numpy.ndarray or float): Charges for each point. If float,
            the same charge is applied to all points (homogenous surface).
        output_path (str): Path to save the surf file
        heterogenous (bool): If True, adds a header comment instructing user
            to edit charges per-point. Defaults to False.
    """
    n_points = coords.shape[0]

    # Handle scalar charge (homogenous) vs array (heterogenous)
    if np.isscalar(charges):
        charges = np.full(n_points, charges)

    with open(output_path, "w") as f:
        if heterogenous:
            f.write("# x          y          z          q (EDIT CHARGES BELOW)\n")
        else:
            f.write("x          y          z          q\n")

        for i in range(n_points):
            f.write(
                f"{coords[i, 0]:<10.6f} {coords[i, 1]:<10.6f} {coords[i, 2]:<10.6f} {charges[i]:<10.6f}\n"
            )

    print(f"Surface saved to: {output_path} ({n_points} points)")


def save_mol2(coords, values, output_path, name="SURF"):
    """
    Save surface points as MOL2 pseudo-atoms (element H), with ``values`` in the
    charge column so any MOL2-reading viewer (PyMOL, Avogadro, VMD) can color by it.

    Args:
        coords (numpy.ndarray): Surface coordinates with shape [N, 3]
        values (numpy.ndarray): Per-point scalar (e.g. potential or charge), shape [N]
        output_path (str): Path to save the mol2 file
        name (str): Substructure name written per atom. Defaults to "SURF".
    """
    n_points = coords.shape[0]

    with open(output_path, "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{name}\n")
        f.write(f"{n_points:5d} 0 0 0\n")
        f.write("SMALL\n")
        f.write("GASTEIGER\n")

        f.write("@<TRIPOS>ATOM\n")
        for i in range(n_points):
            x, y, z = coords[i]
            f.write(
                f"{i + 1:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {name:8s} {values[i]:10.6f}\n"
            )

    print(f"Surface saved to: {output_path} ({n_points} points)")

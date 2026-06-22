"""Surface generation orchestration."""

import os

import numpy as np

from .io import save_surf
from .optimize import optimize_with_pyscf, smiles_to_xyz
from .vdw import get_vdw_surface_coordinates


def generate_surface(
    input_type,
    input_data,
    output_surf="surface.surf",
    surface_density=1.0,
    surface_scale=1.0,
    surface_type="homogenous",
    surface_charge=1.0,
    optimize=None,
    optimize_method="mmff",
    method="dft",
    basis_set="6-31G*",
    functional="b3lyp",
    solvent=None,
    charge=0,
    spin=0,
    optimized_xyz=None,
):
    """
    Generate a VDW surface from SMILES or XYZ input.

    Args:
        input_type (str): 'SMILES' or 'XYZ'
        input_data (str): SMILES string or path to XYZ file
        output_surf (str): Path to save the surf file
        surface_density (float): Surface point density
        surface_scale (float): Scaling factor for VDW radii
        surface_type (str): 'homogenous' or 'heterogenous'
        surface_charge (float): Charge value for homogenous surfaces
        optimize (bool or None): Whether to optimize geometry
        optimize_method (str): 'mmff', 'uff', or 'pyscf'
        method (str): QM method for pyscf optimization
        basis_set (str): Basis set for pyscf optimization
        functional (str): Functional for pyscf optimization
        solvent (str or None): Solvent for pyscf optimization
        charge (int): Molecular charge
        spin (int): Spin (2S notation)
        optimized_xyz (str or None): Custom path for optimized XYZ file

    Returns:
        str: Path to the generated surf file
    """
    input_type = input_type.upper()

    # Determine XYZ file path
    if input_type == "SMILES":
        # Generate XYZ from SMILES
        if optimized_xyz:
            xyz_path = optimized_xyz
        else:
            xyz_path = output_surf.replace(".surf", ".xyz").replace(".etm", ".xyz")

        # Default to optimizing SMILES input
        should_optimize = optimize if optimize is not None else True

        print(f"Converting SMILES to XYZ: {input_data}")
        smiles_to_xyz(
            smiles=input_data,
            output_path=xyz_path,
            optimize=should_optimize,
            optimize_method=optimize_method,
            method=method,
            basis_set=basis_set,
            functional=functional,
            solvent=solvent,
            charge=charge,
            spin=spin,
        )
        print(f"XYZ file saved: {xyz_path}")

    elif input_type == "XYZ":
        xyz_path = input_data

        if not os.path.exists(xyz_path):
            raise FileNotFoundError(f"XYZ file not found: {xyz_path}")

        # Optionally optimize existing XYZ
        should_optimize = optimize if optimize is not None else False

        if should_optimize:
            print(f"Optimizing geometry: {xyz_path}")
            if optimize_method.lower() == "pyscf":
                optimized_path = optimize_with_pyscf(
                    xyz_path,
                    method=method,
                    basis_set=basis_set,
                    functional=functional,
                    solvent=solvent,
                    charge=charge,
                    spin=spin,
                )
                if optimized_xyz:
                    os.rename(optimized_path, optimized_xyz)
                    xyz_path = optimized_xyz
                else:
                    xyz_path = optimized_path
                print(f"Optimized XYZ saved: {xyz_path}")
            else:
                raise ValueError(
                    f"Optimization method '{optimize_method}' not supported for XYZ input. Use 'pyscf'."
                )
    else:
        raise ValueError(f"Unknown input_type: {input_type}. Use 'SMILES' or 'XYZ'.")

    # Generate VDW surface
    print(f"Generating VDW surface (density={surface_density}, scale={surface_scale})...")
    coords = get_vdw_surface_coordinates(xyz_path, surface_density, surface_scale)
    print(f"Generated {len(coords)} surface points")

    # Determine charges
    if surface_type.lower() == "homogenous":
        charges = surface_charge
    else:
        # Heterogenous: use placeholder charges, user will edit
        charges = np.zeros(len(coords))

    # Save surf file
    save_surf(coords, charges, output_surf, heterogenous=(surface_type.lower() == "heterogenous"))

    return output_surf


##############################################
#         Input Parsing & Entry Point        #
##############################################



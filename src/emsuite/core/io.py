import os

from pyscf.geomopt.geometric_solver import optimize
from rdkit import Chem
from rdkit.Chem import AllChem

from .molecule import create_molecule_object, solvate_molecule

#          Molecular File Operations         #
##############################################


def extract_xyz_name(xyz_filepath):
    """
    Extract a clean molecule name from an XYZ file path.

    This function takes a file path and returns the base filename without
    extension, with path separators replaced by underscores for safe filename usage.

    Args:
        xyz_filepath (str): Path to the XYZ file

    Returns:
        str: Clean molecule name suitable for use in output filenames

    Note:
        Replaces both forward slashes and backslashes with underscores
        to ensure cross-platform compatibility.
    """
    molecule_name = os.path.splitext(os.path.basename(xyz_filepath))[0]
    molecule_name = molecule_name.replace("/", "_").replace("\\", "_")
    return molecule_name


def optimize_molecule(
    xyz_filepath,
    basis_set,
    method="dft",
    functional="b3lyp",
    original_charge=0,
    charge_change=0,
    gpu=True,
    spin_guesses=None,
    solvent=None,
):
    """
    Perform geometry optimization on a molecule and save the optimized structure.

    This function takes a molecular structure from an XYZ file, creates a quantum
    mechanical calculation object, optimizes the geometry, and writes the optimized
    coordinates to a new XYZ file.

    Args:
        xyz_filepath (str): Path to input XYZ file with initial geometry
        basis_set (str): Basis set name (e.g., 'sto-3g', '6-31g*'), a list is provided in the method-info basis-sets file
        method (str, optional): Ab initio method ('dft' or 'hf'). Defaults to 'dft'.
        functional (str, optional): DFT functional name. Defaults to 'b3lyp',however, an extensive list is provided in the method-info functionals csv file with codes for easy access
                            e.g HYB_GGA_XC_WB97X_D3 can also be used with code 399.
        original_charge (int, optional): Base molecular charge. Defaults to 0.
        charge_change (int, optional): Charge modification. Defaults to 0. Useful for generating ions.
        gpu (bool, optional): Use GPU acceleration if available. Defaults to True.
        spin_guesses (list, optional): List of spin multiplicities to test.
                             Defaults to [0, 1, 2, 3, 4]. Uses 2S notation not multiplicity (2S+1).
                             Important for open-shell systems.
        solvent (str, optional): Solvent name for implicit solvation. Defaults to None.
    Returns:
        str: Filename of the output XYZ file containing optimized geometry

    Raises:
        ValueError: If molecule object creation fails

    Note:
        - Uses PySCF's geometric solver with convergence tolerance of 1e-7
        - Output filename format: "{molecule_name}_opt.xyz"
        - Coordinates are written in Angstrom units
    """

    molecule_name = extract_xyz_name(xyz_filepath)

    # Create molecule object using the parameters
    mf = create_molecule_object(
        atom_input=xyz_filepath,
        basis_set=basis_set,
        method=method,
        functional=functional,
        original_charge=original_charge,
        charge_change=charge_change,
        gpu=gpu,
        spin_guesses=spin_guesses,
    )

    if mf is None:
        raise ValueError("Failed to create molecule object")

    # Optimize geometry
    if solvent:
        mf = solvate_molecule(mf, solvent=solvent)

    mol_eq = optimize(mf, conv_tol=1e-7)
    coords = mol_eq.atom_coords(unit="Ang")
    atoms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]

    # Write to XYZ file
    output_filename = f"{molecule_name}_opt.xyz"
    with open(output_filename, "w") as f:
        # Write number of atoms
        f.write(f"{len(atoms)}\n")
        # Write comment line
        f.write("Optimized geometry from PySCF\n")
        # Write atom coordinates
        for atom, coord in zip(atoms, coords, strict=True):
            f.write(f"{atom:2s} {coord[0]:12.8f} {coord[1]:12.8f} {coord[2]:12.8f}\n")

    print(f"Optimized geometry written to {output_filename}")
    return output_filename


def smiles_to_xyz(smiles, filename=None):
    """
    Convert a SMILES string to an XYZ coordinate file.

    This function uses RDKit to generate 3D coordinates from a SMILES string,
    including hydrogen atoms and basic geometry optimization using MMFF.

    Args:
        smiles (str): SMILES representation of the molecule
        filename (str, optional): Output filename. If None, generates automatic name.
                                 Defaults to None.

    Returns:
        str: Path to the generated XYZ file

    Note:
        - Automatically adds hydrogen atoms to the molecule
        - Performs 3D embedding and MMFF geometry optimization
        - If no filename provided, uses format "mol_{hash}.xyz"
        - Hash is generated from SMILES string for reproducibility
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    if not filename:
        filename = f"mol_{abs(hash(smiles)) % 10000}.xyz"

    conf = mol.GetConformer()
    with open(filename, "w") as f:
        f.write(f"{mol.GetNumAtoms()}\n{smiles}\n")
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    return filename


def create_mol2_file(molecule_name, coordinates, property_list, property_name):
    """
    Create a MOL2 file with molecular property data mapped to coordinates.

    This function generates a MOL2 format file where each coordinate point
    is assigned a property value. The file can be used for visualization
    of molecular properties in molecular graphics software.

    Args:
        molecule_name (str): Base name for the molecule
        coordinates (array-like): Array of 3D coordinates with shape [N, 3]
        property_list (array-like): Property values for each coordinate point with shape [N]
        property_name (str): Name/type of the property being mapped

    Returns:
        None: Writes output directly to file

    Raises:
        ValueError: If coordinates and property_list have different lengths

    Note:
        - Output filename format: "{molecule_name}_{property_name}.mol2"
        - Each point is represented as a hydrogen atom for visualization
        - Property values are stored in the charge field of the MOL2 format
        - Uses TRIPOS MOL2 format specification
    """
    if len(coordinates) != len(property_list):
        raise ValueError("coordinates and property_list must have same length")

    filename = f"{molecule_name}_{property_name}.mol2"
    num_points = len(coordinates)

    with open(filename, "w") as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{filename}\n")
        f.write(f"    {num_points} 0 0 0\n")
        f.write("SMALL\n")
        f.write("GASTEIGER\n")
        f.write("@<TRIPOS>ATOM\n")

        for i, (coord, prop_val) in enumerate(
            zip(coordinates, property_list, strict=True), start=1
        ):
            x, y, z = coord
            atom_name = "H"
            atom_type = "H1"
            subst_id = 1
            subst_name = property_name.upper()
            f.write(
                f"{i:>4} {atom_name:<4} {x:>9.4f} {y:>9.4f} {z:>9.4f} {atom_type:<4} {subst_id} {subst_name} {prop_val:>10.6f}\n"
            )

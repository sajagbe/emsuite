"""
Surface generation module for emsuite.

This module handles VDW surface generation and surf file I/O,
separated from the tuning calculations for cleaner workflow.
"""

import os
import subprocess
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


##############################################
#              surf File I/O                 #
##############################################

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
    
    with open(output_path, 'w') as f:
        if heterogenous:
            f.write("# x          y          z          q (EDIT CHARGES BELOW)\n")
        else:
            f.write("x          y          z          q\n")
        
        for i in range(n_points):
            f.write(f"{coords[i, 0]:<10.6f} {coords[i, 1]:<10.6f} {coords[i, 2]:<10.6f} {charges[i]:<10.6f}\n")
    
    print(f"Surface saved to: {output_path} ({n_points} points)")


##############################################
#         VDW Surface Generation             #
##############################################

def get_vdw_surface_coordinates(xyz_file, density=1.0, scale=1.0):
    """
    Generate van der Waals surface coordinates for a molecule.
    
    This function uses the external 'vsg' tool to generate points on the 
    van der Waals surface of a molecule from its XYZ coordinates.
    
    Args:
        xyz_file (str): Path to the XYZ file containing molecular coordinates
        density (float, optional): Surface point density. Defaults to 1.0.
        scale (float, optional): Scaling factor for van der Waals radii. Defaults to 1.0.
        
    Returns:
        numpy.ndarray: Array of surface coordinates with shape [N, 3]
        
    Raises:
        RuntimeError: If the vsg command fails
        FileNotFoundError: If the expected surface file is not created
        
    Note:
        - Requires the 'vsg' external tool to be installed and in PATH
        - Automatically cleans up temporary surface files after reading
    """
    ret = subprocess.run(['vsg', xyz_file, '-d', str(density), '-s', str(scale), '-t'], 
                         capture_output=True, text=True)
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
#         Geometry Optimization              #
##############################################

def optimize_with_rdkit(smiles, method='mmff'):
    """
    Generate and optimize 3D coordinates from SMILES using RDKit.
    
    Args:
        smiles (str): SMILES string of the molecule
        method (str): Force field method - 'mmff' or 'uff'. Defaults to 'mmff'.
        
    Returns:
        rdkit.Chem.Mol: RDKit molecule object with optimized 3D coordinates
        
    Raises:
        ValueError: If SMILES parsing fails or embedding fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    
    mol = Chem.AddHs(mol)
    
    # Embed molecule (generate 3D coordinates)
    result = AllChem.EmbedMolecule(mol, randomSeed=42)
    if result == -1:
        raise ValueError(f"Failed to embed molecule from SMILES: {smiles}")
    
    # Optimize geometry
    if method.lower() == 'mmff':
        AllChem.MMFFOptimizeMolecule(mol)
    elif method.lower() == 'uff':
        AllChem.UFFOptimizeMolecule(mol)
    else:
        raise ValueError(f"Unknown RDKit optimization method: {method}. Use 'mmff' or 'uff'.")
    
    return mol


def rdkit_mol_to_xyz(mol, output_path, comment="Generated by emsuite"):
    """
    Write an RDKit molecule to an XYZ file.
    
    Args:
        mol (rdkit.Chem.Mol): RDKit molecule with 3D coordinates
        output_path (str): Path to save the XYZ file
        comment (str): Comment line for the XYZ file
        
    Returns:
        str: Path to the saved XYZ file
    """
    conf = mol.GetConformer()
    
    with open(output_path, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n")
        f.write(f"{comment}\n")
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    
    return output_path


def optimize_with_pyscf(xyz_path, method='dft', basis_set='6-31G*', functional='b3lyp',
                        solvent=None, charge=0, spin=0):
    """
    Optimize molecular geometry using PySCF.
    
    Args:
        xyz_path (str): Path to input XYZ file
        method (str): Calculation method - 'dft' or 'hf'. Defaults to 'dft'.
        basis_set (str): Basis set. Defaults to '6-31G*'.
        functional (str): DFT functional (only for method='dft'). Defaults to 'b3lyp'.
        solvent (str or None): Solvent name for SMD solvation. Defaults to None.
        charge (int): Molecular charge. Defaults to 0.
        spin (int): Spin (2S notation). Defaults to 0.
        
    Returns:
        str: Path to the optimized XYZ file (named <input>_opt.xyz)
    """
    # Import here to avoid circular imports and keep surface module lightweight
    from . import core
    
    output_path = core.optimize_molecule(
        xyz_filepath=xyz_path,
        basis_set=basis_set,
        method=method,
        functional=functional,
        original_charge=charge,
        charge_change=0,
        gpu=core.check_gpu_info() > 0,
        spin_guesses=[spin] if spin is not None else None,
        solvent=solvent
    )
    
    return output_path


def smiles_to_xyz(smiles, output_path, optimize=True, optimize_method='mmff',
                  method='dft', basis_set='6-31G*', functional='b3lyp',
                  solvent=None, charge=0, spin=0):
    """
    Convert SMILES to XYZ file with optional geometry optimization.
    
    Args:
        smiles (str): SMILES string
        output_path (str): Path to save the XYZ file
        optimize (bool): Whether to optimize geometry. Defaults to True.
        optimize_method (str): 'mmff', 'uff', or 'pyscf'. Defaults to 'mmff'.
        method (str): QM method for pyscf optimization. Defaults to 'dft'.
        basis_set (str): Basis set for pyscf optimization. Defaults to '6-31G*'.
        functional (str): Functional for pyscf optimization. Defaults to 'b3lyp'.
        solvent (str or None): Solvent for pyscf optimization. Defaults to None.
        charge (int): Molecular charge. Defaults to 0.
        spin (int): Spin (2S notation). Defaults to 0.
        
    Returns:
        str: Path to the generated (and optionally optimized) XYZ file
    """
    if optimize_method.lower() in ['mmff', 'uff']:
        # Use RDKit for fast optimization
        mol = optimize_with_rdkit(smiles, method=optimize_method) if optimize else None
        
        if mol is None:
            # Just embed without optimization
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            AllChem.EmbedMolecule(mol, randomSeed=42)
        
        rdkit_mol_to_xyz(mol, output_path, comment=smiles)
        return output_path
        
    elif optimize_method.lower() == 'pyscf':
        # First get initial geometry from RDKit (no optimization)
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            raise ValueError(f"Failed to embed molecule from SMILES: {smiles}")
        
        # Save initial geometry
        temp_xyz = output_path.replace('.xyz', '_initial.xyz')
        rdkit_mol_to_xyz(mol, temp_xyz, comment=smiles)
        
        if optimize:
            # Optimize with PySCF
            optimized_path = optimize_with_pyscf(
                temp_xyz, method=method, basis_set=basis_set, 
                functional=functional, solvent=solvent, charge=charge, spin=spin
            )
            # Rename to desired output path
            if optimized_path != output_path:
                os.rename(optimized_path, output_path)
            # Clean up temp file
            if os.path.exists(temp_xyz):
                os.remove(temp_xyz)
        else:
            os.rename(temp_xyz, output_path)
        
        return output_path
    else:
        raise ValueError(f"Unknown optimization method: {optimize_method}. Use 'mmff', 'uff', or 'pyscf'.")


##############################################
#         Main Surface Generation            #
##############################################

def generate_surface(input_type, input_data, output_surf='surface.surf',
                     density=1.0, scale=1.0, surface_type='homogenous',
                     surface_charge=1.0, optimize=None, optimize_method='mmff',
                     method='dft', basis_set='6-31G*', functional='b3lyp',
                     solvent=None, charge=0, spin=0, optimized_xyz=None):
    """
    Generate VDW surface and save as surf file.
    
    This is the main orchestrator function that handles the complete workflow:
    SMILES/XYZ input -> optional optimization -> VDW surface -> surf output.
    
    Args:
        input_type (str): 'XYZ' or 'SMILES'
        input_data (str): Path to XYZ file or SMILES string
        output_surf (str): Output surf file path. Defaults to 'surface.surf'.
        density (float): Surface point density. Defaults to 1.0.
        scale (float): VDW radii scaling factor. Defaults to 1.0.
        surface_type (str): 'homogenous' or 'heterogenous'. Defaults to 'homogenous'.
        surface_charge (float): Charge for homogenous surface. Defaults to 1.0.
        optimize (bool or None): Whether to optimize geometry. 
            Defaults to False for XYZ, True for SMILES.
        optimize_method (str): 'mmff', 'uff', or 'pyscf'. Defaults to 'mmff'.
        method (str): QM method for pyscf. Defaults to 'dft'.
        basis_set (str): Basis set for pyscf. Defaults to '6-31G*'.
        functional (str): Functional for pyscf. Defaults to 'b3lyp'.
        solvent (str or None): Solvent for pyscf. Defaults to None.
        charge (int): Molecular charge. Defaults to 0.
        spin (int): Spin (2S notation). Defaults to 0.
        optimized_xyz (str or None): Output path for optimized XYZ file. If None, auto-generated.
        
    Returns:
        str: Path to the generated surf file
    """
    # Set default optimize behavior based on input type
    if optimize is None:
        optimize = (input_type.upper() == 'SMILES')
    
    # Determine output directory (same as input file or current dir)
    if input_type.upper() == 'XYZ':
        output_dir = os.path.dirname(os.path.abspath(input_data)) or '.'
    else:
        output_dir = '.'
    
    # Make output_surf path absolute if not already
    if not os.path.isabs(output_surf):
        output_surf = os.path.join(output_dir, output_surf)
    
    # Handle input type
    if input_type.upper() == 'SMILES':
        # Determine XYZ output path
        if optimized_xyz:
            if not os.path.isabs(optimized_xyz):
                xyz_path = os.path.join(output_dir, optimized_xyz)
            else:
                xyz_path = optimized_xyz
        else:
            base_name = f"mol_{abs(hash(input_data)) % 10000}"
            xyz_path = os.path.join(output_dir, f"{base_name}.xyz")
        
        print(f"Converting SMILES to XYZ: {input_data}")
        xyz_path = smiles_to_xyz(
            smiles=input_data,
            output_path=xyz_path,
            optimize=optimize,
            optimize_method=optimize_method,
            method=method,
            basis_set=basis_set,
            functional=functional,
            solvent=solvent,
            charge=charge,
            spin=spin
        )
        print(f"XYZ file saved: {xyz_path}")
        
    elif input_type.upper() == 'XYZ':
        xyz_path = input_data
        
        if not os.path.exists(xyz_path):
            raise FileNotFoundError(f"XYZ file not found: {xyz_path}")
        
        # Optionally optimize the XYZ geometry
        if optimize:
            print(f"Optimizing geometry: {xyz_path}")
            if optimize_method.lower() == 'pyscf':
                optimized_path = optimize_with_pyscf(
                    xyz_path, method=method, basis_set=basis_set,
                    functional=functional, solvent=solvent, charge=charge, spin=spin
                )
                # Rename to user-specified path if provided
                if optimized_xyz:
                    if not os.path.isabs(optimized_xyz):
                        optimized_xyz = os.path.join(output_dir, optimized_xyz)
                    if optimized_path != optimized_xyz:
                        os.rename(optimized_path, optimized_xyz)
                    xyz_path = optimized_xyz
                else:
                    xyz_path = optimized_path
            else:
                raise ValueError(
                    f"Cannot use {optimize_method} optimization with XYZ input. "
                    f"Use optimize_method='pyscf' or set optimize=False."
                )
            print(f"Optimized XYZ saved: {xyz_path}")
    else:
        raise ValueError(f"Invalid input_type: {input_type}. Must be 'XYZ' or 'SMILES'.")
    
    # Generate VDW surface
    print(f"Generating VDW surface (density={density}, scale={scale})...")
    coords = get_vdw_surface_coordinates(xyz_path, density=density, scale=scale)
    print(f"Generated {len(coords)} surface points")
    
    # Prepare charges based on surface type
    if surface_type.lower() == 'homogenous':
        charges = surface_charge
    elif surface_type.lower() == 'heterogenous':
        charges = np.zeros(len(coords))  # Placeholder
        print("NOTE: Heterogenous surface created with placeholder charges (0.0).")
        print("      Please edit the surf file to set per-point charges.")
    else:
        raise ValueError(f"Invalid surface_type: {surface_type}. Must be 'homogenous' or 'heterogenous'.")
    
    # Save surf file
    save_surf(coords, charges, output_surf, heterogenous=(surface_type.lower() == 'heterogenous'))
    
    return output_surf


##############################################
#         Input Parsing & Entry Point        #
##############################################

def parse_surface_input(input_file):
    """
    Parse a surface.in input file.
    
    Args:
        input_file (str): Path to the surface input file
        
    Returns:
        dict: Dictionary of parameters with defaults applied
    """
    defaults = {
        'input_type': None,  # Required
        'input_data': None,  # Required
        'output_surf': 'surface.surf',
        'optimized_xyz': None,  # Optional: custom name for optimized XYZ
        'density': 1.0,
        'scale': 1.0,
        'surface_type': 'homogenous',
        'surface_charge': 1.0,
        'optimize': None,  # Auto-determined based on input_type
        'optimize_method': 'mmff',
        'method': 'dft',
        'basis_set': '6-31G*',
        'functional': 'b3lyp',
        'solvent': None,
        'charge': 0,
        'spin': 0,
    }
    
    params = defaults.copy()
    
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Execute the file content to get variables
    local_vars = {}
    exec(content, {}, local_vars)
    
    # Update params with parsed values
    for key in defaults:
        if key in local_vars:
            params[key] = local_vars[key]
    
    # Validation
    if params['input_type'] is None:
        raise ValueError("Missing required parameter: input_type")
    if params['input_data'] is None:
        raise ValueError("Missing required parameter: input_data")
    
    if params['surface_type'].lower() == 'homogenous' and 'surface_charge' not in local_vars:
        print("Warning: surface_charge not specified for homogenous surface, using default 1.0")
    
    return params


def run_surface_calculation(input_file):
    """
    Main entry point for surface generation from input file.
    
    Args:
        input_file (str): Path to surface.in file
        
    Returns:
        str: Path to the generated surf file
    """
    print("\n" + "="*60)
    print("                  Surface Generation Module")
    print("="*60 + "\n")
    
    print(f"Reading input file: {input_file}")
    params = parse_surface_input(input_file)
    
    print(f"\nInput type: {params['input_type']}")
    print(f"Input data: {params['input_data']}")
    print(f"Surface type: {params['surface_type']}")
    print(f"Output surf: {params['output_surf']}")
    if params['optimized_xyz']:
        print(f"Optimized XYZ: {params['optimized_xyz']}")
    
    if params['optimize'] or (params['optimize'] is None and params['input_type'].upper() == 'SMILES'):
        print(f"Optimization: {params['optimize_method']}")
        if params['optimize_method'].lower() == 'pyscf':
            print(f"  Method: {params['method']}")
            print(f"  Basis: {params['basis_set']}")
            if params['method'].lower() == 'dft':
                print(f"  Functional: {params['functional']}")
            if params['solvent']:
                print(f"  Solvent: {params['solvent']}")
    
    print("\n" + "-"*60)
    
    output_path = generate_surface(
        input_type=params['input_type'],
        input_data=params['input_data'],
        output_surf=params['output_surf'],
        density=params['density'],
        scale=params['scale'],
        surface_type=params['surface_type'],
        surface_charge=params['surface_charge'],
        optimize=params['optimize'],
        optimize_method=params['optimize_method'],
        method=params['method'],
        basis_set=params['basis_set'],
        functional=params['functional'],
        solvent=params['solvent'],
        charge=params['charge'],
        spin=params['spin'],
        optimized_xyz=params['optimized_xyz']
    )
    
    print("\n" + "-"*60)
    print(f"Surface generation complete!")
    print(f"surf file: {output_path}")
    print("="*60 + "\n")
    
    return output_path
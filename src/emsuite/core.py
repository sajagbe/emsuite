import os, time, subprocess, requests
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf, dft, tdscf, qmmm, lib
from pyscf.solvent import smd
from pyscf.geomopt.geometric_solver import optimize

# GPU imports
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

OFFICE_API = "https://officeapi.akashrajpurohit.com"



##############################################
#             Hardware checks                #
##############################################

def check_gpu_info():
    """
    This function uses CuPy to detect CUDA-capable GPUs on the system.
    """
    if not GPU_AVAILABLE:
        print("\nCuPy not installed - CPU mode only.")
        print("For GPU acceleration: pip install emsuite[gpu]\n")
        return 0
    
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count < 1:
            print("\nNo GPUs found.\nSwitching to CPU mode.\n")
            return 0
        else:
            print(f"\n{device_count} GPU(s) detected.\n")
            return device_count
    except Exception as e:
        print(f"\nGPU not available: {e}")
        print("Switching to CPU mode.\n")
        return 0


def check_cpu_info():
    """
    Get the number of available CPU cores on the system.
    
    Returns:
        int: Number of CPU cores available, defaults to 1 if unable to determine
        
    Note:
        Uses os.cpu_count() to detect CPU cores and handles exceptions.
    """
    try:
        cpu_cores = os.cpu_count()
        # print(f"Number of CPU cores available: {cpu_cores}")
        return cpu_cores
    except Exception as e:
        print(f"Could not determine CPU cores: {e}")
        return 1  # Default to 1 if unable to determine




##############################################
#              Print Messages                #
##############################################

def print_startup_message():
    """
    Print the startup banner message for the Electrostatic Map Suite.
    
    """
    print(f"\n")
    print(f"="*60)
    print(f"                   Electrostatic Map Suite")
    print(f"                    By Stephen O. Ajagbe")
    print(f"="*60)


def print_office_quote():
    """
    Makes an API request to get a random quote and character from The Office TV show.
    Also cleans up any existing quote SVG files from the current directory.
    
    Raises:
        requests.exceptions.RequestException: If the API request fails
        
    Note:
        Uses the Office API at https://officeapi.akashrajpurohit.com
    """
    print("\nFetching inspirational quote...\n")
    url = f"{OFFICE_API}/quote/random"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    #remove quote*.svg if it exists, * is a wildcard for any characters
    for file in os.listdir():
        if file.startswith("quote") and file.endswith(".svg"):
            try:
                os.remove(file)
            except OSError as e:
                print(f"Warning: could not remove {file}: {e}")
    print(f"\n  {data['quote']} \n                     - {data['character']}\n")




##############################################
#         Molecular Object Creation          #
##############################################

def create_molecule_object(
    atom_input,
    basis_set,
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=True,
    spin_guesses=None #Note PySCF uses 2S = number of unpaired electrons not multiplicity (2S+1), so spin=0 is singlet, spin=1 is doublet, etc.
):
    """
    Create a PySCF molecule object with optimal spin configuration.
    
    This function creates a PySCF molecule object by testing different spin states
    and returns the one with the lowest energy. It supports both DFT and HF methods
    and can utilize GPU acceleration if available.
    
    Args:
        atom_input (str or list): Atomic coordinates (XYZ file path or coordinate list)
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
    
    Returns:
        pyscf.scf object gpu4pyscf.scf object or None: The converged SCF object with lowest energy,
                                 or None if no spin state converged.
    
    Note:
        - Uses 2S notation (0=singlet, 1=doublet, etc.)
        - Automatically tries SOSCF if initial SCF doesn't converge
        - Prints convergence information for each spin state
    """
    charge = original_charge + charge_change
    if spin_guesses is None:
        spin_guesses = [0, 1, 2, 3, 4]
    elif isinstance(spin_guesses, int):
        spin_guesses = [spin_guesses]
    elif isinstance(spin_guesses, list) and len(spin_guesses) == 0:
        spin_guesses = [0, 1, 2, 3, 4]    
        
    results = []  # store (spin, energy, mf)

    for spin in spin_guesses:
        try:
            mol = gto.Mole()
            mol.atom = atom_input
            mol.basis = basis_set
            mol.charge = charge
            mol.spin = spin  
            # mol.verbose = 4
            mol.build()

            # RKS for singlet, UKS for open shell
            if method.lower() == 'dft':
                mf = dft.UKS(mol, xc=functional) if spin > 0 else dft.RKS(mol, xc=functional)
            elif method.lower() == 'hf':
                mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
            else:
                raise ValueError("Method must be 'dft' or 'hf'")

            # Move to GPU if available and requested
            if gpu:
                mf = mf.to_gpu()
            else:
                print("GPU not available or not requested - using CPU.")
            energy = mf.kernel()

            # Try SOSCF if not converged
            if not mf.converged:
                mf = mf.newton()
                energy = mf.kernel()
            
            if mf.converged:
                print(f"Spin {spin} (2S+1={spin+1}) converged: E = {energy:.6f} Ha")
                results.append((spin, energy, mf))
            else:
                print(f"Spin {spin} (2S+1={spin+1}) did NOT converge")

        except Exception as e:
            print(f"Spin {spin} failed: {e}")

    if results:
        # pick lowest energy among converged spins
        best_spin, best_energy, best_mf = min(results, key=lambda x: x[1])
        print(f"\nLowest energy: spin={best_spin} (2S+1={best_spin+1}), E={best_energy:.6f} Ha")
        return best_mf
    else:
        print("No spin converged for this species.")
        return None

def save_chkfile(mf, chkfile_name, functional=None):
    """Save a mean-field object to a checkpoint file.
    
    Args:
        mf: PySCF or GPU4PySCF mean-field object
        chkfile_name: Name of the checkpoint file
        functional: XC functional (optional, for DFT calculations)
    """
    mf.chkfile = chkfile_name
    mf.kernel()
    if functional:
        lib.chkfile.save(chkfile_name, 'scf/xc', functional)
    print(f"Saved to {chkfile_name}, Energy: {mf.e_tot}")
    return mf



def resurrect_mol(chkfile_name):
    """Reconstruct and run a mean-field calculation from a checkpoint file.
    
    Args:
        chkfile_name: Name of the checkpoint file
        
    Returns:
        mf: Reconstructed mean-field object after running kernel()
    """
    print(f"\n=== Resurrecting {chkfile_name} ===")
    
    # Load molecule and scf data
    mol_data = lib.chkfile.load_mol(chkfile_name)
    scf_data = lib.chkfile.load(chkfile_name, 'scf')
    
    # Rebuild molecule
    mol = gto.Mole()
    mol.atom = mol_data.atom
    mol.basis = mol_data.basis
    mol.charge = mol_data.charge
    mol.spin = mol_data.spin
    mol.build()
    
    # Determine method
    is_gpu = 'gpu4pyscf' in str(type(scf_data.get('mo_coeff', '')))
    xc = lib.chkfile.load(chkfile_name, 'scf/xc')
    is_dft = xc is not None
    is_unrestricted = mol.spin != 0 or len(scf_data.get('mo_occ', [[]])) == 2
    
    # Create appropriate method object
    if is_dft:
        mf = dft.UKS(mol) if is_unrestricted else dft.RKS(mol)
        mf.xc = xc.decode('utf-8') if isinstance(xc, bytes) else xc
    else:
        mf = scf.UHF(mol) if is_unrestricted else scf.RHF(mol)
    
    # Convert to GPU if needed
    if is_gpu:
        mf = mf.to_gpu()
    
    mf.chkfile = chkfile_name
    mf.init_guess = 'chkfile'
    mf.kernel()
    print(f"Energy: {mf.e_tot}")
    return mf


##############################################
#        Molecular Object Manipulation       #
##############################################

def solvate_molecule(mf, solvent='water'):
    """
    Apply implicit solvation to a molecule using the Polarizable Continuum Model (PCM).
    
    This function adds solvation effects to an existing SCF object using the 
    C-PCM method with SMD solvent parameters.
    
    Args:
        mf (pyscf.scf object): The molecular SCF object to solvate
        solvent (str, optional): Solvent name from SMD database. Defaults to 'water'.
    
    Returns:
        pyscf.scf object: The solvated SCF object
        
    Note:
        - Uses C-PCM method with Lebedev order 29 for cavity construction
        - Automatically tries SOSCF if initial SCF doesn't converge
        - Solvent parameters are taken from the PySCF SMD database
    """
    solvent = solvent.lower()
    mf = mf.PCM()
    mf.with_solvent.eps = smd.solvent_db[solvent][5]
    mf.with_solvent.method = 'C-PCM'
    mf.with_solvent.lebedev_order = 29
    mf.kernel()
    if not mf.converged:
        print("SCF did not converge with solvent model. Trying SOSCF...")
        mf = mf.newton()
        mf.kernel()
        if not mf.converged:
            print("SOSCF also did not converge.")
        else:
            print("SOSCF converged.")
    return mf



def find_homo_lumo_and_gap(mf):
    """
    Calculate HOMO, LUMO energies and the HOMO-LUMO gap from an SCF object.
    
    This function analyzes the molecular orbitals to identify the highest occupied
    molecular orbital (HOMO) and lowest unoccupied molecular orbital (LUMO).
    
    Args:
        mf (pyscf.scf object): Converged SCF object containing molecular orbitals
        
    Returns:
        tuple: (HOMO energy, LUMO energy, HOMO-LUMO gap) in eV

    Note:
        - HOMO is the highest energy orbital with non-zero occupation
        - LUMO is the lowest energy orbital with zero occupation
        - Gap is calculated as LUMO - HOMO
    """
    homo = -float("inf")
    lumo = float("inf")
    for energy, occ in zip(mf.mo_energy, mf.mo_occ):
        if occ > 0 and energy > homo:
            homo = energy 
        if occ == 0 and energy < lumo:
            lumo = energy 
    return homo, lumo, lumo - homo

def create_qmmm_molecule_object(mf, coord_mm, q_mm, chkfile = None):
    """
    Create a QM/MM (Quantum Mechanics/Molecular Mechanics) calculation object.
    
    This function integrates classical point charges (MM region) with a quantum
    mechanical calculation. It handles both CPU and GPU-based SCF objects.
    
    Args:
        mf (pyscf.scf or gpu4pyscf.scf object): Base SCF object for the QM region
        coord_mm (numpy.ndarray): MM coordinates with shape [N, 3]
        q_mm (numpy.ndarray): MM point charges with shape [N]
        chkfile (str, optional): Checkpoint file for initial guess. Defaults to None.
                                 Typically mf's chkfile, useful for quick convergence.

    Returns:
        pyscf.scf or gpu4pyscf.scf object: New SCF object with MM charges integrated
        
    Note:
        - For GPU calculations, MM integration is performed on CPU then transferred
        - Automatically tries SOSCF if initial SCF doesn't converge
        - The MM charges modify both the core Hamiltonian and nuclear repulsion energy
    """
    if not hasattr(mf, 'to_cpu'):
        mf_new = qmmm.mm_charge(mf, coord_mm, q_mm)
    else:
        mol = mf.mol        
        # Create a new SCF object of the same type and settings
        mf_new = type(mf)(mol)
        mf_new.__dict__.update(mf.__dict__)     
        # Move to CPU for MM charge integration
        temp_mf = mf_new.to_cpu()       
        temp_mf_mm = qmmm.mm_charge(temp_mf, coord_mm, q_mm)
        v_mm = temp_mf_mm.get_hcore() - temp_mf.get_hcore()
        e_nuc_mm = temp_mf_mm.energy_nuc() - temp_mf.energy_nuc()
        v_mm_gpu = cp.asarray(v_mm)
        orig_get_hcore = mf_new.get_hcore
        orig_energy_nuc = mf_new.energy_nuc
        def get_hcore_with_mm(*args):
            hcore = orig_get_hcore()
            return hcore + v_mm_gpu
        def energy_nuc_with_mm(*args):
            return orig_energy_nuc() + e_nuc_mm
        mf_new.get_hcore = get_hcore_with_mm
        mf_new.energy_nuc = energy_nuc_with_mm      
        mf_new.charge = mol.charge
        mf_new.spin = mol.spin
        mf_new.basis = mol.basis
        if hasattr(mf, 'xc'):
            mf_new.xc = mf.xc

    if chkfile:
        mf_new.chkfile = chkfile
        mf_new.init_guess = 'chkfile'
    mf_new.kernel()
    if not mf_new.converged:
        print("SCF did not converge with MM charges. Trying SOSCF...")
        mf_new = mf_new.newton()
        mf_new.kernel()
        if not mf_new.converged:
            print("SOSCF also did not converge.")
        else:
            print("SOSCF converged.")
    return mf_new


def create_td_molecule_object(mf, nstates=5, triplet=False):
    """
    Create a time-dependent (TD) calculation object for excited states.
    
    This function creates either a TDDFT or TDHF object from a converged ground
    state calculation to compute excited state properties and electronic transitions.
    
    Args:
        mf (pyscf.scf or gpu4pyscf.scf object): Converged ground state SCF object
        nstates (int, optional): Number of excited states to calculate. Defaults to 5.
        triplet (bool, optional): Calculate triplet states if True, singlet if False.
                                 Defaults to False.
    
    Returns:
        pyscf.tdscf object: Time-dependent calculation object with computed excited states
        
    Note:
        - Automatically detects DFT vs HF ground state from mf attributes and uses appropriate TD method
        - Handles both solvated and non-solvated molecules
        - Compatible with both CPU and GPU calculations
        
    Raises:
        ValueError: If the ground state object type is not supported
    """
    if hasattr(mf, 'to_cpu') and callable(mf.to_cpu):
        if hasattr(mf, 'TDDFT') and mf.TDDFT is not None:
            td = mf.TDDFT()
        elif hasattr(mf, 'TDHF') and mf.TDHF is not None:
            td = mf.TDHF()
        else:
            raise ValueError("Unsupported ground state object type")
    else:   
        # Handle solvated molecules
        if hasattr(mf, 'with_solvent'):
            # For solvated molecules, use tdscf directly
            if hasattr(mf, 'xc'):  # DFT case
                td = tdscf.TDDFT(mf)
            else:  # HF case
                td = tdscf.TDHF(mf)
        else:
            # For non-solvated molecules, use the original method
            if hasattr(mf, 'TDDFT') and mf.TDDFT is not None:
                td = mf.TDDFT()
            elif hasattr(mf, 'TDHF') and mf.TDHF is not None:
                td = mf.TDHF()
            else:
                raise ValueError("Unsupported ground state object type")

    td.singlet = not triplet
    td.nstates = nstates
    td.kernel()
    return td


##############################################
#          Molecular File Operations         #
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
        - Handles single-point surfaces by reshaping to proper dimensions
    """
    ret = subprocess.run(['vsg', xyz_file, '-d', str(density), '-s', str(scale), '-t'])
    if ret.returncode != 0:
        raise RuntimeError(f"vsg failed: {ret.stderr}")
    base, _ = os.path.splitext(xyz_file)
    surface_file = f"{base}_vdw_surface.txt"
    if not os.path.isfile(surface_file):
        raise FileNotFoundError(f"Expected surface file not found: {surface_file}")
    coords = np.loadtxt(surface_file, dtype=float)
    if coords.ndim == 1 and coords.size == 3:
        coords = coords.reshape(1,3)
    try:
        os.remove(surface_file)
    except OSError as e:
        print(f"Warning: could not remove {surface_file}: {e}")
    return coords

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
    molecule_name = molecule_name.replace('/', '_').replace('\\', '_')
    return molecule_name

def optimize_molecule(xyz_filepath, 
    basis_set,
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=True,
    spin_guesses=None):
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
        spin_guesses=spin_guesses
    )
    
    if mf is None:
        raise ValueError("Failed to create molecule object")
    
    # Optimize geometry
    mol_eq = optimize(mf, conv_tol=1e-7)
    coords = mol_eq.atom_coords(unit='Ang')
    atoms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
    
    # Write to XYZ file
    output_filename = f"{molecule_name}_opt.xyz"
    with open(output_filename, 'w') as f:
        # Write number of atoms
        f.write(f"{len(atoms)}\n")
        # Write comment line
        f.write("Optimized geometry from PySCF\n")
        # Write atom coordinates
        for atom, coord in zip(atoms, coords):
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
    with open(filename, 'w') as f:
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
    
    with open(filename, 'w') as f:
        f.write("@<TRIPOS>MOLECULE\n")
        f.write(f"{filename}\n")
        f.write(f"    {num_points} 0 0 0\n")   
        f.write("SMALL\n")
        f.write("GASTEIGER\n")
        f.write("@<TRIPOS>ATOM\n")
        
        for i, (coord, prop_val) in enumerate(zip(coordinates, property_list), start=1):
            x, y, z = coord
            atom_name = "H"
            atom_type = "H1"
            subst_id = 1
            subst_name = property_name.upper()
            f.write(f"{i:>4} {atom_name:<4} {x:>9.4f} {y:>9.4f} {z:>9.4f} {atom_type:<4} {subst_id} {subst_name} {prop_val:>10.6f}\n")



import os, sys, time, subprocess, requests,tempfile, pickle, shutil
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
            if gpu and GPU_AVAILABLE:
                mf = mf.to_gpu()
            elif gpu and not GPU_AVAILABLE:
                print("GPU requested but not available - using CPU.")
            else:
                print("Using CPU as requested.")
                
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
    """Save a mean-field object to a checkpoint file."""
    is_gpu = hasattr(mf, 'to_cpu') and callable(mf.to_cpu)
    
    print(f"Saving {'GPU' if is_gpu else 'CPU'} object type: {type(mf)}")
    
    mf.chkfile = chkfile_name
    
    # Save molecule structure
    lib.chkfile.save_mol(mf.mol, chkfile_name)
    
    # Handle CuPy arrays for GPU objects
    if is_gpu and GPU_AVAILABLE:
        mo_energy = cp.asnumpy(mf.mo_energy) if isinstance(mf.mo_energy, cp.ndarray) else mf.mo_energy
        mo_coeff = cp.asnumpy(mf.mo_coeff) if isinstance(mf.mo_coeff, cp.ndarray) else mf.mo_coeff
        mo_occ = cp.asnumpy(mf.mo_occ) if isinstance(mf.mo_occ, cp.ndarray) else mf.mo_occ
    else:
        mo_energy = mf.mo_energy
        mo_coeff = mf.mo_coeff
        mo_occ = mf.mo_occ
    
    # Save SCF results
    lib.chkfile.save(chkfile_name, 'scf', {
        'e_tot': float(mf.e_tot),
        'mo_energy': mo_energy,
        'mo_coeff': mo_coeff,
        'mo_occ': mo_occ,
    })
    
    # CRITICAL: Save functional for DFT - use actual xc from object if not provided
    if functional:
        print(f"Saving functional (from parameter): {functional}")
        lib.chkfile.save(chkfile_name, 'scf/xc', functional)
    elif hasattr(mf, 'xc'):
        # Fallback: get XC from the object itself
        print(f"Saving functional (from mf.xc): {mf.xc}")
        lib.chkfile.save(chkfile_name, 'scf/xc', mf.xc)
    else:
        print("No functional to save (HF method)")
    
    print(f"Saved to {chkfile_name}, Energy: {mf.e_tot} ({'GPU' if is_gpu else 'CPU'})")
    return mf

def resurrect_mol(chkfile_name):
    """Reconstruct and run a mean-field calculation from a checkpoint file."""
    print(f"\n=== Resurrecting {chkfile_name} ===")
    
    # Load molecule and scf data
    mol = lib.chkfile.load_mol(chkfile_name)
    scf_data = lib.chkfile.load(chkfile_name, 'scf')
    
    # Determine method - try to load XC functional
    xc = None
    try:
        xc = lib.chkfile.load(chkfile_name, 'scf/xc')
        # Handle bytes encoding
        if isinstance(xc, bytes):
            xc = xc.decode('utf-8')
        elif isinstance(xc, np.ndarray):
            xc = str(xc)
        print(f"Loaded XC functional: {xc}")
    except (KeyError, TypeError) as e:
        print(f"No XC functional found in checkpoint (this is HF): {e}")
        xc = None
    
    is_dft = xc is not None
    is_unrestricted = mol.spin != 0 or len(scf_data.get('mo_occ', [[]])) == 2
    
    # Create appropriate method object
    if is_dft:
        print(f"Creating DFT object with xc={xc}")
        mf = dft.UKS(mol) if is_unrestricted else dft.RKS(mol)
        mf.xc = xc
    else:
        print(f"Creating HF object (no XC functional found)")
        mf = scf.UHF(mol) if is_unrestricted else scf.RHF(mol)
    
    # Convert to GPU if available
    if GPU_AVAILABLE:
        try:
            print(f"Converting {type(mf)} to GPU...")
            mf = mf.to_gpu()
            print(f"Successfully converted to GPU: {type(mf)}")
        except Exception as e:
            print(f"Warning: Could not convert to GPU: {e}")
            print("Continuing with CPU")
    
    # CRITICAL: Use chkfile for initial guess but don't write back to it
    # Create a temporary checkpoint to avoid corruption
    import tempfile
    temp_chk = tempfile.mktemp(suffix='.chk')
    shutil.copy2(chkfile_name, temp_chk)
    
    mf.chkfile = temp_chk  # Use temp file, not original
    mf.init_guess = 'chkfile'
    mf.verbose = 4
    mf.kernel()
    
    # Clean up temp checkpoint
    if os.path.exists(temp_chk):
        try:
            os.remove(temp_chk)
        except:
            pass
    
    # Set chkfile back to original for reference (but won't write to it)
    mf.chkfile = chkfile_name
    
    # Verify final object type
    is_gpu = hasattr(mf, 'to_cpu') and callable(mf.to_cpu)
    method_type = "DFT" if hasattr(mf, 'xc') else "HF"
    print(f"Resurrected {'GPU' if is_gpu else 'CPU'} {method_type} object: {type(mf)}, Energy: {mf.e_tot}")
    
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

def create_td_molecule_object(mf, nstates=5, triplet=False, force_single_gpu=False):
    """
    Create a time-dependent (TD) calculation object for excited states.
    
    For multi-GPU systems, uses subprocess isolation to force single-GPU mode.
    Passes data via pickle to avoid checkpoint file corruption.
    
    Args:
        mf: Converged ground state SCF object
        nstates: Number of excited states to calculate
        triplet: Whether to calculate triplet states (True) or singlet (False)
        force_single_gpu: If True, skip subprocess isolation (for Ray workers)
        
    Returns:
        TD object with calculated excited states
    """
    
    # Check if multiple GPUs are visible
    current_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    visible_devices = current_cuda_devices.split(',')
    
    # Determine if we should use subprocess isolation
    use_subprocess = (
        len(visible_devices) > 1 and 
        hasattr(mf, 'to_cpu') and 
        callable(mf.to_cpu) and 
        GPU_AVAILABLE and
        not force_single_gpu 
    )
    
    if use_subprocess:
        print(f"Multi-GPU detected. Using subprocess for single-GPU TDDFT...")
        
        # Get all necessary data from mf object
        is_dft = hasattr(mf, 'xc')
        xc_functional = mf.xc if is_dft else None
        
        # Convert to CPU to get numpy arrays
        mf_cpu = mf.to_cpu()
        
        # Extract all data needed for subprocess
        import pickle
        import tempfile
        
        data = {
            'atom': mf_cpu.mol.atom,
            'basis': mf_cpu.mol.basis,
            'charge': mf_cpu.mol.charge,
            'spin': mf_cpu.mol.spin,
            'mo_energy': mf_cpu.mo_energy,
            'mo_coeff': mf_cpu.mo_coeff,
            'mo_occ': mf_cpu.mo_occ,
            'is_dft': is_dft,
            'xc': xc_functional,
            'nstates': nstates,
            'triplet': triplet
        }
        
        # Save data to pickle
        with open('td_input.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        # Create subprocess script
        script = f"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '{visible_devices[0]}'

from pyscf import gto, dft, scf
import pickle
import numpy as np

# Load data
with open('td_input.pkl', 'rb') as f:
    data = pickle.load(f)

# Recreate molecule
mol = gto.Mole()
mol.atom = data['atom']
mol.basis = data['basis']
mol.charge = data['charge']
mol.spin = data['spin']
mol.build()

# Recreate SCF object
is_unrestricted = mol.spin > 0
if data['is_dft']:
    mf = dft.UKS(mol) if is_unrestricted else dft.RKS(mol)
    mf.xc = data['xc']
else:
    mf = scf.UHF(mol) if is_unrestricted else scf.RHF(mol)

# Convert to GPU
mf = mf.to_gpu()

# CRITICAL: Don't set chkfile to prevent checkpoint corruption
mf.chkfile = None
mf.verbose = 0

# Inject MO data directly (NO SCF needed!)
import cupy as cp
mf.mo_energy = cp.asarray(data['mo_energy'])
mf.mo_coeff = cp.asarray(data['mo_coeff'])
mf.mo_occ = cp.asarray(data['mo_occ'])

# Manually set converged flag
mf.converged = True

# Create and run TDDFT
td = mf.TDDFT() if data['is_dft'] else mf.TDHF()
td.singlet = not data['triplet']
td.nstates = data['nstates']
td.verbose = 4
td.kernel()

# Convert results to numpy for pickling
def to_numpy(arr):
    '''Convert CuPy array to NumPy array if needed.'''
    if hasattr(arr, 'get'):
        return arr.get()
    return arr

results = {{
    'e': to_numpy(td.e).tolist(),
    'xy': [[to_numpy(xy[0]).tolist(), to_numpy(xy[1]).tolist()] for xy in td.xy],
    'oscillator_strength': to_numpy(td.oscillator_strength()).tolist(),
}}

with open('td_results.pkl', 'wb') as f:
    pickle.dump(results, f)
"""
        
        # Write and run subprocess
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            script_path = f.name
            f.write(script)
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, 'CUDA_VISIBLE_DEVICES': visible_devices[0]}
            )
            
            # Load results
            with open('td_results.pkl', 'rb') as f:
                results = pickle.load(f)
            
            # Reconstruct TD object with results
            if hasattr(mf_cpu, 'TDDFT'):
                td = mf_cpu.TDDFT()
            else:
                td = tdscf.TDDFT(mf_cpu)
            
            # Inject results without running kernel
            td.e = np.array(results['e'])
            td.xy = [(np.array(xy[0]), np.array(xy[1])) for xy in results['xy']]
            td.converged = True
            
            print(f"TDDFT completed in subprocess on GPU {visible_devices[0]}, states: {len(td.e)}")
            
        except subprocess.CalledProcessError as e:
            print("\n" + "="*70)
            print("SUBPROCESS FAILED - TDDFT Error")
            print("="*70)
            print(f"Return code: {e.returncode}")
            print("\n--- STDOUT ---")
            print(e.stdout if e.stdout else "(empty)")
            print("\n--- STDERR ---")
            print(e.stderr if e.stderr else "(empty)")
            print("="*70)
            
            # Try to show the script that failed
            if os.path.exists(script_path):
                print("\n--- Failed Script ---")
                with open(script_path, 'r') as f:
                    print(f.read())
                print("="*70)
            
            raise RuntimeError(f"TDDFT subprocess failed with code {e.returncode}") from e
            
        finally:
            # Cleanup
            if os.path.exists(script_path):
                os.unlink(script_path)
            if os.path.exists('td_input.pkl'):
                os.unlink('td_input.pkl')
            if os.path.exists('td_results.pkl'):
                os.unlink('td_results.pkl')
        
        return td
    
    else:
        # Single GPU or CPU - normal path
        print(f"Running TDDFT in current process (force_single_gpu={force_single_gpu})")
        
        if hasattr(mf, 'with_solvent'):
            td = tdscf.TDDFT(mf) if hasattr(mf, 'xc') else tdscf.TDHF(mf)
        else:
            td = mf.TDDFT() if hasattr(mf, 'TDDFT') else mf.TDHF()
        
        td.singlet = not triplet
        td.nstates = nstates
        td.kernel()
        
        return td


##############################################
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
    molecule_name = molecule_name.replace('/', '_').replace('\\', '_')
    return molecule_name

def optimize_molecule(xyz_filepath, 
    basis_set,
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=True,
    spin_guesses=None,
    solvent=None):
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
        spin_guesses=spin_guesses
    )
    
    if mf is None:
        raise ValueError("Failed to create molecule object")
    
    # Optimize geometry
    if solvent:
        mf = solvate_molecule(mf, solvent=solvent)
    
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



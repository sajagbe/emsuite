import os,time, subprocess
import numpy as np
import cupy as cp
from rdkit import Chem
from rdkit.Chem import AllChem
from pyscf import gto, scf, dft, tdscf, qmmm, lib
from pyscf.solvent import smd
from pyscf.geomopt.geometric_solver import optimize


def check_gpu_info():
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        GPU_AVAILABLE = device_count > 0
        if GPU_AVAILABLE:
            print(f"\n Found {device_count} GPU(s).")
        else:
            print("\n No GPUs found.\n \n Switching to CPU mode.\n")
    except ImportError:
        GPU_AVAILABLE = False
        print("CuPy not installed - CPU mode only.")
    except Exception as e:
        GPU_AVAILABLE = False
        print(f"GPU not available: {e}")
    
    return device_count if GPU_AVAILABLE else 0


def check_cpu_info():
    try:
        cpu_cores = os.cpu_count()
        print(f"Number of CPU cores available: {cpu_cores}")
        return cpu_cores
    except Exception as e:
        print(f"Could not determine CPU cores: {e}")
        return 1  # Default to 1 if unable to determine

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
    charge = original_charge + charge_change
    spin_guesses = spin_guesses or [0, 1, 2, 3, 4]
    results = []  # store (spin, energy, mf)

    for spin in spin_guesses:
        try:
            mol = gto.Mole()
            mol.atom = atom_input
            mol.basis = basis_set
            mol.charge = charge
            mol.spin = spin  
            mol.build()

            # RKS for singlet, UKS for open shell
            if method.lower() == 'dft':
                mol.xc = functional
                mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
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

def solvate_molecule(mf, solvent='water'):
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

def create_qmmm_molecule_object(mf, coord_mm, q_mm, chkfile = None):
    """
    mf: SCF object (already created, possibly converged without MM)
    coord_mm: array of MM coordinates (shape [N,3])
    q_mm: array of MM charges (shape [N])

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
    Ensure mf is a CPU object (not GPU), then create a TD (time‐dependent) object
    from mf: TDDFT or TDHF depending on mf and solvent, with given number of
    states and singlet/triplet option.
    """

    # If mf has a method to convert to CPU (i.e. is GPU variant), do so
    if hasattr(mf, 'to_cpu') and callable(mf.to_cpu):
        try:
            mf_cpu = mf.to_cpu()
        except NotImplementedError:
            # If conversion isn't implemented, fallback to mf itself
            mf_cpu = mf
    else:
        mf_cpu = mf

    # Use mf_cpu for the rest
    mf = mf_cpu

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

def get_vdw_surface_coordinates(xyz_file):
    ret = subprocess.run(['vsg', xyz_file, '--txt'], capture_output=True, text=True)
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


def optimize_and_get_equilibrium(mf):
    """
    Optimize the geometry of the molecule and return the equilibrium geometry.
    """
    mol_eq = optimize(mf,conv_tol_grad=1e-7,conv_tol=1e-10)
    coords = mol_eq.atom_coords(unit='Ang')
    atoms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
    atom_list = [(atom, coord) for atom, coord in zip(atoms, coords)]
    return atom_list


def smiles_to_xyz(smiles, filename=None):
    """Convert SMILES to XYZ file"""
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


def create_optimized_molecule(atom_input, basis_set, method='dft', functional='m06-2x', charge=0, spin=1):
    """Create molecule with optional geometry optimization"""
    mf_initial = create_molecule_object(atom_input, basis_set, method, functional, charge, spin)
    mf_initial.kernel()
    opt_coords = optimize_and_get_equilibrium(mf_initial)
    
    # Create new molecule with optimized coordinates
    return create_molecule_object(opt_coords, basis_set, method, functional, charge, spin)


def create_mol2_file(molecule_name, coordinates, property_list, property_name):
    """
    Write a MOL2 file where each atom corresponds to one point in `coordinates`,
    and has an associated property value from `property_list`.
    """
    if len(coordinates) != len(property_list):
        raise ValueError("coordinates and property_list must have same length")
    
    filename = f"{molecule_name}_{property_name}_tm.mol2"
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




def find_homo_lumo_and_gap(mf):
    homo = -float("inf")
    lumo = float("inf")
    for energy, occ in zip(mf.mo_energy, mf.mo_occ):
        if occ > 0 and energy > homo:
            homo = energy 
        if occ == 0 and energy < lumo:
            lumo = energy 
    return homo, lumo, lumo - homo

def save_chkfile(mf, filename):
    mf.chkfile = f'{filename}.chk'
    mf.dump_chk(mf.__dict__)
    print(f"Saved checkpoint to {filename}.chk")

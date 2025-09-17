import numpy as np
import cupy as cp
import os,time
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


def check_cpu_cores():
    try:
        cpu_cores = os.cpu_count()
        print(f"Number of CPU cores available: {cpu_cores}")
        return cpu_cores
    except Exception as e:
        print(f"Could not determine CPU cores: {e}")
        return 1  # Default to 1 if unable to determine

No_of_GPUs = check_gpu_info()
No_of_CPU_cores = check_cpu_cores()


def create_molecule_object(
    atom_input,
    basis_set,
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=False,
    spin_guesses=None, #Note PySCF uses 2S = number of unpaired electrons not multiplicity (2S+1), so spin=0 is singlet, spin=1 is doublet, etc.
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
            if No_of_GPUs > 0 and gpu:
                mf = mf.to_gpu()
            else :
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
    return mf

def gpu_qmmm(mf_gpu, coord_mm, q_mm, chkfile = None):
    """

    Given a GPU SCF object (RHF/UHF/RKS/UKS) without MM,
    return the SCF total energy including MM charges.  
    mf_gpu: GPU SCF object (already created, possibly converged without MM)
    coord_mm: array of MM coordinates (shape [N,3])
    q_mm: array of MM charges (shape [N])

    """
    mol = mf_gpu.mol

    # Create a new SCF object of the same type and settings
    mf_gpu_new = type(mf_gpu)(mol)
    mf_gpu_new.__dict__.update(mf_gpu.__dict__)

    # Move to CPU for MM charge integration
    temp_mf = mf_gpu_new.to_cpu()


    temp_mf_mm = qmmm.mm_charge(temp_mf, coord_mm, q_mm)
    v_mm = temp_mf_mm.get_hcore() - temp_mf.get_hcore()
    e_nuc_mm = temp_mf_mm.energy_nuc() - temp_mf.energy_nuc()
    v_mm_gpu = cp.asarray(v_mm)
    orig_get_hcore = mf_gpu_new.get_hcore
    orig_energy_nuc = mf_gpu_new.energy_nuc
    def get_hcore_with_mm(*args):
        hcore = orig_get_hcore()
        return hcore + v_mm_gpu
    def energy_nuc_with_mm(*args):
        return orig_energy_nuc() + e_nuc_mm
    mf_gpu_new.get_hcore = get_hcore_with_mm
    mf_gpu_new.energy_nuc = energy_nuc_with_mm

    mf_gpu_new.charge = mol.charge
    mf_gpu_new.spin = mol.spin
    mf_gpu_new.basis = mol.basis
    if hasattr(mf_gpu, 'xc'):
        mf_gpu_new.xc = mf_gpu.xc

    if chkfile:
        mf_gpu_new.chkfile = chkfile
        mf_gpu_new.init_guess = 'chkfile'

    mf_gpu_new.kernel()

    if not mf_gpu_new.converged:
        print("SCF did not converge with MM charges. Trying SOSCF...")
        mf_gpu_new = mf_gpu_new.newton()
        mf_gpu_new.kernel()
        if not mf_gpu_new.converged:
            print("SOSCF also did not converge.")
        else:
            print("SOSCF converged.")
    return mf_gpu_new




def create_td_molecule_object(mf, nstates=5, triplet=False):
    # Handle solvated molecules
    if hasattr(mf, 'with_solvent'):
        if hasattr(mf, 'xc'):  # DFT case
            if hasattr(mf, 'to_cpu'):
                td = mf.TDDFT()
            else:
                td = tdscf.TDDFT(mf)
        else:  # HF case
            if hasattr(mf, 'to_cpu'):
                td = tdscf.TDHF(mf.to_cpu())
            else:
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
    return td



#Testing Parameters
molecule = 'ethanol.xyz'
coord_mm = np.array([[0.0, 0.0, 1.5]])
q_mm = np.array([1.0])


mf_gpu = create_molecule_object(
    atom_input=molecule,
    basis_set="6-31g*",
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=True,
    spin_guesses=[0]
)

mf_gpu.chkfile = 'anion.chk'
mf_gpu.dump_chk(mf_gpu.__dict__)



mf_gpu_qmmm = gpu_qmmm(mf_gpu, coord_mm, q_mm, chkfile = 'anion.chk')

# mf_gpu = mf_gpu.to_cpu()
# mf_gpu_qmmm = qmmm.mm_charge(mf_gpu, coord_mm, q_mm)

solvated_vac = solvate_molecule(mf_gpu, solvent='water')
solvated_wsc = solvate_molecule(mf_gpu_qmmm, solvent='water')

solvated_vac.kernel()
solvated_wsc.kernel()


# td_gpu = create_td_molecule_object(mf_gpu)
# td_gpu_qmmm = create_td_molecule_object(mf_gpu_qmmm)
# td_solvated_vac = create_td_molecule_object(solvated_vac)
# td_solvated_wsc = create_td_molecule_object(solvated_wsc)

# td_gpu.kernel()
# td_gpu_qmmm.kernel()

start = time.time()
td_solvated_vac.kernel()

end = time.time()
print(f"TDDFT w/ chk {end - start:.2f}")
# td_solvated_wsc.kernel()




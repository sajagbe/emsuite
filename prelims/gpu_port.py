import numpy as np
import cupy as cp
from pyscf import gto, scf, dft, tdscf, qmmm, lib
from pyscf.solvent import smd
from pyscf.geomopt.geometric_solver import optimize


try:
    device_count = cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = device_count > 0
    
    if GPU_AVAILABLE:
        print(f"\n Found {device_count} GPU(s):")
        for i in range(device_count):
            cp.cuda.Device(i).use()
            props = cp.cuda.runtime.getDeviceProperties(i)
            name = props['name'].decode()
            print(f"  GPU {i}: {name}")        
        # Test the first GPU
        cp.cuda.Device(0).use()
        test_array = cp.array([1, 2, 3])
        print(f"\n Using GPU 0 for computations. \n")
    else:
        print("\n No GPUs found.\n \n Switching to CPU mode.\n")
        
except (ImportError, cp.cuda.runtime.CUDARuntimeError) as e:
    GPU_AVAILABLE = False
    print(f"GPU not available: {e}")

# if GPU_AVAILABLE:



# else:



def create_qmmm_molecule_object(mf, coord_mm, q_mm, chkfile = None):
    """
    mf: SCF object (already created, possibly converged without MM)
    coord_mm: array of MM coordinates (shape [N,3])
    q_mm: array of MM charges (shape [N])

    """
    # mf_new = qmmm.mm_charge(mf, coord_mm, q_mm)
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




def create_charged_molecule_object(
    atom_input,
    basis_set,
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=False,
    spin_guesses=None,
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
            mol.spin = spin   # number of unpaired electrons (2S)
            mol.build()

            # RKS for singlet, UKS for open shell
            if method.lower() == 'dft':
                mol.xc = functional
                mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
            elif method.lower() == 'hf':
                mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
            else:
                raise ValueError("Method must be 'dft' or 'hf'")
            if gpu:
                mf = mf.to_gpu()
            energy = mf.kernel()

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
        return best_mf, best_spin, best_energy
    else:
        print("No spin converged for this species.")
        return None, None, None


def find_homo_lumo_and_gap(mf):
    homo = -float("inf")
    lumo = float("inf")
    for energy, occ in zip(mf.mo_energy, mf.mo_occ):
        if occ > 0 and energy > homo:
            homo = energy 
        if occ == 0 and energy < lumo:
            lumo = energy 
    return homo, lumo, lumo - homo

def solvate_molecule(mf, solvent='water'):
    solvent = solvent.lower()
    mf = mf.PCM()
    mf.with_solvent.eps = smd.solvent_db[solvent][5]
    mf.with_solvent.method = 'C-PCM'
    mf.with_solvent.lebedev_order = 29
    return mf

def optimize_and_get_equilibrium(mf):
    """
    Optimize the geometry of the molecule and return the equilibrium geometry.
    """
    mf = mf.to_cpu()
    mol_eq = optimize(mf,conv_tol_grad=1e-7,conv_tol=1e-10)
    coords = mol_eq.atom_coords(unit='Ang')
    atoms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
    atom_list = [(atom, coord) for atom, coord in zip(atoms, coords)]
    return atom_list

molecule = "water.xyz"
coord_mm = np.array([[0.0, 0.0, 1.5]])
q_mm = np.array([1.0])

mf_gpu, _, _ = create_charged_molecule_object(
    atom_input=molecule,
    basis_set="6-31g*",
    method='hf',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=True,
    spin_guesses=[0]
)


mf_gpu.chkfile = 'anion.chk'
mf_gpu.dump_chk(mf_gpu.__dict__)


mf_gpu_qmmm = create_qmmm_molecule_object(mf_gpu, coord_mm, q_mm, chkfile = 'anion.chk')



solvated_vac = solvate_molecule(mf_gpu, solvent='water')
solvated_wsc = solvate_molecule(mf_gpu_qmmm, solvent='water')


solvated_vac.kernel()
solvated_wsc.kernel()

print(mf_gpu_qmmm.e_tot, mf_gpu.e_tot)
print(solvated_wsc.e_tot,solvated_vac.e_tot)


homo, lumo, gap = find_homo_lumo_and_gap(mf_gpu)
homo_wsc, lumo_wsc, gap_wsc = find_homo_lumo_and_gap(mf_gpu_qmmm)

homo_sol_vac, lumo_sol_vac, gap_sol_vac = find_homo_lumo_and_gap(solvated_vac)
homo_sol_wsc, lumo_sol_wsc, gap_sol_wsc = find_homo_lumo_and_gap(solvated_wsc)



print(f"VAC: {homo}, {lumo}, {gap}")
print(f"WSC: {homo_wsc}, {lumo_wsc}, {gap_wsc}")
print(f"Solvated VAC: {homo_sol_vac}, {lumo_sol_vac}, {gap_sol_vac}")
print(f"Solvated WSC: {homo_sol_wsc}, {lumo_sol_wsc}, {gap_sol_wsc}")

# #TDSCF Properties




nstates = 5



td_vac = mf_gpu.TDDFT()
td_vac.nstates = nstates
td_vac.singlet = False
td_vac.kernel()


td_sol_vac = solvated_vac.TDDFT()
td_sol_vac.nstates = nstates
td_sol_vac.singlet = True
td_sol_vac.kernel()

td_wsc = mf_gpu_qmmm.TDDFT()
td_wsc.nstates = nstates
td_wsc.singlet = True   
td_wsc.kernel()

td_sol_wsc = solvated_wsc.TDDFT()
td_sol_wsc.nstates = nstates
td_sol_wsc.singlet = True
td_sol_wsc.kernel()


print("Excitation energies (Ha):", td_vac.e)
print("Oscillator strengths:", td_vac.oscillator_strength())

print("Excitation energies (Ha):", td_sol_vac.e)
print("Oscillator strengths:", td_sol_vac.oscillator_strength())

print("Excitation energies (Ha):", td_wsc.e)
print("Oscillator strengths:", td_wsc.oscillator_strength())

print("Excitation energies (Ha):", td_sol_wsc.e)
print("Oscillator strengths:", td_sol_wsc.oscillator_strength())


# srun -p qGPU48 -A CHEM9C4 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 --time=01:00:00 --mem=8G --pty bash





# def gpu_worker(gpu_id, calculation_func, result_dict, key, *args):
#     """Worker function for GPU calculations"""
#     with cp.cuda.Device(gpu_id):  # Switch to specific GPU
#         result_dict[key] = calculation_func(*args)  # Run calc and store result


# def run_scf_calculation(mol_data):
#     """Example SCF calculation function"""
#     mol = gto.M(**mol_data)
#     mf = scf.RHF(mol).to_gpu()
#     return mf.kernel()

# if No_of_GPUs:
#     # Run two different molecules simultaneously




    
#     mol1_data = {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'cc-pVDZ'}
#     mol2_data = {'atom': 'He 0 0 0', 'basis': 'cc-pVDZ'}
    
#     results = {}
#     t1 = threading.Thread(target=gpu_worker, 
#                          args=(0, run_scf_calculation, results, 'H2', mol1_data))
#     t2 = threading.Thread(target=gpu_worker, 
#                          args=(1, run_scf_calculation, results, 'He', mol2_data))
    
#     t1.start()
#     t2.start()
#     t1.join()
#     t2.join()
    
#     print(f"H2 energy: {results['H2']}")
#     print(f"He energy: {results['He']}")
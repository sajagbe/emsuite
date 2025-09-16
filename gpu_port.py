import numpy as np
import cupy as cp
from pyscf import gto, scf, dft, tdscf, qmmm, lib
from pyscf.solvent import smd
# from pyscf.lib import chkfile
from pyscf.geomopt.geometric_solver import optimize

# GPU availability check
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def qmmm_gpu(mf_gpu, coord_mm, q_mm, chkfile = None):
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

    # if soscf:
    #     mf_gpu_new = mf_gpu_new.undo_soscf()

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

    print("Running QMMM on GPU with charge:", mol.charge, "spin:", mol.spin)
    
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

molecule = "LF.xyz"
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


mf_gpu_qmmm = qmmm_gpu(mf_gpu, coord_mm, q_mm, chkfile = 'anion.chk')



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
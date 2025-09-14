
import numpy as np
import cupy as cp
from pyscf import gto, scf, dft, tdscf, qmmm
from pyscf.solvent import smd
from pyscf.geomopt.geometric_solver import optimize

def qmmm_gpu(mf_gpu, coord_mm, q_mm):
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
    # Step 1: Extract MM contributions using a temporary CPU SCF
    scf_class = type(mf_gpu).cpu() if hasattr(mf_gpu, 'cpu') else type(mf_gpu)
    temp_mf = scf_class(mol)
    temp_mf = temp_mf.to_cpu()
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
    mf_gpu_new.kernel()
    return mf_gpu_new

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

coord_mm = np.array([[0.0, 0.0, 1.5]])
q_mm = np.array([1.0])

mol = gto.M(
    atom = "water.xyz",
    basis = "cc-pvtz",
)

mol.xc = 'b3lyp' 


#SCF Energies


mf_gpu = scf.RKS(mol).to_gpu()
mf_gpu.kernel()

opt_coords = optimize_and_get_equilibrium(mf_gpu)


# # print(mf_gpu.e_tot)
# mf_gpu_qmmm = qmmm_gpu(mf_gpu, coord_mm, q_mm)
# mf_gpu_qmmm.kernel()


# solvated_vac = solvate_molecule(mf_gpu, solvent='water')
# solvated_wsc = solvate_molecule(mf_gpu_qmmm, solvent='water')

# solvated_vac.kernel()
# solvated_wsc.kernel()

# print(mf_gpu_qmmm.e_tot, mf_gpu.e_tot)
# print(solvated_wsc.e_tot,solvated_vac.e_tot)


# homo, lumo, gap = find_homo_lumo_and_gap(mf_gpu)
# homo_wsc, lumo_wsc, gap_wsc = find_homo_lumo_and_gap(mf_gpu_qmmm)

# homo_sol_vac, lumo_sol_vac, gap_sol_vac = find_homo_lumo_and_gap(solvated_vac)
# homo_sol_wsc, lumo_sol_wsc, gap_sol_wsc = find_homo_lumo_and_gap(solvated_wsc)



# print(f"VAC: {homo}, {lumo}, {gap}")
# print(f"WSC: {homo_wsc}, {lumo_wsc}, {gap_wsc}")
# print(f"Solvated VAC: {homo_sol_vac}, {lumo_sol_vac}, {gap_sol_vac}")
# print(f"Solvated WSC: {homo_sol_wsc}, {lumo_sol_wsc}, {gap_sol_wsc}")

# #TDSCF Properties

# nstates = 5

# td_vac = mf_gpu.TDDFT()
# td_vac.nstates = nstates
# td_vac.kernel()

# td_sol_vac = solvated_vac.TDDFT()
# td_sol_vac.nstates = nstates
# td_sol_vac.kernel()

# td_wsc = mf_gpu_qmmm.TDDFT()
# td_wsc.nstates = nstates
# td_wsc.kernel()

# td_sol_wsc = solvated_wsc.TDDFT()
# td_sol_wsc.nstates = nstates
# td_sol_wsc.kernel()


# print("Excitation energies (Ha):", td_vac.e)
# print("Oscillator strengths:", td_vac.oscillator_strength())

# print("Excitation energies (Ha):", td_sol_vac.e)
# print("Oscillator strengths:", td_sol_vac.oscillator_strength())

# print("Excitation energies (Ha):", td_wsc.e)
# print("Oscillator strengths:", td_wsc.oscillator_strength())

# print("Excitation energies (Ha):", td_sol_wsc.e)
# print("Oscillator strengths:", td_sol_wsc.oscillator_strength())

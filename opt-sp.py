import numpy as np
import os,subprocess,sys
from pyscf import gto, scf
from pyscf.solvent import pcm,smd
from pyscf.geomopt.berny_solver import optimize

def create_dft_molecule(atom_input, basis_set, functional='m06-2x', charge=0, spin=0):
    """
    Create a PySCF molecule and KS/DFT object from either:
    - atom_input: list of (atom_symbol, (x,y,z)) tuples
    - atom_input: string path to XYZ file
    """
    mol = gto.Mole(basis = basis_set, charge=charge, spin=spin, xc=functional)
    if isinstance(atom_input, str) and os.path.isfile(atom_input):
        mol.atom = atom_input 
    else:
        mol.atom = atom_input  
    mf = scf.UKS(mol)
    return mf


def optimize_and_get_equilibrium(mf):
    """
    Optimize the geometry of the molecule and return the equilibrium geometry.
    """
    mol_eq = optimize(mf)
    coords = mol_eq.atom_coords()
    atoms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
    atom_list = [(atom, coord) for atom, coord in zip(atoms, coords)]
    return atom_list

#Parameters:
molecule = 'water.xyz'
functional = 'm06-2x'
opt_basis_set = '6-31+G*'
sp_basis_set = 'ccpvdz'
solvent = None

mf = create_dft_molecule(molecule, opt_basis_set,functional=functional)
equilibrium_geometry = optimize_and_get_equilibrium(mf)

if solvent is not None:
    mf = create_dft_molecule(equilibrium_geometry, sp_basis_set,functional=functional)
    mf = mf.PCM()
    mf.with_solvent.eps = smd.solvent_db[solvent][5]
    mf.with_solvent.method = 'C-PCM'
    mf.with_solvent.lebedev_order = 29
else:
    mf = create_dft_molecule(equilibrium_geometry, sp_basis_set)

mf.kernel()


# def get_redox_potential(mf, gfec_basis_set, gfec_functional, sscf_basis_set, sscf_functional, charge, spin, method):
#     #Neutral
#     rdx_n = create_molecule_object(mf.mol.atom, gfec_basis_set, method=method,functional=gfec_functional, charge=charge,spin=spin)
#     rdx_n_opt = optimize_and_get_equilibrium(rdx_n)
#     rdx_n = create_molecule_object(rdx_n_opt,gfec_basis_set, method=method,functional=gfec_functional, charge=charge,spin=spin)
#     _, _, G_corr_rdx_n = get_molecule_gfec(rdx_n)

#     #Anion
#     rdx_a,_,_ = create_charged_molecule_object(mf.mol.atom, gfec_basis_set, method=method,functional=gfec_functional, original_charge=charge, original_spin=spin, charge_change=-1)
#     rdx_a_opt = optimize_and_get_equilibrium(rdx_a)
#     rdx_a,_,_ = create_charged_molecule_object(rdx_a_opt, gfec_basis_set, method=method, functional=gfec_functional, original_charge=charge, original_spin=spin, charge_change=-1)
#     _,_, G_corr_rdx_a = get_molecule_gfec(rdx_a)


#     #hbs = higher basis set SCF
#     rdx_n_hbs = create_molecule_object(rdx_n.mol.atom, sscf_basis_set, functional=sscf_functional, method=method, charge=charge, spin=spin)
#     solvated_n = solvate_molecule(rdx_n_hbs, solvent=solvent)

#     rdx_a_hbs,_,_ = create_charged_molecule_object(rdx_a.mol.atom, sscf_basis_set, functional=sscf_functional, method=method, original_charge=charge, original_spin=spin, charge_change=-1)
#     solvated_a = solvate_molecule(rdx_a_hbs, solvent=solvent)

#     solvated_n.kernel()
#     solvated_a.kernel()

#     #GFE Adjustment
#     gfe_n = solvated_n.e_tot + G_corr_rdx_n
#     gfe_a = solvated_a.e_tot + G_corr_rdx_a

#     #Redox Gibbs Free Energy 
#     dG_red  = -1*((gfe_a - gfe_n) * HARTREE_TO_EV)
#     rdx_pot = (dG_red) -4.43

#     print(f"G_corr_rdx_a: {G_corr_rdx_a}, G_corr_rdx_n: {G_corr_rdx_n}")
#     print(f"gfe_a: {gfe_a}, gfe_n: {gfe_n}, dG_red: {dG_red}, rdx_pot: {rdx_pot}")

#     return rdx_pot
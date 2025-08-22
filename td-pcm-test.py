import numpy as np
import os,subprocess,sys
from pyscf import gto,scf,dft,qmmm,tdscf,hessian
from pyscf.solvent import smd
from pyscf.hessian import thermo
from pyscf.geomopt.geometric_solver import optimize


def solvate_molecule(mf, solvent='water'):
    solvent = solvent.lower()
    mf = mf.PCM()
    mf.with_solvent.eps = smd.solvent_db[solvent][5]
    mf.with_solvent.method = 'C-PCM'
    mf.with_solvent.lebedev_order = 29
    return mf


def create_charged_molecule_object(atom_input, basis_set, method='dft', functional='m06-2x', original_charge=0, original_spin=1, charge_change=-1):
    """
    Create charged form of molecule (anion or cation) and return the lowest energy one with its spin and energy
    
    Args:
        charge_change: -1 for anion (add electron), +1 for cation (remove electron)
    """
    new_charge = original_charge + charge_change
    
    # Determine possible spin states based on charge change
    if charge_change == -1:  
        if original_spin == 2:  
            possible_spins = [1, 3] 
        elif original_spin == 1:  
            possible_spins = [2]  
        elif original_spin == 3:  
            possible_spins = [2, 4] 
            
    elif charge_change == +1:  
        if original_spin == 1:  
            possible_spins = [2]  
        elif original_spin == 2:  
            possible_spins = [1, 3]  
        elif original_spin == 3:  
            possible_spins = [2, 4] 
    
    energies = []
    objects = []
    
    for spin in possible_spins:
        pyscf_spin = spin - 1
        mol = gto.Mole(basis=basis_set, charge=new_charge, spin=pyscf_spin)
        mol.atom = atom_input
        mol.build()
        if method.lower() == 'dft':
            mf = dft.UKS(mol) if pyscf_spin > 0 else dft.RKS(mol)
            mf.xc = functional
        elif method.lower() == 'hf':
            mf = scf.UHF(mol) if pyscf_spin > 0 else scf.RHF(mol)
        energy = mf.kernel()
        energies.append(energy)
        objects.append(mf)
    
    # Find the lowest energy
    min_idx = energies.index(min(energies))
    lowest_energy = energies[min_idx]
    optimal_spin = possible_spins[min_idx]
    best_object = objects[min_idx]
    
    return best_object, optimal_spin, lowest_energy

def create_td_molecule_object(mf, nstates=5, triplet=False):

    if hasattr(mf, 'TDDFT') and mf.TDDFT is not None:
      td = mf.TDDFT()
    elif hasattr(mf, 'TDHF') and mf.TDHF is not None:
        td = mf.TDHF()
    else:
        raise ValueError("Unsupported ground state object type")
    td.singlet = not triplet
    td.nstates = nstates
    return td



def create_molecule_object(atom_input, basis_set, method='dft', functional='m06-2x', charge=0, spin=1):
    """
    Create a PySCF molecule and KS/DFT object from either:
    - atom_input: list of (atom_symbol, (x,y,z)) tuples
    - atom_input: string path to XYZ file
    """
    spin -= 1
    mol = gto.M(basis = basis_set, charge=charge, spin=spin,symmetry=False)
    if isinstance(atom_input, str) and os.path.isfile(atom_input):
        mol.atom = atom_input 
    else:
        mol.atom = atom_input  
    mol.build()

    # Choose method and handle spin
    if method.lower() == 'dft':
        mol.xc=functional
        mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
    elif method.lower() == 'hf':
        mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
    else:
        raise ValueError("method must be 'dft' or 'hf'")
    return mf



#Parameters
molecule = 'water'
method = 'dft'
functional = 'b3lyp'
basis_set = 'augccpvdz'
solvent = 'benzene'
charge = 0
spin = 1
state_of_interest = 2
triplet_excitation = False
rdx_solvent = 'acetonitrile'
gfec_functional = 'b3lyp'
gfec_basis_set = '6-31+G*'


molecule_object = create_molecule_object(f'{molecule}.xyz', basis_set, method=method, functional=functional, charge=charge, spin=spin)
molecule_object.run()
solvated_molecule = solvate_molecule(molecule_object, solvent=solvent)
solvated_molecule.run()


td = create_td_molecule_object(molecule_object, nstates=5, triplet=triplet_excitation)
td.kernel()

td = create_td_molecule_object(solvated_molecule, nstates=5, triplet=triplet_excitation)
td.with_solvent.equilibrium_solvation = True
td.kernel()


anion_mf,_,_ = create_charged_molecule_object(f'{molecule}.xyz', basis_set, method=method, functional=functional, original_charge=charge, original_spin=spin, charge_change=-1)

solvated_anion = solvate_molecule(anion_mf, solvent=solvent)
solvated_anion.run()


td = create_td_molecule_object(anion_mf, nstates=5, triplet=triplet_excitation)
td.kernel()

td = create_td_molecule_object(solvated_anion, nstates=5, triplet=triplet_excitation)
td.with_solvent.equilibrium_solvation = True
td.kernel()






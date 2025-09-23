import cupy as cp, numpy as np, pandas as pd
from pyscf import gto, dft,lib
import os,time



#Testing Parameters
molecule_name = 'water.xyz'
coord_mm = np.array([[0.0, 0.0, 1.5]])
q_mm = np.array([1.0])


molecule = create_molecule_object(
    atom_input=molecule_name,
    basis_set="6-31g*",
    method='dft',
    functional='b3lyp',
    original_charge=0,
    charge_change=0,
    gpu=False,
    spin_guesses=[]
)

molecule.chkfile = f'{molecule_name}.chk'
molecule.dump_chk(molecule.__dict__)

qmmm_molecule = create_qmmm_molecule_object(molecule, coord_mm, q_mm, chkfile = f'{molecule_name}.chk')

solvated_molecule_vac = solvate_molecule(molecule, solvent='water')
solvated_molecule_wsc = solvate_molecule(qmmm_molecule, solvent='water')

solvated_molecule_vac.kernel()
solvated_molecule_wsc.kernel()

td_gpu = create_td_molecule_object(molecule,nstates=3, triplet=False)
td_gpu_trip = create_td_molecule_object(molecule,nstates=3, triplet=True)

td_gpu_qmmm = create_td_molecule_object(qmmm_molecule,nstates=3, triplet=True)
td_solvated_vac = create_td_molecule_object(solvated_molecule_vac,nstates=3, triplet=True)
td_solvated_wsc = create_td_molecule_object(solvated_molecule_wsc,nstates=3, triplet=True)

td_gpu.kernel()
td_gpu_trip.kernel()
td_gpu_qmmm.kernel()
td_solvated_vac.kernel()
td_solvated_wsc.kernel()


# srun -p qGPU24 -A CHEM9C4 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 --time=01:00:00 --mem=8G --pty bash
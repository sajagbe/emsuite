import time
import cupy as cp
import numpy as np
from pyscf import qmmm,gto,scf


def qmmm_energy_gpu(mf_gpu, coord_mm, q_mm):
    """
    Given a GPU SCF object (RHF/UHF/RKS/UKS) without MM,
    return the SCF total energy including MM charges.
    
    mf_gpu: GPU SCF object (already created, possibly converged without MM)
    coord_mm: array of MM coordinates (shape [N,3])
    q_mm: array of MM charges (shape [N])
    """
    # Step 1: Extract MM contributions using a temporary CPU SCF
    mol = mf_gpu.mol
    scf_class = type(mf_gpu).cpu() if hasattr(mf_gpu, 'cpu') else type(mf_gpu)
    
    # Temporary CPU SCF object
    temp_mf = scf_class(mol)
    temp_mf = temp_mf.to_cpu()
    temp_mf_mm = qmmm.mm_charge(temp_mf, coord_mm, q_mm)
    
    v_mm = temp_mf_mm.get_hcore() - temp_mf.get_hcore()
    e_nuc_mm = temp_mf_mm.energy_nuc() - temp_mf.energy_nuc()
    
    # Step 2: Convert v_mm to GPU
    v_mm_gpu = cp.asarray(v_mm)
    
    # Step 3: Save original methods
    orig_get_hcore = mf_gpu.get_hcore
    orig_energy_nuc = mf_gpu.energy_nuc
    
    # Step 4: Override to include MM
    def get_hcore_with_mm(*args):
        hcore = orig_get_hcore()
        return hcore + v_mm_gpu
    
    def energy_nuc_with_mm(*args):
        return orig_energy_nuc() + e_nuc_mm
    
    mf_gpu.get_hcore = get_hcore_with_mm
    mf_gpu.energy_nuc = energy_nuc_with_mm
    
    # Step 5: Re-run SCF with MM contributions
    mf_gpu.kernel()
    
    return mf_gpu.e_tot


mol = gto.M(
    atom='taxol.xyz',
    basis='sto-3g'
)
mol.build()

# MM point charge
coord_mm = np.array([[0.0, 0.0, 1.5]])
q_mm = np.array([1.0])


mf_cpu = scf.RHF(mol)
cpu_start_time1 = time.time()
mf_cpu.kernel()
cpu_time1 =  time.time() - cpu_start_time1 


cpu_start_time2 = time.time()
e_cpu_mm = qmmm.mm_charge(mf_cpu, coord_mm, q_mm).kernel()
cpu_time2 =  time.time() -cpu_start_time2


mf_gpu = mf_cpu.to_gpu() 
gpu_start_time1 = time.time()
mf_gpu.kernel()
gpu_time1 = time.time() - gpu_start_time1

gpu_start_time2 = time.time()
# Assume mf_gpu is your GPU SCF object (RHF/UHF/RKS/UKS) without MM
e_gpu_with_mm = qmmm_energy_gpu(mf_gpu, coord_mm, q_mm)
gpu_time2 =  time.time() - gpu_start_time2 


print(f'CPU Vacuum: {cpu_time1}, CPU wsc: {cpu_time2}, GPU Vacuum: {gpu_time1}, GPU wsc: {gpu_time2}')
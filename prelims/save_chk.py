from pyscf import gto, dft, scf, lib

def save_chkfile(mf, chkfile_name, functional=None):
    """Save a mean-field object to a checkpoint file.
    
    Args:
        mf: PySCF or GPU4PySCF mean-field object
        chkfile_name: Name of the checkpoint file
        functional: XC functional (optional, for DFT calculations)
    """
    mf.chkfile = chkfile_name
    mf.kernel()
    if functional:
        lib.chkfile.save(chkfile_name, 'scf/xc', functional)
    print(f"Saved to {chkfile_name}, Energy: {mf.e_tot}")
    return mf


def resurrect_mol(chkfile_name):
    """Reconstruct and run a mean-field calculation from a checkpoint file.
    
    Args:
        chkfile_name: Name of the checkpoint file
        
    Returns:
        mf: Reconstructed mean-field object after running kernel()
    """
    print(f"\n=== Resurrecting {chkfile_name} ===")
    
    # Load molecule and scf data
    mol_data = lib.chkfile.load_mol(chkfile_name)
    scf_data = lib.chkfile.load(chkfile_name, 'scf')
    
    # Rebuild molecule
    mol = gto.Mole()
    mol.atom = mol_data.atom
    mol.basis = mol_data.basis
    mol.charge = mol_data.charge
    mol.spin = mol_data.spin
    mol.build()
    
    # Determine method
    is_gpu = 'gpu4pyscf' in str(type(scf_data.get('mo_coeff', '')))
    xc = lib.chkfile.load(chkfile_name, 'scf/xc')
    is_dft = xc is not None
    is_unrestricted = mol.spin != 0 or len(scf_data.get('mo_occ', [[]])) == 2
    
    # Create appropriate method object
    if is_dft:
        mf = dft.UKS(mol) if is_unrestricted else dft.RKS(mol)
        mf.xc = xc.decode('utf-8') if isinstance(xc, bytes) else xc
    else:
        mf = scf.UHF(mol) if is_unrestricted else scf.RHF(mol)
    
    # Convert to GPU if needed
    if is_gpu:
        mf = mf.to_gpu()
    
    mf.chkfile = chkfile_name
    mf.init_guess = 'chkfile'
    mf.kernel()
    print(f"Energy: {mf.e_tot}")
    return mf


# Test 1: CPU RKS
print("=== Test 1: CPU RKS ===")
mol_cpu = gto.Mole()
mol_cpu.atom = '''
O 0 0 0
H 0 1 0
H 0 0 1
'''
mol_cpu.basis = '6-31g'
mol_cpu.charge = 0
mol_cpu.spin = 0
mol_cpu.build()

mf_cpu = dft.RKS(mol_cpu)
mf_cpu.xc = 'b3lyp'
save_chkfile(mf_cpu, 'cpu_molecule.chk', functional='b3lyp')

# Resurrect CPU
mf_cpu_resurrected = resurrect_mol('cpu_molecule.chk')

# Test 2: GPU RKS
print("\n=== Test 2: GPU RKS ===")
mf_gpu = mf_cpu.to_gpu()
save_chkfile(mf_gpu, 'gpu_molecule.chk', functional='b3lyp')

# Resurrect GPU
mf_gpu_resurrected = resurrect_mol('gpu_molecule.chk')

print("\n=== Summary ===")
print(f"CPU original: {mf_cpu.e_tot}")
print(f"CPU resurrected: {mf_cpu_resurrected.e_tot}")
print(f"GPU original: {mf_gpu.e_tot}")
print(f"GPU resurrected: {mf_gpu_resurrected.e_tot}")
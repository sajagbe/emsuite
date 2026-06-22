import os
from pathlib import Path
import sys

# Add parent directories to path to import emsuite
sys.path.insert(0, str(Path(__file__).parent.parent))

from emsuite.surface import optimize_with_pyscf
from emsuite.core import check_gpu_info

# XYZ files to optimize
xyz_files = ['Indole.xyz', 'LF.xyz', 'p-cresol.xyz']

print("="*60)
print("          Optimizing Molecular Geometries with PySCF")
print("="*60)

for xyz_file in xyz_files:
    if os.path.exists(xyz_file):
        print(f"\nProcessing: {xyz_file}")
        try:
            output_path = optimize_with_pyscf(
                xyz_file,
                method='dft',
                basis_set='6-31+G*',
                functional='b3lyp',
                solvent=None,
                charge=0,
                spin=0,
                gpu=check_gpu_info() > 0
            )
            print(f"✓ Saved optimized geometry to: {output_path}")
        except Exception as e:
            print(f"✗ Error optimizing {xyz_file}: {e}")
    else:
        print(f"✗ File not found: {xyz_file}")

print("\n" + "="*60)
print("                    Optimization Complete")
print("="*60)
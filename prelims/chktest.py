import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.emsuite.core as core

def test_resurrect_existing():
    """Test resurrection of existing molecule_alone.chk file."""
    print("\n" + "="*60)
    print("Testing resurrection of existing checkpoint")
    print("="*60)
    
    chkfile = 'molecule_alone.chk'
    
    if not os.path.exists(chkfile):
        print(f"ERROR: {chkfile} not found!")
        return False
    
    print(f"\nAttempting to resurrect: {chkfile}")
    print(f"File size: {os.path.getsize(chkfile) / 1024:.2f} KB")
    
    try:
        # Resurrect the molecule
        mf = core.resurrect_mol(chkfile)
        
        # Check results
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Object type:     {type(mf).__name__}")
        print(f"Energy:          {mf.e_tot:.10f} Ha")
        print(f"Converged:       {mf.converged}")
        print(f"Is GPU:          {hasattr(mf, 'to_cpu')}")
        print(f"Is DFT:          {hasattr(mf, 'xc')}")
        if hasattr(mf, 'xc'):
            print(f"Functional:      {mf.xc}")
        print(f"Spin:            {mf.mol.spin}")
        print(f"Charge:          {mf.mol.charge}")
        print("="*60)
        
        # Validate
        success = mf.converged and mf.e_tot < 0
        print(f"\n{'✓ PASS' if success else '✗ FAIL'}")
        
        return success
        
    except Exception as e:
        print(f"\n✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    gpu_available = core.GPU_AVAILABLE and core.check_gpu_info() > 0
    print(f"GPU: {gpu_available}\n")
    
    success = test_resurrect_existing()
    
    sys.exit(0 if success else 1)






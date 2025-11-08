import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import src.emsuite.core as core
import numpy as np

def create_test_molecule():
    """Create a simple water molecule for testing."""
    xyz_file = 'test_water.xyz'
    with open(xyz_file, 'w') as f:
        f.write("3\nWater molecule\n")
        f.write("O  0.000000  0.000000  0.117176\n")
        f.write("H  0.000000  0.755453 -0.468706\n")
        f.write("H  0.000000 -0.755453 -0.468706\n")
    return xyz_file

def check_td_converged(td):
    """Helper to check if TD calculation converged (handles CuPy arrays)."""
    if td is None:
        return False
    
    converged = td.converged
    
    # Handle different types
    if hasattr(converged, 'get'):  # CuPy array
        converged = converged.get()
    
    # If it's an array, check if all elements are True
    if hasattr(converged, '__iter__') and not isinstance(converged, (str, bytes)):
        try:
            return bool(np.all(converged))
        except:
            return bool(converged)
    
    # Scalar case
    return bool(converged)

def test_single_gpu_td():
    """Test TD calculation with CUDA_VISIBLE_DEVICES set to single GPU."""
    print("\n" + "="*70)
    print("TEST 1: Single GPU TD Calculation")
    print("="*70)
    
    # Force single GPU
    original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    try:
        xyz_file = create_test_molecule()
        
        # Create molecule
        print("\n1. Creating molecule on GPU 0...")
        mf = core.create_molecule_object(
            atom_input=xyz_file,
            basis_set='sto-3g',
            method='dft',
            functional='b3lyp',
            original_charge=0,
            charge_change=0,
            gpu=True,
            spin_guesses=[0]
        )
        
        if mf is None:
            print("✗ FAIL: Molecule creation failed")
            return False
        
        print(f"   Ground state energy: {mf.e_tot:.10f} Ha")
        
        # Save checkpoint
        print("\n2. Saving checkpoint...")
        chkfile = 'test_single_gpu.chk'
        core.save_chkfile(mf, chkfile, functional='b3lyp')
        
        # Create TD
        print("\n3. Running TDDFT (should use current process)...")
        td = core.create_td_molecule_object(mf, nstates=2, triplet=False, force_single_gpu=False)
        
        if not check_td_converged(td):
            print("✗ FAIL: TDDFT failed to converge")
            return False
        
        print(f"   Excited states: {len(td.e)}")
        
        # Handle CuPy arrays for display
        e_vals = td.e.get() if hasattr(td.e, 'get') else td.e
        print(f"   Excitation energies (eV): {e_vals * 27.2114}")
        
        # Cleanup
        if os.path.exists(chkfile):
            os.remove(chkfile)
        if os.path.exists(xyz_file):
            os.remove(xyz_file)
        
        print("\n✓ PASS: Single GPU TD test")
        return True
        
    except Exception as e:
        print(f"\n✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original CUDA setting
        if original_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)


def test_multi_gpu_td():
    """Test TD calculation with multiple GPUs visible (should use subprocess)."""
    print("\n" + "="*70)
    print("TEST 2: Multi-GPU TD Calculation (Subprocess Mode)")
    print("="*70)
    
    # Check available GPUs
    gpu_count = core.check_gpu_info()
    if gpu_count < 2:
        print("⚠ SKIP: Need at least 2 GPUs for this test")
        return None
    
    # Set multiple GPUs visible
    original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    try:
        xyz_file = create_test_molecule()
        
        # Create molecule
        print(f"\n1. Creating molecule with GPUs {os.environ['CUDA_VISIBLE_DEVICES']} visible...")
        mf = core.create_molecule_object(
            atom_input=xyz_file,
            basis_set='sto-3g',
            method='dft',
            functional='b3lyp',
            original_charge=0,
            charge_change=0,
            gpu=True,
            spin_guesses=[0]
        )
        
        if mf is None:
            print("✗ FAIL: Molecule creation failed")
            return False
        
        # Get energy (handle CuPy)
        e_tot = mf.e_tot.get() if hasattr(mf.e_tot, 'get') else mf.e_tot
        print(f"   Ground state energy: {e_tot:.10f} Ha")
        
        # Save checkpoint
        print("\n2. Saving checkpoint...")
        chkfile = 'test_multi_gpu.chk'
        core.save_chkfile(mf, chkfile, functional='b3lyp')
        
        # Verify checkpoint has functional
        from pyscf import lib
        try:
            xc = lib.chkfile.load(chkfile, 'scf/xc')
            xc_str = xc.decode('utf-8') if isinstance(xc, bytes) else str(xc)
            print(f"   ✓ Checkpoint has functional: {xc_str}")
        except Exception as e:
            print(f"   ✗ WARNING: Could not verify functional in checkpoint: {e}")
        
        # Create TD (should trigger subprocess)
        print("\n3. Running TDDFT (should use subprocess on GPU 0)...")
        td = core.create_td_molecule_object(mf, nstates=2, triplet=False, force_single_gpu=False)
        
        if not check_td_converged(td):
            print("✗ FAIL: TDDFT failed to converge")
            return False
        
        print(f"   ✓ Excited states calculated: {len(td.e)}")
        
        # Handle CuPy arrays
        e_vals = td.e.get() if hasattr(td.e, 'get') else td.e
        print(f"   Excitation energies (eV): {e_vals * 27.2114}")
        
        # Verify checkpoint still has functional (test backup/restore)
        print("\n4. Verifying checkpoint integrity after TDDFT...")
        try:
            xc_after = lib.chkfile.load(chkfile, 'scf/xc')
            xc_after_str = xc_after.decode('utf-8') if isinstance(xc_after, bytes) else str(xc_after)
            print(f"   ✓ Checkpoint still has functional: {xc_after_str}")
            
            if xc_after_str != xc_str:
                print(f"   ✗ WARNING: Functional changed from {xc_str} to {xc_after_str}")
                return False
        except Exception as e:
            print(f"   ✗ FAIL: Checkpoint corrupted after TDDFT: {e}")
            return False
        
        # Test resurrection after TDDFT
        print("\n5. Testing resurrection after TDDFT...")
        mf_resurrected = core.resurrect_mol(chkfile)
        
        if not hasattr(mf_resurrected, 'xc'):
            print(f"   ✗ FAIL: Resurrected object lost DFT functional!")
            return False
        
        print(f"   ✓ Resurrected with functional: {mf_resurrected.xc}")
        
        # Handle CuPy for energy comparison
        e_tot_res = mf_resurrected.e_tot.get() if hasattr(mf_resurrected.e_tot, 'get') else mf_resurrected.e_tot
        energy_diff = abs(e_tot - e_tot_res)
        print(f"   Energy difference: {energy_diff:.2e} Ha")
        
        if energy_diff > 1e-8:
            print(f"   ✗ FAIL: Energy mismatch!")
            return False
        
        # Cleanup
        if os.path.exists(chkfile):
            os.remove(chkfile)
        if os.path.exists(xyz_file):
            os.remove(xyz_file)
        
        print("\n✓ PASS: Multi-GPU TD test with checkpoint preservation")
        return True
        
    except Exception as e:
        print(f"\n✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Restore original CUDA setting
        if original_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)


def test_sequential_multi_gpu_td():
    """Test multiple sequential TD calculations (simulates tuning workflow)."""
    print("\n" + "="*70)
    print("TEST 3: Sequential Multi-GPU TD Calculations")
    print("="*70)
    
    gpu_count = core.check_gpu_info()
    if gpu_count < 2:
        print("⚠ SKIP: Need at least 2 GPUs for this test")
        return None
    
    original_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    
    try:
        xyz_file = create_test_molecule()
        
        # Create base molecule
        print("\n1. Creating base molecule...")
        mf = core.create_molecule_object(
            atom_input=xyz_file,
            basis_set='sto-3g',
            method='dft',
            functional='b3lyp',
            original_charge=0,
            charge_change=0,
            gpu=True,
            spin_guesses=[0]
        )
        
        chkfile = 'test_sequential.chk'
        core.save_chkfile(mf, chkfile, functional='b3lyp')
        
        # Run multiple TD calculations in sequence
        num_iterations = 3
        print(f"\n2. Running {num_iterations} sequential TDDFT calculations...")
        
        for i in range(num_iterations):
            print(f"\n   --- Iteration {i+1}/{num_iterations} ---")
            
            # Resurrect molecule
            mf_res = core.resurrect_mol(chkfile)
            
            if not hasattr(mf_res, 'xc'):
                print(f"   ✗ FAIL at iteration {i+1}: Lost functional after resurrection")
                return False
            
            print(f"   Functional: {mf_res.xc}")
            
            # Run TDDFT
            td = core.create_td_molecule_object(mf_res, nstates=2, triplet=False)
            
            if not check_td_converged(td):
                print(f"   ✗ FAIL at iteration {i+1}: TDDFT did not converge")
                return False
            
            print(f"   ✓ Converged with {len(td.e)} states")
            
            # Verify checkpoint still intact
            from pyscf import lib
            xc = lib.chkfile.load(chkfile, 'scf/xc')
            xc_str = xc.decode('utf-8') if isinstance(xc, bytes) else str(xc)
            
            if xc_str.lower() != 'b3lyp':
                print(f"   ✗ FAIL at iteration {i+1}: Checkpoint corrupted (xc={xc_str})")
                return False
        
        print(f"\n✓ PASS: All {num_iterations} iterations preserved checkpoint")
        
        # Cleanup
        if os.path.exists(chkfile):
            os.remove(chkfile)
        if os.path.exists(xyz_file):
            os.remove(xyz_file)
        
        return True
        
    except Exception as e:
        print(f"\n✗ FAIL: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if original_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Multi-GPU TDDFT Test Suite")
    print("="*70)
    
    # Check GPU availability
    gpu_count = core.check_gpu_info()
    print(f"GPUs detected: {gpu_count}\n")
    
    if gpu_count == 0:
        print("⚠ No GPUs available - cannot run tests")
        sys.exit(1)
    
    results = {}
    
    # Test 1: Single GPU
    results['Single GPU'] = test_single_gpu_td()
    
    # Test 2: Multi-GPU subprocess
    if gpu_count >= 2:
        results['Multi-GPU'] = test_multi_gpu_td()
        results['Sequential'] = test_sequential_multi_gpu_td()
    else:
        print("\n⚠ Skipping multi-GPU tests (only 1 GPU available)")
        results['Multi-GPU'] = None
        results['Sequential'] = None
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, result in results.items():
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{test_name:<20}: {status}")
    print("="*70 + "\n")
    
    # Exit code
    passed = [r for r in results.values() if r is True]
    failed = [r for r in results.values() if r is False]
    
    if failed:
        print(f"⚠ {len(failed)} test(s) FAILED")
        sys.exit(1)
    else:
        print(f"✓ All tests PASSED ({len(passed)} tests)")
        sys.exit(0)

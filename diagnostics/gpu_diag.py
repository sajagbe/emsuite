# Quick GPU diagnostic for the emsuite environment
import os, traceback, sys
def print_line(s=""):
    print("="*80)
    print(s)
    print("="*80)

print_line("ENV")
for k in ("CUDA_VISIBLE_DEVICES","CUDA_LAUNCH_BLOCKING"):
    print(k, os.environ.get(k))

print_line("PYTHON / PACKAGE VERSIONS")
try:
    import cupy as cp, numpy as np
    print("cupy", cp.__version__)
    try:
        print("cuda runtime:", cp.cuda.runtime.runtimeGetVersion())
    except Exception as e:
        print("cuda runtime query failed:", e)
except Exception as e:
    print("cupy import failed:", e)
try:
    import pyscf
    print("pyscf", pyscf.__version__)
except Exception as e:
    print("pyscf import failed:", e)
try:
    import gpu4pyscf
    print("gpu4pyscf", getattr(gpu4pyscf, '__version__', 'unknown'))
except Exception as e:
    print("gpu4pyscf import failed:", e)

print_line("DEVICE INFO & SIMPLE KERNEL")
try:
    import cupy as cp
    ndev = cp.cuda.runtime.getDeviceCount()
    print("device count:", ndev)
    for i in range(ndev):
        prop = cp.cuda.runtime.getDeviceProperties(i)
        print(f"device[{i}] name:", prop['name'])
    # simple alloc / compute
    a = cp.ones((2000,2000), dtype=cp.float32)
    b = a.dot(a.T)
    cp.cuda.Stream.null.synchronize()
    print("simple GEMM OK")
except Exception as e:
    print("device/kernel test failed:")
    traceback.print_exc()

# If you set CHKFILE env to the failing checkpoint, try resurrect + TD (captures exception)
# CHK = os.environ.get("CHKFILE")
CHK = os.environ.get("CHKFILE", "/home/users/sajagbe2/Desktop/packages/emsuite/diagnostics/molecule_alone.chk")  # default path (env overrides)
if CHK:
    print_line("TRY RESURRECT & TD ON CHKFILE")
    try:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        import emsuite.core as core
        print("resurrecting", CHK)
        mf = core.resurrect_mol(CHK)
        print("resurrect ok, energy:", getattr(mf, 'e_tot', 'unknown'))
        print("creating TD object (5 states) ...")
        td = core.create_td_molecule_object(mf, nstates=5, triplet=False)
        print("TD OK")
    except Exception as e:
        print("RESURRECT / TD failed:")
        traceback.print_exc()
else:
    print("No CHKFILE env set — to test resurrection set CHKFILE=/path/to/your.chk")
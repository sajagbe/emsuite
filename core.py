import numpy as np
from pyscf import gto, scf, dft, tdscf, qmmm, lib
from pyscf.solvent import smd
from pyscf.geomopt.geometric_solver import optimize



import cupy as cp

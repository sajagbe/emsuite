"""Print per-atom ESP and RESP charges from the GPU code and the CPU port, side by side.

Usage (on a GPU node): python show_charges.py [molecule]   (default: water)
Molecules are the ones in compare_gpu.py. B3LYP/6-31G*.

Three columns per charge type:
  GPU         gpu4pyscf.pop.esp on the GPU SCF density
  CPU same    resp_cpu on the SAME GPU density  (tests the port itself)
  CPU chain   CPU SCF + resp_cpu                (what a CPU-only run would give)
"""

from __future__ import annotations

import sys

import numpy as np
import resp_cpu
from compare_gpu import MOLECULES, build_mol
from gpu4pyscf.pop import esp as gesp
from pyscf import dft


def main(name: str = "water") -> None:
    xyz, charge = MOLECULES[name]
    mol = build_mol(xyz, charge)

    mf_g = dft.RKS(mol, xc="b3lyp").to_gpu()
    mf_g.kernel()
    dm_g = mf_g.make_rdm1()
    dm_gn = dm_g.get()
    mf_c = dft.RKS(mol, xc="b3lyp").run()
    dm_c = mf_c.make_rdm1()

    print(f"{name}  B3LYP/6-31G*  charge {charge}  ({mol.natm} atoms)")
    print(f"SCF energy  GPU {float(mf_g.e_tot):.10f} Ha   CPU {mf_c.e_tot:.10f} Ha")
    print(f"surface points: {len(resp_cpu.vdw_surface(mol, scales=[1.4, 1.6, 1.8, 2.0]))}\n")

    rows = {
        "ESP": (
            np.asarray(gesp.esp_solve(mol, dm_g)),
            resp_cpu.esp_solve(mol, dm_gn),
            resp_cpu.esp_solve(mol, dm_c),
        ),
        "RESP": (
            np.asarray(gesp.resp_solve(mol, dm_g)),
            resp_cpu.resp_solve(mol, dm_gn),
            resp_cpu.resp_solve(mol, dm_c),
        ),
    }
    for kind, (g, cs, cc) in rows.items():
        print(f"{kind} charges (e)")
        print(f"{'atom':<6}{'GPU':>14}{'CPU same':>14}{'CPU chain':>14}{'same-GPU':>12}{'chain-GPU':>12}")
        for i, sym in enumerate(mol.elements):
            print(
                f"{sym}{i + 1:<5}{g[i]:>14.8f}{cs[i]:>14.8f}{cc[i]:>14.8f}"
                f"{cs[i] - g[i]:>12.1e}{cc[i] - g[i]:>12.1e}"
            )
        print(f"{'sum':<6}{g.sum():>14.8f}{cs.sum():>14.8f}{cc.sum():>14.8f}\n")


if __name__ == "__main__":
    main(*sys.argv[1:])

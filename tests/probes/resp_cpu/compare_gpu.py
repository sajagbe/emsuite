"""Compare the CPU RESP port (resp_cpu.py) with gpu4pyscf.pop.esp.

Per molecule (B3LYP/6-31G*, the emsuite tuning level):
  1. Same density, two codes: one GPU SCF; its density goes to both gpu4pyscf
     and the CPU port. Compares surface points, electrostatic potential at every
     point, the fit matrix A and vector B, ESP charges and RESP charges.
  2. Full CPU chain: CPU SCF + CPU port, vs the GPU chain (charges only).

Without a GPU only step 2's CPU half runs (used for the local smoke test).
Usage: python compare_gpu.py [molecule ...]   writes results_<host>.json
"""

from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import resp_cpu
from pyscf import dft, gto

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CAMP = ROOT / "tests/integration_runs/opsins/campaigns/pose-restrain-2026-10-11"
MOLECULES = {  # name: (xyz, total charge)
    "water": (ROOT / "examples/water-gpu/Water.xyz", 0),
    "LF": (ROOT / "tests/integration_runs/recreate-maps-2026-10-07T035624Z/lf-gas/LF.xyz", 0),
    "SB": (CAMP / "gas/SB/separate_singlet/SB.xyz", 0),
    "rPSB": (CAMP / "gas/rPSB/separate_singlet/rPSB.xyz", 1),
}


def build_mol(xyz: Path, charge: int):
    body = xyz.read_text().split("\n", 2)[2]
    return gto.M(atom=body, basis="6-31g*", charge=charge, spin=0, verbose=0)


def gpu_available() -> bool:
    try:
        import cupy

        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:  # noqa: BLE001
        return False


def maxdiff(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def run(name: str, use_gpu: bool) -> dict:
    xyz, charge = MOLECULES[name]
    mol = build_mol(xyz, charge)
    rec = {"molecule": name, "natm": mol.natm, "nao": mol.nao, "charge": charge}

    # Full CPU chain
    t = time.time()
    mf_c = dft.RKS(mol, xc="b3lyp").run()
    rec["cpu_scf_s"] = round(time.time() - t, 1)
    rec["cpu_e_tot"] = mf_c.e_tot
    dm_c = mf_c.make_rdm1()
    t = time.time()
    q_esp_c = resp_cpu.esp_solve(mol, dm_c)
    q_resp_c = resp_cpu.resp_solve(mol, dm_c)
    rec["cpu_resp_s"] = round(time.time() - t, 2)
    rec["cpu_chain_esp"] = q_esp_c.tolist()
    rec["cpu_chain_resp"] = q_resp_c.tolist()
    rec["cpu_chain_resp_sum"] = float(q_resp_c.sum())

    if not use_gpu:
        return rec

    from gpu4pyscf.gto.int3c1e import int1e_grids
    from gpu4pyscf.pop import esp as gesp

    t = time.time()
    mf_g = dft.RKS(mol, xc="b3lyp").to_gpu()
    mf_g.kernel()
    rec["gpu_scf_s"] = round(time.time() - t, 1)
    rec["gpu_e_tot"] = float(mf_g.e_tot)
    dm_g = mf_g.make_rdm1()
    dm_np = dm_g.get()

    # 1. same density, two codes
    pts_g = gesp.vdw_surface(mol, scales=[1.4, 1.6, 1.8, 2.0])
    pts_c = resp_cpu.vdw_surface(mol, scales=[1.4, 1.6, 1.8, 2.0])
    rec["n_points"] = [len(pts_g), len(pts_c)]
    rec["points_maxdiff_bohr"] = maxdiff(pts_g, pts_c)
    ve_g = int1e_grids(mol, pts_g, dm=dm_g, direct_scf_tol=1e-14).get()
    ve_c = resp_cpu.electron_potential(mol, dm_np, pts_c)
    rec["v_electron_maxdiff_au"] = maxdiff(ve_g, ve_c)
    rec["v_electron_max_au"] = float(np.max(np.abs(ve_c)))
    A_g, B_g = (x.get() for x in gesp.build_ab(mol, dm_g))
    A_c, B_c = resp_cpu.build_ab(mol, dm_np)
    rec["A_maxdiff"] = maxdiff(A_g, A_c)
    rec["B_maxdiff"] = maxdiff(B_g, B_c)
    t = time.time()
    q_esp_g = gesp.esp_solve(mol, dm_g)
    q_resp_g = gesp.resp_solve(mol, dm_g)
    rec["gpu_resp_s"] = round(time.time() - t, 2)
    q_esp_same = resp_cpu.esp_solve(mol, dm_np)
    q_resp_same = resp_cpu.resp_solve(mol, dm_np)
    rec["same_dm_esp_maxdiff"] = maxdiff(q_esp_g, q_esp_same)
    rec["same_dm_resp_maxdiff"] = maxdiff(q_resp_g, q_resp_same)
    rec["gpu_resp"] = np.asarray(q_resp_g).tolist()

    # 2. full chains (includes CPU vs GPU SCF difference)
    rec["e_tot_diff_ha"] = rec["gpu_e_tot"] - rec["cpu_e_tot"]
    rec["chain_esp_maxdiff"] = maxdiff(q_esp_g, q_esp_c)
    rec["chain_resp_maxdiff"] = maxdiff(q_resp_g, q_resp_c)
    return rec


def main() -> None:
    names = sys.argv[1:] or list(MOLECULES)
    use_gpu = gpu_available()
    print(f"host={socket.gethostname()} gpu={use_gpu} molecules={names}", flush=True)
    out = []
    for name in names:
        rec = run(name, use_gpu)
        out.append(rec)
        keys = [
            k
            for k in rec
            if k.endswith(("maxdiff", "_s", "n_points", "diff_ha", "resp_sum"))
            or k.endswith("maxdiff_au")
            or k.endswith("maxdiff_bohr")
        ]
        print(name, {k: rec[k] for k in keys}, flush=True)
    dest = HERE / f"results_{socket.gethostname()}.json"
    dest.write_text(json.dumps(out, indent=1))
    print("wrote", dest)


if __name__ == "__main__":
    main()

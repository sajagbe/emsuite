#!/usr/bin/env python3
"""Full optional-stack smoke: PySCF 2.14 + gpu4pyscf ≥1.8.1 + osc CPU + combined pin.

Runs under repo-local ``.venv`` (see PYSCF214_ENV_NOTES.md); does not touch ``~/.local``.

Checks:
  1. Package versions (numpy / pyscf / gpu4pyscf)
  2. ``lib.einsum`` NumPy≥2.4 fix (stock 3-operand OK)
  3. Stock GPU ``td.oscillator_strength()`` vs ``oscillator_strength_cpu`` identity
  4. Combined methane tuning with fake multi-GPU env → pin + finite exe/osc

Usage::

    PYTHONPATH=src .venv/bin/python tests/manual/integrated_pyscf214_stack.py
    sbatch tests/manual/integrated_pyscf214_stack.slurm
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Water TDDFT osc from probe job 4572332 (PySCF 2.14 + gpu4pyscf 1.8.1, V100)
_WATER_OSC_REF = np.array([1.63466173e-02, 5.37305364e-06, 8.17358774e-02])
_WATER_E_REF = np.array([0.29950995, 0.37429867, 0.3800568])


def _banner(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def check_versions() -> dict:
    _banner("1) Versions (optional .venv stack)")
    import gpu4pyscf
    import pyscf

    info = {
        "python": sys.executable,
        "numpy": np.__version__,
        "pyscf": pyscf.__version__,
        "pyscf_file": pyscf.__file__,
        "gpu4pyscf": getattr(gpu4pyscf, "__version__", "?"),
        "gpu4pyscf_file": gpu4pyscf.__file__,
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    assert "2.14" in str(info["pyscf"]), f"expected pyscf 2.14.x, got {info['pyscf']}"
    # Require paired GPU package from repo .venv (≥1.8.1), not live ~/.local 1.4.3
    assert ".venv" in str(info["gpu4pyscf_file"]), (
        f"gpu4pyscf must come from repo .venv, got {info['gpu4pyscf_file']}"
    )
    ver = str(info["gpu4pyscf"])
    parts = [int(x) for x in ver.split(".")[:3] if x.isdigit()]
    assert parts >= [1, 8, 1], f"expected gpu4pyscf ≥1.8.1, got {ver}"
    print("PASS: paired PySCF 2.14 + gpu4pyscf ≥1.8.1 from .venv")
    return info


def check_einsum() -> None:
    _banner("2) lib.einsum NumPy≥2.4 fix")
    from pyscf import lib

    lib.einsum("ij,jk,kl->ik", np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 5)))
    a = np.zeros((2, 3, 4))
    b = np.zeros((5, 4))
    c = np.zeros((5, 3))
    lib.einsum("xov,pv,qo->xpq", a, b, c)
    print("PASS: stock pyscf.lib.einsum 3-op contractions OK")


def check_osc_identity() -> dict:
    _banner("3) Stock GPU osc vs oscillator_strength_cpu")
    from pyscf import dft, gto

    from emsuite.core.oscillator_strength import oscillator_strength_cpu

    mol = gto.M(atom="O 0 0 0; H 0 0 0.95; H 0.89 0 -0.32", basis="6-31g*", verbose=0)
    mf = dft.RKS(mol)
    mf.xc = "b3lyp"
    mf_g = mf.to_gpu()
    mf_g.kernel()
    td = mf_g.TDDFT()
    td.nstates = 3
    td.singlet = True
    td.kernel()

    e = np.asarray(td.e.get() if hasattr(td.e, "get") else td.e, dtype=float)
    osc_direct = td.oscillator_strength()
    if hasattr(osc_direct, "get"):
        osc_direct = osc_direct.get()
    osc_direct = np.asarray(osc_direct, dtype=float)
    osc_cpu = oscillator_strength_cpu(td)

    rel = float(np.max(np.abs(osc_cpu - osc_direct) / np.maximum(np.abs(osc_direct), 1e-16)))
    e_rel = float(np.max(np.abs(e - _WATER_E_REF) / np.maximum(np.abs(_WATER_E_REF), 1e-16)))
    osc_ref_rel = float(
        np.max(np.abs(osc_direct - _WATER_OSC_REF) / np.maximum(np.abs(_WATER_OSC_REF), 1e-16))
    )
    print(f"  E:          {e}")
    print(f"  osc direct: {osc_direct}")
    print(f"  osc cpu:    {osc_cpu}")
    print(f"  rel(direct vs cpu) = {rel:.3e}")
    print(f"  rel(E vs prior probe 4572332) = {e_rel:.3e}")
    print(f"  rel(osc vs prior probe 4572332) = {osc_ref_rel:.3e}")
    assert rel < 1e-10, f"stock vs cpu helper diverge: {rel}"
    # Backtest tolerance: same molecule/basis/functional; allow mild node/lib noise.
    assert e_rel < 1e-6, f"energy drifted vs prior probe: {e_rel}"
    assert osc_ref_rel < 1e-5, f"osc drifted vs prior probe: {osc_ref_rel}"
    print("PASS: stock GPU osc restored; matches CPU helper + prior probe")
    return {"e": e.tolist(), "osc": osc_direct.tolist(), "rel_direct_cpu": rel}


def check_combined_pin() -> dict:
    _banner("4) Combined methane exe/osc with fake multi-GPU pin")
    import importlib.util

    from emsuite.inputs import TuningInput

    helpers_path = _ROOT / "tests" / "integration" / "helpers.py"
    spec = importlib.util.spec_from_file_location("emsuite_int_helpers", helpers_path)
    assert spec and spec.loader
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    fingerprint_csv = helpers.fingerprint_csv
    install_methane_surf = helpers.install_methane_surf
    latest_results_dir = helpers.latest_results_dir
    write_methane_xyz = helpers.write_methane_xyz

    real = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0"
    # Fake multi-GPU so _pin_combined_to_single_gpu must collapse the list.
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{real},{real}"
    print(f"  fake multi-GPU CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r}")

    with tempfile.TemporaryDirectory(prefix="emsuite_int_") as tmp:
        tmp_path = Path(tmp)
        os.chdir(tmp_path)
        write_methane_xyz(tmp_path)
        surf = install_methane_surf(tmp_path)
        TuningInput(
            molecule="methane.xyz",
            surface_file=str(surf),
            properties=("exe", "osc"),
            basis_set="sto-3g",
            method="dft",
            functional="b3lyp",
            calc_type="combined",
            parallel=False,
            state_of_interest=2,
            triplet=False,
        ).run()

        pinned = os.environ["CUDA_VISIBLE_DEVICES"]
        assert pinned == real, f"expected pin to {real!r}, got {pinned!r}"
        results_dir = latest_results_dir(tmp_path)
        summary = results_dir / "methane_tuning_summary.csv"
        assert summary.is_file()
        text = summary.read_text()
        assert "s1_exe" in text or "s1_osc" in text
        # Combined TD emits per-state files: methane_s1_exe.mol2, ...
        for name in ("s1_exe", "s1_osc", "s2_exe", "s2_osc"):
            assert (results_dir / f"methane_{name}.mol2").is_file(), name
        fp = fingerprint_csv(summary)
        print(f"  pinned CUDA_VISIBLE_DEVICES={pinned!r}")
        print(f"  summary fingerprint: {fp}")
        print("PASS: combined pin + exe/osc completed")
        return {"pinned_cuda": pinned, "summary": fp, "results_dir": str(results_dir)}


def main() -> int:
    print(f"cwd={os.getcwd()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}")
    try:
        check_versions()
        check_einsum()
        osc_info = check_osc_identity()
        combined_info = check_combined_pin()
    except Exception:
        traceback.print_exc()
        print("\nFAIL: integrated stack smoke aborted")
        return 1

    _banner("SUMMARY")
    print("All checks passed on isolated .venv stack.")
    print(f"  osc identity rel: {osc_info['rel_direct_cpu']:.3e}")
    print(f"  combined pin:     {combined_info['pinned_cuda']}")
    print(f"  combined summary: {combined_info['summary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

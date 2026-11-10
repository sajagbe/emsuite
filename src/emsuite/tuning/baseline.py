"""Neutral, anion and cation starting SCFs, run at the same time, one GPU each.

Each state runs in its own Python process with ``CUDA_VISIBLE_DEVICES`` set to a
single GPU, so gpu4pyscf never splits one SCF across several GPUs. Each process
runs the usual spin-state search (``core.create_molecule_object``, including the
CPU energy check) and writes the usual checkpoint file. With fewer GPUs than
states, states share GPUs and run one after another on them.

Timing (tests/probes/multigpu_scf, V100): neutral + anion + cation of rPSB took
~110 s one after another on one GPU and ~53 s at the same time on three GPUs,
with identical energies.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CHKFILES = {
    "neutral": "molecule_alone.chk",
    "anion": "anion_alone.chk",
    "cation": "cation_alone.chk",
}
CHARGE_CHANGE = {"neutral": 0, "anion": -1, "cation": +1}


def visible_gpus() -> list[str]:
    """GPU ids this job may use, read without initialising CUDA in this process."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        return [d.strip() for d in raw.split(",") if d.strip() not in ("", "-1")]
    out = subprocess.run(
        [sys.executable, "-c", "import cupy; print(cupy.cuda.runtime.getDeviceCount())"],
        capture_output=True,
        text=True,
    )
    try:
        return [str(i) for i in range(int(out.stdout.strip()))]
    except ValueError:
        return []


def assign_states(states: list[str], gpus: list[str]) -> dict[str, list[str]]:
    """Map each GPU to the states it runs (one each if enough GPUs, else round robin)."""
    plan: dict[str, list[str]] = {}
    for i, state in enumerate(states):
        plan.setdefault(gpus[i % len(gpus)], []).append(state)
    return plan


def run_states(
    input_data: str,
    basis_set: str,
    method: str,
    functional: str,
    charge: int,
    neutral_spin_guesses,
    states: list[str],
    gpus: list[str],
    workdir: str = ".",
    max_energy_delta: float = 1.0,
) -> dict[str, dict]:
    """Run the requested states at the same time; return {state: result} where result
    has ``chkfile`` (absolute path), ``e_tot`` and ``spin``. Raises RuntimeError naming
    every state without a trustworthy solution."""
    work = Path(workdir).resolve()
    for state in states:  # never pick up results from an earlier run
        for stale in (work / f"baseline_{state}.json", work / CHKFILES[state]):
            stale.unlink(missing_ok=True)
    procs = []
    for gpu, gpu_states in assign_states(states, gpus).items():
        spec = {
            "input_data": str(Path(input_data).resolve()),
            "basis_set": basis_set,
            "method": method,
            "functional": functional,
            "charge": charge,
            "neutral_spin_guesses": neutral_spin_guesses,
            "states": gpu_states,
            "workdir": str(work),
        }
        log = work / f"baseline_gpu{gpu}.log"
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        print(f"Starting {', '.join(gpu_states)} on GPU {gpu} (log: {log.name})")
        with open(log, "w") as fh:
            p = subprocess.Popen(
                [sys.executable, "-m", "emsuite.tuning.baseline", json.dumps(spec)],
                env=env,
                stdout=fh,
                stderr=subprocess.STDOUT,
                cwd=work,
            )
        procs.append((p, log, gpu_states))

    for p, log, gpu_states in procs:
        p.wait()
        print(f"\n----- {', '.join(gpu_states)} (exit {p.returncode}) -----")
        print(log.read_text(errors="replace"))

    results = {}
    for state in states:
        f = work / f"baseline_{state}.json"
        results[state] = json.loads(f.read_text()) if f.exists() else {"e_tot": None}

    failed = [s for s in states if results[s].get("e_tot") is None]
    neutral = results.get("neutral", {}).get("e_tot")
    if neutral is not None:
        for s in ("anion", "cation"):
            e = results.get(s, {}).get("e_tot")
            if e is not None and abs(e - neutral) > max_energy_delta:
                print(
                    f"{s} REJECTED: E = {e:.6f} Ha is {abs(e - neutral):.3f} Ha from the "
                    f"neutral ({neutral:.6f} Ha); limit {max_energy_delta} Ha"
                )
                failed.append(s)
    if failed:
        raise RuntimeError(
            f"No trustworthy SCF solution for the {', '.join(failed)} state(s) of {input_data}: "
            "every spin state either did not converge or was rejected (see the 'REJECTED' "
            "lines above). The properties that need these states cannot be computed."
        )
    return results


def _worker(spec: dict) -> None:
    """Run the given states one after another on this process's single GPU."""
    from emsuite import core

    os.chdir(spec["workdir"])
    for state in spec["states"]:
        print(f"===== {state} =====", flush=True)
        mf = core.create_molecule_object(
            atom_input=spec["input_data"],
            basis_set=spec["basis_set"],
            method=spec["method"],
            functional=spec["functional"],
            original_charge=spec["charge"],
            charge_change=CHARGE_CHANGE[state],
            gpu=True,
            spin_guesses=spec["neutral_spin_guesses"] if state == "neutral" else None,
        )
        out = Path(f"baseline_{state}.json")
        if mf is None:
            out.write_text(json.dumps({"e_tot": None}))
            continue
        core.save_chkfile(mf, CHKFILES[state], spec["functional"])
        out.write_text(
            json.dumps(
                {
                    "chkfile": str(Path(CHKFILES[state]).resolve()),
                    "e_tot": float(mf.e_tot),
                    "spin": int(mf.mol.spin),
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                }
            )
        )


if __name__ == "__main__":
    _worker(json.loads(sys.argv[1]))

"""Run production RESP auto/GPU against CPU; requires one allocated visible GPU."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    assert visible and len(visible.split(",")) == 1, "Allocate one visible GPU before imports"
    import numpy as np

    from emsuite.potential.resp import resp_charges_from_atoms

    reference = json.loads(Path(__file__).with_name("reference_results.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for case in reference["production_validation_cases"]:
        settings = {key: case[key] for key in ("atoms", "charge", "spin", "method", "basis")}
        gpu, cpu = {}, {}
        qgpu = resp_charges_from_atoms(**settings, backend="auto", diagnostics=gpu)
        assert gpu["backend"] == "gpu", "Automatic selection fell back: GPU validation cannot pass"
        qcpu = resp_charges_from_atoms(**settings, backend="cpu", diagnostics=cpu)
        checks = dict(
            energy_abs_hartree=abs(gpu["energy_hartree"] - cpu["energy_hartree"]),
            max_charge_difference_e=float(np.max(abs(qgpu - qcpu))),
            total_charge_error_e=float(abs(qgpu.sum() - case["charge"])),
        )
        gates = dict(
            energy_abs_hartree=1e-7 if case["method"] == "hf" else 1e-6,
            max_charge_difference_e=1e-4,
            total_charge_error_e=1e-8,
        )
        result = dict(
            case=case,
            gpu=gpu,
            cpu=cpu,
            checks=checks,
            gates=gates,
            passed=all(checks[key] <= gate for key, gate in gates.items()),
            gpu_charges=qgpu.tolist(),
            cpu_charges=qcpu.tolist(),
        )
        (args.output / f"{case['name']}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        results.append(result)
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    assert all(result["passed"] for result in results), "Production CPU/GPU gates failed"


if __name__ == "__main__":
    main()

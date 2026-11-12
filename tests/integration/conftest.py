"""Traceable integration test session hooks."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

INTEGRATION_FEATURE_TRACE: dict[str, dict[str, object]] = {
    "tests/integration/test_surface_smiles.py::test_surface_smiles_generates_surf": {
        "feature_id": "v1.1-surface-smiles",
        "version": "1.1.0",
        "channel": "surface",
        "summary": "SMILES → UFF optimize → VDW .surf generation",
        "code_paths": ["src/emsuite/surface/"],
    },
    "tests/integration/test_smoke_methane.py::test_methane_surface_to_tuning_smoke": {
        "feature_id": "v1.1-tuning-smoke",
        "version": "1.1.0",
        "channel": "tuning",
        "summary": "End-to-end surface → serial homo/lumo/gap tuning on methane",
        "code_paths": ["src/emsuite/surface/", "src/emsuite/tuning/runner.py"],
    },
    "tests/integration/test_potential_apbs.py::test_potential_apbs_map": {
        "feature_id": "v1.3-potential-apbs",
        "version": "1.3.0",
        "channel": "potential",
        "summary": "APBS potential map on VDW surface",
        "code_paths": ["src/emsuite/potential/runner.py", "src/emsuite/potential/apbs.py"],
    },
    "tests/integration/test_coupled_smoke.py::test_coupled_pipeline": {
        "feature_id": "v1.1-coupled-pipeline",
        "version": "1.1.0",
        "channel": "coupled",
        "summary": "Potential → tuning coupled channel smoke test",
        "code_paths": ["src/emsuite/coupled/runner.py"],
    },
    "tests/integration/test_consistency_matrix.py::test_surface_heterogenous_generates_edit_header": {
        "feature_id": "consistency-surface-heterogenous",
        "version": "1.6.0",
        "channel": "surface",
        "summary": "Heterogeneous VDW surface with editable per-point charges",
        "code_paths": ["src/emsuite/surface/generate.py", "src/emsuite/surface/io.py"],
    },
    "tests/integration/test_consistency_matrix.py::test_potential_apbs_gauss_charge": {
        "feature_id": "consistency-potential-charge",
        "version": "1.6.0",
        "channel": "potential",
        "summary": "CPU APBS Gauss-law charge map on methane VDW surface",
        "code_paths": ["src/emsuite/potential/runner.py", "src/emsuite/potential/gauss.py"],
    },
    "tests/integration/test_consistency_matrix.py::test_tuning_combined_methane": {
        "feature_id": "consistency-tuning-combined",
        "version": "1.6.0",
        "channel": "tuning",
        "summary": "Standalone combined-mode tuning on methane",
        "code_paths": ["src/emsuite/tuning/runner.py", "src/emsuite/inputs.py"],
    },
    "tests/integration/test_gpu_channels.py::test_gpu_preflight": {
        "feature_id": "gpu-preflight",
        "version": "1.3.0",
        "channel": "gpu",
        "summary": "CuPy/gpu4pyscf GPU detection on compute node",
        "code_paths": ["src/emsuite/core/hardware.py"],
    },
    "tests/integration/test_gpu_channels.py::test_gpu_surface_uff_smoke": {
        "feature_id": "gpu-surface-uff",
        "version": "1.3.0",
        "channel": "surface",
        "summary": "Surface channel smoke on GPU allocation (UFF)",
        "code_paths": ["src/emsuite/surface/"],
    },
    "tests/integration/test_gpu_channels.py::test_gpu_surface_pyscf_optimize": {
        "feature_id": "gpu-surface-pyscf",
        "version": "1.3.0",
        "channel": "surface",
        "summary": "PySCF geometry optimization on GPU (surface channel)",
        "code_paths": ["src/emsuite/surface/optimize.py", "src/emsuite/core/molecule.py"],
    },
    "tests/integration/test_gpu_channels.py::test_gpu_potential_apbs_potential": {
        "feature_id": "gpu-potential-phi",
        "version": "1.3.0",
        "channel": "potential",
        "summary": "APBS potential map on GPU node",
        "code_paths": ["src/emsuite/potential/runner.py"],
    },
    "tests/integration/test_gpu_channels.py::test_gpu_potential_apbs_gauss_charge": {
        "feature_id": "gpu-potential-charge",
        "version": "1.3.0",
        "channel": "potential",
        "summary": "Gauss-law charges from APBS on GPU node",
        "code_paths": ["src/emsuite/potential/gauss.py"],
    },
    "tests/integration/test_gpu_channels.py::test_gpu_tuning_parallel": {
        "feature_id": "gpu-tuning-parallel",
        "version": "1.3.0",
        "channel": "tuning",
        "summary": "Parallel Ray + gpu4pyscf tuning",
        "code_paths": ["src/emsuite/tuning/runner.py", "src/emsuite/tuning/parallel.py"],
    },
    "tests/integration/test_gpu_channels.py::test_gpu_tuning_combined_exe_osc_pinned": {
        "feature_id": "gpu-tuning-combined-exe-osc-pin",
        "version": "1.6.1",
        "channel": "tuning",
        "summary": "Combined MM+TDDFT exe/osc with single-GPU pin (fake multi-GPU env)",
        "code_paths": [
            "src/emsuite/tuning/runner.py",
            "src/emsuite/core/oscillator_strength.py",
            "src/emsuite/core/excited.py",
        ],
    },
    "tests/integration/test_gpu_channels.py::test_gpu_coupled_parallel": {
        "feature_id": "gpu-coupled-parallel",
        "version": "1.3.0",
        "channel": "coupled",
        "summary": "APBS charges → parallel GPU tuning (coupled)",
        "code_paths": ["src/emsuite/coupled/runner.py", "src/emsuite/inputs.py"],
    },
}


def _safe_dirname(nodeid: str) -> str:
    """Unique per-module test directory (avoids collisions across files)."""
    module = Path(nodeid.split("::")[0]).stem
    test_name = nodeid.split("::")[-1]
    return re.sub(r"[^\w.-]+", "_", f"{module}__{test_name}")


@pytest.fixture(autouse=True)
def _bind_integration_workspace(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    request.node._integration_tmp = tmp_path  # type: ignore[attr-defined]


def pytest_configure(config: pytest.Config) -> None:
    run_dir = os.environ.get("EMSUITE_INTEGRATION_RUN_DIR")
    if run_dir:
        config._emsuite_integration_run_dir = Path(run_dir).resolve()  # type: ignore[attr-defined]
        config._emsuite_integration_run_dir.mkdir(parents=True, exist_ok=True)
    else:
        config._emsuite_integration_run_dir = None  # type: ignore[attr-defined]


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]):
    outcome = yield
    report = outcome.get_result()
    if call.when != "call":
        return

    run_dir: Path | None = getattr(item.config, "_emsuite_integration_run_dir", None)
    if run_dir is None:
        return

    test_dir = run_dir / _safe_dirname(item.nodeid)
    test_dir.mkdir(parents=True, exist_ok=True)

    trace = INTEGRATION_FEATURE_TRACE.get(item.nodeid, {})
    manifest = {
        "nodeid": item.nodeid,
        "outcome": report.outcome,
        "duration_s": round(report.duration, 3),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **trace,
    }
    (test_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    tmp_path: Path | None = getattr(item, "_integration_tmp", None)
    if tmp_path is not None and tmp_path.exists():
        workspace = test_dir / "workspace"
        if workspace.exists():
            shutil.rmtree(workspace)

        def _ignore_artifacts(directory: str, names: list[str]) -> set[str]:
            ignored = set()
            for name in names:
                if name.endswith(".chk"):
                    ignored.add(name)
                if name == "integration_runs" or name == "tests":
                    ignored.add(name)
            return ignored

        shutil.copytree(tmp_path, workspace, ignore=_ignore_artifacts)

        assertions_src = tmp_path / ".integration_assertions.json"
        if assertions_src.is_file():
            shutil.copy2(assertions_src, test_dir / "assertions.json")

        inputs_dir = test_dir / "inputs"
        inputs_dir.mkdir(exist_ok=True)
        for pattern in ("*.in", "*.xyz"):
            for src in tmp_path.glob(pattern):
                shutil.copy2(src, inputs_dir / src.name)

    if report.failed:
        (test_dir / "failure.txt").write_text(str(report.longrepr), encoding="utf-8")


# --- PDB fixture for the potential and coupled channels ---


@pytest.fixture
def pdb_complex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run folder with ``complex.pdb`` (ACE-ALA-NME, which Amber knows, plus an acetate
    ``ACT``, which it does not), ``acetate.xyz`` and ``acetate.surf`` in the same frame."""
    import numpy as np

    from emsuite.physconst import vdw_radius_angstrom
    from emsuite.surface import save_surf

    monkeypatch.chdir(tmp_path)
    source = Path(__file__).parents[1] / "unit" / "data" / "potential" / "ala_acetate.pdb"
    shutil.copy2(source, tmp_path / "complex.pdb")
    atoms = [
        (line[76:78].strip(), [float(line[30:38]), float(line[38:46]), float(line[46:54])])
        for line in source.read_text().splitlines()
        if line.startswith(("ATOM", "HETATM")) and line[17:20] == "ACT"
    ]
    xyz = [f"{len(atoms)}", "acetate"] + [f"{s} {x:.4f} {y:.4f} {z:.4f}" for s, (x, y, z) in atoms]
    (tmp_path / "acetate.xyz").write_text("\n".join(xyz) + "\n")

    # deterministic VDW-surface points: a spiral on each atom's sphere, minus buried points
    centers = np.array([c for _, c in atoms])
    radii = np.array([vdw_radius_angstrom(s) for s, _ in atoms])
    k = np.arange(16) + 0.5
    polar, turn = np.arccos(1 - 2 * k / 16), np.pi * (1 + 5**0.5) * k
    unit = np.column_stack(
        [np.cos(turn) * np.sin(polar), np.sin(turn) * np.sin(polar), np.cos(polar)]
    )
    points = []
    for i, center in enumerate(centers):
        for point in center + radii[i] * unit:
            others = np.delete(np.arange(len(centers)), i)
            if (np.linalg.norm(centers[others] - point, axis=1) >= radii[others]).all():
                points.append(point)
    save_surf(np.array(points), np.zeros(len(points)), str(tmp_path / "acetate.surf"))
    return tmp_path

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
    "tests/integration/test_tuning_separate.py::test_methane_surface_to_tuning_smoke": {
        "feature_id": "v1.1-tuning-separate-alias",
        "version": "1.1.0",
        "channel": "tuning",
        "summary": "Alias entry point for separate-mode tuning smoke test",
        "code_paths": ["tests/integration/test_smoke_methane.py"],
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


# --- pdb2pqr fixture (shared by test_potential_pdb2pqr.py, test_coupled_pdb2pqr.py) ---

# A real, standard 3-residue fragment (ASN-ILE-PHE, no altlocs) from PDB 3HTB
# (T4 lysozyme L99A/M102Q), plus a synthetic methane ligand as a second HETATM
# residue. Small and self-contained so this doesn't depend on external files.
PDB2PQR_PROTEIN_PDB = """\
ATOM     14  N   ASN A   2       8.696 -17.109 -10.221  1.00 13.04           N
ATOM     15  CA  ASN A   2       9.444 -18.214 -10.830  1.00 11.28           C
ATOM     16  C   ASN A   2      10.923 -17.749 -10.851  1.00 11.13           C
ATOM     17  O   ASN A   2      11.262 -16.586 -10.453  1.00  8.10           O
ATOM     18  CB  ASN A   2       8.903 -18.572 -12.242  1.00 11.33           C
ATOM     19  CG  ASN A   2       8.979 -17.384 -13.226  1.00 13.64           C
ATOM     20  OD1 ASN A   2      10.036 -16.854 -13.455  1.00  9.75           O
ATOM     21  ND2 ASN A   2       7.826 -16.975 -13.804  1.00 12.13           N
ATOM     22  N   ILE A   3      11.803 -18.653 -11.255  1.00  9.44           N
ATOM     23  CA  ILE A   3      13.215 -18.370 -11.247  1.00  9.32           C
ATOM     24  C   ILE A   3      13.649 -17.195 -12.132  1.00  7.70           C
ATOM     25  O   ILE A   3      14.597 -16.489 -11.782  1.00  9.80           O
ATOM     26  CB  ILE A   3      14.010 -19.669 -11.653  1.00  9.07           C
ATOM     27  CG1 ILE A   3      15.522 -19.542 -11.375  1.00  8.89           C
ATOM     28  CG2 ILE A   3      13.701 -20.044 -13.127  1.00  9.00           C
ATOM     29  CD1 ILE A   3      16.021 -18.938 -10.040  1.00 10.83           C
ATOM     30  N   PHE A   4      12.976 -16.982 -13.265  1.00  8.60           N
ATOM     31  CA  PHE A   4      13.313 -15.827 -14.061  1.00  8.69           C
ATOM     32  C   PHE A   4      12.939 -14.559 -13.370  1.00  9.89           C
ATOM     33  O   PHE A   4      13.695 -13.598 -13.393  1.00  9.06           O
ATOM     34  CB  PHE A   4      12.680 -15.895 -15.481  1.00  8.95           C
ATOM     35  CG  PHE A   4      13.207 -17.104 -16.277  1.00 11.87           C
ATOM     36  CD1 PHE A   4      12.684 -18.381 -16.087  1.00 12.45           C
ATOM     37  CD2 PHE A   4      14.252 -16.959 -17.165  1.00 10.91           C
ATOM     38  CE1 PHE A   4      13.203 -19.507 -16.757  1.00 10.97           C
ATOM     39  CE2 PHE A   4      14.737 -18.073 -17.875  1.00 13.68           C
ATOM     40  CZ  PHE A   4      14.194 -19.334 -17.662  1.00 13.03           C
HETATM    1  C1  MTH A 200      50.000  50.000  50.000  1.00  0.00           C
HETATM    2  H1  MTH A 200      50.630  50.630  50.630  1.00  0.00           H
HETATM    3  H2  MTH A 200      49.370  49.370  50.630  1.00  0.00           H
HETATM    4  H3  MTH A 200      49.370  50.630  49.370  1.00  0.00           H
HETATM    5  H4  MTH A 200      50.630  49.370  49.370  1.00  0.00           H
END
"""

# Gasteiger charges for methane (sum ~0), computed once via `obabel --partialcharge
# gasteiger` on the HETATM block above — real, unique atom names, matching MTH's
# own PDB atom names so pdb2pqr's --ligand atom-name matching succeeds.
PDB2PQR_METHANE_MOL2 = """\
@<TRIPOS>MOLECULE
methane
 5 4 0 0 0
SMALL
GASTEIGER

@<TRIPOS>ATOM
      1  C1        50.0000   50.0000   50.0000 C.3   200  MTH200     -0.0776
      2  H1        50.6300   50.6300   50.6300 H     200  MTH200      0.0194
      3  H2        49.3700   49.3700   50.6300 H     200  MTH200      0.0194
      4  H3        49.3700   50.6300   49.3700 H     200  MTH200      0.0194
      5  H4        50.6300   49.3700   49.3700 H     200  MTH200      0.0194
@<TRIPOS>BOND
     1     4     1    1
     2     5     1    1
     3     1     2    1
     4     1     3    1
"""

# molecule= just needs to be a valid XYZ (surface-generation + box purposes);
# unrelated to the pdb2pqr ligand (MTH), which is entirely inside the PDB.
PDB2PQR_LIGAND_XYZ = """\
1
probe
C 20.000 -17.000 -12.000
"""


@pytest.fixture
def pdb2pqr_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "complex.pdb").write_text(PDB2PQR_PROTEIN_PDB)
    (tmp_path / "methane.mol2").write_text(PDB2PQR_METHANE_MOL2)
    (tmp_path / "ligand.xyz").write_text(PDB2PQR_LIGAND_XYZ)
    return tmp_path

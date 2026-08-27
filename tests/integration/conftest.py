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
        (test_dir / "failure.txt").write_text(str(report.longrepr))

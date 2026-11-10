"""Batch-safe resume decisions and preservation of previous runs."""

import json
from pathlib import Path

import pytest

from emsuite.inputs import TuningInput
from emsuite.tuning.logging import setup_logs_directory
from emsuite.tuning.resume import (
    capture_resume_inputs,
    create_resume_metadata,
    select_resume_run,
    validate_resume_compatibility,
)


@pytest.fixture
def run_inputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("interactive prompt called"))
    Path("ligand.xyz").write_text("1\nligand\nH 0 0 0\n")
    Path("ligand.surf").write_text("x y z q\n1 0 0 0.1\n")
    params = TuningInput(molecule="ligand.xyz", surface_file="ligand.surf").to_dict()
    snapshot = capture_resume_inputs(params, ["lumo", "homo"], 2)
    folder = setup_logs_directory()
    create_resume_metadata(folder, "separate", 2, ["homo", "lumo"], snapshot)
    Path(folder, "point_0000.out").write_text("existing result")
    return params, snapshot, folder


def contents(folder):
    return {str(p): p.read_bytes() for p in Path(folder).iterdir()}


def test_identical_inputs_resume_without_modifying_old_folder(run_inputs):
    _, snapshot, folder = run_inputs
    before = contents(folder)
    selected = select_resume_run(True, snapshot)
    assert selected[0] == folder
    assert contents(folder) == before


def test_disabled_resume_preserves_old_folder_and_creates_new_one(run_inputs):
    _, snapshot, folder = run_inputs
    before = contents(folder)
    assert select_resume_run(False, snapshot) is None
    new_folder = setup_logs_directory()
    assert new_folder != folder
    assert contents(folder) == before


@pytest.mark.parametrize(
    "field",
    [
        "basis_set",
        "method",
        "functional",
        "charge",
        "spin",
        "solvent",
        "state_of_interest",
        "triplet",
        "calc_type",
        "properties",
        "total_points",
        "molecule",
        "surface_file",
    ],
)
def test_changed_inputs_start_fresh(run_inputs, field, capsys):
    params, snapshot, folder = run_inputs
    before = contents(folder)
    changed = dict(snapshot)
    if field in ("molecule", "surface_file"):
        with open(params[field], "a") as handle:
            handle.write("changed content\n")
        changed = capture_resume_inputs(params, ["homo", "lumo"], 2)
    else:
        changed[field] = "different"
    assert select_resume_run(True, changed) is None
    output = capsys.readouterr().out
    assert field in output and "old=" in output and "new=" in output
    assert setup_logs_directory() != folder
    assert contents(folder) == before


def test_legacy_metadata_starts_fresh(run_inputs, capsys):
    _, snapshot, folder = run_inputs
    path = Path(folder, ".resume_metadata.json")
    metadata = json.loads(path.read_text())
    del metadata["inputs"]
    path.write_text(json.dumps(metadata))
    before = contents(folder)
    assert select_resume_run(True, snapshot) is None
    assert "inputs not recorded in the previous run" in capsys.readouterr().out
    assert setup_logs_directory() != folder
    assert contents(folder) == before


def test_hardware_settings_and_file_paths_do_not_change_compatibility(run_inputs):
    params, snapshot, _ = run_inputs
    params.update(parallel=False, num_procs=9)
    Path("copy.xyz").write_bytes(Path(params["molecule"]).read_bytes())
    params["molecule"] = "copy.xyz"
    current = capture_resume_inputs(params, ["homo", "lumo"], 2)
    assert validate_resume_compatibility({"inputs": snapshot}, current) == (True, None)


def test_newest_run_is_selected(run_inputs):
    _, snapshot, _ = run_inputs
    newest = setup_logs_directory()
    create_resume_metadata(newest, "separate", 2, ["homo", "lumo"], snapshot)
    path = Path(newest, ".resume_metadata.json")
    metadata = json.loads(path.read_text())
    metadata["original_start"] = "2099-01-01T00:00:00"
    path.write_text(json.dumps(metadata))
    assert select_resume_run(True, snapshot)[0] == newest


def test_resume_input_defaults_and_file_parsing(tmp_path):
    path = tmp_path / "tuning.in"
    path.write_text("molecule = 'a.xyz'\nsurface_file = 'a.surf'\n")
    assert TuningInput.from_file(path).resume is True
    path.write_text(path.read_text() + "resume = False\n")
    assert TuningInput.from_file(path).resume is False


@pytest.mark.parametrize("value", ["False", 0, 1, None])
def test_resume_requires_boolean(value):
    with pytest.raises(ValueError, match="resume must be True or False"):
        TuningInput(molecule="a.xyz", surface_file="a.surf", resume=value)

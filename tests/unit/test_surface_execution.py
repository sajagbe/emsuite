"""Surface execution preserves coordinate frames and caller files."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from emsuite.inputs import SurfaceInput
from emsuite.surface import vdw
from emsuite.surface.generate import generate_surface
from emsuite.surface.io import load_surf


def test_concurrent_generation_ignores_stale_files(tmp_path, monkeypatch):
    xyz = tmp_path / "ligand.xyz"
    xyz.write_text("1\nshifted oxygen\nO 10 -3 2\n")
    stale = tmp_path / "ligand_vdw_surface.txt"
    stale.write_text("keep this caller file\n")
    monkeypatch.chdir(tmp_path)
    workspaces = []

    def run(args, cwd, **kwargs):
        assert args[0] == sys.executable and args[1] == "-c"
        assert cwd != tmp_path
        assert Path(args[3]).read_bytes() == xyz.read_bytes()
        workspaces.append(cwd)
        (cwd / "ligand_vdw_surface.txt").write_text("11.52 -3 2\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(vdw.subprocess, "run", run)
    with ThreadPoolExecutor(max_workers=2) as pool:
        surfaces = list(pool.map(lambda _: vdw.get_vdw_surface_coordinates(xyz), range(2)))
    assert len(set(workspaces)) == 2
    assert all(not path.exists() for path in workspaces)
    assert stale.read_text() == "keep this caller file\n"
    assert xyz.read_text() == "1\nshifted oxygen\nO 10 -3 2\n"
    for surface in surfaces:
        np.testing.assert_allclose(surface, [[11.52, -3, 2]])


def test_failed_generation_cannot_read_old_surface(tmp_path, monkeypatch):
    xyz = tmp_path / "ligand.xyz"
    xyz.write_text("1\noxygen\nO 0 0 0\n")
    stale = tmp_path / "ligand_vdw_surface.txt"
    stale.write_text("1 2 3\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vdw.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(FileNotFoundError, match="did not write"):
        vdw.get_vdw_surface_coordinates(xyz)
    assert stale.read_text() == "1 2 3\n"


def test_real_surface_keeps_xyz_frame_without_vsg_in_path(tmp_path, monkeypatch):
    from emsuite.surface import generate

    xyz, surf = tmp_path / "oxygen.xyz", tmp_path / "oxygen.surf"
    xyz.write_text("1\nshifted oxygen\nO 10 -3 2\n")
    original = xyz.read_bytes()
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    def no_optimization(*args, **kwargs):
        pytest.fail("Surface generation must not optimize supplied XYZ by default")

    monkeypatch.setattr(generate, "optimize_with_pyscf", no_optimization)
    generate_surface(
        "XYZ", str(xyz), str(surf), surface_density=0.01, surface_scale=1.3, surface_charge=0.1
    )
    points, charges = load_surf(surf)
    assert xyz.read_bytes() == original
    assert len(points) > 0
    np.testing.assert_allclose(np.linalg.norm(points - [10, -3, 2], axis=1), 1.52 * 1.3, atol=1e-6)
    np.testing.assert_allclose(charges, 0.1)


@pytest.mark.parametrize(
    "key,value",
    [
        ("surface_density", 0),
        ("surface_scale", -1),
        ("surface_charge", float("nan")),
        ("surface_type", "invalid"),
    ],
)
def test_surface_rejects_invalid_settings(key, value):
    with pytest.raises(ValueError, match=key):
        SurfaceInput(input_type="XYZ", input_data="m.xyz", **{key: value})

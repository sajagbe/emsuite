"""Geometry warnings use the same exposed VDW spheres as surface generation."""

import inspect
from pathlib import Path

import numpy as np
import pytest
from vdw_surfgen.cli import VDW_RADII, fibonacci_sphere
from vdw_surfgen.cli import generate_surface as vsg_surface

from emsuite.inputs import SurfaceInput, TuningInput
from emsuite.surface.generate import generate_surface
from emsuite.surface.io import load_surf, save_surf
from emsuite.tuning.geometry import check_surface_frame


def write_xyz(path, symbols, centers):
    path.write_text(
        f"{len(symbols)}\ngeometry\n"
        + "".join(f"{symbol} {x} {y} {z}\n" for symbol, (x, y, z) in zip(symbols, centers, strict=True))
    )


def exposed_points(symbols, centers, scale):
    centers = np.asarray(centers)
    radii = np.array([VDW_RADII.get(symbol, 1.5) for symbol in symbols]) * scale
    clouds = []
    for j, (center, radius) in enumerate(zip(centers, radii, strict=True)):
        points = center + fibonacci_sphere(100) * radius
        keep = np.ones(len(points), dtype=bool)
        for k in range(len(centers)):
            if k != j:
                keep &= np.linalg.norm(points - centers[k], axis=1) > radii[k]
        clouds.append(points[keep])
    return np.round(np.concatenate(clouds), 6)


@pytest.mark.parametrize("scale", [0.8, 1.0, 1.5, 2.0])
def test_matching_rounded_surface_reports_scale(tmp_path, capsys, scale):
    xyz = tmp_path / "water.xyz"
    centers = [[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]]
    symbols = ["O", "H", "H"]
    write_xyz(xyz, symbols, centers)
    points = exposed_points(symbols, centers, scale)
    assert check_surface_frame(xyz, points) == pytest.approx(scale, abs=1e-6)
    output = capsys.readouterr().out
    assert "inferred surface scale=" in output and "WARNING" not in output


@pytest.mark.parametrize("kind", ["shifted", "different"])
def test_mismatched_geometry_warns_with_diagnostics(tmp_path, capsys, kind):
    xyz = tmp_path / "ligand.xyz"
    points = exposed_points(["O", "H"], [[0, 0, 0], [1, 0, 0]], 1.2)
    if kind == "shifted":
        write_xyz(xyz, ["O", "H"], [[0.5, 0, 0], [1.5, 0, 0]])
    else:
        write_xyz(xyz, ["C"], [[3, 0, 0]])
    check_surface_frame(xyz, points)
    output = capsys.readouterr().out
    for text in (
        "WARNING",
        "median scale=",
        "min=",
        "max=",
        "points off",
        "moved or rotated",
        "Continuing",
    ):
        assert text in output


@pytest.mark.parametrize("symbol", ["Co", "cl"])
def test_missing_radius_uses_generator_fallback_without_normalizing_symbol(
    tmp_path, capsys, symbol
):
    assert symbol not in VDW_RADII
    xyz = tmp_path / "fallback.xyz"
    write_xyz(xyz, [symbol], [[0, 0, 0]])
    points = fibonacci_sphere(30) * 1.5 * 1.2
    assert check_surface_frame(xyz, points) == pytest.approx(1.2)
    assert "WARNING" not in capsys.readouterr().out


def test_actual_generator_surface_roundtrip(tmp_path, capsys):
    xyz, surf = tmp_path / "water.xyz", tmp_path / "water.surf"
    centers = [[0, 0, 0], [0.76, 0, 0.59], [-0.76, 0, 0.59]]
    symbols = ["O", "H", "H"]
    write_xyz(xyz, symbols, centers)
    points = vsg_surface(centers, symbols, scale_factor=1.3, density=0.01)
    save_surf(points, 0.1, surf)
    loaded, _ = load_surf(surf)
    assert check_surface_frame(xyz, loaded) == pytest.approx(1.3, abs=1e-6)
    assert "WARNING" not in capsys.readouterr().out


def test_runner_warns_and_continues_to_scf_setup(tmp_path, monkeypatch, capsys):
    from emsuite.tuning import runner

    monkeypatch.chdir(tmp_path)
    xyz, surf = Path("ligand.xyz"), Path("ligand.surf")
    write_xyz(xyz, ["O"], [[0.5, 0, 0]])
    save_surf(exposed_points(["O"], [[0, 0, 0]], 1.0), 0.1, surf)
    monkeypatch.setattr(runner.core, "CUPY_AVAILABLE", False)
    monkeypatch.setattr(runner.core, "check_gpu_info", lambda: 0)
    monkeypatch.setattr(runner.core, "check_cpu_info", lambda: 1)

    class ReachedSCF(Exception):
        pass

    def stop_at_scf(*args, **kwargs):
        assert "WARNING: XYZ/surface geometry is inconsistent" in capsys.readouterr().out
        raise ReachedSCF

    monkeypatch.setattr(runner, "create_molecule_objects", stop_at_scf)
    with pytest.raises(ReachedSCF):
        runner._run_tuning(
            TuningInput(molecule=str(xyz), surface_file=str(surf), properties=("gse",))
        )


def test_surface_charge_defaults():
    assert inspect.signature(generate_surface).parameters["surface_charge"].default == 0.1
    assert SurfaceInput(input_type="XYZ", input_data="ligand.xyz").surface_charge == 0.1
    from emsuite.surface import parse_surface_input

    assert parse_surface_input("examples/templates/surface.in")["surface_charge"] == 0.1

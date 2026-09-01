"""End-to-end surface → tuning smoke test on methane (fast QM basis)."""

from __future__ import annotations

from pathlib import Path

import pytest

from emsuite.surface import run_surface_calculation
from emsuite.tuning import run_tuning_calculation

from .helpers import record_assertions

SURFACE_IN = """\
input_type = 'SMILES'
input_data = 'C'
surface_density = 0.5
surface_scale = 1.0
surface_type = 'homogenous'
surface_charge = 0.1
output_surf = 'methane.surf'
optimize = True
optimize_method = 'uff'
optimized_xyz = 'methane.xyz'
"""

TUNING_IN = """\
molecule = 'methane.xyz'
surface_file = 'methane.surf'
properties = ['homo', 'lumo', 'gap']
basis_set = 'sto-3g'
method = 'dft'
functional = 'b3lyp'
charge = 0
spin = 0
solvent = None
calc_type = 'separate'
parallel = False
"""


@pytest.mark.slow
def test_methane_surface_to_tuning_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run a minimal PySCF pipeline: UFF surface, then serial homo/lumo/gap tuning."""
    monkeypatch.chdir(tmp_path)

    (tmp_path / "surface.in").write_text(SURFACE_IN)
    (tmp_path / "tuning.in").write_text(TUNING_IN)

    surf_path = run_surface_calculation("surface.in")
    assert Path(surf_path).is_file()
    assert Path("methane.xyz").is_file()

    surf_lines = Path(surf_path).read_text().strip().splitlines()
    assert len(surf_lines) >= 11  # header + at least 10 surface points

    run_tuning_calculation("tuning.in")

    results_dirs = sorted(tmp_path.glob("results_methane_*"))
    assert results_dirs, "expected timestamped results_methane_* directory"
    results_dir = results_dirs[-1]

    summary_csv = results_dir / "methane_tuning_summary.csv"
    assert summary_csv.is_file()
    csv_lines = summary_csv.read_text().strip().splitlines()
    surf_point_count = len(surf_lines) - 1
    assert len(csv_lines) == surf_point_count + 1  # header + one row per surface point

    for prop in ("homo", "lumo", "gap"):
        assert (results_dir / f"methane_{prop}.mol2").is_file()
        assert (results_dir / f"methane_{prop}_normalized.mol2").is_file()

    assert (results_dir / "logs").is_dir()

    # Checkpoint files should be cleaned up after a successful run.
    for chk in ("molecule_alone.chk", "anion_alone.chk", "cation_alone.chk"):
        assert not (tmp_path / chk).exists()
        assert not (results_dir / chk).exists()

    record_assertions(
        tmp_path,
        surf_points=surf_point_count,
        csv_rows=len(csv_lines) - 1,
        properties=["homo", "lumo", "gap"],
        results_dir=str(results_dir),
    )

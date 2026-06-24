"""CCO golden artifact regression (structure only — no QM re-run)."""

from pathlib import Path

import numpy as np
import pytest

GOLDEN_CSV = Path("examples/tuning/CCO2-exe/separate/CCO_opt2_tuning_summary.csv")


@pytest.mark.skipif(not GOLDEN_CSV.exists(), reason="CCO golden CSV not present")
def test_cco_golden_csv_structure():
    lines = GOLDEN_CSV.read_text().strip().splitlines()
    header = lines[0].split(",")
    assert "point_index" in header
    assert "s1_exe_effect" in header
    assert "s1_exe_effect_normalized" in header
    assert len(lines) > 10


@pytest.mark.skipif(not GOLDEN_CSV.exists(), reason="CCO golden CSV not present")
def test_cco_normalized_values_in_unit_interval():
    import csv

    with GOLDEN_CSV.open() as handle:
        reader = csv.DictReader(handle)
        normalized = [
            float(row["s1_exe_effect_normalized"])
            for row in reader
            if row.get("s1_exe_effect_normalized")
        ]
    assert normalized
    assert min(normalized) >= -1.0 - 1e-6
    assert max(normalized) <= 1.0 + 1e-6


@pytest.mark.skipif(not GOLDEN_CSV.exists(), reason="CCO golden CSV not present")
def test_cco_baseline_constant_per_property():
    import csv

    with GOLDEN_CSV.open() as handle:
        reader = csv.DictReader(handle)
        baselines = [float(row["s1_exe_baseline"]) for row in reader]
    assert len(set(np.round(baselines, 6))) == 1

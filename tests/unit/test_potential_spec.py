"""Potential channel input file: parsing and checks (no calculation)."""

from pathlib import Path

import pytest

from emsuite.config.schemas import ConfigValidationError
from emsuite.potential.spec import DEFAULT_FORCEFIELD, PotentialSpec, Selection

FULL = """\
# comment line
pdb          = 'complex.pdb'
surface_file = 'ligand.surf'
output_surf  = 'asL1.surf'
quantity     = 'Charge'            # trailing comment
forcefield   = ['amber14/protein.ff14SB.xml', 'amber14/tip3p.xml']
add_hydrogens = True
ph            = 6.5
unknown_charge = {'FMN': -2, 'RET': 1}
unknown_spin   = {'FMN': 0}
resp_copies    = 'one'
off            = ['FMN', 'RET:A:296@C15']
pdie = 4
sdie = 80.0
"""


def _write(tmp_path: Path, text: str, files: bool = True) -> Path:
    if files:
        (tmp_path / "complex.pdb").write_text("END\n")
        (tmp_path / "ligand.surf").write_text("x y z q\n")
    cfg = tmp_path / "potential.in"
    cfg.write_text(text)
    return cfg


def test_full_file_is_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    spec = PotentialSpec.from_file(_write(tmp_path, FULL))
    assert spec.pdb == "complex.pdb"
    assert spec.surface_file == "ligand.surf"
    assert spec.output_surf == "asL1.surf"
    assert spec.quantity == "charge"
    assert spec.forcefield == ("amber14/protein.ff14SB.xml", "amber14/tip3p.xml")
    assert spec.add_hydrogens is True
    assert spec.ph == 6.5
    assert spec.unknown_charge == {"FMN": -2, "RET": 1}
    assert spec.spin_of("FMN") == 0 and spec.spin_of("RET") == 0
    assert spec.resp_copies == "one"
    assert spec.off == (Selection("FMN"), Selection("RET", "A", 296, "C15"))
    assert spec.pdie == 4.0 and spec.sdie == 80.0


def test_minimal_file_gets_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    spec = PotentialSpec.from_file(
        _write(tmp_path, "pdb = 'complex.pdb'\nsurface_file = 'ligand.surf'\n")
    )
    assert spec.quantity == "potential"
    assert spec.method == "apbs"
    assert spec.forcefield == DEFAULT_FORCEFIELD
    assert spec.add_hydrogens is False
    assert spec.unknown_charge == {} and spec.unknown_spin == {}
    assert spec.resp_copies == "each"
    assert spec.off == ()
    assert (spec.pdie, spec.sdie) == (2.0, 78.54)


def test_file_matches_constructor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    from_file = PotentialSpec.from_file(
        _write(tmp_path, "pdb = 'complex.pdb'\nsurface_file = 'ligand.surf'\noff = 'FMN'\n")
    )
    assert from_file == PotentialSpec(pdb="complex.pdb", surface_file="ligand.surf", off=["FMN"])


def test_unknown_key_stops(tmp_path: Path):
    cfg = _write(tmp_path, "pdb = 'complex.pdb'\nsurface_file = 'ligand.surf'\npdei = 2.0\n")
    with pytest.raises(ConfigValidationError, match="unknown potential input key 'pdei'"):
        PotentialSpec.from_file(cfg)


@pytest.mark.parametrize(
    ("line", "hint"),
    [
        ("molecule = 'm.xyz'", "read from the PDB"),
        ("protein = 'p.xyz'", "use pdb="),
        ("ligand_atoms = 'absent'", "use off="),
        ("ligand_mol2 = 'l.mol2'", "unknown_charge"),
        ("surface_density = 0.5", "surface channel"),
    ],
)
def test_removed_key_says_what_replaces_it(tmp_path: Path, line: str, hint: str):
    cfg = _write(tmp_path, f"pdb = 'complex.pdb'\nsurface_file = 'ligand.surf'\n{line}\n")
    with pytest.raises(ConfigValidationError, match=hint):
        PotentialSpec.from_file(cfg)


@pytest.mark.parametrize("missing", ["pdb", "surface_file"])
def test_missing_required_key_stops(tmp_path: Path, missing: str):
    lines = {"pdb": "pdb = 'complex.pdb'", "surface_file": "surface_file = 'ligand.surf'"}
    del lines[missing]
    cfg = _write(tmp_path, "\n".join(lines.values()) + "\n")
    with pytest.raises(ConfigValidationError, match=f"missing required parameter: {missing}"):
        PotentialSpec.from_file(cfg)


def test_missing_data_file_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cfg = _write(tmp_path, "pdb = 'complex.pdb'\nsurface_file = 'ligand.surf'\n", files=False)
    with pytest.raises(FileNotFoundError, match="pdb file not found"):
        PotentialSpec.from_file(cfg)
    (tmp_path / "complex.pdb").write_text("END\n")
    with pytest.raises(FileNotFoundError, match="surface_file file not found"):
        PotentialSpec.from_file(cfg)
    assert PotentialSpec.from_file(cfg, check_files=False).pdb == "complex.pdb"


def test_missing_input_file_stops(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="potential input file not found"):
        PotentialSpec.from_file(tmp_path / "nope.in")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pdb": "complex.xyz"}, "must be a .pdb file"),
        ({"quantity": "field"}, "quantity must be"),
        ({"method": "esp"}, "method must be 'apbs'"),
        ({"forcefield": []}, "at least one"),
        ({"forcefield": ["AMBER"]}, "not an OpenMM XML file name"),
        ({"add_hydrogens": "yes"}, "add_hydrogens must be True or False"),
        ({"ph": 15}, "between 0 and 14"),
        ({"unknown_charge": "FMN"}, "must be a dict"),
        ({"unknown_charge": {"FMN": -2.0}}, "must be an integer"),
        ({"unknown_charge": {"FMN": True}}, "must be an integer"),
        ({"unknown_charge": {"FMN:A": -2}}, "not a residue name"),
        ({"unknown_charge": {"FMN": 0}, "unknown_spin": {"FMN": -1}}, "0 or more"),
        ({"unknown_spin": {"FMN": 1}}, "give its charge too"),
        ({"resp_copies": "all"}, "resp_copies must be"),
        ({"off": [3]}, "non-empty string"),
        ({"off": ["FMN:A:x"]}, "not an integer"),
        ({"off": ["FMN@"]}, "atom name missing"),
        ({"off": ["A:B:1:2"]}, "RES\\[:CHAIN\\]"),
        ({"pdie": 0}, "pdie must be a positive number"),
        ({"sdie": "high"}, "sdie must be a positive number"),
    ],
)
def test_bad_values_stop(kwargs: dict, message: str):
    base = {"pdb": "complex.pdb", "surface_file": "ligand.surf"}
    with pytest.raises(ConfigValidationError, match=message):
        PotentialSpec(**{**base, **kwargs})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("FMN", Selection("FMN")),
        ("FMN:12", Selection("FMN", None, 12)),
        ("FMN:A", Selection("FMN", "A")),
        ("FMN:A:12", Selection("FMN", "A", 12)),
        (" FMN:A:12@O2P ", Selection("FMN", "A", 12, "O2P")),
        ("NA@NA", Selection("NA", atom="NA")),
    ],
)
def test_selection_forms(text: str, expected: Selection):
    assert Selection.parse(text) == expected


def test_selection_matching_and_off_switch():
    spec = PotentialSpec(pdb="complex.pdb", surface_file="ligand.surf", off=["FMN", "LYS:A:12@NZ"])
    assert spec.is_off("FMN", "B", 500, "P")
    assert spec.is_off("LYS", "A", 12, "NZ")
    assert not spec.is_off("LYS", "A", 12, "CE")
    assert not spec.is_off("LYS", "B", 12, "NZ")
    assert not spec.is_off("ALA", "A", 1, "CA")


def test_removed_and_grid_spacing():
    spec = PotentialSpec(
        pdb="complex.pdb", surface_file="ligand.surf", removed="FMN", grid_spacing=0.3
    )
    assert spec.removed == (Selection("FMN"),)
    assert spec.is_removed("FMN", "A", 1, "P") and not spec.is_off("FMN", "A", 1, "P")
    assert spec.grid_spacing == 0.3
    assert PotentialSpec(pdb="complex.pdb", surface_file="ligand.surf").grid_spacing == 0.125
    assert (
        PotentialSpec(pdb="complex.pdb", surface_file="ligand.surf", grid_spacing=None).grid_spacing
        is None
    )
    with pytest.raises(ConfigValidationError, match="removed entry"):
        PotentialSpec(pdb="complex.pdb", surface_file="ligand.surf", removed=["FMN@"])
    with pytest.raises(ConfigValidationError, match="grid_spacing"):
        PotentialSpec(pdb="complex.pdb", surface_file="ligand.surf", grid_spacing=0.01)


def test_grid_size_for_spacing():
    from emsuite.potential.apbs import dime_for_spacing

    assert dime_for_spacing(46.7, 0.73) == 65
    assert dime_for_spacing(46.7, 0.3) == 161
    assert 46.7 / (dime_for_spacing(46.7, 0.25) - 1) <= 0.25


def test_unknown_level_of_theory_keys():
    spec = PotentialSpec(pdb="complex.pdb", surface_file="ligand.surf")
    assert (spec.unknown_method, spec.unknown_basis) == ("hf", "6-31g*")
    spec = PotentialSpec(
        pdb="complex.pdb", surface_file="ligand.surf", unknown_method="B3LYP", unknown_basis="3-21G"
    )
    assert (spec.unknown_method, spec.unknown_basis) == ("b3lyp", "3-21g")
    with pytest.raises(ConfigValidationError, match="unknown_basis"):
        PotentialSpec(pdb="complex.pdb", surface_file="ligand.surf", unknown_basis="")

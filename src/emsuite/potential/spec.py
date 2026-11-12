"""Potential channel input: one ``.in`` file names the PDB, the ``.surf`` and every setting.

Known residues get Amber charges and radii from the OpenMM force-field XML files.
Unknown residues get HF/6-31G* RESP charges and vdw-surfgen element radii; the
user states their total charge (and spin) here. The PDB must already be
correctly protonated unless ``add_hydrogens`` is set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Self

from emsuite.config.parser import parse_assignments
from emsuite.config.schemas import ConfigValidationError

DEFAULT_FORCEFIELD: tuple[str, ...] = (
    "amber14/protein.ff14SB.xml",
    "amber14/DNA.OL15.xml",
    "amber14/RNA.OL3.xml",
    "amber14/lipid17.xml",
    "amber14/GLYCAM_06j-1.xml",
    "amber14/tip3p.xml",
)

_QUANTITIES = frozenset({"potential", "charge"})
_RESP_COPIES = frozenset({"each", "one"})

# Keys of the old potential input, with what replaces each.
_REMOVED_KEYS: dict[str, str] = {
    "molecule": "the ligand is read from the PDB; give pdb= and surface_file=",
    "ligand": "the ligand is read from the PDB; give pdb= and surface_file=",
    "protein": "use pdb=",
    "protein_format": "only PDB input is supported; use pdb=",
    "ligand_atoms": "use off= to switch residues or atoms off",
    "ligand_resname": "use off= to switch residues or atoms off",
    "ligand_chain": "use off= to switch residues or atoms off",
    "ligand_resseq": "use off= to switch residues or atoms off",
    "ligand_mol2": "unknown residues get RESP charges; use unknown_charge=",
    "charge": "use unknown_charge={'RES': charge}",
    "spin": "use unknown_spin={'RES': spin}",
    "surface_density": "make the surface in the surface channel; give surface_file=",
    "surface_scale": "make the surface in the surface channel; give surface_file=",
}


@dataclass(frozen=True)
class Selection:
    """Residues or one atom named by ``RES[:CHAIN][:SEQ][@ATOM]``.

    ``FMN`` is every FMN residue; ``FMN:A:1`` is one residue; ``FMN:1@O2P`` is one atom.
    """

    resname: str
    chain: str | None = None
    resseq: int | None = None
    atom: str | None = None

    @classmethod
    def parse(cls, text: Any) -> Self:
        if not isinstance(text, str) or not text.strip():
            raise ConfigValidationError(f"off entry must be a non-empty string, got {text!r}")
        body, at, atom = text.strip().partition("@")
        if at and not atom.strip():
            raise ConfigValidationError(f"off entry {text!r}: atom name missing after '@'")
        parts = [p.strip() for p in body.split(":")]
        if not parts[0] or len(parts) > 3 or any(not p for p in parts):
            raise ConfigValidationError(
                f"off entry {text!r} is not of the form RES[:CHAIN][:SEQ][@ATOM]"
            )
        chain: str | None = None
        resseq: int | None = None
        rest = parts[1:]
        if len(rest) == 2:
            chain, seq_text = rest
            try:
                resseq = int(seq_text)
            except ValueError:
                raise ConfigValidationError(
                    f"off entry {text!r}: residue number {seq_text!r} is not an integer"
                ) from None
        elif len(rest) == 1:
            try:
                resseq = int(rest[0])
            except ValueError:
                chain = rest[0]
        return cls(parts[0], chain, resseq, atom.strip() or None)

    def matches(self, resname: str, chain: str, resseq: int, atom: str) -> bool:
        return (
            resname.strip() == self.resname
            and (self.chain is None or chain.strip() == self.chain)
            and (self.resseq is None or resseq == self.resseq)
            and (self.atom is None or atom.strip() == self.atom)
        )


def _selection(entry: Any, key: str) -> Selection:
    if isinstance(entry, Selection):
        return entry
    try:
        return Selection.parse(entry)
    except ConfigValidationError as exc:
        raise ConfigValidationError(str(exc).replace("off entry", f"{key} entry", 1)) from None


def _strict_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigValidationError(f"{what} must be an integer, got {value!r}")
    return value


def _positive_float(value: Any, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ConfigValidationError(f"{what} must be a positive number, got {value!r}")
    return float(value)


def _residue_map(value: Any, what: str) -> dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigValidationError(
            f"{what} must be a dict of residue name to integer, e.g. {{'FMN': -2}}"
        )
    out: dict[str, int] = {}
    for name, number in value.items():
        if not isinstance(name, str) or not name.strip() or ":" in name or "@" in name:
            raise ConfigValidationError(f"{what}: {name!r} is not a residue name")
        out[name.strip()] = _strict_int(number, f"{what}[{name!r}]")
    return out


def _string_list(value: Any, what: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list | tuple) or not all(
        isinstance(v, str) and v.strip() for v in value
    ):
        raise ConfigValidationError(f"{what} must be a string or a list of strings")
    return tuple(v.strip() for v in value)


@dataclass(frozen=True)
class PotentialSpec:
    """Validated settings of one potential-channel run."""

    pdb: str
    surface_file: str
    output_surf: str = "potential.surf"
    quantity: str = "potential"
    method: str = "apbs"
    forcefield: tuple[str, ...] = DEFAULT_FORCEFIELD
    add_hydrogens: bool = False
    ph: float = 7.0
    unknown_charge: dict[str, int] = field(default_factory=dict)
    unknown_spin: dict[str, int] = field(default_factory=dict)
    resp_copies: str = "each"
    off: tuple[Selection, ...] = ()
    uncharged: tuple[Selection, ...] = ()
    pdie: float = 2.0
    sdie: float = 78.54
    grid_spacing: float | None = None

    def __post_init__(self) -> None:
        def put(name: str, value: Any) -> None:
            object.__setattr__(self, name, value)

        for name in ("pdb", "surface_file", "output_surf"):
            value = getattr(self, name)
            if not isinstance(value, str | Path) or not str(value).strip():
                raise ConfigValidationError(f"potential input missing required parameter: {name}")
            put(name, str(value))
        if Path(self.pdb).suffix.lower() not in {".pdb", ".ent"}:
            raise ConfigValidationError(f"pdb must be a .pdb file, got {self.pdb!r}")

        quantity = str(self.quantity).lower()
        if quantity not in _QUANTITIES:
            raise ConfigValidationError("quantity must be 'potential' or 'charge'")
        put("quantity", quantity)
        method = str(self.method).lower()
        if method != "apbs":
            raise ConfigValidationError(f"method must be 'apbs', got {self.method!r}")
        put("method", method)

        forcefield = _string_list(self.forcefield, "forcefield")
        if not forcefield:
            raise ConfigValidationError("forcefield must name at least one OpenMM XML file")
        bad = [name for name in forcefield if not name.lower().endswith(".xml")]
        if bad:
            raise ConfigValidationError(
                f"forcefield entry {bad[0]!r} is not an OpenMM XML file name "
                "(for example 'amber14/protein.ff14SB.xml')"
            )
        put("forcefield", forcefield)

        if not isinstance(self.add_hydrogens, bool):
            raise ConfigValidationError("add_hydrogens must be True or False")
        ph = _positive_float(self.ph, "ph")
        if ph > 14:
            raise ConfigValidationError(f"ph must be between 0 and 14, got {self.ph!r}")
        put("ph", ph)

        charges = _residue_map(self.unknown_charge, "unknown_charge")
        spins = _residue_map(self.unknown_spin, "unknown_spin")
        for name, spin in spins.items():
            if spin < 0:
                raise ConfigValidationError(
                    f"unknown_spin[{name!r}] must be 0 or more (number of unpaired electrons)"
                )
            if name not in charges:
                raise ConfigValidationError(
                    f"unknown_spin names {name!r} but unknown_charge does not; give its charge too"
                )
        put("unknown_charge", charges)
        put("unknown_spin", spins)

        resp_copies = str(self.resp_copies).lower()
        if resp_copies not in _RESP_COPIES:
            raise ConfigValidationError(
                "resp_copies must be 'each' (RESP on every copy of an unknown residue) "
                "or 'one' (RESP on the first copy, reused for the others)"
            )
        put("resp_copies", resp_copies)

        for name in ("off", "uncharged"):
            entries = getattr(self, name)
            if isinstance(entries, str | Selection):
                entries = [entries]
            if entries is None:
                entries = ()
            if not isinstance(entries, list | tuple):
                raise ConfigValidationError(f"{name} must be a string or a list of strings")
            put(name, tuple(_selection(s, name) for s in entries))

        put("pdie", _positive_float(self.pdie, "pdie"))
        put("sdie", _positive_float(self.sdie, "sdie"))
        if self.grid_spacing is not None:
            spacing = _positive_float(self.grid_spacing, "grid_spacing")
            if spacing < 0.1:
                raise ConfigValidationError(
                    "grid_spacing below 0.1 Å needs too much memory; use 0.2 to 0.5"
                )
            put("grid_spacing", spacing)

    @classmethod
    def from_file(cls, path: str | Path, *, check_files: bool = True) -> Self:
        """Read a potential ``.in`` file. Unknown keys and missing files are errors."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"potential input file not found: {path}")
        params = parse_assignments(path.read_text())
        known = {item.name for item in fields(cls)}
        for key in params:
            if key in _REMOVED_KEYS:
                raise ConfigValidationError(
                    f"{path.name}: {key!r} is no longer a potential input key; {_REMOVED_KEYS[key]}"
                )
            if key not in known:
                raise ConfigValidationError(
                    f"{path.name}: unknown potential input key {key!r} "
                    f"(known keys: {', '.join(sorted(known))})"
                )
        for name in ("pdb", "surface_file"):
            if params.get(name) in (None, ""):
                raise ConfigValidationError(f"potential input missing required parameter: {name}")
        spec = cls(**params)
        if check_files:
            spec.check_files()
        return spec

    def check_files(self) -> None:
        """Stop early when an input file is missing (paths are relative to the run folder)."""
        for name in ("pdb", "surface_file"):
            value = getattr(self, name)
            if not Path(value).is_file():
                raise FileNotFoundError(f"{name} file not found: {value}")

    def is_off(self, resname: str, chain: str, resseq: int, atom: str) -> bool:
        """Off: charge 0 and radius 0, so the solvent fills the atom's place."""
        return any(s.matches(resname, chain, resseq, atom) for s in self.off)

    def is_uncharged(self, resname: str, chain: str, resseq: int, atom: str) -> bool:
        """Uncharged: charge 0, radius kept, so the atom still keeps the solvent out."""
        return any(s.matches(resname, chain, resseq, atom) for s in self.uncharged)

    def spin_of(self, resname: str) -> int:
        """Unpaired electrons of an unknown residue; singlet unless stated."""
        return self.unknown_spin.get(resname, 0)

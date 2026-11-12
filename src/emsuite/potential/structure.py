"""Read the PDB of a potential-channel run into atoms, residues and bonds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from emsuite.config.schemas import ConfigValidationError


@dataclass(frozen=True)
class Structure:
    """Atoms of the PDB in file order. Names are those written in the PDB file."""

    topology: Any  # openmm.app.Topology
    coords: np.ndarray  # (N, 3) Å
    elements: tuple[str, ...]
    atom_names: tuple[str, ...]
    res_names: tuple[str, ...]  # per atom
    chains: tuple[str, ...]  # per atom
    res_seqs: tuple[int, ...]  # per atom
    res_index: np.ndarray  # per atom, index of its residue in the topology
    hydrogens_added: int = 0

    @property
    def n_atoms(self) -> int:
        return len(self.elements)

    def residue_atoms(self, index: int) -> np.ndarray:
        return np.flatnonzero(self.res_index == index)

    def residue_label(self, index: int) -> str:
        first = int(self.residue_atoms(index)[0])
        chain = self.chains[first] or "-"
        return f"{self.res_names[first]}:{chain}:{self.res_seqs[first]}"


def _file_names(path: Path, n_atoms: int) -> tuple[list[str], list[str]] | None:
    """Atom and residue names as written in the file (OpenMM renames some, e.g. HIP to HIS)."""
    from openmm.app.internal.pdbstructure import PdbStructure

    with open(path) as handle:
        raw = PdbStructure(handle)
    atom_names: list[str] = []
    res_names: list[str] = []
    for chain in raw.iter_chains():
        for residue in chain.iter_residues():
            for atom in residue.iter_atoms():
                atom_names.append(atom.get_name().strip())
                res_names.append(residue.get_name().strip())
    if len(atom_names) != n_atoms:
        return None
    return atom_names, res_names


def _seq(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        return -1


def load_structure(
    pdb_path: str | Path, *, add_hydrogens: bool = False, ph: float = 7.0
) -> Structure:
    """Load a PDB. Hydrogens are used as given unless ``add_hydrogens`` is set, in which
    case OpenMM adds the missing ones on residues it knows, at ``ph``; existing atoms
    are not moved."""
    from openmm import app, unit

    path = Path(pdb_path)
    if not path.is_file():
        raise FileNotFoundError(f"pdb file not found: {path}")
    try:
        pdb = app.PDBFile(str(path))
    except Exception as exc:
        raise ConfigValidationError(f"could not read PDB {path}: {exc}") from exc
    topology = pdb.topology
    positions = pdb.positions
    if topology.getNumAtoms() == 0:
        raise ConfigValidationError(f"PDB {path} has no atoms")
    names = _file_names(path, topology.getNumAtoms())

    added = 0
    if add_hydrogens:
        old_heavy = [
            a.index for a in topology.atoms() if a.element is None or a.element.symbol != "H"
        ]
        before = np.array(positions.value_in_unit(unit.angstrom))[old_heavy]
        modeller = app.Modeller(topology, positions)
        modeller.addHydrogens(pH=ph)
        added = modeller.topology.getNumAtoms() - topology.getNumAtoms()
        topology, positions = modeller.topology, modeller.positions
        names = None  # atom list changed: use OpenMM's names throughout
        new_heavy = [
            a.index for a in topology.atoms() if a.element is None or a.element.symbol != "H"
        ]
        after = np.array(positions.value_in_unit(unit.angstrom))[new_heavy]
        if before.shape != after.shape or not np.allclose(before, after, atol=1e-3):
            raise RuntimeError("adding hydrogens changed the heavy atoms; refusing to continue")

    coords = np.array(positions.value_in_unit(unit.angstrom), dtype=float)
    elements, atom_names, res_names, chains, res_seqs, res_index = [], [], [], [], [], []
    for atom in topology.atoms():
        if atom.element is None:
            raise ConfigValidationError(
                f"atom {atom.name} of residue {atom.residue.name} {atom.residue.id} has no "
                "recognizable element; fill the element column of the PDB"
            )
        elements.append(atom.element.symbol)
        atom_names.append(names[0][atom.index] if names else atom.name)
        res_names.append(names[1][atom.index] if names else atom.residue.name)
        chains.append(atom.residue.chain.id.strip())
        res_seqs.append(_seq(atom.residue.id))
        res_index.append(atom.residue.index)
    multi_atom = np.bincount(res_index).max() > 1
    if multi_atom and "H" not in elements:
        raise ConfigValidationError(
            f"PDB {path} has no hydrogen atoms. Provide a protonated PDB, or set "
            "add_hydrogens = True to let OpenMM add them on standard residues."
        )
    return Structure(
        topology=topology,
        coords=coords,
        elements=tuple(elements),
        atom_names=tuple(atom_names),
        res_names=tuple(res_names),
        chains=tuple(chains),
        res_seqs=tuple(res_seqs),
        res_index=np.array(res_index, dtype=int),
        hydrogens_added=added,
    )

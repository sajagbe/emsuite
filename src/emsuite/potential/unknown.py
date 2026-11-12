"""Charges and radii for residues with no Amber template.

Charges: two-stage RESP from a quantum calculation (HF/6-31G* unless the input
says otherwise) on the residue alone, at its PDB geometry (no optimization). Radii: the vdw-surfgen element table. The
user states each residue's total charge (and spin) in the input.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import qcelemental as qcel
from scipy.spatial import cKDTree

from emsuite.config.schemas import ConfigValidationError
from emsuite.physconst import vdw_radius_angstrom

from .structure import Structure

RespFunction = Callable[[list[tuple[str, float, float, float]], int, int, str], np.ndarray]
_BOND_FACTOR = 1.25  # bonded if closer than 1.25 x the sum of covalent radii


@dataclass(frozen=True)
class UnknownResult:
    charges: dict[int, np.ndarray]  # residue index -> per-atom charge (e)
    radii: dict[int, np.ndarray]  # residue index -> per-atom radius (Å)
    sources: dict[int, str]  # residue index -> how the charges were obtained
    hf_runs: int


def _resp_at(method: str, basis: str) -> RespFunction:
    def run(atoms, charge, spin, label):
        from .resp import resp_charges_from_atoms

        return resp_charges_from_atoms(
            atoms, charge=charge, spin=spin, basis=basis, label=label, method=method
        )

    return run


def _covalent_radius(symbol: str) -> float:
    return float(qcel.covalentradii.get(symbol, units="angstrom", missing=1.5))


def bonded_neighbors(structure: Structure, residue: int) -> list[str]:
    """Atoms of other multi-atom residues that are within bonding distance of ``residue``.

    Single-atom residues (ions) are ignored: a metal next to a ligand is not a covalent link.
    """
    mine = structure.residue_atoms(residue)
    sizes = np.bincount(structure.res_index)
    others = np.flatnonzero((structure.res_index != residue) & (sizes[structure.res_index] > 1))
    if len(others) == 0:
        return []
    tree = cKDTree(structure.coords[others])
    found = []
    for i in mine:
        for k in tree.query_ball_point(structure.coords[i], 2.6):
            j = int(others[k])
            if structure.elements[i] == "H" and structure.elements[j] == "H":
                continue
            limit = _BOND_FACTOR * (
                _covalent_radius(structure.elements[i]) + _covalent_radius(structure.elements[j])
            )
            if np.linalg.norm(structure.coords[i] - structure.coords[j]) < limit:
                found.append(
                    f"{structure.atom_names[i]} - {structure.residue_label(int(structure.res_index[j]))}"
                    f"@{structure.atom_names[j]}"
                )
    return found


def check_unknown_residues(
    structure: Structure,
    unmatched: tuple[int, ...],
    unknown_charge: dict[str, int],
    unknown_spin: dict[str, int],
) -> None:
    """Stop before any HF calculation if an unknown residue cannot be handled."""
    from openmm.app import PDBFile

    PDBFile._loadNameReplacementTables()
    standard = set(PDBFile._residueNameReplacements) | set(
        PDBFile._residueNameReplacements.values()
    )
    problems: list[str] = []
    for residue in unmatched:
        atoms = structure.residue_atoms(residue)
        name = structure.res_names[atoms[0]]
        label = structure.residue_label(residue)
        if name not in unknown_charge:
            hint = (
                " It is a standard residue name, so it is probably missing hydrogens or atoms"
                " (set add_hydrogens = True, or fix the PDB)."
                if name in standard
                else f" If it is a new molecule, add unknown_charge = {{'{name}': <charge>}}."
            )
            problems.append(f"{label}: no Amber template matches it.{hint}")
            continue
        electrons = sum(qcel.periodictable.to_Z(structure.elements[i]) for i in atoms)
        electrons -= unknown_charge[name]
        spin = unknown_spin.get(name, 0)
        if (electrons - spin) % 2 != 0:
            problems.append(
                f"{label}: {electrons} electrons with charge {unknown_charge[name]:+d} cannot have "
                f"spin {spin}. Check the charge, the spin, or the hydrogens of this residue."
            )
        links = bonded_neighbors(structure, residue)
        if links:
            problems.append(
                f"{label}: bonded to another residue ({'; '.join(links[:3])}). Unknown residues "
                "bonded to the rest of the structure are not supported yet."
            )
    if problems:
        raise ConfigValidationError(
            "unknown residues cannot be processed:\n  " + "\n  ".join(problems)
        )


def assign_unknown(
    structure: Structure,
    unmatched: tuple[int, ...],
    unknown_charge: dict[str, int],
    unknown_spin: dict[str, int],
    resp_copies: str = "each",
    skip: np.ndarray | None = None,
    resp: RespFunction | None = None,
    method: str = "hf",
    basis: str = "6-31g*",
) -> UnknownResult:
    """RESP charges and element radii for the unmatched residues.

    ``skip`` marks atoms whose charge is switched to zero; a residue whose atoms are all skipped
    gets no HF calculation. ``resp_copies='one'`` runs RESP on the first copy of
    each residue name and reuses its charges for the other copies (matched by atom name).
    """
    resp = resp or _resp_at(method, basis)
    level = f"{method.upper()}/{basis.upper()}"
    skip = np.zeros(structure.n_atoms, dtype=bool) if skip is None else skip
    # a residue whose charges are all switched to zero needs no charge and no HF calculation
    active = tuple(r for r in unmatched if not skip[structure.residue_atoms(r)].all())
    check_unknown_residues(structure, active, unknown_charge, unknown_spin)

    charges: dict[int, np.ndarray] = {}
    radii: dict[int, np.ndarray] = {}
    sources: dict[int, str] = {}
    todo = list(active)
    for residue in unmatched:
        atoms = structure.residue_atoms(residue)
        radii[residue] = np.array([vdw_radius_angstrom(structure.elements[i]) for i in atoms])
        if residue not in todo:
            charges[residue] = np.zeros(len(atoms))
            sources[residue] = (
                "unknown residue, all charges switched to zero: no RESP run; vsg element radii"
            )

    first_of: dict[str, int] = {}
    plan: list[int] = []
    for residue in todo:
        name = structure.res_names[structure.residue_atoms(residue)[0]]
        if resp_copies == "one" and name in first_of:
            continue
        first_of.setdefault(name, residue)
        plan.append(residue)
    if plan:
        sizes = ", ".join(
            f"{structure.residue_label(r)} ({len(structure.residue_atoms(r))} atoms)" for r in plan
        )
        print(f"{level} RESP calculations to run: {len(plan)} -> {sizes}")

    for residue in plan:
        atoms = structure.residue_atoms(residue)
        name = structure.res_names[atoms[0]]
        geometry = [
            (structure.elements[i], *(float(v) for v in structure.coords[i])) for i in atoms
        ]
        label = structure.residue_label(residue)
        q = np.asarray(resp(geometry, unknown_charge[name], unknown_spin.get(name, 0), label))
        if q.shape != (len(atoms),) or not np.all(np.isfinite(q)):
            raise RuntimeError(f"RESP returned unusable charges for {label}")
        if abs(q.sum() - unknown_charge[name]) > 1e-4:
            raise RuntimeError(
                f"RESP charges of {label} sum to {q.sum():+.5f}, not {unknown_charge[name]:+d}"
            )
        charges[residue] = q
        sources[residue] = f"RESP {level} at PDB geometry; vsg element radii"

    for residue in todo:
        if residue in charges:
            continue
        atoms = structure.residue_atoms(residue)
        name = structure.res_names[atoms[0]]
        first = first_of[name]
        first_atoms = structure.residue_atoms(first)
        by_name = {
            structure.atom_names[i]: q for i, q in zip(first_atoms, charges[first], strict=True)
        }
        names = [structure.atom_names[i] for i in atoms]
        if len(by_name) != len(first_atoms) or sorted(names) != sorted(by_name):
            raise ConfigValidationError(
                f"resp_copies = 'one': {structure.residue_label(residue)} does not have the same "
                f"unique atom names as {structure.residue_label(first)}; use resp_copies = 'each'"
            )
        charges[residue] = np.array([by_name[n] for n in names])
        sources[residue] = (
            f"RESP charges reused from {structure.residue_label(first)}; vsg element radii"
        )
    return UnknownResult(charges, radii, sources, hf_runs=len(plan))

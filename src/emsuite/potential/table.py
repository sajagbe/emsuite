"""The per-atom charge and radius table of a potential run, and the files made from it."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emsuite.config.schemas import ConfigValidationError

from .amber import assign_amber
from .spec import PotentialSpec
from .structure import Structure, load_structure
from .unknown import RespFunction, assign_unknown


@dataclass(frozen=True)
class ParameterTable:
    structure: Structure
    charges: np.ndarray  # (N,) e, after the off switches
    radii: np.ndarray  # (N,) Å, after the off switches
    off: np.ndarray  # (N,) bool: charge 0 and radius 0
    uncharged: np.ndarray  # (N,) bool: charge 0, radius kept
    residue_source: dict[int, str]  # residue index -> where its parameters came from
    hf_runs: int = 0

    def write_pqr(self, path: str | Path) -> Path:
        """Whitespace-separated PQR: record serial name resname [chain] resseq x y z q r."""
        s = self.structure
        with open(path, "w") as handle:
            for i in range(s.n_atoms):
                chain = f"{s.chains[i]} " if s.chains[i] else ""
                x, y, z = s.coords[i]
                handle.write(
                    f"ATOM {i + 1} {s.atom_names[i]} {s.res_names[i]} {chain}{s.res_seqs[i]} "
                    f"{x:.3f} {y:.3f} {z:.3f} {self.charges[i]:.6f} {self.radii[i]:.4f}\n"
                )
        return Path(path)

    def residue_rows(self) -> list[dict[str, object]]:
        s = self.structure
        rows = []
        for residue in sorted(self.residue_source):
            atoms = s.residue_atoms(residue)
            first = int(atoms[0])
            rows.append(
                {
                    "residue": s.res_names[first],
                    "chain": s.chains[first],
                    "number": s.res_seqs[first],
                    "atoms": len(atoms),
                    "source": self.residue_source[residue],
                    "atoms_off": int(self.off[atoms].sum()),
                    "atoms_uncharged": int(self.uncharged[atoms].sum()),
                    "net_charge_e": round(float(self.charges[atoms].sum()), 4),
                }
            )
        return rows

    def write_sources(self, path: str | Path) -> Path:
        rows = self.residue_rows()
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return Path(path)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for row in self.residue_rows():
            kind = str(row["source"]).split(":")[0].split(";")[0]
            counts[kind] = counts.get(kind, 0) + 1
        lines = [f"  {n:5d} residues: {kind}" for kind, n in sorted(counts.items())]
        lines.append(f"  atoms off (charge 0, radius 0): {int(self.off.sum())}")
        lines.append(f"  atoms uncharged (charge 0, radius kept): {int(self.uncharged.sum())}")
        lines.append(f"  total charge in PQR: {self.charges.sum():+.4f} e")
        return "\n".join(lines)


def _mask(structure: Structure, selections, key: str, pdb: str) -> np.ndarray:
    """Atoms named by the selections of one input key; every selection must hit something."""
    rows = list(
        zip(
            structure.res_names,
            structure.chains,
            structure.res_seqs,
            structure.atom_names,
            strict=True,
        )
    )
    mask = np.zeros(structure.n_atoms, dtype=bool)
    for selection in selections:
        hits = np.array([selection.matches(*row) for row in rows], dtype=bool)
        if not hits.any():
            raise ConfigValidationError(
                f"{key} entry for residue {selection.resname!r} "
                f"(chain {selection.chain}, number {selection.resseq}, atom {selection.atom}) "
                f"matches no atom in {pdb}"
            )
        mask |= hits
    return mask


def build_parameter_table(spec: PotentialSpec, resp: RespFunction | None = None) -> ParameterTable:
    """PDB -> charge and radius for every atom, following the input file."""
    structure = load_structure(spec.pdb, add_hydrogens=spec.add_hydrogens, ph=spec.ph)
    off = _mask(structure, spec.off, "off", spec.pdb)
    uncharged = _mask(structure, spec.uncharged, "uncharged", spec.pdb) & ~off

    amber = assign_amber(structure, spec.forcefield)
    unknown = assign_unknown(
        structure,
        amber.unmatched,
        spec.unknown_charge,
        spec.unknown_spin,
        resp_copies=spec.resp_copies,
        skip=off | uncharged,  # atoms whose charge ends at zero need no RESP
        resp=resp,
    )
    charges = amber.charges.copy()
    radii = amber.radii.copy()
    source = {index: f"Amber template: {name}" for index, name in amber.templates.items()}
    for residue in amber.unmatched:
        atoms = structure.residue_atoms(residue)
        charges[atoms] = unknown.charges[residue]
        radii[atoms] = unknown.radii[residue]
        source[residue] = unknown.sources[residue]
    if np.isnan(charges).any() or np.isnan(radii).any():
        raise RuntimeError("some atoms were left without a charge or radius")
    charges[off | uncharged] = 0.0
    radii[off] = 0.0
    return ParameterTable(
        structure, charges, radii, off, uncharged, source, hf_runs=unknown.hf_runs
    )

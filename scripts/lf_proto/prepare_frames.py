#!/usr/bin/env python3
"""Extract middle frame from multi-frame GROMACS .gro into ligand/protein XYZ."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NM_TO_ANGSTROM = 10.0
SKIP_RESIDUES = frozenset({"CHR", "SOL", "CL", "NA", "K", "MG", "CA", "ZN"})


def parse_gro_line(line: str) -> tuple[str, str, str, float, float, float]:
    """Parse one GROMACS .gro atom line (fixed-width format)."""
    residue_name = line[5:10].strip()
    atom_name = line[10:15].strip()
    x = float(line[20:28])
    y = float(line[28:36])
    z = float(line[36:44])
    return residue_name, atom_name, element_from_atom_name(atom_name), x, y, z


def element_from_atom_name(atom_name: str) -> str:
    atom_name = atom_name.strip()
    if not atom_name:
        return "X"
    if len(atom_name) >= 2 and atom_name[1].islower():
        return atom_name[:2].capitalize()
    return atom_name[0].upper()


def read_gro_frames(path: Path) -> list[list[tuple[str, str, str, float, float, float]]]:
    frames: list[list[tuple[str, str, str, float, float, float]]] = []
    with path.open() as handle:
        while True:
            title = handle.readline()
            if not title:
                break
            natoms_line = handle.readline()
            if not natoms_line:
                break
            natoms = int(natoms_line.split()[0])
            atoms = [parse_gro_line(handle.readline()) for _ in range(natoms)]
            handle.readline()  # box line
            frames.append(atoms)
    return frames


def write_xyz(path: Path, atoms: list[tuple[str, str, str, float, float, float]]) -> None:
    lines = [str(len(atoms))]
    lines.extend(
        f"{element:>2s}  {x * NM_TO_ANGSTROM:12.6f}  {y * NM_TO_ANGSTROM:12.6f}  {z * NM_TO_ANGSTROM:12.6f}"
        for _, _, element, x, y, z in atoms
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gro", required=True, type=Path, help="Input multi-frame .gro file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for XYZ files")
    parser.add_argument("--system", required=True, help="System label (e.g. AT1)")
    args = parser.parse_args()

    frames = read_gro_frames(args.gro)
    if not frames:
        raise SystemExit(f"No frames found in {args.gro}")

    frame_idx = len(frames) // 2
    atoms = frames[frame_idx]

    ligand = [atom for atom in atoms if atom[0] == "CHR"]
    protein = [atom for atom in atoms if atom[0] not in SKIP_RESIDUES]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ligand_path = args.out_dir / "ligand.xyz"
    protein_path = args.out_dir / "protein.xyz"
    write_xyz(ligand_path, ligand)
    write_xyz(protein_path, protein)

    metadata = {
        "system": args.system,
        "source_gro": str(args.gro.resolve()),
        "n_frames": len(frames),
        "frame_index": frame_idx,
        "frame_number_1based": frame_idx + 1,
        "n_ligand_atoms": len(ligand),
        "n_protein_atoms": len(protein),
        "ligand_xyz": str(ligand_path.resolve()),
        "protein_xyz": str(protein_path.resolve()),
        "coord_units": "angstrom",
        "coord_scale": NM_TO_ANGSTROM,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        f"{args.system}: frame {frame_idx + 1}/{len(frames)} "
        f"({len(ligand)} ligand, {len(protein)} protein atoms)"
    )


if __name__ == "__main__":
    main()

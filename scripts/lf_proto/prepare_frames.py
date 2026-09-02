#!/usr/bin/env python3
"""Extract middle frame from multi-frame GROMACS .gro into ligand XYZ and complex PDB."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

NM_TO_ANGSTROM = 10.0
SKIP_RESIDUES = frozenset({"CHR", "SOL", "CL", "NA", "K", "MG", "CA", "ZN"})
LIGAND_RESNAME = "CHR"


def parse_gro_line(
    line: str,
) -> tuple[str, int, str, str, float, float, float]:
    """Parse one GROMACS .gro atom line (fixed-width format)."""
    residue_number = int(line[0:5].strip())
    residue_name = line[5:10].strip()
    atom_name = line[10:15].strip()
    x = float(line[20:28])
    y = float(line[28:36])
    z = float(line[36:44])
    return (
        residue_name,
        residue_number,
        atom_name,
        element_from_atom_name(atom_name),
        x,
        y,
        z,
    )


def element_from_atom_name(atom_name: str) -> str:
    """Map a GROMACS atom name to an element symbol for XYZ/PDB."""
    cleaned = re.sub(r"\d+", "", atom_name.strip())
    if not cleaned:
        return "X"
    if len(cleaned) >= 2 and cleaned[1].islower():
        return cleaned[:2].capitalize()
    return cleaned[0].upper()


def read_gro_frames(
    path: Path,
) -> list[list[tuple[str, int, str, str, float, float, float]]]:
    frames: list[list[tuple[str, int, str, str, float, float, float]]] = []
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


def validate_xyz(path: Path) -> None:
    """Validate standard XYZ format expected by EMSuite."""
    lines = path.read_text().strip().splitlines()
    if len(lines) < 2:
        raise ValueError(f"{path}: XYZ file must have at least a header and comment line")
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"{path}: invalid atom count on line 1: {lines[0]!r}") from exc
    expected_lines = 2 + n_atoms
    if len(lines) != expected_lines:
        raise ValueError(
            f"{path}: expected {expected_lines} lines (2 header + {n_atoms} atoms), got {len(lines)}"
        )
    for line_no, line in enumerate(lines[2:], start=3):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}: missing coordinates on line {line_no}: {line!r}")
        symbol = parts[0]
        if not symbol or not symbol[0].isalpha():
            raise ValueError(f"{path}: invalid element symbol on line {line_no}: {symbol!r}")


def write_xyz(
    path: Path,
    atoms: list[tuple[str, int, str, str, float, float, float]],
    comment: str = "",
) -> None:
    lines = [str(len(atoms)), comment]
    lines.extend(
        f"{element}  {x * NM_TO_ANGSTROM:.6f}  {y * NM_TO_ANGSTROM:.6f}  {z * NM_TO_ANGSTROM:.6f}"
        for _, _, _, element, x, y, z in atoms
    )
    path.write_text("\n".join(lines) + "\n")
    validate_xyz(path)


def pdb_atom_name(atom_name: str) -> str:
    """Format a GROMACS atom name for a PDB ATOM/HETATM record."""
    name = atom_name.strip()
    if len(name) <= 3:
        return f" {name:<3}"
    return name[:4]


def write_pdb(
    path: Path,
    atoms: list[tuple[str, int, str, str, float, float, float]],
) -> None:
    """Write protein + CHR ligand as ATOM/HETATM records (coords in Å)."""
    lines: list[str] = []
    serial = 1
    for residue_name, residue_number, atom_name, element, x, y, z in atoms:
        record = "HETATM" if residue_name == LIGAND_RESNAME else "ATOM  "
        x_ang = x * NM_TO_ANGSTROM
        y_ang = y * NM_TO_ANGSTROM
        z_ang = z * NM_TO_ANGSTROM
        lines.append(
            f"{record}{serial:5d} {pdb_atom_name(atom_name)} {residue_name:>3} A"
            f"{residue_number:4d}    {x_ang:8.3f}{y_ang:8.3f}{z_ang:8.3f}"
            f"  1.00  0.00          {element:>2}  "
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n")


def parse_mol2_atoms(text: str) -> tuple[str, list[list[str]], list[list[str]]]:
    """Return (molecule_header, atom_lines_as_tokens, bond_lines_as_tokens)."""
    sections: dict[str, list[list[str]]] = {}
    current = ""
    for line in text.splitlines():
        if line.startswith("@<TRIPOS>"):
            current = line.split("@<TRIPOS>", 1)[1].strip()
            sections[current] = []
            continue
        if not current or not line.strip():
            continue
        sections[current].append(line.split())

    header = ""
    if "MOLECULE" in sections and sections["MOLECULE"]:
        header = " ".join(sections["MOLECULE"][0])
    return header, sections.get("ATOM", []), sections.get("BOND", [])


def remap_mol2_atom_names(
    source_mol2: Path,
    atom_names: list[str],
    dest_mol2: Path,
) -> None:
    """Rename MOL2 atoms by index so names match the PDB/GROMACS ligand."""
    text = source_mol2.read_text()
    header, atoms, bonds = parse_mol2_atoms(text)
    if len(atoms) < len(atom_names):
        raise ValueError(
            f"{source_mol2}: expected at least {len(atom_names)} atoms, found {len(atoms)}"
        )

    n_atoms = len(atom_names)
    renamed_atoms: list[str] = []
    for idx, tokens in enumerate(atoms[:n_atoms], start=1):
        if len(tokens) < 8:
            raise ValueError(f"{source_mol2}: malformed ATOM line {idx}: {tokens!r}")
        tokens[1] = atom_names[idx - 1]
        renamed_atoms.append(
            f"{int(tokens[0]):7d} {tokens[1]:<4} "
            f"{float(tokens[2]):9.4f} {float(tokens[3]):9.4f} {float(tokens[4]):9.4f} "
            f"{tokens[5]:<6} {tokens[6]:<3} {tokens[7]:<10} {tokens[8]}"
        )

    kept_bonds = []
    for tokens in bonds:
        if len(tokens) < 4:
            continue
        a, b = int(tokens[1]), int(tokens[2])
        if a <= n_atoms and b <= n_atoms:
            kept_bonds.append(f"{int(tokens[0]):5d} {a:5d} {b:5d}  {tokens[3]}")

    lines = [
        "@<TRIPOS>MOLECULE",
        header or "CHR",
        f"{n_atoms} {len(kept_bonds)} 0 0 0",
        "SMALL",
        "GASTEIGER",
        "",
        "@<TRIPOS>ATOM",
        *renamed_atoms,
        "@<TRIPOS>BOND",
        *kept_bonds,
    ]
    dest_mol2.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gro", required=True, type=Path, help="Input multi-frame .gro file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory for frame files")
    parser.add_argument("--system", required=True, help="System label (e.g. AT1)")
    parser.add_argument(
        "--mol2-source",
        type=Path,
        default=None,
        help="Source FMN MOL2 to remap atom names for pdb2pqr (writes CHR.mol2)",
    )
    args = parser.parse_args()

    frames = read_gro_frames(args.gro)
    if not frames:
        raise SystemExit(f"No frames found in {args.gro}")

    frame_idx = len(frames) // 2
    atoms = frames[frame_idx]

    ligand = [atom for atom in atoms if atom[0] == LIGAND_RESNAME]
    protein = [atom for atom in atoms if atom[0] not in SKIP_RESIDUES]
    complex_atoms = protein + ligand

    if not ligand:
        raise SystemExit(f"No {LIGAND_RESNAME} ligand atoms found in {args.gro}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ligand_path = args.out_dir / "ligand.xyz"
    complex_path = args.out_dir / "complex.pdb"
    write_xyz(ligand_path, ligand, comment=f"{args.system} ligand from {args.gro.name}")
    write_pdb(complex_path, complex_atoms)

    ligand_atom_names = [atom[2] for atom in ligand]
    mol2_path = args.out_dir / "CHR.mol2"
    if args.mol2_source is not None:
        remap_mol2_atom_names(args.mol2_source, ligand_atom_names, mol2_path)

    metadata = {
        "system": args.system,
        "source_gro": str(args.gro.resolve()),
        "n_frames": len(frames),
        "frame_index": frame_idx,
        "frame_number_1based": frame_idx + 1,
        "n_ligand_atoms": len(ligand),
        "n_protein_atoms": len(protein),
        "n_complex_atoms": len(complex_atoms),
        "ligand_xyz": str(ligand_path.resolve()),
        "complex_pdb": str(complex_path.resolve()),
        "ligand_mol2": str(mol2_path.resolve()) if mol2_path.exists() else None,
        "ligand_resname": LIGAND_RESNAME,
        "coord_units": "angstrom",
        "coord_scale": NM_TO_ANGSTROM,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    print(
        f"{args.system}: frame {frame_idx + 1}/{len(frames)} "
        f"({len(ligand)} ligand, {len(protein)} protein atoms, complex.pdb written)"
    )


if __name__ == "__main__":
    main()

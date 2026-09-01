"""Explicit PDB residue selection (line-level, no structural parser).

Selection is by ``(resname, chain, resseq)`` — the same identifiers a PQR line
carries — rather than relying on pdb2pqr's own ``--ligand`` atom-name matching,
which has no residue-identity check and can collide with any other HETATM
residue sharing atom names with the target MOL2 (see pdb2pqr main.py's
``--ligand`` loop: it matches every HETATM residue's atoms by bare name against
the single supplied MOL2, with no residue check at all).
"""

from __future__ import annotations

from pathlib import Path


def _is_hetatm(line: str) -> bool:
    return line[0:6] == "HETATM"


def _matches(line: str, resname: str, chain: str | None, resseq: int | None) -> bool:
    if line[17:20].strip() != resname:
        return False
    if chain is not None and line[21:22].strip() != chain:
        return False
    if resseq is not None and int(line[22:26]) != resseq:
        return False
    return True


def select_residue_lines(
    pdb_path: str | Path,
    resname: str,
    chain: str | None = None,
    resseq: int | None = None,
) -> list[str]:
    """Return the HETATM lines for one residue, erroring if the match isn't unique.

    Uniqueness is checked by distinct (chain, resseq) pairs among matching lines,
    not by line count, since a residue has multiple atom lines.
    """
    lines = Path(pdb_path).read_text().splitlines(keepends=True)
    matched = [line for line in lines if _is_hetatm(line) and _matches(line, resname, chain, resseq)]
    if not matched:
        raise ValueError(f"No HETATM residue named {resname!r} found in {pdb_path}")
    keys = {(line[21:22].strip(), line[22:26].strip()) for line in matched}
    if len(keys) > 1:
        raise ValueError(
            f"{resname!r} matches {len(keys)} distinct residues in {pdb_path} "
            f"(chain, resseq): {sorted(keys)} — pass ligand_chain/ligand_resseq to disambiguate"
        )
    return matched


def strip_residue(
    pdb_path: str | Path,
    resname: str,
    chain: str | None,
    resseq: int | None,
    output_path: str | Path,
) -> Path:
    """Write a copy of ``pdb_path`` with exactly one residue's lines removed.

    Everything else (protein ATOM lines, other HETATM residues) is kept.
    """
    select_residue_lines(pdb_path, resname, chain, resseq)  # validates uniqueness
    lines = Path(pdb_path).read_text().splitlines(keepends=True)
    kept = [
        line
        for line in lines
        if not (_is_hetatm(line) and _matches(line, resname, chain, resseq))
    ]
    output = Path(output_path)
    output.write_text("".join(kept))
    return output


def isolate_residue(
    pdb_path: str | Path,
    resname: str,
    chain: str | None,
    resseq: int | None,
    output_path: str | Path,
) -> Path:
    """Write a copy of ``pdb_path`` keeping ATOM lines plus only the target residue.

    Drops every *other* HETATM residue, so pdb2pqr's ``--ligand`` atom-name
    matching (which checks every HETATM residue, not just the target) has
    nothing else in the file it could collide with.
    """
    select_residue_lines(pdb_path, resname, chain, resseq)  # validates uniqueness
    lines = Path(pdb_path).read_text().splitlines(keepends=True)
    kept = [
        line
        for line in lines
        if not _is_hetatm(line) or _matches(line, resname, chain, resseq)
    ]
    output = Path(output_path)
    output.write_text("".join(kept))
    return output

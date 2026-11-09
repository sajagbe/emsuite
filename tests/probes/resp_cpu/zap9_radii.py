"""Per-atom radii for a ligand under five radius sets, side by side.

Usage: python zap9_radii.py LF        (molecules as in compare_gpu.py)
Writes <name>_radii.txt.

Radius sets
  Mantina      what emsuite writes into the PQR today (qcelemental, Mantina 2009)
  Bondi        Bondi 1964, as listed in Nicholls 2008 Table 2 / Nicholls 2010 Table 2
  pdb2pqr      what pdb2pqr 3.7.1 `--ligand` assigns (its "zap9" dict, Bondi fallback)
  ZAP9-2008    Nicholls et al., J. Med. Chem. 2008, 51, 769, Table 2
  ZAP9-2010    Nicholls et al., J. Comput.-Aided Mol. Des. 2010, 24, 293, Table 2 + text:
               acetamide (amide) carbonyl O back to Bondi 1.52; O doubly bonded to
               S, P or N gets the carbonyl value 1.76 (OpenEye's current default).

ZAP9 typing rules (from the papers):
  C 1.87  H 1.10  S 2.15  F 2.40  Cl 1.82  I 2.65
  carbonyl O (O=C) 1.76, other O Bondi 1.52
  secondary/tertiary N 1.40, other N Bondi 1.55
  everything else Bondi
"Secondary/tertiary" is taken here as N with >= 2 heavy-atom neighbours. The papers
define it for amines; applying it to imine / ring N is our interpretation (flagged).
"""

from __future__ import annotations

import sys
from pathlib import Path

import qcelemental as qcel
from compare_gpu import MOLECULES
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

HERE = Path(__file__).resolve().parent

BONDI = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
    "Si": 2.10,
}
ZAP9_FIXED = {"C": 1.87, "H": 1.10, "S": 2.15, "F": 2.40, "Cl": 1.82, "I": 2.65}
# pdb2pqr 3.7.1 pdb2pqr/ligand/__init__.py RADII["zap9"], keyed by element or Sybyl type
PDB2PQR_ZAP9 = {
    "C": 1.87,
    "H": 1.10,
    "O.co2": 1.76,
    "N": 1.40,
    "S": 2.15,
    "F": 2.40,
    "Cl": 1.82,
    "I": 2.65,
}


def bondi(sym):
    if sym in BONDI:
        return BONDI[sym]
    return float(qcel.vdwradii.get(sym, units="angstrom"))  # no Bondi value: Mantina


def heavy_neighbours(atom):
    return [n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]


def double_partner(atom):
    for b in atom.GetBonds():
        if b.GetBondType() == Chem.BondType.DOUBLE:
            return b.GetOtherAtom(atom)
    return None


def is_amide_carbon(c):
    return any(n.GetSymbol() == "N" for n in c.GetNeighbors())


def is_carboxylate_o(o):
    # Sybyl O.co2: O on a carbon that carries two terminal oxygens
    c = heavy_neighbours(o)
    if len(c) != 1 or c[0].GetSymbol() != "C":
        return False
    term_o = [n for n in c[0].GetNeighbors() if n.GetSymbol() == "O" and n.GetDegree() == 1]
    return len(term_o) == 2


def radii_for(atom):
    sym = atom.GetSymbol()
    note = ""
    # ZAP9 2008 / 2010
    if sym in ZAP9_FIXED:
        z08 = z10 = ZAP9_FIXED[sym]
    elif sym == "N":
        if len(heavy_neighbours(atom)) >= 2:
            z08 = z10 = 1.40
            amine = all(b.GetBondType() == Chem.BondType.SINGLE for b in atom.GetBonds())
            if not amine or atom.GetIsAromatic():
                note = "N typed 2ry/3ry by heavy-neighbour count (not an amine)"
        else:
            z08 = z10 = BONDI["N"]
    elif sym == "O":
        p = double_partner(atom)
        if p is not None and p.GetSymbol() == "C":
            z08 = 1.76
            z10 = BONDI["O"] if is_amide_carbon(p) else 1.76
            if is_amide_carbon(p):
                note = "amide/imide carbonyl O: 1.76 (2008) -> 1.52 (2010)"
        elif p is not None and p.GetSymbol() in ("S", "P", "N"):
            z08, z10 = BONDI["O"], 1.76
            note = f"O={p.GetSymbol()}: 1.52 (2008) -> 1.76 (2010)"
        else:
            z08 = z10 = BONDI["O"]
    else:
        z08 = z10 = bondi(sym)
    # pdb2pqr
    if sym == "O" and is_carboxylate_o(atom):
        p2p = PDB2PQR_ZAP9["O.co2"]
    else:
        p2p = PDB2PQR_ZAP9.get(sym, bondi(sym))
    mant = float(qcel.vdwradii.get(sym, units="angstrom"))
    return mant, bondi(sym), p2p, z08, z10, note


def main(name: str) -> None:
    xyz, charge = MOLECULES[name]
    mol = Chem.MolFromXYZFile(str(xyz))
    rdDetermineBonds.DetermineBonds(mol, charge=charge)
    lines = [
        f"{name}  charge {charge}  ({mol.GetNumAtoms()} atoms)   radii in Angstrom",
        "pdb2pqr = what pdb2pqr --ligand assigns today; ZAP9-2008/2010 = the papers' rules",
        "",
        f"{'atom':<6}{'bonded to':<16}{'Mantina':>8}{'Bondi':>8}{'pdb2pqr':>9}"
        f"{'ZAP9-08':>9}{'ZAP9-10':>9}  note",
    ]
    for a in mol.GetAtoms():
        i = a.GetIdx()
        nb = "".join(sorted(n.GetSymbol() for n in a.GetNeighbors()))
        mant, bon, p2p, z08, z10, note = radii_for(a)
        flag = " *" if abs(p2p - z10) > 1e-6 else ""
        lines.append(
            f"{a.GetSymbol()}{i + 1:<5}{nb:<16}{mant:>8.2f}{bon:>8.2f}{p2p:>9.2f}"
            f"{z08:>9.2f}{z10:>9.2f}  {note}{flag and '  [pdb2pqr differs from paper]'}"
        )
    lines += [
        "",
        "ZAP9 was fitted with AM1-BCC charges (designed to mimic HF/6-31G* RESP),",
        "interior dielectric 1, water 80, nonpolar term 6.3 cal/(mol A^2).",
    ]
    out = HERE / f"{name}_radii.txt"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", out)


if __name__ == "__main__":
    for n in sys.argv[1:] or ["LF", "SB", "rPSB"]:
        main(n)

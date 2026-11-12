"""Amber charges and radii for the residues OpenMM recognizes.

OpenMM matches each residue to a template of the chosen force-field XML files by
its elements and bonds (not by its name). Matched residues get the template's
charge and the Lennard-Jones size of each atom type; the PQR radius is Rmin/2.
Atoms with zero well depth (for example hydroxyl H) get radius 0. Residues with
no matching template are returned as unmatched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from emsuite.config.schemas import ConfigValidationError

from .structure import Structure

_SIGMA_NM_TO_RMIN_HALF_A = 10.0 * 2.0 ** (1.0 / 6.0) / 2.0


@dataclass(frozen=True)
class AmberAssignment:
    charges: np.ndarray  # (N,) e; NaN on unmatched residues
    radii: np.ndarray  # (N,) Å; NaN on unmatched residues
    templates: dict[int, str]  # residue index -> template name
    unmatched: tuple[int, ...]  # residue indices with no template


def _prefer_named_template(forcefield, residue, bonded_to_atom, ignore_external, ignore_extra):
    """Pick one template when several different ones fit a residue.

    The same atoms and bonds can fit templates of two force fields (the acetyl cap is
    ACE in ff14SB and ACX in GLYCAM). OpenMM stops in that case. Here the template
    named like the residue wins; otherwise the one from the force field listed first.
    Returns None when there is nothing to decide, so OpenMM's normal matching runs.
    """
    from openmm.app import forcefield as ff_module

    signature = ff_module._createResidueSignature([atom.element for atom in residue.atoms()])
    fits = [
        template
        for template in forcefield._templateSignatures.get(signature, [])
        if ff_module.compiled.matchResidueToTemplate(
            residue, template, bonded_to_atom, ignore_external, ignore_extra
        )
        is not None
    ]
    if len(fits) < 2:
        return None
    for template in fits:
        if template.name == residue.name:
            return template
    order = {name: rank for rank, name in enumerate(forcefield._templates)}
    return min(fits, key=lambda template: order.get(template.name, len(order)))


def load_forcefield(names: tuple[str, ...]):
    from openmm import app

    try:
        forcefield = app.ForceField(*names)
    except Exception as exc:
        raise ConfigValidationError(
            f"could not load force field files {list(names)}: {exc}. Names are OpenMM XML "
            "files, for example 'amber14/protein.ff14SB.xml'."
        ) from exc
    forcefield.registerTemplateMatcher(_prefer_named_template)
    return forcefield


def assign_amber(structure: Structure, forcefield_names: tuple[str, ...]) -> AmberAssignment:
    from openmm import NonbondedForce, app, unit

    forcefield = load_forcefield(forcefield_names)
    topology = structure.topology
    unmatched = forcefield.getUnmatchedResidues(topology)
    unmatched_index = tuple(sorted(r.index for r in unmatched))

    charges = np.full(structure.n_atoms, np.nan)
    radii = np.full(structure.n_atoms, np.nan)
    templates: dict[int, str] = {}
    kept_residues = [r for r in topology.residues() if r.index not in set(unmatched_index)]
    if not kept_residues:
        return AmberAssignment(charges, radii, templates, unmatched_index)

    modeller = app.Modeller(topology, structure.coords * unit.angstrom)
    modeller.delete(unmatched)
    try:
        system = forcefield.createSystem(modeller.topology)
        matched = forcefield.getMatchingTemplates(modeller.topology)
    except Exception as exc:
        raise ConfigValidationError(
            "OpenMM could not assign Amber parameters after the unrecognized residues "
            f"({', '.join(structure.residue_label(i) for i in unmatched_index) or 'none'}) were "
            f"set aside: {exc}. A recognized residue is probably bonded to an unrecognized one; "
            "that case is not supported yet."
        ) from exc
    nonbonded = next(f for f in system.getForces() if isinstance(f, NonbondedForce))

    kept_atoms = np.flatnonzero(~np.isin(structure.res_index, unmatched_index))
    if len(kept_atoms) != system.getNumParticles():
        raise RuntimeError("atom count changed while assigning Amber parameters")
    for sub_index, atom_index in enumerate(kept_atoms):
        charge, sigma, epsilon = nonbonded.getParticleParameters(sub_index)
        charges[atom_index] = charge.value_in_unit(unit.elementary_charge)
        if epsilon.value_in_unit(unit.kilojoule_per_mole) == 0.0:
            radii[atom_index] = 0.0
        else:
            radii[atom_index] = sigma.value_in_unit(unit.nanometer) * _SIGMA_NM_TO_RMIN_HALF_A
    for residue, template in zip(kept_residues, matched, strict=True):
        templates[residue.index] = template.name
    return AmberAssignment(charges, radii, templates, unmatched_index)

"""Tuning calculation orchestration."""

import json
import logging
import os
import shutil
import sys
from datetime import datetime

import numpy as np
import ray

from emsuite import core
from emsuite.surface import load_surf

from .config_io import get_tuning_parameters
from .logging import (
    append_point_to_summary,
    finalize_summary_log,
    initialize_summary_log,
    log_point_result,
    setup_logs_directory,
)
from .output import (
    append_raw_properties_to_summary,
    check_all_files_created,
    create_output_files,
    organize_results,
)
from .parallel import calculate_point_effect_cpu_remote, calculate_point_effect_gpu
from .properties import calculate_all_properties, interaction_effect_kcal, setup_calculation
from .properties.interaction import water_probe_coords_and_charges
from .resume import (
    create_resume_metadata,
    find_incomplete_logs,
    load_completed_results_from_logs,
    prompt_user_resume,
    update_resume_metadata,
    validate_resume_compatibility,
)

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def calculate_surface_effect_at_point(
    base_chkfiles,
    coord,
    surface_charge,
    solvent,
    state_of_interest,
    triplet,
    properties_to_calculate,
    required_calculations,
    force_single_gpu=False,
):
    """
    Calculate the effect of a surface charge at a single coordinate point.

    Args:
        base_chkfiles (dict): Dictionary with keys 'neutral', 'anion', 'cation'
                             pointing to checkpoint file paths
        coord (array-like): 3D coordinates [x, y, z] of the surface charge
        surface_charge (float): Magnitude of the point charge
        solvent (str or None): Solvent for implicit solvation
        state_of_interest (int): Number of excited states for TD calculations
        triplet (bool): Whether to calculate triplet excited states
        properties_to_calculate (list): List of molecular properties to compute
        required_calculations (dict): Dictionary specifying needed calculations
        functional (str): XC functional for DFT calculations
        force_single_gpu (bool): Skip TD subprocess isolation (for Ray workers)

    Returns:
        dict: Dictionary of property effects
    """
    # BACKUP CHECKPOINT FILES
    backup_files = {}
    for key, chkfile in base_chkfiles.items():
        if chkfile and os.path.exists(chkfile):
            backup_file = f"{chkfile}.bak"
            shutil.copy2(chkfile, backup_file)
            backup_files[key] = backup_file

    try:
        # Resurrect base molecules from checkpoint files
        molecule_alone = (
            core.resurrect_mol(base_chkfiles["neutral"]) if base_chkfiles.get("neutral") else None
        )
        anion_alone = (
            core.resurrect_mol(base_chkfiles["anion"]) if base_chkfiles.get("anion") else None
        )
        cation_alone = (
            core.resurrect_mol(base_chkfiles["cation"]) if base_chkfiles.get("cation") else None
        )

        # Create TD object if needed - pass force_single_gpu flag
        td_alone = None
        if required_calculations.get("td", False) and molecule_alone:
            td_alone = core.create_td_molecule_object(
                molecule_alone,
                nstates=state_of_interest,
                triplet=triplet,
                force_single_gpu=force_single_gpu,
            )

        # Create single-point charge array (optionally include explicit water probe)
        single_coord = np.array([coord])
        q_mm = np.array([surface_charge])
        if "h2o" in properties_to_calculate:
            w_coords, w_charges = water_probe_coords_and_charges(np.asarray(coord))
            single_coord = np.vstack([single_coord, w_coords])
            q_mm = np.concatenate([q_mm, w_charges])

        # Create QM/MM objects for this point
        molecule_wsc, anion_wsc, cation_wsc, td_wsc = create_wsc_objects(
            [molecule_alone, anion_alone, cation_alone, td_alone],
            single_coord,
            q_mm,
            state_of_interest,
            triplet,
            required_calculations,
        )

        # Apply solvation if needed
        if solvent:
            all_molecules = [
                molecule_alone,
                anion_alone,
                cation_alone,
                td_alone,
                molecule_wsc,
                anion_wsc,
                cation_wsc,
                td_wsc,
            ]
            all_molecules = apply_solvation(
                all_molecules, solvent, state_of_interest, triplet, required_calculations
            )
            (
                molecule_alone,
                anion_alone,
                cation_alone,
                td_alone,
                molecule_wsc,
                anion_wsc,
                cation_wsc,
                td_wsc,
            ) = all_molecules

        # Calculate properties
        results = calculate_all_properties(
            molecule_alone,
            anion_mf=anion_alone,
            cation_mf=cation_alone,
            td_obj=td_alone,
            triplet=triplet,
            props_to_calc=properties_to_calculate,
            probe_coord=np.asarray(coord),
            probe_charge=float(surface_charge),
        )
        wsc_results = calculate_all_properties(
            molecule_wsc,
            anion_mf=anion_wsc,
            cation_mf=cation_wsc,
            td_obj=td_wsc,
            triplet=triplet,
            props_to_calc=properties_to_calculate,
            probe_coord=np.asarray(coord),
            probe_charge=float(surface_charge),
        )

        if "eint" in properties_to_calculate:
            results["eint"] = 0.0
            wsc_results["eint"] = interaction_effect_kcal(molecule_alone, molecule_wsc)
        if "h2o" in properties_to_calculate:
            results["h2o"] = 0.0
            wsc_results["h2o"] = interaction_effect_kcal(molecule_alone, molecule_wsc)

        # Calculate differences
        effects = {}
        for prop in results:
            if prop in wsc_results:
                effects[f"{prop}_effect"] = wsc_results[prop] - results[prop]

        return effects

    finally:
        # RESTORE CHECKPOINT FILES FROM BACKUP
        for key, backup_file in backup_files.items():
            original_file = base_chkfiles[key]
            if os.path.exists(backup_file):
                shutil.move(backup_file, original_file)  # Restore original


##########################################################
#        Surface Data Loading and Validation             #
##########################################################


def calculate_combined_surface_effect(
    base_chkfiles,
    coords,
    charges,
    solvent,
    state_of_interest,
    triplet,
    properties_to_calculate,
    required_calculations,
    functional,
):
    """
    Calculate the effect of all surface charges together in a single QM/MM calculation.

    Args:
        base_chkfiles (dict): Dictionary with checkpoint file paths
        coords (numpy.ndarray): All surface coordinates [N, 3]
        charges (numpy.ndarray): All surface charges [N]
        solvent (str or None): Solvent for implicit solvation
        state_of_interest (int): Number of excited states
        triplet (bool): Whether to calculate triplet states
        properties_to_calculate (list): Properties to compute
        required_calculations (dict): Required calculation types
        functional (str): XC functional

    Returns:
        dict: Dictionary of combined property effects
    """
    # Resurrect base molecules
    molecule_alone = (
        core.resurrect_mol(base_chkfiles["neutral"]) if base_chkfiles.get("neutral") else None
    )
    anion_alone = core.resurrect_mol(base_chkfiles["anion"]) if base_chkfiles.get("anion") else None
    cation_alone = (
        core.resurrect_mol(base_chkfiles["cation"]) if base_chkfiles.get("cation") else None
    )

    # Create TD object if needed
    td_alone = None
    if required_calculations.get("td", False) and molecule_alone:
        td_alone = core.create_td_molecule_object(
            molecule_alone, nstates=state_of_interest, triplet=triplet
        )

    # Create QM/MM objects with ALL charges at once
    molecule_wsc, anion_wsc, cation_wsc, td_wsc = create_wsc_objects(
        [molecule_alone, anion_alone, cation_alone, td_alone],
        coords,
        charges,
        state_of_interest,
        triplet,
        required_calculations,
    )

    # Apply solvation if needed
    if solvent:
        all_molecules = [
            molecule_alone,
            anion_alone,
            cation_alone,
            td_alone,
            molecule_wsc,
            anion_wsc,
            cation_wsc,
            td_wsc,
        ]
        all_molecules = apply_solvation(
            all_molecules, solvent, state_of_interest, triplet, required_calculations
        )
        (
            molecule_alone,
            anion_alone,
            cation_alone,
            td_alone,
            molecule_wsc,
            anion_wsc,
            cation_wsc,
            td_wsc,
        ) = all_molecules

    # Calculate properties
    results = calculate_all_properties(
        molecule_alone,
        anion_mf=anion_alone,
        cation_mf=cation_alone,
        td_obj=td_alone,
        triplet=triplet,
        props_to_calc=properties_to_calculate,
    )
    wsc_results = calculate_all_properties(
        molecule_wsc,
        anion_mf=anion_wsc,
        cation_mf=cation_wsc,
        td_obj=td_wsc,
        triplet=triplet,
        props_to_calc=properties_to_calculate,
    )

    # Calculate differences
    effects = {}
    for prop in results:
        if prop in wsc_results:
            effects[f"{prop}_effect"] = wsc_results[prop] - results[prop]

    return effects


##########################################################
# Create Molecular Objects for Any Requested Calculation #
##########################################################
def create_alone_molecule_objects(
    input_data, basis_set, method, functional, charge, charge_change, gpu_available, spin_guesses
):
    """
    Create individual molecule objects with specified charge modifications.

    This is a wrapper function that creates PySCF molecule objects using
    the provided parameters and applies charge modifications for ionic states.

    Args:
        input_data (str): Path to XYZ file with molecular coordinates
        basis_set (str): Basis set for quantum calculations
        method (str): Quantum calculation method (e.g., 'dft')
        functional (str): Functional for DFT calculations
        charge (int): Base molecular charge
        charge_change (int): Charge modification from neutral state (+1 for cation, -1 for anion)
        gpu_available (bool): Whether GPU is available for calculations
        spin_guesses (list or None): List of spin multiplicities to test, e.g [0,1] for singlet and doublet. If None, default of [0, 1, 2, 3, 4] are used.

    Returns:
        pyscf.scf object or None: Converged SCF object for the specified charge state
    """
    return core.create_molecule_object(
        atom_input=input_data,
        basis_set=basis_set,
        method=method,
        functional=functional,
        original_charge=charge,
        charge_change=charge_change,
        gpu=gpu_available,
        spin_guesses=spin_guesses,
    )


def create_molecule_objects(
    input_data,
    basis_set,
    spin_guesses,
    method,
    functional,
    charge,
    gpu_available,
    required_calculations,
    state_of_interest,
    triplet,
):
    """
    Create and save all required molecule objects for property calculations.

    This function creates SCF objects for neutral, anionic, and cationic states
    as needed, along with time-dependent objects for excited state calculations.
    All objects are saved as checkpoint files for later use.

    Args:
        input_data (str): Path to XYZ file with molecular coordinates
        basis_set (str): Basis set for quantum calculations
        method (str): Quantum calculation method (e.g., 'dft')
        functional (str): Functional for DFT calculations
        charge (int): Base molecular charge
        gpu_available (bool): Whether GPU is available for calculations
        required_calculations (dict): Dictionary specifying which calculations are needed
                                    (e.g., {'neutral': True, 'anion': False, ...})
        spin_guesses (list, optional): List of spin multiplicities to test.
                                     Defaults to [0, 1, 2, 3, 4]. Uses 2S notation not multiplicity (2S+1).
                                     Important for open-shell systems.

        state_of_interest (int): Number of excited states to calculate for TD
        triplet (bool): Whether to calculate triplet excited states

    Returns:
        list: [neutral_mf, anion_mf, cation_mf, td_obj] where elements are
              SCF/TD objects or None if not calculated

    Note:
        - Saves checkpoint files as 'molecule_alone.chk', 'anion_alone.chk', 'cation_alone.chk'
        - TD objects are only created if explicitly required
        - Charge states: neutral (0), anion (-1), cation (+1)
    """
    molecules = {}
    calc_configs = [("neutral", 0, spin_guesses), ("anion", -1, None), ("cation", +1, None)]
    for name, charge_change, spin_guesses in calc_configs:
        if required_calculations.get(name, False):
            molecules[name] = create_alone_molecule_objects(
                input_data,
                basis_set,
                method,
                functional,
                charge,
                charge_change,
                gpu_available,
                spin_guesses,
            )

    chkfile_map = {
        "neutral": "molecule_alone.chk",
        "anion": "anion_alone.chk",
        "cation": "cation_alone.chk",
    }
    for key, filename in chkfile_map.items():
        if molecules.get(key):
            core.save_chkfile(molecules[key], filename, functional)

    # Create TD object if needed
    td_obj = None
    if required_calculations.get("td", False) and molecules.get("neutral"):
        td_obj = core.create_td_molecule_object(
            molecules["neutral"],
            nstates=state_of_interest,
            triplet=triplet,
            force_single_gpu=not gpu_available,
        )

    return [molecules.get(k) for k in ["neutral", "anion", "cation"]] + [td_obj]


def create_wsc_objects(molecules, coord, q_mm, state_of_interest, triplet, required_calculations):
    """
    Create QM/MM molecule objects with external point charges.

    This function creates QM/MM calculations by adding external point charges
    to the base molecule objects.

    Args:
        molecules (list): [neutral_mf, anion_mf, cation_mf, td_obj] base objects
        coord (numpy.ndarray): Coordinates of external charges with shape [N, 3]
        q_mm (numpy.ndarray): Values of external point charges with shape [N]
        state_of_interest (int): Number of excited states for TD calculations
        triplet (bool): Whether to calculate triplet excited states
        required_calculations (dict): Dictionary specifying needed calculations

    Returns:
        list: [molecule_wsc, anion_wsc, cation_wsc, td_wsc] QM/MM objects
              where elements are SCF/TD objects or None if not calculated

    Note:
        - 'wsc' suffix indicates "with surface charge"
        - Uses checkpoint files from base calculations as initial guesses
        - TD objects are only created if explicitly needed
    """
    molecule_alone, anion_alone, cation_alone, td_alone = molecules

    qmmm_configs = [
        ("molecule_wsc", molecule_alone, "molecule_alone.chk"),
        ("anion_wsc", anion_alone, "anion_alone.chk"),
        ("cation_wsc", cation_alone, "cation_alone.chk"),
    ]

    qmmm_objects = {}
    for name, mol, chkfile in qmmm_configs:
        if mol is not None:
            qmmm_objects[name] = core.create_qmmm_molecule_object(mol, coord, q_mm, chkfile)

    # Only create TD if explicitly needed
    if qmmm_objects.get("molecule_wsc") and required_calculations.get("td", False):
        qmmm_objects["td_wsc"] = core.create_td_molecule_object(
            qmmm_objects["molecule_wsc"], nstates=state_of_interest, triplet=triplet
        )

    return [qmmm_objects.get(k) for k in ["molecule_wsc", "anion_wsc", "cation_wsc", "td_wsc"]]


def apply_solvation(molecules, solvent, state_of_interest, triplet, required_calculations):
    """
    Apply implicit solvation to all molecule objects.

    This function applies solvation effects using the Polarizable Continuum Model
    to all provided molecule objects (both base and QM/MM calculations).

    Args:
        molecules (list): List of 8 molecule objects:
                         [molecule_alone, anion_alone, cation_alone, td_alone,
                          molecule_wsc, anion_wsc, cation_wsc, td_wsc]
        solvent (str or None): Solvent name from SMD database, or None for gas phase
        state_of_interest (int): Number of excited states for TD calculations
        triplet (bool): Whether to calculate triplet excited states
        required_calculations (dict): Dictionary specifying needed calculations

    Returns:
        list: Solvated molecule objects in the same order as input,
              with new TD objects created from solvated ground states

    Note:
        - Returns original objects unchanged if solvent is None
        - Creates new TD objects from solvated ground states if needed
        - Maintains the same ordering and None values as input
    """
    if not solvent:
        return molecules

    (
        molecule_alone,
        anion_alone,
        cation_alone,
        td_alone,
        molecule_wsc,
        anion_wsc,
        cation_wsc,
        td_wsc,
    ) = molecules

    solvated = [
        core.solvate_molecule(mol, solvent) if mol else None
        for mol in [molecule_alone, anion_alone, cation_alone, molecule_wsc, anion_wsc, cation_wsc]
    ]

    # Only create TD objects if needed
    td_alone_new = None
    td_wsc_new = None

    if required_calculations.get("td", False):
        if solvated[0]:
            td_alone_new = core.create_td_molecule_object(
                solvated[0], nstates=state_of_interest, triplet=triplet
            )
        if solvated[3]:
            td_wsc_new = core.create_td_molecule_object(
                solvated[3], nstates=state_of_interest, triplet=triplet
            )

    return solvated[:3] + [td_alone_new] + solvated[3:] + [td_wsc_new]


##########################################################
def startup_message():
    print("=" * 60)
    print("                  Electrostatic Tuning Maps")
    print("             Built on efforts by the Gozem Lab")
    print("   See: https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489")
    print("=" * 60)
    print("\n")


##########################################################
def main(tuning_input="tuning.in"):
    #######################################
    #           Preliminary Setup         #
    #######################################
    """Main entry point for tuning calculations.

    Args:
        tuning_input (str | Path | dict): Path to a tuning.in file, or a
            parameter dict (defaults are supplied per-key below).
    """
    # Print startup message
    startup_message()

    # Get parameters from tuning file or dict
    if isinstance(tuning_input, dict):
        tuning_params = tuning_input
    else:
        tuning_params = get_tuning_parameters(tuning_input)

    # Extract all parameters
    molecule = tuning_params.get("molecule") or tuning_params.get("xyz_file")
    if not molecule or not os.path.exists(molecule):
        raise FileNotFoundError(f"XYZ file required: {molecule}")
    basis_set = tuning_params.get("basis_set", "6-31G*")
    method = tuning_params.get("method", "dft")
    functional = tuning_params.get("functional", "b3lyp")
    charge = tuning_params.get("charge", 0)
    spin = tuning_params.get("spin", 0)
    solvent = tuning_params.get("solvent", None)

    # Surface calculation parameters
    surface_file = tuning_params.get("surface_file")
    if not surface_file or not os.path.exists(surface_file):
        raise FileNotFoundError(f"Surface file required: {surface_file}")
    calc_type = tuning_params.get("calc_type", "separate")

    # Calculation specifics
    properties = tuning_params.get("properties", ["all"])
    state_of_interest = tuning_params.get("state_of_interest", 2)
    triplet = tuning_params.get("triplet", False)

    # Parallel processing parameter (default: True)
    parallel = tuning_params.get("parallel", True)
    num_procs = tuning_params.get("num_procs", None)

    # Check available hardware
    No_of_GPUs: int = core.check_gpu_info() or 0
    No_of_CPUs: int = core.check_cpu_info() or 1

    gpu_available = No_of_GPUs > 0

    # Resolve property dependencies and required calculations
    properties_to_calculate, required_calculations = setup_calculation(properties)
    print(f"Calculating Tuning of:  {properties_to_calculate}")
    print(f"Using molecular states: {required_calculations}")

    # Prepare input data
    input_data = molecule
    molecule_name = core.extract_xyz_name(input_data)

    # Load surface data
    surface_coords, surface_charges = load_surf(surface_file)

    print("\n")
    print("=" * 60)
    print(f"                Calculation Type: {calc_type}")
    print(f"                Number of surface points: {len(surface_coords)}")
    print(f"                Parallel Processing: {parallel}")
    print("=" * 60)
    print("\n")

    #######################################
    #    Core Tuning Map Calculations     #
    #######################################

    # ========================================
    #          Check for incomplete runs
    # ========================================
    resume_mode = False
    logs_dir = None
    existing_results = {}
    points_to_calculate = None

    incomplete_runs = find_incomplete_logs()

    if incomplete_runs:
        # Get most recent incomplete run
        latest_logs_dir, latest_metadata = incomplete_runs[0]

        # Check compatibility
        current_params = {
            "calc_type": calc_type,
            "total_points": len(surface_coords),
            "properties": sorted(properties_to_calculate),
        }

        is_compatible, error_msg = validate_resume_compatibility(latest_metadata, current_params)

        if is_compatible:
            if prompt_user_resume(latest_logs_dir, latest_metadata):
                resume_mode = True
                logs_dir = latest_logs_dir

                # Load existing results
                print(f"\nLoading existing results from {logs_dir}...")
                existing_results = load_completed_results_from_logs(logs_dir, len(surface_coords))

                # Determine which points still need calculation
                completed_indices = set(existing_results.keys())
                all_indices = set(range(len(surface_coords)))
                points_to_calculate = sorted(all_indices - completed_indices)

                print(f"Loaded {len(existing_results)} existing results")

                # Debug: Show what was actually loaded
                print("\nExisting results summary:")
                for idx in sorted(existing_results.keys()):
                    effects = existing_results[idx]["effects"]
                    print(f"  Point {idx}: {len(effects)} properties")
                    for prop, value in list(effects.items())[:3]:  # Show first 3
                        print(f"    {prop}: {value}")

                print(
                    f"Will calculate {len(points_to_calculate)} remaining points: {points_to_calculate[:10]}{'...' if len(points_to_calculate) > 10 else ''}"
                )

                # Update metadata for resume
                with open(os.path.join(logs_dir, ".resume_metadata.json")) as f:
                    metadata = json.load(f)
                metadata["resume_count"] += 1
                metadata["last_updated"] = datetime.now().isoformat()
                with open(os.path.join(logs_dir, ".resume_metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
        else:
            print("\nWarning: Found incomplete run but parameters don't match:")
            print(error_msg)
            print("Starting new calculation instead.\n")

    # Create logs directory if not resuming
    if not resume_mode:
        logs_dir = setup_logs_directory()
        print(f"Logging results to: {logs_dir}")

        # Create resume metadata
        create_resume_metadata(logs_dir, calc_type, len(surface_coords), properties_to_calculate)

        # Calculate all points
        points_to_calculate = list(range(len(surface_coords)))

    # Create base molecule objects
    base_molecules = create_molecule_objects(
        input_data,
        basis_set,
        spin,
        method,
        functional,
        charge,
        gpu_available,
        required_calculations,
        state_of_interest,
        triplet,
    )

    # Unpack the list returned by create_molecule_objects
    molecule_alone, anion_alone, cation_alone, td_alone = base_molecules

    # Use absolute paths for checkpoint files so Ray worker processes can
    # reliably locate them even if their current working directory differs.
    base_chkfiles = {
        "neutral": os.path.abspath("molecule_alone.chk")
        if required_calculations.get("neutral")
        else None,
        "anion": os.path.abspath("anion_alone.chk") if required_calculations.get("anion") else None,
        "cation": os.path.abspath("cation_alone.chk")
        if required_calculations.get("cation")
        else None,
    }

    # Quick sanity check: ensure checkpoint files that should exist are present
    missing_chk = [v for v in base_chkfiles.values() if v and not os.path.exists(v)]
    if missing_chk:
        raise FileNotFoundError(
            f"Missing checkpoint file(s) required for calculation: {missing_chk}. "
            "Make sure base calculations completed and checkpoint files were saved (e.g. 'molecule_alone.chk')."
        )

    # Calculate raw properties (baseline - no surface effects)
    raw_properties = calculate_all_properties(
        molecule_alone,
        anion_mf=anion_alone,
        cation_mf=cation_alone,
        td_obj=td_alone,
        triplet=triplet,
        props_to_calc=properties_to_calculate,
    )

    if solvent:
        all_molecules = [
            molecule_alone,
            anion_alone,
            cation_alone,
            td_alone,
            None,
            None,
            None,
            None,
        ]
        all_molecules = apply_solvation(
            all_molecules, solvent, state_of_interest, triplet, required_calculations
        )
        molecule_alone, anion_alone, cation_alone, td_alone, _, _, _, _ = all_molecules

    # Calculate raw properties (baseline - no surface effects)
    raw_properties = calculate_all_properties(
        molecule_alone,
        anion_mf=anion_alone,
        cation_mf=cation_alone,
        td_obj=td_alone,
        triplet=triplet,
        props_to_calc=properties_to_calculate,
    )

    print("\n")
    print("=" * 60)
    print("                Raw Properties (No Surface)")
    print(f"                Total raw properties calculated: {len(raw_properties)}")
    for prop, value in sorted(raw_properties.items()):
        print(f"                {prop}: {value:.6f}")
    print("=" * 60)
    print("\n")

    if calc_type == "combined":
        print(f"Running combined calculation with all {len(surface_coords)} surface points...")

        # Ensure logs_dir is set
        assert logs_dir is not None, "logs_dir must be initialized before combined calculation"

        # Initialize or reopen summary log
        summary_file = os.path.join(logs_dir, "calculation_summary.out")
        if not resume_mode:
            summary_file = initialize_summary_log(logs_dir, calc_type, 1)

        q_mm = surface_charges

        try:
            combined_effects = calculate_combined_surface_effect(
                base_chkfiles,
                surface_coords,
                q_mm,
                solvent,
                state_of_interest,
                triplet,
                properties_to_calculate,
                required_calculations,
                functional,
            )

            # Log combined result - use sum of charges for logging since it's a single combined calculation
            total_charge = float(np.sum(q_mm))
            log_point_result(
                logs_dir,
                0,
                np.mean(surface_coords, axis=0),
                total_charge,
                combined_effects,
                success=True,
            )

            # Append to summary
            append_point_to_summary(
                summary_file,
                0,
                np.mean(surface_coords, axis=0),
                total_charge,
                combined_effects,
                success=True,
                total_points=1,
            )

            # Update resume metadata
            update_resume_metadata(logs_dir, 0, True)

            print(f"Combined effects: {combined_effects}")
            all_effects = [combined_effects]
            output_coords = np.mean(surface_coords, axis=0).reshape(1, 3)

        except Exception as e:
            error_msg = f"Combined calculation failed: {e}"
            print(error_msg)

            # Log failure
            total_charge = float(np.sum(q_mm))
            log_point_result(
                logs_dir,
                0,
                np.mean(surface_coords, axis=0),
                total_charge,
                None,
                success=False,
                error_msg=error_msg,
            )

            # Append failure to summary
            append_point_to_summary(
                summary_file,
                0,
                np.mean(surface_coords, axis=0),
                total_charge,
                None,
                success=False,
                error_msg=error_msg,
                total_points=1,
            )

            # Update resume metadata
            update_resume_metadata(logs_dir, 0, False)

            all_effects = [None]
            output_coords = np.mean(surface_coords, axis=0).reshape(1, 3)

        # Finalize summary with statistics
        finalize_summary_log(summary_file, all_effects)

        # Append raw properties to summary
        append_raw_properties_to_summary(summary_file, raw_properties)

    else:  # calc_type == 'separate'
        if resume_mode:
            assert points_to_calculate is not None, "points_to_calculate must be set in resume mode"
            assert logs_dir is not None, "logs_dir must be set in resume mode"
            print(f"\nResuming calculation for {len(points_to_calculate)} remaining points...")
            summary_file = os.path.join(logs_dir, "calculation_summary.out")

            # Append resume header to existing summary
            with open(summary_file, "a") as f:
                f.write("\n" + "=" * 70 + "\n")
                f.write(f"RESUMING CALCULATION - {datetime.now().isoformat()}\n")
                f.write(f"Calculating {len(points_to_calculate)} remaining points\n")
                f.write("=" * 70 + "\n\n")
        else:
            summary_file = initialize_summary_log(logs_dir, calc_type, len(surface_coords))

        # Determine charges for each point
        point_charges = surface_charges.tolist()

        # Initialize all_effects with existing results if resuming
        if resume_mode:
            # Create full array with None placeholders
            all_effects: list[dict | None] = [None] * len(surface_coords)

            # Fill in existing results
            for point_idx, result in existing_results.items():
                all_effects[point_idx] = result["effects"]
                print(f"  Restored point {point_idx}: {result['effects']}")  # Debug

            print("\nInitialized all_effects array:")
            print(f"  Total slots: {len(all_effects)}")
            print(f"  Pre-filled: {sum(1 for x in all_effects if x is not None)}")
            assert points_to_calculate is not None, "points_to_calculate must be initialized"
            print(f"  To calculate: {len(points_to_calculate)}")
        else:
            all_effects = [None] * len(surface_coords)

        if parallel:
            if num_procs is None:
                parallel_processes = No_of_CPUs if No_of_GPUs < 1 else No_of_GPUs
            else:
                parallel_processes = min(
                    No_of_CPUs if No_of_GPUs < 1 else No_of_GPUs, int(num_procs)
                )

            # logging.getLogger("ray").setLevel(logging.ERROR)

            if No_of_GPUs < 1:
                ray.init(
                    num_cpus=parallel_processes,
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    logging_level=logging.ERROR,
                    # log_to_driver=False
                )
                calculate_point_effect = calculate_point_effect_cpu_remote
            else:
                ray.init(
                    num_gpus=parallel_processes,
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    logging_level=logging.ERROR,
                    # log_to_driver=False
                )
                calculate_point_effect = calculate_point_effect_gpu

            print(
                f"Using {parallel_processes} parallel processes on {'GPU' if gpu_available else 'CPU'}"
            )

            assert parallel_processes is not None, "parallel_processes must be initialized"
            # MODIFIED: Only submit jobs for points_to_calculate
            batch_size: int = parallel_processes
            assert points_to_calculate is not None, "points_to_calculate must be initialized"
            for batch_start in range(0, len(points_to_calculate), batch_size):
                batch_end = min(batch_start + batch_size, len(points_to_calculate))
                batch_indices = [points_to_calculate[i] for i in range(batch_start, batch_end)]

                futures = [
                    calculate_point_effect.remote(
                        base_chkfiles,
                        surface_coords[i],
                        point_charges[i],
                        solvent,
                        state_of_interest,
                        triplet,
                        properties_to_calculate,
                        required_calculations,
                        functional,
                        i,
                    )
                    for i in batch_indices
                ]

                # Get results and log them IMMEDIATELY after each completes
                for result in ray.get(futures):
                    point_index = result["point_index"]
                    all_effects[point_index] = result["effects"]

                    # Log individual point file
                    log_point_result(
                        logs_dir,
                        point_index,
                        result["coord"],
                        result["charge"],
                        result["effects"],
                        success=result["success"],
                        error_msg=result["error_msg"],
                    )

                    # Append to summary file IMMEDIATELY
                    append_point_to_summary(
                        summary_file,
                        point_index,
                        result["coord"],
                        result["charge"],
                        result["effects"],
                        success=result["success"],
                        error_msg=result["error_msg"],
                        total_points=len(surface_coords),
                    )

                    # Update resume metadata
                    update_resume_metadata(logs_dir, point_index, result["success"])

                    status = "SUCCESS" if result["success"] else "FAILED"
                    completed_so_far = len([e for e in all_effects if e is not None])
                    print(
                        f"Point {point_index + 1}/{len(surface_coords)}: {status} ({completed_so_far}/{len(surface_coords)} total)"
                    )

            ray.shutdown()

        else:
            # Sequential processing
            print("Using sequential processing (parallel=False)")
            assert points_to_calculate is not None, "points_to_calculate must be initialized"

            for point_idx in points_to_calculate:  # Use point_idx directly
                coord = surface_coords[point_idx]
                try:
                    effects = calculate_surface_effect_at_point(
                        base_chkfiles,
                        coord,
                        point_charges[point_idx],
                        solvent,
                        state_of_interest,
                        triplet,
                        properties_to_calculate,
                        required_calculations,
                        force_single_gpu=True,
                    )

                    all_effects[point_idx] = effects

                    # Log individual file
                    log_point_result(
                        logs_dir, point_idx, coord, point_charges[point_idx], effects, success=True
                    )

                    # Append to summary IMMEDIATELY
                    append_point_to_summary(
                        summary_file,
                        point_idx,
                        coord,
                        point_charges[point_idx],
                        effects,
                        success=True,
                        total_points=len(surface_coords),
                    )

                    # Update resume metadata
                    update_resume_metadata(logs_dir, point_idx, True)

                    completed_so_far = len([e for e in all_effects if e is not None])
                    print(
                        f"Point {point_idx + 1}/{len(surface_coords)}: SUCCESS ({completed_so_far}/{len(surface_coords)} total)"
                    )

                except Exception as e:
                    all_effects[point_idx] = None  # Use point_idx
                    error_msg = f"Error at point {point_idx}: {e}"
                    print(f"Point {point_idx + 1}/{len(surface_coords)}: FAILED - {e}")

                    # Log failure
                    log_point_result(
                        logs_dir,
                        point_idx,
                        coord,
                        point_charges[point_idx],
                        None,
                        success=False,
                        error_msg=error_msg,
                    )

                    # Append failure to summary IMMEDIATELY
                    append_point_to_summary(
                        summary_file,
                        point_idx,
                        coord,
                        point_charges[point_idx],
                        None,
                        success=False,
                        error_msg=error_msg,
                        total_points=len(surface_coords),
                    )

                    # Update resume metadata
                    update_resume_metadata(logs_dir, point_idx, False)

        # Now all_effects contains BOTH old and new results
        output_coords = surface_coords

        # Finalize summary with ALL results (old + new)
        finalize_summary_log(summary_file, all_effects)

        # Append raw properties to summary
        append_raw_properties_to_summary(summary_file, raw_properties)

    # Create output files with ALL results (not just new ones)
    normalization_params = create_output_files(
        output_coords, all_effects, molecule_name, properties_to_calculate, raw_properties
    )

    # Remove temporary checkpoint files FIRST
    temp_files = ["molecule_alone.chk", "anion_alone.chk", "cation_alone.chk"]
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    # Organize results into timestamped folder SECOND (with normalization params)
    results_dir = organize_results(
        molecule_name, properties_to_calculate, logs_dir, normalization_params
    )

    # Change to results directory and check files THIRD
    original_dir = os.getcwd()
    os.chdir(results_dir)
    check_all_files_created(molecule_name, output_coords, properties_to_calculate, all_effects)
    os.chdir(original_dir)


if __name__ == "__main__":
    tuning_file = sys.argv[1] if len(sys.argv) > 1 else "tuning.in"
    main(tuning_file)

"""Tuning output files and normalization."""

import csv
import os
import shutil
from datetime import datetime


def append_raw_properties_to_summary(summary_file, raw_properties):
    """
    Append raw baseline properties to the summary file.

    Args:
        summary_file (str): Path to summary file
        raw_properties (dict): Dictionary of baseline property values (no surface effects)
    """
    with open(summary_file, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("RAW PROPERTIES (NO SURFACE EFFECTS)\n")
        f.write("=" * 70 + "\n\n")

        # Group properties by category for better readability
        energy_props = {}
        orbital_props = {}
        excited_props = {}
        other_props = {}

        for prop, value in raw_properties.items():
            if any(x in prop for x in ["gse", "ie", "ea", "cp", "eng", "hard", "efl", "nfl"]):
                energy_props[prop] = value
            elif any(x in prop for x in ["homo", "lumo", "gap"]):
                orbital_props[prop] = value
            elif "exe" in prop or "osc" in prop:
                excited_props[prop] = value
            else:
                other_props[prop] = value

        # Write energy properties
        if energy_props:
            f.write("Energy Properties:\n")
            f.write("-" * 50 + "\n")
            for prop, value in sorted(energy_props.items()):
                unit = "eV" if any(x in prop for x in ["eng", "hard", "efl", "nfl"]) else "kcal/mol"
                f.write(f"  {prop:<15s}: {value:>12.6f}  {unit}\n")
            f.write("\n")

        # Write orbital properties
        if orbital_props:
            f.write("Orbital Properties:\n")
            f.write("-" * 50 + "\n")
            for prop, value in sorted(orbital_props.items()):
                f.write(f"  {prop:<15s}: {value:>12.6f}  eV\n")
            f.write("\n")

        # Write excited state properties
        if excited_props:
            f.write("Excited State Properties:\n")
            f.write("-" * 50 + "\n")
            for prop, value in sorted(excited_props.items()):
                unit = "eV" if "exe" in prop else "dimensionless"
                f.write(f"  {prop:<15s}: {value:>12.6f}  {unit}\n")
            f.write("\n")

        # Write other properties
        if other_props:
            f.write("Other Properties:\n")
            f.write("-" * 50 + "\n")
            for prop, value in sorted(other_props.items()):
                f.write(f"  {prop:<15s}: {value:>12.6f}  a.u.\n")
            f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"Raw properties appended to: {summary_file}")


def organize_results(molecule_name, properties_to_calculate, logs_dir, normalization_params=None):
    """
    Move all output files into a timestamped results folder.

    Args:
        molecule_name (str): Base name of molecule
        properties_to_calculate (list): List of properties that were calculated
        logs_dir (str): Path to logs directory
        normalization_params (dict, optional): Normalization parameters (min, max) for each property

    Returns:
        str: Path to created results directory
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = f"results_{molecule_name}_{timestamp}"

    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    print(f"\nOrganizing results into: {results_dir}/")

    # Move CSV summary file
    csv_file = f"{molecule_name}_tuning_summary.csv"
    if os.path.exists(csv_file):
        shutil.move(csv_file, os.path.join(results_dir, csv_file))
        print(f"  Moved: {csv_file}")

    # Move all .mol2 files (scan directory for all matching files)
    mol2_files = []
    for file in os.listdir("."):
        if file.startswith(molecule_name) and file.endswith(".mol2"):
            shutil.move(file, os.path.join(results_dir, file))
            mol2_files.append(file)

    if mol2_files:
        print(f"  Moved {len(mol2_files)} MOL2 files")

    # Move logs directory
    if logs_dir and os.path.exists(logs_dir):
        dest_logs = os.path.join(results_dir, "logs")

        # Add normalization parameters to summary file BEFORE moving
        if normalization_params:
            summary_file = os.path.join(logs_dir, "calculation_summary.out")
            if os.path.exists(summary_file):
                with open(summary_file, "a") as f:
                    f.write("\n" + "=" * 70 + "\n")
                    f.write("NORMALIZATION PARAMETERS\n")
                    f.write("=" * 70 + "\n\n")
                    f.write(f"{'Property':<30} {'Min Value':>15} {'Max Value':>15}\n")
                    f.write("-" * 70 + "\n")
                    for key, (min_val, max_val) in sorted(normalization_params.items()):
                        f.write(f"{key:<30} {min_val:>15.10f} {max_val:>15.10f}\n")
                    f.write("\n" + "=" * 70 + "\n")
                print("  Added normalization parameters to summary")

        shutil.move(logs_dir, dest_logs)
        print(f"  Moved: {logs_dir}/ -> logs/")

    # Remove any leftover point_* worker directories
    removed_dirs = []
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("point_"):
            try:
                shutil.rmtree(item)
                removed_dirs.append(item)
            except Exception as e:
                print(f"  Warning: Could not remove {item}: {e}")

    if removed_dirs:
        print(f"  Cleaned up {len(removed_dirs)} worker directories")

    # Create a README file in results directory
    readme_path = os.path.join(results_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ELECTROSTATIC TUNING MAP RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Molecule:           {molecule_name}\n")
        f.write(f"Timestamp:          {timestamp}\n")
        f.write(f"Properties:         {', '.join(properties_to_calculate)}\n\n")
        f.write("Files in this directory:\n")
        f.write("-" * 70 + "\n")
        f.write(f"  {csv_file:<40} - Summary CSV with all data\n")
        f.write(f"  {molecule_name}_*.mol2{'':<24} - MOL2 files (raw values)\n")
        f.write(f"  {molecule_name}_*_normalized.mol2{'':<14} - MOL2 files (normalized)\n")
        f.write(f"  logs/{'':<46} - Individual point logs\n")
        f.write(f"  README.txt{'':<38} - This file\n\n")
        f.write("=" * 70 + "\n")

    return results_dir


def create_output_files(
    surface_coords, all_effects, molecule_name, properties_to_calculate, raw_properties
):
    """
    Create MOL2 files and CSV summary for surface effects analysis.

    This function scans all effect dictionaries for any key ending in '_effect'
    (including sX_exe_effect, tX_osc_effect, etc.) and creates output files for each.
    Creates both normalized and non-normalized versions.

    Args:
        surface_coords (numpy.ndarray): Array of surface coordinates with shape [N, 3]
        all_effects (list): List of effect dictionaries for each surface point
        molecule_name (str): Base name for output files
        properties_to_calculate (list): List of calculated molecular properties
        raw_properties (dict): Dict of baseline property values (no surface effects)

    Returns:
        dict: Normalization parameters (min, max) for each property
    """
    # Gather all effect keys found in all_effects
    effect_keys = set()
    for effect in all_effects:
        if effect:
            effect_keys.update(effect.keys())
    effect_keys = sorted(effect_keys)

    # Normalize the effects
    normalized_effects, normalization_params = normalize_effects(all_effects, effect_keys)

    # Create MOL2 files for non-normalized values
    for key in effect_keys:
        prop_base = key.replace("_effect", "") if key.endswith("_effect") else key

        # Get baseline value for this property
        baseline_value = raw_properties.get(prop_base, 0.0)

        # Create custom MOL2 with baseline in comment line
        filename = f"{molecule_name}_{prop_base}.mol2"
        with open(filename, "w") as f:
            # Header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{prop_base} | baseline={baseline_value:.6f}\n")
            f.write(f"{len(surface_coords):5d} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")

            # Atoms
            f.write("@<TRIPOS>ATOM\n")
            for idx, (coord, effect) in enumerate(zip(surface_coords, all_effects, strict=True), 1):
                x, y, z = coord
                effect_value = effect.get(key, 0.0) if effect else 0.0
                f.write(
                    f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n"
                )

        print(f"Created: {filename}")

    # Create MOL2 files for normalized values
    for key in effect_keys:
        prop_base = key.replace("_effect", "") if key.endswith("_effect") else key

        # Get baseline value for this property
        baseline_value = raw_properties.get(prop_base, 0.0)

        # Create custom MOL2 with baseline in comment line
        filename = f"{molecule_name}_{prop_base}_normalized.mol2"
        with open(filename, "w") as f:
            # Header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{prop_base}_normalized | baseline={baseline_value:.6f}\n")
            f.write(f"{len(surface_coords):5d} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")

            # Atoms
            f.write("@<TRIPOS>ATOM\n")
            for idx, (coord, norm_effect) in enumerate(
                zip(surface_coords, normalized_effects, strict=True), 1
            ):
                x, y, z = coord
                effect_value = norm_effect.get(key, 0.0) if norm_effect else 0.0
                f.write(
                    f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n"
                )

        print(f"Created: {filename}")

    # Create CSV summary with coordinates, effects, normalized effects, AND baseline values
    csv_filename = f"{molecule_name}_tuning_summary.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        # Create fieldnames
        fieldnames = ["point_index", "x", "y", "z"]
        for key in effect_keys:
            prop_base = key.replace("_effect", "") if key.endswith("_effect") else key
            fieldnames.append(key)  # Raw effect (e.g., 'gse_effect')
            fieldnames.append(f"{key}_normalized")  # Normalized
            fieldnames.append(f"{prop_base}_baseline")  # Baseline

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, (coord, effect, norm_effect) in enumerate(
            zip(surface_coords, all_effects, normalized_effects, strict=True)
        ):
            row = {"point_index": i, "x": coord[0], "y": coord[1], "z": coord[2]}

            for key in effect_keys:
                prop_base = key.replace("_effect", "") if key.endswith("_effect") else key

                # Get raw effect value
                raw_val = effect.get(key, 0.0) if effect else 0.0
                norm_val = norm_effect.get(key, 0.0) if norm_effect else 0.0
                base_val = raw_properties.get(prop_base, 0.0)

                row[key] = raw_val
                row[f"{key}_normalized"] = norm_val
                row[f"{prop_base}_baseline"] = base_val

            writer.writerow(row)

    print(f"\nCreated: {csv_filename}")

    # Return normalization parameters
    return normalization_params


def normalize_effects(all_effects, effect_keys):
    """
    Normalize effect values to [-1, 1] range using min-max normalization.

    Args:
        all_effects (list): List of effect dictionaries for each surface point
        effect_keys (list): List of effect keys to normalize

    Returns:
        list: List of dictionaries with normalized values
    """
    normalized_effects = []

    # Calculate min and max for each effect key
    normalization_params = {}
    for key in effect_keys:
        values = [effect.get(key, 0.0) for effect in all_effects]
        min_val = min(values)
        max_val = max(values)
        normalization_params[key] = (min_val, max_val)

    # Normalize each effect dictionary
    for effect in all_effects:
        normalized = {}
        for key in effect_keys:
            min_val, max_val = normalization_params[key]
            value = effect.get(key, 0.0)

            # Min-max normalization to [-1, 1]
            if max_val - min_val != 0:
                normalized[key] = 2 * (value - min_val) / (max_val - min_val) - 1
            else:
                normalized[key] = 0.0
        normalized_effects.append(normalized)

    return normalized_effects, normalization_params


def check_all_files_created(
    molecule_name, surface_coords, properties_to_calculate, all_effects=None
):
    missing = []
    # Use effect keys from all_effects if provided, otherwise fallback to properties_to_calculate
    effect_keys = set()
    if all_effects and len(all_effects) > 0:
        for effect in all_effects:
            effect_keys.update(effect.keys())
        # Only check for keys ending with '_effect'
        effect_props = [
            key.replace("_effect", "") for key in effect_keys if key.endswith("_effect")
        ]
    else:
        effect_props = properties_to_calculate

    for prop in effect_props:
        filepath = f"{molecule_name}_{prop}.mol2"
        if not os.path.exists(filepath):
            missing.append(filepath)

    csv_path = f"{molecule_name}_tuning_summary.csv"
    if not os.path.exists(csv_path):
        missing.append(csv_path)

    if missing:
        print(f"Missing: {', '.join(missing)}")
    else:
        print("All expected output files were created.")


#################
# Miscellaneous #
#################

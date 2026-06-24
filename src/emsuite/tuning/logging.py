"""Tuning run logging."""

import os
from datetime import datetime

import numpy as np


def setup_logs_directory():
    """Create logs directory structure with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = f"logs_{timestamp}"
    os.makedirs(logs_dir, exist_ok=True)
    return logs_dir


def initialize_summary_log(logs_dir, calc_type, total_points):
    """
    Create summary file with header at the start of calculation.

    Args:
        logs_dir (str): Directory for log files
        calc_type (str): 'separate' or 'combined'
        total_points (int): Expected number of surface points

    Returns:
        str: Path to summary file
    """
    summary_file = os.path.join(logs_dir, "calculation_summary.out")

    with open(summary_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{'ELECTROSTATIC TUNING MAP CALCULATION SUMMARY':^70}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Start Time:         {datetime.now().isoformat()}\n")
        f.write(f"Calculation Type:   {calc_type}\n")
        f.write(f"Total Points:       {total_points}\n")
        f.write("Status:             IN PROGRESS\n\n")

    print(f"Summary log initialized: {summary_file}")
    return summary_file


def append_point_to_summary(
    summary_file,
    point_index,
    coord,
    charge,
    effects,
    success=True,
    error_msg=None,
    total_points=None,
):
    """
    Append individual point result to summary file immediately after calculation.

    Args:
        summary_file (str): Path to summary file
        point_index (int): Index of completed point
        coord (array): Coordinates
        charge (float): Surface charge
        effects (dict): Calculated effects
        success (bool): Whether calculation succeeded
        error_msg (str, optional): Error message if failed
        total_points (int, optional): Total points for progress percentage
    """
    with open(summary_file, "a") as f:
        # Write point header
        progress = f"[{point_index + 1}/{total_points}]" if total_points else f"Point {point_index}"
        f.write(f"\n{progress} " + "-" * 55 + "\n")
        f.write(f"Point {point_index:4d}:  ")
        f.write(f"({coord[0]:8.4f}, {coord[1]:8.4f}, {coord[2]:8.4f})  ")
        f.write(f"q = {charge:7.4f}\n")
        f.write(f"Status:     {'SUCCESS' if success else 'FAILED'}\n")
        f.write(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        if success and effects:
            f.write("\nCalculated Effects:\n")
            # Group by type for readability
            for key, value in sorted(effects.items()):
                f.write(f"  {key:<30} = {value:>12.8f}\n")
        elif error_msg:
            f.write(f"\nError: {error_msg}\n")

        f.write("\n")


def finalize_summary_log(summary_file, all_effects):
    """
    Add final statistics section to summary file after all calculations complete.

    Args:
        summary_file (str): Path to summary file
        all_effects (list): List of effects dictionaries (or None for failed)
        surface_coords (array): All surface coordinates
        point_charges (list): All surface charges
    """
    successful = sum(1 for e in all_effects if e is not None)
    failed = sum(1 for e in all_effects if e is None)

    with open(summary_file, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("FINAL STATISTICS\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Completion Time:    {datetime.now().isoformat()}\n")
        f.write(f"Successful Points:  {successful}\n")
        f.write(f"Failed Points:      {failed}\n")
        f.write(f"Success Rate:       {successful / len(all_effects) * 100:.2f}%\n\n")

        # Get all unique property keys from successful calculations
        all_keys = set()
        for effects in all_effects:
            if effects:
                all_keys.update(effects.keys())

        if all_keys:
            f.write("-" * 70 + "\n")
            f.write("STATISTICS FOR EACH PROPERTY EFFECT\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Property':<25} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std Dev':>12}\n")
            f.write("-" * 70 + "\n")

            for key in sorted(all_keys):
                # Convert all values to float to handle both CPU and GPU arrays
                values = []
                for e in all_effects:
                    if e and key in e:
                        val = e[key]
                        # Handle CuPy arrays from GPU calculations
                        if hasattr(val, "get"):
                            val = float(val.get())
                        else:
                            val = float(val)
                        values.append(val)

                if values:
                    f.write(
                        f"{key:<25} {min(values):>12.6f} {max(values):>12.6f} "
                        f"{np.mean(values):>12.6f} {np.std(values):>12.6f}\n"
                    )

            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("Calculation complete. Individual point files: point_XXXX.out\n")
        f.write("=" * 70 + "\n")

    print(f"\nFinal statistics written to: {summary_file}")


def log_point_result(logs_dir, point_index, coord, charge, effects, success=True, error_msg=None):
    """Log individual point calculation result to structured .out file."""
    log_file = os.path.join(logs_dir, f"point_{point_index:04d}.out")

    # Convert charge to scalar if it's an array
    charge = float(charge) if np.isscalar(charge) else float(np.sum(charge))

    with open(log_file, "w") as f:
        f.write("=" * 70 + "\n")
        f.write(f"{'SURFACE POINT CALCULATION RESULTS':^70}\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Point Index:        {point_index}\n")
        f.write(f"Timestamp:          {datetime.now().isoformat()}\n")
        f.write(f"Status:             {'SUCCESS' if success else 'FAILED'}\n\n")

        f.write("-" * 70 + "\n")
        f.write("COORDINATES AND CHARGE\n")
        f.write("-" * 70 + "\n")
        f.write(f"X-coordinate:       {coord[0]:>12.6f} Angstrom\n")
        f.write(f"Y-coordinate:       {coord[1]:>12.6f} Angstrom\n")
        f.write(f"Z-coordinate:       {coord[2]:>12.6f} Angstrom\n")
        f.write(f"Surface Charge:     {charge:>12.6f} a.u.\n\n")

        if success and effects:
            f.write("-" * 70 + "\n")
            f.write("CALCULATED EFFECTS\n")
            f.write("-" * 70 + "\n")
            f.write(f"{'Property':<30} {'Effect Value':>20} {'Unit':>15}\n")
            f.write("-" * 70 + "\n")

            # Group effects by type
            energy_effects = {}
            orbital_effects = {}
            excited_effects = {}
            other_effects = {}

            for key, value in effects.items():
                if any(x in key for x in ["gse", "ie", "ea", "cp", "eng", "hard", "efl", "nfl"]):
                    energy_effects[key] = value
                elif any(x in key for x in ["homo", "lumo", "gap"]):
                    orbital_effects[key] = value
                elif "exe" in key or "osc" in key:
                    excited_effects[key] = value
                else:
                    other_effects[key] = value

            # Write energy effects
            if energy_effects:
                f.write("\n  Energy Properties:\n")
                for key, value in sorted(energy_effects.items()):
                    unit = (
                        "eV"
                        if "eng" in key or "hard" in key or "efl" in key or "nfl" in key
                        else "kcal/mol"
                    )
                    f.write(f"    {key:<36} {value:>20.14f} {unit:>17}\n")

            # Write orbital effects
            if orbital_effects:
                f.write("\n  Orbital Properties:\n")
                for key, value in sorted(orbital_effects.items()):
                    f.write(f"    {key:<36} {value:>20.14f} {'eV':>17}\n")

            # Write excited state effects
            if excited_effects:
                f.write("\n  Excited State Properties:\n")
                for key, value in sorted(excited_effects.items()):
                    unit = "eV" if "exe" in key else "dimensionless"
                    f.write(f"    {key:<36} {value:>20.14f} {unit:>17}\n")

            # Write other effects
            if other_effects:
                f.write("\n  Other Properties:\n")
                for key, value in sorted(other_effects.items()):
                    f.write(f"    {key:<36} {value:>20.14f} {'a.u.':>17}\n")

        elif error_msg:
            f.write("-" * 70 + "\n")
            f.write("ERROR INFORMATION\n")
            f.write("-" * 70 + "\n")
            f.write(f"{error_msg}\n\n")

        f.write("\n" + "=" * 70 + "\n")

"""Resume interrupted tuning runs."""

import json
import os
from datetime import datetime
from pathlib import Path


def create_resume_metadata(logs_dir, calc_type, total_points, properties_to_calculate):
    """
    Create metadata file for resume capability.

    Args:
        logs_dir (str): Logs directory path
        calc_type (str): 'separate' or 'combined'
        total_points (int): Total number of points
        properties_to_calculate (list): Properties being calculated
    """
    metadata = {
        "original_start": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat(),
        "calc_type": calc_type,
        "total_points": total_points,
        "properties": sorted(properties_to_calculate),
        "completed_points": [],
        "failed_points": [],
        "resume_count": 0,
    }

    metadata_file = os.path.join(logs_dir, ".resume_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_file


def update_resume_metadata(logs_dir, point_index, success):
    """
    Update metadata after each point completes.

    Args:
        logs_dir (str): Logs directory path
        point_index (int): Completed point index
        success (bool): Whether point succeeded
    """
    metadata_file = os.path.join(logs_dir, ".resume_metadata.json")

    if os.path.exists(metadata_file):
        with open(metadata_file) as f:
            metadata = json.load(f)

        # Update completion tracking
        if success:
            if point_index not in metadata["completed_points"]:
                metadata["completed_points"].append(point_index)
        else:
            if point_index not in metadata["failed_points"]:
                metadata["failed_points"].append(point_index)

        metadata["last_updated"] = datetime.now().isoformat()

        # Sort for readability
        metadata["completed_points"].sort()
        metadata["failed_points"].sort()

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)


def find_incomplete_logs():
    """
    Search current directory for incomplete log directories.

    Returns:
        list: List of tuples (logs_dir, metadata_dict) sorted by timestamp (newest first)
    """
    incomplete_runs = []

    # Find all logs_* directories
    for item in Path(".").glob("logs_*"):
        if item.is_dir():
            metadata_file = item / ".resume_metadata.json"

            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)

                # Check if incomplete
                completed = len(metadata["completed_points"])
                total = metadata["total_points"]

                if completed < total:
                    incomplete_runs.append((str(item), metadata))

    # Sort by timestamp (newest first)
    incomplete_runs.sort(key=lambda x: x[1]["original_start"], reverse=True)

    return incomplete_runs


def validate_resume_compatibility(metadata, current_params):
    """
    Check if resume is compatible with current run parameters.

    Args:
        metadata (dict): Metadata from previous run
        current_params (dict): Current run parameters

    Returns:
        tuple: (is_compatible, error_message)
    """
    checks = []

    # Check calc_type
    if metadata["calc_type"] != current_params["calc_type"]:
        checks.append(
            f"calc_type mismatch: {metadata['calc_type']} vs {current_params['calc_type']}"
        )

    # Check total_points
    if metadata["total_points"] != current_params["total_points"]:
        checks.append(
            f"total_points mismatch: {metadata['total_points']} vs {current_params['total_points']}"
        )

    # Check properties (must be same or subset)
    old_props = set(metadata["properties"])
    new_props = set(current_params["properties"])
    if old_props != new_props:
        checks.append(f"properties mismatch: {old_props} vs {new_props}")

    if checks:
        return False, "\n".join(checks)

    return True, None


def load_completed_results_from_logs(logs_dir, total_points):
    """Load previously completed results from individual point log files."""
    existing_results = {}

    for point_idx in range(total_points):
        # Look for .out files, not .log files
        point_log = os.path.join(logs_dir, f"point_{point_idx:04d}.out")

        if os.path.exists(point_log):
            try:
                # Parse the .out file (it's a formatted text file, not JSON)
                result = parse_point_log_file(point_log)

                if result and result["success"]:
                    effects = result.get("effects", {})

                    # Verify we have actual effect values (not empty dict)
                    if effects and any(v is not None for v in effects.values()):
                        existing_results[point_idx] = {
                            "effects": effects,
                            "coord": result.get("coord"),
                            "charge": result.get("charge"),
                        }
                        print(f"  Loaded point {point_idx}: {len(effects)} properties")
                    else:
                        print(f"  Point {point_idx}: SUCCESS but no effects data")
                else:
                    print(f"  Point {point_idx}: marked as failed or corrupt")

            except Exception as e:
                print(f"  Warning: Could not load point {point_idx}: {e}")
                continue

    return existing_results


def parse_point_log_file(log_file):
    """
    Parse a point_XXXX.out file to extract calculation results.

    Args:
        log_file (str): Path to log file

    Returns:
        dict: {'coord': [x,y,z], 'charge': X, 'effects': {...}, 'success': bool} or None
    """
    try:
        with open(log_file) as f:
            lines = f.readlines()

        result = {"coord": [0, 0, 0], "charge": None, "effects": {}, "success": False}

        in_effects_section = False

        for line in lines:
            line_stripped = line.strip()

            # Parse status
            if "Status:" in line and "SUCCESS" in line:
                result["success"] = True

            # Parse coordinates
            if "X-coordinate:" in line:
                parts = line.split()
                result["coord"][0] = float(parts[1])
            elif "Y-coordinate:" in line:
                parts = line.split()
                result["coord"][1] = float(parts[1])
            elif "Z-coordinate:" in line:
                parts = line.split()
                result["coord"][2] = float(parts[1])

            # Parse charge
            if "Surface Charge:" in line:
                parts = line.split()
                result["charge"] = float(parts[2])

            # Detect effects section start
            if "CALCULATED EFFECTS" in line:
                in_effects_section = True
                continue

            # Parse effects - look for lines with property names and values
            if in_effects_section:
                # End of effects section
                if line_stripped.startswith("==="):
                    break

                # Skip header lines, dividers, and category labels
                if (
                    line_stripped.startswith("-")
                    or line_stripped.startswith("Property")
                    or line_stripped.endswith("Properties:")
                    or not line_stripped
                ):
                    continue

                # Parse effect line (format: "    s1_exe_effect                         -0.22655508              eV")
                parts = line_stripped.split()
                if len(parts) >= 2:
                    try:
                        prop_name = parts[0]
                        value = float(parts[1])
                        result["effects"][prop_name] = value
                    except (ValueError, IndexError):
                        pass

        # Only return if we got valid data
        if result["success"] and result["coord"] and result["charge"] is not None:
            return result

        return None

    except Exception as e:
        print(f"Warning: Could not parse {log_file}: {e}")
        return None


def prompt_user_resume(logs_dir, metadata):
    """
    Ask user if they want to resume an incomplete run.

    Args:
        logs_dir (str): Logs directory path
        metadata (dict): Metadata from incomplete run

    Returns:
        bool: True if user wants to resume
    """
    completed = len(metadata["completed_points"])
    failed = len(metadata["failed_points"])
    total = metadata["total_points"]
    remaining = total - completed

    print("\n" + "=" * 70)
    print("INCOMPLETE RUN DETECTED")
    print("=" * 70)
    print(f"Logs Directory:     {logs_dir}")
    print(f"Original Start:     {metadata['original_start']}")
    print(f"Last Updated:       {metadata['last_updated']}")
    print(f"Calculation Type:   {metadata['calc_type']}")
    print(f"Total Points:       {total}")
    print(f"Completed:          {completed} ({completed / total * 100:.1f}%)")
    print(f"Failed:             {failed}")
    print(f"Remaining:          {remaining}")
    print("=" * 70)

    response = input("\nDo you want to resume this calculation? [y/n]: ").strip().lower()

    return response in ["", "y", "yes"]



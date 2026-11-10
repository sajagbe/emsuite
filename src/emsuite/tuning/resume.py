"""Resume interrupted tuning runs."""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def create_resume_metadata(logs_dir, calc_type, total_points, properties_to_calculate, inputs):
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
        "inputs": inputs,
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


def capture_resume_inputs(params, properties, total_points):
    """Snapshot calculation settings and file contents before any SCF work."""
    inputs = {
        key: params[key]
        for key in (
            "calc_type",
            "basis_set",
            "method",
            "functional",
            "charge",
            "spin",
            "solvent",
            "state_of_interest",
            "triplet",
        )
    }
    inputs.update(properties=sorted(properties), total_points=total_points)
    for key in ("molecule", "surface_file"):
        path = Path(params[key]).resolve()
        inputs[key] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return inputs


def validate_resume_compatibility(metadata, current_params):
    """Compare all recorded calculation inputs; paths are diagnostic only."""
    previous = metadata.get("inputs")
    if not isinstance(previous, dict):
        return False, "inputs not recorded in the previous run"
    differences = []
    for key in sorted(previous.keys() | current_params.keys()):
        old, new = previous.get(key), current_params.get(key)
        if key in ("molecule", "surface_file") and isinstance(old, dict) and isinstance(new, dict):
            matches = old.get("sha256") == new.get("sha256")
        else:
            matches = key in previous and key in current_params and old == new
        if not matches:
            differences.append(f"{key}: old={old!r}; new={new!r}")
    return not differences, "\n".join(differences) or None


def select_resume_run(enabled, current_inputs):
    """Select the newest compatible unfinished run without prompting or modifying it."""
    runs = find_incomplete_logs()
    if not runs:
        return None
    logs_dir, metadata = runs[0]
    if not enabled:
        print(f"Unfinished run {logs_dir} left untouched (resume=False). Starting a new run.")
        return None
    compatible, differences = validate_resume_compatibility(metadata, current_inputs)
    if not compatible:
        print(f"Unfinished run {logs_dir} has different inputs:\n{differences}")
        print("Starting a new run; the previous folder is left untouched.")
        return None
    print(f"Resuming unfinished run: {logs_dir}")
    return logs_dir, metadata


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

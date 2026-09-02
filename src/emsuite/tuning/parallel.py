"""Ray parallel workers for per-point tuning calculations."""

from __future__ import annotations

import os
import shutil

import ray


def calculate_point_effect_cpu(
    base_chkfiles,
    coord,
    surface_charge,
    solvent,
    state_of_interest,
    triplet,
    properties_to_calculate,
    required_calculations,
    functional,
    point_index,
):
    sched_getaffinity = getattr(os, "sched_getaffinity", None)
    if sched_getaffinity is not None:
        cpu_id = sched_getaffinity(0)
        print(f"[Point {point_index}] Running on CPU cores: {cpu_id}, PID: {os.getpid()}")
    else:
        print(f"[Point {point_index}] Running on PID: {os.getpid()}")

    worker_dir = f"point_{point_index}"
    os.makedirs(worker_dir, exist_ok=True)

    worker_chkfiles = {}
    for key, chkfile in base_chkfiles.items():
        if chkfile:
            worker_chkfile = os.path.join(worker_dir, os.path.basename(chkfile))
            shutil.copy2(chkfile, worker_chkfile)
            worker_chkfiles[key] = worker_chkfile
        else:
            worker_chkfiles[key] = None

    original_dir = os.getcwd()
    os.chdir(worker_dir)

    try:
        from .runner import calculate_surface_effect_at_point

        effects = calculate_surface_effect_at_point(
            {k: os.path.basename(v) if v else None for k, v in worker_chkfiles.items()},
            coord,
            surface_charge,
            solvent,
            state_of_interest,
            triplet,
            properties_to_calculate,
            required_calculations,
            force_single_gpu=False,
        )
        os.chdir(original_dir)
        return {
            "point_index": point_index,
            "coord": coord,
            "charge": surface_charge,
            "effects": effects,
            "success": True,
            "error_msg": None,
        }

    except Exception as e:
        error_msg = f"Error at point {point_index}: {e}"
        print(error_msg)
        os.chdir(original_dir)
        return {
            "point_index": point_index,
            "coord": coord,
            "charge": surface_charge,
            "effects": None,
            "success": False,
            "error_msg": error_msg,
        }

    finally:
        if os.path.exists(worker_dir):
            shutil.rmtree(worker_dir, ignore_errors=True)


@ray.remote(num_cpus=1, max_retries=0)
def calculate_point_effect_cpu_remote(
    base_chkfiles,
    coord,
    surface_charge,
    solvent,
    state_of_interest,
    triplet,
    properties_to_calculate,
    required_calculations,
    functional,
    point_index,
):
    return calculate_point_effect_cpu(
        base_chkfiles,
        coord,
        surface_charge,
        solvent,
        state_of_interest,
        triplet,
        properties_to_calculate,
        required_calculations,
        functional,
        point_index,
    )


@ray.remote(num_cpus=1, num_gpus=1, max_retries=0, memory=4 * 1024 * 1024 * 1024)
def calculate_point_effect_gpu(
    base_chkfiles,
    coord,
    surface_charge,
    solvent,
    state_of_interest,
    triplet,
    properties_to_calculate,
    required_calculations,
    functional,
    point_index,
):
    gpu_id = ray.get_gpu_ids()[0]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Point {point_index}: Using GPU {gpu_id}, PID {os.getpid()}")

    worker_dir = f"point_{point_index}"
    os.makedirs(worker_dir, exist_ok=True)

    worker_chkfiles = {}
    for key, chkfile in base_chkfiles.items():
        if chkfile:
            worker_chkfile = os.path.join(worker_dir, os.path.basename(chkfile))
            shutil.copy2(chkfile, worker_chkfile)
            worker_chkfiles[key] = worker_chkfile
        else:
            worker_chkfiles[key] = None

    original_dir = os.getcwd()
    os.chdir(worker_dir)

    try:
        from .runner import calculate_surface_effect_at_point

        effects = calculate_surface_effect_at_point(
            {k: os.path.basename(v) if v else None for k, v in worker_chkfiles.items()},
            coord,
            surface_charge,
            solvent,
            state_of_interest,
            triplet,
            properties_to_calculate,
            required_calculations,
            force_single_gpu=True,
        )

        os.chdir(original_dir)
        return {
            "point_index": point_index,
            "coord": coord,
            "charge": surface_charge,
            "effects": effects,
            "success": True,
            "error_msg": None,
        }

    except Exception as e:
        error_msg = f"Error at point {point_index}: {e}"
        print(error_msg)
        os.chdir(original_dir)
        return {
            "point_index": point_index,
            "coord": coord,
            "charge": surface_charge,
            "effects": None,
            "success": False,
            "error_msg": error_msg,
        }

    finally:
        if os.path.exists(worker_dir):
            shutil.rmtree(worker_dir, ignore_errors=True)

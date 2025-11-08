import csv, os, argparse, ast, sys, shutil, json, ray, logging
import numpy as np
from datetime import datetime
from pathlib import Path
from . import core


## Suppress Ray Warnings and Logs
# os.environ['RAY_DASHBOARD_LOG_TO_STDERR'] = '0'
# os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'


##############################################
#              Dependency Setup              #
##############################################

HARTREE_TO_KCAL = 627.509
HARTREE_TO_EV = 27.2114


PROPERTY_CONFIG = {
    'gse': {'deps': [], 'calc': [], 'unit': 1},
    'homo': {'deps': [], 'calc': [], 'unit': 1},
    'lumo': {'deps': [], 'calc': [], 'unit': 1},
    'gap': {'deps': ['homo', 'lumo'], 'calc': [], 'unit': 1},
    'dm': {'deps': [], 'calc': [], 'unit': 1},
    'ie': {'deps': [], 'calc': ['cation'], 'unit': HARTREE_TO_KCAL},
    'ea': {'deps': [], 'calc': ['anion'], 'unit': HARTREE_TO_KCAL},
    'cp': {'deps': ['ie', 'ea'], 'calc': [], 'unit': HARTREE_TO_KCAL},
    'eng': {'deps': ['cp'], 'calc': [], 'unit': HARTREE_TO_EV},
    'hard': {'deps': ['ie', 'ea'], 'calc': [], 'unit': HARTREE_TO_EV},
    'efl': {'deps': ['cp', 'hard'], 'calc': [], 'unit': HARTREE_TO_EV},
    'nfl': {'deps': ['efl'], 'calc': [], 'unit': HARTREE_TO_EV},
    'exe': {'deps': [], 'calc': ['td'], 'unit': 1},
    'osc': {'deps': [], 'calc': ['td'], 'unit': 1}
}


#####################################################
# Prepare Calculations' Dependencies on User Inputs #
#####################################################

def setup_calculation(requested_props):
    """
    Setup and resolve property dependencies and required calculations.
    
    This function takes a list of requested molecular properties and determines
    all the dependencies and quantum mechanical calculations needed to compute them.
    It handles the dependency tree resolution and maps properties to required
    calculations (neutral, cation, anion, TD).
    
    Args:
        requested_props (list): List of property names to calculate.
                               Use 'all' to calculate all available properties.
                               
    Returns:
        tuple: (props_needed, calcs_needed) where:
            - props_needed (list): All properties needed including dependencies
            - calcs_needed (dict): Dictionary mapping calculation types to boolean
                                  (e.g., {'neutral': True, 'cation': False, ...})
    
    Note:
        Available properties: 'gse', 'homo', 'lumo', 'gap', 'dm', 'ie', 'ea', 
        'cp', 'eng', 'hard', 'efl', 'nfl', 'exe', 'osc' and the all encompassing 'all'.
        
        Dependencies are automatically resolved (e.g., 'gap' requires 'homo' and 'lumo')
    """
    if 'all' in requested_props:
        requested_props = list(PROPERTY_CONFIG.keys())
    
    # Resolve dependencies
    props_needed = set()
    def add_deps(prop):
        if prop in props_needed: return
        props_needed.add(prop)
        for dep in PROPERTY_CONFIG[prop]['deps']:
            add_deps(dep)
    
    for prop in requested_props:
        add_deps(prop)
    
    # Determine required calculations
    calcs_needed = {'neutral': True}
    for prop in props_needed:
        for calc in PROPERTY_CONFIG[prop]['calc']:
            calcs_needed[calc] = True
    
    return list(props_needed), calcs_needed


###############################################
# Calculate Properties from Molecular Objects #
###############################################

def calculate_all_properties(mf, anion_mf=None, cation_mf=None, td_obj=None, triplet=False, props_to_calc=None):
    """
    Calculate a comprehensive set of molecular properties from quantum calculations.
    
    This function computes various molecular properties including energetic,
    electronic, and excited state properties from converged SCF objects.
    
    Args:
        mf (pyscf.scf or gpu4pyscf.scf object): Converged neutral molecule SCF object
        anion_mf (pyscf.scf or gpu4pyscf.scf object, optional): Converged anion SCF object for EA calculations
        cation_mf (pyscf.scf or gpu4pyscf.scf object, optional): Converged cation SCF object for IE calculations  
        td_obj (pyscf.tdscf or gpu4pyscf.scf object, optional): TD object for excited state properties
        triplet (bool, optional): Whether to calculate triplet excited states. Defaults to False.
        props_to_calc (list, optional): List of properties to calculate
        
    Returns:
        dict: Dictionary containing calculated properties with units:
            - 'gse': Ground state energy (kcal/mol)
            - 'homo'/'lumo'/'gap': Orbital energies (eV)
            - 'dm': Dipole moment magnitude (Debye)
            - 'ie'/'ea': Ionization/electron affinity (kcal/mol)
            - 'cp': Chemical potential (kcal/mol)
            - 'eng': Electronegativity (eV)
            - 'hard': Chemical hardness (eV)
            - 'efl'/'nfl': Electrophilicity/nucleophilicity (eV)
            - 's1_exe'/'t1_exe': Excitation energies (eV)
            - 's1_osc'/'t1_osc': Oscillator strengths (dimensionless)
            
    Note:
        - Energies are converted from Hartree using conversion constants
        - Excited state properties are labeled with state prefix (s/t) and number
        - Missing SCF objects will skip dependent property calculations
    """
    results = {}
    
    # Basic properties
    if 'gse' in props_to_calc:
        results['gse'] = mf.e_tot * HARTREE_TO_KCAL
    
    if any(p in props_to_calc for p in ['homo', 'lumo', 'gap']):
        homo, lumo, gap = [x * HARTREE_TO_EV for x in core.find_homo_lumo_and_gap(mf)]
        results.update({p: v for p, v in zip(['homo', 'lumo', 'gap'], [homo, lumo, gap]) if p in props_to_calc})
    
    if 'dm' in props_to_calc:
        results['dm'] = np.linalg.norm(mf.dip_moment())
    
    # Charged state properties
    if 'ie' in props_to_calc and cation_mf:
        results['ie'] = cation_mf.e_tot - mf.e_tot
    if 'ea' in props_to_calc and anion_mf:
        results['ea'] = mf.e_tot - anion_mf.e_tot
    
    # Derived properties
    if 'cp' in props_to_calc and all(k in results for k in ['ie', 'ea']):
        results['cp'] = -(results['ie'] + results['ea']) / 2
    if 'eng' in props_to_calc and 'cp' in results:
        results['eng'] = -results['cp']
    if 'hard' in props_to_calc and all(k in results for k in ['ie', 'ea']):
        results['hard'] = (results['ie'] - results['ea']) / 2
    if 'efl' in props_to_calc and all(k in results for k in ['cp', 'hard']):
        results['efl'] = results['cp']**2 / (2 * results['hard']) if results['hard'] != 0 else 0
    if 'nfl' in props_to_calc and 'efl' in results:
        results['nfl'] = 1/results['efl'] if results['efl'] != 0 else 0
    
    # Excited state properties - only if TD object exists and exe/osc properties are requested
    if td_obj and any(p in props_to_calc for p in ['exe', 'osc']):
        state_prefix = 't' if triplet else 's'
        
        if 'exe' in props_to_calc:
            excitation_energies = td_obj.e * HARTREE_TO_EV
            for i, energy in enumerate(excitation_energies, 1):
                results[f'{state_prefix}{i}_exe'] = energy
        
        if 'osc' in props_to_calc:
            oscillator_strengths = td_obj.oscillator_strength()
            for i, osc in enumerate(oscillator_strengths, 1):
                results[f'{state_prefix}{i}_osc'] = osc

    
    return results

# def calculate_surface_effect_at_point(base_chkfiles, coord, surface_charge, solvent, 
#                                       state_of_interest, triplet, properties_to_calculate, 
#                                       required_calculations, functional, force_single_gpu=False):
#     """
#     Calculate the effect of a surface charge at a single coordinate point.
    
#     Args:
#         base_chkfiles (dict): Dictionary with keys 'neutral', 'anion', 'cation' 
#                              pointing to checkpoint file paths
#         coord (array-like): 3D coordinates [x, y, z] of the surface charge
#         surface_charge (float): Magnitude of the point charge
#         solvent (str or None): Solvent for implicit solvation
#         state_of_interest (int): Number of excited states for TD calculations
#         triplet (bool): Whether to calculate triplet excited states
#         properties_to_calculate (list): List of molecular properties to compute
#         required_calculations (dict): Dictionary specifying needed calculations
#         functional (str): XC functional for DFT calculations
#         force_single_gpu (bool): Skip TD subprocess isolation (for Ray workers)
        
#     Returns:
#         dict: Dictionary of property effects
#     """
#     # Resurrect base molecules from checkpoint files
#     molecule_alone = core.resurrect_mol(base_chkfiles['neutral']) if base_chkfiles.get('neutral') else None
#     anion_alone = core.resurrect_mol(base_chkfiles['anion']) if base_chkfiles.get('anion') else None
#     cation_alone = core.resurrect_mol(base_chkfiles['cation']) if base_chkfiles.get('cation') else None
    
#     # Create TD object if needed - pass force_single_gpu flag
#     td_alone = None
#     if required_calculations.get('td', False) and molecule_alone:
#         td_alone = core.create_td_molecule_object(
#             molecule_alone, 
#             nstates=state_of_interest, 
#             triplet=triplet,
#             force_single_gpu=force_single_gpu  
#         )
    
#     # Create single-point charge array
#     single_coord = np.array([coord])
#     q_mm = np.array([surface_charge])
    
#     # Create QM/MM objects for this point
#     molecule_wsc, anion_wsc, cation_wsc, td_wsc = create_wsc_objects(
#         [molecule_alone, anion_alone, cation_alone, td_alone], 
#         single_coord, q_mm, state_of_interest, triplet, required_calculations
#     )
    
#     # Apply solvation if needed
#     if solvent:
#         all_molecules = [molecule_alone, anion_alone, cation_alone, td_alone, 
#                        molecule_wsc, anion_wsc, cation_wsc, td_wsc]
#         all_molecules = apply_solvation(all_molecules, solvent, state_of_interest, triplet, required_calculations)
#         molecule_alone, anion_alone, cation_alone, td_alone, \
#         molecule_wsc, anion_wsc, cation_wsc, td_wsc = all_molecules
    
#     # Calculate properties
#     results = calculate_all_properties(molecule_alone, anion_mf=anion_alone, 
#                                      cation_mf=cation_alone, td_obj=td_alone, 
#                                      triplet=triplet, props_to_calc=properties_to_calculate)
#     wsc_results = calculate_all_properties(molecule_wsc, anion_mf=anion_wsc, 
#                                          cation_mf=cation_wsc, td_obj=td_wsc, 
#                                          triplet=triplet, props_to_calc=properties_to_calculate)
    
#     # Calculate differences
#     effects = {}
#     for prop in results:
#         if prop in wsc_results:
#             effects[f'{prop}_effect'] = wsc_results[prop] - results[prop]
    
#     return effects


def calculate_surface_effect_at_point(base_chkfiles, coord, surface_charge, solvent, 
                                      state_of_interest, triplet, properties_to_calculate, 
                                      required_calculations, functional, force_single_gpu=False):
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
        molecule_alone = core.resurrect_mol(base_chkfiles['neutral']) if base_chkfiles.get('neutral') else None
        anion_alone = core.resurrect_mol(base_chkfiles['anion']) if base_chkfiles.get('anion') else None
        cation_alone = core.resurrect_mol(base_chkfiles['cation']) if base_chkfiles.get('cation') else None
        
        # Create TD object if needed - pass force_single_gpu flag
        td_alone = None
        if required_calculations.get('td', False) and molecule_alone:
            td_alone = core.create_td_molecule_object(
                molecule_alone, 
                nstates=state_of_interest, 
                triplet=triplet,
                force_single_gpu=force_single_gpu  
            )
        
        # Create single-point charge array
        single_coord = np.array([coord])
        q_mm = np.array([surface_charge])
        
        # Create QM/MM objects for this point
        molecule_wsc, anion_wsc, cation_wsc, td_wsc = create_wsc_objects(
            [molecule_alone, anion_alone, cation_alone, td_alone], 
            single_coord, q_mm, state_of_interest, triplet, required_calculations
        )
        
        # Apply solvation if needed
        if solvent:
            all_molecules = [molecule_alone, anion_alone, cation_alone, td_alone, 
                           molecule_wsc, anion_wsc, cation_wsc, td_wsc]
            all_molecules = apply_solvation(all_molecules, solvent, state_of_interest, triplet, required_calculations)
            molecule_alone, anion_alone, cation_alone, td_alone, \
            molecule_wsc, anion_wsc, cation_wsc, td_wsc = all_molecules
        
        # Calculate properties
        results = calculate_all_properties(molecule_alone, anion_mf=anion_alone, 
                                         cation_mf=cation_alone, td_obj=td_alone, 
                                         triplet=triplet, props_to_calc=properties_to_calculate)
        wsc_results = calculate_all_properties(molecule_wsc, anion_mf=anion_wsc, 
                                             cation_mf=cation_wsc, td_obj=td_wsc, 
                                             triplet=triplet, props_to_calc=properties_to_calculate)
        
        # Calculate differences
        effects = {}
        for prop in results:
            if prop in wsc_results:
                effects[f'{prop}_effect'] = wsc_results[prop] - results[prop]
        
        return effects
        
    finally:
        # RESTORE CHECKPOINT FILES FROM BACKUP
        for key, backup_file in backup_files.items():
            original_file = base_chkfiles[key]
            if os.path.exists(backup_file):
                shutil.move(backup_file, original_file)  # Restore original


##########################################################
#        Logging Infrastructure                         #
##########################################################

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
    
    with open(summary_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"{'ELECTROSTATIC TUNING MAP CALCULATION SUMMARY':^70}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Start Time:         {datetime.now().isoformat()}\n")
        f.write(f"Calculation Type:   {calc_type}\n")
        f.write(f"Total Points:       {total_points}\n")
        f.write(f"Status:             IN PROGRESS\n\n")
    
    print(f"Summary log initialized: {summary_file}")
    return summary_file

def append_point_to_summary(summary_file, point_index, coord, charge, effects, 
                           success=True, error_msg=None, total_points=None):
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
    with open(summary_file, 'a') as f:
        # Write point header
        progress = f"[{point_index+1}/{total_points}]" if total_points else f"Point {point_index}"
        f.write(f"\n{progress} " + "-"*55 + "\n")
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

def finalize_summary_log(summary_file, all_effects, surface_coords, point_charges):
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
    
    with open(summary_file, 'a') as f:
        f.write("\n" + "="*70 + "\n")
        f.write("FINAL STATISTICS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Completion Time:    {datetime.now().isoformat()}\n")
        f.write(f"Successful Points:  {successful}\n")
        f.write(f"Failed Points:      {failed}\n")
        f.write(f"Success Rate:       {successful/len(all_effects)*100:.2f}%\n\n")
        
        # Get all unique property keys from successful calculations
        all_keys = set()
        for effects in all_effects:
            if effects:
                all_keys.update(effects.keys())
        
        if all_keys:
            f.write("-"*70 + "\n")
            f.write("STATISTICS FOR EACH PROPERTY EFFECT\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Property':<25} {'Min':>12} {'Max':>12} {'Mean':>12} {'Std Dev':>12}\n")
            f.write("-"*70 + "\n")
            
            for key in sorted(all_keys):
                # Convert all values to float to handle both CPU and GPU arrays
                values = []
                for e in all_effects:
                    if e and key in e:
                        val = e[key]
                        # Handle CuPy arrays from GPU calculations
                        if hasattr(val, 'get'):
                            val = float(val.get())
                        else:
                            val = float(val)
                        values.append(val)
                
                if values:
                    f.write(f"{key:<25} {min(values):>12.6f} {max(values):>12.6f} "
                           f"{np.mean(values):>12.6f} {np.std(values):>12.6f}\n")
            
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("Calculation complete. Individual point files: point_XXXX.out\n")
        f.write("="*70 + "\n")
    
    print(f"\nFinal statistics written to: {summary_file}")

def log_point_result(logs_dir, point_index, coord, charge, effects, success=True, error_msg=None):
    """Log individual point calculation result to structured .out file."""
    log_file = os.path.join(logs_dir, f"point_{point_index:04d}.out")
    
    with open(log_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"{'SURFACE POINT CALCULATION RESULTS':^70}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Point Index:        {point_index}\n")
        f.write(f"Timestamp:          {datetime.now().isoformat()}\n")
        f.write(f"Status:             {'SUCCESS' if success else 'FAILED'}\n\n")
        
        f.write("-"*70 + "\n")
        f.write("COORDINATES AND CHARGE\n")
        f.write("-"*70 + "\n")
        f.write(f"X-coordinate:       {coord[0]:>12.6f} Angstrom\n")
        f.write(f"Y-coordinate:       {coord[1]:>12.6f} Angstrom\n")
        f.write(f"Z-coordinate:       {coord[2]:>12.6f} Angstrom\n")
        f.write(f"Surface Charge:     {charge:>12.6f} a.u.\n\n")
        
        if success and effects:
            f.write("-"*70 + "\n")
            f.write("CALCULATED EFFECTS\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Property':<30} {'Effect Value':>20} {'Unit':>15}\n")
            f.write("-"*70 + "\n")
            
            # Group effects by type
            energy_effects = {}
            orbital_effects = {}
            excited_effects = {}
            other_effects = {}
            
            for key, value in effects.items():
                if any(x in key for x in ['gse', 'ie', 'ea', 'cp', 'eng', 'hard', 'efl', 'nfl']):
                    energy_effects[key] = value
                elif any(x in key for x in ['homo', 'lumo', 'gap']):
                    orbital_effects[key] = value
                elif 'exe' in key or 'osc' in key:
                    excited_effects[key] = value
                else:
                    other_effects[key] = value
            
            # Write energy effects
            if energy_effects:
                f.write("\n  Energy Properties:\n")
                for key, value in sorted(energy_effects.items()):
                    unit = 'eV' if 'eng' in key or 'hard' in key or 'efl' in key or 'nfl' in key else 'kcal/mol'
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
                    unit = 'eV' if 'exe' in key else 'dimensionless'
                    f.write(f"    {key:<36} {value:>20.14f} {unit:>17}\n")
            
            # Write other effects
            if other_effects:
                f.write("\n  Other Properties:\n")
                for key, value in sorted(other_effects.items()):
                    f.write(f"    {key:<36} {value:>20.14f} {'a.u.':>17}\n")
                    
        elif error_msg:
            f.write("-"*70 + "\n")
            f.write("ERROR INFORMATION\n")
            f.write("-"*70 + "\n")
            f.write(f"{error_msg}\n\n")
        
        f.write("\n" + "="*70 + "\n")


###############################################
#              Parallel Processing            #
###############################################
##########################################################
# Modified Remote Functions - Return Metadata            #
##########################################################

@ray.remote(num_cpus=1, num_gpus=0, max_retries=0, memory=4*1024*1024*1024)
def calculate_point_effect_cpu(base_chkfiles, coord, surface_charge, solvent, state_of_interest, triplet, properties_to_calculate, required_calculations, functional, point_index):
    cpu_id = os.sched_getaffinity(0)
    print(f"[Point {point_index}] Running on CPU cores: {cpu_id}, PID: {os.getpid()}")
    
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
        effects = calculate_surface_effect_at_point(
            {k: os.path.basename(v) if v else None for k, v in worker_chkfiles.items()},
            coord, surface_charge, 
            solvent, state_of_interest, triplet, properties_to_calculate, 
            required_calculations, functional
        )
        os.chdir(original_dir)
        return {
            'point_index': point_index,
            'coord': coord,
            'charge': surface_charge,
            'effects': effects,
            'success': True,
            'error_msg': None
        }
        
    except Exception as e:
        error_msg = f"Error at point {point_index}: {e}"
        print(error_msg)
        os.chdir(original_dir)
        return {
            'point_index': point_index,
            'coord': coord,
            'charge': surface_charge,
            'effects': None,
            'success': False,
            'error_msg': error_msg
        }
        
    finally:
        if os.path.exists(worker_dir):
            shutil.rmtree(worker_dir, ignore_errors=True)


@ray.remote(num_cpus=1, num_gpus=1, max_retries=0, memory=4*1024*1024*1024)
def calculate_point_effect_gpu(base_chkfiles, coord, surface_charge, solvent, 
                               state_of_interest, triplet, properties_to_calculate, 
                               required_calculations, functional, point_index):
    gpu_id = ray.get_gpu_ids()[0]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
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
        effects = calculate_surface_effect_at_point(
            {k: os.path.basename(v) if v else None for k, v in worker_chkfiles.items()},
            coord, surface_charge, 
            solvent, state_of_interest, triplet, properties_to_calculate, 
            required_calculations, functional,
            force_single_gpu=True
        )
        
        os.chdir(original_dir)
        return {
            'point_index': point_index,
            'coord': coord,
            'charge': surface_charge,
            'effects': effects,
            'success': True,
            'error_msg': None
        }
        
    except Exception as e:
        error_msg = f"Error at point {point_index}: {e}"
        print(error_msg)
        os.chdir(original_dir)
        return {
            'point_index': point_index,
            'coord': coord,
            'charge': surface_charge,
            'effects': None,
            'success': False,
            'error_msg': error_msg
        }
        
    finally:
        if os.path.exists(worker_dir):
            shutil.rmtree(worker_dir, ignore_errors=True)

##########################################################
#        Surface Data Loading and Validation             #
##########################################################

def load_surface_data(surface_type, calc_type, surface_file='surface.etm', xyz_file=None, density=1.0, scale=1.0):
    """
    Load or generate surface coordinates and charges based on surface type.
    
    Args:
        surface_type (str): 'homogenous' or 'heterogenous'
        calc_type (str): 'separate' or 'combined'
        surface_file (str): Path to surface file (default: 'surface.etm')
        xyz_file (str, optional): Path to XYZ file for VDW surface generation
        density (float): Surface point density for VDW generation
        scale (float): Scaling factor for VDW radii
        
    Returns:
        tuple: (surface_coords, surface_charges) where:
            - surface_coords: numpy array of shape [N, 3]
            - surface_charges: numpy array of shape [N] or None for homogenous
            
    Raises:
        FileNotFoundError: If surface file is required but not found
        ValueError: If surface_type is invalid
    """
    if surface_file is None:
        surface_file = 'surface.etm'

    if surface_type == 'homogenous':
        # Check if surface file exists
        if not os.path.exists(surface_file):
            # Generate VDW surface and save to surface file
            if xyz_file is None:
                raise ValueError("xyz_file required to generate VDW surface")
            
            print(f"Generating VDW surface and saving to {surface_file}...")
            coords = core.get_vdw_surface_coordinates(xyz_file, density=density, scale=scale)
            
            # Save to surface file (x, y, z format only)
            with open(surface_file, 'w') as f:
                f.write(f"{'x':<10} {'y':<10} {'z':<10}\n")
                for coord in coords:
                    f.write(f"{coord[0]:<10.6f} {coord[1]:<10.6f} {coord[2]:<10.6f}\n")
        else:
            # Load existing surface file
            print(f"Loading surface coordinates from {surface_file}...")
            data = np.loadtxt(surface_file, skiprows=1)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            coords = data[:, :3]  # Take only x, y, z columns
        
        # Return coords with None for charges (will be set per calculation)
        return coords, None
        
    elif surface_type == 'heterogenous':
        # Must have surface file with 4 columns
        if not os.path.exists(surface_file):
            raise FileNotFoundError(
                f"For heterogenous surfaces, {surface_file} with x, y, z, q columns is required"
            )
        
        print(f"Loading surface coordinates and charges from {surface_file}...")
        data = np.loadtxt(surface_file, skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        if data.shape[1] < 4:
            raise ValueError(
                f"{surface_file} must have 4 columns (x, y, z, q) for heterogenous surfaces"
            )
        
        coords = data[:, :3]
        charges = data[:, 3]
        
        return coords, charges
    
    else:
        raise ValueError(f"Invalid surface_type: {surface_type}. Must be 'homogenous' or 'heterogenous'")


def calculate_combined_surface_effect(base_chkfiles, coords, charges, solvent, state_of_interest, 
                                      triplet, properties_to_calculate, required_calculations, functional):
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
    molecule_alone = core.resurrect_mol(base_chkfiles['neutral']) if base_chkfiles.get('neutral') else None
    anion_alone = core.resurrect_mol(base_chkfiles['anion']) if base_chkfiles.get('anion') else None
    cation_alone = core.resurrect_mol(base_chkfiles['cation']) if base_chkfiles.get('cation') else None
    
    # Create TD object if needed
    td_alone = None
    if required_calculations.get('td', False) and molecule_alone:
        td_alone = core.create_td_molecule_object(molecule_alone, nstates=state_of_interest, triplet=triplet)
    
    # Create QM/MM objects with ALL charges at once
    molecule_wsc, anion_wsc, cation_wsc, td_wsc = create_wsc_objects(
        [molecule_alone, anion_alone, cation_alone, td_alone], 
        coords, charges, state_of_interest, triplet, required_calculations
    )
    
    # Apply solvation if needed
    if solvent:
        all_molecules = [molecule_alone, anion_alone, cation_alone, td_alone, 
                       molecule_wsc, anion_wsc, cation_wsc, td_wsc]
        all_molecules = apply_solvation(all_molecules, solvent, state_of_interest, triplet, required_calculations)
        molecule_alone, anion_alone, cation_alone, td_alone, \
        molecule_wsc, anion_wsc, cation_wsc, td_wsc = all_molecules
    
    # Calculate properties
    results = calculate_all_properties(molecule_alone, anion_mf=anion_alone, 
                                     cation_mf=cation_alone, td_obj=td_alone, 
                                     triplet=triplet, props_to_calc=properties_to_calculate)
    wsc_results = calculate_all_properties(molecule_wsc, anion_mf=anion_wsc, 
                                         cation_mf=cation_wsc, td_obj=td_wsc, 
                                         triplet=triplet, props_to_calc=properties_to_calculate)
    
    # Calculate differences
    effects = {}
    for prop in results:
        if prop in wsc_results:
            effects[f'{prop}_effect'] = wsc_results[prop] - results[prop]
    
    return effects

##########################################################
# Create Molecular Objects for Any Requested Calculation #
##########################################################
def create_alone_molecule_objects(input_data, basis_set, method, functional, charge, charge_change, gpu_available, spin_guesses):
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
        spin_guesses=spin_guesses
    )

def create_molecule_objects(input_data, basis_set, spin_guesses, method, functional, charge, gpu_available, required_calculations, state_of_interest, triplet):
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
    calc_configs = [('neutral', 0, spin_guesses), ('anion', -1, None), ('cation', +1, None)]
    for name, charge_change, spin_guesses in calc_configs:
        if required_calculations.get(name, False):
            molecules[name] = create_alone_molecule_objects(
                input_data, basis_set, method, functional, charge, 
                charge_change, gpu_available, spin_guesses
            )

    chkfile_map = {'neutral': 'molecule_alone.chk', 'anion': 'anion_alone.chk', 'cation': 'cation_alone.chk'}
    for key, filename in chkfile_map.items():
        if molecules.get(key):
            core.save_chkfile(molecules[key], filename, functional)

    # Create TD object if needed
    td_obj = None
    if required_calculations.get('td', False) and molecules.get('neutral'):
        td_obj = core.create_td_molecule_object(
            molecules['neutral'], 
            nstates=state_of_interest, 
            triplet=triplet,
            force_single_gpu=not gpu_available
        )

    return [molecules.get(k) for k in ['neutral', 'anion', 'cation']] + [td_obj]

    
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
        ('molecule_wsc', molecule_alone, 'molecule_alone.chk'),
        ('anion_wsc', anion_alone, 'anion_alone.chk'),
        ('cation_wsc', cation_alone, 'cation_alone.chk')
    ]
    
    qmmm_objects = {}
    for name, mol, chkfile in qmmm_configs:
        if mol is not None:
            qmmm_objects[name] = core.create_qmmm_molecule_object(mol, coord, q_mm, chkfile)
    
    # Only create TD if explicitly needed
    if qmmm_objects.get('molecule_wsc') and required_calculations.get('td', False):
        qmmm_objects['td_wsc'] = core.create_td_molecule_object(
            qmmm_objects['molecule_wsc'], nstates=state_of_interest, triplet=triplet
        )
    
    return [qmmm_objects.get(k) for k in ['molecule_wsc', 'anion_wsc', 'cation_wsc', 'td_wsc']]

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
    
    molecule_alone, anion_alone, cation_alone, td_alone, molecule_wsc, anion_wsc, cation_wsc, td_wsc = molecules
    
    solvated = [core.solvate_molecule(mol, solvent) if mol else None 
                for mol in [molecule_alone, anion_alone, cation_alone, molecule_wsc, anion_wsc, cation_wsc]]
    
    # Only create TD objects if needed
    td_alone_new = None
    td_wsc_new = None
    
    if required_calculations.get('td', False):
        if solvated[0]:
            td_alone_new = core.create_td_molecule_object(solvated[0], nstates=state_of_interest, triplet=triplet)
        if solvated[3]:
            td_wsc_new = core.create_td_molecule_object(solvated[3], nstates=state_of_interest, triplet=triplet)
    
    return solvated[:3] + [td_alone_new] + solvated[3:] + [td_wsc_new]



##########################################################
#        Handling Input and Output Data And Files       #
##########################################################

def get_tuning_parameters(filepath='tuning.in'):
    """
    Search for tuning.in file and return parameters.
    
    Args:
        filepath (str): Path to tuning file, defaults to 'tuning.in'
        
    Returns:
        dict: Dictionary of tuning parameters
    """
    if not os.path.exists(filepath):
        return {}
    
    params = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Parse value
                    try:
                        params[key] = ast.literal_eval(value)
                    except (ValueError, SyntaxError):
                        params[key] = value
    except Exception as e:
        print(f"Error parsing tuning.in file: {e}")
        return {}
    
    return params


def prepare_input_data(input_type, input_data, basis_set, method='dft', functional='b3lyp', charge=0, spin=0, gpu_available=False):
    """
    Prepare molecular coordinates from different input formats.
    
    This function handles different input types and converts them to XYZ format where necessary.
    For XYZ file input, it simply returns the file path. For SMILES input, it generates 3D coordinates and saves
    them in an XYZ file. For SMILES input, it performs automatic geometry optimization.
    Dependent on global variables for optimization parameters and core module's optimize_molecule and smiles_to_xyz functions.
    Args:
        input_type (str): Type of input - 'xyz' for XYZ file path or 'smiles' for SMILES string
        input_data (str): Either path to XYZ file or SMILES string
        basis_set (str): Basis set for quantum calculations
        method (str): Quantum calculation method (e.g., 'dft')
        functional (str): Functional for DFT calculations
        charge (int): Molecular charge
        spin (int): Spin multiplicity
        gpu_available (bool): Whether GPU is available for calculations
        
    Returns:
        str: Path to XYZ file with molecular coordinates
        
    Raises:
        ValueError: If input_type is not 'xyz' or 'smiles'
        
    Note:
        - For SMILES input: converts to 3D coordinates and performs geometry optimization
        - For XYZ input: returns the file path unchanged
        - Uses global variables for optimization: basis_set, method, functional, charge, gpu_available
    """
    if input_type.lower() == 'xyz':
        return input_data
    elif input_type.lower() == 'smiles':
        raw_xyz = core.smiles_to_xyz(input_data)
        optimized_xyz = core.optimize_molecule(raw_xyz,
        basis_set=basis_set,
        method=method,
        functional=functional,
        original_charge=charge,
        charge_change=0,
        gpu=gpu_available,
        spin_guesses=None)
        return optimized_xyz
    else:
        raise ValueError("input_type must be 'xyz' or 'smiles'")


def append_raw_properties_to_summary(summary_file, raw_properties):
    """
    Append raw baseline properties to the summary file.
    
    Args:
        summary_file (str): Path to summary file
        raw_properties (dict): Dictionary of baseline property values (no surface effects)
    """
    with open(summary_file, 'a') as f:
        f.write("\n" + "="*70 + "\n")
        f.write("RAW PROPERTIES (NO SURFACE EFFECTS)\n")
        f.write("="*70 + "\n\n")
        
        # Group properties by category for better readability
        energy_props = {}
        orbital_props = {}
        excited_props = {}
        other_props = {}
        
        for prop, value in raw_properties.items():
            if any(x in prop for x in ['gse', 'ie', 'ea', 'cp', 'eng', 'hard', 'efl', 'nfl']):
                energy_props[prop] = value
            elif any(x in prop for x in ['homo', 'lumo', 'gap']):
                orbital_props[prop] = value
            elif 'exe' in prop or 'osc' in prop:
                excited_props[prop] = value
            else:
                other_props[prop] = value
        
        # Write energy properties
        if energy_props:
            f.write("Energy Properties:\n")
            f.write("-" * 50 + "\n")
            for prop, value in sorted(energy_props.items()):
                unit = 'eV' if any(x in prop for x in ['eng', 'hard', 'efl', 'nfl']) else 'kcal/mol'
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
                unit = 'eV' if 'exe' in prop else 'dimensionless'
                f.write(f"  {prop:<15s}: {value:>12.6f}  {unit}\n")
            f.write("\n")
        
        # Write other properties
        if other_props:
            f.write("Other Properties:\n")
            f.write("-" * 50 + "\n")
            for prop, value in sorted(other_props.items()):
                f.write(f"  {prop:<15s}: {value:>12.6f}  a.u.\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
    
    print(f"Raw properties appended to: {summary_file}")

# def organize_results(molecule_name, properties_to_calculate, logs_dir):
#     """
#     Move all output files into a timestamped results folder.
    
#     Args:
#         molecule_name (str): Base name of molecule
#         properties_to_calculate (list): List of properties that were calculated
#         logs_dir (str): Path to logs directory
        
#     Returns:
#         str: Path to created results directory
#     """
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#     results_dir = f"results_{molecule_name}_{timestamp}"
    
#     # Create results directory
#     os.makedirs(results_dir, exist_ok=True)
#     print(f"\nOrganizing results into: {results_dir}/")
    
#     # Move CSV summary file
#     csv_file = f"{molecule_name}_tuning_summary.csv"
#     if os.path.exists(csv_file):
#         shutil.move(csv_file, os.path.join(results_dir, csv_file))
#         print(f"  Moved: {csv_file}")
    
#     # Move all .mol2 files (scan directory for all matching files)
#     mol2_files = []
#     for file in os.listdir('.'):
#         if file.startswith(molecule_name) and file.endswith('.mol2'):
#             shutil.move(file, os.path.join(results_dir, file))
#             mol2_files.append(file)
    
#     if mol2_files:
#         print(f"  Moved {len(mol2_files)} MOL2 files")
    
#     # Move logs directory
#     if logs_dir and os.path.exists(logs_dir):
#         dest_logs = os.path.join(results_dir, 'logs')
#         shutil.move(logs_dir, dest_logs)
#         print(f"  Moved: {logs_dir}/ -> logs/")
    
#     # Create a README file in results directory
#     readme_path = os.path.join(results_dir, 'README.txt')
#     with open(readme_path, 'w') as f:
#         f.write("="*70 + "\n")
#         f.write("ELECTROSTATIC TUNING MAP RESULTS\n")
#         f.write("="*70 + "\n\n")
#         f.write(f"Molecule:           {molecule_name}\n")
#         f.write(f"Timestamp:          {timestamp}\n")
#         f.write(f"Properties:         {', '.join(properties_to_calculate)}\n\n")
#         f.write("Files in this directory:\n")
#         f.write("-" * 70 + "\n")
#         f.write(f"  {csv_file:<40} - Summary CSV with all data\n")
#         f.write(f"  {molecule_name}_*.mol2{'':<24} - MOL2 files (raw values)\n")
#         f.write(f"  {molecule_name}_*_normalized.mol2{'':<14} - MOL2 files (normalized)\n")
#         f.write(f"  logs/{'':<46} - Individual point logs\n")
#         f.write(f"  README.txt{'':<38} - This file\n\n")
#         f.write("="*70 + "\n")
    
#     return results_dir

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
    for file in os.listdir('.'):
        if file.startswith(molecule_name) and file.endswith('.mol2'):
            shutil.move(file, os.path.join(results_dir, file))
            mol2_files.append(file)
    
    if mol2_files:
        print(f"  Moved {len(mol2_files)} MOL2 files")
    
    # Move logs directory
    if logs_dir and os.path.exists(logs_dir):
        dest_logs = os.path.join(results_dir, 'logs')
        
        # Add normalization parameters to summary file BEFORE moving
        if normalization_params:
            summary_file = os.path.join(logs_dir, 'calculation_summary.out')
            if os.path.exists(summary_file):
                with open(summary_file, 'a') as f:
                    f.write("\n" + "="*70 + "\n")
                    f.write("NORMALIZATION PARAMETERS\n")
                    f.write("="*70 + "\n\n")
                    f.write(f"{'Property':<30} {'Min Value':>15} {'Max Value':>15}\n")
                    f.write("-"*70 + "\n")
                    for key, (min_val, max_val) in sorted(normalization_params.items()):
                        f.write(f"{key:<30} {min_val:>15.10f} {max_val:>15.10f}\n")
                    f.write("\n" + "="*70 + "\n")
                print(f"  Added normalization parameters to summary")
        
        shutil.move(logs_dir, dest_logs)
        print(f"  Moved: {logs_dir}/ -> logs/")
    
    # Remove any leftover point_* worker directories
    removed_dirs = []
    for item in os.listdir('.'):
        if os.path.isdir(item) and item.startswith('point_'):
            try:
                shutil.rmtree(item)
                removed_dirs.append(item)
            except Exception as e:
                print(f"  Warning: Could not remove {item}: {e}")
    
    if removed_dirs:
        print(f"  Cleaned up {len(removed_dirs)} worker directories")
    
    # Create a README file in results directory
    readme_path = os.path.join(results_dir, 'README.txt')
    with open(readme_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ELECTROSTATIC TUNING MAP RESULTS\n")
        f.write("="*70 + "\n\n")
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
        f.write("="*70 + "\n")
    
    return results_dir

def create_output_files(surface_coords, all_effects, molecule_name, properties_to_calculate, raw_properties):
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
        prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
        
        # Get baseline value for this property
        baseline_value = raw_properties.get(prop_base, 0.0)
        
        # Create custom MOL2 with baseline in comment line
        filename = f"{molecule_name}_{prop_base}.mol2"
        with open(filename, 'w') as f:
            # Header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{prop_base} | baseline={baseline_value:.6f}\n")
            f.write(f"{len(surface_coords):5d} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")
            
            # Atoms
            f.write("@<TRIPOS>ATOM\n")
            for idx, (coord, effect) in enumerate(zip(surface_coords, all_effects), 1):
                x, y, z = coord
                effect_value = effect.get(key, 0.0) if effect else 0.0
                f.write(f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n")
        
        print(f"Created: {filename}")

    # Create MOL2 files for normalized values
    for key in effect_keys:
        prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
        
        # Get baseline value for this property
        baseline_value = raw_properties.get(prop_base, 0.0)
        
        # Create custom MOL2 with baseline in comment line
        filename = f"{molecule_name}_{prop_base}_normalized.mol2"
        with open(filename, 'w') as f:
            # Header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{prop_base}_normalized | baseline={baseline_value:.6f}\n")
            f.write(f"{len(surface_coords):5d} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")
            
            # Atoms
            f.write("@<TRIPOS>ATOM\n")
            for idx, (coord, norm_effect) in enumerate(zip(surface_coords, normalized_effects), 1):
                x, y, z = coord
                effect_value = norm_effect.get(key, 0.0) if norm_effect else 0.0
                f.write(f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n")
        
        print(f"Created: {filename}")

    # Create CSV summary with coordinates, effects, normalized effects, AND baseline values
    csv_filename = f"{molecule_name}_tuning_summary.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        # Create fieldnames
        fieldnames = ['point_index', 'x', 'y', 'z']
        for key in effect_keys:
            prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
            fieldnames.append(key)  # Raw effect (e.g., 'gse_effect')
            fieldnames.append(f"{key}_normalized")  # Normalized
            fieldnames.append(f"{prop_base}_baseline")  # Baseline
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (coord, effect, norm_effect) in enumerate(zip(surface_coords, all_effects, normalized_effects)):
            row = {
                'point_index': i,
                'x': coord[0],
                'y': coord[1],
                'z': coord[2]
            }
            
            for key in effect_keys:
                prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
                
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

def create_output_files(surface_coords, all_effects, molecule_name, properties_to_calculate, raw_properties):
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
        prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
        
        # Get baseline value for this property
        baseline_value = raw_properties.get(prop_base, 0.0)
        
        # Create custom MOL2 with baseline in comment line
        filename = f"{molecule_name}_{prop_base}.mol2"
        with open(filename, 'w') as f:
            # Header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{prop_base} | baseline={baseline_value:.6f}\n")
            f.write(f"{len(surface_coords):5d} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")
            
            # Atoms
            f.write("@<TRIPOS>ATOM\n")
            for idx, (coord, effect) in enumerate(zip(surface_coords, all_effects), 1):
                x, y, z = coord
                effect_value = effect.get(key, 0.0) if effect else 0.0
                f.write(f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n")
        
        print(f"Created: {filename}")

    # Create MOL2 files for normalized values
    for key in effect_keys:
        prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
        
        # Get baseline value for this property
        baseline_value = raw_properties.get(prop_base, 0.0)
        
        # Create custom MOL2 with baseline in comment line
        filename = f"{molecule_name}_{prop_base}_normalized.mol2"
        with open(filename, 'w') as f:
            # Header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{prop_base}_normalized | baseline={baseline_value:.6f}\n")
            f.write(f"{len(surface_coords):5d} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")
            
            # Atoms
            f.write("@<TRIPOS>ATOM\n")
            for idx, (coord, norm_effect) in enumerate(zip(surface_coords, normalized_effects), 1):
                x, y, z = coord
                effect_value = norm_effect.get(key, 0.0) if norm_effect else 0.0
                f.write(f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n")
        
        print(f"Created: {filename}")

    # Create CSV summary with coordinates, effects, normalized effects, AND baseline values
    csv_filename = f"{molecule_name}_tuning_summary.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        # Create fieldnames
        fieldnames = ['point_index', 'x', 'y', 'z']
        for key in effect_keys:
            prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
            fieldnames.append(key)  # Raw effect (e.g., 'gse_effect')
            fieldnames.append(f"{key}_normalized")  # Normalized
            fieldnames.append(f"{prop_base}_baseline")  # Baseline
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (coord, effect, norm_effect) in enumerate(zip(surface_coords, all_effects, normalized_effects)):
            row = {
                'point_index': i,
                'x': coord[0],
                'y': coord[1],
                'z': coord[2]
            }
            
            for key in effect_keys:
                prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
                
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

# def create_output_files(surface_coords, all_effects, molecule_name, properties_to_calculate, raw_properties):
#     """
#     Create MOL2 files and CSV summary for surface effects analysis.

#     This function scans all effect dictionaries for any key ending in '_effect'
#     (including sX_exe_effect, tX_osc_effect, etc.) and creates output files for each.
#     Creates both normalized and non-normalized versions.

#     Args:
#         surface_coords (numpy.ndarray): Array of surface coordinates with shape [N, 3]
#         all_effects (list): List of effect dictionaries for each surface point
#         molecule_name (str): Base name for output files
#         properties_to_calculate (list): List of calculated molecular properties
#         raw_properties (dict): Dict of baseline property values (no surface effects)

#     Returns:
#         None: Creates files directly on disk
#     """
#     # Gather all effect keys found in all_effects
#     effect_keys = set()
#     for effect in all_effects:
#         if effect:
#             effect_keys.update(effect.keys())
#     effect_keys = sorted(effect_keys)
    
#     # print(f"\nDEBUG: Found {len(effect_keys)} effect keys: {effect_keys[:5]}...")
#     # print(f"DEBUG: First effect dict: {all_effects[0] if all_effects else 'None'}")

#     # Normalize the effects
#     normalized_effects, normalization_params = normalize_effects(all_effects, effect_keys)

#     # Create MOL2 files for non-normalized values
#     for key in effect_keys:
#         prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
        
#         # Get baseline value for this property
#         baseline_value = raw_properties.get(prop_base, 0.0)
        
#         # Create custom MOL2 with baseline in comment line
#         filename = f"{molecule_name}_{prop_base}.mol2"
#         with open(filename, 'w') as f:
#             # Header
#             f.write("@<TRIPOS>MOLECULE\n")
#             f.write(f"{prop_base} | baseline={baseline_value:.6f}\n")
#             f.write(f"{len(surface_coords):5d} 0 0 0\n")
#             f.write("SMALL\n")
#             f.write("GASTEIGER\n")
            
#             # Atoms
#             f.write("@<TRIPOS>ATOM\n")
#             for idx, (coord, effect) in enumerate(zip(surface_coords, all_effects), 1):
#                 x, y, z = coord
#                 effect_value = effect.get(key, 0.0) if effect else 0.0
#                 f.write(f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n")
        
#         print(f"Created: {filename}")

#     # Create MOL2 files for normalized values
#     for key in effect_keys:
#         prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
        
#         # Get baseline value for this property
#         baseline_value = raw_properties.get(prop_base, 0.0)
        
#         # Create custom MOL2 with baseline in comment line
#         filename = f"{molecule_name}_{prop_base}_normalized.mol2"
#         with open(filename, 'w') as f:
#             # Header
#             f.write("@<TRIPOS>MOLECULE\n")
#             f.write(f"{prop_base}_normalized | baseline={baseline_value:.6f}\n")
#             f.write(f"{len(surface_coords):5d} 0 0 0\n")
#             f.write("SMALL\n")
#             f.write("GASTEIGER\n")
            
#             # Atoms
#             f.write("@<TRIPOS>ATOM\n")
#             for idx, (coord, norm_effect) in enumerate(zip(surface_coords, normalized_effects), 1):
#                 x, y, z = coord
#                 effect_value = norm_effect.get(key, 0.0) if norm_effect else 0.0
#                 f.write(f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n")
        
#         print(f"Created: {filename}")

#     # Create CSV summary with coordinates, effects, normalized effects, AND baseline values
#     csv_filename = f"{molecule_name}_tuning_summary.csv"
#     with open(csv_filename, 'w', newline='') as csvfile:
#         # Create fieldnames
#         fieldnames = ['point_index', 'x', 'y', 'z']
#         for key in effect_keys:
#             prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
#             fieldnames.append(key)  # Raw effect (e.g., 'gse_effect')
#             fieldnames.append(f"{key}_normalized")  # Normalized
#             fieldnames.append(f"{prop_base}_baseline")  # Baseline
        
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
        
#         for i, (coord, effect, norm_effect) in enumerate(zip(surface_coords, all_effects, normalized_effects)):
#             row = {
#                 'point_index': i,
#                 'x': coord[0],
#                 'y': coord[1],
#                 'z': coord[2]
#             }
            
#             for key in effect_keys:
#                 prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
                
#                 # Get raw effect value
#                 raw_val = effect.get(key, 0.0) if effect else 0.0
#                 norm_val = norm_effect.get(key, 0.0) if norm_effect else 0.0
#                 base_val = raw_properties.get(prop_base, 0.0)
                
#                 # # Debug first row
#                 # if i == 0:
#                 #     print(f"DEBUG Row 0: {key} = {raw_val}, normalized = {norm_val}, baseline = {base_val}")
                
#                 row[key] = raw_val
#                 row[f"{key}_normalized"] = norm_val
#                 row[f"{prop_base}_baseline"] = base_val
            
#             writer.writerow(row)
    
#     print(f"\nCreated: {csv_filename}")
    
#     # Print normalization parameters
#     print("\nNormalization parameters (min, max):")
#     for key, (min_val, max_val) in normalization_params.items():
#         print(f"  {key}: ({min_val:.6f}, {max_val:.6f})")



def create_output_files(surface_coords, all_effects, molecule_name, properties_to_calculate, raw_properties):
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
        prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
        
        # Get baseline value for this property
        baseline_value = raw_properties.get(prop_base, 0.0)
        
        # Create custom MOL2 with baseline in comment line
        filename = f"{molecule_name}_{prop_base}.mol2"
        with open(filename, 'w') as f:
            # Header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{prop_base} | baseline={baseline_value:.6f}\n")
            f.write(f"{len(surface_coords):5d} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")
            
            # Atoms
            f.write("@<TRIPOS>ATOM\n")
            for idx, (coord, effect) in enumerate(zip(surface_coords, all_effects), 1):
                x, y, z = coord
                effect_value = effect.get(key, 0.0) if effect else 0.0
                f.write(f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n")
        
        print(f"Created: {filename}")

    # Create MOL2 files for normalized values
    for key in effect_keys:
        prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
        
        # Get baseline value for this property
        baseline_value = raw_properties.get(prop_base, 0.0)
        
        # Create custom MOL2 with baseline in comment line
        filename = f"{molecule_name}_{prop_base}_normalized.mol2"
        with open(filename, 'w') as f:
            # Header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{prop_base}_normalized | baseline={baseline_value:.6f}\n")
            f.write(f"{len(surface_coords):5d} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")
            
            # Atoms
            f.write("@<TRIPOS>ATOM\n")
            for idx, (coord, norm_effect) in enumerate(zip(surface_coords, normalized_effects), 1):
                x, y, z = coord
                effect_value = norm_effect.get(key, 0.0) if norm_effect else 0.0
                f.write(f"{idx:5d} H    {x:8.4f} {y:8.4f} {z:8.4f} H1   1 {prop_base.upper():8s} {effect_value:10.6f}\n")
        
        print(f"Created: {filename}")

    # Create CSV summary with coordinates, effects, normalized effects, AND baseline values
    csv_filename = f"{molecule_name}_tuning_summary.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        # Create fieldnames
        fieldnames = ['point_index', 'x', 'y', 'z']
        for key in effect_keys:
            prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
            fieldnames.append(key)  # Raw effect (e.g., 'gse_effect')
            fieldnames.append(f"{key}_normalized")  # Normalized
            fieldnames.append(f"{prop_base}_baseline")  # Baseline
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (coord, effect, norm_effect) in enumerate(zip(surface_coords, all_effects, normalized_effects)):
            row = {
                'point_index': i,
                'x': coord[0],
                'y': coord[1],
                'z': coord[2]
            }
            
            for key in effect_keys:
                prop_base = key.replace('_effect', '') if key.endswith('_effect') else key
                
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

        
def check_all_files_created(molecule_name, surface_coords, properties_to_calculate, all_effects=None):
    if len(surface_coords) == 1:
        core.print_office_quote()
    else:
        missing = []
        # Use effect keys from all_effects if provided, otherwise fallback to properties_to_calculate
        effect_keys = set()
        if all_effects and len(all_effects) > 0:
            for effect in all_effects:
                effect_keys.update(effect.keys())
            # Only check for keys ending with '_effect'
            effect_props = [key.replace('_effect', '') for key in effect_keys if key.endswith('_effect')]
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
            core.print_office_quote()

#################
# Miscellaneous #
#################

def startup_message():
    print(f"="*60)
    print(f"                  Electrostatic Tuning Maps")
    print(f"             Built on efforts by the Gozem Lab")
    print(f"   See: https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489")
    print(f"="*60)
    print(f"\n")


##########################################################
#        Resume/Recovery Infrastructure                  #
##########################################################

def create_resume_metadata(logs_dir, calc_type, surface_type, total_points, 
                          properties_to_calculate, surface_charge=None):
    """
    Create metadata file for resume capability.
    
    Args:
        logs_dir (str): Logs directory path
        calc_type (str): 'separate' or 'combined'
        surface_type (str): 'homogenous' or 'heterogeneous'
        total_points (int): Total number of points
        properties_to_calculate (list): Properties being calculated
        surface_charge (float, optional): Surface charge value
    """
    metadata = {
        'original_start': datetime.now().isoformat(),
        'last_updated': datetime.now().isoformat(),
        'calc_type': calc_type,
        'surface_type': surface_type,
        'total_points': total_points,
        'properties': sorted(properties_to_calculate),
        'surface_charge': surface_charge,
        'completed_points': [],
        'failed_points': [],
        'resume_count': 0
    }
    
    metadata_file = os.path.join(logs_dir, '.resume_metadata.json')
    with open(metadata_file, 'w') as f:
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
    metadata_file = os.path.join(logs_dir, '.resume_metadata.json')
    
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Update completion tracking
        if success:
            if point_index not in metadata['completed_points']:
                metadata['completed_points'].append(point_index)
        else:
            if point_index not in metadata['failed_points']:
                metadata['failed_points'].append(point_index)
        
        metadata['last_updated'] = datetime.now().isoformat()
        
        # Sort for readability
        metadata['completed_points'].sort()
        metadata['failed_points'].sort()
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

def find_incomplete_logs():
    """
    Search current directory for incomplete log directories.
    
    Returns:
        list: List of tuples (logs_dir, metadata_dict) sorted by timestamp (newest first)
    """
    incomplete_runs = []
    
    # Find all logs_* directories
    for item in Path('.').glob('logs_*'):
        if item.is_dir():
            metadata_file = item / '.resume_metadata.json'
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Check if incomplete
                completed = len(metadata['completed_points'])
                total = metadata['total_points']
                
                if completed < total:
                    incomplete_runs.append((str(item), metadata))
    
    # Sort by timestamp (newest first)
    incomplete_runs.sort(key=lambda x: x[1]['original_start'], reverse=True)
    
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
    if metadata['calc_type'] != current_params['calc_type']:
        checks.append(f"calc_type mismatch: {metadata['calc_type']} vs {current_params['calc_type']}")
    
    # Check surface_type
    if metadata['surface_type'] != current_params['surface_type']:
        checks.append(f"surface_type mismatch: {metadata['surface_type']} vs {current_params['surface_type']}")
    
    # Check total_points
    if metadata['total_points'] != current_params['total_points']:
        checks.append(f"total_points mismatch: {metadata['total_points']} vs {current_params['total_points']}")
    
    # Check properties (must be same or subset)
    old_props = set(metadata['properties'])
    new_props = set(current_params['properties'])
    if old_props != new_props:
        checks.append(f"properties mismatch: {old_props} vs {new_props}")
    
    # Check surface_charge for homogenous surfaces
    if metadata['surface_type'] == 'homogenous':
        if abs(metadata.get('surface_charge', 0) - current_params.get('surface_charge', 0)) > 1e-6:
            checks.append(f"surface_charge mismatch: {metadata['surface_charge']} vs {current_params['surface_charge']}")
    
    if checks:
        return False, "\n".join(checks)
    
    return True, None

def load_completed_results_from_logs(logs_dir, total_points):
    """Load previously completed results from individual point log files."""
    existing_results = {}
    
    for point_idx in range(total_points):
        # Look for .out files, not .log files
        point_log = os.path.join(logs_dir, f'point_{point_idx:04d}.out')
        
        if os.path.exists(point_log):
            try:
                # Parse the .out file (it's a formatted text file, not JSON)
                result = parse_point_log_file(point_log)
                
                if result and result['success']:
                    effects = result.get('effects', {})
                    
                    # Verify we have actual effect values (not empty dict)
                    if effects and any(v is not None for v in effects.values()):
                        existing_results[point_idx] = {
                            'effects': effects,
                            'coord': result.get('coord'),
                            'charge': result.get('charge')
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
        with open(log_file, 'r') as f:
            lines = f.readlines()
        
        result = {'coord': [0, 0, 0], 'charge': None, 'effects': {}, 'success': False}
        
        in_effects_section = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Parse status
            if 'Status:' in line and 'SUCCESS' in line:
                result['success'] = True
            
            # Parse coordinates
            if 'X-coordinate:' in line:
                parts = line.split()
                result['coord'][0] = float(parts[1])
            elif 'Y-coordinate:' in line:
                parts = line.split()
                result['coord'][1] = float(parts[1])
            elif 'Z-coordinate:' in line:
                parts = line.split()
                result['coord'][2] = float(parts[1])
            
            # Parse charge
            if 'Surface Charge:' in line:
                parts = line.split()
                result['charge'] = float(parts[2])
            
            # Detect effects section start
            if 'CALCULATED EFFECTS' in line:
                in_effects_section = True
                continue
            
            # Parse effects - look for lines with property names and values
            if in_effects_section:
                # End of effects section
                if line_stripped.startswith('==='):
                    break
                
                # Skip header lines, dividers, and category labels
                if (line_stripped.startswith('-') or 
                    line_stripped.startswith('Property') or
                    line_stripped.endswith('Properties:') or
                    not line_stripped):
                    continue
                
                # Parse effect line (format: "    s1_exe_effect                         -0.22655508              eV")
                parts = line_stripped.split()
                if len(parts) >= 2:
                    try:
                        prop_name = parts[0]
                        value = float(parts[1])
                        result['effects'][prop_name] = value
                    except (ValueError, IndexError):
                        pass
        
        # Only return if we got valid data
        if result['success'] and result['coord'] and result['charge'] is not None:
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
    completed = len(metadata['completed_points'])
    failed = len(metadata['failed_points'])
    total = metadata['total_points']
    remaining = total - completed
    
    print("\n" + "="*70)
    print("INCOMPLETE RUN DETECTED")
    print("="*70)
    print(f"Logs Directory:     {logs_dir}")
    print(f"Original Start:     {metadata['original_start']}")
    print(f"Last Updated:       {metadata['last_updated']}")
    print(f"Calculation Type:   {metadata['calc_type']}")
    print(f"Surface Type:       {metadata['surface_type']}")
    print(f"Total Points:       {total}")
    print(f"Completed:          {completed} ({completed/total*100:.1f}%)")
    print(f"Failed:             {failed}")
    print(f"Remaining:          {remaining}")
    print("="*70)
    
    response = input("\nDo you want to resume this calculation? [y/n]: ").strip().lower()
    
    return response in ['', 'y', 'yes']

def main(tuning_file='tuning.in'):

    #######################################
    #           Preliminary Setup         #
    #######################################
    """Main entry point for tuning calculations"""
    # Print startup message
    startup_message()

    # Get parameters from tuning file
    tuning_params = get_tuning_parameters(tuning_file)
    
    # Extract all parameters
    input_type = tuning_params.get('input_type', 'smiles')
    input_data = tuning_params.get('input_data', 'O')
    basis_set = tuning_params.get('basis_set', '6-31G*')
    method = tuning_params.get('method', 'dft')
    functional = tuning_params.get('functional')
    charge = tuning_params.get('charge', 0)
    spin = tuning_params.get('spin', 0)
    solvent = tuning_params.get('solvent', None)
    density = tuning_params.get('density', 1.0)  
    scale = tuning_params.get('scale', 1.0)

    # Surface calculation parameters
    surface_type = tuning_params.get('surface_type', 'homogenous')
    surface_charge = tuning_params.get('surface_charge', 0.1)
    surface_file = tuning_params.get('surface_file', None)
    calc_type = tuning_params.get('calc_type', 'separate')
    
    # Parallel processing parameter (default: True)
    parallel = tuning_params.get('parallel', True)
    num_procs = tuning_params.get('num_procs', None)

    # Calculation specifics
    properties = tuning_params.get('properties', ['all'])
    state_of_interest = tuning_params.get('state_of_interest', 2)
    triplet = tuning_params.get('triplet', False)

    # Check available hardware
    No_of_GPUs = core.check_gpu_info()
    No_of_CPUs = core.check_cpu_info()

    gpu_available = No_of_GPUs > 0

    # Resolve property dependencies and required calculations
    properties_to_calculate, required_calculations = setup_calculation(properties)
    print(f"Calculating Tuning of:  {properties_to_calculate}")
    print(f"Using molecular states: {required_calculations}")

    # Prepare input data
    input_data = prepare_input_data(input_type, input_data, basis_set, method, functional, charge, spin, gpu_available=gpu_available)
    molecule_name = core.extract_xyz_name(input_data)
    
    # Load surface data based on surface_type
    surface_coords, surface_charges = load_surface_data(
        surface_type, calc_type, surface_file=surface_file, 
        xyz_file=input_data, density=density, scale=scale
    )

    print(f"\n")
    print(f"="*60)
    print(f"                Surface Type: {surface_type}")
    print(f"                Calculation Type: {calc_type}")
    print(f"                Number of surface points: {len(surface_coords)}")
    print(f"                Parallel Processing: {parallel}")
    print(f"="*60)
    print(f"\n")

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
            'calc_type': calc_type,
            'surface_type': surface_type,
            'total_points': len(surface_coords),
            'properties': sorted(properties_to_calculate),
            'surface_charge': surface_charge if surface_type == 'homogenous' else None
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
                    effects = existing_results[idx]['effects']
                    print(f"  Point {idx}: {len(effects)} properties")
                    for prop, value in list(effects.items())[:3]:  # Show first 3
                        print(f"    {prop}: {value}")

                print(f"Will calculate {len(points_to_calculate)} remaining points: {points_to_calculate[:10]}{'...' if len(points_to_calculate) > 10 else ''}")
                
                # Update metadata for resume
                with open(os.path.join(logs_dir, '.resume_metadata.json'), 'r') as f:
                    metadata = json.load(f)
                metadata['resume_count'] += 1
                metadata['last_updated'] = datetime.now().isoformat()
                with open(os.path.join(logs_dir, '.resume_metadata.json'), 'w') as f:
                    json.dump(metadata, f, indent=2)
        else:
            print(f"\nWarning: Found incomplete run but parameters don't match:")
            print(error_msg)
            print("Starting new calculation instead.\n")
    
    # Create logs directory if not resuming
    if not resume_mode:
        logs_dir = setup_logs_directory()
        print(f"Logging results to: {logs_dir}")
        
        # Create resume metadata
        create_resume_metadata(
            logs_dir, calc_type, surface_type, len(surface_coords),
            properties_to_calculate, 
            surface_charge if surface_type == 'homogenous' else None
        )
        
        # Calculate all points
        points_to_calculate = list(range(len(surface_coords)))
    
    
    
    # Create base molecule objects
    base_molecules = create_molecule_objects(
        input_data, basis_set, spin, method, functional, charge, 
        gpu_available, required_calculations, state_of_interest, triplet)


    # Unpack the list returned by create_molecule_objects
    molecule_alone, anion_alone, cation_alone, td_alone = base_molecules

    base_chkfiles = {
        'neutral': 'molecule_alone.chk' if required_calculations.get('neutral') else None,
        'anion': 'anion_alone.chk' if required_calculations.get('anion') else None,
        'cation': 'cation_alone.chk' if required_calculations.get('cation') else None
    }
    
    
    # print(f"\nBase molecule types:")
    # print(f"  molecule_alone: {type(molecule_alone)}")
    # print(f"  Has to_cpu: {hasattr(molecule_alone, 'to_cpu')}")
    
    # Calculate raw properties (baseline - no surface effects)
    raw_properties = calculate_all_properties(
        molecule_alone,
        anion_mf=anion_alone,
        cation_mf=cation_alone,
        td_obj=td_alone,
        triplet=triplet,
        props_to_calc=properties_to_calculate
    )


    if solvent:
        all_molecules = [molecule_alone, anion_alone, cation_alone, td_alone, None, None, None, None]
        all_molecules = apply_solvation(all_molecules, solvent, state_of_interest, triplet, required_calculations)
        molecule_alone, anion_alone, cation_alone, td_alone, _, _, _, _ = all_molecules
    
    # Calculate raw properties (baseline - no surface effects)
    raw_properties = calculate_all_properties(
        molecule_alone,
        anion_mf=anion_alone,
        cation_mf=cation_alone,
        td_obj=td_alone,
        triplet=triplet,
        props_to_calc=properties_to_calculate
    )


    print(f"\n")
    print(f"="*60)
    print(f"                Raw Properties (No Surface)")
    print(f"                Total raw properties calculated: {len(raw_properties)}")
    for prop, value in sorted(raw_properties.items()):
        print(f"                {prop}: {value:.6f}")
    print(f"="*60)
    print(f"\n")

    
    if calc_type == 'combined':
        print(f"Running combined calculation with all {len(surface_coords)} surface points...")
        
        # Initialize or reopen summary log
        summary_file = os.path.join(logs_dir, "calculation_summary.out")
        if not resume_mode:
            summary_file = initialize_summary_log(logs_dir, calc_type, 1)
        
        if surface_type == 'homogenous':
            q_mm = np.full(len(surface_coords), surface_charge)
        else:
            q_mm = surface_charges
        
        try:
            combined_effects = calculate_combined_surface_effect(
                base_chkfiles, surface_coords, q_mm,
                solvent, state_of_interest, triplet, properties_to_calculate,
                required_calculations, functional
            )
            
            # Log combined result (individual file)
            log_point_result(logs_dir, 0, np.mean(surface_coords, axis=0), 
                           surface_charge, combined_effects, success=True)
            
            # Append to summary
            append_point_to_summary(summary_file, 0, np.mean(surface_coords, axis=0),
                                   surface_charge, combined_effects, success=True, total_points=1)
            
            # Update resume metadata
            update_resume_metadata(logs_dir, 0, True)
            
            print(f"Combined effects: {combined_effects}")
            all_effects = [combined_effects]
            output_coords = np.mean(surface_coords, axis=0).reshape(1, 3)
            
        except Exception as e:
            error_msg = f"Combined calculation failed: {e}"
            print(error_msg)
            
            # Log failure
            log_point_result(logs_dir, 0, np.mean(surface_coords, axis=0), 
                           surface_charge, None, success=False, error_msg=error_msg)
            
            # Append failure to summary
            append_point_to_summary(summary_file, 0, np.mean(surface_coords, axis=0),
                                   surface_charge, None, success=False, 
                                   error_msg=error_msg, total_points=1)
            
            # Update resume metadata
            update_resume_metadata(logs_dir, 0, False)
            
            all_effects = [None]
            output_coords = np.mean(surface_coords, axis=0).reshape(1, 3)
        
        # Finalize summary with statistics
        finalize_summary_log(summary_file, all_effects, output_coords, [surface_charge])
        
        # Append raw properties to summary
        append_raw_properties_to_summary(summary_file, raw_properties)

    else:  # calc_type == 'separate'
        if resume_mode:
            print(f"\nResuming calculation for {len(points_to_calculate)} remaining points...")
            summary_file = os.path.join(logs_dir, "calculation_summary.out")
            
            # Append resume header to existing summary
            with open(summary_file, 'a') as f:
                f.write("\n" + "="*70 + "\n")
                f.write(f"RESUMING CALCULATION - {datetime.now().isoformat()}\n")
                f.write(f"Calculating {len(points_to_calculate)} remaining points\n")
                f.write("="*70 + "\n\n")
        else:
            summary_file = initialize_summary_log(logs_dir, calc_type, len(surface_coords))
        
        # Determine charges for each point
        if surface_type == 'homogenous':
            point_charges = [surface_charge] * len(surface_coords)
        else:
            point_charges = surface_charges.tolist()


        # Initialize all_effects with existing results if resuming
        if resume_mode:
            # Create full array with None placeholders
            all_effects = [None] * len(surface_coords)
            
            # Fill in existing results
            for point_idx, result in existing_results.items():
                all_effects[point_idx] = result['effects']
                print(f"  Restored point {point_idx}: {result['effects']}")  # Debug
            
            print(f"\nInitialized all_effects array:")
            print(f"  Total slots: {len(all_effects)}")
            print(f"  Pre-filled: {sum(1 for x in all_effects if x is not None)}")
            print(f"  To calculate: {len(points_to_calculate)}")
        else:
            all_effects = [None] * len(surface_coords)
        
        if parallel:
            if num_procs is None:
                parallel_processes = No_of_CPUs if No_of_GPUs < 1 else No_of_GPUs
            else:
                parallel_processes = min(No_of_CPUs if No_of_GPUs < 1 else No_of_GPUs, num_procs)
            
            # logging.getLogger("ray").setLevel(logging.ERROR)

            if No_of_GPUs < 1:
                ray.init(
                    num_cpus=parallel_processes, 
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    logging_level=logging.ERROR,
                    # log_to_driver=False
                )
                calculate_point_effect = calculate_point_effect_cpu
            else:
                ray.init(
                    num_gpus=parallel_processes, 
                    include_dashboard=False,
                    ignore_reinit_error=True,
                    logging_level=logging.ERROR,
                    # log_to_driver=False
                )
                calculate_point_effect = calculate_point_effect_gpu

            print(f"Using {parallel_processes} parallel processes on {'GPU' if gpu_available else 'CPU'}")

            # MODIFIED: Only submit jobs for points_to_calculate
            batch_size = parallel_processes
            for batch_start in range(0, len(points_to_calculate), batch_size):
                batch_end = min(batch_start + batch_size, len(points_to_calculate))
                batch_indices = [points_to_calculate[i] for i in range(batch_start, batch_end)]
            
                futures = [
                    calculate_point_effect.remote(
                        base_chkfiles, surface_coords[i], point_charges[i], 
                        solvent, state_of_interest, triplet, properties_to_calculate, 
                        required_calculations, functional, i
                    )
                    for i in batch_indices
                ]
            
                # Get results and log them IMMEDIATELY after each completes
                for result in ray.get(futures):
                    point_index = result['point_index']
                    all_effects[point_index] = result['effects']
                    
                    # Log individual point file
                    log_point_result(
                        logs_dir, 
                        point_index, 
                        result['coord'], 
                        result['charge'], 
                        result['effects'],
                        success=result['success'],
                        error_msg=result['error_msg']
                    )
                    
                    # Append to summary file IMMEDIATELY
                    append_point_to_summary(
                        summary_file,
                        point_index,
                        result['coord'],
                        result['charge'],
                        result['effects'],
                        success=result['success'],
                        error_msg=result['error_msg'],
                        total_points=len(surface_coords)
                    )
                    
                    # Update resume metadata
                    update_resume_metadata(logs_dir, point_index, result['success'])
                    
                    status = "SUCCESS" if result['success'] else "FAILED"
                    completed_so_far = len([e for e in all_effects if e is not None])
                    print(f"Point {point_index+1}/{len(surface_coords)}: {status} ({completed_so_far}/{len(surface_coords)} total)")
            
            ray.shutdown()
            
        else:
            # Sequential processing
            print(f"Using sequential processing (parallel=False)")
            for point_idx in points_to_calculate:  # Use point_idx directly
                coord = surface_coords[point_idx]
                try:
                    
                    effects = calculate_surface_effect_at_point(
                        base_chkfiles, coord, point_charges[point_idx],
                        solvent, state_of_interest, triplet, 
                        properties_to_calculate, required_calculations, functional, force_single_gpu=True
                    )
                    
                    all_effects[point_idx] = effects  # Use point_idx
                    
                    # Log individual file
                    log_point_result(logs_dir, point_idx, coord, point_charges[point_idx], effects, success=True)
                    
                    # Append to summary IMMEDIATELY
                    append_point_to_summary(summary_file, point_idx, coord, point_charges[point_idx], 
                                           effects, success=True, total_points=len(surface_coords))
                    
                    # Update resume metadata
                    update_resume_metadata(logs_dir, point_idx, True)
                    
                    completed_so_far = len([e for e in all_effects if e is not None])
                    print(f"Point {point_idx+1}/{len(surface_coords)}: SUCCESS ({completed_so_far}/{len(surface_coords)} total)")
                    
                except Exception as e:
                    all_effects[point_idx] = None  # Use point_idx
                    error_msg = f"Error at point {point_idx}: {e}"
                    print(f"Point {point_idx+1}/{len(surface_coords)}: FAILED - {e}")
                    
                    # Log failure
                    log_point_result(logs_dir, point_idx, coord, point_charges[point_idx], None, 
                                   success=False, error_msg=error_msg)
                    
                    # Append failure to summary IMMEDIATELY
                    append_point_to_summary(summary_file, point_idx, coord, point_charges[point_idx], None,
                                           success=False, error_msg=error_msg, 
                                           total_points=len(surface_coords))
                    
                    # Update resume metadata
                    update_resume_metadata(logs_dir, point_idx, False)
        
        # Now all_effects contains BOTH old and new results
        output_coords = surface_coords
        
        # Finalize summary with ALL results (old + new)
        finalize_summary_log(summary_file, all_effects, output_coords, point_charges)
        
        # Append raw properties to summary
        append_raw_properties_to_summary(summary_file, raw_properties)

    # # Create output files with ALL results (not just new ones)
    # create_output_files(output_coords, all_effects, molecule_name, properties_to_calculate, raw_properties)
    # check_all_files_created(molecule_name, output_coords, properties_to_calculate, all_effects)

    # # Organize results into timestamped folder
    # results_dir = organize_results(molecule_name, properties_to_calculate, logs_dir)


    # # Remove temporary checkpoint files
    # temp_files = ['molecule_alone.chk', 'anion_alone.chk', 'cation_alone.chk']
    # for temp_file in temp_files:
    #     if os.path.exists(temp_file):
    #         os.remove(temp_file)

    # Create output files with ALL results (not just new ones)
    normalization_params = create_output_files(output_coords, all_effects, molecule_name, properties_to_calculate, raw_properties)

    # Remove temporary checkpoint files FIRST
    temp_files = ['molecule_alone.chk', 'anion_alone.chk', 'cation_alone.chk']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    # Organize results into timestamped folder SECOND (with normalization params)
    results_dir = organize_results(molecule_name, properties_to_calculate, logs_dir, normalization_params)
    
    # Change to results directory and check files THIRD
    original_dir = os.getcwd()
    os.chdir(results_dir)
    check_all_files_created(molecule_name, output_coords, properties_to_calculate, all_effects)
    os.chdir(original_dir)

if __name__ == "__main__":
    tuning_file = sys.argv[1] if len(sys.argv) > 1 else 'tuning.in'
    main(tuning_file)

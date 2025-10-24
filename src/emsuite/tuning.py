from . import core
import csv, os, argparse, ast, sys
import numpy as np
import ray

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

def calculate_surface_effect_at_point(base_molecules, coord, surface_charge, solvent, state_of_interest, triplet, properties_to_calculate, required_calculations):
    """
    Calculate the effect of a surface charge at a single coordinate point.
    
    This function computes how a point charge at a specific location affects
    various molecular properties by comparing calculations with and without
    the external charge.
    
    Args:
        base_molecules (list): [neutral_mf, anion_mf, cation_mf, td_obj] base objects
        coord (array-like): 3D coordinates [x, y, z] of the surface charge
        surface_charge (float): Magnitude of the point charge
        solvent (str or None): Solvent for implicit solvation
        state_of_interest (int): Number of excited states for TD calculations
        triplet (bool): Whether to calculate triplet excited states
        properties_to_calculate (list): List of molecular properties to compute
        required_calculations (dict): Dictionary specifying needed calculations
        
    Returns:
        dict: Dictionary of property effects with keys like '{property}_effect'
              containing the difference (with_charge - without_charge)
              
    Note:
        - Creates QM/MM calculations with the single point charge
        - Applies solvation if specified
        - Calculates the difference in properties due to the external charge
        - Effects show how the external charge modifies each property
    """
    molecule_alone, anion_alone, cation_alone, td_alone = base_molecules
    
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
    
    # Only create TD if explicitly needed
    if required_calculations.get('td', False):
        molecules['td'] = core.create_td_molecule_object(molecules['neutral'], nstates=state_of_interest, triplet=triplet)
    
    chkfile_map = {'neutral': 'molecule_alone', 'anion': 'anion_alone', 'cation': 'cation_alone'}
    for key, filename in chkfile_map.items():
        if molecules.get(key):
            core.save_chkfile(molecules[key], filename)
    
    return [molecules.get(k) for k in ['neutral', 'anion', 'cation', 'td']]

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
        - TD objects are only created if explicitly required
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
        - Uses global variables for optimization: basis_set, method, functional, charge, No_of_GPUs
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

def create_output_files(surface_coords, all_effects, molecule_name, properties_to_calculate):
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

    Returns:
        None: Creates files directly on disk
    """
    # Gather all effect keys found in all_effects
    effect_keys = set()
    for effect in all_effects:
        effect_keys.update(effect.keys())
    effect_keys = sorted(effect_keys)

    # Normalize the effects
    normalized_effects, normalization_params = normalize_effects(all_effects, effect_keys)

    # Create MOL2 files for non-normalized values
    for key in effect_keys:
        if key.endswith('_effect'):
            prop_base = key.replace('_effect', '')
            core.create_mol2_file(
                molecule_name,
                surface_coords,
                [effect.get(key, 0.0) for effect in all_effects],
                prop_base
            )

    # Create MOL2 files for normalized values
    for key in effect_keys:
        if key.endswith('_effect'):
            prop_base = key.replace('_effect', '')
            core.create_mol2_file(
                molecule_name,
                surface_coords,
                [effect.get(key, 0.0) for effect in normalized_effects],
                f"{prop_base}_normalized"
            )

    # Create CSV summary with both original and normalized values
    csv_filename = f"{molecule_name}_tuning_summary.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        # Create fieldnames with both original and normalized columns
        fieldnames = ['point_index', 'x', 'y', 'z']
        for key in effect_keys:
            fieldnames.append(key)
            fieldnames.append(f"{key}_normalized")
        
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
                row[key] = effect.get(key, 0.0)
                row[f"{key}_normalized"] = norm_effect.get(key, 0.0)
            writer.writerow(row)
    
    # Print normalization parameters for reference
    print("\nNormalization parameters (min, max):")
    for key, (min_val, max_val) in normalization_params.items():
        print(f"  {key}: ({min_val:.6f}, {max_val:.6f})")
        
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



#######################################
#            Main Execution           #
#######################################

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
    functional = tuning_params.get('functional', 'b3lyp')
    charge = tuning_params.get('charge', 0)
    spin = tuning_params.get('spin', 0)
    surface_charge = tuning_params.get('surface_charge', 1.0)
    solvent = tuning_params.get('solvent', 'water')
    density = tuning_params.get('density', 1.0)  
    scale = tuning_params.get('scale', 1.0)

    # Calculation specifics
    properties = tuning_params.get('properties', ['lumo'])
    state_of_interest = tuning_params.get('state_of_interest', 2)
    triplet = tuning_params.get('triplet', False)
    print(spin)

    # Check available hardware
    No_of_GPUs = core.check_gpu_info()
    No_of_CPUs = core.check_cpu_info()
    
    # Resolve property dependencies and required calculations
    properties_to_calculate, required_calculations = setup_calculation(properties)
    print(f"Calculating Tuning of:  {properties_to_calculate}")
    print(f"Using molecular states: {required_calculations}")


    # Prepare input data
    q_mm = np.array([surface_charge])
    input_data = prepare_input_data(input_type, input_data, basis_set, method, functional, charge, spin, gpu_available=No_of_GPUs > 0)
    molecule_name = core.extract_xyz_name(input_data)
    surface_coords = core.get_vdw_surface_coordinates(input_data, density=density, scale=scale)
    print(f"\n")
    print(f"="*60)
    print(f"         Running calculations on {len(surface_coords)} surface points")
    print(f"="*60)
    print(f"\n")


    # #######################################
    # #    Core Tuning Map Calculations     #
    # #######################################

    # # Create base molecule objects
    base_molecules = create_molecule_objects(
        input_data, basis_set, spin, method, functional, charge, 
        No_of_GPUs > 0, required_calculations, state_of_interest, triplet)
    
    # Calculate effects at each surface point
    all_effects = []
    for i, coord in enumerate(surface_coords):
        effects = calculate_surface_effect_at_point(
            base_molecules, coord, surface_charge, 
            solvent, state_of_interest, triplet, properties_to_calculate, required_calculations
        )
        all_effects.append(effects)
        print(f"Point {i+1}/{len(surface_coords)}: {effects}")

     # Create output files
    create_output_files(surface_coords, all_effects, molecule_name, properties_to_calculate)
    check_all_files_created(molecule_name, surface_coords, properties_to_calculate, all_effects)

    #Remove temporary checkpoint files
    temp_files = ['molecule_alone.chk', 'anion_alone.chk', 'cation_alone.chk']
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

if __name__ == "__main__":
    tuning_file = sys.argv[1] if len(sys.argv) > 1 else 'tuning.in'
    main(tuning_file)

import core,csv, os
import numpy as np


#Constants
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


def setup_calculation(requested_props):
    """Setup properties and calculations needed"""
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


def calculate_all_properties(mf, anion_mf=None, cation_mf=None, td_obj=None, triplet=False, props_to_calc=None):
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


def create_alone_molecule_objects(charge_change, spin_guesses):
    return core.create_molecule_object(
        atom_input=input_data,
        basis_set=basis_set,
        method=method,
        functional=functional,
        original_charge=charge,
        charge_change=charge_change,
        gpu=True if No_of_GPUs > 0 else False,
        spin_guesses=spin_guesses
    )

def create_molecule_objects(required_calculations, state_of_interest, triplet):
    """Create and save required molecule objects"""
    molecules = {}
    
    calc_configs = [('neutral', 0, None), ('anion', -1, None), ('cation', +1, None)]
    for name, charge_change, spin_guesses in calc_configs:
        if required_calculations.get(name, False):
            molecules[name] = create_alone_molecule_objects(charge_change, spin_guesses)
    
    # Only create TD if explicitly needed
    if required_calculations.get('td', False):
        molecules['td'] = core.create_td_molecule_object(molecules['neutral'], nstates=state_of_interest, triplet=triplet)
    
    chkfile_map = {'neutral': 'molecule_alone', 'anion': 'anion_alone', 'cation': 'cation_alone'}
    for key, filename in chkfile_map.items():
        if molecules.get(key):
            core.save_chkfile(molecules[key], filename)
    
    return [molecules.get(k) for k in ['neutral', 'anion', 'cation', 'td']]

def create_wsc_objects(molecules, coord, q_mm, state_of_interest, triplet, required_calculations):
    """Create QM/MM molecule objects from base molecules"""
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
    """Apply solvation to all molecule objects"""
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

def prepare_input_data(input_type, input_data):
    if input_type == 'xyz':
        return input_data
    elif input_type == 'smiles':
        return core.smiles_to_xyz(input_data)
    else:
        raise ValueError("input_type must be 'file' or 'smiles'")


def calculate_surface_effect_at_point(base_molecules, coord, surface_charge, solvent, state_of_interest, triplet, properties_to_calculate, required_calculations):
    """Calculate the effect of a surface charge at a single coordinate point"""
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

def create_output_files(surface_coords, all_effects, molecule_name, properties_to_calculate):
    """Create MOL2 files and CSV summary for surface effects"""
    
    # Collect all effects by property
    property_effects = {}
    for prop in properties_to_calculate:
        prop_effect_key = f'{prop}_effect'
        property_effects[prop] = [effect.get(prop_effect_key, 0.0) for effect in all_effects]
    
    # Create MOL2 files for each property
    for prop, effect_values in property_effects.items():
        core.create_mol2_file(molecule_name, surface_coords, effect_values, prop)
        # print(f"Created {molecule_name}_{prop}_tuning.mol2")
    
    # Create CSV summary
    csv_filename = f"{molecule_name}_tuning_summary.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['point_index', 'x', 'y', 'z'] + [f'{prop}_effect' for prop in properties_to_calculate]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for i, (coord, effect) in enumerate(zip(surface_coords, all_effects)):
            row = {
                'point_index': i,
                'x': coord[0],
                'y': coord[1], 
                'z': coord[2]
            }
            # Add property effects
            for prop in properties_to_calculate:
                row[f'{prop}_effect'] = effect.get(f'{prop}_effect', 0.0)
            writer.writerow(row)
    
    # print(f"Created {csv_filename}")

















####User Input####
#Input options
input_type = 'xyz'  # 'xyz' or 'smiles'
input_data = 'water.xyz' #'O' 

#Calculation options
method = 'dft'
basis_set = "6-31g*"
functional = 'b3lyp'
charge = 0
spin = 0
surface_charge = 1.0 # Charge of the surface point
solvent = None

#Calculation specifics
properties =  ['gse']
state_of_interest = 2
triplet = False

#Options: 'hard', 'cp', 'ea', 'exe', 'gap', 'osc', 'eng', 'lumo', 'gse', 'ie', 'homo', 'dm', 'nfl', 'efl'






####Main Execution####
### Setup Required Calculations and Properties

core.print_startup_message()
print(f"="*60)
print(f"                  Electrostatic Tuning Maps")
print(f"             Built on efforts by the Gozem Lab")
print(f"   See: https://pubs.acs.org/doi/10.1021/acs.jpcb.9b00489")
print(f"="*60)
print(f"\n")

No_of_GPUs = core.check_gpu_info()
No_of_CPUs = core.check_cpu_info()


properties_to_calculate, required_calculations = setup_calculation(properties)
print(f"Properties: {properties_to_calculate}")
print(f"Calculations: {required_calculations}")


# Prepare input data
q_mm = np.array([surface_charge])
input_data = prepare_input_data(input_type, input_data)
molecule_name = os.path.splitext(os.path.basename(input_data))[0]  # Always extract base name from xyz file
molecule_name = molecule_name.replace('/', '_').replace('\\', '_')  # Clean filename
# surface_coords = core.get_vdw_surface_coordinates(input_data)

surface_coords = np.array([[0.143397, -0.368820, 1.585376]])

# Create base molecule objects
base_molecules = create_molecule_objects(required_calculations, state_of_interest, triplet)

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

if len(surface_coords) == 1:
    core.print_office_quote()
else:
    missing = []
    for prop in properties_to_calculate:
        filepath = f"{molecule_name}_{prop}_tm.mol2"
        if not os.path.exists(filepath):
            missing.append(filepath)
    
    csv_path = f"{molecule_name}_tuning_summary.csv"
    if not os.path.exists(csv_path):
        missing.append(csv_path)

    if missing:
        print(f"Missing: {', '.join(missing)}")
    else:
        core.print_office_quote()
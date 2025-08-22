import numpy as np
import os
import subprocess
import sys
from pyscf import gto, scf, dft, qmmm, tdscf
from pyscf.solvent import smd
from pyscf.hessian import thermo
from pyscf.geomopt.geometric_solver import optimize
from rdkit import Chem
from rdkit.Chem import AllChem


class TuningCalculator:
    """
    A comprehensive class for calculating molecular properties and their electronic responses
    to external electrostatics.
    """
    
    # Constants
    HARTREE_TO_KCAL = 627.509
    HARTREE_TO_EV = 27.2114
    SHE_REFERENCE_POTENTIAL = 4.43
    
    # Property dependency and unit mapping
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
    
    def __init__(self, molecule='water', method='dft', functional='b3lyp', 
                 basis_set='augccpvdz', charge=0, spin=1, gfec_functional='b3lyp',
                 gfec_basis_set='6-31+G*', state_of_interest=2, triplet_excitation=False,
                 solvent=None, rdx_solvent='acetonitrile', input_type='xyz',
                 smiles_input='O', optimize_geometry=True):
        """
        Initialize the TuningCalculator with molecular parameters.
        """
        self.molecule = molecule
        self.method = method
        self.functional = functional
        self.basis_set = basis_set
        self.charge = charge
        self.spin = spin
        self.gfec_functional = gfec_functional
        self.gfec_basis_set = gfec_basis_set
        self.state_of_interest = state_of_interest
        self.triplet_excitation = triplet_excitation
        self.solvent = solvent
        self.rdx_solvent = rdx_solvent
        self.input_type = input_type
        self.smiles_input = smiles_input
        self.optimize_geometry = optimize_geometry
        
        # Will be populated during calculations
        self.molecule_object = None
        self.anion_mf = None
        self.cation_mf = None
        self.td_object = None
        self.xyz_file = None
        self.molecule_name = None
        self.surface_coords = []
        self.properties_alone = {}
        self.tuning = {}
    
    def setup_calculation(self, requested_props):
        """Setup properties and calculations needed"""
        if 'all' in requested_props:
            requested_props = list(self.PROPERTY_CONFIG.keys())
        
        # Resolve dependencies
        props_needed = set()
        def add_deps(prop):
            if prop in props_needed: return
            props_needed.add(prop)
            for dep in self.PROPERTY_CONFIG[prop]['deps']:
                add_deps(dep)
        
        for prop in requested_props:
            add_deps(prop)
        
        # Determine required calculations
        calcs_needed = {'neutral': True}
        for prop in props_needed:
            for calc in self.PROPERTY_CONFIG[prop]['calc']:
                calcs_needed[calc] = True
        
        return list(props_needed), calcs_needed
    
    def create_molecule_object(self, atom_input, basis_set, method='dft', 
                              functional='m06-2x', charge=0, spin=1):
        """
        Create a PySCF molecule and KS/DFT object from either:
        - atom_input: list of (atom_symbol, (x,y,z)) tuples
        - atom_input: string path to XYZ file
        """
        spin -= 1
        mol = gto.M(basis=basis_set, charge=charge, spin=spin, symmetry=False)
        if isinstance(atom_input, str) and os.path.isfile(atom_input):
            mol.atom = atom_input 
        else:
            mol.atom = atom_input  
        mol.build()

        # Choose method and handle spin
        if method.lower() == 'dft':
            mol.xc = functional
            mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
        elif method.lower() == 'hf':
            mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
        else:
            raise ValueError("method must be 'dft' or 'hf'")
        return mf

    def create_charged_molecule_object(self, atom_input, basis_set, method='dft', 
                                     functional='m06-2x', original_charge=0, 
                                     original_spin=1, charge_change=-1):
        """
        Create charged form of molecule (anion or cation) and return the lowest energy one
        """
        new_charge = original_charge + charge_change
        
        # Determine possible spin states based on charge change
        if charge_change == -1:  
            if original_spin == 2:  
                possible_spins = [1, 3] 
            elif original_spin == 1:  
                possible_spins = [2]  
            elif original_spin == 3:  
                possible_spins = [2, 4] 
                
        elif charge_change == +1:  
            if original_spin == 1:  
                possible_spins = [2]  
            elif original_spin == 2:  
                possible_spins = [1, 3]  
            elif original_spin == 3:  
                possible_spins = [2, 4] 
        
        energies = []
        objects = []
        
        for spin in possible_spins:
            pyscf_spin = spin - 1
            mol = gto.Mole(basis=basis_set, charge=new_charge, spin=pyscf_spin)
            mol.atom = atom_input
            mol.build()
            if method.lower() == 'dft':
                mf = dft.UKS(mol) if pyscf_spin > 0 else dft.RKS(mol)
                mf.xc = functional
            elif method.lower() == 'hf':
                mf = scf.UHF(mol) if pyscf_spin > 0 else scf.RHF(mol)
            energy = mf.kernel()
            energies.append(energy)
            objects.append(mf)
        
        # Find the lowest energy
        min_idx = energies.index(min(energies))
        lowest_energy = energies[min_idx]
        optimal_spin = possible_spins[min_idx]
        best_object = objects[min_idx]
        
        return best_object, optimal_spin, lowest_energy

    def create_td_molecule_object(self, mf, nstates=5, triplet=False):
        """Create time-dependent DFT/HF object"""
        # Handle solvated molecules
        if hasattr(mf, 'with_solvent'):
            # For solvated molecules, use tdscf directly
            if hasattr(mf, 'xc'):  # DFT case
                td = tdscf.TDDFT(mf)
            else:  # HF case
                td = tdscf.TDHF(mf)
        else:
            # For non-solvated molecules, use the original method
            if hasattr(mf, 'TDDFT') and mf.TDDFT is not None:
                td = mf.TDDFT()
            elif hasattr(mf, 'TDHF') and mf.TDHF is not None:
                td = mf.TDHF()
            else:
                raise ValueError("Unsupported ground state object type")
        
        td.singlet = not triplet
        td.nstates = nstates
        return td

    def solvate_molecule(self, mf, solvent='water'):
        """Apply solvation to molecule"""
        solvent = solvent.lower()
        mf = mf.PCM()
        mf.with_solvent.eps = smd.solvent_db[solvent][5]
        mf.with_solvent.method = 'C-PCM'
        mf.with_solvent.lebedev_order = 29
        return mf

    def optimize_and_get_equilibrium(self, mf):
        """Optimize the geometry of the molecule and return the equilibrium geometry"""
        mol_eq = optimize(mf, conv_tol_grad=1e-7, conv_tol=1e-10)
        coords = mol_eq.atom_coords(unit='Ang')
        atoms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
        atom_list = [(atom, coord) for atom, coord in zip(atoms, coords)]
        return atom_list

    def smiles_to_xyz(self, smiles, filename=None):
        """Convert SMILES to XYZ file"""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
        
        if not filename:
            filename = f"mol_{abs(hash(smiles)) % 10000}.xyz"
        
        conf = mol.GetConformer()
        with open(filename, 'w') as f:
            f.write(f"{mol.GetNumAtoms()}\n{smiles}\n")
            for i, atom in enumerate(mol.GetAtoms()):
                pos = conf.GetAtomPosition(i)
                f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
        return filename

    def create_optimized_molecule(self, atom_input, basis_set, method='dft', 
                                functional='m06-2x', charge=0, spin=1, optimize=True):
        """Create molecule with optional geometry optimization"""
        if optimize:
            # Create initial molecule for optimization
            mf_initial = self.create_molecule_object(atom_input, basis_set, method, functional, charge, spin)
            mf_initial.kernel()
            
            # Optimize geometry using existing function
            opt_coords = self.optimize_and_get_equilibrium(mf_initial)
            
            # Create new molecule with optimized coordinates
            return self.create_molecule_object(opt_coords, basis_set, method, functional, charge, spin)
        else:
            return self.create_molecule_object(atom_input, basis_set, method, functional, charge, spin)

    def find_homo_lumo_and_gap(self, mf):
        """Find HOMO, LUMO, and gap energies"""
        homo = -float("inf")
        lumo = float("inf")
        for energy, occ in zip(mf.mo_energy, mf.mo_occ):
            if occ > 0 and energy > homo:
                homo = energy 
            if occ == 0 and energy < lumo:
                lumo = energy 
        return homo, lumo, lumo - homo

    def get_molecule_gfec(self, mf):
        """Calculate Gibbs free energy correction"""
        mf.kernel()
        hess = mf.Hessian().kernel()

        vib_data = thermo.harmonic_analysis(
            mf.mol,
            hess,
            imaginary_freq=False
        )

        thermo_data = thermo.thermo(
            mf,
            vib_data['freq_au'],
            temperature=298.15,
            pressure=101325
        )

        G_corr = thermo_data['G_tot'][0] - thermo_data['E0'][0]
        return vib_data, thermo_data, G_corr

    def calculate_all_properties(self, mf, anion_mf=None, cation_mf=None, 
                               td_obj=None, props_to_calc=None):
        """Calculate all requested molecular properties"""
        results = {}
        
        # Basic properties
        if 'gse' in props_to_calc:
            results['gse'] = mf.e_tot * self.HARTREE_TO_KCAL
        
        if any(p in props_to_calc for p in ['homo', 'lumo', 'gap']):
            homo, lumo, gap = [x * self.HARTREE_TO_EV for x in self.find_homo_lumo_and_gap(mf)]
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
        
        # Excited state properties
        if td_obj and any(p in props_to_calc for p in ['exe', 'osc']):
            state_idx = self.state_of_interest - 1
            if 'exe' in props_to_calc:
                results['exe'] = td_obj.e[state_idx] * self.HARTREE_TO_EV
            if 'osc' in props_to_calc:
                results['osc'] = td_obj.oscillator_strength()[state_idx]
        
        return results

    def create_mol2_files(self, tuning, molecule_name, tuning_names):
        """Create MOL2 files for tuning data"""
        for property_name in tuning_names:
            filename = f"{molecule_name}_{property_name}_tm.mol2"
            property_data = tuning[property_name]
            num_points = len(property_data)
            
            with open(filename, 'w') as f:
                # Write header
                f.write("@<TRIPOS>MOLECULE\n")
                f.write(f"{filename}\n")
                f.write(f"     {num_points} 0 0 0\n")
                f.write("SMALL\n")
                f.write("GASTEIGER\n")
                f.write("@<TRIPOS>ATOM\n")
                
                # Write atom records
                for i, (coord, delta_value) in enumerate(property_data, 1):
                    x, y, z = coord[0]  # Extract coordinates from (1,3) array
                    f.write(f"{i:>4}    H  {x:>8.4f}  {y:>8.4f}  {z:>8.4f}  H1  1  {property_name.upper()}  {delta_value:>10.6f}\n")

    def prepare_molecule(self):
        """Prepare the initial molecule structure"""
        # Handle input and create molecule
        if self.input_type == 'smiles':
            self.xyz_file = self.smiles_to_xyz(self.smiles_input)
            self.molecule_name = self.xyz_file.replace('.xyz', '')
        else:
            self.xyz_file = f'{self.molecule}.xyz'
            self.molecule_name = self.molecule

        self.molecule_object = self.create_molecule_object(
            self.xyz_file, self.basis_set, method=self.method, 
            functional=self.functional, charge=self.charge, spin=self.spin
        )

        if self.solvent:
            self.molecule_object = self.solvate_molecule(self.molecule_object, solvent=self.solvent)
            molecule_opt = self.optimize_and_get_equilibrium(self.molecule_object)
            self.molecule_object = self.create_molecule_object(
                molecule_opt, self.basis_set, method=self.method, 
                functional=self.functional, charge=self.charge, spin=self.spin
            )
            self.molecule_object = self.solvate_molecule(self.molecule_object, solvent=self.solvent)

        if self.optimize_geometry and self.solvent is None:
            self.molecule_object = self.create_optimized_molecule(
                self.molecule_object.mol.atom, self.basis_set, method=self.method, 
                functional=self.functional, charge=self.charge, spin=self.spin, 
                optimize=self.optimize_geometry
            )

        print(self.molecule_object.mol.atom)

        # Create XYZ file from molecule_object.mol.atom
        atom_data = self.molecule_object.mol.atom
        self.xyz_file = f"{self.molecule_name}_opt.xyz"
        with open(self.xyz_file, 'w') as f:
            f.write(f"{len(atom_data)}\n")
            f.write(f"\n")
            for element, coords in atom_data:
                f.write(f"{element} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
        print(f"Created XYZ file: {self.xyz_file}")

    def prepare_calculations(self, required_calculations):
        """Prepare all required calculation objects"""
        self.molecule_object.kernel()

        # Handle charged species with optional solvation
        if required_calculations.get('anion'):
            self.anion_mf = self.create_charged_molecule_object(
                self.xyz_file, self.basis_set, method=self.method, 
                functional=self.functional, original_charge=self.charge, 
                original_spin=self.spin, charge_change=-1
            )[0]
            if self.solvent:
                self.anion_mf = self.solvate_molecule(self.anion_mf, solvent=self.solvent)
            self.anion_mf.kernel()

        if required_calculations.get('cation'):
            self.cation_mf = self.create_charged_molecule_object(
                self.xyz_file, self.basis_set, method=self.method, 
                functional=self.functional, original_charge=self.charge, 
                original_spin=self.spin, charge_change=+1
            )[0]
            if self.solvent:
                self.cation_mf = self.solvate_molecule(self.cation_mf, solvent=self.solvent)
            self.cation_mf.kernel()

        # Handle excited state calculations with optional solvation
        if required_calculations.get('td'):
            self.td_object = self.create_td_molecule_object(
                self.molecule_object, triplet=self.triplet_excitation, 
                nstates=self.state_of_interest
            )
            if self.solvent:
                self.td_object.with_solvent.equilibrium_solvation = True
            self.td_object.kernel()

    def calculate_surface_coordinates(self):
        """Calculate van der Waals surface coordinates"""
        vdw_surface_operation = subprocess.run(
            ['vsg', self.xyz_file, '--txt'], capture_output=True, text=True
        )
        self.surface_coords = []
        surface_file = self.xyz_file.replace('.xyz', '_vdw_surface.txt')
        with open(surface_file, 'r') as f:
            for line in f:
                arr = np.fromstring(line.strip(), sep=' ').reshape(1, 3)
                self.surface_coords.append(arr)

    def calculate_tuning_properties(self, properties_to_calculate):
        """Calculate tuning properties for all surface coordinates"""
        self.tuning = {name: [] for name in properties_to_calculate}

        for coord in self.surface_coords:
            # Create perturbed molecules
            molecule_wsc = qmmm.mm_charge(self.molecule_object, coord, np.array([1.0]))
            molecule_wsc.kernel()
            
            anion_wsc = qmmm.mm_charge(self.anion_mf, coord, np.array([1.0])) if self.anion_mf else None
            if anion_wsc: anion_wsc.kernel()
            
            cation_wsc = qmmm.mm_charge(self.cation_mf, coord, np.array([1.0])) if self.cation_mf else None
            if cation_wsc: cation_wsc.kernel()
            
            td_wsc = None
            if self.td_object:
                td_wsc = self.create_td_molecule_object(
                    molecule_wsc, triplet=self.triplet_excitation, 
                    nstates=self.state_of_interest
                )
                if self.solvent:
                    td_wsc.with_solvent.equilibrium_solvation = True
                td_wsc.kernel()
            
            # Calculate perturbed properties and store deltas
            properties_wsc = self.calculate_all_properties(
                molecule_wsc, anion_wsc, cation_wsc, td_wsc, properties_to_calculate
            )
            
            for prop in properties_to_calculate:
                if prop in properties_wsc and prop in self.properties_alone:
                    delta = (properties_wsc[prop] - self.properties_alone[prop]) * self.PROPERTY_CONFIG[prop]['unit']
                    self.tuning[prop].append([coord, delta])

    def run_calculation(self, requested_properties=['gse', 'homo', 'lumo', 'gap']):
        """Main method to run the entire tuning calculation"""
        properties_to_calculate, required_calculations = self.setup_calculation(requested_properties)
        print(f"Properties: {properties_to_calculate}")
        print(f"Calculations: {required_calculations}")

        # Prepare molecule and calculations
        self.prepare_molecule()
        self.prepare_calculations(required_calculations)

        # Calculate reference properties
        self.properties_alone = self.calculate_all_properties(
            self.molecule_object, self.anion_mf, self.cation_mf, 
            self.td_object, properties_to_calculate
        )

        # Calculate surface coordinates and tuning properties
        self.calculate_surface_coordinates()
        self.calculate_tuning_properties(properties_to_calculate)

        # Create output files
        self.create_mol2_files(self.tuning, self.molecule_name, properties_to_calculate)
        print(f"Created {len(properties_to_calculate)} MOL2 files for {self.molecule_name}")

        return self.tuning, self.properties_alone


# Keep constants at module level for backward compatibility
HARTREE_TO_KCAL = 627.509
HARTREE_TO_EV = 27.2114
SHE_REFERENCE_POTENTIAL = 4.43

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


# Keep the original functions for backward compatibility
def setup_calculation(requested_props):
    """Setup properties and calculations needed"""
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

# Keep other original functions for backward compatibility
def calculate_all_properties(mf, anion_mf=None, cation_mf=None, td_obj=None, props_to_calc=None):
    results = {}
    
    # Basic properties
    if 'gse' in props_to_calc:
        results['gse'] = mf.e_tot * HARTREE_TO_KCAL
    
    if any(p in props_to_calc for p in ['homo', 'lumo', 'gap']):
        homo, lumo, gap = [x * HARTREE_TO_EV for x in find_homo_lumo_and_gap(mf)]
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
    
    # Excited state properties
    if td_obj and any(p in props_to_calc for p in ['exe', 'osc']):
        # Use a default state index for backward compatibility
        state_idx = 1  # Default to first excited state
        if 'exe' in props_to_calc:
            results['exe'] = td_obj.e[state_idx] * HARTREE_TO_EV
        if 'osc' in props_to_calc:
            results['osc'] = td_obj.oscillator_strength()[state_idx]
    
    return results

def create_molecule_object(atom_input, basis_set, method='dft', functional='m06-2x', charge=0, spin=1):
    """
    Create a PySCF molecule and KS/DFT object from either:
    - atom_input: list of (atom_symbol, (x,y,z)) tuples
    - atom_input: string path to XYZ file
    """
    spin -= 1
    mol = gto.M(basis=basis_set, charge=charge, spin=spin, symmetry=False)
    if isinstance(atom_input, str) and os.path.isfile(atom_input):
        mol.atom = atom_input 
    else:
        mol.atom = atom_input  
    mol.build()

    # Choose method and handle spin
    if method.lower() == 'dft':
        mol.xc = functional
        mf = dft.UKS(mol) if spin > 0 else dft.RKS(mol)
    elif method.lower() == 'hf':
        mf = scf.UHF(mol) if spin > 0 else scf.RHF(mol)
    else:
        raise ValueError("method must be 'dft' or 'hf'")
    return mf

def create_charged_molecule_object(atom_input, basis_set, method='dft', functional='m06-2x', original_charge=0, original_spin=1, charge_change=-1):
    """
    Create charged form of molecule (anion or cation) and return the lowest energy one with its spin and energy
    
    Args:
        charge_change: -1 for anion (add electron), +1 for cation (remove electron)
    """
    new_charge = original_charge + charge_change
    
    # Determine possible spin states based on charge change
    if charge_change == -1:  
        if original_spin == 2:  
            possible_spins = [1, 3] 
        elif original_spin == 1:  
            possible_spins = [2]  
        elif original_spin == 3:  
            possible_spins = [2, 4] 
            
    elif charge_change == +1:  
        if original_spin == 1:  
            possible_spins = [2]  
        elif original_spin == 2:  
            possible_spins = [1, 3]  
        elif original_spin == 3:  
            possible_spins = [2, 4] 
    
    energies = []
    objects = []
    
    for spin in possible_spins:
        pyscf_spin = spin - 1
        mol = gto.Mole(basis=basis_set, charge=new_charge, spin=pyscf_spin)
        mol.atom = atom_input
        mol.build()
        if method.lower() == 'dft':
            mf = dft.UKS(mol) if pyscf_spin > 0 else dft.RKS(mol)
            mf.xc = functional
        elif method.lower() == 'hf':
            mf = scf.UHF(mol) if pyscf_spin > 0 else scf.RHF(mol)
        energy = mf.kernel()
        energies.append(energy)
        objects.append(mf)
    
    # Find the lowest energy
    min_idx = energies.index(min(energies))
    lowest_energy = energies[min_idx]
    optimal_spin = possible_spins[min_idx]
    best_object = objects[min_idx]
    
    return best_object, optimal_spin, lowest_energy

def create_td_molecule_object(mf, nstates=5, triplet=False):
    # Handle solvated molecules
    if hasattr(mf, 'with_solvent'):
        # For solvated molecules, use tdscf directly
        if hasattr(mf, 'xc'):  # DFT case
            td = tdscf.TDDFT(mf)
        else:  # HF case
            td = tdscf.TDHF(mf)
    else:
        # For non-solvated molecules, use the original method
        if hasattr(mf, 'TDDFT') and mf.TDDFT is not None:
            td = mf.TDDFT()
        elif hasattr(mf, 'TDHF') and mf.TDHF is not None:
            td = mf.TDHF()
        else:
            raise ValueError("Unsupported ground state object type")
    
    td.singlet = not triplet
    td.nstates = nstates
    return td

#Manipulation and Extraction
def solvate_molecule(mf, solvent='water'):
    solvent = solvent.lower()
    mf = mf.PCM()
    mf.with_solvent.eps = smd.solvent_db[solvent][5]
    mf.with_solvent.method = 'C-PCM'
    mf.with_solvent.lebedev_order = 29
    return mf

def optimize_and_get_equilibrium(mf):
    """
    Optimize the geometry of the molecule and return the equilibrium geometry.
    """
    mol_eq = optimize(mf,conv_tol_grad=1e-7,conv_tol=1e-10)
    coords = mol_eq.atom_coords(unit='Ang')
    atoms = [mol_eq.atom_symbol(i) for i in range(mol_eq.natm)]
    atom_list = [(atom, coord) for atom, coord in zip(atoms, coords)]
    return atom_list

def smiles_to_xyz(smiles, filename=None):
    """Convert SMILES to XYZ file"""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    
    if not filename:
        filename = f"mol_{abs(hash(smiles)) % 10000}.xyz"
    
    conf = mol.GetConformer()
    with open(filename, 'w') as f:
        f.write(f"{mol.GetNumAtoms()}\n{smiles}\n")
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    return filename

def create_optimized_molecule(atom_input, basis_set, method='dft', functional='m06-2x', charge=0, spin=1, optimize=True):
    """Create molecule with optional geometry optimization"""
    if optimize:
        # Create initial molecule for optimization
        mf_initial = create_molecule_object(atom_input, basis_set, method, functional, charge, spin)
        mf_initial.kernel()
        
        # Optimize geometry using existing function
        opt_coords = optimize_and_get_equilibrium(mf_initial)
        
        # Create new molecule with optimized coordinates
        return create_molecule_object(opt_coords, basis_set, method, functional, charge, spin)
    else:
        return create_molecule_object(atom_input, basis_set, method, functional, charge, spin)


def find_homo_lumo_and_gap(mf):
    homo = -float("inf")
    lumo = float("inf")
    for energy, occ in zip(mf.mo_energy, mf.mo_occ):
        if occ > 0 and energy > homo:
            homo = energy 
        if occ == 0 and energy < lumo:
            lumo = energy 
    return homo, lumo, lumo - homo

def get_molecule_gfec(mf):
    mf.kernel()
    hess = mf.Hessian().kernel()

    vib_data = thermo.harmonic_analysis(
        mf.mol,
        hess,
        imaginary_freq=False
    )

    thermo_data = thermo.thermo(
        mf,
        vib_data['freq_au'],
        temperature=298.15,
        pressure=101325
    )

    G_corr = thermo_data['G_tot'][0] - thermo_data['E0'][0]
    return vib_data, thermo_data, G_corr

#File Manipulation
def create_mol2_files(tuning, molecule_name, tuning_names):
    for property_name in tuning_names:
        filename = f"{molecule_name}_{property_name}_tm.mol2"
        property_data = tuning[property_name]
        num_points = len(property_data)
        
        with open(filename, 'w') as f:
            # Write header
            f.write("@<TRIPOS>MOLECULE\n")
            f.write(f"{filename}\n")
            f.write(f"     {num_points} 0 0 0\n")
            f.write("SMALL\n")
            f.write("GASTEIGER\n")
            f.write("@<TRIPOS>ATOM\n")
            
            # Write atom records
            for i, (coord, delta_value) in enumerate(property_data, 1):
                x, y, z = coord[0]  # Extract coordinates from (1,3) array
                f.write(f"{i:>4}    H  {x:>8.4f}  {y:>8.4f}  {z:>8.4f}  H1  1  {property_name.upper()}  {delta_value:>10.6f}\n")

# Example usage
if __name__ == "__main__":
    # Initialize calculator
    calculator = TuningCalculator(
        molecule='water',
        method='dft',
        functional='b3lyp',
        basis_set='augccpvdz',
        charge=0,
        spin=1,
        optimize_geometry=True
    )
    
    # Run calculation
    tuning_results, reference_properties = calculator.run_calculation(
        requested_properties=['gse', 'homo', 'lumo', 'gap']
    )
    
    print("Reference properties:", reference_properties)
    print(f"Tuning calculation completed with {len(tuning_results)} properties")


# Original script execution (for backward compatibility)
# Uncomment below to use the original function-based approach
"""
#Parameters
molecule = 'water'
method = 'dft'
functional = 'b3lyp'
basis_set = 'augccpvdz'
charge = 0
spin = 1
gfec_functional = 'b3lyp'
gfec_basis_set = '6-31+G*'
state_of_interest = 2
triplet_excitation = False
solvent = None
rdx_solvent = 'acetonitrile'
input_type = 'xyz'  # 'xyz' or 'smiles'
smiles_input = 'O' 
optimize_geometry = True 

# User input and setup
requested_properties = ['gse', 'homo', 'lumo', 'gap']  # User specifies
# requested_properties = ['all']  # Or everything

properties_to_calculate, required_calculations = setup_calculation(requested_properties)
print(f"Properties: {properties_to_calculate}")
print(f"Calculations: {required_calculations}")

# Handle input and create molecule
if input_type == 'smiles':
    xyz_file = smiles_to_xyz(smiles_input)
    molecule_name = xyz_file.replace('.xyz', '')
else:
    xyz_file = f'{molecule}.xyz'
    molecule_name = molecule

molecule_object = create_molecule_object(xyz_file, basis_set, method=method, functional=functional, charge=charge, spin=spin)

if solvent:
    molecule_object = solvate_molecule(molecule_object, solvent=solvent)
    molecule_opt = optimize_and_get_equilibrium(molecule_object)
    molecule_object = create_molecule_object(molecule_opt, basis_set, method=method, functional=functional, charge=charge, spin=spin)
    molecule_object = solvate_molecule(molecule_object, solvent=solvent)

if optimize_geometry and solvent is None:
    molecule_object = create_optimized_molecule(molecule_object.mol.atom, basis_set, method=method, functional=functional, charge=charge, spin=spin, optimize=optimize_geometry)

print(molecule_object.mol.atom)

# Create XYZ file from molecule_object.mol.atom
atom_data = molecule_object.mol.atom
xyz_file = f"{molecule_name}_opt.xyz"
with open(xyz_file, 'w') as f:
    f.write(f"{len(atom_data)}\n")
    f.write(f"\n")
    for element, coords in atom_data:
        f.write(f"{element} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
print(f"Created XYZ file: {xyz_file}")

molecule_object.kernel()

# Handle charged species with optional solvation
anion_mf = None
if required_calculations.get('anion'):
    anion_mf = create_charged_molecule_object(xyz_file, basis_set, method=method, functional=functional, original_charge=charge, original_spin=spin, charge_change=-1)[0]
    if solvent:
        anion_mf = solvate_molecule(anion_mf, solvent=solvent)
    anion_mf.kernel()

cation_mf = None
if required_calculations.get('cation'):
    cation_mf = create_charged_molecule_object(xyz_file, basis_set, method=method, functional=functional, original_charge=charge, original_spin=spin, charge_change=+1)[0]
    if solvent:
        cation_mf = solvate_molecule(cation_mf, solvent=solvent)
    cation_mf.kernel()

# Handle excited state calculations with optional solvation
td_object = None
if required_calculations.get('td'):
    td_object = create_td_molecule_object(molecule_object, triplet=triplet_excitation, nstates=state_of_interest)
    if solvent:
        td_object.with_solvent.equilibrium_solvation = True
    td_object.kernel()

# Calculate reference properties
properties_alone = calculate_all_properties(molecule_object, anion_mf, cation_mf, td_object, properties_to_calculate)

# Surface calculations
vdw_surface_operation = subprocess.run(['vsg', xyz_file, '--txt'], capture_output=True, text=True)
surface_coords = []
surface_file = xyz_file.replace('.xyz', '_vdw_surface.txt')
with open(surface_file, 'r') as f:
    for line in f:
        arr = np.fromstring(line.strip(), sep=' ').reshape(1, 3)
        surface_coords.append(arr)

tuning = {name: [] for name in properties_to_calculate}

for coord in surface_coords:
    # Create perturbed molecules
    molecule_wsc = qmmm.mm_charge(molecule_object, coord, np.array([1.0]))
    molecule_wsc.kernel()
    
    anion_wsc = qmmm.mm_charge(anion_mf, coord, np.array([1.0])) if anion_mf else None
    if anion_wsc: anion_wsc.kernel()
    
    cation_wsc = qmmm.mm_charge(cation_mf, coord, np.array([1.0])) if cation_mf else None
    if cation_wsc: cation_wsc.kernel()
    
    td_wsc = None
    if td_object:
        td_wsc = create_td_molecule_object(molecule_wsc, triplet=triplet_excitation, nstates=state_of_interest)
        if solvent:
            td_wsc.with_solvent.equilibrium_solvation = True
        td_wsc.kernel()
    
    # Calculate perturbed properties and store deltas
    properties_wsc = calculate_all_properties(molecule_wsc, anion_wsc, cation_wsc, td_wsc, properties_to_calculate)
    
    for prop in properties_to_calculate:
        if prop in properties_wsc and prop in properties_alone:
            delta = (properties_wsc[prop] - properties_alone[prop]) * PROPERTY_CONFIG[prop]['unit']
            tuning[prop].append([coord, delta])

create_mol2_files(tuning, molecule_name, properties_to_calculate)
print(f"Created {len(properties_to_calculate)} MOL2 files for {molecule_name}")
"""


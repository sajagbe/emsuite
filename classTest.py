from emsuite.tuning import TuningCalculator

# Basic usage
calculator = TuningCalculator(
    input_type='smiles',
    smiles_input ='O',
    method='dft',
    functional='b3lyp',
    basis_set='augccpvdz',
    state_of_interest = 1,
    optimize_geometry=False
)

tuning_results, reference_properties = calculator.run_calculation(
        requested_properties=['all']
    )


#Parameters extended 
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
"""
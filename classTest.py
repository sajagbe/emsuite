from emsuite.tuning import TuningCalculator

# Basic usage
waterTuning = TuningCalculator(
    input_type='smiles',
    molecule ='O',
    method='dft',
    functional='b3lyp',
    basis_set='augccpvdz',
    state_of_interest = 1,
    optimize_geometry=True
)

waterTuning.run_calculation(['gse'])





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
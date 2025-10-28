import src.emsuite.core as core

# Test resurrecting molecule_alone.chk
print("=== Testing molecule_alone.chk ===")
mf_neutral = core.resurrect_mol('molecule_alone.chk')

# # Test resurrecting anion_alone.chk
# print("\n=== Testing anion_alone.chk ===")
# mf_anion = core.resurrect_mol('anion_alone.chk')

print("\n=== Summary ===")
print(f"Neutral molecule energy: {mf_neutral.e_tot}")
# print(f"Anion molecule energy: {mf_anion.e_tot}")






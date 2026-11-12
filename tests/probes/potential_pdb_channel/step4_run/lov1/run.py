"""Step 4 test: full potential run on AsLOV LOV1, new channel vs the old pdb2pqr channel."""
import time, shutil, numpy as np
from pathlib import Path
from emsuite.potential.spec import PotentialSpec
from emsuite.potential.runner import run_potential_spec, _apbs_values
from emsuite.inputs import PotentialInput
from emsuite.surface import load_surf
from emsuite.physconst import vdw_radius_angstrom
open('potential.in','w').write("pdb = 'complex.pdb'\nsurface_file = 'coupled.surf'\noutput_surf = 'new.surf'\nquantity = 'potential'\noff = ['CHR']\n")
t=time.time(); p=run_potential_spec(PotentialSpec.from_file('potential.in')); tn=time.time()-t
pts,new=load_surf('new.surf')
t=time.time(); PotentialInput(molecule='ligand.xyz',surface_file='coupled.surf',output_surf='old.surf',protein='complex.pdb',protein_format='pdb',ligand_resname='CHR',ligand_atoms='absent',forcefield='AMBER',ph=7.0,quantity='potential').run(); to=time.time()-t
_,old=load_surf('old.surf')
def stats(a,b,label):
    d=a-b; print(f'{label}: mean abs diff {abs(d).mean():.3f}, median {np.median(abs(d)):.3f}, max {abs(d).max():.3f} kT/e; correlation {np.corrcoef(a,b)[0,1]:.5f}')
print(f'RESULT points {len(new)}; new run {tn:.1f} s, old run {to:.1f} s')
print(f'RESULT new potential: mean {new.mean():.2f}, sd {new.std():.2f}, min {new.min():.2f}, max {new.max():.2f} kT/e; median |V| {np.median(abs(new)):.2f}')
stats(new,old,'RESULT new (off CHR) vs old (pdb2pqr, ligand absent)')
# control: same PQR parameters, ligand kept as an uncharged cavity (charge 0, vsg radii)
rows=[l.split() for l in open('new.pqr')]; xyz=np.array([[float(x) for x in r[6:9]] for r in rows])
with open('cavity.pqr','w') as f:
    for r in rows:
        if r[3]=='CHR':
            el=''.join(c for c in r[2] if c.isalpha())[0]; r[-1]=f'{vdw_radius_angstrom(el):.4f}'
        f.write(' '.join(r)+'\n')
cav=_apbs_values(pts,None,None,np.vstack([xyz,pts]),2.0,78.54,'potential',pqr_path=str(Path('cavity.pqr').resolve()))
np.savetxt('cavity_values.txt',cav)
stats(cav,new,'RESULT ligand as uncharged cavity (charge 0, vsg radii) vs ligand fully off (radius 0)')
print(f'RESULT cavity potential: mean {cav.mean():.2f}, sd {cav.std():.2f}, median |V| {np.median(abs(cav)):.2f}')
print('RESULT sources:'); import collections,csv
c=collections.Counter(r['source'].split(':')[0] for r in csv.DictReader(open('new_sources.csv'))); print('RESULT',dict(c))

"""Step 3 checks (stops, copies) and step 4 small end-to-end runs."""
import numpy as np, time, csv
from pathlib import Path
from emsuite.potential.spec import PotentialSpec
from emsuite.potential.table import build_parameter_table
from emsuite.potential.runner import run_potential_spec
from emsuite.potential.structure import load_structure
from emsuite.surface.generate import generate_surface
from emsuite.surface import load_surf
calls=[]
def counting(atoms,charge,spin,label):
    from emsuite.potential.resp import resp_charges_from_atoms
    calls.append(label); return resp_charges_from_atoms(atoms,charge=charge,spin=spin,label=label)
def stop(title,**kw):
    try: build_parameter_table(PotentialSpec(surface_file='x.surf',**kw),resp=counting); print('NOT STOPPED:',title)
    except Exception as e: print(f'STOP [{title}] -> {type(e).__name__}: {str(e)[:230]}')
print('--- stops before any HF run'); n0=len(calls)
stop('unknown residue without charge',pdb='mix.pdb')
stop('charge that gives odd electron count',pdb='mix.pdb',unknown_charge={'ACT':0})
stop('unknown residue bonded to protein (phosphoserine)',pdb='sep.pdb',unknown_charge={'SEP':-1})
stop('off entry that matches nothing',pdb='mix.pdb',unknown_charge={'ACT':-1},off=['XYZ'])
print('HF runs started during stops:',len(calls)-n0)
# strip hydrogens from ALA part to test missing-H message and add_hydrogens
lines=[l for l in open('mix.pdb') if not (l.startswith(('ATOM','HETATM')) and l[76:78].strip()=='H' and l[17:20]!='ACT')]
open('mix_noH.pdb','w').writelines(lines)
stop('standard residues without hydrogens',pdb='mix_noH.pdb',unknown_charge={'ACT':-1})
t=build_parameter_table(PotentialSpec(pdb='mix_noH.pdb',surface_file='x.surf',unknown_charge={'ACT':-1},add_hydrogens=True,off=['ACT']))
full=load_structure('mix.pdb'); print('add_hydrogens: added',t.structure.hydrogens_added,'-> atoms',t.structure.n_atoms,'(original file has',full.n_atoms,'); net charge of ALA part %.4f'%t.charges.sum())
print('--- copies: two acetates')
for mode in ('each','one'):
    calls.clear(); t=build_parameter_table(PotentialSpec(pdb='two.pdb',surface_file='x.surf',unknown_charge={'ACT':-1,'ALA':0},resp_copies=mode),resp=counting)
    s=t.structure; res=[r for r in sorted(t.residue_source) if 'RESP' in t.residue_source[r] and s.res_names[s.residue_atoms(r)[0]]=='ACT']
    q=[t.charges[s.residue_atoms(r)] for r in res]
    print(mode,': HF runs',len(calls),'| max charge difference between the two copies %.4f e'%np.abs(q[0]-q[1]).max(),'|',[t.residue_source[r] for r in res])
print('--- off switches on the ALA + acetate case')
for off in ([],['ACT'],['ACT@O3'],['ALA']):
    calls.clear(); t=build_parameter_table(PotentialSpec(pdb='mix.pdb',surface_file='x.surf',unknown_charge={'ACT':-1},off=off),resp=counting)
    print('off =',off,'| HF runs',len(calls),'| atoms off',int(t.off.sum()),'| total charge %.4f'%t.charges.sum(),'| zero-radius atoms',int((t.radii==0).sum()))
print('--- end-to-end run (APBS) on the ALA + acetate case')
s=load_structure('mix.pdb'); act=[i for i in range(s.n_atoms) if s.res_names[i]=='ACT']
with open('act.xyz','w') as f:
    f.write(f'{len(act)}\nacetate\n'); [f.write(f'{s.elements[i]} {s.coords[i][0]:.4f} {s.coords[i][1]:.4f} {s.coords[i][2]:.4f}\n') for i in act]
generate_surface(input_type='XYZ',input_data='act.xyz',output_surf='act.surf',surface_density=1.0,surface_scale=1.0,surface_type='homogenous',surface_charge=0.0,optimize=False)
vals={}
for q in ('potential','charge'):
    t0=time.time(); p=run_potential_spec(PotentialSpec(pdb='mix.pdb',surface_file='act.surf',output_surf=f'mix_{q}.surf',quantity=q,unknown_charge={'ACT':-1},off=['ACT']))
    c,v=load_surf(p['surf']); vals[q]=v; print(q,'points',len(v),'range %.4f .. %.4f'%(v.min(),v.max()),'time %.1f s'%(time.time()-t0),'files:',sorted(Path(x).name for x in p.values()),'all exist:',all(Path(x).is_file() for x in p.values()))
print(open('mix_potential_sources.csv').read()); print(''.join(open('mix_potential.pqr').readlines()[:3]+open('mix_potential.pqr').readlines()[-2:])); print(''.join(open('mix_potential.mol2').readlines()[:9]))

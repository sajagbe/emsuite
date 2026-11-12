import numpy as np, collections
from pathlib import Path
from emsuite.potential.apbs import run_apbs_grids
from emsuite.potential.gauss import potential_at_points
from emsuite.surface import load_surf
pts,new=load_surf('new.surf'); _,old=load_surf('old.surf')
def read(p):
    rows=[l.split() for l in open(p) if l.startswith(('ATOM','HETATM'))]
    return rows
newr=read('new.pqr'); oldr=read('../../step2/pdb2pqr_ph7.0.pqr')
def key(r): return (int(r[-6]),r[2])
def xyzqr(r): return np.array([float(x) for x in r[-5:]])
N={key(r):xyzqr(r) for r in newr if r[3]!='CHR'}; O={key(r):xyzqr(r) for r in oldr}
common=[k for k in N if k in O]
d=np.array([np.linalg.norm(N[k][:3]-O[k][:3]) for k in common]); isH=np.array([k[1][0]=='H' for k in common])
print(f'RESULT name-matched atoms {len(common)}: heavy atoms moved by pdb2pqr: max {d[~isH].max():.3f} A; hydrogens: {int((d[isH]>0.05).sum())} of {int(isH.sum())} moved > 0.05 A, max {d[isH].max():.2f} A')
# per-residue multiset check of (charge, radius) over all atoms
def per_res(rows):
    m=collections.defaultdict(list)
    for r in rows:
        if r[3]!='CHR': m[int(r[-6])].append((round(float(r[-2]),4),round(float(r[-1]),3)))
    return {k:sorted(v) for k,v in m.items()}
a=per_res(newr); b=per_res(oldr); print('RESULT residues whose set of (charge, radius) values is identical in both files:',sum(a[k]==b.get(k) for k in a),'of',len(a), '| differing:',[k for k in a if a[k]!=b.get(k)][:10])
xyz=np.array([xyzqr(r)[:3] for r in newr]); box=np.vstack([xyz,pts])
def run(pqr,dime,box=box): return potential_at_points(run_apbs_grids(pqr_path=str(Path(pqr).resolve()),box_coords=box,dime=(dime,)*3).potential,pts)
def st(a,b,l): d=abs(a-b); print(f'RESULT {l}: mean abs {d.mean():.3f}, median {np.median(d):.3f}, max {d.max():.3f} kT/e, corr {np.corrcoef(a,b)[0,1]:.4f}')
old_same=run('../../step2/pdb2pqr_ph7.0.pqr',65)
st(old_same,new,'old PQR vs new PQR, same box and grid (65)')
res={}
for dime in (65,97,129,161,193):
    res[dime]=run('new.pqr',dime)
for dime in (65,97,129,161): st(res[dime],res[193],f'new PQR grid {dime} vs grid 193')
o193=run('../../step2/pdb2pqr_ph7.0.pqr',193); st(o193,res[193],'old PQR vs new PQR, both at grid 193')
c65=np.loadtxt('cavity_values.txt'); c193=run('cavity.pqr',193); st(c65,c193,'cavity PQR grid 65 vs 193'); st(c193,res[193],'cavity vs fully off, both at grid 193')
print('RESULT box length %.1f A; fine spacing at 65: %.2f A, at 193: %.2f A'%((box.max(0)-box.min(0)).max()+12, 0.8*((box.max(0)-box.min(0)).max()+12)/64, 0.8*((box.max(0)-box.min(0)).max()+12)/192))
np.savez('diag.npz',**{f'new{k}':v for k,v in res.items()},old65=old_same,old193=o193,cav193=c193)

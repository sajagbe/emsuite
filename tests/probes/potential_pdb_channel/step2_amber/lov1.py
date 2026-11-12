"""Step 2 test: Amber assignment on the AsLOV LOV1 protein, against pdb2pqr (independent program)."""
import sys, time, collections, numpy as np
from pathlib import Path
from emsuite.potential.structure import load_structure
from emsuite.potential.amber import assign_amber
from emsuite.potential.spec import DEFAULT_FORCEFIELD
from emsuite.potential.pdb_select import strip_residue
from emsuite.potential.pdb2pqr_runner import run_pdb2pqr
S=Path(sys.argv[1]); P='examples/families/AsLOV/LOV1/complex.pdb'
t=time.time(); s=load_structure(P); a=assign_amber(s,DEFAULT_FORCEFIELD); print('assignment time %.1f s'%(time.time()-t))
print('atoms',s.n_atoms,'residues',len(set(s.res_index)),'unmatched',[s.residue_label(r) for r in a.unmatched])
k=~np.isnan(a.charges); print('protein net charge %.4f'%a.charges[k].sum(), 'zero radii',int((a.radii[k]==0).sum()))
print('templates differing from PDB name:',collections.Counter((s.res_names[s.residue_atoms(r)[0]],n) for r,n in a.templates.items() if n!=s.res_names[s.residue_atoms(r)[0]]))
strip_residue(P,'CHR',None,None,S/'stripped.pdb')
for ph in (None,7.0):
    out=S/f'pdb2pqr_ph{ph}.pqr'; run_pdb2pqr(S/'stripped.pdb',out,forcefield='AMBER',ph=ph)
    ref={}
    for l in open(out):
        f=l.split()
        if f[0] in('ATOM','HETATM'): ref[(int(f[4]) if len(f)==10 else int(f[5]),f[2])]=(f[3],float(f[-2]),float(f[-1]))
    # pdb2pqr renumbers? keep residue number as given
    n=m=0; dq=[]; dr=[]; miss=collections.Counter(); badres=collections.Counter()
    for i in np.flatnonzero(k):
        key=(s.res_seqs[i],s.atom_names[i]); n+=1
        if key not in ref: miss[(s.res_names[i],s.atom_names[i])]+=1; continue
        m+=1; dq.append(abs(a.charges[i]-ref[key][1])); dr.append(abs(a.radii[i]-ref[key][2]))
        if dq[-1]>1e-3 or dr[-1]>1e-3: badres[(s.res_names[i],s.res_seqs[i],ref[key][0])]+=1
    dq=np.array(dq); dr=np.array(dr)
    print(f'pdb2pqr ph={ph}: pqr atoms {len(ref)} net {sum(v[1] for v in ref.values()):.3f}; our atoms {n}, same residue number+atom name {m}; charge equal(<1e-3) {int((dq<1e-3).sum())}, radius equal {int((dr<1e-3).sum())}; max dq {dq.max():.3f} max dr {dr.max():.3f}')
    print('  residues with differences:',dict(badres)); print('  names not found in pdb2pqr file:',dict(miss))

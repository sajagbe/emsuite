"""Step 3 test: production unknown-residue path on small parked fixtures vs the benchmark RESP charges."""
import csv, json, sys, time
from pathlib import Path
import numpy as np
from emsuite.potential.structure import load_structure
from emsuite.potential.amber import assign_amber
from emsuite.potential.unknown import assign_unknown
from emsuite.potential.spec import DEFAULT_FORCEFIELD
R=Path('/data/PHO_WORK/sajagbe2/packages/emsuite/tests/probes/ddec_radii')
cases=sys.argv[1:]
w=csv.writer(sys.stdout); w.writerow('case atoms charge seconds sum_charge mean_abs_diff_e max_abs_diff_e radii'.split())
for c in cases:
    d=R/'fixtures'/c; meta=json.loads((d/'case.json').read_text())
    s=load_structure(d/'as_built/geometry.pdb'); a=assign_amber(s,DEFAULT_FORCEFIELD)
    names={s.res_names[s.residue_atoms(r)[0]] for r in a.unmatched}; assert len(a.unmatched)==1 and len(set(s.res_index))==1,(c,names)
    t=time.time(); u=assign_unknown(s,a.unmatched,{n:int(meta['charge']) for n in names},{}); dt=time.time()-t
    q=u.charges[a.unmatched[0]]
    ref=None
    for rnd in ('round1','round2'):
        f=R/rnd/c/'fits/as_built/charges.csv'
        if f.exists(): ref=np.array([float(r['resp_charge_e']) for r in csv.DictReader(open(f))])
    order=list(range(s.n_atoms)); m=d/'as_built/pdb_atom_map.csv'
    if m.exists(): order=[int(r['chemical_index'])-1 for r in sorted(csv.DictReader(open(m)),key=lambda r:int(r['pdb_serial']))]
    diff=np.abs(q-ref[order]) if ref is not None and len(ref)==len(q) else np.array([np.nan])
    w.writerow([c,s.n_atoms,meta['charge'],f'{dt:.1f}',f'{q.sum():+.5f}',f'{diff.mean():.4f}',f'{diff.max():.4f}',' '.join(f'{e}={r:.2f}' for e,r in sorted(set(zip(s.elements,u.radii[a.unmatched[0]]))))]); sys.stdout.flush()

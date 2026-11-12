"""Step 2 test: Amber assignment on every parked fixture that has reference values."""
import csv, sys
from pathlib import Path
import numpy as np
from emsuite.potential.structure import load_structure
from emsuite.potential.amber import assign_amber
from emsuite.potential.spec import DEFAULT_FORCEFIELD
R=Path('/data/PHO_WORK/sajagbe2/packages/emsuite/tests/probes/ddec_radii')
L21=tuple('amber19/lipid21.xml' if f.endswith('lipid17.xml') else f for f in DEFAULT_FORCEFIELD)
rows=[]
for d in sorted(list((R/'cases').iterdir())+list((R/'fixtures').iterdir())):
    pdb=d/'as_built'/'geometry.pdb'; ref=d/'atoms.csv'
    if not pdb.exists() or not ref.exists(): rows.append([d.name,'no pdb or reference file']+['']*8); continue
    ra=list(csv.DictReader(open(ref)))
    ff=L21 if 'lipid21' in (d/'case.json').read_text() else DEFAULT_FORCEFIELD
    try:
        s=load_structure(pdb); a=assign_amber(s,ff)
    except Exception as e:
        rows.append([d.name,'stopped',str(e).replace('\n',' ')[:160]]+['']*7); continue
    if len(ra)!=s.n_atoms: rows.append([d.name,'atom count differs']+['']*8); continue
    order=list(range(s.n_atoms)); m=d/'as_built'/'pdb_atom_map.csv'
    if m.exists():
        mm=list(csv.DictReader(open(m))); order=[int(r['chemical_index'])-1 for r in sorted(mm,key=lambda r:int(r['pdb_serial']))]
    cmp=[(i,order[i]) for i in range(s.n_atoms) if not np.isnan(a.charges[i]) and ra[order[i]].get('amber_charge_e') not in (None,'','nan','None')]
    elem_ok=all(s.elements[i]==ra[order[i]]['element'] for i in range(s.n_atoms))
    dq=max((abs(a.charges[i]-float(ra[j]['amber_charge_e'])) for i,j in cmp),default=float('nan'))
    dr=max((abs(a.radii[i]-float(ra[j]['amber_rmin_half_A'])) for i,j in cmp),default=float('nan'))
    un=sorted({s.res_names[s.residue_atoms(r)[0]] for r in a.unmatched})
    rows.append([d.name,'ok','','lipid21' if ff is L21 else 'default',len(set(s.res_index)),len(a.unmatched),' '.join(un),len(cmp),f'{dq:.1e}',f'{dr:.1e}',elem_ok,' '.join(sorted(set(a.templates.values())))[:120]][:12])
w=csv.writer(sys.stdout); w.writerow('case status message forcefield residues unmatched_residues unmatched_names compared_atoms max_charge_diff_e max_radius_diff_A elements_agree templates'.split()); w.writerows(rows)

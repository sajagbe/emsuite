"""The two optional settings on AsLOV LOV1: grid_spacing and uncharged."""
import time, numpy as np
from emsuite.potential.spec import PotentialSpec
from emsuite.potential.runner import run_potential_spec
from emsuite.surface import load_surf
out={}
for name,kw in (('off_default_grid',dict(off=['CHR'])),('off_0.3',dict(off=['CHR'],grid_spacing=0.3)),('uncharged_0.3',dict(uncharged=['CHR'],grid_spacing=0.3)),('off_0.25',dict(off=['CHR'],grid_spacing=0.25))):
    t=time.time(); p=run_potential_spec(PotentialSpec(pdb='complex.pdb',surface_file='coupled.surf',output_surf=f'{name}.surf',**kw)); out[name]=load_surf(p['surf'])[1]; print(f'RESULT {name}: {time.time()-t:.1f} s, median |V| {np.median(abs(out[name])):.2f} kT/e')
def st(a,b): d=abs(out[a]-out[b]); print(f'RESULT {a} vs {b}: mean abs {d.mean():.3f}, median {np.median(d):.3f}, max {d.max():.2f} kT/e')
st('off_default_grid','off_0.25'); st('off_0.3','off_0.25'); st('uncharged_0.3','off_0.3')


# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from gridlib140 import *
def run(args):
 i,tri,x,y=args;o=OUT/'pen140';k='h'+str(i)
 with open('/tmp/pen140_'+k,'w') as f:subprocess.run([sys.executable,'long14.py','hover',k],env=dict(os.environ,TRI=str(tri),AX=str(x),AY=str(y)),stdout=f,stderr=f,check=True)
 a=np.load(o/(k+'.npz'))['a'];print(k,tri,x,y,'ink',round(a[:,9].max(),4),'held',round((a[:,10:12].min(1)>.01).mean(),3),'end',round(np.linalg.norm(a[-1,4:6]-a[-1,2:4])*1000,2),'gap',round((a[:,6].min()-np.loadtxt(o/(k+'_origin.txt'))[3]-.001)*1000,2),flush=True)
with ThreadPoolExecutor(8) as ex:list(ex.map(run,[(i,*v) for i,v in enumerate(itertools.product([0,1],[-.4,0,.18,.7,1.3],[-.4,.15]))]))

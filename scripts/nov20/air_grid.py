
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from concurrent.futures import ThreadPoolExecutor
from evaluate_long import *
import sys,subprocess,os
def run(x):
 k='air'+str(x);o=OUT/'pen140'
 with open('/tmp/pen140_'+k,'w') as log:subprocess.run([sys.executable,'air_long.py','hover',k],env=dict(os.environ,AX='.18',AY='-.4',DP=str(x)),stdout=log,stderr=log,check=True)
 a=np.load(o/(k+'.npz'))['a'];print(k,a[:,9].max(),a[a[:,1]>0,10:12].min(0),np.linalg.norm(a[-1,4:6]-a[-1,2:4]),a[:,6].min()-np.loadtxt(o/(k+'_origin.txt'))[3]-.001,flush=True)
with ThreadPoolExecutor(5) as ex:list(ex.map(run,[.45,.9,1.5,2.1,3,4]))

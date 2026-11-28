
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from concurrent.futures import ThreadPoolExecutor
from evaluate_long import *
import itertools,sys,subprocess,os
def run(item):
 i,(x,y,p,g,f)=item;k='v'+str(i)
 with open('/tmp/pen140_'+k,'w') as log:subprocess.run([sys.executable,'servo_long.py','write',k],env=dict(os.environ,AX=str(x),AY=str(y),PINCH=str(p),GAIN=str(g),FORCE=str(f),TRI='0'),stdout=log,stderr=log,check=True)
 print(k,x,y,p,g,f,stats(OUT/'pen140',k),flush=True)
with ThreadPoolExecutor(5) as ex:list(ex.map(run,enumerate(itertools.product([0],[-.2,-.12],[1.2,1.6,2.1],[.0003],[.035,.075]))))


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
 i,x,y=item;k='t'+str(i)
 with open('/tmp/pen140_'+k,'w') as f:subprocess.run([sys.executable,'long14.py','write',k],env=dict(os.environ,AX=str(x),AY=str(y)),stdout=f,stderr=f,check=True)
 print(k,x,y,stats(OUT/'pen140',k),flush=True)
with ThreadPoolExecutor(5) as ex:list(ex.map(run,[(i,x,y) for i,(x,y) in enumerate(itertools.product([-.2,0,.18,.42,.7],[-.4,-.15]))]))


# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import OUT
from pathlib import Path
src=(Path(__file__).with_name('stack09.py')).read_text().replace('u,d=s.body_q.numpy()',"""if t==440:
   np.savez(OUT/'pen14/warm.npz',q=s.joint_q.numpy(),act=act);return
  u,d=s.body_q.numpy()""");exec(compile(src.replace("run(sys.argv[1])","run('custom')"),'stack09.py','exec'))

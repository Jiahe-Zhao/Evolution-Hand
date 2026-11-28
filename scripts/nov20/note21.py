
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from audit21 import *
import csv
o=OUT/'pour21';d=evaluate('final');assert d['ok'];a=d['a'];k=np.flatnonzero(a[:,2]>=2)[0];idle=np.load(o/'idle.npz')['a'];assert idle[:,2].max()==0
grip=list(csv.DictReader(open(o/'final_grip.csv')));assert len(set(int(g['step']) for g in grip))==len(a)
text=f"""控制使用仿真内的过沿粒数反馈（不假设真实相机能无延时数粒）：达到 2 粒即发起回转。首次超过阈值的采样步已有 {int(a[k,2])} 粒越沿（{a[k,0]:.3f} s），后续仍有 {int(a[-1,2]-a[k,2])} 粒越沿。因此提前收腕比见到第 10 粒再收腕更合适，此提前量仅对应本次装料和物性。另跑只抬腕不倾倒对照，{idle[-1,0]:.2f} s 内出料 0 粒。逐步握柄载荷覆盖全部物理步。碗面接触启用滚动阻力；固定材料与装料只验证本次结果。原始接触记录：final_rim.csv、final_grip.csv。"""
for path in [o/'RESULTS.md',OUT.parent.parent/'docs/SCENES_20_20261116.md']:
 st=path.read_text()
 if text not in st:path.write_text(st.rstrip()+'\n\n'+text+'\n')


# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt,csv
from build import OUT
o=OUT/'keys17';a=np.loadtxt(o/'hand.csv',delimiter=',',skiprows=1);p=np.loadtxt(o/'probe.csv',delimiter=',',skiprows=1);cs=list(csv.DictReader(open(o/'hand_contacts.csv')));fig,axs=plt.subplots(2,1,figsize=(9,5),sharex=True);lines=[]
for ax,arr,title in zip(axs,[p,a],['Isolated axial load (0.72 N)','hand_111: four sequential finger presses']):
 ax.plot(arr[:,0],arr[:,1:]);ax.axhline(2,color='.4',ls='--');ax.axhline(.5,color='.4',ls=':');ax.set(ylabel='Travel (mm)',title=title);ax.legend(['Index','Middle','Ring','Little'],ncol=4,fontsize=8)
axs[-1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(o/'travel.png',dpi=150)
for j in range(4):
 q=a[:,j+1];hits=[];ups=[];active=False
 for t,d in zip(a[:,0],q):
  if not active and d>=2:hits.append(t);active=True
  elif active and d<=.5:ups.append(t);active=False
 assert len(hits)==len(ups)==1 and (j==0 or hits[0]>prev);prev=ups[0]
 cc=[x for x in cs if int(x['key'])==j+1];assert all(f'/link_{j+2}_' in x['contact_geom'] and 120+480*j<=round(float(x['time_s'])*480)<600+480*j for x in cc)
 assert any(abs(float(x['time_s'])-hits[0])<.001 and f'/link_{j+2}_3/' in x['contact_geom'] for x in cc)
 lines.append(f"|{['食指','中指','无名指','小指'][j]} / K{j+1}|{hits[0]:.3f}|{ups[0]:.3f}|{q.max():.2f}|")
doc='''## 17 多指按键：固定初态通过（2026-11-16）

四个独立滑动键，限程 4 mm，回弹刚度 210 N/m，阻尼 0.8 N·s/m；键帽 18 × 15 mm，横向节距 19 mm，净隙 4 mm，纵向按指长错列。按下阈值 2 mm，释放阈值 0.5 mm（均相对滑轨零点）。单键依次直压 0.72 N，各键峰值 3.65 mm，未加载键保持自重静挠度约 0.19 mm；卸载回弹，无邻键触发。

Newton SolverMuJoCo CPU，480 Hz，hand_111 左手。依次控制四指屈曲；键上只保留重力、阻尼和随实测位移的被动回弹。逐步核对载荷 >1 mN 的接触：每次越过触发线时均有指定指末节接触，整段没有错指触键；无名指和小指另有同指相邻指节接触。各键各按下、释放一次。

|手指 / 键|触发 s|释放 s|峰值 mm|
|---|---:|---:|---:|
'''+ '\n'.join(lines)+'''

证据：outputs/nov20/keys17/{operation.mp4,travel.png,hand.csv,hand_contacts.csv,probe.csv,keys.xml}。接触 CSV 每行是一条接触点的法向力（N），不是整指合力。脚本 scripts/nov20/keys17.py（probe / hand video），report17.py。此为固定初态验证。
'''
(o/'RESULTS.md').write_text(doc);f=OUT.parent.parent/'docs/SCENES_20_20261116.md';old=f.read_text();header=doc.splitlines()[0];f.write_text(old.split('\n'+header)[0].rstrip()+'\n\n'+doc);print(lines)

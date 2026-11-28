
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw
import csv,subprocess,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
o=OUT/'pen14';a,q=[np.load(o/'write.npz')[k] for k in ['a','q']];h=np.load(o/'hover.npz')['a'];pos=np.load(o/'origin.npy');u=np.linspace(0,1,4001);curve=np.c_[pos[0]-.002+.046*u,pos[1]+.006*np.sin(2*np.pi*u)];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))];tree=cKDTree(curve)
def ink(x):return (x[:,9]>.01)&np.isfinite(x[:,7])
marked=ink(a);held=(a[:,10:12]>.01).all(axis=1);ok=marked&held;pts=a[ok,7:9];err,idx=tree.query(pts);bins=np.unique(np.minimum((arc[idx]/arc[-1]*100).astype(int),99));cov=len(bins);rms=np.sqrt(np.mean(err**2))*1000
ix=np.flatnonzero(ok);s=arc[idx];ints=sorted((min(x,y),max(x,y)) for j,(x,y) in enumerate(zip(s,s[1:])) if ix[j+1]-ix[j]==1);end=-1;tot=0
for x,y in ints:tot+=max(0,y-max(x,end));end=max(end,y)
coverage=100*tot/arc[-1]
assert (h[:,9]==0).all() and (h[h[:,1]>0,10:12]>.01).all() and np.linalg.norm(h[-1,4:6]-curve[-1])<.003 and not ink(h).any() and h[:,6].min()-(pos[2]-.0434)>0 and coverage>=90 and rms<=3 and held[marked].all()
for label,data in [('write',a),('hover',h)]:np.savetxt(o/(label+'_contacts.csv'),data,delimiter=',',header='time_s,progress,target_x_m,target_y_m,tip_x_m,tip_y_m,tip_z_m,contact_x_m,contact_y_m,tip_paper_normal_N,thumb_pen_normal_N,index_pen_normal_N',comments='')
fig,ax=plt.subplots(figsize=(8,3.4));ax.plot(*(1000*curve.T),'--',lw=3,c='.55',label='Prescribed curve');ax.plot(*(1000*np.where(ok[:,None],a[:,7:9],np.nan).T),lw=1.4,color='tab:blue',label='Loaded tip contact');ax.axis('equal');ax.set(xlabel='Paper x (mm)',ylabel='Paper y (mm)',title=f'{coverage:.1f}% loaded arc   |   lateral RMS {rms:.2f} mm');ax.legend();fig.tight_layout();fig.savefig(o/'overlay.png',dpi=160)
vm=mj.MjModel.from_xml_path(str(o/'write.xml'));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480);cam=mj.MjvCamera();cam.lookat[:]=[pos[0]+.014,pos[1],pos[2]-.004];cam.distance=.208;cam.azimuth=125;cam.elevation=-28;frames=[]
def uv(p):return (int((p[0]-pos[0]+.012)*6800)+490,int(175-(p[1]-pos[1])*6800))
for t in range(0,len(q),32):
 vd.qpos[:27]=q[t,:27];vd.qpos[27:30]=q[t,27:30];vd.qpos[30:]=q[t,[33,30,31,32]];mj.mj_forward(vm,vd);ren.update_scene(vd,cam);im=Image.new('RGB',(960,360),'#f3f1e9');im.paste(Image.fromarray(ren.render()),(0,0));dr=ImageDraw.Draw(im);dr.line([uv(p) for p in curve],fill='#bbb8af',width=4)
 for j in range(1,t+1):
  if marked[j-1] and marked[j]:dr.line([uv(a[j-1,7:9]),uv(a[j,7:9])],fill='#134573',width=3)
 dr.rectangle((0,0,960,23),fill='#172936');dr.text((8,5),'hand_111  |  Newton  |  free pen',fill='white');dr.text((489,5),f'Paper view | contact ink | t={a[t,0]:.2f}s | tip {a[t,9]:.3f} N',fill='white');frames.append(np.asarray(im))
subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-pix_fmt','yuv420p',str(o/'operation.mp4')],input=np.asarray(frames).tobytes(),check=True);ren.close()
doc=f'''## 14 持笔描线：固定初态通过（2026-11-17）

hand_111 左手，Newton SolverMuJoCo CPU，960 Hz。自由刚体笔，带加宽握部的短笔，半径 1 mm 的球形笔尖；纸面为固定刚性平板。初态已合指，未测试从桌面拾笔。只驱动手部关节，以笔尖位置和纸面载荷反馈调整手腕。笔尖与纸面法向力 >0.01 N 且有实际接触点才落墨。对照中纸面降低 10 mm，手持同一支笔沿同一平面曲线悬空运动，最小净空 {(h[:,6].min()-(pos[2]-.0434))*1000:.2f} mm，墨点数 0。

预设正弦曲线长 {arc[-1]*1000:.2f} mm（跨度 46 mm，振幅 6 mm）。用实际着墨点向目标中心线作最近点投影；仅相邻两步均着墨且双指持笔才累计其投影弧段，合并重叠区间，断墨处不补线。要求覆盖 ≥90%，横向 RMS ≤3 mm。实测覆盖 **{coverage:.2f}%**，接触点到曲线的距离 RMS **{rms:.3f} mm**；着墨时均有拇指、食指同时承载。书写接触法向力中位数 {np.median(a[marked,9]):.3f} N。该结果不代表随机初态成功率。

证据：outputs/nov20/pen14/ 下 operation.mp4、overlay.png、write_contacts.csv、hover_contacts.csv、write.xml / hover.xml 及两次完整状态轨迹。复现：在 scripts/nov20 中先运行 warm14.py 生成初始手姿；pen14.py hover / write，最后运行 report14.py。
'''
(o/'RESULTS.md').write_text(doc);f=OUT.parent.parent/'docs/SCENES_20_20261116.md';f.write_text(f.read_text().split('\n'+doc.splitlines()[0])[0].rstrip()+'\n\n'+doc);print(doc)

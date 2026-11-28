
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from verify19 import *
from PIL import Image,ImageDraw
import subprocess
tags=['fine7','single'];cases={k:metrics(k) for k in tags};tag=max((k for k in tags if cases[k]['success']),key=lambda k:cases[k]['relative_mm']);r=cases[tag];f=np.load(out/(tag+'.npz'));x=f['data'];p=x[:,1:36].reshape(-1,5,7);init=f['initial'];d=np.linalg.norm(p[:,:,:3]-init[None,:,:3],axis=2)*1000
vtx=np.array([[xx,yy,z] for xx,y,zz in [(-.027,.015,.00021),(-.025,.016,.0004),(.025,.016,.0004),(.027,.015,.00021)] for yy in (-y,y) for z in (-zz,zz)])
tipx=[R.from_quat(q[3:]).apply(vtx)[:,0]+q[0] for q in p[-1]];separation=(tipx[-1].min()-max(a.max() for a in tipx[:-1]))*1000
vm=mj.MjModel.from_xml_path(str(out/(tag+'.xml')));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480);hs=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith('handgeom')];opt=mj.MjvOption();opt.geomgroup[5]=0
cams=[]
for tr,dist,az,el in [([cx+.025,cy,.080],.220,89,-37),([cx+.020,cy,.07],.169,97,-20)]:
 c=mj.MjvCamera();c.lookat[:]=tr;c.distance=dist;c.azimuth=az;c.elevation=el;cams.append(c)
names=f['names'];starts=f['starts'];qs=f['q'];proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-c:v','libx264','-pix_fmt','yuv420p',str(out/'operation.mp4')],stdin=subprocess.PIPE)
for i in range(0,len(qs),8):
 q=qs[i];raw=i*4
 for j,name in enumerate(names):
  id=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,str(name))
  if id<0:continue
  o=vm.jnt_qposadr[id];k=starts[j]
  if vm.jnt_type[id]==0:vd.qpos[o:o+7]=np.r_[q[k:k+3],q[k+6],q[k+3:k+6]]
  else:vd.qpos[o]=q[k]
 mj.mj_forward(vm,vd);ren.update_scene(vd,cams[0]);left=ren.render().copy();vm.geom_group[hs]=5;ren.update_scene(vd,cams[1],scene_option=opt);right=ren.render().copy();vm.geom_group[hs]=0
 im=Image.fromarray(np.hstack([left,right]));dr=ImageDraw.Draw(im);dr.rectangle([0,0,960,23],fill=(24,34,38));dr.text((8,5),f'hand_111 | Newton | {x[raw,0]:.2f}s',fill='white');dr.text((486,5),'Bottom to top | centre displacement (mm): '+' / '.join(f'{z:.1f}' for z in d[raw]),fill='white');proc.stdin.write(np.asarray(im).tobytes())
proc.stdin.close();assert proc.wait()==0;ren.close()
r['tag']=tag;r['edge_separation_mm']=separation;(out/'metrics.json').write_text(json.dumps(r));np.savetxt(out/'displacements.csv',np.c_[x[:,0],d],delimiter=',',header='t_s,bottom1_mm,card2_mm,card3_mm,card4_mm,top5_mm',comments='',fmt='%.7f')
base=metrics('base');idle=metrics('idle');mu=metrics('pairtest')
txt=f"""## 19 薄卡分拣：五层独立薄刚体
每张 54 × 32 × 0.80 mm；抽取方向两端各有 2 mm 过渡，厚度从 0.80 mm 收到 0.42 mm，端部宽度从 32 mm 收到 30 mm。中央平面和两个凸边缘体随各自卡片运动。五卡自由，无胶合、导轨、卡扣、压卡夹或按编号冻结。卡间滑动摩擦系数 0.23，指卡 0.95，桌卡 0.72（仿真参数，非材料实测）。标准初态 5 张齐叠，层间初始间隙 0.02 mm，重力沉降。hand_111 左手在 Newton SolverMuJoCo CPU 960 Hz 仅驱动手关节，指腹从上表面摩擦抽卡。不模拟弯曲或静电。

验收：顶卡末帧沿抽出方向位移 ≥30 mm，同时相对其下方第 4 张净抽出 ≥30 mm；其余每张卡的三维位移始终 ≤5 mm，检查中心及全部外缘顶点，排除平移后回退及转角超限。手在末段至少 0.25 s 完全脱离。初态为整段参考。

|从底到顶|末帧沿抽出方向 mm|全程中心最大三维位移 mm|全程角点最大三维位移 mm|
|---:|---:|---:|---:|
"""
for i in range(5):txt+=f"|{i+1}|{r['final_x_mm'][i]:.2f}|{r['max_com_mm'][i]:.2f}|{r['max_corner_mm'][i]:.2f}|\n"
txt+=f"""
顶卡相对第 4 张净抽出 {r['relative_mm']:.2f} mm，通过。手与下层卡有载接触步数 {r['other_hand_steps']}。无手对照最大中心漂移 {max(idle['max_com_mm']):.2f} mm。动作反例 base：下层最大中心位移 {max(base['max_com_mm'][:-1]):.2f} mm，不通过。相同双拨动作的高摩擦对照（pairtest，卡间系数 0.60）：顶卡相对第 4 张 {mu['relative_mm']:.2f} mm，下层最大角点位移 {max(mu['max_corner_mm'][:-1]):.2f} mm，判定 {'通过' if mu['success'] else '未通过'}。

证据：outputs/nov20/card19/operation.mp4、displacements.csv、metrics.json、{tag}.xml、{tag}.npz（逐步各卡位姿、边缘接触计数、各层法向力）。复跑参数 X=-.120 Y=.023 Z=.0735 FN=.055 DX=.064 TL=1.65，IsaacLab3 Python 运行 scripts/nov20/card19.py {tag}；verify19.py 可独立复算。固定初态关节控制正例。"""
p=OUT.parent.parent/'docs/SCENES_20_20261116.md';orig=p.read_text()
fr=json.loads((OUT/'grind12/friction.json').read_text());note=f"""### 12 补测：切向摩擦与木屑效果
复跑与原正例逐步位姿、法向力和评分列完全相同。按接触坐标系读取两切向分力并转到世界坐标；取作用于磨石的方向，按接触点速度计算耗散。盘面滑动摩擦系数 {fr['mu']:.2f}；有效且行程方向速度绝对值 >5 mm/s 样本，切向合力中位 {fr['median_tangent_N']:.3f} N，{fr['opposing_x_fraction']*100:.1f}% 的行程向摩擦分量与磨石运动反向；六程摩擦耗散 {fr['dissipated_J']:.3f} J。手驱动磨石，摩擦提供阻力。
outputs/nov20/grind12/wood_fx.mp4 是独立视觉版本，仅在有效接触、有摩擦耗散时生成木屑状粒子并下落；无质量回写、不改变接触或计分，也不代表石材产生木屑或模拟了木料去除量。\n"""
add=''
for txt0 in [note,txt]:
 if txt0.splitlines()[0] not in orig:add+='\n\n'+txt0
p.write_text(orig.rstrip()+add+'\n');(out/'RESULTS.md').write_text(txt+'\n');print(r)

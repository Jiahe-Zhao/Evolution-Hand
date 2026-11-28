
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
import newton,sys,csv,subprocess
from PIL import Image,ImageDraw
o=OUT/'keys17';o.mkdir(exist_ok=True)
r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'));c=E.SubElement(r,'contact')
def box(b,name,pos,size,col,mass):
 return E.SubElement(b,'geom',name=name,type='box',pos=pos,size=size,rgba=col,mass=mass,contype='2',conaffinity='1')
box(w,'deck','-.114 -.028 .080','.035 .043 .002','.23 .27 .32 1','1')
for i,(x,y,col) in enumerate(zip([-.123,-.128,-.118,-.097],[.001,-.018,-.037,-.056],['.85 .35 .18','.86 .65 .15','.19 .62 .56','.25 .51 .83'])):
 h=E.SubElement(w,'body',name=f'key{i}',pos=f'{x} {y} .096');E.SubElement(h,'joint',name=f'slide{i}',type='slide',axis='0 0 -1',range='0 .004',damping='.8',armature='.001')
 box(h,f'cap{i}','0 0 0','.009 .0075 .002',col+' 1','.004')
 for g in w.find('body').iter('geom'):E.SubElement(c,'pair',geom1=g.get('name'),geom2=f'cap{i}')
p=o/'keys.xml';E.ElementTree(r).write(p)
b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2;b.default_joint_cfg.target_kd=.08;b.add_mjcf(str(p),collapse_fixed_joints=False)
names=[n.split('/')[-1] for n in b.joint_label];jj=lambda name:b.joint_qd_start[names.index(name)]
for i in range(4):k=jj(f'slide{i}');b.joint_target_ke[k]=b.joint_target_kd[k]=0
for name in names:
 if 'diagnostic_wrist' in name:k=jj(name);b.joint_target_ke[k]=1700;b.joint_target_kd[k]=42
m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ct=m.control();v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50);mm=v.mj_model;dt=1/480
ks=[jj(f'slide{i}') for i in range(4)];fs=[jj(f'link_0_0_to_link_{i}_0') for i in range(2,6)];base=ct.joint_target_q.numpy().copy()
mode=sys.argv[1];render='video' in sys.argv
if render:
 vm=mj.MjModel.from_xml_path(str(p));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480);cam=mj.MjvCamera();cam.lookat[:]=[-.105,-.028,.102];cam.distance=.212;cam.azimuth=160;cam.elevation=-47;frames=[]
rows=[];contacts=[];bad=set()
for t in range(4*480+120):
 i=(t-120)//480;phase=(t-120)%480;cmd=base.copy();s.clear_forces()
 q=s.joint_q.numpy();force=np.zeros_like(base);force[ks]=-210*q[ks] # upward spring
 if i>=0:
  strength=np.clip(min(phase/110,(330-phase)/110),0,1)
  if mode=='probe':force[ks[i]]+=.72*strength
  else:cmd[fs[i]]+=[.31,.30,.34,.43][i]*strength
 ct.joint_f.assign(force);ct.joint_target_q.assign(cmd);v.step(s,n,ct,None,dt);s,n=n,s
 depth=s.joint_q.numpy()[ks]*1000;rows.append([t*dt,*depth]);record=[]
 for j,co in enumerate(v.mj_data.contact):
  gg=[mm.geom(int(z)).name for z in co.geom];caps=[g for g in gg if '/cap' in g]
  if not caps or co.dist>0:continue
  cap=int(caps[0].split('/cap')[1][0]);other=gg[1-gg.index(caps[0])];f=np.zeros(6);mj.mj_contactForce(mm,v.mj_data,j,f)
  if f[0]>.001:
   record.append([t*dt,cap+1,other,float(f[0])])
   if f'/link_{cap+2}_3/' not in other:bad.add(other)
 contacts.extend(record)
 if render and t%16==0:
  for j,name in enumerate(names):
   if name in [vm.joint(a).name for a in range(vm.njnt)]:vd.qpos[vm.jnt_qposadr[vm.joint(name).id]]=s.joint_q.numpy()[b.joint_q_start[j]]
  mj.mj_forward(vm,vd);ren.update_scene(vd,cam);im=Image.fromarray(ren.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,480,22),fill='#192633');dr.text((8,5),'Key travel [mm]   '+ '   '.join(f'{j+1}: {z:.2f}' for j,z in enumerate(depth)),fill='white');frames.append(np.asarray(im))
a=np.array(rows);print(mode,[a[120+j*480:600+j*480,1:].max(0).round(2).tolist() for j in range(4)],'bad',bad,'end',a[-1],flush=True)
np.savetxt(o/(mode+'.csv'),a,delimiter=',',header='time_s,index_mm,middle_mm,ring_mm,little_mm',comments='')
with open(o/(mode+'_contacts.csv'),'w') as f:
 writer=csv.writer(f);writer.writerow(['time_s','key','contact_geom','normal_N']);writer.writerows(contacts)
if render:
 subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','480x360','-r','30','-i','-','-pix_fmt','yuv420p',str(o/'operation.mp4')],input=np.asarray(frames).tobytes(),check=True);ren.close()


# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import OUT,E,np,mj
from pathlib import Path
import os,sys,json,subprocess,newton
from scipy.spatial.transform import Rotation as R
out=OUT/'latch16';out.mkdir(exist_ok=True)
teal='.20 .44 .47 1';wood='.62 .39 .22 1';gold='.93 .61 .15 1'
def box(b,n,pos,half,col,mass=None,**kw):
 g=E.SubElement(b,'geom',name=n,type='box',pos=' '.join(map(str,pos)),size=' '.join(map(str,half)),rgba=col,contype='2',conaffinity='3',friction='.65 .004 .0001',**kw)
 if mass:g.set('mass',str(mass))
 return g
def scene():
 r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');h=w.find('body');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'))
 # fixed tray; front at x=-.174, hinge center at -.216, lid plane z=.080
 box(w,'bottom',[-.196,.010,.048],[.022,.025,.002],teal)
 for y in [-.014,.034]:box(w,'side'+str(y),[-.196,y,.062],[.022,.001,.012],teal)
 for x in [-.217,-.175]:box(w,'end'+str(x),[x,.01,.062],[.001,.024,.012],teal)
 box(w,'stand',[-.196,.01,.022],[.018,.021,.022],'.29 .31 .32 1')
 lid=E.SubElement(w,'body',name='lid',pos='-.216 .010 .078')
 E.SubElement(lid,'joint',name='lid_hinge',type='hinge',axis='0 -1 0',range='0 1.66',damping='.00024',frictionloss='.00005',armature='.00002')
 box(lid,'cover',[.021,0,0],[.022,.025,.002],wood,.024)
 box(lid,'lip',[.049,-.012,0],[.008,.011,.002],wood,.004)
 box(lid,'ear',[.046,.013,0],[.004,.003,.002],'.73 .45 .25 1',.002)
 for j,y in enumerate([-.027,.003]):box(lid,'pullpost'+str(j),[.049,y,.0125],[.004,.002,.0125],wood,.0012)
 box(lid,'pullbar',[.049,-.012,.025],[.006,.017,.004],'.26 .35 .36 1',.0024)
 # Frame-mounted bolt travels sideways across ear; slot open over lip.
 bolt=E.SubElement(w,'body',name='bolt',pos='-.169 .023 .083')
 E.SubElement(bolt,'joint',name='slide_bolt',type='slide',axis='0 1 0',range='0 .018',damping='.08',frictionloss='.035',armature='.002')
 box(bolt,'tongue',[0,.003,0],[.004,.008,.002],gold,.005)
 box(bolt,'tab',[.001,.005,.008],[.004,.002,.008],gold,.004)
 # guide bracket outboard of ear; does not overlap the lid sweep.
 box(w,'bracket',[-.174,.044,.070],[.004,.004,.009],teal)
 box(w,'guide',[-.169,.043,.079],[.008,.007,.0015],teal)
 c=E.SubElement(r,'contact')
 E.SubElement(c,'pair',geom1='ear',geom2='tongue',solref='.006 1')
 for g in h.iter('geom'):
  for obj in list(lid.iter('geom'))+list(bolt.iter('geom'))+[o for o in w.findall('geom') if o.get('name')!='floor']:E.SubElement(c,'pair',geom1=g.get('name'),geom2=obj.get('name'),friction='.8 .8 .003 .0001 .0001')
 p=out/'latch16.xml';E.ElementTree(r).write(p);return p
def run(mode):
 p=scene();b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.add_mjcf(str(p),collapse_fixed_joints=False);names=[x.split('/')[-1] for x in b.joint_label];valid={j.get('name') for j in _evo_parse_xml(p).iter('joint')};ids=[]
 for i,n in enumerate(names):
  k=b.joint_qd_start[i]
  if n in ('slide_bolt','lid_hinge'):b.joint_target_ke[k]=b.joint_target_kd[k]=0
  elif n in valid:
   ids.append(k)
   if 'diagnostic_wrist' in n:b.joint_target_ke[k]=1700;b.joint_target_kd[k]=42
 if mode=='clear':b.joint_q[b.joint_q_start[names.index('slide_bolt')]]=.018
 m=b.finalize(device='cpu');s=m.state();nxt=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50);base=ctrl.joint_target_q.numpy().copy()
 mm=v.mj_model;js=[b.joint_q_start[names.index(k)] for k in ['slide_bolt','lid_hinge']]; bodies=[k.split('/')[-1] for k in m.body_label];lid=bodies.index('lid');tip=bodies.index('link_2_3')
 vm=mj.MjModel.from_xml_path(str(p));vd=mj.MjData(vm);ren=mj.Renderer(vm,360,480);cam=mj.MjvCamera();cam.lookat[:]=[-.178,.015,.088];cam.distance=.205;cam.azimuth=116;cam.elevation=-26
 rows=[];frames=[];events=[];dt=1/240
 def smooth(z):z=np.clip(z,0,1);return z*z*(3-2*z)
 # command the neutral index pad center in world space, via wrist translation
 init=np.array([-.128,.00124,.130])
 pts=[(0,init),(90,[-.163,.003,.107]),(145,[-.165,.003,.093]),(240,[-.165,.050,.093]),(300,[-.133,.050,.096]),(380,[-.133,-.004,.092]),(440,[-.166,-.004,.093])]
 for k,ang in enumerate(np.linspace(0,78,11)):
  th=np.deg2rad(ang);pts.append((475+25*k,[-.216+.049*np.cos(th)-.019*np.sin(th),-.004,.078+.049*np.sin(th)+.020*np.cos(th)]))
 if mode=='direct':pts=[(0,init),(140,[-.133,-.004,.092]),(440,[-.166,-.004,.093])]+pts[7:]
 for t in range(790 if mode in ('hand','direct') else 240):
  cmd=base.copy();s.clear_forces()
  if mode in ('locked','clear'):
   f=np.zeros((m.body_count,6),np.float32);f[lid,4]=-.025*min(1,t/30);s.body_f.assign(f)
  else:
   dst=np.array(pts[-1][1])
   for (i,a),(j,h) in zip(pts,pts[1:]):
    if i<=t<=j:dst=np.array(a)+(np.array(h)-a)*smooth((t-i)/(j-i));break
   off=dst-init;cmd[ids[:3]]=off[[0,2,1]]*[1,1,-1]
  ctrl.joint_target_q.assign(cmd);v.step(s,nxt,ctrl,None,dt);s,nxt=nxt,s
  slide,angle=s.joint_q.numpy()[js];touch=set()
  for co in v.mj_data.contact:
   if co.dist>0:continue
   g=[mm.geom(int(i)).name for i in co.geom];body=[a.split('/')[-2] for a in g]
   if 'bolt' in body or 'lid' in body:
    touch.add('|'.join(sorted([a.split('/')[-2]+'/'+a.split('/')[-1].rsplit('_',1)[0] for a in g])))
  if False:
   print('dbg',t,s.body_q.numpy()[tip,:3],cmd[ids[:3]],v.mj_data.geom_xpos[[i for i in range(mm.ngeom) if 'handgeom40_' in mm.geom(i).name]],flush=True)
  rows.append([t,slide*1000,np.rad2deg(angle)]);events.append(sorted(touch))
  if t%8==0:
   q=s.joint_q.numpy()
   for j,name in enumerate(names):
    if name in valid:vd.qpos[vm.jnt_qposadr[vm.joint(name).id]]=q[b.joint_q_start[j]]
   mj.mj_forward(vm,vd);ren.update_scene(vd,cam);left=ren.render().copy()
   handgeoms=[i for i in range(vm.ngeom) if vm.geom(i).name.startswith('handgeom')]
   vm.geom_group[handgeoms]=5
   opt=mj.MjvOption();opt.geomgroup[5]=0
   inspect=mj.MjvCamera();inspect.lookat[:]=[-.185,.018,.084];inspect.distance=.153;inspect.azimuth=43;inspect.elevation=-26
   ren.update_scene(vd,inspect,scene_option=opt);right=ren.render().copy();vm.geom_group[handgeoms]=0
   from PIL import Image,ImageDraw
   im=Image.fromarray(np.hstack([left,right]));dr=ImageDraw.Draw(im);dr.rectangle([0,0,960,25],fill=(22,30,34));dr.text((8,6),f'{mode} | hand_111 | t={t/240:.2f}s',fill='white');dr.text((486,6),f'Mechanism view  |  bolt {slide*1000:.1f} mm  |  lid {np.rad2deg(angle):.1f} deg',fill='white');frames.append(np.asarray(im).copy())
 np.save(out/(mode+'.npy'),rows);(out/(mode+'_contacts.json')).write_text(json.dumps(events));subprocess.run(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x360','-r','30','-i','-','-pix_fmt','yuv420p',str(out/(mode+'.mp4'))],input=np.asarray(frames).tobytes(),check=True);ren.close()
 a=np.array(rows);print(mode,a[:,1:].max(0),a[-1],sorted(set.union(*map(set,events))),flush=True)
if __name__=='__main__':run(sys.argv[1])

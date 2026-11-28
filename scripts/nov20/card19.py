
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import OUT,np,E,mj
from surface_objects import geom
import os,sys,json,newton
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
out=OUT/'card19';out.mkdir(exist_ok=True);cx=-.123;cy=.021;table=.067;th=.0008;gap=.00002;count=5
colors=['.33 .40 .60 1','.71 .44 .30 1','.42 .59 .48 1','.82 .69 .46 1','.32 .65 .68 1']
def scene(tag):
 r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');h=w.find('body');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'));a=r.find('asset')
 geom(w,'deck_table','box',(cx+.012,cy,table/2),'.097 .058 .0335','.30 .32 .34 1')
 cards=[]
 for i in range(count):
  b=E.SubElement(w,'body',name='card'+str(i+1),pos=f'{cx} {cy} {table+(i+.5)*th+(i+1)*gap}');E.SubElement(b,'freejoint',name='free'+str(i));cards.append(b);col=colors[i]
  geom(b,f'face{i}','box',(0,0,0),'.025 .016 .0004',col,mass=str(1350*.050*.032*th))
  for side in [-1,1]:
   pts=np.array([[side*x,y*wid,z*ht] for x,wid,ht in [(.025,.016,.0004),(.027,.015,.00021)] for y in [-1,1] for z in [-1,1]])
   n=f'edge{i}'+('a' if side<0 else 'b');E.SubElement(a,'mesh',name=n,vertex=' '.join(map(str,pts.ravel())))
   geom(b,n,'mesh',(0,0,0),'.01',col,mesh=n,mass=str(1350*ConvexHull(pts).volume))
 c=E.SubElement(r,'contact')
 def pair(g,h,mu,ref):E.SubElement(c,'pair',geom1=g,geom2=h,friction=f'{mu} {mu} .00005 .000001 .000001',solref=f'{ref} 1',solimp='.95 .99 .0001 .5 2')
 for i,b in enumerate(cards):
  for g in b.iter('geom'):
   n=g.get('name');pair(n,'deck_table',.72,.003)
   for o in h.iter('geom'):pair(n,o.get('name'),.95,.005)
   for earlier in cards[:i]:
    for o in earlier.iter('geom'):pair(n,o.get('name'),float(os.getenv('MU','.23')),.003)
 p=out/(tag+'.xml');E.ElementTree(r).write(p);return p
def run(tag):
 p=scene(tag);b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(p),collapse_fixed_joints=False)
 names=[s.split('/')[-1] for s in b.joint_label];valid={j.get('name') for j in _evo_parse_xml(p).iter('joint')};ids=[]
 for j,k in enumerate(names):
  if k not in valid or not k.startswith(('diagnostic_wrist','wrist_','link_')):continue
  i=b.joint_qd_start[j];ids.append(i)
  if k.startswith('diagnostic_wrist'):b.joint_target_ke[i]=1700;b.joint_target_kd[i]=42
 m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ctrl=m.control();base=ctrl.joint_target_q.numpy().copy();v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50);mm=v.mj_model;dd=v.mj_data;body=[k.split('/')[-1] for k in m.body_label];idx=[body.index('card'+str(i+1)) for i in range(5)]
 gn=[mm.geom(i).name for i in range(mm.ngeom)];hz=960;init=s.body_q.numpy()[idx].copy();qs=[];data=[];contacts=[];zs=0.;force=0.;target=float(os.getenv('FN','.12'));start=2.;distance=float(os.getenv('DX','.045'));y=float(os.getenv('Y','.008'));tlen=float(os.getenv('TL','2'));xp=float(os.getenv('X','-.124'));above=float(os.getenv('Z', '.077'));near=np.array([xp,y,above]);orig=np.array([-.12794,.00124,.130])
 sm=lambda v:(lambda u:u*u*(3-2*u))(np.clip(v,0,1))
 second=tag in ('single','pairtest');end1=start+tlen+.9
 for i in range(int((end1+(3.0 if second else 0))*hz)):
  t=i/hz;dest=orig+(near-orig)*sm(t/.85)
  if .85<t<start+tlen+.15:zs=np.clip(zs+.000012*(force-target),-.010,.005)
  dest[2]+=zs+.037*sm((t-start-tlen-.15)/.38)
  dest[0]+=distance*sm((t-start)/tlen)
  if second and t>=end1:
   tt=t-end1
   if tt<1/hz:home2=dest.copy();c2=poses[-1,:3].copy();zs2=0.
   anchor=c2+[.005,-.001,.0075];dest=home2+(anchor-home2)*sm(tt/.65)
   if .65<tt<2.32:zs2=np.clip(zs2+.000015*(force-.085),-.010,.005)
   dest[2]+=zs2+.038*sm((tt-2.32)/.38)
   dest[0]+=.039*sm((tt-1.)/1.2)
  cmd=base.copy()
  if tag!='idle':
   delta=dest-orig;cmd[ids[:3]]=delta[[0,2,1]]*[1,1,-1]
  ctrl.joint_target_q.assign(cmd);s.clear_forces();v.step(s,n,ctrl,None,1/hz);s,n=n,s;poses=s.body_q.numpy()[idx];force=0.;hc=0;hc_other=0;edge=0;loads=np.zeros(4)
  for j,co in enumerate(dd.contact):
   if co.dist>0 or co.efc_address<0:continue
   g=[gn[h] for h in co.geom];bo=[z.split('/')[-2] for z in g]
   if not any(z.startswith('card') for z in bo):continue
   f=np.zeros(6);mj.mj_contactForce(mm,dd,j,f)
   if f[0]<1e-6:continue
   if all(z.startswith('card') for z in bo):
    layers=sorted(int(z[-1]) for z in bo)
    if layers[1]==layers[0]+1:loads[layers[0]-1]+=f[0]
    if any('/edge' in z for z in g):edge+=1
   if any(z.startswith('link_') for z in bo):
    if 'card5' in bo:force+=f[0];hc+=1
    else:hc_other+=1
  data.append([t,*poses.flatten(),force,hc,hc_other,edge,*loads])
  if i%4==0:qs.append(s.joint_q.numpy().copy())
 x=np.array(data);p=x[:,1:36].reshape(-1,5,7);disp=np.linalg.norm(p[:,:,:3]-init[None,:,:3],axis=2)*1000
 np.savez_compressed(out/(tag+'.npz'),data=x,q=qs,names=names,starts=b.joint_q_start,initial=init)
 print('MAX',disp.max(0),'final signed x',(p[-1,:,0]-init[:,0])*1000,'force',x[:,36].max(),flush=True)
if __name__=='__main__':run(sys.argv[1])

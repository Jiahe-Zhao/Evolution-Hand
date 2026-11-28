
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
import os,sys,newton,csv,json
from scipy.spatial.transform import Rotation as R
o=OUT/'scan20';o.mkdir(exist_ok=True);mode=sys.argv[1];tag=sys.argv[2] if len(sys.argv)>2 else mode
r=_evo_parse_xml(OUT/'carry.xml').getroot();w=r.find('worldbody');hand=w.find('body');w.remove(w.find("body[@name='item']"));r.remove(r.find('contact'));c=E.SubElement(r,'contact')
fmt=lambda x:' '.join('%.9g'%v for v in x)
def g(b,n,kind,x,size,col,mass=None,**kw):
 a=E.SubElement(b,'geom',name=n,type=kind,pos=fmt(x),size=fmt(size),rgba=col,**kw)
 if mass:a.set('mass',str(mass))
 return a
ob=E.SubElement(w,'body',name='scanner',pos='0 0 0');E.SubElement(ob,'freejoint',name='scan_free')
# handle slopes to follow the three curled supporting fingers.
g(ob,'grip','capsule',[-.012,-.032,.085],[.017,.023],'.11 .22 .29 1',.037,euler='0.98 0 0')
g(ob,'heel','ellipsoid',[-.012,-.055,.102],[.022,.011,.020],'.16 .37 .43 1',.008)
g(ob,'bridge','capsule',[-.011,-.011,.071],[.010,.014],'.18 .34 .38 1',.009,euler='1.0 0 0')
g(ob,'housing','box',[-.020,.006,.073],[.029,.012,.012],'.23 .52 .54 1',.024)
g(ob,'window','box',[-.050,.006,.073],[.001,.008,.006],'.12 .18 .23 1',.001)
# depress towards +x
tr=E.SubElement(ob,'body',name='trigger',pos=os.getenv('TRPOS','-.031 -.022 .037'));E.SubElement(tr,'joint',name='pull',type='slide',axis='1 0 0',range='0 .006',damping='.8',armature='.001')
g(tr,'paddle','box',[0,0,0],[.0025,.005,.007],'.94 .48 .14 1',.003)
g(tr,'stem','capsule',[.007,0,0],[.002,.006],'.70 .48 .25 1',.0007,euler='0 1.5707963 0')
g(ob,'guidepost','capsule',[-.016,-.017,.052],[.003,.014],'.19 .29 .33 1',.002,euler='-.35 0 0')
for hg in hand.iter('geom'):
 for sg in ob.iter('geom'):E.SubElement(c,'pair',geom1=hg.get('name'),geom2=sg.get('name'),solref='.009 1',friction='1.0 1.0 .003 .0001 .0001')
L=['0001101','0011001','0010011','0111101','0100011','0110001','0101111','0111011','0110111','0001011'];digits='7351353';digits+=str((-sum(int(x)*(3 if j%2==0 else 1) for j,x in enumerate(digits)))%10);Rcodes=[''.join('1' if c=='0' else '0' for c in x) for x in L];bits='101'+''.join(L[int(x)] for x in digits[:4])+'01010'+''.join(Rcodes[int(x)] for x in digits[4:])+'101'
if (o/'board.txt').exists():B,U,V,N=np.loadtxt(o/'board.txt')
else:B=np.array([-.19,.006,.073]);U=np.array([0,1.,0]);V=np.array([0,0,1.]);N=np.cross(U,V)
qr=R.from_matrix(np.stack([U,V,N],axis=1)).as_quat()[[3,0,1,2]];module=.00062
g(w,'label','box',B-.0006*N,[.027,.014,.0006],'.96 .95 .90 1',quat=fmt(qr))
for j,bit in enumerate(bits):
 if bit=='1':g(w,'bar'+str(j),'box',B+U*(j-33)*module+.00003*N,[module*.497,.010,.00003],'.022 .03 .04 1',quat=fmt(qr))
if mode=='blocked':g(w,'mask','box',B+.012*N,[.028,.018,.001],'.38 .37 .40 1',quat=fmt(qr))
xml=o/(tag+'.xml');E.ElementTree(r).write(xml)
b=newton.ModelBuilder();b.default_joint_cfg.target_ke=2.;b.default_joint_cfg.target_kd=.08;b.default_joint_cfg.armature=.0003;b.add_mjcf(str(xml),collapse_fixed_joints=False)
ns=[n.split('/')[-1] for n in b.joint_label];ki=b.joint_qd_start[ns.index('pull')];qi=b.joint_q_start[ns.index('pull')];b.joint_target_ke[ki]=b.joint_target_kd[ki]=0
for i in range(3):b.joint_target_ke[i]=1700;b.joint_target_kd[i]=42
q=np.array(b.joint_q);q[6:11]=[.25,.7,.55,.65,.55];q[12:15]=[.80,.70,.50]
for i in (16,20,24):q[i:i+3]=[1.05,1.05,1.0]
b.joint_q[:]=q.tolist();m=b.finalize(device='cpu');s=m.state();n=m.state();newton.eval_fk(m,m.joint_q,m.joint_qd,s);ct=m.control();base=ct.joint_target_q.numpy().copy();base[:27]=q[:27];v=newton.solvers.SolverMuJoCo(m,use_mujoco_cpu=True,iterations=50);mm=v.mj_model;dd=v.mj_data;gn=[mm.geom(i).name.split('/')[-1].rsplit('_',1)[0] for i in range(mm.ngeom)];win=gn.index('window');scanid=int(mm.geom_bodyid[win]);dt=1/480;rows=[];contacts=[];qs=[];poses=[];latch=False;events=[]
def decode(lens,axis):
 vals=[]
 for j in range(-1,68):
  delta=B+(j-33)*module*U-lens;direction=delta/np.linalg.norm(delta);
  if direction@axis<np.cos(np.deg2rad(15)):return ''
  gid=np.array([-1],dtype=np.int32);mj.mj_ray(mm,dd,lens,direction,None,1,scanid,gid)
  n=gn[gid[0]] if gid[0]>=0 else '';vals.append('1' if n.startswith('bar') else '0' if n=='label' else '?')
 word=''.join(vals);good=word[0]==word[-1]=='0' and word[1:4]=='101' and word[32:37]=='01010' and word[-4:-1]=='101'
 if not good:return ''
 try:
  t=word[1:-1];decoded=''.join(str(L.index(t[3+j*7:10+j*7])) for j in range(4))+''.join(str(Rcodes.index(t[36+j*7:43+j*7])) for j in range(4))
 except ValueError:return ''
 return decoded if sum(int(x)*(3 if j%2==0 else 1) for j,x in enumerate(decoded))%10==0 else ''
for k in range(1300):
 T=k*dt;cmd=base.copy();cmd[1]+=.008*np.clip((T-.35)/.55,0,1);phase=np.clip(min((T-1.25)/.35,(2.30-T)/.35),0,1) if mode!='idle' else 0
 cmd[12:15]+=.56*phase*np.array([float(os.getenv('C1','.15')),float(os.getenv('C2','.34')),float(os.getenv('C3','.32'))]);ct.joint_target_q.assign(cmd);f=np.zeros(m.joint_dof_count,dtype=np.float32);f[ki]=-float(os.getenv('SPR','130'))*s.joint_q.numpy()[qi];ct.joint_f.assign(f);s.clear_forces();v.step(s,n,ct,None,dt);s,n=n,s
 dep=s.joint_q.numpy()[qi]*1000;loads={};idx=other=0.
 for j,co in enumerate(dd.contact):
  if co.dist>0 or co.efc_address<0:continue
  pair=[gn[z] for z in co.geom]
  if not any(x.startswith('handgeom') for x in pair):
   if set(pair).intersection({'grip','bridge','housing','window','heel','paddle','guidepost','stem'}) and set(pair).intersection({'floor','label','mask'}):
    F=np.zeros(6);mj.mj_contactForce(mm,dd,j,F)
    if F[0]>.001:contacts.append([T,'external_support','/'.join(pair),F[0]])
   continue
  hi=next(i for i,x in enumerate(pair) if x.startswith('handgeom'));body=mm.geom(co.geom[hi]).name.split('/')[-2];obj=pair[1-hi];F=np.zeros(6);mj.mj_contactForce(mm,dd,j,F)
  if F[0]<=.001:continue
  loads[(obj,body)]=loads.get((obj,body),0)+F[0]
  if obj=='paddle':
   if body.startswith('link_2_'):idx+=F[0]
   else:other+=F[0]
 for (obj,bod),val in loads.items():contacts.append([T,obj,bod,val])
 lens=dd.geom_xpos[win]-dd.geom_xmat[win].reshape(3,3)[:,0]*.0011;axis=-dd.geom_xmat[win].reshape(3,3)[:,0];delta=B-lens;dist=np.linalg.norm(delta);ang=np.degrees(np.arccos(np.clip(delta@axis/dist,-1,1)));gid=np.array([-1],dtype=np.int32);mj.mj_ray(mm,dd,lens,delta/dist,None,1,scanid,gid);los=gid[0]>=0 and (gn[gid[0]]=='label' or gn[gid[0]].startswith('bar'))
 grip=sorted(set(bod.split('_')[1] for (obj,bod),val in loads.items() if obj in ['grip','heel','bridge','housing'] and bod.startswith('link_') and bod.split('_')[1] in ['1','3','4','5']))
 edge=dep>=3 and not latch;event=''
 if edge:
  latch=True
  if .08<=dist<=.2 and ang<=10 and los and idx>.01 and other==0 and len(grip)>=2:event=decode(lens,axis)
 if dep<=.6:latch=False
 if event:events.append((T,event))
 rows.append([T,dist*1000,ang,int(los),dep,idx,other,len(grip),int(edge),int(bool(event))]);qs.append(s.joint_q.numpy().copy());poses.append([*lens,*axis,*dd.geom_xpos[gn.index('paddle')],*dd.geom_xpos[gn.index('handgeom40')]])
a=np.array(rows);np.savez_compressed(o/(tag+'.npz'),a=a,q=qs,poses=poses);np.savetxt(o/(tag+'_trace.csv'),a,delimiter=',',header='time_s,distance_mm,angle_deg,line_of_sight,trigger_mm,index_N,other_trigger_N,supporting_digits,rising_edge,read',comments='')
with open(o/(tag+'_contacts.csv'),'w') as ff:
 wr=csv.writer(ff);wr.writerow(['time_s','object','hand_link','normal_N']);wr.writerows(contacts)
(o/(tag+'_reads.json')).write_text(json.dumps(events));print(tag,events,'stages',a[[120,480,600,700,810,1000,1220],1:9],flush=True)

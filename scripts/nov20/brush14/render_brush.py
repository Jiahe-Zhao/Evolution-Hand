
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import os,sys,json,subprocess
from pathlib import Path
import numpy as np,mujoco as mj
from PIL import Image,ImageDraw,ImageFont
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));tag=sys.argv[1];D=np.load(P/(tag+'.npz'));a=D['a'];q=D['q'];ink=D['ink'];info=json.loads((P/(tag+'_model.json')).read_text());m=mj.MjModel.from_xml_path(str(P/(tag+'.xml')));d=mj.MjData(m);ren=mj.Renderer(m,480,640,max_geom=50000);center=np.array(info['rest_center']);z=info['paper_z'];shaft=m.geom('shaft').id
cam=mj.MjvCamera();cam.lookat[:]=center+[.01,0,.017];cam.distance=.265;cam.azimuth=115;cam.elevation=-23
close=mj.MjvCamera();close.lookat[:]=[center[0]+.009,center[1],z+.012];close.distance=.075;close.azimuth=100;close.elevation=-18
if info['mode']=='hao':
 cam.lookat[:]=center+[.012,-.016,.017];close.lookat[:]=[center[0]+.010,center[1]-.018,z+.012];close.distance=.14;close.elevation=-40
font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),16);small=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),13)
marks=ink[(ink[:,0].astype(int)%8==0)&(ink[:,5]>1e-5)] if len(ink) else ink
# Verify replay coordinates against recorded physical hair geometry, not merely export success.
hair_ids=[g.id for g in [m.geom(name) for name in [m.geom(i).name for i in range(m.ngeom) if 'bristle_' in m.geom(i).name]]]
if 'hair_names' in D:hair_ids=[m.geom(str(name)).id for name in D['hair_names']]
elif len(hair_ids)!=D['hair'].shape[1]:hair_ids=[i for i in hair_ids if m.geom(i).name.endswith('_capsule')]
errors=[]
for t in [0,len(q)//2,len(q)-1]:
 d.qpos[:]=q[t];mj.mj_forward(m,d);errors.append(float(np.max(np.abs(d.geom_xpos[hair_ids]-D['hair'][t]))))
if max(errors)>1e-6:raise RuntimeError(f'Replay geometry mismatch {errors}')
# Seven visual strands per physical bundle, advected by its actual segment poses.
# They stay within the coarse collision envelope; no separate scripted animation.
capids=[[m.geom(f'bristle_{b:02d}_{j:02d}_capsule').id for j in range(5)] for b in range(13)]
for group in capids:
 for g in group:m.geom_rgba[g,3]=0
for i in range(m.ngeom):
 if m.geom(i).name.endswith('_tip'):m.geom_rgba[i,3]=0
def fibers():
 for group in capids:
  nodes=[]
  for j,gid in enumerate(group):
   axis=d.geom_xmat[gid].reshape(3,3)[:,2];c=d.geom_xpos[gid];half=m.geom_size[gid,1];ends=[c-axis*half,c+axis*half];bp=d.xpos[m.geom_bodyid[gid]]
   if np.linalg.norm(ends[0]-bp)>np.linalg.norm(ends[1]-bp):ends.reverse()
   if j==0:nodes.append(ends[0])
   nodes.append(ends[1])
  nodes=np.array(nodes);tan=np.gradient(nodes,axis=0);tan/=np.linalg.norm(tan,axis=1)[:,None];ref=np.tile([1.,0,0],(6,1));normal=ref-(ref*tan).sum(axis=1)[:,None]*tan;normal/=np.linalg.norm(normal,axis=1)[:,None];binormal=np.cross(tan,normal)
  for strand in range(7):
   ang=2*np.pi*strand/6;radius=np.linspace(.00023,.00011,6)*(strand<6);points=nodes+radius[:,None]*(np.cos(ang)*normal+np.sin(ang)*binormal)
   for j in range(5):
    g=ren.scene.geoms[ren.scene.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([.47,.39,.25,1],dtype=np.float32));mj.mjv_connector(g,mj.mjtGeom.mjGEOM_CAPSULE,.00007,points[j],points[j+1]);ren.scene.ngeom+=1
def add_marks(t):
 for row in marks:
  if row[0]>t:break
  g=ren.scene.geoms[ren.scene.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_SPHERE,np.array([.00025]*3),np.r_[row[2:4],z+.00006],np.eye(3).ravel(),np.array([.025,.023,.019,1],dtype=np.float32));ren.scene.ngeom+=1
proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x590','-r','30','-i','-','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(P/(tag+'_operation.mp4'))],stdin=subprocess.PIPE)
phase_names=['Grip / settle','Press','Hold pressure','Drag','Lift','Air travel','Recover']
selected=[int(np.argmin(abs(a[:,0]-v))) for v in ([1.,2.9,4.6,6.6] if info['mode']!='hao' else [3.1,a[-1,0]*.35,a[-1,0]*.7,a[-1,0]-.1])]
shots=[]
for t in range(0,len(q),16):
 d.qpos[:]=q[t];mj.mj_forward(m,d);im=Image.new('RGB',(1280,590),'#eeeae0')
 for x,c in [(0,cam),(640,close)]:
  ren.update_scene(d,c);fibers();add_marks(t);im.paste(Image.fromarray(ren.render()),(x,0))
 dr=ImageDraw.Draw(im);dr.rectangle((0,0,1280,30),fill='#172936');phase=int(a[t,1]);dr.text((10,6),f'Free brush held by hand_111 | {phase_names[phase]} | {a[t,0]:.2f} s',fill='white',font=font);dr.text((650,6),'Elastic bundles + paper contact | real-time replay',fill='white',font=font)
 dr.text((12,491),f'Paper reaction: {a[t,10]:.4f} N   |   Hand on brush Z: {a[t,13]:.4f} N   |   Bristle deflection: {a[t,19]*1000:.2f} mm',fill='#152932',font=font)
 for j,label,color,x0,scale in [(10,'Paper normal force','#a04929',15,max(.02,a[:,10].max())),(19,'Bristle deflection','#235c81',650,max(.001,a[:,19].max()))]:
  dr.text((x0,516),label,fill=color,font=small)
  pts=[(x0+int(600*a[k,0]/a[-1,0]),574-int(35*a[k,j]/scale)) for k in range(0,t+1,4)]
  if len(pts)>1:dr.line(pts,fill=color,width=2)
  dr.line((x0,575,x0+600,575),fill='#b5b2a6')
 proc.stdin.write(np.asarray(im).tobytes())
 if any(abs(t-v)<8 for v in selected):shots.append(im.copy())
 if t==len(range(0,len(q),16))*16-16:im.save(P/(tag+'_last.jpg'))
proc.stdin.close();code=proc.wait();ren.close()
if code:raise RuntimeError('ffmpeg failed')
if shots:
 grid=Image.new('RGB',(1280,590*len(shots)));
 for k,im in enumerate(shots):grid.paste(im,(0,590*k))
 grid.save(P/(tag+'_preview.jpg'))
(P/(tag+'_replay_audit.json')).write_text(json.dumps({'max_geom_replay_error_m':max(errors),'frames':len(range(0,len(q),16))},indent=2))
print('rendered',tag,flush=True)

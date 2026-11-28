
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
P=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/brush14'));tag=sys.argv[1];D=np.load(P/(tag+'.npz'));a=D['a'];q=D['q'];ink=D['ink'];info=json.loads((P/(tag+'_model.json')).read_text());m=mj.MjModel.from_xml_path(str(P/(tag+'.xml')));d=mj.MjData(m);m.vis.global_.offwidth=1280;m.vis.global_.offheight=960;ren=mj.Renderer(m,960,1280,max_geom=50000);center=np.array(info['rest_center']);z=info['paper_z'];shaft=m.geom('shaft').id
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

import hashlib
curves=[np.array(v) for v in json.loads((P/(tag+'_curves.json')).read_text())]
allpts=np.concatenate(curves);mid=(allpts.min(axis=0)+allpts.max(axis=0))/2
paper_cam=mj.MjvCamera();paper_cam.lookat[:]=[*mid,z];paper_cam.distance=.078;paper_cam.azimuth=90;paper_cam.elevation=-89.9
full_cam=mj.MjvCamera();full_cam.lookat[:]=center+[.012,-.016,.017];full_cam.distance=.265;full_cam.azimuth=115;full_cam.elevation=-23
original_rgba=m.geom_rgba.copy();occluders=[i for i in range(m.ngeom) if m.geom(i).name.startswith('handgeom') or m.geom(i).name=='shaft']
font_big=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),21)
phase_names=['Grip / settle','Touch down','Hold','Write','Lift','Air travel','Finish']
output=P/(tag+'_paper_view.mp4')
proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x800','-r','30','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(output)],stdin=subprocess.PIPE)
# Keep static contact marks in the scene buffer, adding only newly visible marks.
# Each view uses the same number of visible geoms because alpha-zero geoms are retained
# only according to the renderer; reinitialize if its base offset changes.
for t in range(0,len(q),16):
 d.qpos[:]=q[t];mj.mj_forward(m,d);im=Image.new('RGB',(1280,800),'#e9e7df')
 m.geom_rgba[:]=original_rgba;m.geom_rgba[occluders,3]=0
 ren.update_scene(d,paper_cam);ren.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False;fibers();add_marks(t)
 im.paste(Image.fromarray(ren.render()).resize((960,720),Image.Resampling.LANCZOS),(0,40))
 m.geom_rgba[:]=original_rgba
 ren.update_scene(d,full_cam);ren.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=True;fibers();add_marks(t)
 im.paste(Image.fromarray(ren.render()).resize((320,240),Image.Resampling.LANCZOS),(960,40))
 dr=ImageDraw.Draw(im);dr.rectangle((0,0,1280,40),fill='#172936')
 dr.text((14,8),'PAPER TOP VIEW | hand + shaft hidden to reveal writing',font=font_big,fill='white')
 dr.text((974,8),'Full scene',font=font_big,fill='white')
 sid=int(a[t,27])+1;phase=int(a[t,1]);contact=a[t,10]>1e-4
 entries=[f'Time: {a[t,0]:.2f} s',f'Stroke: {max(0,sid)} / 6',phase_names[phase],('CONTACT / INK' if contact else 'AIR / NO INK'),f'Paper force: {a[t,10]:.3f} N','Same physical replay','Speed: 1x']
 for j,line in enumerate(entries):dr.text((977,307+38*j),line,font=font,fill='#145d50' if j==3 and contact else '#253640')
 dr.text((15,775),'Actual contact ink only | No simulation or acceptance results changed',font=font,fill='#253640')
 proc.stdin.write(np.asarray(im).tobytes())
 for at,name in [(4.5,'early'),(12.,'middle'),(21.,'late'),(29.9,'final')]:
  if abs(a[t,0]-at)<1/60:im.save(P/(tag+'_paper_view_'+name+'.jpg'))
proc.stdin.close();assert proc.wait()==0;ren.close()
(P/(tag+'_paper_view_audit.json')).write_text(json.dumps({'source_npz_sha256':hashlib.sha256((P/(tag+'.npz')).read_bytes()).hexdigest(),'source_xml_sha256':hashlib.sha256((P/(tag+'.xml')).read_bytes()).hexdigest(),'video_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'max_replay_geometry_error_m':max(errors),'main_view':'overhead, hand and shaft hidden, physical bristle deformation and contact ink retained','inset_view':'full scene, all objects visible','physical_trajectory_changed':False,'frames':len(range(0,len(q),16))},indent=2))
print(output,flush=True)

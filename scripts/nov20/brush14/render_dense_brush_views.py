
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
# Dense visual hairs interpolate the 13 physical guide bundles.
# They add neither contact bodies nor independent dynamics.
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline
rng=np.random.default_rng(14019)
root_xy=np.array([[0.,0.]]+[[.0035*(.5 if b<=6 else 1)*np.cos(2*np.pi*((b-1)%6)/6+(0 if b<=6 else np.pi/6)),.0035*(.5 if b<=6 else 1)*np.sin(2*np.pi*((b-1)%6)/6+(0 if b<=6 else np.pi/6))] for b in range(1,13)])
tri=Delaunay(root_xy);sample=[]
while len(sample)<900:
 x=rng.uniform(-.0035,.0035,2)
 if tri.find_simplex(x)>=0:sample.append(x)
sample=np.array(sample);si=tri.find_simplex(sample);bary=np.einsum('nij,nj->ni',tri.transform[si,:2],sample-tri.transform[si,2]);weights=np.c_[bary,1-bary.sum(1)];vertices=tri.simplices[si]
# Most hairs reach near the physical tips; shorter outer hairs soften the silhouette.
rad=np.linalg.norm(sample,axis=1)/.0035
lengths=1-rng.uniform(.0,.02,900)-.12*rad**2
lengths[::12]=1.0  # Keep full-length hairs at physical contact depth.
widths=rng.uniform(.000022,.000040,900)
shades=rng.uniform(.7,1.13,900)
hair_rgba=np.array([np.r_[np.array([.48,.31,.15])*shade,1.] for shade in shades],dtype=np.float32)
segments=9
hair_positions=None
def update_hairs():
 global hair_positions
 guides=[]
 for group in capids:
  nodes=[]
  for j,gid in enumerate(group):
   axis=d.geom_xmat[gid].reshape(3,3)[:,2];c=d.geom_xpos[gid];half=m.geom_size[gid,1];ends=[c-axis*half,c+axis*half];bp=d.xpos[m.geom_bodyid[gid]]
   if np.linalg.norm(ends[0]-bp)>np.linalg.norm(ends[1]-bp):ends.reverse()
   if j==0:nodes.append(ends[0])
   nodes.append(ends[1])
  guides.append(nodes)
 guide=np.array(guides)
 # Convex interpolation fills gaps while retaining the guide motion.
 nodes=np.einsum('nk,nkjc->njc',weights,guide[vertices])
 spline=CubicSpline(np.linspace(0,1,6),nodes,axis=1,bc_type='natural')
 # Evaluate each strand at its own length, using the corresponding spline column.
 ts=np.linspace(0,1,segments+1)
 uv=lengths[:,None]*ts[None,:];idx=np.minimum((uv*5).astype(int),4);dx=(uv-idx/5)[...,None];co=spline.c[:,idx,np.arange(900)[:,None],:];hair_positions=((co[0]*dx+co[1])*dx+co[2])*dx+co[3]
def fibers():
 for b,points in enumerate(hair_positions):
  for j in range(segments):
   u=(j+.5)/segments
   color=hair_rgba[b].copy()
   # Dark distal hairs distinguish the inked brush tip from the warm roots.
   color[:3]*=1-.68*max(0,(u-.65)/.35)
   radius=widths[b]*(1-.72*u**2)
   g=ren.scene.geoms[ren.scene.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_CAPSULE,np.zeros(3),np.zeros(3),np.eye(3).ravel(),color);mj.mjv_connector(g,mj.mjtGeom.mjGEOM_CAPSULE,radius,points[j],points[j+1]);g.shininess=.08;g.specular=.08;ren.scene.ngeom+=1
def add_marks(t):
 for row in marks:
  if row[0]>t:break
  g=ren.scene.geoms[ren.scene.ngeom];mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_SPHERE,np.array([.00025]*3),np.r_[row[2:4],z+.00006],np.eye(3).ravel(),np.array([.025,.023,.019,1],dtype=np.float32));ren.scene.ngeom+=1

import hashlib
preview='--preview' in sys.argv
original=m.geom_rgba.copy();hands=[i for i in range(m.ngeom) if m.geom(i).name.startswith('handgeom')]
curves=[np.array(v) for v in json.loads((P/(tag+'_curves.json')).read_text())];pts=np.concatenate(curves);mid=(pts.min(0)+pts.max(0))/2
full=mj.MjvCamera();full.azimuth=100;full.elevation=-18;full.distance=.28
macro=mj.MjvCamera();macro.azimuth=100;macro.elevation=-12;macro.distance=.065
paper=mj.MjvCamera();paper.lookat[:]=[*mid,z];paper.distance=.16;paper.azimuth=90;paper.elevation=-89.9
out=P/(tag+'_dense_brush_views.mp4')
if not preview:proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1440x900','-r','30','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(out)],stdin=subprocess.PIPE)
indices=[int(v*480) for v in (5,9,12)] if preview else range(0,len(q),16)
for t in indices:
 d.qpos[:]=q[t];mj.mj_forward(m,d);update_hairs()
 root=np.mean([d.geom_xpos[g[0]] for g in capids],axis=0)
 tip=np.mean([d.geom_xpos[g[-1]] for g in capids],axis=0)
 full.lookat[:]=d.geom_xpos[shaft]+[0,0,-.015]
 macro.lookat[:]=(root+tip)/2
 im=Image.new('RGB',(1440,900),'#e9e7df')
 def render(cam,box,hide=False):
  m.geom_rgba[:]=original
  if hide:m.geom_rgba[hands,3]=0
  ren.update_scene(d,cam);ren.scene.flags[mj.mjtRndFlag.mjRND_SHADOW]=False;fibers();add_marks(t)
  frame=Image.fromarray(ren.render())
  # Preserve proportions; crop to the panel aspect ratio.
  w,h=box[2:];ratio=w/h;fw,fh=frame.size
  if fw/fh>ratio:
   nw=int(fh*ratio);frame=frame.crop(((fw-nw)//2,0,(fw+nw)//2,fh))
  else:
   nh=int(fw/ratio);frame=frame.crop((0,(fh-nh)//2,fw,(fh+nh)//2))
  im.paste(frame.resize((w,h),Image.Resampling.LANCZOS),box[:2])
 render(full,(0,40,640,800))
 render(macro,(640,40,800,450),True)
 # Paper panel: hide shaft as well only in this panel.
 save=original[shaft,3];original[shaft,3]=0
 render(paper,(640,530,800,310),True);original[shaft,3]=save
 dr=ImageDraw.Draw(im)
 for box in [(0,0,1440,40),(640,490,1440,530)]:dr.rectangle(box,fill='#172936')
 dr.text((15,10),'FULL BRUSH + HAND',font=font,fill='white')
 dr.text((655,10),'BRISTLE CLOSE-UP | hand hidden, shaft retained',font=font,fill='white')
 dr.text((655,500),'PAPER / ACTUAL CONTACT INK | hand + shaft hidden',font=font,fill='white')
 dr.text((15,852),f'Time {a[t,0]:.2f} s    Paper force {a[t,10]:.3f} N    Same physics | 1x speed',font=font,fill='#253640')
 dr.text((15,877),'900 visual hairs driven by 13 physical bundles | Display refinement; same physics.',font=small,fill='#253640')
 if preview:im.save(P/(tag+f'_dense_brush_views_preview_{t//480}.jpg'))
 else:
  proc.stdin.write(np.asarray(im).tobytes())
  if abs(a[t,0]-9)<1/60:im.save(P/(tag+'_dense_brush_views_sample.jpg'))
if not preview:
 proc.stdin.close();assert proc.wait()==0
 (P/(tag+'_dense_brush_views_audit.json')).write_text(json.dumps({'video_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'source_npz_sha256':hashlib.sha256((P/(tag+'.npz')).read_bytes()).hexdigest(),'source_xml_sha256':hashlib.sha256((P/(tag+'.xml')).read_bytes()).hexdigest(),'max_replay_geometry_error_m':max(errors),'physical_trajectory_changed':False,'frames':len(indices),'visual_hairs':900,'physical_guide_bundles':13,'display_changes':'dense interpolated tapered hairs; same three views; no new hair dynamics'},indent=2))
ren.close();print('done',flush=True)

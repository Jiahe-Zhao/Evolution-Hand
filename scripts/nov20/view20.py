
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from PIL import Image,ImageDraw,ImageFont
import sys,subprocess
o=OUT/'scan20';z=np.load(o/'hand.npz');q=z['q'];a=z['a'];p=z['poses'];B=np.loadtxt(o/'board.txt')[0]
m=mj.MjModel.from_xml_path(str(o/'hand.xml'));d=mj.MjData(m);m.vis.global_.offwidth=960;m.vis.global_.offheight=600
for i in range(m.nlight):m.light_ambient[i]=[.52,.52,.52];m.light_castshadow[i]=0
m.vis.headlight.ambient[:]=.48;m.vis.headlight.diffuse[:]=.65
for i in range(m.ngeom):
 name=m.geom(i).name
 if name.startswith('handgeom'):m.geom_group[i]=5
 if name in ('housing','heel','bridge','grip','window','paddle','stem','guidepost'):
  m.geom_rgba[i,:3]=dict(housing=[.21,.67,.72],heel=[.11,.45,.54],bridge=[.16,.45,.49],grip=[.15,.32,.39],window=[.04,.14,.2],paddle=[1,.48,.03],stem=[.82,.54,.28],guidepost=[.17,.3,.34])[name]
sheet='sheet' in sys.argv;W,H=(320,240) if sheet else (800,540);ren=mj.Renderer(m,H,W);opt=mj.MjvOption();opt.geomgroup[5]=0;visible=mj.MjvOption();visible.geomgroup[5]=1
def pose(j):
 x=q[j];d.qpos[:27]=x[:27];d.qpos[27:34]=x[[27,28,29,33,30,31,32]];d.qpos[34]=x[34];mj.mj_forward(m,d)
def render(az,el,dist,hide):
 cam=mj.MjvCamera();cam.lookat[:]=.5*d.geom('heel').xpos+.35*d.geom('window').xpos+.15*d.geom('paddle').xpos;cam.distance=dist;cam.azimuth=az;cam.elevation=el;ren.update_scene(d,cam,scene_option=opt if hide else visible);return Image.fromarray(ren.render())
if sheet:
 pose(735);im=Image.new('RGB',(W*4,H*3));dr=ImageDraw.Draw(im)
 for i,az in enumerate([0,45,90,135,180,225,270,315,30,120,210,300]):
  x=i%4*W;y=i//4*H;el=-15 if i<8 else 18;im.paste(render(az,el,.19,False),(x,y));dr.text((x+8,y+6),str((az,el)),fill='white')
 im.save(o/'views.jpg');ren.close();sys.exit()
font=lambda n:ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),n)
e=np.flatnonzero(a[:,9])[0];out=o/'scanner_clear.mp4';enc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x640','-r','30','-i','-','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],stdin=subprocess.PIPE)
for j in range(0,len(q),16):
 pose(j);im=Image.new('RGB',(1600,640),'#182a35');im.paste(render(float(sys.argv[1]),float(sys.argv[2]),.175,False),(0,48));im.paste(render(float(sys.argv[3]),float(sys.argv[4]),.160,True),(800,48));dr=ImageDraw.Draw(im)
 dr.text((20,8),'Hand + scanner  |  index on trigger',font=font(24),fill='white');dr.text((820,8),'Scanner only  |  same instant, hand hidden',font=font(24),fill='white')
 dr.text((20,596),f'{a[j,0]:.2f} s     {a[j,1]:.1f} mm     {a[j,2]:.1f} deg',font=font(23),fill='white')
 dr.text((817,596),f'Trigger: {max(0,a[j,4]):.2f} mm    '+('READ  73513537' if j>=e else 'Waiting'),font=font(24),fill='#ffca80')
 if j==int(np.ceil(e/16)*16):im.save(o/'scanner_clear.jpg',quality=94)
 enc.stdin.write(im.tobytes())
enc.stdin.close();assert enc.wait()==0;ren.close()

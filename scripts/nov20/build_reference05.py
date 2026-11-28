
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from build import *
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from PIL import Image,ImageDraw,ImageFont
import json
o=OUT/'hammerstone_reference';o.mkdir(exist_ok=True)
# Continuous rounded, slightly asymmetric cobble; no narrow shaft or detached lobes.
N=96;M=48;verts=[];uv=[]
for j in range(M+1):
 t=np.pi*j/M
 for i in range(N+1):
  p=2*np.pi*i/N
  sp=lambda x:np.sign(x)*abs(x)**.88
  x=sp(np.sin(t)*np.cos(p));y=sp(np.sin(t)*np.sin(p));z=sp(np.cos(t))
  rad=1+.035*np.sin(3*p+.6)*np.sin(t)**2+.025*np.cos(5*t+2*p)
  verts.append([x*rad*(1+.055*z)+.035*z*z,y*rad*(1-.045*x),z*(1+.025*y)])
  uv.append([i/N,j/M])
v=np.array(verts);v=(v-(v.max(0)+v.min(0))/2)/np.ptp(v,axis=0)*[.064,.057,.045];faces=[]
for j in range(M):
 for i in range(N):
  a=j*(N+1)+i;b=a+N+1
  if j:faces.append([a,b,a+1])
  if j<M-1:faces.append([a+1,b,b+1])
with open(o/'hammerstone.obj','w') as f:
 for x in v:f.write('v '+' '.join(map(str,x))+'\n')
 for x in uv:f.write('vt '+' '.join(map(str,x))+'\n')
 for x in faces:f.write('f '+' '.join(f'{i+1}/{i+1}' for i in x)+'\n')
rng=np.random.default_rng(5);sz=(512,1024);noise=np.zeros(sz)
for sigma,amp in [(42,1),(12,.55),(3,.20),(.6,.08)]:
 z=gaussian_filter(rng.normal(size=sz),sigma,mode='wrap');noise+=z/z.std()*amp
base=np.array([.47,.42,.31]);rgb=base+noise[:,:,None]*np.array([.056,.047,.034]);rust=gaussian_filter(rng.normal(size=sz),15,mode='wrap');rust=rust/rust.std();rgb+=np.maximum(rust-1,0)[:,:,None]*np.array([.075,.010,-.022]);Image.fromarray((np.clip(rgb,0,1)*255).astype('uint8')).save(o/'stone_texture.png')
root=E.Element('mujoco',model='Majuangou_reference_cobble');E.SubElement(root,'compiler',angle='radian');asset=E.SubElement(root,'asset');E.SubElement(asset,'mesh',name='cobble',file=str(o/'hammerstone.obj'));E.SubElement(asset,'texture',name='skin',type='2d',file=str(o/'stone_texture.png'));E.SubElement(asset,'material',name='stone',texture='skin',specular='.04',shininess='.02')
w=E.SubElement(root,'worldbody');E.SubElement(w,'light',pos='-.1 -.15 .3',diffuse='.85 .85 .85',ambient='.25 .25 .25');ob=E.SubElement(w,'body',name='hammerstone',pos='0 0 .030');E.SubElement(ob,'freejoint');mass=ConvexHull(v).volume*2600;E.SubElement(ob,'geom',name='hammerstone',type='mesh',mesh='cobble',material='stone',mass=str(mass),friction='.8 .005 .0001')
E.ElementTree(root).write(o/'asset.xml')
m=mj.MjModel.from_xml_path(str(o/'asset.xml'));d=mj.MjData(m);mj.mj_forward(m,d);m.vis.global_.offwidth=1200;m.vis.global_.offheight=800;r=mj.Renderer(m,500,550);canvas=Image.new('RGB',(1650,600),'#222931');font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),22)
for i,(az,el) in enumerate([(90,-80),(90,-13),(5,-18)]):
 cam=mj.MjvCamera();cam.lookat[:]=[0,0,.03];cam.distance=.165;cam.azimuth=az;cam.elevation=el;r.update_scene(d,camera=cam);canvas.paste(Image.fromarray(r.render()),(550*i,50))
dr=ImageDraw.Draw(canvas);dr.text((20,12),'05 | Single-piece cobble | Reference-inspired reconstruction, not a scan',font=font);dr.text((20,561),f'64 x 57 x 45 mm | density 2600 kg/m3 | mass {mass*1000:.0f} g | Geometry / grasp acceptance pending',font=font);canvas.save(o/'preview.jpg');r.close()
note=f"""## 05 考古实物参考重建（2026-11-18）
参考 Smithsonian 马圈沟锤石，约 166 万年前：
https://humanorigins.si.edu/evidence/behavior/stone-tools/early-stone-age-tools/hammerstone-majuangou-china
从实物照片观察：整体浑圆、略不对称，无细柄或独立球头；参考该轮廓重建单体天然卵石。纹理为程序生成的石皮斑驳，不把色斑认定为敲击痕。不是扫描，不声称一比一复制。64×57×45 mm 为适配手型的设计尺寸；密度假设 2600 kg/m³，凸包体积折算质量 {mass*1000:.1f} g。
OBJ 可视网格和物理代理同一轮廓；MuJoCo/Newton 刚体碰撞按凸包处理。当前完成模型、资产加载与三视角预览；尚未重做该石型的握持和敲击验收，不继承旧锤石的通过指标。历史通过轨迹保留，05 新外形的复验待完成。不得继续以细柄球头作为本阶段考古外形。
资产 outputs/nov20/hammerstone_reference/asset.xml、hammerstone.obj、stone_texture.png、preview.jpg；脚本 scripts/nov20/build_reference05.py。
"""
(o/'MODEL.md').write_text(note);print(note)

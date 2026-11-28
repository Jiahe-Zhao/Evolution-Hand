
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
from PIL import Image,ImageDraw,ImageFont
import json,newton
from scipy.ndimage import gaussian_filter
o=OUT/'hafted05';o.mkdir(exist_ok=True)
root=E.Element('mujoco',model='Grooved_stone_short_wooden_handle');E.SubElement(root,'compiler',angle='radian');assets=E.SubElement(root,'asset')
w=E.SubElement(root,'worldbody');E.SubElement(w,'light',pos='.1 -.25 .3',diffuse='.9 .9 .9',ambient='.8 .8 .8')
body=E.SubElement(w,'body',name='hafted_hammer',pos='0 0 .18');E.SubElement(body,'freejoint',name='hammer_free')
rng=np.random.default_rng(85)
for name in ['stone','wood']:
 shape=(512,512);noise=np.zeros(shape)
 for sigma,amp in [(24,.42),(6,.24),(1,.09)]:
  f=gaussian_filter(rng.normal(size=shape),sigma);noise+=f/f.std()*amp
 if name=='wood':
  yy,xx=np.mgrid[:512,:512];grain=np.sin(xx*.29+3*np.sin(yy*.011)+.5*noise)
  grain+=.5*np.sin(xx*.89+np.sin(yy*.02))
  noise=.18*noise+.20*grain
  color=np.array([.43,.28,.15])+noise[:,:,None]*np.array([.22,.17,.10])
 else:
  color=np.array([.51,.48,.40])+noise[:,:,None]*np.array([.18,.16,.13])
 Image.fromarray((np.clip(color,0,1)*255).astype('uint8')).save(o/(name+'.png'))
 E.SubElement(assets,'texture',name=name+'_tex',type='2d',file=str(o/(name+'.png')))
 E.SubElement(assets,'material',name=name,texture=name+'_tex',specular='.01',shininess='.01')
def mesh(n,v,mass,col,uv):
 E.SubElement(assets,'mesh',name=n,vertex=' '.join(map(str,np.array(v).ravel())),texcoord=' '.join(map(str,np.array(uv).ravel())))
 E.SubElement(body,'geom',name=n,type='mesh',mesh=n,mass=str(mass),material='wood' if n.startswith('wood') else 'stone',friction='.8 .004 .0001')
# Head is an oval cobble, with a circumferential groove centred on x=0.
# Each axial band has its own convex collider; a single convex hull would fill the groove.
xs=np.array([-.042,-.037,-.026,-.012,-.006,-.004,.004,.006,.012,.026,.037,.042])
rads=np.array([.006,.016,.022,.024,.023,.0195,.0195,.023,.024,.022,.016,.006])
totalmass=0
from scipy.spatial import ConvexHull
rings=[]
for x,rr in zip(xs,rads):
 ring=[]
 for k in range(25):
  a=k*2*np.pi/24
  rad=rr*(1+.10*np.sin(3*a+26*x)+.055*np.cos(5*a-37*x)+.025*np.sin(9*a+75*x))
  # a few broad uneven facets, continuous across segment boundaries
  rad=min(rad,rr*(1.00+.13*np.cos(a+.4)))
  ring.append([x+.0011*np.sin(2*a+60*x),rad*np.cos(a)+.0011*np.sin(x*55),rad*.90*np.sin(a)])
 rings.append(ring)
for j in range(len(xs)-1):
 v=rings[j]+rings[j+1]
 mass=ConvexHull(v).volume*2600;totalmass+=mass
 uv=[[k/24,(x+.045)/.090] for x in xs[j:j+2] for k in range(25)]
 mesh('stone_band'+str(j),v,mass,'',uv)
# subtly curved branch, angular shaved faces, irregular thickness
zs=np.linspace(-.137,-.010,10)
wr=[]
for j,z in enumerate(zs):
 rr=.0088*(1+.09*np.sin(j*1.7))
 if j==0:rr*=.78
 center=[.0025*np.sin(np.pi*j/9),.0012*np.sin(j*.6)]
 ring=[]
 for k in range(17):
  a=k*2*np.pi/16;rad=rr*(1+.09*np.sin(3*a+j*.25)+.035*np.sin(7*a))
  ring.append([center[0]+rad*np.cos(a),center[1]+rad*np.sin(a),z+.00065*np.sin(2*a+j)])
 wr.append(ring)
for j in range(9):
 mesh('wood_grip'+str(j),wr[j]+wr[j+1],.030/9,'',[[k/16,h/9] for h in [j,j+1] for k in range(17)])
# Continuous visible skins avoid the seams of the convex collision decomposition.
for g in body.findall('geom'):g.set('rgba','0 0 0 0')
def skin(name,rs,mat):
 nv=len(rs[0]);vs=np.array(rs);faces=[]
 for j in range(len(rs)-1):
  for k in range(nv-1):
   a=j*nv+k;b=a+nv
   faces.extend([[a,a+1,b],[a+1,b+1,b]])
 faces.extend([[0,k+1,k] for k in range(1,nv-2)])
 last=(len(rs)-1)*nv;faces.extend([[last,last+k,last+k+1] for k in range(1,nv-2)])
 file=o/(name+'.obj')
 with open(file,'w') as f:
  for v in vs.reshape(-1,3):f.write('v '+' '.join(map(str,v))+'\n')
  for j in range(len(rs)):
   for k in range(nv):f.write(f'vt {k/(nv-1)} {j/(len(rs)-1)}\n')
  for face in faces:f.write('f '+' '.join(f'{i+1}/{i+1}' for i in face)+'\n')
 E.SubElement(assets,'mesh',name=name,file=str(file))
 E.SubElement(body,'geom',name=name,type='mesh',mesh=name,material=mat,mass='.000001',contype='0',conaffinity='0')
skin('stone_surface',rings,'stone');skin('wood_surface',wr,'wood')
p=o/'asset.xml';E.ElementTree(root).write(p)
# Load the same fixed composite into Newton: head and wood share one free body.
b=newton.ModelBuilder();b.add_mjcf(str(p),collapse_fixed_joints=True);model=b.finalize(device='cpu');print('Newton shapes',model.shape_count,'bodies',model.body_count)
# integration asset for existing current-hand scene, object initially on a raised stand
r=_evo_parse_xml(OUT/'carry.xml').getroot();r.remove(r.find('contact'));wb=r.find('worldbody');wb.remove(wb.find("body[@name='item']"))
for asset in list(assets):r.find('asset').append(asset)
import copy
placed=copy.deepcopy(body);placed.set('pos','-.19 .035 .16');wb.append(placed)
E.SubElement(wb,'geom',name='display_stand',type='box',pos='-.19 .035 .012',size='.055 .035 .012',rgba='.27 .30 .32 1')
E.ElementTree(r).write(o/'hand_scene.xml')
m=mj.MjModel.from_xml_path(str(p));d=mj.MjData(m);mj.mj_forward(m,d);m.vis.global_.offwidth=700;m.vis.global_.offheight=700;ren=mj.Renderer(m,600,540);im=Image.new('RGB',(1620,680),'#222a30');font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/DejaVuSans.ttf'),22)
for i,(az,el) in enumerate([(-90,-8),(-30,-18),(0,-7)]):
 cam=mj.MjvCamera();cam.lookat[:]=[0,0,.126];cam.distance=.34;cam.azimuth=az;cam.elevation=el;ren.update_scene(d,camera=cam);im.paste(Image.fromarray(ren.render()),(i*540,40))
draw=ImageDraw.Draw(im);draw.text((20,9),'05 | Rough stone head + irregular wooden grip',font=font);draw.text((20,647),f'Nominal head 84 x 48 x 43 mm | irregular grip ~18 mm | height ~159 mm | {1000*(totalmass+.030):.0f} g',font=font);im.save(o/'preview.jpg');ren.close()
msg=f"""## 05 粗糙石头与木柄外观修订（2026-11-18）
保留无绑缚的石头＋短木柄。石头增加不对称轮廓、大小不一的浅削面和粗颗粒石皮，中央槽缘也不再完全规整；木柄改为略弯的木枝轮廓，粗细不均、端部不齐，并加入纵向木纹与削面。程序纹理只表达外观，不作为文物年代或制造痕迹的证据。名义尺寸沿用原设计，局部轮廓有毫米级改变。
实际碰撞代理也采用更新后的分段凸几何，而不是只改贴图。石木固定连接未变，组合质量约{1000*(totalmass+.030):.1f} g。Newton资产加载通过（{model.body_count}刚体、{model.shape_count}碰撞体），手操作仍待复验，不能沿用历史冲击结果。
outputs/nov20/hafted05/asset.xml、hand_scene.xml、stone.png、wood.png、preview.jpg；生成脚本 scripts/nov20/build_hafted05.py。
"""
(o/'MODEL.md').write_text(msg);print(msg)


# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from build import *
from scipy.spatial.transform import Rotation
import os
d=OUT/'pen140';d.mkdir(exist_ok=True);D=np.load(OUT/'pen14/warm.npz');q=D['q'][:34].copy();p=q[27:30].copy();rot=Rotation.from_quat(q[30:34]);fmt=lambda v:' '.join(map(str,v))
r=_evo_parse_xml(OUT/'pen14/write.xml').getroot();w=r.find('worldbody');ob=w.find("body[@name='pen']");c=r.find('contact')
for g in ob.findall('geom'):ob.remove(g)
for g in list(c):
 if 'paper' not in g.attrib.values() or not any('handgeom' in x for x in g.attrib.values()):c.remove(g)
# axis chosen relative to the pinching fingertips
old=mj.MjModel.from_xml_path(str(OUT/'pen14/write.xml'));dd=mj.MjData(old);dd.qpos[:27]=q[:27];mj.mj_forward(old,dd)
v1=dd.geom('handgeom20').xpos;v2=dd.geom('handgeom40').xpos
print('tips',v1,v2,'origin',p)
axis=np.array([float(os.getenv('AX','-.4')),float(os.getenv('AY','.35')),1.]);axis/=np.linalg.norm(axis)
center=(v1+v2)/2
def loc(t):return rot.inv().apply(center-p+axis*t)
def geom(name,typ,**kw):return E.SubElement(ob,'geom',name=name,type=typ,**kw)
for n,(lo,hi,mass,col) in enumerate([(-.023,0,.0035,'.12 .39 .54 1'),(0,.110,.003,'.19 .44 .66 1')]):
 geom(['front','rear'][n],'cylinder',fromto=fmt(np.r_[loc(lo),loc(hi)]),size='.0045',mass=str(mass),rgba=col)
geom('nib','capsule',fromto=fmt(np.r_[loc(-.027),loc(-.023)]),size='.0018',mass='.0004',rgba='.55 .55 .55 1')
geom('tip','sphere',pos=fmt(loc(-.029)),size='.001',mass='.0001',rgba='.06 .08 .13 1')
for g in ob.findall('geom'):
 for other in ['paper']+[x.get('name') for x in w.find('body').iter('geom')]:
  mu=.12 if other=='paper' else 1.1
  E.SubElement(c,'pair',geom1=g.get('name'),geom2=other,condim='6',friction=f'{mu} {mu} .003 .0004 .0004',solref='.04 1.4' if other=='paper' else '.005 1')
tip=center-.029*axis
print('offset',tip-p)

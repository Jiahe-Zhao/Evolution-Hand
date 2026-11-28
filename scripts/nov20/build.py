
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from pathlib import Path
from xml.etree import ElementTree as E
import numpy as np,mujoco as mj
ROOT=_EVOLUTION_ROOT
OUT=ROOT/'outputs/nov20'
def hand():
    src=ROOT/'outputs/v5_acceptance/hand_111/left/hand.urdf'
    u=_evo_parse_xml(src); assets={}
    for mesh in u.iter('mesh'):
        f=(src.parent/mesh.get('filename')).resolve();assets[f.name]=f.read_bytes();mesh.set('filename',f.name)
    ext=E.SubElement(u.getroot(),'mujoco');E.SubElement(ext,'compiler',fusestatic='false',discardvisual='true',balanceinertia='true')
    model=mj.MjModel.from_xml_string(E.tostring(u.getroot(),encoding='unicode'),assets)
    mj.mj_saveLastXML(str(OUT/'hand.xml'),model)
    tree=_evo_parse_xml(OUT/'hand.xml');r=tree.getroot()
    for a in r.iter('mesh'):a.set('file',str(src.parent.parent/'meshes'/Path(a.get('file')).name))
    return r
if __name__=='__main__':
 r=hand(); m=mj.MjModel.from_xml_string(E.tostring(r,encoding='unicode'));d=mj.MjData(m);mj.mj_forward(m,d)
 for i in range(m.nbody):print(m.body(i).name,np.round(d.xpos[i],3))
 for j in range(m.njnt):print(m.joint(j).name,m.jnt_range[j])
def scene(task):
 r=hand(); w=r.find('worldbody'); root=w.find('body');root.set('pos','0 0 .13');root.set('quat','.70710678 .70710678 0 0')
 if task in ('stone','phone'):root.set('quat','.70710678 -.70710678 0 0')
 if task=='handle':root.set('pos','0 0 .044')
 for i,g in enumerate(root.iter('geom')):g.set('name','handgeom'+str(i));g.set('contype','1');g.set('conaffinity','2')
 def geom(b,name,typ,size,pos,color,**kw):
    return E.SubElement(b,'geom',name=name,type=typ,size=size,pos=pos,rgba=color,friction='1 .01 .001',contype='2',conaffinity='3',**kw)
 geom(w,'floor','plane','1 1 .01','0 0 0','.35 .39 .42 1')
 places={'carry':'.004 -.018 .147','stone':'-.076 .021 .151','phone':'-.096 .021 .138','handle':'-.176 -.022 .026'}
 E.SubElement(w,'light',pos='0 0 .55',dir='0 0 -1',diffuse='.8 .8 .8',ambient='.4 .4 .4')
 obj=E.SubElement(w,'body',name='item',pos=places[task]);E.SubElement(obj,'freejoint',name='item_free')
 if task=='carry':geom(obj,'fruit','ellipsoid','.017 .014 .012','0 0 0','.9 .35 .12 1',mass='.03')
 if task=='stone':
    mesh=E.SubElement(r.find('asset'),'mesh',name='stone_mesh',vertex='-.022 -.013 -.011  .017 -.015 -.011  .021 .012 -.011  -.016 .015 -.011  -.014 -.01 .015  .011 -.009 .012  .013 .009 .011  -.014 .01 .015')
    geom(obj,'core','mesh','.01','0 0 0','.63 .37 .18 1',mesh='stone_mesh',mass='.035')
 if task=='phone':
    geom(obj,'phone','box','.032 .060 .003','0 0 0','.12 .25 .35 1',mass='.085')
    for n,y in enumerate([-.039,-.053]):E.SubElement(obj,'site',name='target'+str(n),pos=f'{[.025,.025][n]} {y} .0032',type='cylinder',size='.006 .0002',rgba='1 .55 .14 1')
 if task=='handle':
    geom(obj,'base','box','.016 .025 .002','0 0 -.023','.22 .47 .55 1',mass='.04')
    for v in (-1,1):
     geom(obj,'end'+str(v),'box','.002 .025 .022',f'{v*.014} 0 .001','.22 .47 .55 1',mass='.01')
     geom(obj,'side'+str(v),'box','.012 .002 .022',f'0 {v*.023} .001','.22 .47 .55 1',mass='.01')
    h=E.SubElement(obj,'body',name='bail',pos='0 0 .026');E.SubElement(h,'joint',name='hinge',axis='0 -1 0',range='0 2.7',damping='.0008')
    for i,y in enumerate([-.028,.028]):geom(h,'arm'+str(i),'box','.032 .002 .002','.032 '+str(y)+' 0','.9 .57 .18 1',mass='.003')
    geom(h,'grip','box','.003 .030 .003','.064 0 0','.9 .57 .18 1',mass='.005')
 contact=E.SubElement(r,'contact')
 for g in root.iter('geom'):
    for o in obj.iter('geom'):E.SubElement(contact,'pair',geom1=g.get('name'),geom2=o.get('name'),friction='0.9 0.9 .004 .0001 .0001')
 act=E.SubElement(r,'actuator')
 for joint in root.iter('joint'):
    name=joint.get('name'); slide='diagnostic_wrist' in name; E.SubElement(act,'position',name='servo_'+name,joint=name,kp='2500' if slide else '2.5',kv='50' if slide else '.06')
 p=OUT/(task+'.xml');E.ElementTree(r).write(p);return p

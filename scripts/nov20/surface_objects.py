"""Asset variants for selective pinch and contact-qualified scraping (SI units)."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap, parse_scene_xml as _evo_parse_xml
_evo_bootstrap()
from xml.etree import ElementTree as E
from pathlib import Path
import numpy as np
from build import OUT
def geom(b,name,typ,pos,size,color,**attrs):
    return E.SubElement(b,'geom',name=name,type=typ,pos=' '.join(map(str,pos)),size=size,rgba=color,friction='.85 .005 .0001',contype='2',conaffinity='3',**attrs)
def scene(task):
    from build import scene as original
    base=OUT/'carry.xml'
    r=_evo_parse_xml(base if base.exists() else original('carry')).getroot();w=r.find('worldbody');obj=w.find("body[@name='item']");w.remove(obj);r.remove(r.find('contact'))
    hand=w.find('body');hand.set('pos','0 0 .091')
    if task=='pebble':
        geom(w,'tray','box',(-.095,0,.024),'.059 .047 .003','.27 .34 .37 1')
        for k in (-1,1):
            geom(w,'rim_x'+str(k),'box',(-.095+k*.058,0,.030),'.001 .047 .003','.24 .30 .31 1')
            geom(w,'rim_y'+str(k),'box',(-.095,k*.046,.030),'.057 .001 .003','.24 .30 .31 1')
        points=np.array([[-1,-.45,-.4],[-.3,-1,-.52],[.65,-.78,-.55],[1,.2,-.34],[.32,.93,-.4],[-.85,.72,-.45],[-.73,-.32,.42],[.15,-.58,.8],[.73,.12,.49],[-.20,.7,.64],[.0,.05,-.72]])
        for i,(x,y,scale,angle,col) in enumerate([(-.093,.003,1,.2,'.64 .44 .24 1'),(-.111,.021,.86,1,'.43 .47 .43 1'),(-.075,-.016,.78,-.6,'.46 .41 .38 1')]):
            verts=points*(1 if i==0 else np.random.default_rng(83+i).uniform(.83,1.12,points.shape))*np.array([.010,.008,.008])*scale
            verts[:6,2]=-.0042*scale;verts[-1,2]=-.0040*scale
            E.SubElement(r.find('asset'),'mesh',name='peb'+str(i),vertex=' '.join(map(str,verts.ravel())))
            body=E.SubElement(w,'body',name='item' if i==0 else 'neighbor'+str(i),pos=f'{x} {y} {.027+.0042*scale+.00015}',euler=f'0 0 {angle}');E.SubElement(body,'freejoint',name='free'+str(i))
            geom(body,'stone'+str(i),'mesh',(0,0,0),'.01',col,mesh='peb'+str(i),mass=str(.0021*scale**3))
            geom(body,'foot'+str(i),'box',(0,0,-.0041*scale),f'{.0045*scale} {.0035*scale} {.00045*scale}',col,mass='.00001')
    else:
        geom(w,'workpiece','box',(-.107,.01,.025),'.047 .028 .006','.49 .35 .23 1')
        # Flush tiles form a single bonded, instrumented layer. Never loose boxes.
        for row in range(3):
            for col in range(8):
                geom(w,f'coat_{row}_{col}','box',(-.135+col*.008,-.002+row*.010,.0316),'.004 .005 .0006',f'{.72+.04*np.sin(col+2*row):.3f} {.52+.025*np.sin(col+row):.3f} .32 1')
        tool=E.SubElement(w,'body',name='item',pos='-.080 -.042 .040',euler='0 0 .2');E.SubElement(tool,'freejoint',name='free0')
        geom(tool,'grip','ellipsoid',(0,0,0),'.024 .009 .007','.31 .33 .32 1',mass='.016')
        E.SubElement(r.find('asset'),'mesh',name='shoulder_mesh',vertex='-.025 -.009 -.003  -.025 .009 -.003  -.015 -.007 -.004  -.015 .007 -.004  -.013 -.006 .003  -.013 .006 .003')
        geom(tool,'shoulder','mesh',(0,0,0),'.01','.39 .41 .40 1',mesh='shoulder_mesh',mass='.002')
        geom(tool,'edge','box',(-.025,0,-.003),'.002 .009 .0015','.62 .65 .64 1',mass='.002')
        geom(w,'rest','box',(-.080,-.042,.018),'.027 .010 .015','.25 .31 .34 1')
    c=E.SubElement(r,'contact')
    if task=='pebble':
        for i in range(3):
            E.SubElement(c,'pair',geom1='foot'+str(i),geom2='tray',solref='.009 1',friction='1 1 .003 .0001 .0001')
            for base in ['tray','rim_x-1','rim_x1','rim_y-1','rim_y1']+['stone'+str(j) for j in range(i)]:
                E.SubElement(c,'pair',geom1='stone'+str(i),geom2=base,solref='.01 1',friction='.9 .9 .003 .0001 .0001')
    for g in hand.iter('geom'):
        for o in w.findall("body[@name='item']/geom"):E.SubElement(c,'pair',geom1=g.get('name'),geom2=o.get('name'),friction='.9 .9 .004 .0001 .0001')
    p=OUT/(task+'.xml');E.ElementTree(r).write(p);return p

class Score:
    def __init__(self,task):
        self.task=task;self.count=0;self.best=0;self.tiles=set();self.travel={};self.old=None;self.max_neighbor=0.
    def update(self,mm,dd,x,init,ids,time):
        from scipy.spatial.transform import Rotation
        import mujoco as mj
        obj=ids['item'];thumb=index=other=surface=grip=False;force=0.;candidates=[]
        for k,con in enumerate(dd.contact):
            if con.dist>0:continue
            from re import sub
            ge=[sub(r'_\d+$','',mm.geom(g).name.split('/')[-1]) for g in con.geom]
            bo=[mm.geom(g).name.split('/')[-2] for g in con.geom]
            if 'item' not in bo:continue
            j=bo.index('item');opp=bo[1-j];src=ge[j];dest=ge[1-j]
            thumb |= opp=='link_1_3';index |= opp=='link_2_3'
            other |= opp.startswith('link_') and opp not in ('link_1_3','link_2_3')
            surface |= dest.startswith(('tray','rim','stone1','stone2','floor'))
            grip |= src=='grip' and opp.startswith('link_')
            if src=='edge' and dest.startswith('coat_'):
                f=np.zeros(6);mj.mj_contactForce(mm,dd,k,f);force+=max(0.,f[0]);candidates.append(dest)
        rot=Rotation.from_quat(x[obj,3:]);p=x[obj,:3]+rot.apply([-.025,0,-.003]);v=0. if self.old is None else np.linalg.norm(p[:2]-self.old[:2])*240;self.old=p
        if self.task=='pebble':
            d=max(np.linalg.norm(x[ids[n],:3]-init[ids[n],:3]) for n in ('neighbor1','neighbor2'));self.max_neighbor=max(d,self.max_neighbor)
            lift=x[obj,2]-init[obj,2];ok=thumb and index and not other and not surface and lift>=.020 and self.max_neighbor<=.005
            metrics=dict(lift_mm=float(lift*1000),neighbor_max_mm=float(self.max_neighbor*1000),thumb=bool(thumb),index=bool(index))
        else:
            ang=np.rad2deg(np.arcsin(np.clip(rot.apply([1,0,0])[2],-1,1)))
            if grip and .3<=force<=3 and abs(ang-30)<=15 and .005<=v<=.20:
                for cell in set(candidates):
                    self.travel[cell]=self.travel.get(cell,0.)+v/240
                    if self.travel[cell]>=.003:self.tiles.add(cell)
            ok=len(self.tiles)>=20;metrics=dict(coverage=len(self.tiles)/24,force_N=float(force),angle_deg=float(ang),speed_m_s=float(v),held=bool(grip))
        self.count=self.count+1 if ok else 0;self.best=max(self.best,self.count)
        return dict(**metrics,success=self.best>=(72 if self.task=='pebble' else 1))

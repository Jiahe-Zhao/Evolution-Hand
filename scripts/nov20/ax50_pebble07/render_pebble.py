"""Render only saved pebble-07 poses; never runs or interpolates physics."""
from pathlib import Path
import csv,json,sys,os
import numpy as np
ROOT=Path('/home/zjh/Evolution_PC');OUT=ROOT/'outputs/nov20/ax50_pebble07'
PKG=ROOT/'outputs/ax50_visualskin_v3_0_0'
sys.path[:0]=[str(PKG),str(PKG/'tools'),str(PKG/'tools/smooth_render')]
import mujoco as mj
from ax50_visualskin.visual import VisualSkin
from smooth_render import panel
from render_core import encode,strip,FONTS
from PIL import Image,ImageDraw,ImageFont
W,H=600,448
SIDE=os.environ.get('AX50_SIDE','left')
BOX_F=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]

def rot(q):
    x,y,z,w=np.asarray(q,float).T
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))
def box(s):
    x,y,z=s
    return np.array([[-x if i<4 else x,-y if (i//2)%2==0 else y,-z if i%2==0 else z] for i in range(8)]),np.array(BOX_F)
def basis(v):
    zz=np.asarray(v,float);zz/=np.linalg.norm(zz);xx=np.cross([0,0,1],zz);xx/=np.linalg.norm(xx)
    return np.array([xx,-np.cross(zz,xx),zz]).T
def camera(points,v,margin=.82):
    b=basis(v);p=np.asarray(points);c=(p.min(0)+p.max(0))/2;d=p-c
    s=min(margin*(W/2)/max(abs(d@b[:,0]).max(),1e-6),margin*(H/2)/max(abs(d@b[:,1]).max(),1e-6))
    return b,s,np.array([W/2,H/2,0.])-c@b*s
def load_object():
    m=mj.MjModel.from_xml_path(str(OUT/'object.xml'));out=[];fixed=[]
    for g in range(m.ngeom):
        typ=m.geom_type[g];name=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or ''
        if name=='floor' or typ==mj.mjtGeom.mjGEOM_PLANE:continue
        if typ==mj.mjtGeom.mjGEOM_BOX:v,f=box(m.geom_size[g])
        elif typ==mj.mjtGeom.mjGEOM_MESH:
            k=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]];f=m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]]
        else:continue
        bn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
        p=np.array(m.geom_pos[g]);R=rot(m.geom_quat[g][[1,2,3,0]])
        d=(np.asarray(v,float),np.asarray(f,int),tuple(int(x*255) for x in m.geom_rgba[g,:3]),name)
        if bn=='world':fixed.append((np.asarray(v,float)@R.T+p,f,d[2],name))
        else:out.append(dict(body=bn,geom=d,v=np.asarray(v,float),p=p,R=R))
    return out,fixed

def trace(tag,t,z):
    out={};p=OUT/(tag+'_trace.csv')
    if not p.exists():return out
    with p.open(newline='') as f:rows=list(csv.DictReader(f))
    for k in ['phase','thumb_terminal_N','index_terminal_N','other_hand_item_N','lift_mm','neighbor1_displacement_mm','neighbor2_displacement_mm']:
        if k in z.files:continue
        if k not in rows[0]:continue
        vals=np.array([r[k] for r in rows]);rt=np.array([float(r['time_s']) for r in rows])
        out[k]=np.array([vals[np.argmin(abs(rt-x))] for x in t])
    return out

def setup_font():
    for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc']:
        if not Path(p).exists():continue
        try:
            f=ImageFont.truetype(p,16)
            for size in list(FONTS):FONTS[size]=ImageFont.truetype(p,size)
            return p
        except OSError:pass
    return None

def render(tag='pinch_trial_v1'):
    z=np.load(OUT/(tag+'.npz'));t=np.asarray(z['time'],float).ravel();bq=np.asarray(z['body_q'],float)
    names=[str(n) for n in z['body_names']];idx={n.rsplit('/',1)[-1]:i for i,n in enumerate(names)}
    phases=np.asarray(z['phase']) if 'phase' in z.files else None
    extra=trace(tag,t,z)
    def arr(k):
        if k in z.files:return np.asarray(z[k]).ravel()
        return np.asarray(extra.get(k,np.full(len(t),np.nan)))
    phases=phases if phases is not None else extra.get('phase',np.array(['recorded']*len(t)))
    thumb,index,other=arr('thumb_terminal_N'),arr('index_terminal_N'),arr('other_hand_item_N')
    geoms,fixed=load_object();skin=VisualSkin(SIDE,names,device='cpu')
    tri=np.asarray(z['tri']);parts=z['particles'] if 'particles' in z.files else None
    if parts is not None:tri=tri[(tri>=0).all(1)&(tri<parts.shape[1]).all(1)]
    def poses(i):
        R=rot(bq[i,:,3:]);p=bq[i,:,:3];out=list(fixed)
        for g in geoms:
            j=idx[g['body']];M=R[j]@g['R'];out.append((g['v']@M.T+p[j]+R[j]@g['p'],g['geom'][1],g['geom'][2],g['geom'][3]))
        return out,p,R
    valid=np.isfinite(bq).all((1,2))
    if not valid.any():raise RuntimeError('No finite recorded poses')
    valid_ids=np.flatnonzero(valid)
    sample=valid_ids[np.unique(np.linspace(0,len(valid_ids)-1,min(len(valid_ids),40),dtype=int))];near=[];allpts=[]
    for i in sample:
        _,p,R=poses(i);v,_=skin.update(bq[i]);allpts.extend(v[::max(1,len(v)//120)])
        for n in ['item','neighbor1','neighbor2']:
            j=idx[n];near.append(p[j]);allpts.append(p[j])
        if n=='item':near.extend([p[j]+R[j]@np.array([.018,0,0]),p[j]+R[j]@np.array([0,.018,0])])
        # Close-up volume is bounded to target and fingertips via recorded skin vertices.
        ti=p[idx['item']];near.extend(v[np.linalg.norm(v-ti,axis=1)<.035])
    near=np.asarray(near);allpts=np.asarray(allpts)
    cams=[camera(allpts,[-.65,-1,.48]),camera(near,[-.15,-1,.20],.88),camera(near,[-1,-.08,.16],.88)]
    if tag in ['pinch_trial_v5','pinch_trial_v6','pinch_trial_v7','pinch_trial_v8']:cams=[cams[0],camera(near,[1,-.8,.3],.88),camera(near,[1,.08,.16],.88)]
    p0=bq[0,:,:3].copy();ii=idx['item'];n1,n2=idx['neighbor1'],idx['neighbor2']
    measured_lift=arr('lift_mm').astype(float)
    measured_neighbor=np.maximum.accumulate(np.maximum(arr('neighbor1_displacement_mm').astype(float),arr('neighbor2_displacement_mm').astype(float)))
    font=setup_font()
    def frame(i):
        obj,p,R=poses(i);v,f=skin.update(bq[i]);obj.append((v,f,(222,174,143),'AX50'))
        if parts is not None:obj.append((parts[i],tri,(235,187,156),'soft'))
        lift=(p[ii,2]-p0[ii,2])*1000;nd=max(np.linalg.norm(p[n]-p0[n]) for n in [n1,n2])*1000
        if np.isfinite(measured_lift[i]):lift=measured_lift[i]
        if np.isfinite(measured_neighbor[i]):nd=measured_neighbor[i]
        a=[f'实际阶段 {phases[i]}  时间 {t[i]:.2f} s',f'目标石抬升 {lift:.2f} mm  邻石最大位移 {nd:.2f} mm',f'拇指力 {thumb[i]:.5f} N  食指力 {index[i]:.5f} N  其余手-目标力 {other[i]:.5f} N']
        title='AX50 07 双指夹石：实际记录回放 2倍慢放  |  阶段试验未通过'
        if tag=='pinch_trial_v6':title='AX50 07 低速限力续测（预夹持状态起步） | 实际记录 2倍慢放 | 未通过'
        if tag in ['pinch_trial_v7','pinch_trial_v8']:title='AX50 07 位置夹紧后提起（预夹持起步） | 实际记录 2倍慢放 | 验收见记录'
        return strip([panel(obj,cams[0]),panel(obj,cams[1]),panel(obj,cams[2])],['整手与石群','目标石与双指近景','侧面'],[a,a,a],title)
    valid=np.isfinite(bq).all((1,2));times=t.copy();dt=np.median(np.diff(times)) if len(times)>1 else 1/100
    frames=[i for i in np.unique(np.searchsorted(times,np.arange(times[0],times[-1]+1e-12,1/30)).clip(0,len(times)-1)) if valid[i]]
    if not frames:raise RuntimeError('No finite recorded poses')
    frames_img=[frame(i) for i in frames]
    frames_img[len(frames_img)//2].save(OUT/(tag+'_preview.png'))
    if not valid.all():
        card=frames_img[-1].copy();draw=ImageDraw.Draw(card);draw.rectangle((0,0,card.width,65),fill=(50,20,20));font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',23);draw.text((12,12),f'{t[np.flatnonzero(~valid)[0]]:.2f} 秒出现非有限状态，仿真停止；此处停留在最后有限帧',fill='white',font=font);frames_img.extend([card]*15)
    encode(OUT/(tag+'.mp4'),15,frames_img)
    info=dict(tag=tag,source=str(OUT/(tag+'.npz')),frames=len(frames),fps=15,sampling='recorded frames selected at 30 Hz; no interpolation',font=str(font),passed=False,title='阶段试验未通过',invalid_frames=int((~valid).sum()))
    (OUT/(tag+'_render.json')).write_text(json.dumps(info,indent=2,ensure_ascii=False))
    print(json.dumps(info,indent=2,ensure_ascii=False))
if __name__=='__main__':render(sys.argv[1] if len(sys.argv)>1 else 'pinch_trial_v1')

"""Replay a recorded AX50 task-13 active bail (提梁) run; no physics is run.

Geometry: original outputs/nov20/ax50_handle13/object.xml, parsed on CPU by MuJoCo.
Poses: recorded {tag}.npz body_q (frame by frame, never interpolated or invented).
Soft surfaces: the saved particles/tri. Hand: AX50 VisualSkin.
Acceptance verdict: audit_handle.audit(tag), recomputed from the same recorded files.
"""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import sys,os,json
import numpy as np
ROOT=Path((str(_EVOLUTION_ROOT)));OUT=ROOT/'outputs/nov20/ax50_handle13'
PKG=ROOT/'outputs/ax50_visualskin_v3_0_0';sys.path[:0]=[str(PKG),str(PKG/'tools'),str(Path(__file__).resolve().parent)]
import ax50_visualskin,warp as wp,mujoco as mj
from ax50_visualskin.visual import VisualSkin
from smooth_render import panel
import render_core
from render_core import encode,strip,FONTS
from PIL import Image,ImageDraw,ImageFont
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_handle import audit as audit_tag,quat_to_matrix
W,H=600,448
SIDE=os.environ.get('AX50_SIDE','left')
BOX_F=[[0,1,3],[0,3,2],[4,6,7],[4,7,5],[0,4,5],[0,5,1],[2,3,7],[2,7,6],[0,2,6],[0,6,4],[1,5,7],[1,7,3]]

def box(sz):
    x,y,z=sz;v=np.array([[-x if i<4 else x,-y if (i//2)%2==0 else y,-z if i%2==0 else z] for i in range(8)],float)
    return v,np.array(BOX_F)
def camera_basis(view):
    zz=np.asarray(view,float);zz/=np.linalg.norm(zz);xx=np.cross([0,0,1],zz);xx/=np.linalg.norm(xx)
    return np.array([xx,-np.cross(zz,xx),zz]).T
def fit_camera(points,view,margin=.84):
    b=camera_basis(view);pts=np.asarray(points,float);c=(pts.min(0)+pts.max(0))/2;d=pts-c
    sx=max(float(np.max(np.abs(d@b[:,0]))),1e-6);sy=max(float(np.max(np.abs(d@b[:,1]))),1e-6)
    s=min(margin*(W/2)/sx,margin*(H/2)/sy)
    return b,s,np.array([W/2,H/2,0.])-c@b*s
def load_object():
    m=mj.MjModel.from_xml_path(str(OUT/'object.xml'));geoms=[]
    for g in range(m.ngeom):
        nm=mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g) or '';typ=m.geom_type[g]
        if nm=='floor' or typ==mj.mjtGeom.mjGEOM_PLANE:continue
        if typ==mj.mjtGeom.mjGEOM_BOX:v,f=box(m.geom_size[g])
        elif typ==mj.mjtGeom.mjGEOM_MESH:
            k=m.geom_dataid[g];v=np.array(m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]],float)
            f=np.array(m.mesh_face[m.mesh_faceadr[k]:m.mesh_faceadr[k]+m.mesh_facenum[k]],int)
        else:continue
        bn=mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
        geoms.append(dict(body=bn.rsplit('/',1)[-1],pl=np.array(m.geom_pos[g],float),Rl=quat_to_matrix(m.geom_quat[g][[1,2,3,0]]),
            v=np.asarray(v,float),f=np.asarray(f,int),col=tuple(int(round(c*255)) for c in m.geom_rgba[g,:3]),name=nm))
    return geoms,np.array(m.body_pos[mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'bail')],float),next(np.array(g['pl']) for g in geoms if g['name']=='grip')
def floor_quad(cx,cy,half=.09,col=(206,210,214)):
    return np.array([[-half,-half,0],[half,-half,0],[half,half,0],[-half,half,0]],float)+[cx,cy,0],np.array([[0,1,2],[0,2,3]]),col,'floor'
def bar(x,y,z0,z1,w=.0006,col=(30,30,30)):
    if z1<=z0+1e-6:return None
    v,f=box([w,w,(z1-z0)/2])
    return v+[x,y,(z0+z1)/2],f,col,'clearance'
def marker(p,sz=(.0012,.005,.0012),col=(198,40,40)):
    v,f=box(sz)
    return v+p,f,col,'hinge_axis'
CJK=[str(_EVOLUTION_ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc'),str(_EVOLUTION_ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc'),str(_EVOLUTION_ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc'),str(_EVOLUTION_ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc')]
ZH=dict(title='AX50 13 活动提梁：实测回放 2x慢放',v1='视角1：器身底部与地面净距',v2='视角2：提梁铰链',
    t1='时间 {t:.2f} s  提升 {lift:.1f} mm  底部净距 {gap:.1f} mm',t2='提梁转角 {hinge:.1f}°  器身倾斜 {tilt:.1f}°  阶段 {phase}',
    t3='手–器身 {body:.5f} N  手–提梁 {bail:.4f} N  子步 {q}/{sub}',h1='提梁转角 {hinge:.1f}°（要求 >=45°）',
    h2='器身倾斜 {tilt:.1f}°（要求 <=15°）',h3='本帧 {ok}  连续合格 {hold:.2f} s',yes='合格',no='未合格',pass_='验收通过',fail='验收未通过')
EN=dict(title='AX50 13 active bail: recorded replay 2x slow',v1='View 1: vessel bottom to floor',v2='View 2: bail hinge',
    t1='t {t:.2f} s  lift {lift:.1f} mm  bottom gap {gap:.1f} mm',t2='bail {hinge:.1f} deg  tilt {tilt:.1f} deg  phase {phase}',
    t3='hand-item {body:.5f} N  hand-bail {bail:.4f} N  substeps {q}/{sub}',h1='bail {hinge:.1f} deg (needs >=45)',
    h2='vessel tilt {tilt:.1f} deg (needs <=15)',h3='frame {ok}  hold {hold:.2f} s',yes='OK',no='not OK',pass_='PASS',fail='FAIL')
def glyph_mask(font,ch):
    im=Image.new('L',(72,72),0);ImageDraw.Draw(im).text((4,4),ch,font=font,fill=255);return np.asarray(im)
def font_ok(path,text):
    f=ImageFont.truetype(path,16);ref=glyph_mask(f,'\U000f0000')
    return all(not np.array_equal(glyph_mask(f,c),ref) for c in text)
def setup_text(sample):
    for p in CJK:
        if not Path(p).exists():continue
        try:
            if not font_ok(p,sample):continue
            for s in list(FONTS):FONTS[s]=ImageFont.truetype(p,s)
            from matplotlib import font_manager
            try:font_manager.fontManager.addfont(p);plt.rcParams['font.sans-serif']=[font_manager.FontProperties(fname=p).get_name(),'DejaVu Sans']
            except Exception:pass
            plt.rcParams['axes.unicode_minus']=False
            return ZH,p
        except OSError:continue
    return EN,None

def render(tag):
    z=np.load(OUT/(tag+'.npz'));t=np.asarray(z['time'],float).ravel();bq_all=np.asarray(z['body_q'],float)
    names=[x.decode() if isinstance(x,bytes) else str(x) for x in z['body_names']];sh=[n.rsplit('/',1)[-1] for n in names]
    idx={n:i for i,n in enumerate(sh)}
    res,s=audit_tag(tag,series=True)
    T,fontpath=setup_text(''.join(ZH.values()))
    geoms,hinge_local,grip_local=load_object()
    skin=VisualSkin(SIDE,names,device='cpu')
    tri=np.asarray(z['tri']);parts_all=np.asarray(z['particles']) if 'particles' in z.files else None
    if parts_all is not None:tri=tri[(tri>=0).all(axis=1)&(tri<parts_all.shape[1]).all(axis=1)]
    item=[g for g in geoms if g['body']=='item']
    def world(i):
        bq=bq_all[i];Rb=quat_to_matrix(bq[:,3:]);pb=bq[:,:3];out=[]
        for g in geoms:
            j=idx[g['body']];M=Rb[j]@g['Rl'];out.append((g['v']@M.T+(pb[j]+Rb[j]@g['pl']),g['f'],g['col'],g['name']))
        return out,pb,Rb
    def bottom(i,pb,Rb):
        lo=None
        for g in item:
            j=idx[g['body']];w=g['v']@(Rb[j]@g['Rl']).T+(pb[j]+Rb[j]@g['pl']);k=int(np.argmin(w[:,2]))
            if lo is None or w[k,2]<lo[2]:lo=(w[k,0],w[k,1],w[k,2])
        return lo
    valid=np.array([np.isfinite(bq_all[i]).all() for i in range(len(t))])
    if not valid.any():raise RuntimeError('no finite recorded poses')
    step=max(1,len(t)//60);sample=list(range(0,len(t),step))
    cpts=[];hpts=[];whole=[]
    for i in sample:
        if not valid[i]:continue
        _,pb,Rb=world(i)
        sv,_=skin.update(bq_all[i]);whole.extend(sv[::max(1,len(sv)//100)])
        for g in item:
            j=idx[g['body']];cpts+=list(g['v']@(Rb[j]@g['Rl']).T+(pb[j]+Rb[j]@g['pl']))
        cpts+=[[pb[idx['item'],0],pb[idx['item'],1],0.]]
        hinge=pb[idx['item']]+Rb[idx['item']]@hinge_local;grip=pb[idx['bail']]+Rb[idx['bail']]@grip_local
        hpts+=[hinge,grip,hinge+[.04,.04,.04],hinge-[.04,.04,.04]]
    cpts=np.array(cpts);floor=floor_quad(float(cpts[:,0].mean()),float(cpts[:,1].mean()),.05)
    cam_c=fit_camera(np.vstack([cpts,[[floor[0][:,0].min(),floor[0][:,1].min(),0],[floor[0][:,0].max(),floor[0][:,1].max(),0]]]),[-.85,-1,.18])
    cam_h=fit_camera(np.array(hpts),[-.30,-1,.10])
    cam_all=fit_camera(np.vstack([whole,cpts]),[-.5,-1,.45])
    gaps=[];hinges=[]
    def frame(i):
        obj,pb,Rb=world(i);lo=bottom(i,pb,Rb);gap=lo[2];gaps.append(gap)
        hinge=pb[idx['item']]+Rb[idx['item']]@hinge_local;hinges.append(hinge)
        p1=floor[0],floor[1],floor[2],'floor'
        parts1=[(p1[0],p1[1],p1[2],p1[3])]+obj
        # Clearance is numeric; do not draw a 3D rod that resembles support.
        parts2=[(p1[0],p1[1],p1[2],p1[3])]+obj
        v,f=skin.update(bq_all[i]);parts1.append((v,f,(222,174,143),'AX50 skin'));parts2.append((v,f,(222,174,143),'AX50 skin'))
        if 'particles' in z.files:
            pq=np.asarray(z['particles'][i]);parts1.append((pq,tri,(235,187,156),'soft'));parts2.append((pq,tri,(235,187,156),'soft'))
        verdict=T['pass_'] if res['passed'] else T['fail'];hold=res['hold']['duration_s']
        if tag.startswith('open_'):verdict='阶段试验：完整握持尚未验收'
        ok=T['yes'] if s['ok'][i] else T['no']
        a=[T['t1'].format(t=t[i],lift=s['lift_m'][i]*1000,gap=gap*1000),T['t2'].format(hinge=s['hinge_deg'][i],tilt=s['tilt_deg'][i],phase=s['phase'][i]),
           T['t3'].format(body=s['body_N'][i],bail=s['bail_N'][i],q=int(s['qualified_substeps'][i]),sub=int(s['substeps'][i]))]
        c=[T['h1'].format(hinge=s['hinge_deg'][i]),T['h2'].format(tilt=s['tilt_deg'][i]),T['h3'].format(ok=ok,hold=hold)]
        return strip([panel(parts1,cam_all),panel(parts1,cam_c),panel(parts2,cam_h)],['AX50 整手与物体',T['v1'],T['v2']],[a,a,c],f"{T['title']}  |  {verdict}")
    ts=s['time']
    frames=[i for i in np.unique(np.searchsorted(ts,np.arange(ts[0],ts[-1]+1e-9,1/30)).clip(0,len(ts)-1)) if valid[i]]
    if not frames:raise RuntimeError('no finite frames to encode')
    encode(OUT/(tag+'.mp4'),15,(frame(i) for i in frames))
    fig,ax=plt.subplots(4,1,figsize=(10,9),sharex=True)
    ax[0].plot(s['time'],s['hinge_deg']);ax[0].axhline(45,color='r',ls='--');ax[0].set_ylabel('bail hinge (deg)')
    ax[1].plot(s['time'],s['lift_m']*1000);ax[1].axhline(25,color='r',ls='--');ax[1].set_ylabel('vessel lift (mm)')
    ax[2].plot(s['time'],s['tilt_deg']);ax[2].axhline(15,color='r',ls='--');ax[2].set_ylabel('vessel tilt (deg)')
    ax[3].plot(s['time'],s['body_N'],label='hand-item N');ax[3].plot(s['time'],s['bail_N'],label='hand-bail N')
    ax[3].axhline(.0005,color='r',ls=':');ax[3].axhline(.01,color='g',ls=':');ax[3].set_ylabel('contact (N)');ax[3].legend()
    ax[3].set_xlabel('simulation time (s)')
    if res['hold']['interval_s']:
        a,b=res['hold']['interval_s']
        for x in ax:x.axvspan(a,b,color='C2',alpha=.12)
    for x in ax:x.grid(alpha=.25)
    fig.tight_layout();fig.savefig(OUT/(tag+'_curves.png'),dpi=160);plt.close(fig)
    info=dict(tag=tag,frames=len(frames),fps=15,slow_motion='2x (30 Hz samples encoded at 15 fps)',side=SIDE,font='CJK '+fontpath if fontpath else 'English fallback',
        views=dict(clearance=dict(aim_center=cam_c[2].tolist(),scale=float(cam_c[1])),hinge=dict(aim_center=cam_h[2].tolist(),scale=float(cam_h[1]))),
        item_bottom_gap_m=dict(min=float(np.min(gaps)),max=float(np.max(gaps))),hinge_point_range_m=[np.min(hinges,0).tolist(),np.max(hinges,0).tolist()],
        passed=res['passed'],scope='geometry from object.xml, poses/particles from recorded npz; no simulation, no interpolation',files=dict(npz=str(OUT/(tag+'.npz')),mp4=str(OUT/(tag+'.mp4')),curves=str(OUT/(tag+'_curves.png'))))
    (OUT/(tag+'_render.json')).write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2));return info

if __name__=='__main__':
    render(sys.argv[1] if len(sys.argv)>1 else 'handle_v1')

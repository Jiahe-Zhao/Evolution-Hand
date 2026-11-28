"""Read measured contact constraints from one Newton world; apply the unmodified task checker offline."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from env import episode,OUT,np,mj
from surface_objects import Score
from scipy.spatial.transform import Rotation as R
from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
import hashlib,sys,json,re,subprocess
src=OUT/sys.argv[1];rec=np.load(src);vid='--video' in sys.argv;dest=OUT/'verified_objects'/sys.argv[2];dest.mkdir(exist_ok=True,parents=True)
g=episode('scrape');meta=next(g);loc=g.gi_frame.f_locals;m=loc['mm'];d=mj.MjData(m);vm=loc['vm'];vd=mj.MjData(vm)
def key(n):return '/'.join(n.split('/')[-2:-1]+[re.sub(r'_\d+$','',n.split('/')[-1])])
geom={key(m.geom(j).name):j for j in range(m.ngeom)};remap=[geom[key(n)] for n in rec['geom_names']]
assert m.opt.cone==int(rec['cone']) and max(rec['counts'])<=32
score=Score('scrape');hist=[];initial=meta['poses'].copy();x=initial.copy();ids={n:i for i,n in enumerate(meta['names'])};ix=ids['item'];mapq=[]
for j,name in enumerate(loc['joints']):
 k=mj.mj_name2id(vm,mj.mjtObj.mjOBJ_JOINT,name)
 if k>=0:mapq.append((vm.jnt_qposadr[k],loc['b'].joint_q_start[j],7 if vm.jnt_type[k]==0 else 1))
if vid:
 render=mj.Renderer(vm,height=360,width=480);cams=[]
 for az,el,dist in [(112,-36,.235),(235,-41,.210)]:
    c=mj.MjvCamera();c.lookat[:]=[-.104,-.009,.045];c.azimuth=az;c.elevation=el;c.distance=dist;cams.append(c)
 proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x408','-r','30','-i','-','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'operation.mp4')],stdin=subprocess.PIPE)
 font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc'),19,index=2)
for i,(q,contacts,n) in enumerate(zip(rec['qpos'],rec['contacts'],rec['counts'])):
 raw=contacts[:n];lengths=[(max(1,2*(int(row[3])-1)) if int(m.opt.cone)==0 else int(row[3])) if row[4]>=0 else 0 for row in raw]
 nf=sum(lengths);mj._functions._realloc_con_efc(d,ncon=int(n),nefc=nf,nJ=nf*m.nv);d.ncon=n;d.nefc=nf;idx=0;rest=False
 for j,(row,L) in enumerate(zip(raw,lengths)):
    c=d.contact[j];aa,bb=remap[int(row[0])],remap[int(row[1])];c.geom[:]=[aa,bb];c.dim=int(row[3]);c.dist=row[2];c.friction[:]=row[5:10];c.efc_address=idx if L else -1
    if L:d.efc_force[idx:idx+L]=row[10:10+L];idx+=L
    rest |= any(key(m.geom(jj).name).endswith('/rest') for jj in (aa,bb))
 x[ix]=q[[27,28,29,31,32,33,30]];h=score.update(m,d,x,initial,ids,i/240)
 assert abs(h['force_N']-rec['sensors'][i,24])<.001 and h['held']==bool(rec['sensors'][i,25]),(i,h,rec['sensors'][i,24:])
 h.update(tiles=sorted(score.tiles),time_s=i/240,lift_mm=float((x[ix,2]-initial[ix,2])*1000),rest=bool(rest),
          edge=(x[ix,:3]+R.from_quat(x[ix,3:]).apply([-.025,0,-.003])).tolist())
 hist.append(h)
 if vid and i%8==0:
    for dst,org,l in mapq:vd.qpos[dst:dst+l]=q[org:org+l]
    for cell,travel in score.travel.items():
     f=min(travel/.003,1);vm.geom_rgba[vm.geom(cell).id,:3]=np.array([.74,.53,.32])*(1-f)+np.array([.49,.35,.23])*f
    mj.mj_forward(vm,vd);can=Image.new('RGB',(960,408),'#172a35')
    for j,c in enumerate(cams):render.update_scene(vd,c);can.paste(Image.fromarray(render.render()),(480*j,0))
    ImageDraw.Draw(can).text((14,365),f"{i/240:.2f} s    有效区域 {len(score.tiles)}/24     刃缘 {h['force_N']:.2f} N     工具角 {h['angle_deg']:.1f}°     "+('已触发成功' if h['success'] else '进行中'),font=font,fill='white');proc.stdin.write(can.tobytes())
if vid:proc.stdin.close();assert proc.wait()==0
(dest/'metrics.json').write_text(json.dumps(hist));lift=max((z['lift_mm'] for z in hist if z['held'] and not z['rest'] and not z['coverage']),default=0)
res=dict(task='scrape',solver='Newton SolverMuJoCo / MJWarp CUDA',batch=int(rec['batch']),lane=int(rec['lane']),source=str(src),cells=len(score.tiles),success=bool(hist[-1]['success']),held_off_rest_before_first_patch_mm=lift,first_success=next((h['time_s'] for h in hist if h['success']),None),steps=len(hist),source_score_sha=hashlib.sha256(Path(__file__).with_name('surface_objects.py').read_bytes()).hexdigest())
(dest/'result.json').write_text(json.dumps(res,indent=2));print(res)


# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from env import episode,OUT,np,mj
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import json,sys,subprocess,hashlib
t,path,key=sys.argv[1:4];video='--video' in sys.argv
g=episode(t,video,'--cpu' not in sys.argv);meta=next(g);v=g.gi_frame.f_locals;act=np.load(OUT/path)['actions'];dest=OUT/'verified_objects'/key;dest.mkdir(parents=True,exist_ok=True)
names=meta['names'];ix=names.index('item');initial=meta['poses'][ix,:3].copy();hist=[];poses=[];actions=[]
if video:
 cam=v['cam'];cam.lookat[:]=[-.10,0,.049] if t=='pebble' else [-.10,-.009,.048];cam.distance=.22;cam.azimuth=101;cam.elevation=-32
 cam2=mj.MjvCamera();cam2.lookat[:]=cam.lookat[:];cam2.distance=.19;cam2.azimuth=243;cam2.elevation=-28
 proc=subprocess.Popen(['ffmpeg','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x408','-r','30','-i','-','-pix_fmt','yuv420p','-movflags','+faststart',str(dest/'operation.mp4')],stdin=subprocess.PIPE)
 font=ImageFont.truetype(str(_EVOLUTION_ROOT / 'assets/fonts/NotoSansCJK-Regular.ttc'),19,index=2)
policy=None
if '--closed' in sys.argv:
 from scrape_policy import ScrapePolicy
 policy=ScrapePolicy(OUT/path,OUT/'scrape_gpu_search/candidate_002.npz')
for i in range(meta['horizon']):
 a=act[min(i,len(act)-1)] if policy is None else policy.act(i,(poses[-1][ix,:3],poses[-1][ix,3:],poses[-1][meta['names'].index('link_0_0'),:3],hist[-1]['force_N']) if i else None)
 z=g.send(a);x=z['poses'].copy();h=z['task_metrics'];score=v['score'];o=x[ix];mm=v['mm'];dd=v['dd']
 support=any(c.dist<=0 and 'rest' in ' '.join(mm.geom(j).name for j in c.geom) and '/item/' in ' '.join(mm.geom(j).name for j in c.geom) for c in dd.contact) if t=='scrape' else False
 h.update(time_s=i/240,streak=score.count,item_xyz=o[:3].tolist(),lift_mm=float(1000*(o[2]-initial[2])),rest=bool(support))
 if t=='scrape':h.update(edge=(o[:3]+R.from_quat(o[3:]).apply([-.025,0,-.003])).tolist(),tiles=sorted(score.tiles))
 hist.append(h);poses.append(x);actions.append(a)
 if video and z['frame'] is not None:
    rend=v['renderer'];rend.update_scene(v['vd'],cam2);can=Image.new('RGB',(960,408),'#172a35');can.paste(Image.fromarray(z['frame']),(0,0));can.paste(Image.fromarray(rend.render()),(480,0));draw=ImageDraw.Draw(can)
    line=(f"目标抬升 {h['lift_mm']:.1f} mm   邻石最大位移 {h['neighbor_max_mm']:.2f} mm   连续达标 {h['streak']/240:.2f} s" if t=='pebble' else f"已完成 {len(score.tiles)}/24 区   刃缘 {h['force_N']:.2f} N   工具角 {h['angle_deg']:.1f}°")
    draw.text((12,365),f"{i/240:.2f} s    {line}",font=font,fill='white');draw.text((790,365),'已触发成功' if h['success'] else '进行中',font=font,fill='#91e6bd');proc.stdin.write(can.tobytes())
if video:proc.stdin.close();assert proc.wait()==0
np.savez_compressed(dest/'trajectory.npz',actions=actions,poses=poses,names=names,action_names=meta['action_names'])
(dest/'metrics.json').write_text(json.dumps(hist));best=max(h['streak'] for h in hist)
res=dict(task=t,source=path,steps=len(hist),success=any(h['success'] for h in hist),first_success=next((h['time_s'] for h in hist if h['success']),None),max_lift_mm=max(h['lift_mm'] for h in hist),maximum_streak=best,source_score_sha=hashlib.sha256(Path(__file__).with_name('surface_objects.py').read_bytes()).hexdigest())
if t=='scrape':res.update(cells=len(score.tiles),held_off_rest_peak_mm=max((h['lift_mm'] for h in hist if h['held'] and not h['rest']),default=0))
else:res.update(neighbor_mm=hist[-1]['neighbor_max_mm'])
(dest/'result.json').write_text(json.dumps(res,indent=2));print(res)

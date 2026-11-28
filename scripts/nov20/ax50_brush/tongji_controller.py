"""Contact-gated writing trajectory; targets are not actual ink."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
from pathlib import Path
import json,numpy as np
class TongjiTrajectory:
 def __init__(self,path,speed=.012,hover=.006,contact=-.002):
  source=json.loads(Path(path).read_text());self.text=source['text'];self.phases=[];clock=0.;last=np.zeros(2)
  for i,s in enumerate(source['strokes']):
   curve=np.array(s['xy_m']);arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]
   for mode,duration,xy0,xy1,z0,z1 in [('lift',.45,last,last,contact,hover),('travel',.65,last,curve[0],hover,hover),('lower',.65,curve[0],curve[0],hover,contact),('write',float(arc[-1]/speed),curve[0],curve[-1],contact,contact)]:
    self.phases.append(dict(start=clock,end=clock+duration,mode=mode,character=s['character'],stroke=s['stroke'],xy0=xy0,xy1=xy1,z0=z0,z1=z1,curve=curve,arc=arc));clock+=duration
   last=curve[-1]
  self.duration=clock+.6;self.last=last;self.hover=hover;self.contact=contact
 def sample(self,t):
  for p in self.phases:
   if t<p['end']:
    u=np.clip((t-p['start'])/(p['end']-p['start']),0,1);u=u*u*(3-2*u)
    xy=np.array([np.interp(u*p['arc'][-1],p['arc'],p['curve'][:,k]) for k in (0,1)]) if p['mode']=='write' else p['xy0']+(p['xy1']-p['xy0'])*u
    return np.r_[xy,p['z0']+(p['z1']-p['z0'])*u],p['mode'],p['character'],p['stroke']
  u=np.clip((t-(self.duration-.6))/.6,0,1)
  return np.r_[self.last,self.contact+(self.hover-self.contact)*u],'finish','济',9
 def set_wrist_target(self,hand,t,origin):
  xyz,phase,ch,stroke=self.sample(t);xyz+=np.asarray(origin)
  hand.set_targets(dict(zip(['diagnostic_wrist_x','diagnostic_wrist_y','diagnostic_wrist_z'],xyz)))
  return phase,ch,stroke
if __name__=='__main__':
 p=Path((str(_EVOLUTION_ROOT) + '/outputs/nov20/ax50_brush'));tr=TongjiTrajectory(p/'tongji_paths.json');(p/'tongji_schedule.json').write_text(json.dumps({'text':tr.text,'stroke_count':15,'duration_s':tr.duration,'speed_m_s':.012,'status':'planned targets only; physical writing blocked until stable grasp','phases':[{k:v for k,v in row.items() if k not in ('xy0','xy1','curve','arc')} for row in tr.phases]},ensure_ascii=False,indent=2));print(tr.text,tr.duration)

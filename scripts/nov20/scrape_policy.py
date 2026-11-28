
# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import numpy as np
from scipy.spatial.transform import Rotation as R
class ScrapePolicy:
 def __init__(self,path,grasp):
    self.p=np.pad(np.load(path)['p'],(0,max(0,20-len(np.load(path)['p']))));self.A=np.load(grasp)['actions'];self.mount=R.from_euler('x',90,degrees=True);self.feedback=np.zeros(3);self.fc=0.
 def act(self,i,state):
    p=self.p;a=self.A[min(i,439)].copy()
    if i<300:return a
    x,quat,root,force=state;edge=x+R.from_quat(quat).apply([-.025,0,-.003])
    if i==300:self.start=edge.copy();self.lever=x-root;self.end=self.mount*R.from_euler('XYZ',self.A[-1,3:6])
    smooth=lambda v:(lambda h:h*h*(3-2*h))(np.clip(v,0,1))
    t=smooth((i-300)/135);u=smooth((i-417)/58);sx=np.clip((i-469)/97,0,1)-np.clip((i-598)/103,0,1);sy=smooth((i-566)/32)
    ro=R.from_rotvec([0,np.deg2rad(p[0]+p[18]*u)*t,0]);a[3:6]=(self.mount.inv()*ro*self.end).as_euler('XYZ');a[[10,13,14,6,12]]+=p[[5,6,7,12,13]]*t
    rev=p[14]>.5;goal=np.array([(-.078 if rev else -.134)+(-1 if rev else 1)*(.056+p[15])*sx+p[1]/1000,.0004+p[16]+(.014+p[17])*sy+p[2]/1000,.035+p[3]/1000+(.018+p[4]/1000)*(1-u)+p[19]*sy])
    target=(1-t)*self.start+t*goal
    self.feedback=np.clip(self.feedback*.987+p[8]*(target-edge),-.016,.016)
    if i>460:self.fc=np.clip(self.fc+p[10]*(force-.95),-.010,.005)
    a[:3]=self.A[300,:3]+self.mount.inv().apply(target-self.start-(ro.apply(self.lever)-self.lever)+self.feedback+[0,0,self.fc])
    return a

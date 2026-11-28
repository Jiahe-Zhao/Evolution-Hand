"""Single-environment baselines. Import this module once to register the four tasks."""

# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
import gymnasium as gym
import numpy as np
from env import episode
TASKS={'Carry':'carry','StoneReorient':'stone','BailLift':'handle','ThumbTap':'phone','PebblePinch':'pebble','SurfaceScrape':'scrape'}
class NewtonHandEnv(gym.Env):
    def __init__(self,task,gpu=True):
        self.task,self.gpu=task,gpu
        self._roll=None
        self.reset()
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.close()
        self._roll=episode(self.task,False,self.gpu)
        meta=next(self._roll)
        self.action_space=gym.spaces.Box(meta['lo'],meta['hi'],dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,meta['poses'].shape,dtype=np.float32)
        self._done=False
        return meta['poses'].astype('f'),{k:meta[k] for k in ['base','action_names','names','horizon']}
    def step(self,action):
        if self._done:raise RuntimeError('Episode finished; call reset().')
        z=self._roll.send(np.asarray(action,dtype=np.float32))
        success=z['success'];timeout=z['timeout'] and not success
        self._done=success or timeout
        return z['poses'].astype('f'),float(success),success,timeout,{k:z[k] for k in ['metrics','events','best_s','success','task_metrics']}
    def close(self):
        if self._roll is not None:self._roll.close();self._roll=None
for suffix,task in TASKS.items():
    gym.register(id=f'EvolutionNewton-{suffix}-v0',entry_point=NewtonHandEnv,kwargs={'task':task})

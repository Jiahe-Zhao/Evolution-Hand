import torch,numpy as np
from scipy.spatial.transform import Rotation
def wrist(env):
    # Matched interventions: in each env replay the same reset state for + / - / 0.
    import copy
    rng=env.generator.get_state().clone();counts=env.episode_counts.copy()
    rows=[]
    for axis in range(6):
        samples=[]
        for sign in (0,1,-1):
            env.generator.set_state(rng);env.episode_counts[:]=counts;env.reset()
            ids=torch.arange(env.num_envs,device=env.device)
            for name in env.object_names:
                pose=env.pose(name).clone();pose[:,0]+=2.;env.put(name,pose,ids)
            base=env.root().clone().cpu().numpy()
            a=torch.zeros(env.num_envs,env.action_size,device=env.device);l=env.limits[env.fids]
            a[:,:len(l)]=2*(env.target[:,env.fids]-l[:,0])/(l[:,1]-l[:,0])-1
            a[:,len(l)+axis]=sign*.4
            # Free-space fixture may terminate normally under task rules; suppress only reset for this diagnostic.
            orig=env.reset;env.reset=lambda ids=None: None
            try:
                for _ in range(8):env.step(a)
            finally:env.reset=orig
            last=env.root().clone().cpu().numpy();mount=Rotation.from_quat(env.mount_quaternion.cpu().numpy())
            delta=mount.inv().apply(last[:,:3]-base[:,:3]);rot=mount.inv().apply((Rotation.from_quat(last[:,3:])*Rotation.from_quat(base[:,3:]).inv()).as_rotvec())
            samples.append(np.concatenate([delta,rot],axis=1))
        pos=samples[1]-samples[0];neg=samples[2]-samples[0]
        rows.append(dict(axis=axis,positive=pos.tolist(),negative=neg.tolist(),passed=bool((pos[:,axis]>1e-5).all() and (neg[:,axis]<-1e-5).all())))
    env.reset()
    return rows

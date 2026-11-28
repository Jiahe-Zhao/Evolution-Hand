"""Per-body Newton tactile mapping with separate normal/friction channels."""
import re,json
import torch
from pxr import Usd,UsdPhysics
from isaaclab.sensors import ContactSensorCfg

def target_permutation(requested, actual):
    """Resolve semantic target order; never assume backend preserves filter order."""
    if len(set(requested))!=len(requested) or len(set(actual))!=len(actual) or set(requested)!=set(actual):
        raise ValueError(f'Ambiguous or missing tactile targets: requested={requested}, actual={actual}')
    return [actual.index(name) for name in requested]


def configure(scene_cfg,asset,targets=None):
    targets = targets or ['{ENV_REGEX_NS}/Object']
    stage=Usd.Stage.Open(str(asset));root=stage.GetDefaultPrim().GetPath().pathString
    bodies=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    rows=[]
    scene_cfg.hand.spawn.activate_contact_sensors=True
    if hasattr(scene_cfg,'contact'):scene_cfg.contact.track_friction_forces=True
    for i,p in enumerate(bodies):
        key=f'tactile_{i}';path='{ENV_REGEX_NS}/Hand'+p.GetPath().pathString[len(root):]
        setattr(scene_cfg,key,ContactSensorCfg(prim_path=path,filter_prim_paths_expr=targets,track_friction_forces=True,update_period=0.,history_length=1))
        rows.append(dict(name=p.GetName(),key=key,path=path))
    return rows

class Tactile:
    def __init__(self,scene,hand,rows):
        self.scene=scene;self.hand=hand;byname={r['name']:r for r in rows}
        assert set(byname)==set(hand.body_names)
        self.rows=[byname[n] for n in hand.body_names]
        self.names=list(hand.body_names)
        self.target_orders=[]
        for row in self.rows:
            sensor=scene[row['key']]
            requested=[path.rsplit('/',1)[-1] for path in sensor.cfg.filter_prim_paths_expr]
            actual=list(sensor._filter_object_names)
            self.target_orders.append(target_permutation(requested,actual))
        self.enabled=None
        self.normal=None;self.friction=None;self.max_balance_error=0.;self.contact_bodies=set();self.checked_steps=0
    def read_targets(self):
        """Return [environment, body, target, xyz] without losing target identity."""
        normal=[];friction=[]
        for row,order in zip(self.rows,self.target_orders):
            data=self.scene[row['key']].data
            a=data.normal_force_matrix_w.torch
            b=data.friction_force_matrix_w.torch
            assert a.ndim==4 and a.shape[1]==1 and a.shape[-1]==3,a.shape
            assert a.shape==b.shape
            normal.append(a[:,0][:,order]);friction.append(b[:,0][:,order])
        normal=torch.stack(normal,1);friction=torch.stack(friction,1)
        if self.enabled is not None:
            normal=normal*self.enabled[:,None,None,None]
            friction=friction*self.enabled[:,None,None,None]
        assert torch.isfinite(normal).all() and torch.isfinite(friction).all()
        return normal,friction
    def read(self):
        normal,friction=self.read_targets()
        self.normal=normal.sum(2);self.friction=friction.sum(2)
        return self.normal,self.friction
    def digit_vectors(self,include_friction=False):
        normal,friction=self.read_targets()
        force=normal+friction if include_friction else normal
        # Net vector is used for opposition; contact thresholds use max per body below.
        return torch.stack([force[:,[i for i,n in enumerate(self.names)
            if n.startswith(f'link_{digit}_')]].sum(1) for digit in range(1,6)],1)
    def digit_forces(self,target=0):
        normal,_=self.read_targets()
        magnitude=normal[:,:,target].norm(dim=-1)
        return torch.stack([magnitude[:,[i for i,n in enumerate(self.names)
            if n.startswith(f'link_{digit}_')]].amax(-1) for digit in range(1,6)],-1)
    def opposition_cosine(self,target=0):
        vectors=self.digit_vectors()[:,:,target]
        thumb=vectors[:,0];other=vectors[:,1:].sum(1)
        return (thumb*other).sum(-1)/(thumb.norm(dim=-1)*other.norm(dim=-1)).clamp_min(1e-8)
    def local_forces(self):
        normal,friction=self.read_targets()
        from frame_math import inverse_rotate_xyzw
        quat=self.hand.data.body_link_pose_w.torch[...,3:7].unsqueeze(2)
        return inverse_rotate_xyzw(quat,normal),inverse_rotate_xyzw(quat,friction)
    def verify(self):
        normal,friction=self.read();obj=self.scene['contact'].data.net_forces_w.torch[:,0]
        # After settling, Object has contacts only with the hand; action/reaction must match.
        error=(normal.sum(1)+friction.sum(1)+obj).norm(dim=-1)
        self.max_balance_error=max(self.max_balance_error,float(error.max()))
        for i in torch.where(normal.norm(dim=-1).amax(0)>.01)[0].tolist():self.contact_bodies.add(self.names[i])
        self.checked_steps+=1
        assert error.max()<.003,('Force imbalance',float(error.max()))
    def report(self):
        return dict(bodies=self.rows,body_count=len(self.rows),normal_and_friction_separate=True,checked_steps=self.checked_steps,max_action_reaction_error_N=self.max_balance_error,contact_bodies=sorted(self.contact_bodies),passed=self.checked_steps>0 and bool(self.contact_bodies))
    def check_partial_reset(self):
        # Inspect underlying buffers before lazy data refresh recomputes current contacts.
        import warp as wp
        assert self.hand.data.joint_pos.torch.shape[0] >= 2
        keys=['_normal_force_matrix_w','_friction_force_matrix_w']
        for row in self.rows:
            sensor=self.scene[row['key']]
            before={k:wp.to_torch(getattr(sensor._data,k)).clone() for k in keys}
            sensor.reset(env_ids=[0])
            for k in keys:
                after=wp.to_torch(getattr(sensor._data,k))
                assert torch.count_nonzero(after[0])==0,(row['name'],k)
                assert torch.equal(after[1:],before[k][1:]),(row['name'],'cross-environment reset')
        return dict(passed=True,reset_environment=0,other_environments_preserved=True,scope='sensor buffers only; full environment reset pending')

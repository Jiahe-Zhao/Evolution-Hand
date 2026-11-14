"""Newton per-contact virtual fingertip pad, 1 mm pitch in local Y/Z.

Only distal skin/cover contacts on negative-X pad face and Object are sampled.
This is a rigid-contact sampling model, not an elastomer/taxel sensor model.
"""
import re
import torch
import warp as wp
from frame_math import rotate_xyzw, inverse_rotate_xyzw
from tactile_slip import SlipDetector

class FingertipContactGrid:
    def __init__(self, env):
        from isaaclab_newton.physics import NewtonManager
        self.manager=NewtonManager;self.env=env;self.model=NewtonManager.get_model()
        self.detector=SlipDetector(dt=env.dt,grid_mm=1.)
        self.size=32;self.pitch=.001
        self.tipids=[env.hand.body_names.index(f'link_{d}_3') for d in range(1,6)]
        body=self.model.shape_body.numpy();labels=self.model.shape_label
        self.target_names=list(env.target_names)
        metadata=[]
        for i,label in enumerate(labels):
            match=re.search(r'/env_(\d+)/',label)
            e=int(match.group(1)) if match else -1
            digit=re.search(r'/(?:skin_link_|visual_finger_cover_)([1-5])_3(?:/|$)',label)
            metadata.append((e,int(digit.group(1))-1 if digit else -1,next((i for i,name in enumerate(self.target_names,1) if name in label.split('/')),0)))
        self.meta=torch.tensor(metadata,device=env.device,dtype=torch.long)
        self.shape_body=wp.to_torch(self.model.shape_body).long()
        self.grid=torch.zeros(env.num_envs,5,self.size,self.size,device=env.device)
        self.slip=torch.zeros(env.num_envs,5,device=env.device,dtype=torch.bool)
        self.warning=self.slip.clone();self.speed=torch.zeros_like(self.slip,dtype=torch.float32)
        self.contact_count=torch.zeros_like(self.speed);self.total_force=torch.zeros_like(self.speed)
        self.dropped_force=torch.zeros_like(self.speed)
        self.contract=dict(pitch_mm=1.,grid_shape=[5,32,32],axes='distal link Y/Z',origin_m=[-.016,-.0085],scope='negative-X fingertip pad; task target bodies; per-contact force',slip_speed_threshold_m_s=.005,normal_contact_threshold_N=.02,material_sensor_equivalence=False)
        assert (self.meta[:,1]>=0).any(), 'No fingertip skin shapes mapped'

    def reset(self, ids):
        for x in [self.grid,self.slip,self.warning,self.speed,self.contact_count,self.total_force,self.dropped_force]:x[ids]=0
        self.detector.reset(ids)

    def sample(self):
        c=self.manager.get_contacts();count=int(wp.to_torch(c.rigid_contact_count)[0])
        if count>c.rigid_contact_max:raise RuntimeError('Contact buffer overflow')
        self.grid.zero_();self.contact_count.zero_();self.total_force.zero_();self.dropped_force.zero_();self.speed.zero_()
        friction=torch.zeros(self.env.num_envs,5,3,device=self.env.device)
        centroid=torch.zeros(self.env.num_envs,5,2,device=self.env.device)
        if count:
            a=wp.to_torch(c.rigid_contact_shape0)[:count].long();b=wp.to_torch(c.rigid_contact_shape1)[:count].long()
            ma=self.meta[a];mb=self.meta[b]
            first=(ma[:,1]>=0)&(mb[:,2]>0)&(ma[:,0]==mb[:,0])&(ma[:,0]>=0)
            second=(mb[:,1]>=0)&(ma[:,2]>0)&(ma[:,0]==mb[:,0])&(mb[:,0]>=0)
            chosen=torch.where(first|second)[0];first=first[chosen]
            if len(chosen):
                shape=torch.where(first,a[chosen],b[chosen]);other_id=torch.where(first,mb[chosen,2],ma[chosen,2]);metadata=self.meta[shape];e=metadata[:,0];f=metadata[:,1]
                normal=wp.to_torch(c.rigid_contact_normal)[chosen];normal=normal/normal.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                force=wp.to_torch(c.force)[chosen,:3]*torch.where(first,1.,-1.)[:,None]
                fn=(force*normal).sum(-1).abs();ft=force-(force*normal).sum(-1,keepdim=True)*normal
                p0=wp.to_torch(c.rigid_contact_point0)[chosen]+wp.to_torch(c.rigid_contact_offset0)[chosen]
                p1=wp.to_torch(c.rigid_contact_point1)[chosen]+wp.to_torch(c.rigid_contact_offset1)[chosen]
                p=torch.where(first[:,None],p0,p1)
                tip=torch.tensor(self.tipids,device=e.device)[f]
                pose=self.env.hand.data.body_link_pose_w.torch[e,tip]
                local_n=inverse_rotate_xyzw(pose[:,3:],normal)
                pad=((p[:,0]>0) if self.env.config.get("left_pad") else (p[:,0]<0))&(local_n[:,0].abs()>.5)&(fn>0)
                e=e[pad];f=f[pad];p=p[pad];pose=pose[pad];tip=tip[pad];normal=normal[pad];fn=fn[pad];ft=ft[pad]
                if len(e):
                    other_id=other_id[pad]
                    world=pose[:,:3]+rotate_xyzw(pose[:,3:],p)
                    hcom=self.env.hand.data.body_com_pose_w.torch[e,tip,:3];hv=self.env.hand.data.body_com_vel_w.torch[e,tip]
                    obj_v=torch.zeros_like(world)
                    for index,name in enumerate(self.target_names,1):
                        mask=other_id==index
                        obj=self.env.scene[name].data
                        ids=e[mask];com=obj.root_com_pose_w.torch[ids,:3];vel=obj.root_com_vel_w.torch[ids]
                        obj_v[mask]=vel[:,:3]+torch.linalg.cross(vel[:,3:],world[mask]-com)
                    rel=obj_v-hv[:,:3]-torch.linalg.cross(hv[:,3:],world-hcom)
                    tangent=rel-(rel*normal).sum(-1,keepdim=True)*normal
                    xy=p[:,[1,2]]-torch.tensor([0.,.0075],device=p.device)
                    ij=torch.floor(xy/self.pitch+self.size/2).long()
                    inside=(ij>=0).all(-1)&(ij<self.size).all(-1)
                    group=e*5+f
                    self.total_force.flatten().scatter_add_(0,group,fn)
                    self.dropped_force.flatten().scatter_add_(0,group,torch.where(inside,0.,fn))
                    self.contact_count.flatten().scatter_add_(0,group,torch.ones_like(fn))
                    # A fast, near-zero-force contact must not borrow load from a sticking point.
                    loaded_speed=torch.where(fn>.02,tangent.norm(dim=-1),0.)
                    self.speed.flatten().scatter_reduce_(0,group,loaded_speed,reduce='amax',include_self=True)
                    friction.index_put_((e,f),ft,accumulate=True)
                    centroid.index_put_((e,f),xy*fn[:,None],accumulate=True)
                    index=group[inside]*self.size*self.size+ij[inside,1]*self.size+ij[inside,0]
                    self.grid.flatten().scatter_add_(0,index,fn[inside])
        centroid/=self.total_force.clamp_min(1e-8)[...,None]
        velocity=torch.stack([self.speed,torch.zeros_like(self.speed)],-1)
        self.slip=self.detector.update(self.total_force,friction,centroid,velocity)
        self.warning=self.detector.last['friction_warning']
        assert torch.isfinite(self.grid).all()
        assert torch.allclose(self.grid.sum((-1,-2))+self.dropped_force,self.total_force,atol=1e-5,rtol=1e-4)

    def vector(self):
        return torch.cat([self.grid.flatten(1)/5,self.slip.float(),self.warning.float(),self.speed],-1)

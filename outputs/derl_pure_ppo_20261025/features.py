"""Frame-consistent features; scalar-first unit quaternions (w,x,y,z)."""
import torch

def inverse_rotate(q, v):
    qv=-q[...,1:]
    t=2*torch.linalg.cross(qv,v,dim=-1)
    return v+q[...,:1]*t+torch.linalg.cross(qv,t,dim=-1)

def relative_quaternion(parent, child):
    a=parent.clone();a[...,1:]=-a[...,1:]
    aw,av=a[...,:1],a[...,1:];bw,bv=child[...,:1],child[...,1:]
    out=torch.cat((aw*bw-(av*bv).sum(-1,keepdim=True),aw*bv+bw*av+torch.linalg.cross(av,bv,dim=-1)),dim=-1)
    return torch.where(out[...,:1]<0,-out,out)

def object_contact_forces(object_quaternion, world_forces):
    return inverse_rotate(object_quaternion[:,None,:].expand(-1,world_forces.shape[1],-1),world_forces)

def relative_features(hand_position,hand_quaternion,hand_velocity,hand_angular_velocity,object_state,forces):
    displacement=object_state[:,:3]-hand_position
    velocity=object_state[:,7:10]-hand_velocity-torch.linalg.cross(hand_angular_velocity,displacement,dim=-1)
    return torch.cat((inverse_rotate(hand_quaternion,displacement)/.05,
                      inverse_rotate(hand_quaternion,velocity)/.1,
                      relative_quaternion(hand_quaternion,object_state[:,3:7]),
                      object_contact_forces(object_state[:,3:7],forces).flatten(1)/2),dim=1)

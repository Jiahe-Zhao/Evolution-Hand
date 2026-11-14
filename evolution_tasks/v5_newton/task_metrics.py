"""Backend-independent success primitives preserving the current task predicates.

Inputs use SI units and XYZW. Physics, contact acquisition and reset distribution
are separate responsibilities; passing these functions is not simulation parity.
"""
import torch
from frame_math import rotate_xyzw


def grasp_held(rise,support_force,digit_force_local,upright,speed,target,tolerance):
    normal=digit_force_local[...,0].abs()
    touch=(normal[:,0]>.1)&(normal[:,1:].sum(-1)>.1)
    return (rise>.015)&(support_force<.05)&touch&(upright>.98)&(speed<.1)&((target-rise).abs()<=tolerance)


def branch_qualified(digit_force,opposition,position_delta,rotation_delta,
                     force_threshold,min_long,position_tolerance,rotation_tolerance,require_stability):
    stable=(position_delta<=position_tolerance)&(rotation_delta<=rotation_tolerance)
    if not require_stability:stable=torch.ones_like(stable)
    return (digit_force[:,0]>=force_threshold)&((digit_force[:,1:]>=force_threshold).sum(-1)>=min_long)&stable&(opposition<0)


def projected_exposure(food_position,food_size,leaf_positions,leaf_quaternions,leaf_sizes):
    device=food_position.device
    x=(torch.arange(25,device=device)+.5)/25-.5
    y=(torch.arange(23,device=device)+.5)/23-.5
    grid=torch.cartesian_prod(x*food_size[0],y*food_size[1])
    points=food_position[:,None,:2]+grid[None]
    covered=torch.zeros(points.shape[:2],device=device,dtype=torch.bool)
    # Rotation columns are rotated local basis vectors.
    basis=torch.eye(3,device=device,dtype=food_position.dtype)
    for i in range(leaf_positions.shape[1]):
        rotated=rotate_xyzw(leaf_quaternions[:,i,None,:],basis[None])
        extent=(rotated[...,:2].abs()*leaf_sizes[i][None,:,None]/2).sum(1)
        covered |= ((points-leaf_positions[:,i,None,:2]).abs()<=extent[:,None]).all(-1)
    return 1-covered.float().mean(-1)


def forage_qualified(exposure,contact_seen,speeds,distances):
    invalid=(~torch.isfinite(speeds).all(-1))|(distances>.5).any(-1)
    return (exposure>=.98)&contact_seen.all(-1)&(speeds.amax(-1)<.15)&~invalid


def update_grasp_state(contact, streak, missing, held, lost, rewarded, hold_steps, grace_steps):
    streak=torch.where(contact,streak+1,torch.zeros_like(streak))
    missing=torch.where(contact,torch.zeros_like(missing),missing+1)
    acquired=contact&(streak>=hold_steps)
    confirmed_loss=held&(missing>grace_steps)
    held=(held&~confirmed_loss)|acquired
    lost=(lost|confirmed_loss)&~acquired
    first_grasp=acquired&~rewarded
    rewarded=rewarded|first_grasp
    return streak,missing,held,lost,rewarded,first_grasp


def strike_hit(held,holding_now,tip_position,target_position,impact_force,distance_threshold,force_threshold):
    distance=(tip_position[:,:2]-target_position[:,:2]).norm(dim=-1)
    return held&holding_now&(distance<distance_threshold)&(impact_force.norm(dim=-1)>=force_threshold)


def bounded_progress(current,best,weights,event,cost):
    current=current.clamp(0,1)
    gain=(current-best).clamp_min(0)
    return (gain*weights).sum(-1)+10*event.float()-cost,torch.maximum(best,current)

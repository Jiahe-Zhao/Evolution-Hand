"""Explicit XYZW transforms at the Newton boundary; no implicit legacy WXYZ."""
import torch

def rotate_xyzw(quaternion, vector):
    xyz,w=quaternion[...,:3],quaternion[...,3:4]
    xyz,vector=torch.broadcast_tensors(xyz,vector)
    cross=2*torch.linalg.cross(xyz,vector,dim=-1)
    return vector+w*cross+torch.linalg.cross(xyz,cross,dim=-1)

def inverse_rotate_xyzw(quaternion,vector):
    return rotate_xyzw(torch.cat((-quaternion[...,:3],quaternion[...,3:4]),-1),vector)

def from_legacy_wxyz(quaternion):
    return quaternion[...,[1,2,3,0]]

def to_legacy_wxyz(quaternion):
    return quaternion[...,[3,0,1,2]]

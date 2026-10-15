"""Check the new control contract without requiring Isaac Sim."""
import torch
from palm_coordination import coordinate_palm_joints, ROOT_FLEX, ROOT_SPLAY, MAX_DEVIATION_RAD

def main():
    names = ['link_1_thumb_spread_joint'] + list(ROOT_SPLAY) + [f'link_0_0_to_link_{d}_0' for d in range(1,6)] + [f'link_{d}_0_to_link_{d}_1' for d in range(1,6)] + [f'link_{d}_1_to_link_{d}_2' for d in range(1,6)]
    assert len(names) == 20
    torch.manual_seed(7)
    original = torch.rand(1000,20)*1.57
    saved = original.clone()
    limits = torch.zeros(1,20,2);limits[...,1] = 1.57
    limits[:,names.index(ROOT_FLEX[0]),1] = 1.18
    target = coordinate_palm_joints(original,names,limits)
    flex = [names.index(n) for n in ROOT_FLEX]
    spread = [names.index(n) for n in ROOT_SPLAY]
    untouched = [i for i in range(20) if i not in flex+spread]
    assert torch.equal(original[:,untouched],target[:,untouched])
    assert torch.equal(target[:,spread],torch.zeros_like(target[:,spread]))
    assert float((target[:,flex].max(1).values-target[:,flex].min(1).values).max()) <= 2*MAX_DEVIATION_RAD+1e-6
    assert bool((target[:,flex] >= limits[...,flex,0]).all())
    assert bool((target[:,flex] <= limits[...,flex,1]).all())
    extreme = coordinate_palm_joints(limits[...,1],names,limits)
    assert bool((extreme[:,flex] <= limits[...,flex,1]).all())
    assert torch.equal(original,saved)
    print('PASS: 1000 inputs; 20 physical joints; thumb/distal unchanged; root target separation <=10deg; limits preserved')
if __name__=='__main__':main()

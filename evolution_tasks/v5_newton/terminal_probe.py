"""Action-only closed-loop diagnostic, no reward or state overrides."""
import json,torch
from frame_math import inverse_rotate_xyzw
def run(e,out):
    e.record_terminal_evidence=True
    e.reset()
    ids=torch.arange(e.num_envs,device=e.device)
    lo,hi=e.limits[e.fids].T;initial=e.q0[:,e.fids].clone()
    mount=e.mount_quaternion; rows=[]; peak=torch.zeros_like(e.age);seen=torch.zeros_like(e.held)
    for i in range(e.max_steps):
        t=e.age*e.dt;force=e.tactile.digit_forces()
        name={'branch':'branch','grasp':'Object','forage':'leaf_one','strike':'Cone'}[e.task]
        obj=e.pose(name)[:,:3];tips=e.tips();palm=tips.mean(1)
        grip=force[:,0].gt(.1)&force[:,1:].gt(.1).sum(1).ge(2)
        phase=(t/.8).clamp(0,1)
        q=initial.clone()
        for j,jid in enumerate(e.fids):
            n=e.hand.joint_names[jid]
            if 'spread' not in n:q[:,j]=initial[:,j]+phase*(hi[j]-initial[:,j])*(.35+.12*(ids%4))
        delta=obj-palm
        if e.task=='forage':
            leaf=torch.where(e.seen[:,0,None],e.pose('leaf_two')[:,:3],obj)
            delta=leaf-palm;delta[:,0]+=torch.where(e.seen.any(1),.08,-.008)
        elif e.task=='grasp':
            rise=obj[:,2]-e.initial['Object'][:,2]
            delta=torch.where(grip[:,None],torch.stack([rise*0,rise*0,e.goals-rise],1),delta)
        elif e.task=='strike':
            dest=e.origins+e.tensor(e.native['target_position']);dest[:,2]+=e.native['tool_impact_offset']
            delta=torch.where(e.held[:,None],dest-obj,delta)
        wrist=(e.wrist+.4*inverse_rotate_xyzw(mount,delta)).clamp(-.12,.12)
        a=torch.zeros_like(e.last_action);a[:,:len(lo)]=2*(q-lo)/(hi-lo)-1;a[:,len(lo):len(lo)+3]=wrist/.12
        _,r,term,trunc,info=e.step(a);success=info['episode_success'];seen|=success;peak=torch.maximum(peak,e.streak)
        if i%30==0 or (term|trunc).any():
            rows.append(dict(step=i,reward=r.tolist(),success=success.tolist(),terminated=term.tolist(),truncated=trunc.tolist(),forces_before=force.tolist(),peak_streak=peak.tolist()))
            (out/'terminal_trial.json').write_text(json.dumps(dict(task=e.task,steps=i+1,successes=seen.tolist(),samples=rows)))
        if success.any():
            physics=info["physics"]
            events=[dict(environment=k,step=i,reward=r[k].item(),terminated=term[k].item(),truncated=trunc[k].item(),
                         evidence={n:v[k].tolist() for n,v in physics.items() if n!="objects"},
                         poses={n:v[k].tolist() for n,v in physics["objects"].items()}) for k in torch.where(success)[0].tolist()]
            (out/"successful_terminals.json").write_text(json.dumps(events,indent=2))
        if success.any() or (term|trunc).all():break
    return seen.any().item()

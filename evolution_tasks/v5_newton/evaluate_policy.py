"""Evaluate a new-interface checkpoint with frozen observation statistics."""
import torch
from short_ppo import Policy,Moments

@torch.no_grad()
def evaluate(env,checkpoint,action_seed=31,mode='sample',scene_seed=1062802):
    if mode not in ('sample','mean'):raise ValueError(mode)
    state=torch.load(checkpoint,map_location=env.device,weights_only=False)
    if state['interface_signature']!=env.signature:raise ValueError('Checkpoint interface signature mismatch')
    net=Policy(env.observation_size,env.action_size).to(env.device);net.load_state_dict(state['policy']);net.eval()
    rms=Moments((env.observation_size,),env.device)
    for key,value in state['observation_moments'].items():setattr(rms,key,value.to(env.device) if torch.is_tensor(value) else value)
    before=rms.mean.clone();count=rms.count
    env.evaluating=True
    env.seed=scene_seed;env.generator.manual_seed(scene_seed);env.episode_counts[:]=0
    obs=env.reset();torch.manual_seed(action_seed)
    alive=torch.ones(env.num_envs,device=env.device,dtype=torch.bool);success=~alive;returns=torch.zeros(env.num_envs,device=env.device)
    for _ in range(env.max_steps):
        dist,_=net(rms.normalize(obs));action=dist.mean if mode=='mean' else dist.sample()
        obs,reward,terminated,truncated,info=env.step(action)
        success |= alive&info['episode_success'];returns+=alive*reward;alive &= ~(terminated|truncated)
        if not alive.any():break
    assert torch.equal(before,rms.mean) and count==rms.count
    env.evaluating=False
    return dict(successes=int(success.sum()),episodes=env.num_envs,unfinished=int(alive.sum()),returns=returns.tolist(),action_seed=action_seed,scene_seed=scene_seed,mode=mode,normalization_frozen=True)

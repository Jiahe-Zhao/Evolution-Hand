"""First-episode stage audit; same checkpoint with neutral/wrist-fixed controls."""
import json,os
from pathlib import Path
import torch
from native_goal_ppo import Policy,Moments
from whole_hand_tactile import WholeHandTactile

def run(env,args,out):
 e=env.raw;n=e.num_envs;c=e.graph_control
 state=torch.load(Path(os.environ['STRIKE_AUDIT_CHECKPOINT']),map_location=e.device,weights_only=False)
 net=Policy(env.observation_space.shape[0],env.action_space.shape[0]).to(e.device);net.load_state_dict(state['policy']);net.eval()
 rms=Moments((env.observation_space.shape[0],),e.device)
 for k,v in state['observation_moments'].items():setattr(rms,k,v)
 tactile=WholeHandTactile(e,'strike',out) # diagnostics only; no policy input change
 original=e._get_rewards;rows=[];active={}
 def capture():
  rew=original();tactile.update();a=active
  if not a:return rew
  live=a['alive'];f=e.tool_finger_contact_forces;full=tactile.digit_forces()
  speed=e.cone.data.root_lin_vel_w.norm(dim=-1);impact=e.strike_object_force.norm(dim=-1)
  holding=(f[:,0]>=e.cfg.tool_finger_contact_force_threshold)&((f[:,1:]>=e.cfg.tool_finger_contact_force_threshold).sum(-1)>=e.cfg.min_long_finger_contacts)
  for key,value in [('thumb',f[:,0]>=e.cfg.tool_finger_contact_force_threshold),('long',(f[:,1:]>=e.cfg.tool_finger_contact_force_threshold).sum(-1)>=e.cfg.min_long_finger_contacts),('simultaneous',holding),('held',e.tool_was_held),('impact',impact>=e.cfg.success_force_threshold),('success',env.event)]:a[key]|=live&value
  a['lost'] |= live & a['held'] & ~e.tool_was_held
  a['holdmax']=torch.maximum(a['holdmax'],torch.where(live,e.tool_grasp_streak,0))
  a['force']=torch.maximum(a['force'],torch.where(live,f.max(-1).values,0));a['fullforce']=torch.maximum(a['fullforce'],torch.where(live,full.max(-1).values,0));a['speed']=torch.maximum(a['speed'],torch.where(live,speed,0))
  a['heldframes']+=(live&e.tool_was_held).long()
  a['native_full_gap']=max(a['native_full_gap'],float((f-full).abs()[live].max()) if live.any() else 0.)
  for i in torch.where(live&(speed>2)&~a['speed_seen'])[0].tolist():
   a['events'].append(dict(env=i,step=int(a['length'][i])+1,speed=float(speed[i]),finger_force=f[i].tolist(),ever_held=bool(a['held'][i]),held_now=bool(e.tool_was_held[i]),impact=float(impact[i]),tool=e.cone.data.root_state_w[i].tolist(),wrist=e.graph_wrist_offset[i].tolist()))
  a['speed_seen']|=live&(speed>2)
  # Save the state before DirectRLEnv automatically resets terminal environments.
  for i in torch.where(live & (e.reset_terminated|e.reset_time_outs))[0].tolist():
   a['terminal'].append(dict(env=i,step=int(a['length'][i])+1,timeout=bool(e.reset_time_outs[i]),success=bool(env.event[i]),z=float(e.cone_pos[i,2]),target_xy_distance=float((e.cone_pos[i,:2]-e.strike_target_pos[i,:2]).norm()),speed=float(speed[i]),ever_held=bool(a['held'][i])))
  return rew
 e._get_rewards=capture
 for mode in ['hold','sample_fixed_wrist','sample','mean']:
  active.clear();env.reset_seed=args.eval_seed;obs=env.reset();tactile.reset();torch.manual_seed(args.eval_seed+1)
  z=torch.zeros(n,dtype=torch.bool,device=e.device);active.update({k:z.clone() for k in ['thumb','long','simultaneous','held','lost','impact','success','speed_seen']})
  active.update(alive=~z,length=z.long(),holdmax=z.long(),heldframes=z.long(),force=z.float(),fullforce=z.float(),speed=z.float(),events=[],terminal=[],native_full_gap=0.)
  initial={'tool':e.cone.data.root_state_w[0].tolist(),'support':e.tool_support.data.root_state_w[0].tolist(),'tips':e.cartesian_ik.fingertip_positions_world()[0].tolist(),'target':e.strike_target_pos[0].tolist()}
  q=e.hand.data.joint_pos.index_select(-1,c.indices);neutral=torch.zeros_like(e.graph_actions);neutral[:,:len(c.indices)]=2*(q-c.lower)/(c.upper-c.lower)-1
  for step in range(e.max_episode_length):
   with torch.no_grad():
    dist,_=net(rms.normalize(obs));action=neutral if mode=='hold' else dist.mean if mode=='mean' else dist.sample()
    if mode=='sample_fixed_wrist':action[:,-3:]=0
   obs,reward,done,info=env.step(action);active['length']+=active['alive'].long();active['alive']&=~done
   if not active['alive'].any():break
  row={'mode':mode,'num_envs':n,'initial_env0':initial,'counts':{k:int(active[k].sum()) for k in ['thumb','long','simultaneous','held','lost','impact','success','speed_seen']},'holdmax':active['holdmax'].tolist(),'heldframes':active['heldframes'].tolist(),'first_episode_lengths':active['length'].tolist(),'peak_force_N':float(active['force'].max()),'peak_full_force_N':float(active['fullforce'].max()),'peak_speed_m_s':float(active['speed'].max()),'native_full_max_difference_N':active['native_full_gap'],'speed_events':active['events'],'terminal':active['terminal']}
  rows.append(row);(out/'failure_audit.json').write_text(json.dumps(dict(task='strike',checkpoint=os.environ['STRIKE_AUDIT_CHECKPOINT'],note='Fixed-wrist controller is closed-loop; finger actions can differ after state divergence. No geometry or reward change.',trials=rows),indent=2));print('STRIKE_AUDIT',mode,row['counts'],row['peak_force_N'],row['peak_speed_m_s'],flush=True)
 (out/'completed.json').write_text(json.dumps({'status':'completed'}))

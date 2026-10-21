"""Opt-in joint_graph_v2 adapter for the four existing physical tasks.

Keep object dynamics and reward methods. Replace fixed-width joint actions,
legacy observation packing, and anatomy-specific wrist motion gates.
"""
import importlib
import os
import re
from joint_graph_control import JointGraph, JointTargetController

OBJECTS = {'grasp': ('grasp_object',), 'branch': ('branch',),
           'forage': ('food','leaf_one','leaf_two'), 'strike': ('cone',)}


def observation_names(graph, task):
    names=[]
    for field in ('q','dq','command'):
        names += [f'{field}:{j.name}' for j in graph.joints]
    for obj in ('hand',)+OBJECTS[task]:
        names += [f'{obj}.state.{i}' for i in range(13)]
    names += [f'tip{f}.position.{a}' for f in range(5) for a in 'xyz']
    names += [f'contact.{i}' for i in range(2 if task=='forage' else 5)]
    names += [f'morph:{j.name}.{field}' for j in graph.joints for field in ('axis_x','axis_y','axis_z','origin_x','origin_y','origin_z','roll','pitch','yaw','lower','upper','velocity','effort','depth')]
    names += ['episode.elapsed_s','grasp.held','grasp.hold_steps','grasp.missing_steps']
    names += ['previous_action:'+a for a in graph.contract(task in ('strike','forage'))['actions']]
    return names


def install(env_id, cfg, task, urdf, wrist_speed_m_s=0.006, driven_contract=None):
    import gymnasium as gym
    import torch
    graph=JointGraph.from_urdf(urdf)
    wrist=task in ('strike','forage');n=len(graph.joints)
    entry=gym.spec(env_id).entry_point
    module,name=entry.split(':');cls=getattr(importlib.import_module(module),name)
    if getattr(cls,'_joint_graph_installed',False):
        raise RuntimeError('Use a fresh process for a different graph task adapter')
    os.environ.pop('EVOLUTION_STRIKE_BC_MODE',None)
    os.environ.pop('EVOLUTION_BRANCH_BC_MODE',None)
    os.environ.pop('EVOLUTION_STRIKE_BC_TEACHER_CHECKPOINT',None)
    cfg.action_space=n+(3 if wrist else 0)
    cfg.actuated_joint_names=[j.name for j in graph.joints]
    cfg.observation_space=len(observation_names(graph,task))
    cfg.asymmetric_obs=False
    cfg.state_space=0
    original_reset=cls._reset_idx

    def ensure(self):
        if not hasattr(self,'graph_control'):
            self.graph_control=JointTargetController(graph,list(self.hand.joint_names),self.device,
                external_joint_names=driven_contract['wrist_joint_names'] if driven_contract else ())
            if driven_contract:
                self.driven_wrist_ids=[self.hand.joint_names.index(x) for x in driven_contract['wrist_joint_names']]
                self.driven_hand_body_id=self.hand.body_names.index(driven_contract['hand_root'])
            # Respect the effective actuator configuration, not only URDF maxima.
            caps=[]
            for joint in graph.joints:
                cap=joint.velocity
                for actuator in self.cfg.robot_cfg.actuators.values():
                    if not any(re.fullmatch(pattern,joint.name) for pattern in actuator.joint_names_expr):
                        continue
                    configured=getattr(actuator,'velocity_limit_sim',None)
                    if isinstance(configured,dict):
                        for pattern,value in configured.items():
                            if re.fullmatch(pattern,joint.name):cap=min(cap,float(value))
                    elif configured is not None:cap=min(cap,float(configured))
                caps.append(cap)
            self.graph_control.velocity=torch.tensor(caps,device=self.device)
            self.graph_actions=torch.zeros((self.num_envs,cfg.action_space),device=self.device)
            self.graph_wrist_offset=torch.zeros((self.num_envs,3),device=self.device)
            self.mechanical_controller_name='joint_graph_v2'
            self.graph_contract=graph.contract(wrist)
            self.graph_contract.update(task=task,observations=observation_names(graph,task),
                                       wrist_speed_m_s=wrist_speed_m_s, effective_joint_velocity_limits=caps)
        if driven_contract:self.graph_contract['external_wrist_drive']=driven_contract
        return self.graph_control

    def reset(self, env_ids):
        if driven_contract:
            ensure(self)
            ids=self.driven_wrist_ids
            # Legacy reset randomizes every simulator DOF. External stage must
            # start at zero before the tool placement is calculated.
            lower=self.hand_dof_lower_limits[:,ids].clone()
            upper=self.hand_dof_upper_limits[:,ids].clone()
            self.hand_dof_lower_limits[:,ids]=0
            self.hand_dof_upper_limits[:,ids]=0
            try:
                original_reset(self,env_ids)
            finally:
                self.hand_dof_lower_limits[:,ids]=lower
                self.hand_dof_upper_limits[:,ids]=upper
        else:
            original_reset(self,env_ids)
        if task=='branch':
            from branch_reset_clearance_v5 import correct
            correct(self,env_ids)
        ensure(self)
        if env_ids is None:env_ids=slice(None)
        self.graph_actions[env_ids]=0
        self.graph_wrist_offset[env_ids]=0
        # Carry the reset's physically bounded targets, not stale commands.
        limits=self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.prev_targets[env_ids]=self.hand.data.joint_pos[env_ids].clamp(limits[env_ids,:,0],limits[env_ids,:,1])
        self.cur_targets[env_ids]=self.prev_targets[env_ids]

    def pre(self, actions):
        ensure(self)
        if actions.shape != self.graph_actions.shape:
            raise ValueError('joint_graph_v2 action shape mismatch')
        self.graph_actions=actions.clamp(-1,1).clone()
        self.actions=self.graph_actions
        self.raw_actions=self.graph_actions
        self.hand_actions=self.graph_actions[:,:n]
        self.joint_actions=self.hand_actions
        if wrist:self.wrist_actions=self.graph_actions[:,n:]
        if task=='branch':
            # Preserve the coordination diagnostic in terms of actual digits.
            ids_by_name={j.name:i for i,j in enumerate(graph.joints)}
            self.finger_action_scores=torch.stack([
                self.hand_actions[:,[ids_by_name[self.hand.joint_names[i]] for i in ids.tolist()]].mean(-1)
                if len(ids) else torch.zeros(self.num_envs,device=self.device)
                for ids in self.cartesian_ik.joint_ids],dim=-1)

    def apply(self):
        control=ensure(self)
        targets=control.targets(self.graph_actions[:,:n],self.prev_targets,self.cfg.sim.dt)
        self.cur_targets[:]=targets;self.prev_targets[:]=targets
        self.hand.set_joint_position_target(targets)
        if wrist:
            scale=self.wrist_action_scale if task=='strike' else self.wrist_limits
            desired=self.graph_actions[:,n:]*scale
            delta=(desired-self.graph_wrist_offset).clamp(-wrist_speed_m_s*self.cfg.sim.dt,wrist_speed_m_s*self.cfg.sim.dt)
            self.graph_wrist_offset+=delta
            state=self.hand.data.default_root_state.clone()
            state[:,:3]+=self.scene.env_origins+self.graph_wrist_offset
            if task=='strike':
                state[:,:3]+=self.hand_reset_offset
                self.wrist_offset[:]=self.graph_wrist_offset
                self.wrist_allowed=torch.ones(self.num_envs,dtype=torch.bool,device=self.device)
            else:self.wrist_offsets[:]=self.graph_wrist_offset
            state[:,7:]=0;state[:,7:10]=delta/self.cfg.sim.dt
            if driven_contract:
                from driven_wrist_adapter import apply as apply_driven
                apply_driven(self,targets)
            else:
                self.hand.write_root_pose_to_sim(state[:,:7]);self.hand.write_root_velocity_to_sim(state[:,7:])

    def observations(self):
        control=ensure(self);self._compute_intermediate_values()
        q,dq=control.joint_observations(self.hand.data.joint_pos,self.hand.data.joint_vel)
        command,_=control.joint_observations(self.prev_targets,torch.zeros_like(self.prev_targets))
        parts=[q,dq,command]
        for obj in (self.hand,)+tuple(getattr(self,name) for name in OBJECTS[task]):
            from driven_wrist_adapter import hand_state
            state=(hand_state(self) if obj is self.hand else obj.data.root_state_w).clone()
            state[:,:3]-=self.scene.env_origins;parts.append(state)
        tips=self.cartesian_ik.fingertip_positions_world()-self.scene.env_origins[:,None,:]
        parts.append(tips.reshape(self.num_envs,-1))
        if task=='strike':contact=self.tool_finger_contact_forces
        else:
            sensor=self.leaf_contact if task=='forage' else self.branch_contact_sensor if task=='branch' else self.contact_sensor
            matrix=sensor.data.force_matrix_w
            if matrix is None:raise RuntimeError('Missing filtered contact observations')
            contact=torch.linalg.vector_norm(matrix,dim=-1).sum(dim=1)
        if contact.shape!=(self.num_envs,2 if task=='forage' else 5):
            raise RuntimeError(f'Unexpected contact observation shape {contact.shape}')
        parts.append(contact)
        morphology=torch.tensor([list(j.axis)+list(j.origin_xyz)+list(j.origin_rpy)+[j.lower,j.upper,j.velocity,j.effort,j.depth] for j in graph.joints],device=self.device,dtype=q.dtype).reshape(1,-1)
        parts.append(morphology.expand(self.num_envs,-1))
        zero=torch.zeros(self.num_envs,device=self.device)
        parts.append(torch.stack((self.episode_length_buf*self.step_dt,
            getattr(self,'tool_was_held',zero).float(),getattr(self,'tool_grasp_streak',zero).float(),
            getattr(self,'tool_grasp_missing_steps',zero).float()),dim=-1))
        parts.append(self.graph_actions)
        obs=torch.cat(parts,dim=-1)
        if obs.shape[-1]!=cfg.observation_space:raise RuntimeError('Graph observation contract mismatch')
        return {'policy':obs}

    cls._reset_idx=reset;cls._pre_physics_step=pre;cls._apply_action=apply;cls._get_observations=observations
    cls._joint_graph_installed=True
    return graph

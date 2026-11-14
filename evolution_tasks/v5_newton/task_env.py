"""Unified tensor API for Newton tasks. New observation signature; no legacy policy reuse.

Material calibration and geometric reset parity are separate acceptance gates.
This adapter exposes actual simulation state and never labels a milestone Strike success.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
import torch
from frame_math import rotate_xyzw,inverse_rotate_xyzw
from tactile_adapter import Tactile
from task_metrics import grasp_held,branch_qualified,projected_exposure,forage_qualified,update_grasp_state,strike_hit,bounded_progress

class TaskEnv:
    def __init__(self,ctx,scene,config,rows,objects,targets,seed=31,urdf=None):
        self.ctx=ctx;self.scene=scene;self.config=config;self.task=config['task'];self.hand=scene['hand']
        self.device=self.hand.device;self.num_envs=self.hand.data.joint_pos.torch.shape[0]
        self.dt=config['control_dt'];self.physics_dt=config['physics_dt'];self.max_steps=config['episode_steps']
        self.native=config['source_contract'].get('config',{});self.object_names=list(objects);self.target_names=targets
        self.tactile=Tactile(scene,self.hand,rows);self.tactile.enabled=torch.zeros(self.num_envs,device=self.device,dtype=torch.bool);self.generator=torch.Generator(device=self.device).manual_seed(seed)
        self.clearance=None
        if self.task in ('branch','strike'):
            from reset_clearance import Clearance
            self.clearance=Clearance(urdf,self.hand,config.get("collision_asset"))
        self.evaluating=False
        self.seed=seed;self.episode_counts=np.zeros(self.num_envs,dtype=int)
        self.origins=scene.env_origins.torch if hasattr(scene.env_origins,'torch') else scene.env_origins
        self.q0=self.hand.data.default_joint_pos.torch.clone();self.limits=self.hand.data.joint_pos_limits.torch[0].clone()
        self.fids=[i for i,n in enumerate(self.hand.joint_names) if not n.startswith(('diagnostic_wrist_','wrist_'))]
        self.wids=[self.hand.joint_names.index('diagnostic_wrist_'+a) for a in 'xyz']
        self.rids=[self.hand.joint_names.index(n) for n in ['wrist_roll','wrist_pitch','wrist_yaw'] if n in self.hand.joint_names]
        assert len(self.rids) in (0,3)
        self.action_size=len(self.fids)+(3 if self.wids else 0)+len(self.rids)
        self.mount_quaternion=self.hand.data.body_link_pose_w.torch[:,self.hand.body_names.index('diagnostic_stage_y'),3:].clone() if self.rids else None
        self.target=self.q0.clone();self.wrist=torch.zeros(self.num_envs,3,device=self.device)
        self.last_action=torch.zeros(self.num_envs,self.action_size,device=self.device)
        self.initial={name:self.pose(name).clone() for name in objects}
        self.age=torch.zeros(self.num_envs,device=self.device,dtype=torch.long)
        self.streak=torch.zeros_like(self.age);self.missing=torch.zeros_like(self.age);self.outside_streak=torch.zeros_like(self.age)
        self.held=torch.zeros_like(self.age,dtype=torch.bool);self.lost=self.held.clone();self.rewarded=self.held.clone()
        self.seen=torch.zeros(self.num_envs,2,device=self.device,dtype=torch.bool)
        self.best=torch.zeros(self.num_envs,7,device=self.device);self.progress=self.best.clone()
        self.goals=torch.zeros(self.num_envs,device=self.device);self.hold=torch.zeros_like(self.goals)
        self.initial_bottom=torch.zeros_like(self.goals);self.previous_branch=None
        self.tipids=[self.hand.body_names.index(n) for n in ['link_1_3','link_2_2','link_3_2','link_4_2','link_5_2']]
        self.rootid=self.hand.body_names.index('link_0_0')
        self.skin_observation=None
        if config.get('skin_observation',False):
            from skin_observation import SkinObservation
            self.skin_observation=SkinObservation(self.hand,self.tactile)
        self.fingertip_grid=None
        if config.get('fingertip_grid',False):
            from fingertip_contact_grid import FingertipContactGrid
            self.fingertip_grid=FingertipContactGrid(self)
        self.reset()
        self.observation_size=self.observe().shape[-1]
        self.contract=dict(version='newton_task_tensor_v1',task=self.task,observation_size=self.observation_size,
            action_size=self.action_size,body_names=self.hand.body_names,joint_names=self.hand.joint_names,
            targets=targets,quaternion='xyzw',config_signature=config['signature'],
            observation='joint position, velocity, target, action, root-relative object state, all-body local normal/friction, age, progress history, task history',
            legacy_checkpoint_compatible=False,material_parity_verified=False,reset_clearance_verified=self.clearance is not None)
        if self.fingertip_grid is not None:self.contract['fingertip_grid']=self.fingertip_grid.contract
        if self.skin_observation is not None:self.contract['skin_observation']=self.skin_observation.contract
        files=['task_env.py','task_scene.py','task_metrics.py','task_materials.py','tactile_adapter.py','reset_clearance.py','controller_config.py','skin_observation.py','fingertip_contact_grid.py','tactile_slip.py']
        self.contract['runtime_source_hashes']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files}
        self.signature=hashlib.sha256(json.dumps(self.contract,sort_keys=True).encode()).hexdigest()
    def tensor(self,x):return torch.as_tensor(x,device=self.device,dtype=torch.float32)
    def pose(self,name):return self.scene[name].data.root_link_pose_w.torch
    def velocity(self,name):return self.scene[name].data.root_com_vel_w.torch
    def tips(self):return self.hand.data.body_link_pose_w.torch[:,self.tipids,:3]
    def root(self):return self.hand.data.body_link_pose_w.torch[:,self.rootid]
    def put(self,name,pose,ids):
        self.scene[name].write_root_link_pose_to_sim_index(root_pose=pose,env_ids=ids)
        self.scene[name].write_root_com_velocity_to_sim_index(root_velocity=torch.zeros(len(ids),6,device=self.device),env_ids=ids)
    def write_q(self,q,ids):
        self.hand.write_joint_state_to_sim_index(position=q,velocity=torch.zeros_like(q),env_ids=ids)
        self.hand.set_joint_position_target_index(target=q,env_ids=ids)
        self.scene.update(0.)
    def bottom(self,name,size):
        pose=self.pose(name);basis=torch.eye(3,device=self.device)
        columns=rotate_xyzw(pose[:,None,3:],basis[None])
        extent=(columns[:,:,2].abs()*self.tensor(size)[None]/2).sum(-1)
        return pose[:,2]-extent
    def reset(self,ids=None):
        ids=torch.arange(self.num_envs,device=self.device) if ids is None else torch.as_tensor(ids,device=self.device,dtype=torch.long)
        if len(ids)==0:return self.observe()
        q=self.q0[ids].clone();noise=float(self.native.get('reset_dof_pos_noise',0.))
        if self.task=='branch':
            q[:,self.fids]+=(torch.rand(len(ids),len(self.fids),device=self.device,generator=self.generator)*2-1)*noise*(self.limits[self.fids,1]-self.limits[self.fids,0])
        elif self.task=='forage':
            lo=self.limits[:,0]-q;hi=self.limits[:,1]-q
            q+=noise*(lo+(hi-lo)*torch.rand(q.shape,device=self.device,generator=self.generator))
        if self.task=='strike':
            q[:,self.fids]=self.limits[self.fids,0]+(.5*(float(self.native['pregrasp_action'])+1))*(self.limits[self.fids,1]-self.limits[self.fids,0])
            thumb=self.hand.joint_names.index('link_1_thumb_spread_joint')
            q[:,thumb]=.8*float(self.limits[thumb,1])/(2*math.pi/3)
        if self.wids:q[:,self.wids]=0
        if self.rids:q[:,self.rids]=0
        q=q.clamp(self.limits[:,0],self.limits[:,1]);self.write_q(q,ids)
        self.target[ids]=q;self.wrist[ids]=0;self.last_action[ids]=0
        for name in self.object_names:
            self.put(name,self.initial[name][ids].clone(),ids)
            self.scene[name].permanent_wrench_composer.reset(env_ids=ids)
        if self.task=='grasp':
            pose=self.initial['Object'][ids].clone()
            cases=[(0,0,0),(-.5,0,0),(.5,0,0),(0,-.5,0),(0,.5,0),(0,0,-2),(0,0,2),(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(-.5,-.5,-1),(.5,.5,1)]
            rng=np.random.default_rng(self.seed)
            all_cases=np.array([cases[i] if i<len(cases) else (*rng.uniform(-.5,.5,2),rng.uniform(-1.5,1.5)) for i in range(self.num_envs)])
            perturb=self.tensor(all_cases[ids.cpu().numpy()]);pose[:,:2]+=perturb[:,:2]/1000
            yaw=perturb[:,2]*math.pi/180;pose[:,3:]=0;pose[:,5]=torch.sin(yaw/2);pose[:,6]=torch.cos(yaw/2)
            self.put('Object',pose,ids)
            levels=self.tensor(self.config['goals']['height_levels_m']);self.goals[ids]=levels[ids%len(levels) if self.evaluating else torch.randint(len(levels),(len(ids),),device=self.device,generator=self.generator)]
        elif self.task=='branch':
            plan=self.config['reset']['placement'];pose=self.initial['branch'][ids].clone();centers=[]
            for i in ids.tolist():
                rng=np.random.default_rng(np.random.SeedSequence([self.seed,i,int(self.episode_counts[i])]))
                centers.append(np.asarray(plan['center_world'])+np.asarray(plan['anatomical_rotation'])@rng.uniform(-.002,.002,3))
            pose[:,:3]=self.origins[ids]+self.tensor(np.array(centers));quat=plan['quat_world'];pose[:,3:]=self.tensor(quat[1:]+quat[:1])
            for row,i in enumerate(ids.tolist()):
                for attempt in range(33):
                    if self.clearance.branch_valid(i,pose[row].cpu().numpy(),self.native['branch_cfg']['spawn']['radius'],self.native['branch_cfg']['spawn']['height']):break
                    if attempt==32:raise RuntimeError('No nonpenetrating Branch initial sample after 32 retries')
                    rng=np.random.default_rng(np.random.SeedSequence([self.seed,i,int(self.episode_counts[i]),attempt+1]))
                    pose[row,:3]=self.origins[i]+self.tensor(np.asarray(plan['center_world'])+np.asarray(plan['anatomical_rotation'])@rng.uniform(-.002,.002,3))
                    q[row]=self.q0[i];q[row,self.fids]+=(torch.rand(len(self.fids),device=self.device,generator=self.generator)*2-1)*noise*(self.limits[self.fids,1]-self.limits[self.fids,0])
                    q[row]=q[row].clamp(self.limits[:,0],self.limits[:,1]);self.write_q(q[row:row+1],ids[row:row+1]);self.target[i]=q[row]
            self.put('branch',pose,ids)
        elif self.task=='forage':
            center=self.tips()[ids,1:].mean(1)
            for name,offset in [('food',(0,0,.08)),('leaf_one',(-.018,.006,.1022)),('leaf_two',(.018,.006,.1064))]:
                pose=self.initial[name][ids].clone();pose[:,:2]=center[:,:2]+self.tensor(offset[:2]);pose[:,2]=self.origins[ids,2]+offset[2];self.put(name,pose,ids)
        else:
            # Use closure envelope, never the nominal cfg tool position.
            closure=q.clone();closure[:,self.fids]=self.limits[self.fids,0]+.85*(self.limits[self.fids,1]-self.limits[self.fids,0])
            thumb=self.hand.joint_names.index('link_1_thumb_spread_joint');closure[:,thumb]=1.1*float(self.limits[thumb,1])/(2*math.pi/3)
            self.write_q(closure,ids);tips=self.tips()[ids];thumb=tips[:,0,:2];error=torch.full((len(ids),),float('inf'),device=self.device);center=tips[:,0].clone()
            radius=float(self.native['Cone_cfg']['spawn']['size'][0])/2+.006
            for first in range(1,5):
                for second in range(first+1,5):
                    b=tips[:,first,:2]-thumb;c=tips[:,second,:2]-thumb;det=2*(b[:,0]*c[:,1]-b[:,1]*c[:,0]);safe=torch.where(det.abs()>1e-7,det,torch.ones_like(det))
                    bs=(b*b).sum(-1);cs=(c*c).sum(-1)
                    xy=thumb+torch.stack(((bs*c[:,1]-cs*b[:,1])/safe,(cs*b[:,0]-bs*c[:,0])/safe),-1)
                    candidate=torch.cat((xy,tips[:,[0,first,second],2].mean(1,keepdim=True)),1)
                    delta=torch.where(det.abs()>1e-7,((xy-thumb).norm(dim=-1)-radius).abs(),float('inf'));better=delta<error
                    center=torch.where(better[:,None],candidate,center);error=torch.minimum(error,delta)
            self.write_q(q,ids);pose=self.initial['Cone'][ids].clone();pose[:,:3]=center
            for row,i in enumerate(ids.tolist()):pose[row,:3]+=self.tensor(self.clearance.strike_offset(i,pose[row].cpu().numpy(),self.native['Cone_cfg']['spawn']['size']))
            center=pose[:,:3];self.put('Cone',pose,ids)
            support=self.initial['tool_support'][ids].clone();support[:,:3]=center;support[:,2]=self.bottom('Cone',self.native['Cone_cfg']['spawn']['size'])[ids]-.01;self.put('tool_support',support,ids)
            self.initial_bottom[ids]=self.bottom('Cone',self.native['Cone_cfg']['spawn']['size'])[ids]
        for value in [self.age,self.streak,self.missing,self.outside_streak,self.held,self.lost,self.rewarded,self.seen,self.best,self.progress,self.hold]:value[ids]=0
        for sensor in self.scene.sensors.values():sensor.reset(env_ids=ids)
        self.tactile.enabled[ids]=False
        if self.fingertip_grid is not None:self.fingertip_grid.reset(ids)
        self.scene.write_data_to_sim();self.scene.update(0.)
        self.episode_counts[ids.cpu().numpy()]+=1
        if self.task=='branch':
            if self.previous_branch is None:self.previous_branch=self.pose('branch').clone()
            self.previous_branch[ids]=self.pose('branch')[ids]
        if not torch.allclose(self.hand.data.joint_pos.torch[ids],q,atol=1e-6):raise RuntimeError('Reset posture mismatch')
        if self.task!='grasp':
            names=['streak','missing','held','lost','rewarded','seen','best','progress','outside_streak']
            saved={name:getattr(self,name).clone() for name in names}
            previous=None if self.previous_branch is None else self.previous_branch.clone()
            self.compute_reward();initial_progress=self.progress[ids].clone()
            for name,value in saved.items():setattr(self,name,value)
            self.previous_branch=previous;self.best[ids]=initial_progress;self.progress[ids]=initial_progress
        return self.observe()
    def observe(self):
        q=self.hand.data.joint_pos.torch;dq=self.hand.data.joint_vel.torch;lo=self.limits[:,0];span=(self.limits[:,1]-lo).clamp_min(1e-6)
        root=self.root();parts=[2*(q-lo)/span-1,dq,2*(self.target-lo)/span-1,self.last_action,self.wrist]
        for name in self.object_names:
            pose=self.pose(name);parts.extend([inverse_rotate_xyzw(root[:,3:],pose[:,:3]-root[:,:3]),pose[:,3:],self.velocity(name)])
        normal,friction=self.tactile.local_forces();parts.extend([normal.flatten(1)/5,friction.flatten(1)/5,
            self.age[:,None]/self.max_steps,self.tactile.enabled[:,None].float(),self.progress,self.best,self.streak[:,None]/180,self.hold[:,None]/3,
            self.seen.float(),self.held[:,None].float(),self.lost[:,None].float(),self.rewarded[:,None].float(),self.missing[:,None]/10,self.outside_streak[:,None]/10,self.goals[:,None]/.05,self.initial_bottom[:,None]])
        if self.task=='grasp':
            rise=self.pose('Object')[:,2]-self.initial['Object'][:,2]
            parts.extend([(self.goals-rise)[:,None]/.05,torch.full((self.num_envs,1),self.config['goals']['tolerance_m']/.05,device=self.device)])
        elif self.task=='strike':
            target=self.origins+self.tensor(self.native['target_position'])
            tool=self.pose('Cone');tip=tool[:,:3]+rotate_xyzw(tool[:,3:],self.tensor([0,0,-self.native['tool_impact_offset']]).expand(self.num_envs,-1))
            parts.extend([inverse_rotate_xyzw(root[:,3:],target-root[:,:3])/.1,(target-tip)/.1,
                self.tensor([self.native['success_distance'],self.native['success_force_threshold']]).expand(self.num_envs,-1)])
        elif self.task=='branch':
            parts.append(self.tensor([self.native['branch_contact_force_threshold'],self.native['branch_success_hold_steps']*self.dt]).expand(self.num_envs,-1))
        else:
            parts.append(self.tensor([.98,.15,30*self.dt]).expand(self.num_envs,-1))
        if getattr(self,'skin_observation',None) is not None:parts.append(self.skin_observation.vector())
        if self.fingertip_grid is not None:parts.append(self.fingertip_grid.vector())
        obs=torch.cat(parts,-1)
        if not torch.isfinite(obs).all():raise RuntimeError('Nonfinite observation')
        return obs
    def step(self,action):
        if action.shape!=(self.num_envs,self.action_size) or not torch.isfinite(action).all():raise ValueError('Invalid actions')
        self.last_action=action.clamp(-1,1).clone();lo=self.limits[self.fids,0];hi=self.limits[self.fids,1]
        desired=lo+(self.last_action[:,:len(self.fids)]+1)*.5*(hi-lo)
        speed=.25 if self.task=='grasp' else .45
        if self.fingertip_grid is not None:self.fingertip_grid.begin_interval()
        for _ in range(self.config['decimation']):
            self.target[:,self.fids]+=(desired-self.target[:,self.fids]).clamp(-speed*self.physics_dt,speed*self.physics_dt)
            if self.wids:
                bounds=self.tensor([.12,.12,.12]);rate=.02
                desired_w=self.last_action[:,len(self.fids):len(self.fids)+3]*bounds
                self.wrist+=(desired_w-self.wrist).clamp(-rate*self.physics_dt,rate*self.physics_dt)
                local=self.wrist  # mount-relative displacement
                self.target[:,self.wids]=local.clamp(self.limits[self.wids,0],self.limits[self.wids,1])
            if self.rids:
                desired_r=self.last_action[:,-3:]*self.tensor([.5,.5,.5])
                self.target[:,self.rids]+=(desired_r-self.target[:,self.rids]).clamp(-self.physics_dt,self.physics_dt)
            self.hand.set_joint_position_target_index(target=self.target)
            if self.task=='grasp' and self.config['source_contract']['args'].get('push',False):
                force=torch.zeros(self.num_envs,1,3,device=self.device);torque=torch.zeros_like(force)
                pulse=(self.age>=1080)&(self.age<1110);index=torch.arange(self.num_envs,device=self.device)
                force[:,0,0]=pulse*(index%2*2-1)*.15;torque[:,0,1]=pulse*(index%3-1)*.0005
                self.scene['Object'].set_external_force_and_torque(force,torque,is_global=True)
            elif self.task=='forage':
                for name in ['leaf_one','leaf_two']:
                    cfg=self.native[name+'_cfg']['spawn'];damping=(cfg.get('rigid_props') or {}).get('linear_damping') or 0.
                    mass=cfg['mass_props']['mass'];force=-mass*damping*self.velocity(name)[:,:3,None].transpose(1,2)
                    self.scene[name].set_external_force_and_torque(force,torch.zeros_like(force),is_global=True)
            self.scene.write_data_to_sim();self.ctx.step(render=False);self.scene.update(self.physics_dt)
            if self.fingertip_grid is not None:self.fingertip_grid.add_substep()
            if getattr(self,"contact_trace",None) is not None:
                self.contact_trace.append(self.fingertip_grid.raw_sample())
        self.tactile.enabled[:]=True
        if self.fingertip_grid is not None:self.fingertip_grid.finish_interval(self.config['decimation'])
        self.age+=1
        reward,event,failed=self.compute_reward()
        truncated=self.age>=self.max_steps;terminated=event|failed
        terminal_obs=self.observe().clone();done=terminated|truncated
        info=dict(episode_success=event.clone(),terminal_observation=terminal_obs,held=self.held.clone())
        if getattr(self,"record_terminal_evidence",False):
            info["physics"]=dict(age=self.age.clone(),streak=self.streak.clone(),hold_s=self.hold.clone(),
                seen=self.seen.clone(),force=self.tactile.digit_forces().clone(),
                objects={name:self.pose(name).clone() for name in self.object_names})
        if done.any():self.reset(done.nonzero().flatten())
        return self.observe(),reward,terminated,truncated,info
    def compute_reward(self):
        n=self.num_envs;zero=torch.zeros(n,device=self.device,dtype=torch.bool)
        force=self.tactile.digit_forces();vectors=self.tactile.digit_vectors()[:,:,0];tips=self.tips()
        cost=self.dt*(.01+.05*((self.tactile.read_targets()[0].norm(dim=-1).flatten(1).amax(-1)-40).clamp_min(0)/40).square().clamp_max(5))
        if self.task=='grasp':
            pose=self.pose('Object');rise=pose[:,2]-self.initial['Object'][:,2];speed=self.velocity('Object')[:,:3].norm(dim=-1)
            local=inverse_rotate_xyzw(pose[:,None,3:],vectors);normal=local[...,0].abs()
            support_force=self.scene['pair_contact'].data.normal_force_matrix_w.torch.norm(dim=-1).flatten(1).amax(-1)
            object_size=self.config['scene']['objects'][0]['size']; object_bottom=self.bottom('Object',object_size)
            sr=next(x for x in self.config['scene']['objects'] if x['name']=='Support'); support_top=float(sr['position'][2])+float(sr['size'][2])/2
            support_clearance=object_bottom-(self.origins[:,2]+support_top)
            support=torch.where(support_clearance>0.001,torch.zeros_like(support_force),support_force)
            upright=rotate_xyzw(pose[:,3:],self.tensor([0.,0.,1.]).expand(n,-1))[:,2]
            self.held=grasp_held(rise,support,local,upright,speed,self.goals,self.config['goals']['tolerance_m'])
            self.hold=torch.where(self.held,self.hold+self.dt,0)
            balance=torch.minimum(normal[:,0]/1.5,normal[:,1:].sum(1)/1.5).clamp(0,1)
            opposed=(local[:,0,0]*local[:,1:,0].sum(1)<0).float()
            quality=balance*opposed*torch.exp(-torch.relu(speed-.08)*10)*upright.clamp(0,1)
            points=[]
            for _,name,offset,_ in self.config['source_contract']['pad_specs']:
                p=self.hand.data.body_link_pose_w.torch[:,self.hand.body_names.index(name)]
                points.append(p[:,:3]+rotate_xyzw(p[:,3:],self.tensor(offset).expand(n,-1)))
            relative=inverse_rotate_xyzw(pose[:,None,3:],torch.stack(points,1)-pose[:,None,:3])
            reach=torch.exp(-torch.relu(relative.abs()-self.tensor([.009,.009,.09])).norm(dim=-1)/.02).mean(1)
            lift=(rise/self.goals).clamp(0,1)*torch.exp(-(rise-self.goals-self.config['goals']['tolerance_m']).clamp_min(0)/self.config['goals']['tolerance_m'])
            failed=pose[:,2]-self.origins[:,2]<.15;event=(self.age>=self.max_steps)&(self.hold>=3)&~failed
            reward=reach+balance+5*lift*quality+3*self.held.float()-.1*torch.relu(vectors.norm(dim=-1)-4).square().sum(1).clamp_max(100)-2*torch.relu(speed-.15).square().clamp_max(25)+10*event.float()-10*failed.float()
            return reward,event,failed
        if self.task=='branch':
            pose=self.pose('branch');delta=(pose[:,:3]-self.previous_branch[:,:3]).norm(dim=-1)
            angle=2*torch.acos((pose[:,3:]*self.previous_branch[:,3:]).sum(-1).abs().clamp(max=1))
            c=self.native;qualified=branch_qualified(force,self.tactile.opposition_cosine(),delta,angle,c['branch_contact_force_threshold'],c['min_long_finger_contacts'],c['branch_relative_position_tolerance'],c['branch_relative_rotation_tolerance'],c['require_pose_stability'])
            self.streak=torch.where(qualified,self.streak+1,0);self.previous_branch=pose.clone();self.held=qualified
            event=self.streak>=c['branch_success_hold_steps']
            rel=inverse_rotate_xyzw(pose[:,None,3:],tips-pose[:,None,:3]);radius=c['branch_cfg']['spawn']['radius'];height=c['branch_cfg']['spawn']['height']
            distance=(torch.relu(rel[:,:,:2].norm(dim=-1)-radius).square()+torch.relu(rel[:,:,2].abs()-height/2).square()).sqrt();near=torch.exp(-distance/.04)
            f=(force/c['branch_contact_force_threshold']).clamp(0,1);simultaneous=torch.minimum(f[:,0],f[:,1:].topk(2).values[:,-1]);opposed=simultaneous*(-self.tactile.opposition_cosine()).clamp(0,1)
            progress=torch.stack((near[:,0],near[:,1:].topk(2).values.mean(-1),f[:,0],f[:,1:].topk(2).values.mean(-1),simultaneous,opposed,(self.streak/c['branch_success_hold_steps']).clamp(0,1)),-1);weights=[.5,.5,.5,.5,1,3,4]
        elif self.task=='forage':
            leaves=torch.stack([self.pose(name) for name in ['leaf_one','leaf_two']],1);food=self.pose('food')
            sizes=self.tensor([self.native[name+'_cfg']['spawn']['size'] for name in ['leaf_one','leaf_two']]);food_size=self.tensor(self.native['food_cfg']['spawn']['size'])
            self.seen |= self.tactile.read_targets()[0][:,:,:2].norm(dim=-1).sum(1)>=self.native.get('minimum_leaf_contact_force',.05)
            speed=torch.stack([self.velocity(name)[:,:3].norm(dim=-1) for name in ['leaf_one','leaf_two']],1)
            distance=(leaves[:,:,:2]-food[:,None,:2]).norm(dim=-1)
            exposure=projected_exposure(food[:,:3],food_size,leaves[:,:,:3],leaves[:,:,3:],sizes)
            qualified=forage_qualified(exposure,self.seen,speed,distance);self.streak=torch.where(qualified,self.streak+1,0);event=self.streak>=30;self.held=self.seen.all(-1)
            clear=[];basis=torch.eye(3,device=self.device)
            for i in range(2):
                columns=rotate_xyzw(leaves[:,i,None,3:],basis[None]);extent=(columns[...,:2].abs()*sizes[i][None,:,None]/2).sum(1)
                value=((leaves[:,i,:2]-food[:,:2]).abs()/(extent+food_size[:2]/2)).amax(-1).clamp(0,1)
                clear.append(value*self.seen[:,i]/(1+(speed[:,i]/.3).square()))
            reach=torch.exp(-torch.cdist(tips,leaves[:,:,:3]).amin(1)/.08).mean(-1)
            progress=torch.stack((reach,self.seen[:,0].float(),self.seen[:,1].float(),*clear,exposure*self.seen.all(-1),(self.streak/30).clamp(0,1)),-1);weights=[1,.5,.5,1,1,4,2]
            cost+=self.dt*.05*torch.relu(speed.amax(-1)-1).square().clamp_max(5)
        else:
            c=self.native;pose=self.pose('Cone');f=(force/c['tool_finger_contact_force_threshold']).clamp(0,1)
            current=(f[:,0]>=1)&((f[:,1:]>=1).sum(-1)>=c['min_long_finger_contacts'])
            self.streak,self.missing,self.held,self.lost,self.rewarded,_=update_grasp_state(current,self.streak,self.missing,self.held,self.lost,self.rewarded,c['tool_grasp_hold_steps'],c.get('tool_grasp_loss_grace_steps',2))
            tip=pose[:,:3]+rotate_xyzw(pose[:,3:],self.tensor([0,0,-c['tool_impact_offset']]).expand(n,-1))
            target=self.origins+self.tensor(c['target_position'])
            impact=self.scene['pair_contact'].data.normal_force_matrix_w.torch[:,0,0]
            event=strike_hit(self.held,current,tip,target,impact,c['success_distance'],c['success_force_threshold'])
            near=torch.exp(-(tips-pose[:,None,:3]).norm(dim=-1)/.08).mean(-1);error=target-tip
            lift=((self.bottom('Cone',c['Cone_cfg']['spawn']['size'])-self.initial_bottom)/.01).clamp(0,1)*current
            progress=torch.stack((near,f[:,0],f[:,1:].topk(2).values[:,-1],(self.streak/c['tool_grasp_hold_steps']).clamp(0,1),lift,torch.exp(-error[:,:2].norm(dim=-1)/.05)*current,torch.exp(-error.norm(dim=-1)/.05)*current),-1);weights=[1,.5,.5,2,2,1,3]
        self.progress=progress
        failed=zero.clone()
        if self.task=='strike':
            pos=self.pose('Cone')[:,:3]-self.origins;target=self.tensor(self.native['target_position']).expand(self.num_envs,-1)
            outside=((pos[:,:2]-target[:,:2]).norm(dim=-1)>=self.native['workspace_xy_radius'])|(pos[:,2]<=self.native['workspace_min_height'])|(pos[:,2]>=self.native['workspace_max_height'])
            self.outside_streak=torch.where(outside,self.outside_streak+1,0);failed=self.outside_streak>=math.ceil(.10/self.dt)
        elif self.task=='forage':
            outside=(self.pose('food')[:,:2]-self.origins[:,:2]).norm(dim=-1)>.22
            self.outside_streak=torch.where(outside,self.outside_streak+1,0)
            failed=(distance>.5).any(-1)|(self.outside_streak>=math.ceil(.10/self.dt))
        reward,self.best=bounded_progress(progress,self.best,self.tensor(weights),event,cost)
        if not torch.isfinite(reward).all():raise RuntimeError('Nonfinite reward')
        return reward,event,failed

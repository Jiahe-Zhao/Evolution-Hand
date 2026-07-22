# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause



from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.sensors import ContactSensor,ContactSensorCfg

if TYPE_CHECKING:
    from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.evolution_tasks.task_strike.evolution_strike_env_cfg import EvolutionStrikeEnvCfg

class EvolutionStrikeEnv(DirectRLEnv):
    cfg:EvolutionStrikeEnvCfg

    def __init__(self, cfg: EvolutionStrikeEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # self.cfg.observation_space=155
        # print(self.cfg.observation_space)
        self.num_hand_dofs = self.hand.num_joints
        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        
        # 锥形物体的初始状态
        self.conical_object_initial_state = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        #击打目标物体的受力
        self.strike_object_force=torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        #目标受力
        self.target_force=-10
        # 手的初始状态
        # self.hand_initial_state = torch.zeros((self.num_envs, 13), dtype=torch.float, device=self.device)
        # self.hand_initial_state=self.hand.data.default_root_state.clone()

        # angle = torch.tensor(-np.pi / 2, device=self.device)
        # self.hand_initial_state[:, 3:7] = torch.tensor([0.70710678, 0.0, 0.70710678, 0.0], dtype=torch.float, device=self.device)

        # self.hand.write_root_state_to_sim(self.hand_initial_state)
        # # 地面受力
        # self.ground_force = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # # 地面目标点
        # self.ground_target_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.ground_target_pos[:, 2] = -0.01  # 假设地面目标点在地面下方0.01米处

        # used to compare object position
        self.in_hand_pos = self.cone.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04
        # print("in_hand_pos:",self.in_hand_pos)
        # initialize goal marker
        # self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        
        self.hand = Articulation(self.cfg.robot_cfg)
        # self.object = RigidObject(self.cfg.object_cfg)
        self.cone=RigidObject(self.cfg.Cone_cfg)
        self.strike_object=RigidObject(self.cfg.strike_object_cfg)
        self.contact_sensor=ContactSensor(self.cfg.contact_sensor_cfg)
        
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["cone"] = self.cone
        self.scene.rigid_objects["strike_object"] = self.strike_object
        
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        if self.contact_sensor is None:
            print("Contact sensor initialization failed!")
        else:
            print("Contact sensor initialized successfully.")
        
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        if self.cfg.asymmetric_obs:
            self.fingertip_force_sensors = self.hand.root_physx_view.get_link_incoming_joint_force()[
                :, self.finger_bodies
            ]

        if self.cfg.obs_type == "openai":
            obs = self.compute_reduced_observations()
        elif self.cfg.obs_type == "full":
            obs = self.compute_full_observations()
        else:
            print("Unknown observations type!")

        if self.cfg.asymmetric_obs:
            states = self.compute_full_state()

        observations = {"policy": obs}
        if self.cfg.asymmetric_obs:
            observations = {"policy": obs, "critic": states}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:]
        )= compute_rewards(
                self.reset_buf,
                self.reset_goal_buf,
                self.successes,
                self.consecutive_successes,
                self.max_episode_length,
                self.cone_pos,
                self.strike_object_force,
                self.in_hand_pos,
                self.target_force,    # print("z_force:",z_force)

                self.cfg.dist_reward_scale,
                self.cfg.force_reward_scale,
                self.actions,
                self.cfg.action_penalty_scale,
                self.cfg.success_tolerance,
                self.cfg.reach_goal_bonus,
                self.cfg.fall_dist,
                self.cfg.fall_penalty,
                self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # reset goals if the goal has been reached
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(goal_env_ids) > 0:
        #     self._reset_target_pose(goal_env_ids)

        return total_reward
    
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cone has fallen
        goal_dist = torch.norm(self.cone_pos - self.in_hand_pos, p=2, dim=-1)
        # print("1111111111:",goal_dist)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            # 计算 Z 方向的受力差异
            z_force = self.strike_object_force[:, 2]  # 提取 Z 方向的受力
            z_force_diff = torch.abs(z_force - self.target_force)
            self.episode_length_buf = torch.where(
                torch.abs(z_force_diff) <= self.cfg.success_tolerance,
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            )
            max_success_reached = self.successes >= self.cfg.max_consecutive_success
        #时间超过最大时间长度
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return out_of_reach, time_out
    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES.tolist()
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals buff
        # self._reset_target_pose(env_ids)
        self.reset_goal_buf[env_ids] = 0

        # reset cone
        cone_default_state = self.cone.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        cone_default_state[:, 0:3] = (
            cone_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        # rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # cone_default_state[:, 3:7] = randomize_rotation(
        #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )

        cone_default_state[:, 7:] = torch.zeros_like(self.cone.data.default_root_state[env_ids, 7:])
        self.cone.write_root_state_to_sim(cone_default_state, env_ids)

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()

    #更新环境类中的属性
    def _compute_intermediate_values(self):
        # 手的数据
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel
        
        #目标物体的位置
        #self.strike_object_pos=self.strike_object.data.root_pos_w-self.scene.env_origins

         # 锥体数据
        self.cone_pos = self.cone.data.root_pos_w - self.scene.env_origins  # 锥体的位置
        self.cone_rot = self.cone.data.root_quat_w  # 锥体的旋转
        self.cone_velocities = self.cone.data.root_vel_w  # 锥体的速度
        self.cone_linvel = self.cone.data.root_lin_vel_w  # 锥体的线速度
        self.cone_angvel = self.cone.data.root_ang_vel_w  # 锥体的角速度
        # 物体的受力数据
        # print("force_matrix_w:",self.contact_sensor.data.force_matrix_w.shape)
        if self.contact_sensor.data.force_matrix_w is not None:
            # print(self.contact_sensor.data.force_matrix_w)  # 查看数据是否有效
               self.strike_object_force = self.contact_sensor.data.force_matrix_w[:,0,0,:]
        else:
            # print("No contact forces detected.")
            self.strike_object_force = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # print("strike_object_force:",self.strike_object_force.shape)
    #简化观测值
    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.cone_pos,
                self.strike_object_force,
                # quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )
        return obs

    def compute_full_observations(self):
        obs = torch.cat(
            (
                # hand
                #len(joint)
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                #len(joint)
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # cone
                #3
                self.cone_pos,
                #4
                self.cone_rot,
                #3
                self.cone_linvel,
                #3
                self.cone_angvel,
                # self.cfg.vel_obs_scale * self.object_angvel,
                # strike_object
                # self.strike_object_force.view(self.num_envs,3),
                #3
                self.strike_object_force,
                # fingertips len(links)*13
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions len(joint)
                self.actions,
                
            ),
            dim=-1,
        )
        
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # cone
                self.cone_pos,
                self.cone_rot,
                self.cone_velocities, 
                self.cone_linvel,
                self.cone_angvel,
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                self.cfg.force_torque_obs_scale
                * self.fingertip_force_sensors.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return states
       
@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
#reward:锥体相对与手的位置；（锥体的旋转？）；物体受力大小；动作平滑度；  成功条件：受力大小在一定阈值内  失败条件：锥体脱离手
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor, #敲击的了的大小在一定阈值内算成功
    successes: torch.Tensor,    #成功的次数
    consecutive_successes: torch.Tensor, #连续成功的此书
    max_episode_length: float,  #每个回合的最大步数（？）没有设置？
    # object_pos: torch.Tensor, #物体的位置
    # object_rot: torch.Tensor,   #物体的旋转
    # strink_object_pos:torch.Tensor,#被敲击物体的位置
    cone_pos:torch.Tensor,      #锥体的位置
    object_force:torch.Tensor,  #物体的受力
    target_pos: torch.Tensor,   #目标位置 
    target_force: int,   #目标受力
    dist_reward_scale: float,   #距离奖励因子
    force_reward_scale: float,    #受力奖励因子
    # rot_eps: float, #用于防止除零错误的一个小值（？）
    actions: torch.Tensor,  #动作  
    action_penalty_scale: float,    #动作惩罚因子
    success_tolerance: float,   #受力成功的容忍度
    reach_goal_bonus: float,    #达到目标时的奖励
    fall_dist: float,   #掉落阈值
    fall_penalty: float,    #掉落惩罚
    av_factor: float,   #用于平滑连续成功的奖励
):
    #增加中间奖励
    #当锥体靠近被敲击物体位置时，给予奖励
    #物体的位置：pos=(0.0, -0.39, 0), rot=(1.0, 0.0, 0.0, 0.0)
    #print("--------------:",cone_pos,cone_pos.shape)
    distance_to_target = torch.norm(cone_pos - target_pos, p=2, dim=-1)
    proximity_reward = -distance_to_target * dist_reward_scale
    
    #当施加的力接近目标力时，给予奖励
    force_diff = torch.abs(object_force[:, 2] - target_force)
    force_reward = -force_diff * force_reward_scale
    
    # print("cone_pos",cone_pos)
    goal_dist = torch.norm(cone_pos - target_pos, p=2, dim=-1)
    # print("goal_dist:",goal_dist)
    dist_rew = goal_dist * dist_reward_scale
    # print(dist_rew.shape)
    # rot_dist = rotation_distance(object_rot, target_rot)
    # 计算 Z 方向的受力差异
    # print(object_force.shape)
    z_force = object_force[:, 2]  # 提取 Z 方向的受力
    # print("z_force:",z_force)
    z_force_diff = torch.abs(z_force - target_force)  # 计算受力差异
    # 根据差异计算受力奖励
    z_force_rew = z_force_diff * force_reward_scale
    # print(z_force_rew.shape)
    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    #reward = proximity_reward+force_reward+dist_rew + z_force_rew + action_penalty * action_penalty_scale
    reward = dist_rew + z_force_rew + action_penalty * action_penalty_scale
    #reward = force_reward+ dist_rew+action_penalty * action_penalty_scale
    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(z_force_diff) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threshold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes
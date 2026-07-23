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
    from isaaclab_tasks.evolution_tasks.task_grasp.evolution_grasp_env_cfg import EvolutionGraspEnvCfg


def _object_in_palm_region(object_pos_w, hand_pos_w, hand_quat_w, center, half_extents):
    """Classify the object in a hand-root local oriented palm box."""
    relative_w = object_pos_w - hand_pos_w
    inverse_vec = -hand_quat_w[:, 1:4]
    tangent = 2.0 * torch.cross(inverse_vec, relative_w, dim=-1)
    relative_local = relative_w + hand_quat_w[:, :1] * tangent + torch.cross(inverse_vec, tangent, dim=-1)
    centered = torch.abs(relative_local - center)
    return torch.all(centered <= half_extents, dim=-1)

class EvolutionGraspEnv(DirectRLEnv):
    cfg:  EvolutionGraspEnvCfg

    def __init__(self, cfg: EvolutionGraspEnvCfg  , render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

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
        # used to compare object position
        self.in_hand_pos = self.grasp_object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04
        #击打目标物体的受力
        self.grasp_object_force=torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        #目标受力
        self.target_force=10

        # #施加外力
        # force=torch.tensor([[[0.0, 0.0, 10.0]]])
        # torque = torch.tensor([[[0.0, 0.0, 0.0]]])
        # self.grasp_object.set_external_force_and_torque(forces=force, torques=torque)
        force_value = [0.0, 0.0, -10.0]
        force=torch.tensor(force_value).repeat(self.num_envs, self.grasp_object.num_bodies, 1)
        torque_value= [0.0,0.0,0.0]
        torque=torch.tensor(torque_value).repeat(self.num_envs, self.grasp_object.num_bodies, 1)
        # print("force_shape:",force.shape)
        # print("torque_shape:",torque.shape)
        self.grasp_object.set_external_force_and_torque(forces=force, torques=torque)
        self.grasp_object.write_data_to_sim()


        # # default goal positions
        # self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # self.goal_rot[:, 0] = 1.0
        # self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        # initialize goal marker
        # self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.success_streaks = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        print(f"init finsh")

    def _setup_scene(self):
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)
        self.grasp_object=RigidObject(self.cfg.grasp_object_cfg)
        self.contact_sensor=ContactSensor(self.cfg.contact_sensor_cfg)
        
        

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["grasp_object"] = self.grasp_object
        self.scene.sensors["contact_sensor"] = self.contact_sensor
        if self.contact_sensor is None:
            print("Contact sensor initialization failed!")
        else:
            print("Contact sensor initialized successfully.")

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        print(f"setup finsh")
        # self.scene.write_data_to_sim()
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
        # print(f"obsser:{observations}")
        return observations
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.success_streaks[:],
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.success_streaks,
            self.consecutive_successes,
            self.max_episode_length,
            self.grasp_object_pos,
            self.grasp_object_angvel,
            self.grasp_object_force,
            self.object_in_palm,
            self.in_hand_pos,
            self.target_force,
            self.cfg.dist_reward_scale,
            self.cfg.angle_reward_scale,
            self.cfg.force_reward_scale,
            self.actions,
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            getattr(self.cfg, "min_success_hold_steps", 10),
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
            self.cfg.palm_region_reward_scale,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()
        self.extras["log"]["success_streaks"] = self.success_streaks.mean()
        self.extras["log"]["palm_region_occupancy"] = self.object_in_palm.float().mean()

        # # reset goals if the goal has been reached
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(goal_env_ids) > 0:
        #     self._reset_target_pose(goal_env_ids)

        return total_reward
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.grasp_object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.cfg.fall_dist

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            # 计算 Z 方向的受力差异
            z_force = self.grasp_object_force[:, 2]  # 提取 Z 方向的受力
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

        # reset goals
        self.reset_goal_buf[env_ids] = 0

        # reset object
        object_default_state = self.grasp_object.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        # rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # object_default_state[:, 3:7] = randomize_rotation(
        #     rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )

        object_default_state[:, 7:] = torch.zeros_like(self.grasp_object.data.default_root_state[env_ids, 7:])
        self.grasp_object.write_root_state_to_sim(object_default_state, env_ids)

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
        self.success_streaks[env_ids] = 0
        self._compute_intermediate_values()
    def _compute_intermediate_values(self):
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        
         # 物体数据
        self.grasp_object_pos = self.grasp_object.data.root_pos_w - self.scene.env_origins  # 物体的位置
        self.grasp_object_rot = self.grasp_object.data.root_quat_w  # 物体的旋转
        self.grasp_object_velocities = self.grasp_object.data.root_vel_w  # 物体的速度
        self.grasp_object_linvel = self.grasp_object.data.root_lin_vel_w  # 物体的线速度
        self.grasp_object_angvel = self.grasp_object.data.root_ang_vel_w  # 物体的角速度
        palm_center = torch.tensor(self.cfg.palm_region_center, dtype=torch.float, device=self.device)
        palm_half_extents = torch.tensor(self.cfg.palm_region_half_extents, dtype=torch.float, device=self.device)
        self.object_in_palm = _object_in_palm_region(
            self.grasp_object.data.root_pos_w,
            self.hand.data.root_pos_w,
            self.hand.data.root_quat_w,
            palm_center,
            palm_half_extents,
        )

        # 物体的受力数据 ？？ z轴吗
        # print("force_matrix_w:",self.contact_sensor.data.force_matrix_w.shape)
        if self.contact_sensor.data.net_forces_w is not None:
            # print(self.contact_sensor.data.force_matrix_w)  # 查看数据是否有效
               self.grasp_object_force = self.contact_sensor.data.net_forces_w[:,0,:]
            #    print("strike_object_force:",self.grasp_object_force)

        else:
            # print("No contact forces detected.")
            self.grasp_object_force = torch.ones((self.num_envs, 3), dtype=torch.float, device=self.device)
        # print("strike_object_force:",self.grasp_object_force)
    def compute_reduced_observations(self):
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.grasp_object_pos,
                self.grasp_object_force,
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
                # grasp object
                #(3)
                self.grasp_object_pos,
                #(4)
                self.grasp_object_rot,
                #(3)
                self.grasp_object_linvel,
                #(3)
                self.cfg.vel_obs_scale*self.grasp_object_angvel,
                # force (3)
                self.grasp_object_force,
                # fingertips
                #len(links)*3
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                #len(links)*4
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                #len(links)*6
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions len(joint)
                self.actions,
            ),
            dim=-1,
        )
        # 看报错
        # obs = torch.cat(
        #     (
        #         obs,  # 原始 155 维观测值
        #         torch.zeros((obs.shape[0], 2), device=obs.device)  # 补充 2 个维度为 0
        #     ),
        #     dim=-1
        # )
        # print("Observation shape:", obs.shape)
        return obs

    def compute_full_state(self):
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # grasp object
                self.grasp_object_pos,
                self.grasp_object_rot,
                
                self.grasp_object_linvel,
                self.grasp_object_angvel,
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
#reward:物体位置的变化；受力的变化；球体是否旋转；
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor, #是否成功达到目标 是否接近理想的受力指/持续一段时间稳定
    successes: torch.Tensor,    #成功的次数
    success_streaks: torch.Tensor, #连续命中受力窗口的步数
    consecutive_successes: torch.Tensor, #连续成功的此书
    max_episode_length: float,  #每个回合的最大步数
    object_pos: torch.Tensor, #物体的位置
    object_angvel: torch.Tensor,   #物体的角速度/角加速度（？）
    object_force:torch.Tensor,  #物体的受力
    object_in_palm: torch.Tensor,
    target_pos: torch.Tensor,   #目标位置 
    target_force: int,   #目标受力
    dist_reward_scale: float,   #距离奖励因子
    angvel_reward_scale:float,  #角速度受力因子
    force_reward_scale: float,    #受力奖励因子
    # rot_eps: float, #用于防止除零错误的一个小值
    actions: torch.Tensor,  #动作  
    action_penalty_scale: float,    #动作惩罚因子
    success_tolerance: float,   #旋转成功的容忍度
    min_success_hold_steps: int, #连续命中多少步后才开始计入奖励
    reach_goal_bonus: float,    #达到目标时的奖励
    fall_dist: float,   #掉落阈值
    fall_penalty: float,    #掉落惩罚
    av_factor: float,   #用于平滑连续成功的奖励
    palm_region_reward_scale: float,
):
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    z_force = object_force[:, 2]  # 提取 Z 方向的受力
    z_force_diff = torch.abs(z_force - target_force)  # 计算受力差异

    # Only count a "successful" grasp signal after the force window is held
    # for several consecutive control steps.
    hit_goal = (torch.abs(z_force_diff) <= success_tolerance) & object_in_palm
    success_streaks = torch.where(hit_goal, success_streaks + 1.0, torch.zeros_like(success_streaks))
    qualified_hits = success_streaks >= float(min_success_hold_steps)
    goal_resets = torch.where(qualified_hits, torch.ones_like(reset_goal_buf), torch.zeros_like(reset_goal_buf))
    successes = successes + goal_resets

    reward = object_in_palm.float() * palm_region_reward_scale
    reward = reward + torch.where(
        qualified_hits,
        torch.full_like(goal_dist, reach_goal_bonus),
        torch.zeros_like(goal_dist),
    )

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(success_streaks * resets.float())
    success_streaks = torch.where(resets > 0, torch.zeros_like(success_streaks), success_streaks)

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, success_streaks, cons_successes

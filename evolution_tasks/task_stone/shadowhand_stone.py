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
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg


import numpy as np
from isaaclab_assets import SHADOW_HAND_CFG

class StoneGrindEnv(DirectRLEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg

    #cone_cfg
    Cone_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cone",
        spawn=sim_utils.ConeCfg(
            radius=0.02,
            height=0.20,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=500.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,  # 可以尝试增加此值
                rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.025, -0.38, 0.37),  #(0.055, -0.375, 0.42)
            rot=tuple(
                quat_from_angle_axis(
                    torch.tensor(-np.pi, dtype=torch.float32), #旋转180度
                    torch.tensor([0.0, 1.0, 0.0])) #绕y轴旋转 使尖端朝下
                    .tolist()),  
            ),#初始状态 
    )

    #grasp_object_cfg
    grasp_object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        spawn=sim_utils.SphereCfg(
            radius=0.04,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=500.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.005,  # 可以尝试增加此值
                rest_offset=0.001,     # 可以尝试增加此值
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.39, 0.6),  
            rot=(1.0,0.0,0.0,0.0),#初始状态 
        )
    )
    # contact_sensor_cfg
    contact_sensor_cfg:ContactSensorCfg=ContactSensorCfg(
        prim_path="/World/envs/env_.*/grasp_object",
        filter_prim_paths_expr=["/World/envs/env_.*/Cone"],
    )
    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        #strinkehand
        #初始化
        self.num_strinkhand_dofs = self.strinkhand.num_joints
        # buffers for position targets
        self.strinkhand_dof_targets = torch.zeros((self.num_envs, self.num_strinkhand_dofs), dtype=torch.float, device=self.device)
        self.strink_prev_targets = torch.zeros((self.num_envs, self.num_strinkhand_dofs), dtype=torch.float, device=self.device)
        self.strink_cur_targets = torch.zeros((self.num_envs, self.num_strinkhand_dofs), dtype=torch.float, device=self.device)
        # list of actuated joints
        self.strinkhand_actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.strinkhand_actuated_dof_indices.append(self.strinkhand.joint_names.index(joint_name))
        self.strinkhand_actuated_dof_indices.sort()

        # finger bodies
        self.strinkhand_finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.strinkhand_finger_bodies.append(self.strinkhand.body_names.index(body_name))
        self.strinkhand_finger_bodies.sort()
        self.num_fingertips = len(self.strinkhand_finger_bodies)

        # joint limits
        joint_pos_limits = self.strinkhand.root_physx_view.get_dof_limits().to(self.device)
        self.strinkhand_dof_lower_limits = joint_pos_limits[..., 0]
        self.strinkhand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        self.strinkhand_reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 锥形物体的初始状态
        self.conical_object_initial_state = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        #击打目标物体的受力
        self.strike_object_force=torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        #目标受力
        self.target_force=-10

        # used to compare object position
        self.in_strinkhand_pos = self.cone.data.default_root_state[:, 0:3].clone()
        self.in_strinkhand_pos[:, 2] -= 0.04

        # track successes
        self.strinkhand_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.strinkhand_consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        #grasphand
        self.num_grasphand_dofs = self.grasphand.num_joints
        # buffers for position targets
        self.grasphand_dof_targets = torch.zeros((self.num_envs, self.num_grasphand_dofs), dtype=torch.float, device=self.device)
        self.grasphand_prev_targets = torch.zeros((self.num_envs, self.num_grasphand_dofs), dtype=torch.float, device=self.device)
        self.grasphand_cur_targets = torch.zeros((self.num_envs, self.num_grasphand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.grasphand_actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.grasphand_actuated_dof_indices.append(self.grasphand.joint_names.index(joint_name))
        self.grasphand_actuated_dof_indices.sort()

        # finger bodies
        self.grasphand_finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.grasphand_finger_bodies.append(self.grasphand.body_names.index(body_name))
        self.grasphand_finger_bodies.sort()
        self.num_fingertips = len(self.grasphand_finger_bodies)

        # joint limits
        grasphand_joint_pos_limits = self.grasphand.root_physx_view.get_dof_limits().to(self.device)
        self.grasphand_dof_lower_limits = grasphand_joint_pos_limits[..., 0]
        self.grasphand_dof_upper_limits = grasphand_joint_pos_limits[..., 1]

        # track goal resets
        self.grasphand_reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        self.in_grasphand_pos = self.grasp_object.data.default_root_state[:, 0:3].clone()
        self.in_grasphand_pos[:, 2] -= 0.04
        #击打目标物体的受力
        self.grasp_object_force=torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        #目标受力
        self.target_force=10

        # initialize goal marker
        self.grasphand_goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        self.grasphand_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.grasphan_consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        
        self.grasphand = Articulation(self.cfg.robot_cfg)
        self.grasp_object=RigidObject(self.grasp_object_cfg)
        self.contact_sensor=ContactSensor(self.contact_sensor_cfg)

        self.cfg.robot_cfg=SHADOW_HAND_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.4),
                rot=(0.70710678, 0.0,0.70710678,  0.0),
                # rot=(1.0, 0.0,0.0,  0.0),
                joint_pos={
                    # 设置每个关节的初始角度，以模拟抓取锥体的姿态 
                    "robot0_WRJ1": 0.0, # 手腕 上下[-0.489, 0.140]
                    "robot0_WRJ0": 0.0, #手腕 左右[-0.698, 0.489]
                    #食指
                    "robot0_FFJ1": 0.2,  # 食指外关节弯曲
                    "robot0_FFJ2": 0.3,  # 食指根部弯曲
                    "robot0_FFJ3": 0.3,  # 食指根部上下 [-0.349, 0.349]
                    #中指
                    "robot0_MFJ1": 0.2,  # 中指外关节弯曲
                    "robot0_MFJ2": 0.3,  # 中指根部弯曲
                    "robot0_MFJ3": 0.3,  # 中指根部上下 [-0.349, 0.349]
                    #无名指
                    "robot0_RFJ1": 0.2,  # 无名指外关节弯曲
                    "robot0_RFJ2": 0.3,  # 无名指根部弯曲
                    "robot0_RFJ3": 0.3,  # 无名指根部上下[-0.349, 0.349]
                    #小指
                    "robot0_LFJ1": 0.3,  # 小指外关节弯曲
                    "robot0_LFJ2": 0.4,  # 小指根部弯曲
                    "robot0_LFJ3": 0.3,  # 小指根部上下 [-0.349, 0.349]
                    "robot0_LFJ4": 0.2,  
                    #拇指
                    "robot0_THJ0": -1.5,    # 拇指弯曲[-1.571, 0.000] -0.5
                    "robot0_THJ1": 0,  #  拇指关节旋转[-0.524, 0.524] 0.5
                    "robot0_THJ2": 0.2,  #  拇指关节旋转[-0.209, 0.209] -0.2
                    "robot0_THJ3": 1.2,  #  拇指根部旋转[[0.000, 1.222]
                    "robot0_THJ4": 0.8   # 拇指根部弯曲
                },
            )
        )
        self.strinkhand=Articulation(self.cfg.robot_cfg)
        self.cone=RigidObject(self.Cone_cfg)

        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.strinkhand
        self.scene.articulations["robot2"] = self.grasphand
        self.scene.rigid_objects["cone"] = self.cone
        self.scene.rigid_objects["grasp_object"] = self.grasp_object
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
            self.consecutive_successes[:],
        ) = compute_rewards(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.grasp_object_pos,
            self.grasp_object_angvel,
            self.grasp_object_force,
            self.in_hand_pos,
            self.target_force,
            self.dist_reward_scale,
            self.angle_reward_scale,
            self.force_reward_scale,
            self.actions,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # # reset goals if the goal has been reached
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(goal_env_ids) > 0:
        #     self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # reset when cube has fallen
        goal_dist = torch.norm(self.grasp_object_pos - self.in_hand_pos, p=2, dim=-1)
        out_of_reach = goal_dist >= self.fall_dist

        if self.cfg.max_consecutive_success > 0:
            # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
            # 计算 Z 方向的受力差异
            z_force = self.grasp_object_force[:, 2]  # 提取 Z 方向的受力
            z_force_diff = torch.abs(z_force - self.target_force)
            self.episode_length_buf = torch.where(
                torch.abs(z_force_diff) <= self.success_tolerance,
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
        self._compute_intermediate_values()

    # def _reset_target_pose(self, env_ids):
    #     # reset goal rotation
    #     rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
    #     new_rot = randomize_rotation(
    #         rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
    #     )

    #     # update goal pose and markers
    #     self.goal_rot[env_ids] = new_rot
    #     goal_pos = self.goal_pos + self.scene.env_origins
    #     self.goal_markers.visualize(goal_pos, self.goal_rot)

    #     self.reset_goal_buf[env_ids] = 0

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
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # grasp object
                self.grasp_object_pos,
                self.grasp_object_rot,
                self.grasp_object_velocities, 
                self.grasp_object_linvel,
                self.grasp_object_angvel,
                # force
                self.grasp_object_force,
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        # 看报错
        obs = torch.cat(
            (
                obs,  # 原始 155 维观测值
                torch.zeros((obs.shape[0], 2), device=obs.device)  # 补充 2 个维度为 0
            ),
            dim=-1
        )
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
                self.grasp_object_velocities, 
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
    consecutive_successes: torch.Tensor, #连续成功的此书
    max_episode_length: float,  #每个回合的最大步数
    object_pos: torch.Tensor, #物体的位置
    object_angvel: torch.Tensor,   #物体的角速度/角加速度（？）
    object_force:torch.Tensor,  #物体的受力
    target_pos: torch.Tensor,   #目标位置 
    target_force: int,   #目标受力
    dist_reward_scale: float,   #距离奖励因子
    angvel_reward_scale:float,  #角速度受力因子
    force_reward_scale: float,    #受力奖励因子
    # rot_eps: float, #用于防止除零错误的一个小值
    actions: torch.Tensor,  #动作  
    action_penalty_scale: float,    #动作惩罚因子
    success_tolerance: float,   #旋转成功的容忍度
    reach_goal_bonus: float,    #达到目标时的奖励
    fall_dist: float,   #掉落阈值
    fall_penalty: float,    #掉落惩罚
    av_factor: float,   #用于平滑连续成功的奖励
):
    #距离
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    dist_rew = goal_dist * dist_reward_scale
    # print(dist_rew.shape)
    #受力
    # 计算 Z 方向的受力差异
    # print(object_force.shape)
    z_force = object_force[:, 2]  # 提取 Z 方向的受力
    # print("z_force:",z_force)
    z_force_diff = torch.abs(z_force - target_force)  # 计算受力差异
    
    # 根据差异计算受力奖励
    z_force_rew = z_force_diff * force_reward_scale
    # print(z_force_rew.shape)

    # rot_dist = rotation_distance(object_rot, target_rot)

    #角速度
    angvel_rew=torch.norm(object_angvel,p=2,dim=-1)*angvel_reward_scale
    # print(angvel_rew.shape)
    #动作平滑
    action_penalty = torch.sum(actions**2, dim=-1)
    action_rew=action_penalty * action_penalty_scale
    # print(action_rew.shape)
    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    
    reward = dist_rew + z_force_rew + angvel_rew + action_rew

    # Find out which envs hit the goal and update successes count 受力
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
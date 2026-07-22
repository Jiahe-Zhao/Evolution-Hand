# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.sensors import ContactSensor,ContactSensorCfg

from .evolution_stone_grind_env_cfg import EvolutionStoneGrindEnvCfg
import time
# 添加新的对象手，要新加一类
# from typing import TYPE_CHECKING
# if TYPE_CHECKING:
#     from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
#     from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg

#lefthand-grasphand;righthand-strinkhand 后续考虑交换左右手
class EvolutionStoneGrindEnv(DirectMARLEnv):
    cfg: EvolutionStoneGrindEnvCfg
    
    def __init__(self, cfg: EvolutionStoneGrindEnvCfg, render_mode: str | None = None, **kwargs):
        # print(f"env_cfg:{EvolutionStoneGrindEnvCfg}")
        super().__init__(cfg, render_mode, **kwargs)
        # print(f"env_cfg:{EvolutionStoneGrindEnvCfg}")

        self.num_hand_dofs = self.right_hand.num_joints
        print(f"self.num_hand_dofs;{self.num_hand_dofs}")
        # self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # buffers for position targets
        self.right_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.right_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_dof_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_prev_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        self.left_hand_curr_targets = torch.zeros(
            (self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device
        )
        # self.right_hand_save_root_state=self.right_hand.data.root_state_w
        # self.left_hand_save_root_state=self.left_hand.data.root_state_w
        self.right_hand_save_root_state=self.right_hand.data.root_com_state_w
        self.left_hand_save_root_state=self.left_hand.data.root_com_state_w
        # self.right_hand_save_root_state=self.right_hand.data.default_root_state
        # self.left_hand_save_root_state=self.left_hand.data.default_root_state
        self.dt=self.cfg.sim.dt
        self.transition_scale=self.cfg.transition_scale
        self.orientation_scale=self.cfg.orientation_scale

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.right_hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()
        # print(f"actuated_dof_indices{self.actuated_dof_indices}")
        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.right_hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)
        print(f"self.right_hand.body_names{self.finger_bodies}")
        # joint limits
        joint_pos_limits = self.right_hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]
        
        joint_pos_limits_left = self.left_hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits_left  = joint_pos_limits_left[..., 0]
        self.hand_dof_upper_limits_left  = joint_pos_limits_left[..., 1]

        # used to compare object position 要改 righthand
        self.in_lefthand_pos = self.grasp_object.data.default_root_state[:, 0:3].clone()
        self.in_lefthand_pos[:, 2] -= 0.04

        self.in_righthand_pos = self.cone.data.default_root_state[:, 0:3].clone()
        self.in_righthand_pos[:, 2] -= 0.04
        
        #不一定要
        # # default goal positions
        # self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # self.goal_rot[:, 0] = 1.0
        # self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.goal_pos[:, :] = torch.tensor([0.0, -0.64, 0.54], device=self.device)
        # # initialize goal marker
        # self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track goal resets
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # 锥形物体的初始状态
        self.conical_object_initial_state = torch.zeros((self.num_envs, 7), dtype=torch.float, device=self.device)
        

    
        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        #击打目标物体的受力
        self.grasp_object_force=torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        #目标受力
        self.target_force=10

        # initialize goal marker
        # self.grasphand_goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # # track successes
        # self.grasphand_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # self.grasphan_consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        #add hand,cone,grasp_object,contact_sensor
        self.right_hand = Articulation(self.cfg.right_robot_cfg)
        self.left_hand = Articulation(self.cfg.left_robot_cfg)
        self.grasp_object=RigidObject(self.cfg.grasp_object_cfg)
        self.contact_sensor=ContactSensor(self.cfg.contact_sensor_cfg)
        self.cone=RigidObject(self.cfg.Cone_cfg)
        # self.right_hand_save_root_state=self.right_hand.data.root_state_w
        # self.left_hand_save_root_state=self.left_hand.data.root_state_w
        # Allow headless cluster runs to continue when the default online ground asset is unavailable.
        if os.environ.get("DISABLE_DEFAULT_GROUND_PLANE", "0") != "1":
            try:
                spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            except Exception as exc:
                print(f"[WARN] Failed to spawn default ground plane: {exc}")
        
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["right_robot"] = self.right_hand
        self.scene.articulations["left_robot"] = self.left_hand
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

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions
    
    def _apply_action(self) -> None:
        # #解析动作
        # right_hand_base_translation_action = self.actions["right_hand"][:, 0:3]  # 平移 xyz
        # # right_hand_base_translation_action = torch.clamp(right_hand_base_translation_action, -1.0, 1.0)#限制动作幅度，防止运动幅度过大
        # right_hand_base_rotation_action = self.actions["right_hand"][:, 3:6]  # 姿态
        # # right_hand_base_rotation_action = torch.clamp(right_hand_base_rotation_action, -1.0, 1.0)#限制动作幅度，防止运动幅度过大
        
        # left_hand_base_translation_action = self.actions["left_hand"][:, 0:3]  # 平移
        # # left_hand_base_translation_action = torch.clamp(left_hand_base_translation_action, -1.0, 1.0)#限制动作幅度，防止运动幅度过大
        # left_hand_base_rotation_action = self.actions["left_hand"][:, 3:6]  # 姿态
        # # left_hand_base_rotation_action = torch.clamp(left_hand_base_rotation_action, -1.0, 1.0)#限制动作幅度，防止运动幅度过大
        
        # right_hand_joint_action = self.actions["left_hand"][:, 6:] #20维动作
        # left_hand_joint_action = self.actions["left_hand"][:, 6:] #20维动作


        # right_hand_force = self.actions["right_hand"][:, 0:3]* self.dt * self.transition_scale * 100000  
        # right_hand_force = right_hand_force.unsqueeze(1)  # shape = [20, 1, 3]
        # # print(f"right force:{right_hand_force}")
        # right_hand_torque = self.actions["right_hand"][:, 3:6]* self.dt * self.orientation_scale * 1000  
        # right_hand_torque = right_hand_torque.unsqueeze(1)  # shape: [num_envs, 1, 3]
        
        # left_hand_force = self.actions["left_hand"][:, 0:3]* self.dt * self.transition_scale * 100000  
        # left_hand_force = left_hand_force.unsqueeze(1)
        # left_hand_torque = self.actions["left_hand"][:, 3:6] * self.dt * self.orientation_scale * 1000 
        # left_hand_torque = left_hand_torque.unsqueeze(1)

        # # print(f"right force shape{right_hand_force.shape}")
        # # print(f"right torque shape{right_hand_torque.shape}")
        # self.right_hand.set_external_force_and_torque(right_hand_force,right_hand_torque,body_ids=[0])
        # self.left_hand.set_external_force_and_torque(left_hand_force,left_hand_torque,body_ids=[0])
        # #应用平移/姿态控制
        # #右手
        # #平移
        # right_root_pos_w = self.right_hand.data.root_pos_w.clone()  # shape: (num_envs, 3)
        # right_delta_translation = 0.05 * right_hand_base_translation_action 
        
        # right_root_pos_w += right_delta_translation  # 平移叠加
        # #姿态
        # right_root_quat_w = self.right_hand.data.root_quat_w.clone()  # shape: (num_envs, 4)
        # right_delta_angle = 0.05 * torch.norm(right_hand_base_rotation_action, dim=-1)  # shape: (num_envs, 1)
        # right_delta_axis = torch.nn.functional.normalize(right_hand_base_rotation_action, dim=-1)  # shape: (num_envs, 3)
        # right_delta_quat = quat_from_angle_axis(right_delta_angle, right_delta_axis)  # shape: (num_envs, 4)
        # right_root_quat_w = quat_mul(right_root_quat_w, right_delta_quat)
        
        # right_root_pose = torch.cat([right_root_pos_w, right_root_quat_w], dim=-1)  # shape: (num_envs, 7)

        # # self.right_hand.write_root_pose_to_sim(right_root_pose)

        # #左手
        # #平移
        # left_root_pos_w = self.left_hand.data.root_pos_w.clone()  # shape: (num_envs, 3)
        # left_delta_translation = 0.05 * left_hand_base_translation_action 
        
        # left_root_pos_w += left_delta_translation  # 平移叠加
        # #姿态
        # left_root_quat_w = self.left_hand.data.root_quat_w.clone()  # shape: (num_envs, 4)
        # left_delta_angle = 0.05 * torch.norm(left_hand_base_rotation_action, dim=-1)  # shape: (num_envs, 1)
        # left_delta_axis = torch.nn.functional.normalize(left_hand_base_rotation_action, dim=-1)  # shape: (num_envs, 3)
        # left_delta_quat = quat_from_angle_axis(left_delta_angle, left_delta_axis)  # shape: (num_envs, 4)
        # left_root_quat_w = quat_mul(left_root_quat_w, left_delta_quat)
        
        # left_root_pose = torch.cat([left_root_pos_w, left_root_quat_w], dim=-1)  # shape: (num_envs, 7)

        # self.left_hand.write_root_pose_to_sim(left_root_pose)

        # right hand target
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["right_hand"],#right_hand_joint_action
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.right_hand_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.right_hand_prev_targets[:, self.actuated_dof_indices]
        )
        self.right_hand_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.right_hand_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # left hand target
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = scale(
            self.actions["left_hand"],#left_hand_joint_action,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.left_hand_curr_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.left_hand_prev_targets[:, self.actuated_dof_indices]
        )
        self.left_hand_curr_targets[:, self.actuated_dof_indices] = saturate(
            self.left_hand_curr_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # save current targets 要改
        self.right_hand_prev_targets[:, self.actuated_dof_indices] = self.right_hand_curr_targets[
            :, self.actuated_dof_indices
        ]
        self.left_hand_prev_targets[:, self.actuated_dof_indices] = self.left_hand_curr_targets[
            :, self.actuated_dof_indices
        ]

        # set targets 要改
        self.right_hand.set_joint_position_target(
            self.right_hand_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        self.left_hand.set_joint_position_target(
            self.left_hand_curr_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )
        

    
    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {
            "right_hand": torch.cat(
                (
                    # ---- right hand ----
                    # DOF positions (24) len(joint)
                    unscale(self.right_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                    # DOF velocities (24) len(joint)
                    self.cfg.vel_obs_scale * self.right_hand_dof_vel,
                    # fingertip positions (5 * 3) len(link)*3
                    self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                    # fingertip rotations (5 * 4) len(link)*4
                    self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                    # fingertip linear and angular velocities (5 * 6) len(link)*6
                    self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                    # applied actions (26) len(joint)
                    self.actions["right_hand"],
                    # ---- grasp_object ----
                    # positions (3)
                    self.grasp_object_pos,
                    # rotations (4)
                    self.grasp_object_rot,
                    # linear velocities (3)
                    self.grasp_object_linvel,
                    # angular velocities (3)
                    self.cfg.vel_obs_scale * self.grasp_object_angvel,
                    #force（3）
                    self.grasp_object_force,
                    # # ---- goal ---- 要改
                    # # positions (3)
                    # self.goal_pos,
                    # # rotations (4)
                    # self.goal_rot,
                    # # goal-object rotation diff (4)
                    # quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                ),
                dim=-1,
            ),
            "left_hand": torch.cat(
                (
                    # ---- left hand ----
                    # DOF positions (24) len(joint)左手每个关节的当前位置 unscale说明这些位置是经过归一化还原的
                    unscale(self.left_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                    # DOF velocities (24) len(joint)左手每个关节的速度（24维） 被缩放因子vel_obs_scale归一化
                    self.cfg.vel_obs_scale * self.left_hand_dof_vel,
                    # fingertip positions (5 * 3) len(link)*3 左手5个指尖的空间位置，每个是3D坐标（x, y, z）
                    self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                    # fingertip rotations (5 * 4) len(link)*4 每个指尖的四元数旋转（单位为四元数 quaternion，4维）
                    self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                    # fingertip linear and angular velocities (5 * 6)  len(link)*6 每个指尖的线速度 + 角速度（每个指6维）
                    self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                    # applied actions (26) 左手执行的动作值 len(joint)
                    self.actions["left_hand"],
                    # ---- cone ----
                    # positions (3) 物体在环境中的位置
                    self.cone_pos,
                    # rotations (4) 物体当前朝向（四元数形式）
                    self.cone_rot,
                    # linear velocities (3) 物体的线速度（vx, vy, vz）
                    self.cone_linvel,
                    # angular velocities (3) 物体的角速度（wx, wy, wz），同样做了归一化处理
                    self.cfg.vel_obs_scale * self.cone_angvel,
                    # # ---- goal ---- 要改
                    # # positions (3)
                    # self.goal_pos,
                    # # rotations (4)
                    # self.goal_rot,
                    # # goal-object rotation diff (4) 当前物体旋转与目标旋转之间的差异，用四元数差表示
                    # quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                ),
                dim=-1,
            ),
        }
        # print("obs shape: ", observations["left_hand"].shape, observations["right_hand"].shape)  # 加一行调试用的
        return observations
    
    def _get_states(self) -> torch.Tensor:
        states = torch.cat(
            (
                # ---- right hand ----
                # DOF positions (24)
                unscale(self.right_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                # DOF velocities (24)
                self.cfg.vel_obs_scale * self.right_hand_dof_vel,
                # fingertip positions (5 * 3)
                self.right_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # fingertip rotations (5 * 4)
                self.right_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # fingertip linear and angular velocities (5 * 6)
                self.right_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # applied actions (26)
                self.actions["right_hand"],
                # ---- left hand ----
                # DOF positions (24)
                unscale(self.left_hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                # DOF velocities (24)
                self.cfg.vel_obs_scale * self.left_hand_dof_vel,
                # fingertip positions (5 * 3)
                self.left_fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                # fingertip rotations (5 * 4)
                self.left_fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                # fingertip linear and angular velocities (5 * 6)
                self.left_fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # applied actions (26)
                self.actions["left_hand"],
                # ---- grasp_object ----
                # positions (3)
                self.grasp_object_pos,
                # rotations (4)
                self.grasp_object_rot,
                # linear velocities (3)
                self.grasp_object_linvel,
                # angular velocities (3)
                self.cfg.vel_obs_scale * self.grasp_object_angvel,
                #force（3）
                self.grasp_object_force,
                # ----cone------------
                self.cone_pos,
                self.cone_rot, 
                self.cone_linvel,
                self.cfg.vel_obs_scale *self.cone_angvel,
                # # ---- goal ---- 不要
                # # positions (3)
                # self.goal_pos,
                # # rotations (4)
                # self.goal_rot,
                # # goal-object rotation diff (4)
                # quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
            ),
            dim=-1,
        )
        return states
    #要改
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        (
            total_reward,
            self.reset_goal_buf,
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_rewards_new(
            self.reset_buf,
            self.reset_goal_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.left_fingertip_pos.view(self.num_envs, self.num_fingertips, 3),
            self.right_fingertip_pos.view(self.num_envs, self.num_fingertips, 3),
            self.grasp_object_pos,
            self.grasp_object_angvel,
            self.grasp_object_force,
            self.cone_pos,
            self.in_lefthand_pos,
            self.in_righthand_pos,
            self.target_force,
            self.cfg.dist_reward_scale,
            self.cfg.angle_reward_scale,
            self.cfg.force_reward_scale,
            self.actions["left_hand"],
            self.actions["right_hand"],
            self.cfg.action_penalty_scale,
            self.cfg.success_tolerance,
            self.cfg.reach_goal_bonus,
            self.cfg.fall_dist,
            self.cfg.grasp_dist,
            self.cfg.fall_penalty,
            self.cfg.av_factor,
        )

        if "log" not in self.extras:
            self.extras["log"] = dict()
        self.extras["log"]["consecutive_successes"] = self.consecutive_successes.mean()

        # # reset goals if the goal has been reached
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        # if len(goal_env_ids) > 0:
        #     self._reset_target_pose(goal_env_ids)

        return {"right_hand": total_reward, "left_hand": total_reward}
       
    
    
    
    
    # def _get_rewards(self) -> dict[str, torch.Tensor]:
    #     # compute reward
    #     goal_dist = torch.norm(self.object_pos - self.goal_pos, p=2, dim=-1)
    #     rew_dist = 2 * torch.exp(-self.cfg.dist_reward_scale * goal_dist)

    #     # log reward components
    #     if "log" not in self.extras:
    #         self.extras["log"] = dict()
    #     self.extras["log"]["dist_reward"] = rew_dist.mean()
    #     self.extras["log"]["dist_goal"] = goal_dist.mean()

    #     return {"right_hand": rew_dist, "left_hand": rew_dist}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        self._compute_intermediate_values()

        # left_fingertip_pos = self.left_fingertip_pos.view(self.num_envs, self.num_fingertips, 3)

        # # 物体的位置，扩展维度用于广播，shape: (num_envs, 1, 3)
        # grasp_object_pos = self.grasp_object_pos.unsqueeze(1)

        # # 计算每个指尖到物体的距离，shape: (num_envs, 5)
        # left_finger_dists = torch.norm(left_fingertip_pos - grasp_object_pos, dim=-1)

        # # 或者平均距离
        # left_avg_dist = torch.mean(left_finger_dists, dim=1)
        # #右手
        # right_fingertip_pos = self.right_fingertip_pos.view(self.num_envs, self.num_fingertips, 3)
        # cone_pos=self.cone_pos.unsqueeze(1)
        # right_finger_dists=torch.norm(right_fingertip_pos - cone_pos, dim=-1)
        # right_avg_dist = torch.mean(right_finger_dists, dim=1)
        # reset when object has fallen
        left_goal_dist = torch.norm(self.grasp_object_pos - self.in_lefthand_pos, p=2, dim=-1)
    
        # （2）右手：与cone距离
        
        right_goal_dist = torch.norm(self.cone_pos - self.in_righthand_pos, p=2, dim=-1)
        # print(f"right dist:{right_goal_dist}")
        out_of_reach = torch.logical_or(
        left_goal_dist >= self.cfg.fall_dist,
        right_goal_dist >= self.cfg.fall_dist
        )
        # print(f"out of reach:{out_of_reach}")
        # print(self.reset_buf)
        # out_of_reach = torch.logical_or(
        # left_avg_dist >= self.cfg.fall_dist,
        # right_avg_dist >= self.cfg.fall_dist
        # )
        # out_of_reach = left_goal_dist >= self.cfg.fall_dist | right_goal_dist >= self.cfg.fall_dist
        # reset when episode ends
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = {agent: out_of_reach for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.right_hand._ALL_INDICES
        # reset articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals
        # self._reset_target_pose(env_ids)
        self.reset_goal_buf[env_ids] = 0

        # reset grasp_object
        object_default_state = self.grasp_object.data.default_root_state.clone()[env_ids]
        # print(f"object_default_state:{object_default_state}")
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
         # global object positions 还需要改？
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )

        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_default_state[:, 7:] = torch.zeros_like(self.grasp_object.data.default_root_state[env_ids, 7:])
        self.grasp_object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.grasp_object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)

        # reset cone
        cone_default_state = self.cone.data.default_root_state.clone()[env_ids]
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
         # global object positions 还需要改？
        cone_default_state[:, 0:3] = (
            cone_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )
        # 获取默认旋转（四元数）
        cone_default_quat = cone_default_state[:, 3:7]

        # cone_rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        # perturb_quat = randomize_rotation(
        #     cone_rot_noise[:, 0], cone_rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )
        # cone_default_state[:, 3:7] = quat_mul(perturb_quat, cone_default_quat)
        cone_default_state[:, 7:] = torch.zeros_like(self.cone.data.default_root_state[env_ids, 7:])
        self.cone.write_root_pose_to_sim(cone_default_state[:, :7], env_ids)
        self.cone.write_root_velocity_to_sim(cone_default_state[:, 7:], env_ids)
        # ====== reset right hand root state ======
        right_hand_root_state = self.right_hand_save_root_state[env_ids].clone()
        right_state=self.right_hand.data.default_root_state[env_ids].clone()
        # print(f"env_ids:{env_ids}")
        # print(f"right_hand_root_state:{right_hand_root_state}")
        # print(f"self.right_hand_save_root_state:{self.right_hand_save_root_state}")

        # self.right_hand.write_root_pose_to_sim(right_hand_root_state[:, :7], env_ids)
        # self.right_hand.write_root_velocity_to_sim(right_hand_root_state[:, 7:], env_ids)


        # reset right hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.right_hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.right_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.right_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise
        # print(f"1num_dof:{self.num_hand_dofs}")
        # print("1dof_pos shape:", dof_pos.shape)
        # print("1dof_vel_left shape:", dof_vel.shape)
        # print("1env_ids:", env_ids)
        self.right_hand_prev_targets[env_ids] = dof_pos
        self.right_hand_curr_targets[env_ids] = dof_pos
        self.right_hand_dof_targets[env_ids] = dof_pos
        # print(f"dof_pos:{dof_pos}")
        self.right_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.right_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        # self.right_hand.write_root_pose_to_sim(right_hand_root_state[:, :7], env_ids)
        # self.right_hand.write_root_velocity_to_sim(right_hand_root_state[:, 7:], env_ids)
        # print(f"right_hand_dof_pos:{dof_pos}")
        # print(f"right_hand_joint_state:{dof_pos,dof_vel}")
        # print(f"right_hand_root_state:{right_hand_root_state}")
        # print("--------------------")
        # time.sleep(1)
        # ====== reset left hand root state ======
        left_hand_root_state = self.left_hand_save_root_state[env_ids].clone()
        # print(f"env_ids:{env_ids}")
        # print(f"left_hand_root_state:{left_hand_root_state}")
        # print(f"self.left_hand_save_root_state:{self.left_hand_save_root_state}")
        # self.left_hand.write_root_pose_to_sim(left_hand_root_state[:, :7], env_ids)
        # self.left_hand.write_root_velocity_to_sim(left_hand_root_state[:, 7:], env_ids)

        # reset left hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.left_hand.data.default_joint_pos[env_ids]
        # print(f"1111111111111111111delta_max shape:{delta_max.shape}")
        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.left_hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.left_hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        # print("dof_pos shape:", dof_pos.shape)
        # print("dof_vel_left shape:", dof_vel.shape)
        # print("env_ids:", env_ids)
        # print("joint_pos:", self._data.joint_pos.shape)
        self.left_hand_prev_targets[env_ids] = dof_pos
        self.left_hand_curr_targets[env_ids] = dof_pos
        self.left_hand_dof_targets[env_ids] = dof_pos
        # print(f"left_dof_pos:{dof_pos_left}")
        self.left_hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.left_hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        # self.left_hand.write_root_pose_to_sim(left_hand_root_state[:, :7], env_ids)
        # self.left_hand.write_root_velocity_to_sim(left_hand_root_state[:, 7:], env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()
        # print("----------------------------------")
        # time.sleep(1)
    #更换为其他角度 其他受力？ 不确定是否合适
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

    def _compute_intermediate_values(self):
        # data for right hand
        self.right_fingertip_pos = self.right_hand.data.body_pos_w[:, self.finger_bodies]
        self.right_fingertip_rot = self.right_hand.data.body_quat_w[:, self.finger_bodies]
        self.right_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.right_fingertip_velocities = self.right_hand.data.body_vel_w[:, self.finger_bodies]

        self.right_hand_dof_pos = self.right_hand.data.joint_pos
        # print(f"shapeeeee dof pos:{self.right_hand_dof_pos.shape} ")
        self.right_hand_dof_vel = self.right_hand.data.joint_vel

        # data for left hand
        self.left_fingertip_pos = self.left_hand.data.body_pos_w[:, self.finger_bodies]
        self.left_fingertip_rot = self.left_hand.data.body_quat_w[:, self.finger_bodies]
        self.left_fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.left_fingertip_velocities = self.left_hand.data.body_vel_w[:, self.finger_bodies]

        self.left_hand_dof_pos = self.left_hand.data.joint_pos
        self.left_hand_dof_vel = self.left_hand.data.joint_vel

        #data for grasp_object
        self.grasp_object_pos = self.grasp_object.data.root_pos_w - self.scene.env_origins  # 物体的位置
        self.grasp_object_rot = self.grasp_object.data.root_quat_w  # 物体的旋转
        # self.grasp_object_velocities = self.grasp_object.data.root_vel_w  # 物体的速度
        self.grasp_object_linvel = self.grasp_object.data.root_lin_vel_w  # 物体的线速度
        self.grasp_object_angvel = self.grasp_object.data.root_ang_vel_w  # 物体的角速度
       
        # data for cone
        self.cone_pos = self.cone.data.root_pos_w - self.scene.env_origins  # 锥体的位置
        self.cone_rot = self.cone.data.root_quat_w  # 锥体的旋转
        # self.cone_velocities = self.cone.data.root_vel_w  # 锥体的速度
        self.cone_linvel = self.cone.data.root_lin_vel_w  # 锥体的线速度
        self.cone_angvel = self.cone.data.root_ang_vel_w  # 锥体的角速度
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
        # # data for object
        # self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        # self.object_rot = self.object.data.root_quat_w
        # self.object_velocities = self.object.data.root_vel_w
        # self.object_linvel = self.object.data.root_lin_vel_w
        # self.object_angvel = self.object.data.root_ang_vel_w



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
#reward:物体位置的变化；受力的变化；球体是否旋转；
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor, #是否成功达到目标 是否接近理想的受力指/持续一段时间稳定
    successes: torch.Tensor,    #成功的次数
    consecutive_successes: torch.Tensor, #连续成功的此书
    max_episode_length: float,  #每个回合的最大步数
    grasp_object_pos: torch.Tensor, #物体的位置
    grasp_object_angvel: torch.Tensor,   #物体的角速度/角加速度（？）
    grasp_object_force:torch.Tensor,  #物体的受力
    cone_pos:torch.Tensor,      #锥体的位置
    in_lefthand_pos: torch.Tensor,   #左手 
    in_righthand_pos: torch.Tensor,   #右手
    target_force: int,   #目标受力
    dist_reward_scale: float,   #距离奖励因子
    angvel_reward_scale:float,  #角速度受力因子
    force_reward_scale: float,    #受力奖励因子
    # rot_eps: float, #用于防止除零错误的一个小值
    left_actions: torch.Tensor,  #动作
    right_actions: torch.Tensor,  #动作  
    action_penalty_scale: float,    #动作惩罚因子
    success_tolerance: float,   #受力成功的容忍度
    reach_goal_bonus: float,    #达到目标时的奖励
    fall_dist: float,   #掉落阈值
    fall_penalty: float,    #掉落惩罚
    av_factor: float,   #用于平滑连续成功的奖励
):
    
    #(1)左手：与grasp object距离
    left_goal_dist = torch.norm(grasp_object_pos - in_lefthand_pos, p=2, dim=-1)
    left_dist_rew = left_goal_dist * dist_reward_scale
    # （2）右手：与cone距离
    right_goal_dist = torch.norm(cone_pos - in_righthand_pos, p=2, dim=-1)
    right_dist_rew = right_goal_dist * dist_reward_scale
    # （3）左右手动作的平滑度
    left_action_penalty = torch.sum(left_actions**2, dim=-1)
    right_action_penalty = torch.sum(right_actions**2, dim=-1)
    # （4）接近阶段：两手靠近 距离
    approach_dist = torch.norm(cone_pos - grasp_object_pos,p=2, dim=-1)
    approach_rew = approach_dist * dist_reward_scale
    # （5）受力（大小，方向）目前是z轴
    force_diff = torch.abs(grasp_object_force[:, 2] - target_force)
    force_reward = force_diff * force_reward_scale
    # （6）受力的点（不知道能否获取）

    #  (7)grasp object 角速度
    angvel_rew=torch.norm(grasp_object_angvel,p=2,dim=-1)*angvel_reward_scale
    #total reward
    reward=left_dist_rew+right_dist_rew+(left_action_penalty+right_action_penalty)*action_penalty_scale+approach_rew+force_reward+angvel_rew
    print("approach_dist:", approach_dist.shape)
    print("force_reward:", force_reward.shape)
    print("angvel_rew:", angvel_rew.shape)
    print("reward shape:",reward.shape)
    #判断敲击受力
    goal_resets = torch.where(torch.abs(force_diff) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets
    #成功奖励
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    
    # 惩罚：grasp object &cone 掉落
    reward = torch.where((left_goal_dist >= fall_dist) | (right_goal_dist >= fall_dist), reward + fall_penalty, reward)
    # 成功：敲击是否完成
    #Check env termination conditions
    resets = torch.where((left_goal_dist >= fall_dist) | (right_goal_dist >= fall_dist), torch.ones_like(reset_buf), reset_buf)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes

def compute_rewards_new(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor, #是否成功达到目标 是否接近理想的受力指/持续一段时间稳定
    successes: torch.Tensor,    #成功的次数
    consecutive_successes: torch.Tensor, #连续成功的此书
    max_episode_length: float,  #每个回合的最大步数
    left_fingertip_pos:torch.Tensor,#左手手指的位置
    right_fingertip_pos:torch.Tensor,#右手手指的位置
    grasp_object_pos: torch.Tensor, #物体的位置
    grasp_object_angvel: torch.Tensor,   #物体的角速度/角加速度（？）
    grasp_object_force:torch.Tensor,  #物体的受力
    cone_pos:torch.Tensor,      #锥体的位置
    in_lefthand_pos: torch.Tensor,   #左手 
    in_righthand_pos: torch.Tensor,   #右手
    target_force: int,   #目标受力
    dist_reward_scale: float,   #距离奖励因子
    angvel_reward_scale:float,  #角速度受力因子
    force_reward_scale: float,    #受力奖励因子
    # rot_eps: float, #用于防止除零错误的一个小值
    left_actions: torch.Tensor,  #动作
    right_actions: torch.Tensor,  #动作  
    action_penalty_scale: float,    #动作惩罚因子
    success_tolerance: float,   #受力成功的容忍度
    reach_goal_bonus: float,    #达到目标时的奖励
    fall_dist: float,   #掉落阈值
    grasp_dist:float, #抓握阈值
    fall_penalty: float,    #掉落惩罚
    av_factor: float,   #用于平滑连续成功的奖励
):
    #(1)左手：与grasp object距离
    left_goal_dist = torch.norm(grasp_object_pos - in_lefthand_pos, p=2, dim=-1)
    
    # （2）右手：与cone距离
    right_goal_dist = torch.norm(cone_pos - in_righthand_pos, p=2, dim=-1)

    # 物体的位置，扩展维度用于广播，shape: (num_envs, 1, 3)
    grasp_object_pos = grasp_object_pos.unsqueeze(1)

    # 计算每个指尖到物体的距离，shape: (num_envs, 5)
    left_finger_dists = torch.norm(left_fingertip_pos - grasp_object_pos,p=2, dim=-1)

    # 或者平均距离
    left_avg_dist = torch.mean(left_finger_dists, dim=1)
    #右手
    cone_pos=cone_pos.unsqueeze(1)
    right_finger_dists=torch.norm(right_fingertip_pos - cone_pos,p=2, dim=-1)
    right_avg_dist = torch.mean(right_finger_dists, dim=1)
   
    
    
    # （3）左右手动作的平滑度
    left_action_penalty = torch.sum(left_actions**2, dim=-1)
    right_action_penalty = torch.sum(right_actions**2, dim=-1)
    # （4）接近阶段：两手靠近 距离
    approach_dist = torch.norm(cone_pos - grasp_object_pos,p=2, dim=-1)
    approach_dist = approach_dist.view(-1)

    # （5）受力（大小，方向）目前是z轴
    force_diff = torch.abs(grasp_object_force[:, 2] - target_force)
    
    # （6）受力的点（不知道能否获取）

    #  (7)grasp object 角速度
    angvel_rew=torch.norm(grasp_object_angvel,p=2,dim=-1)*angvel_reward_scale
    #total reward
    # is_grasp=torch.zeros_like(left_goal_dist)
    is_grasped = (left_avg_dist < grasp_dist) & (right_avg_dist < grasp_dist)
    # print(f"left dist:{left_avg_dist}")
    # print(f"right dist:{right_avg_dist}")
    # left_dist_rew = left_goal_dist * dist_reward_scale
    # right_dist_rew = right_goal_dist * dist_reward_scale
    # approach_rew = approach_dist * dist_reward_scale
    force_reward = force_diff * force_reward_scale
    up_rew=torch.zeros_like(force_reward)
    up_rew = torch.where(is_grasped,
                        torch.abs(grasp_object_force[:, 2]) * 2, up_rew)
    low_reward_scale=0.0 #5.0
    # reward=torch.where(is_grasped,
    #                    (left_avg_dist+right_avg_dist)*dist_reward_scale+approach_dist * low_reward_scale+(left_action_penalty+right_action_penalty)*action_penalty_scale+force_reward+angvel_rew,
    #                    (left_avg_dist+right_avg_dist)*low_reward_scale+approach_dist * dist_reward_scale+(left_action_penalty+right_action_penalty)*action_penalty_scale+force_reward+angvel_rew)
    # reward=torch.where(is_grasped,
    #                    torch.exp(-(left_avg_dist+right_avg_dist)*dist_reward_scale)+torch.exp(-approach_dist * low_reward_scale)+(left_action_penalty+right_action_penalty)*action_penalty_scale+up_rew+force_reward+angvel_rew,
    #                    torch.exp(-(left_avg_dist+right_avg_dist)*low_reward_scale)+torch.exp(-approach_dist * dist_reward_scale)+(left_action_penalty+right_action_penalty)*action_penalty_scale+up_rew+force_reward+angvel_rew)
    reward=torch.where(is_grasped,
                       torch.exp(-(left_avg_dist+right_avg_dist)*dist_reward_scale)+torch.exp(-approach_dist * low_reward_scale)+(left_action_penalty+right_action_penalty)*action_penalty_scale+up_rew+force_reward+angvel_rew,
                       500+torch.exp(-approach_dist * dist_reward_scale)+(left_action_penalty+right_action_penalty)*action_penalty_scale+up_rew+force_reward+angvel_rew)
    # print(f"is_grasp:{is_grasped}")
    # print("left_avg_dist:", left_avg_dist.shape)
    # print("right_avg_dist:", right_avg_dist.shape)
    # print("approach_dist:", approach_dist.shape)
    # print("force_reward:", force_reward.shape)
    # print("angvel_rew:", angvel_rew.shape)
    # print("reward shape:",reward.shape)
    # print("is_grasped shape:", is_grasped.shape)

    # reward=left_dist_rew+right_dist_rew+(left_action_penalty+right_action_penalty)*action_penalty_scale+approach_rew+force_reward+angvel_rew
    
    #判断敲击受力 以及抓握物体
    goal_resets = torch.where(is_grasped,
                              torch.where(torch.abs(force_diff) <= success_tolerance , 
                                          torch.ones_like(reset_goal_buf), reset_goal_buf),reset_goal_buf)
    # print(f"goal_reset:{goal_resets}")
    successes = successes + goal_resets
    #成功奖励
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)
    # print(f"left shape:{left_goal_dist.shape}")
    # print(f"1 shape:{reward.shape}")
    # 惩罚：grasp object &cone 掉落
    
    reward = torch.where((left_goal_dist >= fall_dist) | (right_goal_dist >= fall_dist), reward + fall_penalty, reward)
    # 成功：敲击是否完成
    # print(f"left shape:{left_goal_dist.shape}")
    #Check env termination conditions
    # print(f"1 reset:{reset_buf}")
    # print(f"left_goal_dist:{left_goal_dist}")
    # print(f"left_goal_dist:{right_goal_dist}")
    resets = torch.where((left_goal_dist >= fall_dist) | (right_goal_dist >= fall_dist), torch.ones_like(reset_buf), reset_buf)
    # print(f"2 shape:{reward.shape}")
    # print(f"reset:{resets}")
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, goal_resets, successes, cons_successes

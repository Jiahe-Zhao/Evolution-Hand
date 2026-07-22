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
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

# 如果要添加新的对象手，要新加一类
if TYPE_CHECKING:
    from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg
    from isaaclab_tasks.direct.shadow_hand.shadow_hand_env_cfg import ShadowHandEnvCfg

# 非基于管理器的，而是直接模式，继承了DirectRLEnv
class InHandManipulationEnv(DirectRLEnv):
    cfg: AllegroHandEnvCfg | ShadowHandEnvCfg  # 适用于AllegroHand和ShadowHand

    def __init__(self, cfg: AllegroHandEnvCfg | ShadowHandEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # 进行一堆初始化
        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        # 此张量存储所有环境中手部每个自由度 (DOF) 的目标位置。它初始化为零，但在模拟过程中将设置为特定目标位置。
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        # self.prev_targets& self.cur_targets：这些存储每个手部自由度的先前和当前目标位置。
        # 它们可用于跟踪每个自由度位置随时间的变化，这对于需要平稳过渡或自适应控制的任务很有帮助。
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        # This list contains the indices of the actuated joints in the hand,
        # based on the names provided in cfg.actuated_joint_names.
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        # 此列表保存fingertip_body_names中指定的手指体的索引
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies) # 读取手指的个数

        # joint limits
        # 这些张量存储了从手模型的根 PhysX 视图中提取的每个手部 DOF 的下限和上限位置限制。
        # 它们设置在设备上以实现高效计算，并确保 DOF 移动保持在物理限制范围内。
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track goal resets
        # 此缓冲区跟踪需要重置其目标状态的环境。它是一个布尔张量，每个环境有一个条目，用于监控目标重置。
        # 每个环境就是一个编号，这里储存了第几个环境需要被重置
        self.reset_goal_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # used to compare object position
        # 从物体的根位置上-0.04得到物体在手中的相对位置
        self.in_hand_pos = self.object.data.default_root_state[:, 0:3].clone()
        self.in_hand_pos[:, 2] -= 0.04
        # default goal positions
        # 定义目标旋转（四元数）和位置
        self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        self.goal_rot[:, 0] = 1.0
        self.goal_pos = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        self.goal_pos[:, :] = torch.tensor([-0.2, -0.45, 0.68], device=self.device)
        # initialize goal marker
        # 从手部文件中的goal_object_cfg类来可视化一个目标对象？
        self.goal_markers = VisualizationMarkers(self.cfg.goal_object_cfg)

        # track successes
        # 该张量跟踪每个环境的成功率，并为每个环境存储一个成功分数。
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        # 这将跟踪所有环境中的累积成功率，从而可以监控一段时间内的成功率
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        # unit tensors
        # 这些张量分别表示沿 x、y 和 z 轴的单位向量。每个向量在所有环境中重复，它们在模拟中用作参考方向，例如用于计算对象方向或与特定轴对齐。
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

    def _setup_scene(self):
        # 创建场景
        # add hand, in-hand object, and goal object
        self.hand = Articulation(self.cfg.robot_cfg)  # 铰链类  robot_cfg和object_cfg都在手的env_cfg文件中
        self.object = RigidObject(self.cfg.object_cfg)  # 刚体类
        # add ground plane 地面是直接生成的
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment) 克隆场景
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene - we must register to scene to randomize with EventManager 添加对象
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # add lights 添加光照
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        # 这步是缩放动作self.actions以适应驱动自由度（DOF）的有效关节限制。
        # self.cur_targets[:, self.actuated_dof_indices]：驱动关节的目标关节位置。这会将缩放动作分配给驱动自由度的当前目标。
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # 应用移动平均滤波器来平滑动作。它通过将当前动作与先前目标融合来帮助减少突然的动作。
        # self.cfg.act_moving_average：可配置参数（可能介于 0 和 1 之间），
        # 用于控制当前动作（self.cur_targets）和前一个目标（self.prev_targets）的权重。
        # 值越高表示越重视当前动作，值越低表示越重视前一个目标。
        # self.cur_targets[:, self.actuated_dof_indices]：这些是缩放的当前目标关节位置（来自上一行）。
        # self.prev_targets[:, self.actuated_dof_indices]：这些是上一步中的目标位置。
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )

        # 通过应用饱和函数确保目标关节位置保持在驱动关节的物理极限内。，
        # 再将结果存储回中self.cur_targets[:, self.actuated_dof_indices]，确保关节目标不超过其物理限制。
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        # 更新先前的目标位置（self.prev_targets）以存储当前目标关节位置。
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # 将计算出的目标关节位置（即self.cur_targets[:, self.actuated_dof_indices]）发送到手模型或机器人的控制系统，以更新实际关节位置
        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _get_observations(self) -> dict:
        # 计算观察，返回观察值
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
        # 计算奖励，返回奖励值
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
            self.object_pos,
            self.object_rot,
            self.in_hand_pos,
            self.goal_rot,
            self.cfg.dist_reward_scale,
            self.cfg.rot_reward_scale,
            self.cfg.rot_eps,
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
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(goal_env_ids) > 0:
            self._reset_target_pose(goal_env_ids)

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 该函数用于根据物体掉落、实现目标或达到情节时间限制等不同条件确定批次中每个环境的情节何时结束。
        self._compute_intermediate_values()

        # reset when cube has fallen
        # 计算手中物体与目标位置的距离
        goal_dist = torch.norm(self.object_pos - self.in_hand_pos, p=2, dim=-1)
        # out_of_reach其中每个元素表示True对象是否超出了该环境的可及范围。
        out_of_reach = goal_dist >= self.cfg.fall_dist

        # Reset progress (episode length buf) on goal envs if max_consecutive_success > 0
        # 检查一集中允许的连续成功的最大次数是否有限制。一个可配置参数，指定允许连续成功尝试的次数（例如，将物体保持在正确的位置）。
        if self.cfg.max_consecutive_success > 0:
            # 计算当前物体方向（ ）与目标方向（ ）之间的旋转距离（角度差）
            rot_dist = rotation_distance(self.object_rot, self.goal_rot)

            #如果旋转距离满足成功公差，则将（跟踪当前情节中的步数）重置为零。（就是指成功了的环境编号，传递到下面开启下一轮训练）
            self.episode_length_buf = torch.where(
                torch.abs(rot_dist) <= self.cfg.success_tolerance, #检查绝对旋转距离是否在允许的公差范围内
                torch.zeros_like(self.episode_length_buf),
                self.episode_length_buf,
            ) # 目的是在已达到目标的环境中重置进度。

            # 检查是否达到允许连续成功的最大次数。
            max_success_reached = self.successes >= self.cfg.max_consecutive_success
        # 检查情节是否已达到最大长度，发出超时信号。（超时重置）
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 如果连续成功次数有限制，则将超时条件与最大成功条件结合起来。（最大成功次数重置）
        if self.cfg.max_consecutive_success > 0:
            time_out = time_out | max_success_reached # 确保了情节将在超时或达到最大连续成功次数时结束。

        return out_of_reach, time_out  # 返回两种重置的情况：
        # out_of_reach：表示物体已掉落到无法触及的环境。
        #time_out：表示情节已达到其最大长度或成功阈值的环境。

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        # resets articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # reset goals # 调用父类的 _reset_idx 方法，重置该环境中的关节和刚体属性。
        self._reset_target_pose(env_ids)

        # reset object
        # 获取物体的默认状态，并创建它的副本，确保在后续操作中不会影响原始默认状态数据。
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        # 生成随机位置噪声 pos_noise，应用于物体的初始位置（前3个坐标）。
        pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=self.device)
        # global object positions 应用噪声
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.cfg.reset_position_noise * pos_noise + self.scene.env_origins[env_ids]
        )
        # 生成旋转噪声 rot_noise，用于物体的X和Y轴旋转角度，然后调用 randomize_rotation 函数以加入随机旋转。
        rot_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # noise for X and Y rotation
        object_default_state[:, 3:7] = randomize_rotation(  # 应用噪声并随机化一个姿态
            rot_noise[:, 0], rot_noise[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # 将物体的默认速度和角速度重置为零，确保物体在重置时静止。
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_state_to_sim(object_default_state, env_ids) # 将物体的状态写入模拟环境中，确保重置生效。

        # reset hand 计算每个手部自由度（DOF）的最大和最小允许偏差，即与默认关节位置的上下界差值。
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        # 目的：生成随机的关节位置噪声 dof_pos_noise，使每个关节的初始位置带有一定随机性。
        # 效果：计算新的关节位置 dof_pos，以此来初始化手部每个关节的起始位置。
        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        # 目的：生成关节速度的噪声 dof_vel_noise，应用到每个手部关节的初始速度。
        # 效果：确保每个关节的速度在重置时有随机化的初始值。
        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        # 目的：设置当前和先前的关节位置目标为新的 dof_pos 值，以确保一致性。
        # 效果：更新关节的目标状态，以便后续控制逻辑可以正常执行。
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        #目的：将计算好的关节位置 dof_pos 设置为手部的目标位置，并写入模拟环境。
        #效果：确保手部的目标关节位置和速度生效于模拟环境。
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        #目的：将成功计数 self.successes 重置为 0，确保每次重置时的计数状态更新。
        #效果：调用 _compute_intermediate_values 函数来计算与成功状态相关的中间量，为重置后的模拟过程做准备。
        self.successes[env_ids] = 0
        self._compute_intermediate_values()

    def _reset_target_pose(self, env_ids):
        # 用于更新目标（生成新的不同的学习目标）
        # reset goal rotation
        # 目的：生成一个随机数 rand_floats，用于目标的旋转随机化。随机值在 -1.0 到 1.0 之间生成，其中 rand_floats[:, 0] 和 rand_floats[:, 1] 分别表示随机的旋转角度。
        # randomize_rotation 函数使用这些随机值以及 X、Y 轴单位向量（self.x_unit_tensor 和 self.y_unit_tensor）生成一个新的四元数 new_rot，代表目标的新旋转。
        rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        # update goal pose and markers
        # 将生成的随机旋转 new_rot 分配给目标旋转 self.goal_rot 的对应环境索引 env_ids。
        self.goal_rot[env_ids] = new_rot
        # 计算目标的全局位置 goal_pos。
        # 这是基于 self.goal_pos（目标的默认位置）加上 self.scene.env_origins（环境的起始位置）得出的，以确保目标位置相对环境起点偏移。
        goal_pos = self.goal_pos + self.scene.env_origins
        #
        self.goal_markers.visualize(goal_pos, self.goal_rot)

        #表示这些目标已完成初始化或重置，清除之前的状态记录
        self.reset_goal_buf[env_ids] = 0

    def _compute_intermediate_values(self):
        # 这个函数计算并存储了一些关键的中间值，为观察和状态计算提供基础数据，包括手指位置、旋转、速度等，以及物体的位置、旋转和速度。
        # 这些数据会在后续的观察和状态计算中被多次使用，确保信息可以重复利用，而不用每次重复计算。
        # data for hand
        self.fingertip_pos = self.hand.data.body_pos_w[:, self.finger_bodies]
        self.fingertip_rot = self.hand.data.body_quat_w[:, self.finger_bodies]
        self.fingertip_pos -= self.scene.env_origins.repeat((1, self.num_fingertips)).reshape(
            self.num_envs, self.num_fingertips, 3
        )
        self.fingertip_velocities = self.hand.data.body_vel_w[:, self.finger_bodies]

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

    def compute_reduced_observations(self):
        # 该函数计算“简化观察值”用于强化学习的低维状态表示。观察值包括以下内容：
        # 手指位置：手指位置，表示手的相对位置状态。
        # 物体位置：物体的位置，但不包含物体的旋转，简化了物体的状态表示。
        # 目标相对旋转：将物体当前旋转和目标旋转做四元数相乘，表示目标旋转的相对差异。
        # 手的动作：手的当前控制动作。
        # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
        #   Fingertip positions
        #   Object Position, but not orientation
        #   Relative target orientation
        obs = torch.cat(
            (
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.object_pos,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                self.actions,
            ),
            dim=-1,
        )

        return obs

    def compute_full_observations(self):
        # 该函数计算“完整观察值”，包含了更多的物理状态信息，用于更复杂的策略学习。观察值包括：
        #
        # 手的自由度：手指关节位置、速度，并进行标准化处理。
        # 物体信息：物体的位置、旋转、线速度和角速度。
        # 目标位置和旋转：手中的物体位置以及目标旋转。
        # 手指详细信息：手指位置、旋转和速度，包含更多细粒度的信息。
        # 手的动作：手当前控制的动作。
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
                # fingertips
                self.fingertip_pos.view(self.num_envs, self.num_fingertips * 3),
                self.fingertip_rot.view(self.num_envs, self.num_fingertips * 4),
                self.fingertip_velocities.view(self.num_envs, self.num_fingertips * 6),
                # actions
                self.actions,
            ),
            dim=-1,
        )
        return obs

    # 得到全部对象的观测值
    def compute_full_state(self):
        # 该函数计算“完整状态”，比compute_full_observations提供更多的信息，适合更高精度的场景。状态包括：
        #
        # 手的自由度：手指关节位置、速度（标准化）。
        # 物体信息：位置、旋转、线速度、角速度。
        # 目标信息：物体在手中位置、目标旋转及相对旋转。
        # 手指信息：手指位置、旋转、速度、力传感器数据（表示施加的力矩）。
        # 手的动作：手当前控制的动作。
        states = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # object
                self.object_pos,
                self.object_rot,
                self.object_linvel,
                self.cfg.vel_obs_scale * self.object_angvel,
                # goal
                self.in_hand_pos,
                self.goal_rot,
                quat_mul(self.object_rot, quat_conjugate(self.goal_rot)),
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

#是 PyTorch 中的一种装饰器，用于将 Python 函数或类转换为 TorchScript。TorchScript 是 PyTorch 的一种中间表示，
# 可以让模型和函数在不依赖 Python 解释器的情况下运行，从而实现更高效的执行（特别是加速推理）以及跨平台部署（比如在 C++ 环境中运行）。
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

# 计算奖励的函数，要修改目标就在这里，有时需要把script这个修饰删了以便正常运行（和之前跑的强化学习一样）
@torch.jit.script
def compute_rewards(
    reset_buf: torch.Tensor,
    reset_goal_buf: torch.Tensor,
    successes: torch.Tensor,
    consecutive_successes: torch.Tensor,
    max_episode_length: float,
    object_pos: torch.Tensor,
    object_rot: torch.Tensor,
    target_pos: torch.Tensor,
    target_rot: torch.Tensor,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions: torch.Tensor,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    av_factor: float,
):

    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)
    rot_dist = rotation_distance(object_rot, target_rot)

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
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

# 47 处未提交修改审查

范围：逐文件差异、依赖归属及语法审查；不冒充完成旧 PhysX 训练回归。当前发布主线是独立 Newton 场景。

已合入必要的通用修复；展开机构模型独立发布，避免替换已验收 AX50 手型。其他改动以完整补丁公开，未应用到默认运行代码。

|文件|处理|原因|
|---|---|---|
|`Isaaclab_other/australopithecus_hand_assets/left/urdf/australopithecus_hand_mirror.urdf`|published_as_separate_model|包含展开机构的旧骨手，单独发布完整 URDF/STL，保留默认基准及 AX50 v3|
|`Isaaclab_other/australopithecus_hand_assets/right/urdf/australopithecus_hand.urdf`|published_as_separate_model|包含展开机构的旧骨手，单独发布完整 URDF/STL，保留默认基准及 AX50 v3|
|`Isaaclab_other/build_compliant_palm.py`|deferred_patch|旧 PhysX 软掌实验，非 AX50 Newton 运行依赖；需另做 PhysX 回归|
|`Isaaclab_other/compliant_palm_runtime.py`|deferred_patch|旧 PhysX 软掌实验，非 AX50 Newton 运行依赖；需另做 PhysX 回归|
|`Isaaclab_other/compliant_palm_task.py`|deferred_patch|旧 PhysX 软掌实验，非 AX50 Newton 运行依赖；需另做 PhysX 回归|
|`Isaaclab_other/evaluation_interface.py`|deferred_patch|旧 IsaacLab 进化/训练/校验链路扩展，非本次原生 Newton 场景依赖；保留差异待对应环境回归|
|`Isaaclab_other/isaaclab_tool.py`|deferred_patch|旧 IsaacLab 进化/训练/校验链路扩展，非本次原生 Newton 场景依赖；保留差异待对应环境回归|
|`Isaaclab_other/main_evolution.py`|deferred_patch|旧 IsaacLab 进化/训练/校验链路扩展，非本次原生 Newton 场景依赖；保留差异待对应环境回归|
|`Isaaclab_other/mechanical_evolution_interface.py`|deferred_patch|旧 IsaacLab 进化/训练/校验链路扩展，非本次原生 Newton 场景依赖；保留差异待对应环境回归|
|`Isaaclab_other/mirror_agent.py`|included|显式左右手标记，避免名称推断失效|
|`Isaaclab_other/persistent_isaac_worker.py`|deferred_patch|旧 IsaacLab 进化/训练/校验链路扩展，非本次原生 Newton 场景依赖；保留差异待对应环境回归|
|`Isaaclab_other/policy_inheritance.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`Isaaclab_other/read_results.py`|included|奖励文件名解析；另修复双下划线负号丢失|
|`Isaaclab_other/run_formal_mechanical_bc_20261018.sh`|deferred_patch|旧 IsaacLab 进化/训练/校验链路扩展，非本次原生 Newton 场景依赖；保留差异待对应环境回归|
|`Isaaclab_other/train_worker.py`|deferred_patch|旧 IsaacLab 进化/训练/校验链路扩展，非本次原生 Newton 场景依赖；保留差异待对应环境回归|
|`Isaaclab_other/validate_collision_runtime.py`|deferred_patch|旧 IsaacLab 进化/训练/校验链路扩展，非本次原生 Newton 场景依赖；保留差异待对应环境回归|
|`Isaaclab_other/validate_compliant_palm.py`|deferred_patch|旧 PhysX 软掌实验，非 AX50 Newton 运行依赖；需另做 PhysX 回归|
|`evolution_tasks/cartesian_hand_controller.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/current_evolution_agent/current_hand_cfg.py`|deferred_patch|当前运行生成的配置/个体快照，不覆盖默认配置|
|`evolution_tasks/current_evolution_agent/human_hand_414.py`|deferred_patch|当前运行生成的配置/个体快照，不覆盖默认配置|
|`evolution_tasks/current_evolution_agent/human_mirror_hand_414.py`|deferred_patch|当前运行生成的配置/个体快照，不覆盖默认配置|
|`evolution_tasks/current_left_hand/current_left_hand_cfg.py`|deferred_patch|当前运行生成的配置/个体快照，不覆盖默认配置|
|`evolution_tasks/current_right_hand/current_right_hand_cfg.py`|deferred_patch|当前运行生成的配置/个体快照，不覆盖默认配置|
|`evolution_tasks/mechanical_hand_interface.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_branch_grasp/branch_grasp_env.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_branch_grasp/branch_grasp_env_cfg.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_branch_grasp/calibrate_branch_trajectory.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_forage/forage_env.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_forage/forage_env_cfg.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_grasp/evolution_grasp_env.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_grasp/evolution_grasp_env_cfg.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_grasp/scripted_adaptive_grasp.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_strike/__init__.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_strike/evolution_strike_env.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_strike/evolution_strike_env_cfg.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_strike/probe_strike_grasp.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_strike/strike_bc_teacher.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_suite/evaluate_task.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/task_suite/scripted_adaptive_task_demo.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`evolution_tasks/train_interface.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|
|`outputs/derl_pure_ppo_20261025/NORMALIZATION_AB_20261026.md`|included|补齐已有实验结论，不启动训练|
|`outputs/derl_pure_ppo_20261025/branch_policy_video.py`|deferred_patch|依赖旧策略 checkpoint 的回放扩展，不是场景从零启动依赖|
|`outputs/derl_pure_ppo_20261025/forage_policy_video.py`|deferred_patch|依赖旧策略 checkpoint 的回放扩展，不是场景从零启动依赖|
|`outputs/derl_pure_ppo_20261025/native_contact_hooks.py`|included|USD 缓存目录规范化|
|`outputs/derl_pure_ppo_20261025/native_goal_ppo.py`|included|训练步数 frames 初值补齐|
|`outputs/derl_pure_ppo_20261025/native_goal_worker.py`|included|优先使用实际末节骨段|
|`outputs/derl_pure_ppo_20261025/task_bottleneck_audit.py`|deferred_patch|涉及控制器、动作观测或任务判据；非本次原生 Newton 场景依赖，保留差异待原后端回归|

# v5 四任务终局正例验收

此项验收独立于单指接触 A/B。旧 branch 演示的形态是 human_hand，不能计入 v5 成绩。hand_111 与 hand_211 通过同一个 Newton 入口，从标准 reset 开始，仅允许提交手指与腕部动作。无物体传送，不覆盖奖励或状态机，不把 timeout 当成功。

证据目录：outputs/v5_acceptance/terminal_{111,211}_{branch,forage,grasp,strike}。terminal_trial.json 是运行中快照，不能仅凭文件存在判定完成。成功事件应核对当步奖励、完成状态、接触以及物体位姿。正式训练要求八组均有物理成功事件。

当前状态：试验启动；尚未验收，未启动进化。

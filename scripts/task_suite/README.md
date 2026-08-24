# Four-Task Suite

Every task exports an MP4 plus JSON metrics. Success is unified as
`reward >= 1000`; each task environment remains responsible for determining
when that terminal reward is emitted.

`demo_<task>.sh` records a deterministic scene/demo video.

`train_<task>.sh --num_envs 128 --max_iterations 300 --run_name NAME` starts
RL-Games training with the shared training interface.

`evaluate_<task>.sh CHECKPOINT --episodes 5` loads a checkpoint, records a
video, and writes `metrics.json` under `outputs/task_suite/evaluation`.

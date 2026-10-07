"""Assemble only physically successful, native-scene evaluation trajectories for BC."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input_root", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", choices=("forage", "strike"), required=True)
parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage2")
args = parser.parse_args()

expected_shape = {"forage": (117, 23), "strike": (162, 23)}[args.task]
obs_parts, action_parts, episode_parts, episodes = [], [], [], []
checkpoint = morphology = None
for report_path in sorted(args.input_root.glob("seed_*/evaluation.json")):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = report.get("episode_records", [])
    if report.get("task") != args.task or report.get("curriculum_stage") != args.stage or len(records) != 1:
        raise ValueError(f"Task/stage/episode mismatch: {report_path}")
    record = records[0]
    if not record.get("success") or not any(step.get("reward", 0) >= 999 for step in record.get("trace", [])):
        raise ValueError(f"Incomplete task success: {report_path}")
    if report.get("strike_wrist_teacher") and args.task != "strike":
        raise ValueError(f"Unexpected teacher override: {report_path}")
    if checkpoint is None:
        checkpoint = report.get("checkpoint")
        morphology = report.get("morphology")
    if report.get("checkpoint") != checkpoint or report.get("morphology") != morphology:
        raise ValueError(f"Mixed checkpoint or morphology: {report_path}")
    trace_path = report_path.parent / "successful_policy_trace.npz"
    if not trace_path.is_file():
        raise FileNotFoundError(trace_path)
    with np.load(trace_path) as trace:
        obs = np.asarray(trace["observations_before_step"], dtype=np.float32)
        actions = np.asarray(trace["submitted_actions"], dtype=np.float32)
        if obs.shape != (record["steps"], expected_shape[0]) or actions.shape != (record["steps"], expected_shape[1]):
            raise ValueError(f"Shape mismatch: {trace_path}")
        if not np.isfinite(obs).all() or not np.isfinite(actions).all():
            raise ValueError(f"Nonfinite BC data: {trace_path}")
        if not bool(np.asarray(trace["actions_control_fingers"]).all()) or not bool(np.asarray(trace["scene_unmodified"]).all()):
            raise ValueError(f"Invalid action or scene provenance: {trace_path}")
        if not np.any(np.abs(actions[:, :20]) > 1e-4):
            raise ValueError(f"No finger actions: {trace_path}")
    seed = int(record["seed"])
    if any(item["seed"] == seed for item in episodes):
        raise ValueError(f"Duplicate seed: {seed}")
    obs_parts.append(obs)
    action_parts.append(actions)
    episode_parts.append(np.full(len(obs), len(episodes), dtype=np.int32))
    episodes.append({"seed": seed, "steps": len(obs), "report": str(report_path),
                     "sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest()})

if len(episodes) < 2:
    raise ValueError("Need successful demonstrations from at least two seeds")
args.output.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(args.output,
    observations_before_step=np.concatenate(obs_parts),
    submitted_actions=np.concatenate(action_parts),
    actions_control_fingers=np.ones(sum(map(len, obs_parts)), dtype=bool),
    scene_unmodified=np.ones(sum(map(len, obs_parts)), dtype=bool),
    episode_index=np.concatenate(episode_parts),
)
manifest = {"task": args.task, "stage": args.stage, "checkpoint": checkpoint,
            "morphology": morphology, "episodes": episodes,
            "samples": int(sum(map(len, obs_parts))), "dataset": str(args.output)}
args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(json.dumps({"episodes": len(episodes), "samples": manifest["samples"], "dataset": str(args.output)}))

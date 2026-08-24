"""Merge one-episode IsaacLab evaluation records into a reproducible report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--episodes_dir", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--task", required=True)
args = parser.parse_args()

records = []
for path in sorted(Path(args.episodes_dir).glob("episode_*/evaluation.json")):
    with path.open(encoding="utf-8") as file:
        report = json.load(file)
    record = report["episode_records"][0]
    record["episode_dir"] = str(path.parent)
    records.append(record)

if not records:
    raise SystemExit("No one-episode evaluation records were found.")
successes = sum(bool(record["success"]) for record in records)
summary = {
    "task": args.task,
    "episodes": len(records),
    "successes": successes,
    "success_rate": successes / len(records),
    "success_definition": "The environment's sparse terminal success event.",
    "episode_records": records,
}
with Path(args.output).open("w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)
print(json.dumps({key: summary[key] for key in ("task", "episodes", "successes", "success_rate")}))


# Resolve this checkout, not the original experiment machine.
from pathlib import Path as _EvoPath
import sys as _evo_sys
_EVOLUTION_ROOT = next(p for p in _EvoPath(__file__).resolve().parents if (p / 'scripts/unpack_ax50.py').is_file())
_evo_sys.path.insert(0, str(_EVOLUTION_ROOT))
from portable_runtime import bootstrap as _evo_bootstrap
_evo_bootstrap()
#!/usr/bin/env python3
import argparse
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def nearest_row(rows, times, time_s):
    index = bisect.bisect_left(times, time_s)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(rows)]
    return rows[min(candidates, key=lambda i: abs(times[i] - time_s))]


def contact_summary(trace, contacts):
    phases = [float(row["time_s"]) for row in trace]
    groups = defaultdict(list)
    for contact in contacts:
        if float(contact["normal_N"]) <= 1e-5:
            continue
        phase = nearest_row(trace, phases, float(contact["time_s"]))["phase"]
        groups[(phase, contact["object"], contact["object_shape"])].append(contact)

    result = []
    for (phase, obj, object_shape), rows in sorted(groups.items()):
        result.append({
            "phase": phase,
            "object": obj,
            "object_shape": object_shape,
            "loaded_contact_record_count": len(rows),
            "peak_normal_N": max(float(row["normal_N"]) for row in rows),
            "hand_segments": sorted({row["hand_part"] for row in rows}),
        })
    return result


def palm_side_summary(contacts):
    segments = {f"link_{digit}_{link}" for digit in (3, 4) for link in (1, 2, 3)}
    loaded = [row for row in contacts
              if row["object_shape"] == "grip"
              and row["hand_part"] in segments
              and float(row["normal_N"]) > 1e-5]
    positive = sum(float(row["hand_local_y_m"]) > 0.001 for row in loaded)
    negative = sum(float(row["hand_local_y_m"]) < -0.001 for row in loaded)
    count = len(loaded)
    return {
        "basis": "recorded loaded grip contacts, not per-substep statistics",
        "segments": sorted(segments),
        "loaded_contact_record_count": count,
        "positive_y_over_1mm": {"count": positive, "ratio": positive / count if count else None},
        "negative_y_below_minus_1mm": {"count": negative, "ratio": negative / count if count else None},
        "central_region_abs_y_at_most_1mm": {
            "count": count - positive - negative,
            "ratio": (count - positive - negative) / count if count else None,
            "classification": "unknown",
        },
    }


def actual_joint_summary(model, npz):
    joint_q = np.asarray(npz["joint_q"], dtype=float)
    names = model["action_names"]
    indices = model["action_indices"]
    wanted = [f"digit_{digit}_flex_{link}" for digit in (3, 4) for link in (1, 2, 3)]
    result = {}
    for name in wanted:
        if name not in names:
            raise ValueError(f"Missing action name: {name}")
        action_index = names.index(name)
        joint_index = int(indices[action_index])
        angle = float(np.max(np.abs(joint_q[:, joint_index])))
        result[name] = {
            "action_index": action_index,
            "joint_index": joint_index,
            "max_abs_actual_rad": angle,
            "max_abs_actual_deg": math.degrees(angle),
        }
    return result


def trajectory_summary(trace):
    initial_hinge = float(trace[0]["hinge_rad"])
    hinge_deg = [math.degrees(float(row["hinge_rad"]) - initial_hinge) for row in trace]
    max_hinge_deg = max(hinge_deg)
    end_time = float(trace[-1]["time_s"])
    final = [row for row in trace if float(row["time_s"]) >= end_time - 0.2]
    final_hinge = [math.degrees(float(row["hinge_rad"]) - initial_hinge) for row in final]
    bail_values = [float(row["min_bail_contact_N"]) for row in final]
    return {
        "max_bail_angle_deg": max_hinge_deg,
        "exceeds_90_deg": max_hinge_deg > 90.0,
        "final_0.2s_bail_angle_range_deg": {"min": min(final_hinge), "max": max(final_hinge)},
        "hand_item_full_run_max_body_contact_N": max(float(row["max_body_contact_N"]) for row in trace),
        "final_0.2s_min_bail_contact_N": min(bail_values) if bail_values else None,
        "final_window_basis": "per-substep extrema recorded by the runtime",
        "scope": "open-hand bail support only; not a complete grasp success claim",
    }


def main():
    parser = argparse.ArgumentParser(description="Read-only audit of an open-hand support run")
    parser.add_argument("tag")
    parser.add_argument("--output-root", type=Path,
                        default=Path(__file__).resolve().parents[3] / "outputs/nov20/ax50_handle13")
    args = parser.parse_args()
    root = args.output_root
    paths = {suffix: root / f"{args.tag}{suffix}" for suffix in
             ("_trace.csv", "_contacts.csv", "_model.json", ".npz")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    trace = read_csv(paths["_trace.csv"])
    contacts = read_csv(paths["_contacts.csv"])
    model = json.loads(paths["_model.json"].read_text())
    with np.load(paths[".npz"], allow_pickle=False) as npz:
        audit = {
            "tag": args.tag,
            "loaded_contact_normal_threshold_N": 1e-5,
            "contacts_by_phase_and_object_shape": contact_summary(trace, contacts),
            "grip_hand_local_y": palm_side_summary(contacts),
            "actual_joint_positions": actual_joint_summary(model, npz),
            "bail_support": trajectory_summary(trace),
        }
    output = root / f"{args.tag}_open_support_audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()

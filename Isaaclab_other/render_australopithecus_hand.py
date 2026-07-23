from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from australopithecus_hand_agent import AUSTRALOPITHECUS_HAND


JOINT_POSE = {
    "link_0_0_to_link_1_0": 0.42,
    "link_1_0_to_link_1_1": 0.35,
    "link_1_1_to_link_1_2": 0.22,
    "link_0_0_to_link_2_0": 0.18,
    "link_2_0_to_link_2_1": 0.46,
    "link_2_1_to_link_2_2": 0.34,
    "link_2_2_to_link_2_3": 0.18,
    "link_0_0_to_link_3_0": 0.20,
    "link_3_0_to_link_3_1": 0.50,
    "link_3_1_to_link_3_2": 0.38,
    "link_3_2_to_link_3_3": 0.22,
    "link_0_0_to_link_4_0": 0.24,
    "link_4_0_to_link_4_1": 0.56,
    "link_4_1_to_link_4_2": 0.42,
    "link_4_2_to_link_4_3": 0.24,
    "link_0_0_to_link_5_0": 0.30,
    "link_5_0_to_link_5_1": 0.62,
    "link_5_1_to_link_5_2": 0.45,
    "link_5_2_to_link_5_3": 0.26,
}


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy_matrix(rpy: list[float]) -> np.ndarray:
    rx, ry, rz = rpy
    return _rot_z(rz) @ _rot_y(ry) @ _rot_x(rx)


def _transform(translation: list[float], rotation: np.ndarray) -> np.ndarray:
    tf = np.eye(4)
    tf[:3, :3] = rotation
    tf[:3, 3] = np.asarray(translation, dtype=float)
    return tf


def _draw_box(ax, size_xyz: list[float]) -> None:
    sx, sy, sz = size_xyz
    vertices = np.array(
        [
            [-sx / 2, -sy / 2, 0],
            [sx / 2, -sy / 2, 0],
            [sx / 2, sy / 2, 0],
            [-sx / 2, sy / 2, 0],
            [-sx / 2, -sy / 2, sz],
            [sx / 2, -sy / 2, sz],
            [sx / 2, sy / 2, sz],
            [-sx / 2, sy / 2, sz],
        ]
    )
    faces = [
        [vertices[i] for i in [0, 1, 2, 3]],
        [vertices[i] for i in [4, 5, 6, 7]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [1, 2, 6, 5]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [3, 0, 4, 7]],
    ]
    poly = Poly3DCollection(faces, facecolors="#e1ad64", edgecolors="#7f5c2c", linewidths=0.8, alpha=0.88)
    ax.add_collection3d(poly)


def _transform_points(points: np.ndarray, tf: np.ndarray) -> np.ndarray:
    hom = np.hstack([points, np.ones((points.shape[0], 1))])
    return (tf @ hom.T).T[:, :3]


def _draw_transformed_box(ax, size_xyz: list[float], tf: np.ndarray, color: str = "#d5a062") -> np.ndarray:
    sx, sy, sz = size_xyz
    vertices = np.array(
        [
            [-sx / 2, -sy / 2, 0],
            [sx / 2, -sy / 2, 0],
            [sx / 2, sy / 2, 0],
            [-sx / 2, sy / 2, 0],
            [-sx / 2, -sy / 2, sz],
            [sx / 2, -sy / 2, sz],
            [sx / 2, sy / 2, sz],
            [-sx / 2, sy / 2, sz],
        ]
    )
    transformed = _transform_points(vertices, tf)
    faces = [
        [transformed[i] for i in [0, 1, 2, 3]],
        [transformed[i] for i in [4, 5, 6, 7]],
        [transformed[i] for i in [0, 1, 5, 4]],
        [transformed[i] for i in [1, 2, 6, 5]],
        [transformed[i] for i in [2, 3, 7, 6]],
        [transformed[i] for i in [3, 0, 4, 7]],
    ]
    poly = Poly3DCollection(faces, facecolors=color, edgecolors="#8b6234", linewidths=0.55, alpha=0.28)
    ax.add_collection3d(poly)
    return transformed


def _segment_points(link_data: dict, parent_tf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_name = link_data["joint_name"]
    joint_tf = _transform(
        link_data["joint_origin_translation"],
        _rpy_matrix(link_data["joint_origin_rpy"]) @ _rot_x(JOINT_POSE.get(joint_name, 0.0)),
    )
    link_tf = parent_tf @ joint_tf
    start = link_tf[:3, 3]
    if "geometry_length" in link_data:
        span = float(link_data["geometry_length"])
    elif "geometry_size" in link_data:
        span = float(link_data["geometry_size"][2])
    else:
        span = 0.0
    end = (link_tf @ np.array([0.0, 0.0, span, 1.0]))[:3]
    return start, end, link_tf


def render_hand(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.5, 8.0), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#f6efe5")
    ax.set_facecolor("#f6efe5")

    base_link = AUSTRALOPITHECUS_HAND["base_link"][0]
    _draw_box(ax, base_link["geometry_size"])

    parent_map = {"link_0_0": np.eye(4)}
    points = [
        np.array([-base_link["geometry_size"][0] / 2, -base_link["geometry_size"][1] / 2, 0.0]),
        np.array([base_link["geometry_size"][0] / 2, base_link["geometry_size"][1] / 2, base_link["geometry_size"][2]]),
    ]
    color_map = {
        "1": "#b35a36",
        "2": "#d58753",
        "3": "#cc8b65",
        "4": "#c89d79",
        "5": "#b98c73",
    }

    for link_data in AUSTRALOPITHECUS_HAND["links"]:
        start, end, link_tf = _segment_points(link_data, parent_map[link_data["joint_parent"]])
        parent_map[link_data["name_code"]] = link_tf
        if link_data.get("joint_type") == "fixed" and "geometry_size" in link_data:
            box_points = _draw_transformed_box(ax, link_data["geometry_size"], link_tf)
            points.extend(list(box_points))
        points.extend([start, end])
        finger_id = link_data["name_code"].split("_")[1] if link_data["name_code"].startswith("link_") else "palm"
        line_width = link_data.get("geometry_radius", 0.004) * 1400
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=color_map.get(finger_id, "#8a5a44"),
            linewidth=line_width,
            solid_capstyle="round",
        )
        ax.scatter(end[0], end[1], end[2], s=12, color="#5f3b27", alpha=0.8)

    points_arr = np.vstack(points)
    mins = points_arr.min(axis=0)
    maxs = points_arr.max(axis=0)
    center = (mins + maxs) / 2.0
    extents = np.maximum(maxs - mins, np.array([0.05, 0.04, 0.08]))
    radius = float(extents.max() * 0.65)

    ax.view_init(elev=18, azim=-58)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius * 0.8, center[1] + radius * 0.8)
    ax.set_zlim(center[2] - radius * 0.9, center[2] + radius * 0.9)
    ax.set_box_aspect((0.9, 0.7, 1.35))
    ax.set_axis_off()
    ax.text2D(
        0.03,
        0.95,
        "Australopithecus-style hand\nlong thumb + broader palm + curved phalanges",
        transform=ax.transAxes,
        fontsize=12,
        color="#5f3b27",
        ha="left",
        va="top",
    )

    plt.tight_layout(pad=0)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.06, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a schematic preview of the Australopithecus-style hand.")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    render_hand(args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()

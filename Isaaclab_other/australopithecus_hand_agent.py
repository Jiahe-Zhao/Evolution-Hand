from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

from code_to_urdf import generate_urdf_from_dict
from isaaclab_tool import parse_urdf_and_generate_articulation_cfg
from mirror_agent import create_mirror_hand


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISAACLAB_OTHER_ROOT = PROJECT_ROOT / "Isaaclab_other"
TASK_ROOT = PROJECT_ROOT / "evolution_tasks"


def _joint_limit(lower: float, upper: float, effort: float, velocity: float) -> dict:
    return {"lower": lower, "upper": upper, "effort": effort, "velocity": velocity}


def _fixed_link(
    name: str,
    parent: str,
    geometry_size: list[float],
    translation: list[float],
    rpy: list[float],
    density: float = 780.0,
    geometry_type: str = "box",
    visual_color: list[float] | None = None,
) -> dict:
    link = {
        "name_code": name,
        "geometry_type": "box",
        "geometry_size": geometry_size,
        "joint_name": f"{parent}_to_{name}",
        "joint_parent": parent,
        "joint_type": "fixed",
        "joint_axis": [0, 0, 1],
        "joint_limit": _joint_limit(0.0, 0.0, 0.0, 0.0),
        "joint_origin_translation": translation,
        "joint_origin_rpy": rpy,
        "density": density,
    }
    link["geometry_type"] = geometry_type
    if visual_color is not None:
        link["visual_color"] = visual_color
    return link


def _segment(
    name: str,
    parent: str,
    radius: float,
    length: float,
    translation: list[float],
    rpy: list[float],
    limit: dict,
    density: float = 920.0,
) -> dict:
    return {
        "name_code": name,
        "geometry_type": "capsule",
        "geometry_radius": radius,
        "geometry_length": length,
        "joint_name": f"{parent}_to_{name}",
        "joint_parent": parent,
        "joint_type": "revolute",
        "joint_axis": [1, 0, 0],
        "joint_limit": limit,
        "joint_origin_translation": translation,
        "joint_origin_rpy": rpy,
        "density": density,
    }


def _finger_chain(
    finger_id: int,
    base_translation: list[float],
    base_rpy: list[float],
    lengths: list[float],
    radii: list[float],
    limits: list[dict],
) -> list[dict]:
    if len(lengths) != len(radii) or len(lengths) != len(limits):
        raise ValueError("lengths/radii/limits must have equal length")

    links: list[dict] = []
    parent = "link_0_0"
    for index, (length, radius, limit) in enumerate(zip(lengths, radii, limits)):
        name = f"link_{finger_id}_{index}"
        if index == 0:
            translation = base_translation
            rpy = base_rpy
        else:
            prev_length = lengths[index - 1]
            prev_radius = radii[index - 1]
            translation = [0.0, 0.0, round(prev_length + prev_radius, 5)]
            # Mild progressive flexion bias to mimic curved phalanges.
            rpy = [0.08 + 0.02 * index, 0.0, 0.0]
        links.append(_segment(name, parent, radius, length, translation, rpy, limit))
        parent = name
    return links


BASE_LINK = {
    "name_code": "link_0_0",
    # Original skeletal wrist/metacarpal core.  All 19 finger joints remain
    # attached here, so the original kinematic structure is preserved.
    "geometry_type": "box",
    "geometry_size": [0.03, 0.046, 0.02],
    "density": 900.0,
}


# Design intent:
# 1. Keep the 19 actuated joint names used by the current training tasks.
# 2. Add fixed palm/wrist sub-links so the asset is no longer a single rigid block.
# 3. Use an Australopithecus-inspired morphology: broader palm, long opposable thumb,
#    slightly curved fingers, and a mixed power/precision grasp layout.
# Keep the original skeletal asset.  Palm membership is evaluated in the
# training environments, not represented by an additional physical link.
PALM_LINKS: list[dict] = []


THUMB_LINKS = _finger_chain(
    finger_id=1,
    base_translation=[0.014, -0.013, 0.001],
    base_rpy=[0.30, 2.16, -0.88],
    lengths=[0.03, 0.024, 0.018],
    radii=[0.0052, 0.0045, 0.0039],
    limits=[
        _joint_limit(0.0, 1.15, 18.0, 2.0),
        _joint_limit(0.0, 1.35, 12.0, 1.7),
        _joint_limit(0.0, 1.05, 7.5, 1.5),
    ],
)

INDEX_LINKS = _finger_chain(
    finger_id=2,
    base_translation=[0.0155, -0.0048, 0.02],
    base_rpy=[0.06, 1.55, -0.14],
    lengths=[0.03, 0.028, 0.019, 0.014],
    radii=[0.0042, 0.0039, 0.0033, 0.0026],
    limits=[
        _joint_limit(0.0, 1.0, 16.0, 1.9),
        _joint_limit(0.0, 1.45, 11.0, 1.7),
        _joint_limit(0.0, 1.25, 6.2, 1.5),
        _joint_limit(0.0, 1.0, 5.0, 1.4),
    ],
)

MIDDLE_LINKS = _finger_chain(
    finger_id=3,
    base_translation=[0.017, 0.0, 0.026],
    base_rpy=[0.03, 1.48, -0.02],
    lengths=[0.034, 0.03, 0.02, 0.015],
    radii=[0.0044, 0.0041, 0.0035, 0.0027],
    limits=[
        _joint_limit(0.0, 1.02, 16.0, 1.9),
        _joint_limit(0.0, 1.48, 11.0, 1.7),
        _joint_limit(0.0, 1.28, 6.2, 1.5),
        _joint_limit(0.0, 1.02, 5.0, 1.4),
    ],
)

RING_LINKS = _finger_chain(
    finger_id=4,
    base_translation=[0.016, 0.004, 0.03],
    base_rpy=[0.03, 1.38, 0.05],
    lengths=[0.031, 0.028, 0.019, 0.014],
    radii=[0.0041, 0.0038, 0.0033, 0.0026],
    limits=[
        _joint_limit(0.0, 1.0, 15.0, 1.8),
        _joint_limit(0.0, 1.45, 10.5, 1.7),
        _joint_limit(0.0, 1.24, 6.0, 1.5),
        _joint_limit(0.0, 1.0, 4.8, 1.35),
    ],
)

LITTLE_LINKS = _finger_chain(
    finger_id=5,
    base_translation=[0.014, 0.0088, 0.033],
    base_rpy=[0.02, 1.28, 0.12],
    lengths=[0.026, 0.022, 0.016, 0.012],
    radii=[0.0038, 0.0034, 0.0030, 0.0023],
    limits=[
        _joint_limit(0.0, 0.95, 13.5, 1.8),
        _joint_limit(0.0, 1.35, 9.0, 1.6),
        _joint_limit(0.0, 1.16, 5.2, 1.35),
        _joint_limit(0.0, 0.95, 4.0, 1.2),
    ],
)


AUSTRALOPITHECUS_HAND = {
    "agent_code": "australopithecus_hand",
    "base_link": [BASE_LINK],
    "links": PALM_LINKS + THUMB_LINKS + INDEX_LINKS + MIDDLE_LINKS + RING_LINKS + LITTLE_LINKS,
    "evolution_information": [
        {
            "label": "design_basis",
            "value": "Australopithecus-inspired hand with long thumb, broad palm and mild phalange curvature.",
        }
    ],
    "evolution_id": "australopithecus_hand_asset_v5_skeletal_palm_region",
}


def _ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_file():
            child.unlink()


def _patch_default_root(cfg_path: Path, urdf_path: Path) -> None:
    text = cfg_path.read_text(encoding="utf-8")
    text = text.replace(
        'os.path.join(os.path.expanduser("~"), "Evolution")',
        'os.path.join(os.path.expanduser("~"), "Evolution_PC")',
    )
    rel_asset_path = urdf_path.relative_to(PROJECT_ROOT).as_posix()
    text = re.sub(
        r"ASSET_PATH = .+",
        f"ASSET_PATH = os.path.join(EVOLUTION_ROOT, {rel_asset_path!r})",
        text,
        count=1,
    )
    cfg_path.write_text(text, encoding="utf-8")


def _write_hand_cfg(urdf_path: Path, output_py_path: Path) -> None:
    parse_urdf_and_generate_articulation_cfg(str(urdf_path), str(urdf_path), str(output_py_path))
    _patch_default_root(output_py_path, urdf_path)


def _build_asset_variant(hand_dict: dict, root: Path, urdf_name: str) -> Path:
    mesh_root = root / "mesh"
    urdf_root = root / "urdf"
    urdf_path = urdf_root / urdf_name
    _ensure_clean_dir(mesh_root)
    urdf_root.mkdir(parents=True, exist_ok=True)
    generate_urdf_from_dict(hand_dict, output_dir=str(mesh_root), output_urdf=str(urdf_path))
    return urdf_path


def build_current_hand_assets(project_root: Path | None = None) -> dict[str, Path]:
    project_root = project_root or PROJECT_ROOT
    other_root = project_root / "Isaaclab_other"
    task_root = project_root / "evolution_tasks"

    live_right_root = other_root / "agent_for_isaaclab"
    live_left_root = other_root / "agent_for_isaaclab_mirror"
    archive_right_root = other_root / "australopithecus_hand_assets" / "right"
    archive_left_root = other_root / "australopithecus_hand_assets" / "left"

    right_cfg = task_root / "current_right_hand" / "current_right_hand_cfg.py"
    left_cfg = task_root / "current_left_hand" / "current_left_hand_cfg.py"

    mirror_hand = create_mirror_hand(deepcopy(AUSTRALOPITHECUS_HAND), "australopithecus_hand_mirror")

    live_right_urdf = _build_asset_variant(AUSTRALOPITHECUS_HAND, live_right_root, "current_agent.urdf")
    live_left_urdf = _build_asset_variant(mirror_hand, live_left_root, "current_agent.urdf")
    archive_right_urdf = _build_asset_variant(AUSTRALOPITHECUS_HAND, archive_right_root, "australopithecus_hand.urdf")
    archive_left_urdf = _build_asset_variant(mirror_hand, archive_left_root, "australopithecus_hand_mirror.urdf")

    right_cfg.parent.mkdir(parents=True, exist_ok=True)
    left_cfg.parent.mkdir(parents=True, exist_ok=True)
    _write_hand_cfg(live_right_urdf, right_cfg)
    _write_hand_cfg(live_left_urdf, left_cfg)

    return {
        "right_urdf": live_right_urdf,
        "left_urdf": live_left_urdf,
        "right_cfg": right_cfg,
        "left_cfg": left_cfg,
        "archive_right": archive_right_urdf,
        "archive_left": archive_left_urdf,
    }


if __name__ == "__main__":
    outputs = build_current_hand_assets()
    for name, path in outputs.items():
        print(f"{name}: {path}")

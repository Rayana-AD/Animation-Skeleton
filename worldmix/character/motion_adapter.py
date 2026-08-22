"""Motion adapters for BVH/Mixamo, AMASS, and text-command smoke demos."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from worldmix.character.standard_skeleton_gaussians import rotation_from_euler_xyz


EulerPose = dict[str, np.ndarray]


BVH_TO_WORLDMIX = {
    "Hips": "pelvis",
    "Spine": "spine",
    "Spine1": "chest",
    "Spine2": "chest",
    "Neck": "neck",
    "Head": "head",
    "LeftShoulder": "left_clavicle",
    "LeftArm": "left_upper_arm",
    "LeftForeArm": "left_lower_arm",
    "LeftHand": "left_hand",
    "RightShoulder": "right_clavicle",
    "RightArm": "right_upper_arm",
    "RightForeArm": "right_lower_arm",
    "RightHand": "right_hand",
    "LeftUpLeg": "left_upper_leg",
    "LeftLeg": "left_lower_leg",
    "LeftFoot": "left_foot",
    "LeftToeBase": "left_toe",
    "RightUpLeg": "right_upper_leg",
    "RightLeg": "right_lower_leg",
    "RightFoot": "right_foot",
    "RightToeBase": "right_toe",
}


SMPL_TO_WORLDMIX = {
    0: "pelvis",
    3: "spine",
    6: "chest",
    12: "neck",
    15: "head",
    13: "left_clavicle",
    16: "left_upper_arm",
    18: "left_lower_arm",
    20: "left_hand",
    14: "right_clavicle",
    17: "right_upper_arm",
    19: "right_lower_arm",
    21: "right_hand",
    1: "left_upper_leg",
    4: "left_lower_leg",
    7: "left_foot",
    10: "left_toe",
    2: "right_upper_leg",
    5: "right_lower_leg",
    8: "right_foot",
    11: "right_toe",
}


@dataclass(frozen=True)
class ActionKeyframe:
    """A semantic keyframe in normalized action time."""

    t: float
    pose_euler_deg: dict[str, tuple[float, float, float]]


@dataclass(frozen=True)
class ActionTemplate:
    """Rule-based smoke action template.

    This layer is intentionally replaceable: BVH/Mixamo/AMASS retrieval or a
    motion diffusion model can return the same pose sequence contract.
    """

    name: str
    aliases: tuple[str, ...]
    participating_bones: tuple[str, ...]
    keyframes: tuple[ActionKeyframe, ...]
    description: str


def _rot_axis(axis: str, angle_deg: float) -> np.ndarray:
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    if axis == "X":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "Y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "Z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"Unsupported rotation axis: {axis}")


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle vector to 3x3 rotation matrix."""
    theta = float(np.linalg.norm(axis_angle))
    if theta < 1e-12:
        return np.eye(3)
    axis = axis_angle / theta
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(theta) * k + (1.0 - np.cos(theta)) * (k @ k)


def read_bvh_motion(
    path: str | Path,
    bone_map: dict[str, str] | None = None,
    max_frames: int | None = None,
    downsample: int = 1,
) -> list[dict[str, np.ndarray]]:
    """Read BVH/Mixamo rotations into World-Mix pose dictionaries.

    This parser intentionally focuses on channel rotations. Translation/root
    trajectory can be added later without changing the motion JSON contract.
    """
    bone_map = bone_map or BVH_TO_WORLDMIX
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    channels: list[tuple[str, list[str]]] = []
    stack: list[str] = []
    in_motion = False
    frame_lines: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line == "MOTION":
            in_motion = True
            continue
        if not in_motion:
            parts = line.split()
            if parts[0] in {"ROOT", "JOINT"}:
                stack.append(parts[1])
            elif parts[0] == "End":
                stack.append("__end__")
            elif parts[0] == "}":
                if stack:
                    stack.pop()
            elif parts[0] == "CHANNELS" and stack and stack[-1] != "__end__":
                count = int(parts[1])
                channels.append((stack[-1], parts[2 : 2 + count]))
        else:
            if line.startswith("Frames:") or line.startswith("Frame Time:"):
                continue
            frame_lines.append(line)

    sequence: list[dict[str, np.ndarray]] = []
    selected_lines = frame_lines[:: max(1, downsample)]
    if max_frames is not None:
        selected_lines = selected_lines[:max_frames]

    for frame_line in selected_lines:
        values = [float(v) for v in frame_line.split()]
        cursor = 0
        pose: dict[str, np.ndarray] = {}
        for source_bone, channel_names in channels:
            local_rot = np.eye(3)
            for channel in channel_names:
                value = values[cursor]
                cursor += 1
                if channel.endswith("rotation"):
                    local_rot = local_rot @ _rot_axis(channel[0].upper(), value)
            target_bone = bone_map.get(source_bone)
            if target_bone:
                pose[target_bone] = local_rot
        sequence.append(pose)
    return sequence


def read_amass_motion(
    path: str | Path,
    max_frames: int | None = None,
    downsample: int = 1,
) -> list[dict[str, np.ndarray]]:
    """Read AMASS/SMPL .npz poses into World-Mix pose dictionaries."""
    data = np.load(path)
    poses = data["poses"]
    selected = poses[:: max(1, downsample)]
    if max_frames is not None:
        selected = selected[:max_frames]

    sequence: list[dict[str, np.ndarray]] = []
    for frame in selected:
        pose: dict[str, np.ndarray] = {}
        for joint_idx, target_bone in SMPL_TO_WORLDMIX.items():
            start = joint_idx * 3
            if start + 3 <= len(frame):
                pose[target_bone] = axis_angle_to_matrix(frame[start : start + 3])
        sequence.append(pose)
    return sequence


def write_motion_json(
    sequence: list[dict[str, np.ndarray]],
    path: str | Path,
    source: str = "adapter",
    fps: int = 30,
) -> None:
    """Write World-Mix motion JSON from pose dictionaries."""
    frames = []
    for pose in sequence:
        bones = {
            bone_name: {"matrix": matrix.tolist()}
            for bone_name, matrix in pose.items()
        }
        frames.append({"bones": bones})
    payload = {
        "format": "worldmix_motion_v1",
        "source": source,
        "fps": fps,
        "frames": frames,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pose(**bones: tuple[float, float, float]) -> dict[str, tuple[float, float, float]]:
    return dict(bones)


def _natural_stand_pose() -> dict[str, tuple[float, float, float]]:
    """Natural standing pose layered on top of the T/rest binding pose."""
    pose = _pose(
        chest=(0.0, 0.0, 0.0),
        neck=(0.0, 0.0, 0.0),
        head=(0.0, 0.0, 0.0),
        left_clavicle=(0.0, 0.0, 2.0),
        right_clavicle=(0.0, 0.0, -2.0),
        left_upper_arm=(0.0, 0.0, 0.0),
        right_upper_arm=(0.0, 0.0, 0.0),
        left_lower_arm=(0.0, 0.0, 3.0),
        right_lower_arm=(0.0, 0.0, -3.0),
        left_wrist=(0.0, 0.0, 0.0),
        right_wrist=(0.0, 0.0, 0.0),
        left_upper_leg=(0.0, 0.0, 0.0),
        right_upper_leg=(0.0, 0.0, 0.0),
        left_lower_leg=(0.0, 0.0, 0.0),
        right_lower_leg=(0.0, 0.0, 0.0),
        left_foot=(0.0, 0.0, 0.0),
        right_foot=(0.0, 0.0, 0.0),
    )
    for side, sign in [("left", 1.0), ("right", -1.0)]:
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            pose[f"{side}_{finger}_01"] = (0.0, 0.0, sign * 7.0)
            pose[f"{side}_{finger}_02"] = (0.0, 0.0, sign * 5.0)
            pose[f"{side}_{finger}_03"] = (0.0, 0.0, sign * 3.0)
    return pose


def _merge_pose(*poses: dict[str, tuple[float, float, float]]) -> dict[str, tuple[float, float, float]]:
    merged: dict[str, tuple[float, float, float]] = {}
    for pose in poses:
        merged.update(pose)
    return merged


def _action_templates() -> list[ActionTemplate]:
    stand = _natural_stand_pose()
    walk_bones = (
        "chest",
        "neck",
        "head",
        "left_clavicle",
        "right_clavicle",
        "left_upper_arm",
        "right_upper_arm",
        "left_lower_arm",
        "right_lower_arm",
        "left_upper_leg",
        "right_upper_leg",
        "left_lower_leg",
        "right_lower_leg",
        "left_foot",
        "right_foot",
        "left_toe",
        "right_toe",
    )
    wave_bones = (
        "chest",
        "neck",
        "head",
        "right_clavicle",
        "right_upper_arm",
        "right_lower_arm",
        "right_wrist",
        "right_hand",
    )
    jump_bones = (
        "chest",
        "left_upper_arm",
        "right_upper_arm",
        "left_upper_leg",
        "right_upper_leg",
        "left_lower_leg",
        "right_lower_leg",
        "left_foot",
        "right_foot",
    )
    return [
        ActionTemplate(
            name="walk",
            aliases=("walk", "walking", "走", "走路", "行走"),
            participating_bones=walk_bones,
            description="Natural arm-down walk smoke template with opposite arm/leg swing.",
            keyframes=(
                ActionKeyframe(0.0, stand),
                ActionKeyframe(
                    0.25,
                    _merge_pose(
                        stand,
                        _pose(
                            chest=(0.0, 2.0, 0.0),
                            left_upper_arm=(8.0, 0.0, 4.0),
                            right_upper_arm=(-8.0, 0.0, -4.0),
                            left_lower_arm=(2.0, 0.0, 5.0),
                            right_lower_arm=(-2.0, 0.0, -3.0),
                            left_upper_leg=(8.0, 0.0, 0.0),
                            right_upper_leg=(-8.0, 0.0, 0.0),
                            left_lower_leg=(-8.0, 0.0, 0.0),
                            right_lower_leg=(0.0, 0.0, 0.0),
                            left_foot=(3.0, 0.0, 0.0),
                            right_foot=(-2.0, 0.0, 0.0),
                        ),
                    ),
                ),
                ActionKeyframe(0.5, stand),
                ActionKeyframe(
                    0.75,
                    _merge_pose(
                        stand,
                        _pose(
                            chest=(0.0, -2.0, 0.0),
                            left_upper_arm=(-8.0, 0.0, 4.0),
                            right_upper_arm=(8.0, 0.0, -4.0),
                            left_lower_arm=(-2.0, 0.0, 3.0),
                            right_lower_arm=(2.0, 0.0, -5.0),
                            left_upper_leg=(-8.0, 0.0, 0.0),
                            right_upper_leg=(8.0, 0.0, 0.0),
                            left_lower_leg=(0.0, 0.0, 0.0),
                            right_lower_leg=(-8.0, 0.0, 0.0),
                            left_foot=(-2.0, 0.0, 0.0),
                            right_foot=(3.0, 0.0, 0.0),
                        ),
                    ),
                ),
                ActionKeyframe(1.0, stand),
            ),
        ),
        ActionTemplate(
            name="wave",
            aliases=("wave", "waving", "挥手", "招手"),
            participating_bones=wave_bones,
            description="Raise right arm and wave with forearm/wrist while body remains balanced.",
            keyframes=(
                ActionKeyframe(0.0, stand),
                ActionKeyframe(
                    0.2,
                    _merge_pose(
                        stand,
                        _pose(
                            chest=(0.0, -3.0, 0.0),
                            right_clavicle=(0.0, 0.0, -16.0),
                            right_upper_arm=(0.0, 0.0, -105.0),
                            right_lower_arm=(0.0, 0.0, -42.0),
                            right_wrist=(0.0, 0.0, -10.0),
                        ),
                    ),
                ),
                ActionKeyframe(
                    0.45,
                    _merge_pose(stand, _pose(right_upper_arm=(0.0, 0.0, -105.0), right_lower_arm=(0.0, 0.0, -62.0), right_wrist=(0.0, 0.0, 18.0))),
                ),
                ActionKeyframe(
                    0.7,
                    _merge_pose(stand, _pose(right_upper_arm=(0.0, 0.0, -105.0), right_lower_arm=(0.0, 0.0, -38.0), right_wrist=(0.0, 0.0, -18.0))),
                ),
                ActionKeyframe(1.0, stand),
            ),
        ),
        ActionTemplate(
            name="jump",
            aliases=("jump", "jumping", "跳", "跳跃"),
            participating_bones=jump_bones,
            description="Crouch then extend arms and legs for a jump smoke pose.",
            keyframes=(
                ActionKeyframe(0.0, stand),
                ActionKeyframe(
                    0.25,
                    _merge_pose(
                        stand,
                        _pose(
                            left_upper_leg=(-12.0, 0.0, 0.0),
                            right_upper_leg=(-12.0, 0.0, 0.0),
                            left_lower_leg=(-16.0, 0.0, 0.0),
                            right_lower_leg=(-16.0, 0.0, 0.0),
                            left_upper_arm=(0.0, 0.0, 40.0),
                            right_upper_arm=(0.0, 0.0, -40.0),
                        ),
                    ),
                ),
                ActionKeyframe(
                    0.55,
                    _merge_pose(stand, _pose(left_upper_arm=(0.0, 0.0, 72.0), right_upper_arm=(0.0, 0.0, -72.0), left_lower_leg=(6.0, 0.0, 0.0), right_lower_leg=(6.0, 0.0, 0.0))),
                ),
                ActionKeyframe(1.0, stand),
            ),
        ),
    ]


def list_action_templates() -> list[str]:
    """Return available smoke action template names."""
    return [template.name for template in _action_templates()]


def select_action_template(command: str) -> ActionTemplate:
    """Map a language command to an action template."""
    lowered = command.lower()
    for template in _action_templates():
        if any(alias in lowered for alias in template.aliases):
            return template
    return _action_templates()[0]


def _interpolate_euler_pose(
    a: dict[str, tuple[float, float, float]],
    b: dict[str, tuple[float, float, float]],
    alpha: float,
) -> EulerPose:
    bones = set(a) | set(b)
    pose: EulerPose = {}
    for bone in bones:
        va = np.asarray(a.get(bone, (0.0, 0.0, 0.0)), dtype=np.float64)
        vb = np.asarray(b.get(bone, (0.0, 0.0, 0.0)), dtype=np.float64)
        pose[bone] = (1.0 - alpha) * va + alpha * vb
    return pose


def render_action_template(
    template: ActionTemplate,
    frames: int = 24,
) -> list[dict[str, np.ndarray]]:
    """Render an action template into World-Mix per-frame rotation matrices."""
    keyframes = sorted(template.keyframes, key=lambda k: k.t)
    sequence: list[dict[str, np.ndarray]] = []
    for frame_idx in range(frames):
        phase = frame_idx / max(frames - 1, 1)
        left = keyframes[0]
        right = keyframes[-1]
        for i in range(len(keyframes) - 1):
            if keyframes[i].t <= phase <= keyframes[i + 1].t:
                left, right = keyframes[i], keyframes[i + 1]
                break
        span = max(right.t - left.t, 1e-6)
        alpha = np.clip((phase - left.t) / span, 0.0, 1.0)
        # Smoothstep interpolation keeps start/end frames from snapping.
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        euler_pose = _interpolate_euler_pose(left.pose_euler_deg, right.pose_euler_deg, float(alpha))
        matrix_pose = {
            bone: rotation_from_euler_xyz(np.radians(euler_deg))
            for bone, euler_deg in euler_pose.items()
        }
        sequence.append(matrix_pose)
    return sequence


def text_command_to_motion(
    command: str,
    frames: int = 24,
) -> list[dict[str, np.ndarray]]:
    """Rule-based language-to-motion baseline for smoke E2E validation."""
    return render_action_template(select_action_template(command), frames=frames)

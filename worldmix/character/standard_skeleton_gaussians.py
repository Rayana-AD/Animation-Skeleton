"""Standard skeleton + bone-attached Gaussian splats for animation validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from worldmix.core.gaussian import GaussianSplat, rotation_matrix_to_quaternion


@dataclass
class StandardBone:
    """Rest-pose humanoid bone segment."""

    name: str
    parent: int
    head: np.ndarray
    tail: np.ndarray
    radius: float


@dataclass
class BoneGaussian:
    """Gaussian stored in the local coordinate system of one bone."""

    bone_id: int
    local_mean: np.ndarray
    local_scale: np.ndarray
    local_rotation: np.ndarray


@dataclass
class SkinnedBoneGaussian:
    """Gaussian with linear blend skinning weights across multiple bones."""

    bone_ids: np.ndarray
    weights: np.ndarray
    local_means: np.ndarray
    local_scale: np.ndarray
    local_rotation: np.ndarray


@dataclass
class RiggedGaussianAvatar:
    """Standard humanoid skeleton with Gaussian splats attached per bone."""

    bones: list[StandardBone]
    gaussians: list[BoneGaussian | SkinnedBoneGaussian]
    colors: np.ndarray | None = None


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        return v
    return v / norm


def _axis_frame(head: np.ndarray, tail: np.ndarray) -> np.ndarray:
    """Build a right-handed frame whose Y axis follows the bone."""
    y_axis = _normalize(tail - head)
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(y_axis, helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    x_axis = _normalize(np.cross(helper, y_axis))
    z_axis = _normalize(np.cross(y_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_from_euler_xyz(euler_rad: np.ndarray) -> np.ndarray:
    """Build rotation matrix from XYZ Euler angles in radians."""
    x, y, z = euler_rad
    return _rot_z(z) @ _rot_y(y) @ _rot_x(x)


def build_standard_humanoid_skeleton() -> list[StandardBone]:
    """Create a full humanoid rest skeleton with hands, fingers, feet, and toes."""
    bones: list[StandardBone] = []

    def add(name: str, parent: int, head, tail, radius: float) -> int:
        bones.append(
            StandardBone(
                name=name,
                parent=parent,
                head=np.asarray(head, dtype=np.float64),
                tail=np.asarray(tail, dtype=np.float64),
                radius=radius,
            )
        )
        return len(bones) - 1

    pelvis = add("pelvis", -1, [0.0, 0.0, 0.0], [0.0, 0.22, 0.0], 0.18)
    spine = add("spine", pelvis, [0.0, 0.22, 0.0], [0.0, 0.55, 0.0], 0.16)
    chest = add("chest", spine, [0.0, 0.55, 0.0], [0.0, 0.82, 0.0], 0.20)
    neck = add("neck", chest, [0.0, 0.82, 0.0], [0.0, 0.93, 0.0], 0.08)
    add("head", neck, [0.0, 0.93, 0.0], [0.0, 1.08, 0.0], 0.13)

    def add_arm(side: str, sign: float) -> None:
        clavicle = add(f"{side}_clavicle", chest, [0.0, 0.77, 0.0], [sign * 0.17, 0.77, 0.0], 0.055)
        upper_arm = add(
            f"{side}_upper_arm",
            clavicle,
            [sign * 0.17, 0.77, 0.0],
            [sign * 0.27, 0.50, 0.0],
            0.07,
        )
        lower_arm = add(
            f"{side}_lower_arm",
            upper_arm,
            [sign * 0.27, 0.50, 0.0],
            [sign * 0.24, 0.23, 0.0],
            0.055,
        )
        wrist = add(f"{side}_wrist", lower_arm, [sign * 0.24, 0.23, 0.0], [sign * 0.23, 0.18, 0.0], 0.04)
        hand = add(f"{side}_hand", wrist, [sign * 0.23, 0.18, 0.0], [sign * 0.23, 0.08, 0.0], 0.055)

        finger_specs = [
            ("thumb", -0.035, -0.045, 0.035, 0.045),
            ("index", -0.018, -0.060, 0.020, 0.050),
            ("middle", 0.000, -0.070, 0.000, 0.055),
            ("ring", 0.018, -0.060, -0.020, 0.050),
            ("pinky", 0.034, -0.045, -0.038, 0.040),
        ]
        for fname, dz, length0, spread, radius in finger_specs:
            start = np.array([sign * 0.23, 0.08, dz])
            direction = np.array([sign * 0.15, -1.0, 0.15 * dz + spread])
            direction = direction / (np.linalg.norm(direction) + 1e-12)
            p0 = start
            p1 = p0 + direction * abs(length0)
            p2 = p1 + direction * abs(length0) * 0.75
            p3 = p2 + direction * abs(length0) * 0.55
            mcp = add(f"{side}_{fname}_01", hand, p0, p1, radius * 0.45)
            pip = add(f"{side}_{fname}_02", mcp, p1, p2, radius * 0.38)
            add(f"{side}_{fname}_03", pip, p2, p3, radius * 0.30)

    add_arm("left", -1.0)
    add_arm("right", 1.0)

    def add_leg(side: str, sign: float) -> None:
        upper_leg = add(f"{side}_upper_leg", pelvis, [sign * 0.09, 0.0, 0.0], [sign * 0.13, -0.42, 0.0], 0.09)
        lower_leg = add(
            f"{side}_lower_leg",
            upper_leg,
            [sign * 0.13, -0.42, 0.0],
            [sign * 0.12, -0.82, 0.0],
            0.07,
        )
        foot = add(f"{side}_foot", lower_leg, [sign * 0.12, -0.82, 0.0], [sign * 0.13, -0.92, 0.18], 0.065)
        add(f"{side}_toe", foot, [sign * 0.13, -0.92, 0.18], [sign * 0.13, -0.92, 0.32], 0.045)

    add_leg("left", -1.0)
    add_leg("right", 1.0)
    return bones


def _sample_bone_gaussians(
    bone_id: int,
    bone: StandardBone,
    count: int,
    rng: np.random.Generator,
) -> list[BoneGaussian]:
    length = float(np.linalg.norm(bone.tail - bone.head))
    splats: list[BoneGaussian] = []
    for _ in range(count):
        y = rng.uniform(0.05 * length, 0.95 * length)
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radial = abs(rng.normal(0.0, bone.radius * 0.45))
        x = np.cos(angle) * radial
        z = np.sin(angle) * radial
        local_mean = np.array([x, y, z])
        local_scale = np.array([bone.radius * 0.22, length * 0.035, bone.radius * 0.22])
        local_rotation = np.eye(3)
        splats.append(BoneGaussian(bone_id, local_mean, local_scale, local_rotation))
    return splats


def build_rigged_gaussian_avatar(points_per_bone: int = 24, seed: int = 42) -> RiggedGaussianAvatar:
    """Create a standard skeleton avatar with splats attached to bones."""
    bones = build_standard_humanoid_skeleton()
    return build_rigged_gaussian_avatar_from_bones(bones, points_per_bone=points_per_bone, seed=seed)


def build_rigged_gaussian_avatar_from_bones(
    bones: list[StandardBone],
    points_per_bone: int = 24,
    seed: int = 42,
) -> RiggedGaussianAvatar:
    """Attach bone-local Gaussians to an existing skeleton (e.g. ViT-retargeted proportions)."""
    rng = np.random.default_rng(seed)
    gaussians: list[BoneGaussian] = []
    for bone_id, bone in enumerate(bones):
        count = points_per_bone
        if bone.name in {"pelvis", "spine", "chest"}:
            count = int(points_per_bone * 1.4)
        elif bone.name in {"neck", "head"}:
            count = int(points_per_bone * 0.8)
        elif "finger" in bone.name or any(name in bone.name for name in ["thumb", "index", "middle", "ring", "pinky"]):
            count = max(3, int(points_per_bone * 0.25))
        elif bone.name.endswith("_toe") or bone.name.endswith("_wrist"):
            count = max(4, int(points_per_bone * 0.35))
        elif bone.name.endswith("_hand") or bone.name.endswith("_foot") or bone.name.endswith("_clavicle"):
            count = max(5, int(points_per_bone * 0.5))
        gaussians.extend(_sample_bone_gaussians(bone_id, bone, count, rng))
    return RiggedGaussianAvatar(bones=bones, gaussians=gaussians)


def walk_pose(frame: int, total_frames: int) -> dict[str, np.ndarray]:
    """Conservative debug walk pose in local bone coordinates.

    This is intentionally small-amplitude. It validates skeleton-driven
    Gaussians without pretending to be a trained gait model or IK solver.
    """
    t = 2.0 * np.pi * frame / max(total_frames, 1)
    swing = np.sin(t)
    counter = np.sin(t + np.pi)
    left_lift = max(0.0, swing)
    right_lift = max(0.0, counter)
    pose = {
        "neck": _rot_y(0.025 * counter),
        "head": _rot_y(0.020 * counter),
        "left_clavicle": _rot_z(0.012 * counter),
        "left_upper_arm": _rot_x(0.14 * counter),
        "left_lower_arm": _rot_x(0.05 + 0.04 * counter),
        "left_wrist": _rot_z(0.035 * counter),
        "right_clavicle": _rot_z(-0.012 * swing),
        "right_upper_arm": _rot_x(-0.14 * swing),
        "right_lower_arm": _rot_x(-0.05 - 0.04 * swing),
        "right_wrist": _rot_z(-0.035 * swing),
        "left_upper_leg": _rot_x(0.16 * swing),
        "left_lower_leg": _rot_x(-0.22 * left_lift),
        "left_foot": _rot_x(0.10 * left_lift - 0.05 * right_lift),
        "left_toe": _rot_x(0.06 * left_lift),
        "right_upper_leg": _rot_x(0.16 * counter),
        "right_lower_leg": _rot_x(-0.22 * right_lift),
        "right_foot": _rot_x(0.10 * right_lift - 0.05 * left_lift),
        "right_toe": _rot_x(0.06 * right_lift),
        "chest": _rot_y(0.035 * swing),
    }
    for side, phase in [("left", counter), ("right", swing)]:
        curl = 0.12 + 0.08 * max(0.0, phase)
        for finger in ["thumb", "index", "middle", "ring", "pinky"]:
            pose[f"{side}_{finger}_01"] = _rot_z(curl * (0.7 if finger == "thumb" else 1.0))
            pose[f"{side}_{finger}_02"] = _rot_z(curl * 0.7)
            pose[f"{side}_{finger}_03"] = _rot_z(curl * 0.45)
    return pose


def load_motion_json(path: str | Path) -> list[dict[str, np.ndarray]]:
    """Load a simple motion JSON sequence.

    Expected format:
    {
      "frames": [
        {"bones": {"left_upper_arm": {"euler_deg": [0, 0, 20]}}},
        {"bones": {"left_upper_arm": {"matrix": [[...], [...], [...]]}}}
      ]
    }
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    sequence: list[dict[str, np.ndarray]] = []
    for frame in frames:
        pose: dict[str, np.ndarray] = {}
        for bone_name, spec in frame.get("bones", {}).items():
            if "matrix" in spec:
                pose[bone_name] = np.asarray(spec["matrix"], dtype=np.float64)
            elif "euler_deg" in spec:
                pose[bone_name] = rotation_from_euler_xyz(np.radians(np.asarray(spec["euler_deg"], dtype=np.float64)))
            elif "euler_rad" in spec:
                pose[bone_name] = rotation_from_euler_xyz(np.asarray(spec["euler_rad"], dtype=np.float64))
        sequence.append(pose)
    return sequence


def write_sample_motion_json(path: str | Path, frames: int = 24) -> None:
    """Write a retarget-friendly sample motion JSON generated from walk_pose."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_payload = []
    for frame_idx in range(frames):
        t = 2.0 * np.pi * frame_idx / max(frames, 1)
        swing = np.sin(t)
        counter = np.sin(t + np.pi)
        frame = {
            "bones": {
                "left_upper_arm": {"euler_deg": [0.0, 0.0, 14.0 * counter]},
                "right_upper_arm": {"euler_deg": [0.0, 0.0, -14.0 * swing]},
                "left_upper_leg": {"euler_deg": [9.0 * swing, 0.0, 0.0]},
                "right_upper_leg": {"euler_deg": [9.0 * counter, 0.0, 0.0]},
                "chest": {"euler_deg": [0.0, 2.0 * swing, 0.0]},
                "neck": {"euler_deg": [0.0, 1.5 * counter, 0.0]},
            }
        }
        frame_payload.append(frame)
    payload = {
        "format": "worldmix_motion_v1",
        "description": "Sample retarget-friendly walk pose sequence.",
        "frames": frame_payload,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pose_avatar(
    avatar: RiggedGaussianAvatar,
    pose: dict[str, np.ndarray] | None = None,
) -> tuple[list[GaussianSplat], list[tuple[np.ndarray, np.ndarray]]]:
    """Evaluate world-space Gaussian splats and bone segments for a pose."""
    pose = pose or {}
    frames: list[np.ndarray] = []
    heads: list[np.ndarray] = []
    tails: list[np.ndarray] = []

    for bone in avatar.bones:
        rest_frame = _axis_frame(bone.head, bone.tail)
        local_rot = pose.get(bone.name, np.eye(3))
        if bone.parent < 0:
            head = bone.head.copy()
            frame = rest_frame @ local_rot
        else:
            parent = avatar.bones[bone.parent]
            parent_rest_frame = _axis_frame(parent.head, parent.tail)
            offset = parent_rest_frame.T @ (bone.head - parent.tail)
            head = tails[bone.parent] + frames[bone.parent] @ offset
            relative = parent_rest_frame.T @ rest_frame
            frame = frames[bone.parent] @ relative @ local_rot
        length = np.linalg.norm(bone.tail - bone.head)
        tail = head + frame @ np.array([0.0, length, 0.0])
        frames.append(frame)
        heads.append(head)
        tails.append(tail)

    world_gaussians: list[GaussianSplat] = []
    for splat in avatar.gaussians:
        if isinstance(splat, SkinnedBoneGaussian):
            mean = np.zeros(3)
            for bone_id, weight, local_mean in zip(splat.bone_ids, splat.weights, splat.local_means):
                mean += float(weight) * (heads[int(bone_id)] + frames[int(bone_id)] @ local_mean)
            primary_id = int(splat.bone_ids[int(np.argmax(splat.weights))])
            rot = frames[primary_id] @ splat.local_rotation
        else:
            frame = frames[splat.bone_id]
            mean = heads[splat.bone_id] + frame @ splat.local_mean
            rot = frame @ splat.local_rotation
        world_gaussians.append(
            GaussianSplat(
                mean=mean,
                scale=splat.local_scale,
                rotation=rotation_matrix_to_quaternion(rot),
            )
        )
    bone_segments = list(zip(heads, tails))
    return world_gaussians, bone_segments


def plot_avatar_frame(
    gaussians: list[GaussianSplat],
    bone_segments: list[tuple[np.ndarray, np.ndarray]],
    title: str,
    bone_radii: list[float] | None = None,
    show_ellipsoids: bool = False,
    show_skin_proxy: bool = True,
    point_size: float = 8.0,
    point_colors: np.ndarray | None = None,
) -> plt.Figure:
    """Plot a posed avatar frame with bone lines and Gaussian centers."""
    fig = plt.figure(figsize=(6, 7))
    ax = fig.add_subplot(111, projection="3d")
    pts = np.stack([g.mean for g in gaussians])
    scatter_color = point_colors if point_colors is not None else "#A23B72"
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], s=point_size, c=scatter_color, depthshade=True)

    if show_skin_proxy:
        bone_radii = bone_radii or [0.05] * len(bone_segments)
        for (head, tail), radius in zip(bone_segments, bone_radii):
            x, z, y = _bone_capsule_surface(head, tail, radius)
            ax.plot_surface(x, z, y, color="#E8A6C8", alpha=0.28, linewidth=0, shade=True)

    for head, tail in bone_segments:
        ax.plot3D([head[0], tail[0]], [head[2], tail[2]], [head[1], tail[1]], c="#1F2937", lw=2.0)

    if show_ellipsoids:
        from worldmix.viz.ellipsoid import _ellipsoid_faces
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        step = max(1, len(gaussians) // 70)
        for g in gaussians[::step]:
            faces = _ellipsoid_faces(g.mean, g.covariance, n=8)
            poly = Poly3DCollection(faces, alpha=0.08, facecolor="#A23B72", edgecolor="none")
            ax.add_collection3d(poly)

    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.9, 1.15)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.set_title(title)
    return fig


def _bone_capsule_surface(
    head: np.ndarray,
    tail: np.ndarray,
    radius: float,
    rings: int = 8,
    sides: int = 14,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Low-poly cylinder surface used as a skin proxy around one bone."""
    frame = _axis_frame(head, tail)
    length = float(np.linalg.norm(tail - head))
    ys = np.linspace(0.0, length, rings)
    angles = np.linspace(0.0, 2.0 * np.pi, sides)
    x_grid = np.zeros((rings, sides))
    y_grid = np.zeros((rings, sides))
    z_grid = np.zeros((rings, sides))
    for i, y in enumerate(ys):
        taper = 0.85 + 0.15 * np.sin(np.pi * i / max(rings - 1, 1))
        for j, angle in enumerate(angles):
            local = np.array([np.cos(angle) * radius * taper, y, np.sin(angle) * radius * taper])
            world = head + frame @ local
            x_grid[i, j] = world[0]
            y_grid[i, j] = world[1]
            z_grid[i, j] = world[2]
    return x_grid, z_grid, y_grid


def save_avatar_animation(
    avatar: RiggedGaussianAvatar,
    output_dir: str | Path,
    frames: int = 24,
    fps: int = 12,
    show_ellipsoids: bool = False,
    show_skin_proxy: bool = True,
    motion_sequence: list[dict[str, np.ndarray]] | None = None,
    point_size: float = 8.0,
) -> Path:
    """Render a procedural avatar animation to PNG frames and GIF."""
    output_dir = Path(output_dir)
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    images: list[Image.Image] = []
    frame_count = len(motion_sequence) if motion_sequence else frames
    for frame_idx in range(frame_count):
        pose = motion_sequence[frame_idx] if motion_sequence else walk_pose(frame_idx, frame_count)
        gaussians, segments = pose_avatar(avatar, pose)
        fig = plot_avatar_frame(
            gaussians,
            segments,
            title=f"Full Skeleton GS Animation frame {frame_idx:02d}",
            bone_radii=[b.radius for b in avatar.bones],
            show_ellipsoids=show_ellipsoids,
            show_skin_proxy=show_skin_proxy,
            point_size=point_size,
            point_colors=avatar.colors,
        )
        frame_path = frame_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        images.append(Image.open(frame_path).convert("RGB"))

    gif_path = output_dir / "standard_skeleton_gaussian_walk.gif"
    images[0].save(
        gif_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        disposal=2,
        optimize=False,
    )
    return gif_path

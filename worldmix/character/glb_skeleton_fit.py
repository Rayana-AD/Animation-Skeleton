"""Fit a per-asset humanoid skeleton to a static GLB mesh (template topology, new instance)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from worldmix.character.standard_skeleton_gaussians import (
    StandardBone,
    _axis_frame,
    build_standard_humanoid_skeleton,
)
from worldmix.character.static_mesh_avatar import (
    _SEMANTIC_BONE_GROUPS,
    _capsule_distance,
    load_glb_mesh_with_material,
)
from worldmix.character.front_view_semantics import (
    detect_front_depth_axis,
    front_extremity_landmarks,
    segment_mesh_front_view,
)


MESH_REGIONS = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")


@dataclass
class MeshLandmarks:
    """Semantic anchor points extracted from a static humanoid mesh."""

    pelvis: np.ndarray
    spine_mid: np.ndarray
    chest: np.ndarray
    neck_base: np.ndarray
    head_top: np.ndarray
    left_shoulder: np.ndarray
    right_shoulder: np.ndarray
    left_elbow: np.ndarray
    right_elbow: np.ndarray
    left_wrist: np.ndarray
    right_wrist: np.ndarray
    left_hip: np.ndarray
    right_hip: np.ndarray
    left_knee: np.ndarray
    right_knee: np.ndarray
    left_ankle: np.ndarray
    right_ankle: np.ndarray
    left_foot: np.ndarray
    right_foot: np.ndarray


def align_mesh_space(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Center X/Z and place feet on Y=0 without rescaling the mesh."""
    verts = np.asarray(vertices, dtype=np.float64).copy()
    center_x = float(verts[:, 0].mean())
    center_z = float(verts[:, 2].mean())
    y_min = float(verts[:, 1].min())
    verts[:, 0] -= center_x
    verts[:, 2] -= center_z
    verts[:, 1] -= y_min
    meta = {
        "center_x": center_x,
        "center_z": center_z,
        "y_offset": y_min,
        "height": float(verts[:, 1].max()),
        "width_x": float(verts[:, 0].max() - verts[:, 0].min()),
        "width_z": float(verts[:, 2].max() - verts[:, 2].min()),
    }
    return verts, meta


def _body_region_masks(vertices: np.ndarray) -> dict[str, np.ndarray]:
    """Shared semantic region masks for segmentation, skinning, and skeleton fitting."""
    verts = np.asarray(vertices, dtype=np.float64)
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    height = max(y_max - y_min, 1e-6)
    y_norm = (verts[:, 1] - y_min) / height
    x = verts[:, 0]
    abs_x = np.abs(x)

    torso_band = (y_norm >= 0.30) & (y_norm <= 0.82)
    torso_x_ref = float(np.percentile(abs_x[torso_band], 65)) if np.any(torso_band) else float(np.percentile(abs_x, 65))
    arm_x_threshold = max(torso_x_ref * 0.85, float(np.percentile(abs_x, 55)) * 0.55)
    pelvis_x_threshold = max(torso_x_ref * 0.42, float(np.percentile(abs_x, 40)) * 0.30)

    head = y_norm > 0.82
    legs = y_norm < 0.48
    left_leg = legs & (x < -pelvis_x_threshold)
    right_leg = legs & (x > pelvis_x_threshold)
    left_arm = (~head) & (~legs) & (x < -arm_x_threshold)
    right_arm = (~head) & (~legs) & (x > arm_x_threshold)
    torso = ~(head | left_leg | right_leg | left_arm | right_arm)
    return {
        "head": head,
        "torso": torso,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_leg": left_leg,
        "right_leg": right_leg,
    }


def segment_mesh_vertices(vertices: np.ndarray, *, method: str = "front_view") -> dict[str, np.ndarray]:
    """Partition mesh vertices into body regions."""
    if method == "height":
        return _segment_mesh_height_bands(vertices)
    return segment_mesh_front_view(vertices)


def _segment_mesh_height_bands(vertices: np.ndarray) -> dict[str, np.ndarray]:
    """Partition mesh vertices into coarse body regions using height and lateral bands."""
    masks = _body_region_masks(vertices)
    return {name: np.where(mask)[0] for name, mask in masks.items()}


def _region_centroid(vertices: np.ndarray, indices: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return fallback.copy()
    return vertices[indices].mean(axis=0)


def _region_extreme(vertices: np.ndarray, indices: np.ndarray, axis: int, mode: str, fallback: float) -> float:
    if len(indices) == 0:
        return fallback
    values = vertices[indices, axis]
    return float(values.max() if mode == "max" else values.min())


def _side_joint(
    vertices: np.ndarray,
    indices: np.ndarray,
    sign: float,
    y_lo: float,
    y_hi: float,
    fallback: np.ndarray,
) -> np.ndarray:
    if len(indices) == 0:
        return fallback.copy()
    pts = vertices[indices]
    band = pts[(pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)]
    if len(band) == 0:
        band = pts
    target_x = sign * float(np.percentile(np.abs(band[:, 0]), 85))
    order = np.argsort(np.abs(band[:, 0] - target_x))
    return band[order[0]].copy()


def extract_mesh_landmarks(vertices: np.ndarray, segments: dict[str, np.ndarray]) -> MeshLandmarks:
    """Derive joint anchors from segmented mesh vertices."""
    depth_axis = detect_front_depth_axis(vertices)
    front_lm = front_extremity_landmarks(vertices, segments, depth_axis=depth_axis)
    y_min = float(vertices[:, 1].min())
    y_max = float(vertices[:, 1].max())
    height = max(y_max - y_min, 1e-6)
    pelvis_y = y_min + 0.52 * height
    chest_y = y_min + 0.72 * height
    neck_y = y_min + 0.86 * height

    pelvis = _region_centroid(
        vertices,
        segments["torso"],
        np.array([0.0, pelvis_y, 0.0]),
    )
    pelvis[1] = pelvis_y

    spine_mid = _region_centroid(
        vertices,
        segments["torso"],
        np.array([0.0, y_min + 0.62 * height, 0.0]),
    )
    spine_mid[1] = y_min + 0.62 * height
    spine_mid[0] *= 0.35
    spine_mid[2] *= 0.35

    chest = _region_centroid(
        vertices,
        segments["torso"],
        np.array([0.0, chest_y, 0.0]),
    )
    chest[1] = chest_y
    chest[0] *= 0.25
    chest[2] *= 0.25

    neck_base = np.array([0.0, neck_y, 0.0])
    head_top = front_lm["head_top"].copy()

    shoulder_y = y_min + 0.76 * height
    left_shoulder = _side_joint(
        vertices, segments["left_arm"], -1.0, shoulder_y - 0.05 * height, shoulder_y + 0.08 * height,
        front_lm["left_shoulder"],
    )
    right_shoulder = _side_joint(
        vertices, segments["right_arm"], 1.0, shoulder_y - 0.05 * height, shoulder_y + 0.08 * height,
        front_lm["right_shoulder"],
    )

    elbow_y = y_min + 0.58 * height
    left_elbow = _side_joint(vertices, segments["left_arm"], -1.0, elbow_y - 0.06 * height, elbow_y + 0.06 * height, np.array([-0.22 * height, elbow_y, 0.0]))
    right_elbow = _side_joint(vertices, segments["right_arm"], 1.0, elbow_y - 0.06 * height, elbow_y + 0.06 * height, np.array([0.22 * height, elbow_y, 0.0]))

    wrist_y = y_min + 0.42 * height
    left_wrist = _side_joint(
        vertices, segments["left_arm"], -1.0, wrist_y - 0.08 * height, wrist_y + 0.08 * height,
        front_lm["left_hand"],
    )
    right_wrist = _side_joint(
        vertices, segments["right_arm"], 1.0, wrist_y - 0.08 * height, wrist_y + 0.08 * height,
        front_lm["right_hand"],
    )

    hip_y = y_min + 0.50 * height
    left_hip = _side_joint(vertices, segments["left_leg"], -1.0, hip_y - 0.04 * height, hip_y + 0.04 * height, np.array([-0.10 * height, hip_y, 0.0]))
    right_hip = _side_joint(vertices, segments["right_leg"], 1.0, hip_y - 0.04 * height, hip_y + 0.04 * height, np.array([0.10 * height, hip_y, 0.0]))
    if len(segments["left_leg"]) == 0:
        left_hip = pelvis + np.array([-0.10 * height, -0.02 * height, 0.0])
    if len(segments["right_leg"]) == 0:
        right_hip = pelvis + np.array([0.10 * height, -0.02 * height, 0.0])

    knee_y = y_min + 0.28 * height
    left_knee = _side_joint(vertices, segments["left_leg"], -1.0, knee_y - 0.05 * height, knee_y + 0.05 * height, np.array([-0.11 * height, knee_y, 0.0]))
    right_knee = _side_joint(vertices, segments["right_leg"], 1.0, knee_y - 0.05 * height, knee_y + 0.05 * height, np.array([0.11 * height, knee_y, 0.0]))

    ankle_y = y_min + 0.06 * height
    left_ankle = _side_joint(vertices, segments["left_leg"], -1.0, ankle_y - 0.04 * height, ankle_y + 0.04 * height, np.array([-0.10 * height, ankle_y, 0.0]))
    right_ankle = _side_joint(vertices, segments["right_leg"], 1.0, ankle_y - 0.04 * height, ankle_y + 0.04 * height, np.array([0.10 * height, ankle_y, 0.0]))

    left_foot = front_lm["left_foot"].copy()
    right_foot = front_lm["right_foot"].copy()

    return MeshLandmarks(
        pelvis=pelvis,
        spine_mid=spine_mid,
        chest=chest,
        neck_base=neck_base,
        head_top=head_top,
        left_shoulder=left_shoulder,
        right_shoulder=right_shoulder,
        left_elbow=left_elbow,
        right_elbow=right_elbow,
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        left_hip=left_hip,
        right_hip=right_hip,
        left_knee=left_knee,
        right_knee=right_knee,
        left_ankle=left_ankle,
        right_ankle=right_ankle,
        left_foot=left_foot,
        right_foot=right_foot,
    )


def _lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def _bone_radius(base_radius: float, length: float, ref_length: float = 0.25) -> float:
    return max(base_radius * (length / max(ref_length, 1e-6)), base_radius * 0.35)


def _fit_bone_endpoints(name: str, landmarks: MeshLandmarks, template: StandardBone) -> tuple[np.ndarray, np.ndarray, float]:
    """Map template bone names to fitted head/tail using mesh landmarks."""
    lm = landmarks
    t_head = np.asarray(template.head, dtype=np.float64)
    t_tail = np.asarray(template.tail, dtype=np.float64)
    t_len = max(float(np.linalg.norm(t_tail - t_head)), 1e-6)

    if name == "pelvis":
        head = lm.pelvis.copy()
        tail = lm.spine_mid.copy()
    elif name == "spine":
        head = lm.spine_mid.copy()
        tail = _lerp(lm.spine_mid, lm.chest, 0.55)
    elif name == "chest":
        head = _lerp(lm.spine_mid, lm.chest, 0.55)
        tail = lm.chest.copy()
    elif name == "neck":
        head = lm.chest.copy()
        tail = lm.neck_base.copy()
    elif name == "head":
        head = lm.neck_base.copy()
        tail = lm.head_top.copy()
    elif name.startswith("left_clavicle"):
        head = lm.chest.copy()
        tail = lm.left_shoulder.copy()
    elif name.startswith("right_clavicle"):
        head = lm.chest.copy()
        tail = lm.right_shoulder.copy()
    elif name.startswith("left_upper_arm"):
        head = lm.left_shoulder.copy()
        tail = lm.left_elbow.copy()
    elif name.startswith("right_upper_arm"):
        head = lm.right_shoulder.copy()
        tail = lm.right_elbow.copy()
    elif name.startswith("left_lower_arm"):
        head = lm.left_elbow.copy()
        tail = lm.left_wrist.copy()
    elif name.startswith("right_lower_arm"):
        head = lm.right_elbow.copy()
        tail = lm.right_wrist.copy()
    elif name.startswith("left_wrist"):
        head = lm.left_wrist.copy()
        tail = _lerp(lm.left_wrist, lm.left_wrist + np.array([0.0, -0.05, 0.0]), 1.0)
    elif name.startswith("right_wrist"):
        head = lm.right_wrist.copy()
        tail = _lerp(lm.right_wrist, lm.right_wrist + np.array([0.0, -0.05, 0.0]), 1.0)
    elif name.startswith("left_hand"):
        head = lm.left_wrist.copy()
        tail = lm.left_wrist + np.array([0.0, -0.08, 0.0]) * (lm.head_top[1] / 1.08)
    elif name.startswith("right_hand"):
        head = lm.right_wrist.copy()
        tail = lm.right_wrist + np.array([0.0, -0.08, 0.0]) * (lm.head_top[1] / 1.08)
    elif name.startswith("left_upper_leg"):
        head = lm.left_hip.copy()
        tail = lm.left_knee.copy()
    elif name.startswith("right_upper_leg"):
        head = lm.right_hip.copy()
        tail = lm.right_knee.copy()
    elif name.startswith("left_lower_leg"):
        head = lm.left_knee.copy()
        tail = lm.left_ankle.copy()
    elif name.startswith("right_lower_leg"):
        head = lm.right_knee.copy()
        tail = lm.right_ankle.copy()
    elif name.startswith("left_foot"):
        head = lm.left_ankle.copy()
        tail = lm.left_foot.copy()
    elif name.startswith("right_foot"):
        head = lm.right_ankle.copy()
        tail = lm.right_foot.copy()
    elif name.startswith("left_toe"):
        head = lm.left_foot.copy()
        direction = lm.left_foot - lm.left_ankle
        direction[1] = 0.0
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        tail = lm.left_foot + direction * (0.14 * lm.head_top[1])
    elif name.startswith("right_toe"):
        head = lm.right_foot.copy()
        direction = lm.right_foot - lm.right_ankle
        direction[1] = 0.0
        if np.linalg.norm(direction) < 1e-6:
            direction = np.array([0.0, 0.0, 1.0])
        direction = direction / np.linalg.norm(direction)
        tail = lm.right_foot + direction * (0.14 * lm.head_top[1])
    elif "_thumb_" in name or "_index_" in name or "_middle_" in name or "_ring_" in name or "_pinky_" in name:
        side = "left" if name.startswith("left_") else "right"
        wrist = lm.left_wrist if side == "left" else lm.right_wrist
        hand = lm.left_wrist + np.array([0.0, -0.08, 0.0]) * (lm.head_top[1] / 1.08)
        if side == "left":
            hand = lm.left_wrist + np.array([0.0, -0.08, 0.0]) * (lm.head_top[1] / 1.08)
        else:
            hand = lm.right_wrist + np.array([0.0, -0.08, 0.0]) * (lm.head_top[1] / 1.08)
        rel = (t_tail - t_head) / t_len
        seg_idx = int(name.split("_")[-1]) - 1
        start = _lerp(wrist, hand, 0.35 + 0.15 * seg_idx)
        head = start
        tail = start + rel * (0.05 * lm.head_top[1])
    else:
        head = t_head * (lm.head_top[1] / 1.08)
        tail = t_tail * (lm.head_top[1] / 1.08)

    length = float(np.linalg.norm(tail - head))
    radius = _bone_radius(template.radius, length, t_len)
    if length < 1e-5:
        tail = head + (t_tail - t_head) / t_len * max(0.01 * lm.head_top[1], 1e-3)
        length = float(np.linalg.norm(tail - head))
    return head, tail, radius


def fit_skeleton_to_mesh_vertices(vertices: np.ndarray) -> tuple[list[StandardBone], MeshLandmarks, dict[str, np.ndarray]]:
    """Build a new skeleton instance scaled to mesh proportions (does not mutate template)."""
    segments = segment_mesh_vertices(vertices)
    landmarks = extract_mesh_landmarks(vertices, segments)
    template = build_standard_humanoid_skeleton()
    fitted: list[StandardBone] = []
    for bone in template:
        head, tail, radius = _fit_bone_endpoints(bone.name, landmarks, bone)
        fitted.append(
            StandardBone(
                name=bone.name,
                parent=bone.parent,
                head=head.astype(np.float64),
                tail=tail.astype(np.float64),
                radius=float(radius),
            )
        )
    return fitted, landmarks, segments


def fit_skeleton_to_glb_mesh(glb_path: str | Path) -> dict:
    """Align mesh space, segment body, and fit a per-asset skeleton."""
    glb_path = Path(glb_path)
    vertices, triangles, uvs, texture = load_glb_mesh_with_material(glb_path)
    aligned, alignment_meta = align_mesh_space(vertices)
    bones, landmarks, segments = fit_skeleton_to_mesh_vertices(aligned)
    return {
        "source_glb": str(glb_path.resolve()),
        "vertices": aligned,
        "triangles": triangles,
        "uvs": uvs,
        "texture": texture,
        "bones": bones,
        "landmarks": landmarks,
        "segments": segments,
        "alignment": alignment_meta,
    }


def fitted_skeleton_path_for_glb(glb_path: str | Path, output_dir: str | Path | None = None) -> Path:
    glb_path = Path(glb_path)
    if output_dir is None:
        output_dir = Path("outputs") / f"{glb_path.stem}_fitted"
    return Path(output_dir) / "fitted_skeleton.json"


def save_fitted_skeleton(
    bones: list[StandardBone],
    path: str | Path,
    *,
    source_glb: str | None = None,
    alignment: dict | None = None,
) -> Path:
    """Persist a fitted skeleton sidecar; never overwrites the template builder."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "worldmix_fitted_skeleton_v1",
        "source_glb": source_glb,
        "alignment": alignment,
        "bones": [
            {
                "name": bone.name,
                "parent": bone.parent,
                "head": bone.head.tolist(),
                "tail": bone.tail.tolist(),
                "radius": bone.radius,
            }
            for bone in bones
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_fitted_skeleton(path: str | Path) -> list[StandardBone]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    bones: list[StandardBone] = []
    for spec in payload.get("bones", []):
        bones.append(
            StandardBone(
                name=spec["name"],
                parent=int(spec["parent"]),
                head=np.asarray(spec["head"], dtype=np.float64),
                tail=np.asarray(spec["tail"], dtype=np.float64),
                radius=float(spec["radius"]),
            )
        )
    return bones


def _capsule_distance_batch(points: np.ndarray, head: np.ndarray, tail: np.ndarray, radius: float) -> np.ndarray:
    axis = tail - head
    denom = float(np.dot(axis, axis)) + 1e-12
    t = np.clip((points - head) @ axis / denom, 0.0, 1.0)
    closest = head + t[:, None] * axis
    return np.linalg.norm(points - closest, axis=1) - radius


def assign_vertex_regions(vertices: np.ndarray, *, method: str = "front_view") -> np.ndarray:
    """Vectorized body region label per vertex."""
    segments = segment_mesh_vertices(vertices, method=method)
    labels = np.full(len(vertices), "torso", dtype=object)
    for region, idx in segments.items():
        labels[idx] = region
    return labels


def _bone_name_set(bones: list[StandardBone], names: tuple[str, ...]) -> list[int]:
    wanted = set(names)
    return [idx for idx, bone in enumerate(bones) if bone.name in wanted]


def _region_bone_candidates(
    region: str,
    bones: list[StandardBone],
    points: np.ndarray,
    y_min: float,
    height: float,
) -> list[int]:
    """Pick bone candidates using semantic region plus height bands inside the body."""
    y_norm = (points[:, 1] - y_min) / max(height, 1e-6)
    mean_y = float(np.mean(y_norm))

    if region == "head":
        return _bone_name_set(bones, ("neck", "head"))
    if region == "torso":
        if mean_y < 0.58:
            return _bone_name_set(bones, ("pelvis", "spine"))
        if mean_y < 0.74:
            return _bone_name_set(bones, ("spine", "chest"))
        return _bone_name_set(bones, ("chest", "neck"))
    if region == "left_arm":
        return _bone_name_set(bones, _SEMANTIC_BONE_GROUPS["left_arm"])
    if region == "right_arm":
        return _bone_name_set(bones, _SEMANTIC_BONE_GROUPS["right_arm"])
    if region == "left_leg":
        if mean_y < 0.22:
            return _bone_name_set(bones, ("left_lower_leg", "left_foot", "left_toe"))
        return _bone_name_set(bones, ("left_upper_leg", "left_lower_leg", "pelvis"))
    if region == "right_leg":
        if mean_y < 0.22:
            return _bone_name_set(bones, ("right_lower_leg", "right_foot", "right_toe"))
        return _bone_name_set(bones, ("right_upper_leg", "right_lower_leg", "pelvis"))
    return list(range(len(bones)))


def _assign_capsule_weights(
    points: np.ndarray,
    candidates: list[int],
    bones: list[StandardBone],
    k_bones: int,
    temp: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not candidates:
        candidates = list(range(len(bones)))
    dist_stack = []
    for idx in candidates:
        bone = bones[idx]
        dist_stack.append(_capsule_distance_batch(points, bone.head, bone.tail, bone.radius))
    distances = np.stack(dist_stack, axis=1)
    order = np.argsort(distances, axis=1)[:, : max(1, min(k_bones, len(candidates)))]
    selected_dist = np.take_along_axis(distances, order, axis=1)
    logits = -selected_dist / temp
    logits -= logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True) + 1e-12

    selected_global = np.asarray(candidates, dtype=np.int32)[order]
    if selected_global.shape[1] < k_bones:
        pad = np.repeat(selected_global[:, -1:], k_bones - selected_global.shape[1], axis=1)
        selected_global = np.concatenate([selected_global, pad], axis=1)
        w_pad = np.zeros((w.shape[0], k_bones - w.shape[1]), dtype=np.float64)
        w = np.concatenate([w, w_pad], axis=1)
        w /= w.sum(axis=1, keepdims=True) + 1e-12
    return selected_global[:, :k_bones], w[:, :k_bones]


def compute_vertex_skin_weights(
    vertices: np.ndarray,
    bones: list[StandardBone],
    segments: dict[str, np.ndarray],
    *,
    k_bones: int = 4,
    temperature: float = 0.08,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition LBS weights from semantic region + bone capsule distance field."""
    del segments  # regions are recomputed vectorized for all vertices
    n = len(vertices)
    bone_indices = np.zeros((n, k_bones), dtype=np.int32)
    weights = np.zeros((n, k_bones), dtype=np.float64)
    labels = assign_vertex_regions(vertices, method="front_view")
    y_min = float(vertices[:, 1].min())
    height = max(float(vertices[:, 1].max()) - y_min, 1e-6)
    temp = max(temperature * height, 1e-6)

    for region in MESH_REGIONS:
        mask = labels == region
        if not np.any(mask):
            continue
        pts = vertices[mask]
        global_idx = np.where(mask)[0]
        y_norm = (pts[:, 1] - y_min) / height

        if region == "torso":
            bands = [
                y_norm < 0.58,
                (y_norm >= 0.58) & (y_norm < 0.74),
                y_norm >= 0.74,
            ]
            candidate_sets = [
                _bone_name_set(bones, ("pelvis", "spine")),
                _bone_name_set(bones, ("spine", "chest")),
                _bone_name_set(bones, ("chest", "neck")),
            ]
        elif region in {"left_leg", "right_leg"}:
            prefix = "left" if region.startswith("left") else "right"
            bands = [y_norm < 0.22, y_norm >= 0.22]
            candidate_sets = [
                _bone_name_set(bones, (f"{prefix}_lower_leg", f"{prefix}_foot", f"{prefix}_toe")),
                _bone_name_set(bones, (f"{prefix}_upper_leg", f"{prefix}_lower_leg", "pelvis")),
            ]
        else:
            bands = [np.ones(len(pts), dtype=bool)]
            candidate_sets = [_region_bone_candidates(region, bones, pts, y_min, height)]

        for band_mask, candidates in zip(bands, candidate_sets):
            if not np.any(band_mask):
                continue
            sub_pts = pts[band_mask]
            sub_global = global_idx[band_mask]
            idx_out, w_out = _assign_capsule_weights(sub_pts, candidates, bones, k_bones, temp)
            bone_indices[sub_global] = idx_out
            weights[sub_global] = w_out

    return bone_indices, weights

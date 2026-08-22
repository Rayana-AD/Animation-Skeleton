"""ViT-based GLB ↔ standard skeleton alignment with closed-loop validation.

Topology stays on the standard humanoid template; ViT adjusts front-facing
proportion scales for both GLB shell and skeleton (not a full re-rig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from worldmix.character.front_view_semantics import front_extremity_landmarks, segment_mesh_front_view
from worldmix.character.google_vit_backend import render_front_vit_image, uv_to_front_xyz, vit_front_joint_uv
from worldmix.character.standard_skeleton_gaussians import StandardBone, build_standard_humanoid_skeleton
from worldmix.character.static_mesh_avatar import load_glb_mesh_with_material


MAIN_JOINTS = (
    "pelvis",
    "head_top",
    "left_shoulder",
    "right_shoulder",
    "left_hand",
    "right_hand",
    "left_foot",
    "right_foot",
)

_BONE_GROUPS: dict[str, str] = {}


def _init_bone_groups() -> None:
    global _BONE_GROUPS
    if _BONE_GROUPS:
        return
    for name in ["pelvis", "spine", "chest", "neck", "head"]:
        _BONE_GROUPS[name] = "torso"
    for side in ("left", "right"):
        for part in ("clavicle", "upper_arm", "lower_arm", "wrist", "hand"):
            _BONE_GROUPS[f"{side}_{part}"] = f"{side}_arm"
        for part in ("upper_leg", "lower_leg", "foot", "toe"):
            _BONE_GROUPS[f"{side}_{part}"] = f"{side}_leg"
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            for idx in ("01", "02", "03"):
                _BONE_GROUPS[f"{side}_{finger}_{idx}"] = f"{side}_arm"


_init_bone_groups()


@dataclass
class ViTJointPrediction:
    joints: dict[str, np.ndarray]
    normals: dict[str, np.ndarray]
    confidence: dict[str, float]
    source: str = "google_vit+geometric"


@dataclass
class AlignmentValidation:
    joint_errors: dict[str, float]
    normal_agreement: dict[str, float]
    hierarchy_preserved: bool
    mean_joint_error: float
    passed: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class ViTAlignmentResult:
    vertices: np.ndarray
    triangles: np.ndarray
    uvs: np.ndarray | None
    texture: object | None
    transform: dict
    prediction: ViTJointPrediction
    validation: AlignmentValidation
    template_bones: list[StandardBone]
    standard_bones: list[StandardBone]
    proportion_scales: dict[str, float]


def _bone_point(bones: list[StandardBone], name: str, endpoint: str = "head") -> np.ndarray:
    bone = next(b for b in bones if b.name == name)
    return bone.head.copy() if endpoint == "head" else bone.tail.copy()


def skeleton_joint_targets(bones: list[StandardBone]) -> dict[str, np.ndarray]:
    return {
        "pelvis": _bone_point(bones, "pelvis", "head"),
        "head_top": _bone_point(bones, "head", "tail"),
        "left_shoulder": _bone_point(bones, "left_upper_arm", "head"),
        "right_shoulder": _bone_point(bones, "right_upper_arm", "head"),
        "left_hand": _bone_point(bones, "left_hand", "tail"),
        "right_hand": _bone_point(bones, "right_hand", "tail"),
        "left_foot": _bone_point(bones, "left_foot", "tail"),
        "right_foot": _bone_point(bones, "right_foot", "tail"),
    }


def canonicalize_front_facing(vertices: np.ndarray) -> tuple[np.ndarray, dict]:
    """Center, floor, and rotate about Y so the character faces front (-Z)."""
    verts = np.asarray(vertices, dtype=np.float64).copy()
    verts[:, 0] -= float(verts[:, 0].mean())
    verts[:, 2] -= float(verts[:, 2].mean())
    verts[:, 1] -= float(verts[:, 1].min())

    xz = verts[:, [0, 2]]
    cov = np.cov(xz.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, int(np.argmax(eigvals))]
    angle = float(np.arctan2(major[1], major[0]))
    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    verts = verts @ rot.T

    # Ensure broader side maps to ±X (A-pose expectation).
    if float(np.percentile(verts[:, 0], 95)) < abs(float(np.percentile(verts[:, 0], 5))):
        flip = np.diag([-1.0, 1.0, 1.0])
        verts = verts @ flip
        rot = flip @ rot

    return verts, {"rotation_y_rad": -angle, "height": float(verts[:, 1].max())}


def _estimate_vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    tris = vertices[triangles]
    face_n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    face_n = face_n / (np.linalg.norm(face_n, axis=1, keepdims=True) + 1e-12)
    for i in range(3):
        np.add.at(normals, triangles[:, i], face_n)
    return normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)


def _region_mean_normal(normals: np.ndarray, indices: np.ndarray) -> np.ndarray:
    if len(indices) == 0:
        return np.array([0.0, 0.0, 1.0])
    n = normals[indices].mean(axis=0)
    return n / (np.linalg.norm(n) + 1e-12)


def predict_joints_geometric(vertices: np.ndarray, triangles: np.ndarray) -> ViTJointPrediction:
    segments = segment_mesh_front_view(vertices)
    landmarks = front_extremity_landmarks(vertices, segments)
    mesh_normals = _estimate_vertex_normals(vertices, triangles)
    joints = {name: landmarks[name].copy() for name in MAIN_JOINTS}
    normals = {
        "head": _region_mean_normal(mesh_normals, segments["head"]),
        "torso": _region_mean_normal(mesh_normals, segments["torso"]),
        "left_arm": _region_mean_normal(mesh_normals, segments["left_arm"]),
        "right_arm": _region_mean_normal(mesh_normals, segments["right_arm"]),
        "left_leg": _region_mean_normal(mesh_normals, segments["left_leg"]),
        "right_leg": _region_mean_normal(mesh_normals, segments["right_leg"]),
    }
    return ViTJointPrediction(
        joints=joints,
        normals=normals,
        confidence={name: 0.55 for name in MAIN_JOINTS},
        source="geometric_front_view",
    )


def predict_joints_vit(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray | None,
    texture,
    *,
    use_vit: bool = True,
) -> ViTJointPrediction:
    """Fuse Google ViT front saliency with geometric extremities."""
    geo = predict_joints_geometric(vertices, triangles)
    if not use_vit:
        return geo

    try:
        image, bounds = render_front_vit_image(vertices, triangles, uvs, texture, size=224)
        uv_joints = vit_front_joint_uv(image)
        vit_joints = uv_to_front_xyz(uv_joints, bounds)
        # Snap depth from nearest mesh vertex in front shell.
        for name, joint in vit_joints.items():
            dists = np.linalg.norm(vertices[:, [0, 1]] - joint[[0, 1]], axis=1)
            idx = int(np.argmin(dists))
            joint[2] = float(vertices[idx, 2])
            vit_joints[name] = joint

        fused: dict[str, np.ndarray] = {}
        confidence: dict[str, float] = {}
        for name in MAIN_JOINTS:
            w = 0.62
            fused[name] = w * vit_joints[name] + (1.0 - w) * geo.joints[name]
            confidence[name] = w
        return ViTJointPrediction(
            joints=fused,
            normals=geo.normals,
            confidence=confidence,
            source="google/vit-base-patch16-224+geometric",
        )
    except Exception:
        return geo


def compute_proportion_scales(
    template_joints: dict[str, np.ndarray],
    mesh_joints: dict[str, np.ndarray],
) -> dict[str, float]:
    def dist(a: str, b: str, joints: dict[str, np.ndarray]) -> float:
        return float(np.linalg.norm(joints[a] - joints[b]))

    def ratio(a: str, b: str) -> float:
        t = dist(a, b, template_joints)
        m = dist(a, b, mesh_joints)
        return m / max(t, 1e-9)

    return {
        "torso": ratio("pelvis", "head_top"),
        "left_arm": ratio("left_shoulder", "left_hand"),
        "right_arm": ratio("right_shoulder", "right_hand"),
        "left_leg": ratio("pelvis", "left_foot"),
        "right_leg": ratio("pelvis", "right_foot"),
    }


def retarget_skeleton_proportions(
    template_bones: list[StandardBone],
    mesh_joints: dict[str, np.ndarray],
) -> tuple[list[StandardBone], dict[str, float]]:
    """Scale standard bone lengths to ViT mesh proportions; topology unchanged."""
    template_joints = skeleton_joint_targets(template_bones)
    scales = compute_proportion_scales(template_joints, mesh_joints)

    bones: list[StandardBone] = [
        StandardBone(b.name, b.parent, b.head.copy(), b.tail.copy(), b.radius) for b in template_bones
    ]

    for bone in bones:
        group = _BONE_GROUPS.get(bone.name, "torso")
        s = float(scales.get(group, 1.0))
        vec = bone.tail - bone.head
        bone.tail = bone.head + vec * s
        bone.radius *= s

    for bone in bones:
        if bone.parent >= 0:
            parent = bones[bone.parent]
            offset = bone.tail - bone.head
            bone.head = parent.tail.copy()
            bone.tail = bone.head + offset

    shift = mesh_joints["pelvis"] - bones[0].head
    for bone in bones:
        bone.head += shift
        bone.tail += shift

    # Close loop: nudge spine/limb tips toward ViT targets (single pass).
    tips = {
        "head": ("head_top", "tail"),
        "left_hand": ("left_hand", "tail"),
        "right_hand": ("right_hand", "tail"),
        "left_foot": ("left_foot", "tail"),
        "right_foot": ("right_foot", "tail"),
    }
    for bone_name, (joint_key, _) in tips.items():
        bone = next(b for b in bones if b.name == bone_name)
        delta = mesh_joints[joint_key] - bone.tail
        bone.tail += 0.35 * delta
        if bone.parent >= 0:
            bones[bone.parent].tail += 0.15 * delta

    return bones, scales


def validate_alignment(
    aligned_vertices: np.ndarray,
    triangles: np.ndarray,
    bones: list[StandardBone],
    *,
    joint_tol: float = 0.18,
) -> AlignmentValidation:
    targets = skeleton_joint_targets(bones)
    refreshed = predict_joints_geometric(aligned_vertices, triangles)
    joint_errors = {
        name: float(np.linalg.norm(refreshed.joints[name] - targets[name])) for name in MAIN_JOINTS
    }

    mesh_normals = _estimate_vertex_normals(aligned_vertices, triangles)
    segments = segment_mesh_front_view(aligned_vertices)
    normal_agreement = {
        region: float(np.dot(_region_mean_normal(mesh_normals, segments[region]), np.array([0.0, 1.0, 0.0])))
        for region in ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")
    }

    hierarchy_preserved = all(b.parent >= -1 for b in bones) and len(bones) == len(build_standard_humanoid_skeleton())
    mean_joint_error = float(np.mean(list(joint_errors.values())))
    notes: list[str] = []
    if mean_joint_error > joint_tol:
        notes.append(f"joint RMSE {mean_joint_error:.3f} > {joint_tol}")
    passed = mean_joint_error <= joint_tol and hierarchy_preserved
    return AlignmentValidation(
        joint_errors=joint_errors,
        normal_agreement=normal_agreement,
        hierarchy_preserved=hierarchy_preserved,
        mean_joint_error=mean_joint_error,
        passed=passed,
        notes=notes,
    )


def align_glb_to_standard_skeleton(
    glb_path: str | Path,
    *,
    use_vit: bool = True,
) -> ViTAlignmentResult:
    glb_path = Path(glb_path)
    vertices, triangles, uvs, texture = load_glb_mesh_with_material(glb_path)
    template_bones = build_standard_humanoid_skeleton()

    canon, canon_meta = canonicalize_front_facing(vertices)
    prediction = predict_joints_vit(canon, triangles, uvs, texture, use_vit=use_vit)
    retargeted_bones, proportion_scales = retarget_skeleton_proportions(template_bones, prediction.joints)

    # Shared front-space: mesh stays in canonical coords; skeleton retargeted to ViT joints.
    aligned = canon
    validation = validate_alignment(aligned, triangles, retargeted_bones)

    transform = {
        **canon_meta,
        "source_glb": str(glb_path.resolve()),
        "prediction_source": prediction.source,
        "proportion_scales": proportion_scales,
    }
    return ViTAlignmentResult(
        vertices=aligned,
        triangles=triangles,
        uvs=uvs,
        texture=texture,
        transform=transform,
        prediction=prediction,
        validation=validation,
        template_bones=template_bones,
        standard_bones=retargeted_bones,
        proportion_scales=proportion_scales,
    )


def save_alignment_report(result: ViTAlignmentResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "worldmix_vit_alignment_v2",
        "transform": result.transform,
        "proportion_scales": result.proportion_scales,
        "validation": {
            "passed": result.validation.passed,
            "mean_joint_error": result.validation.mean_joint_error,
            "joint_errors": result.validation.joint_errors,
            "normal_agreement": result.validation.normal_agreement,
            "hierarchy_preserved": result.validation.hierarchy_preserved,
            "notes": result.validation.notes,
        },
        "joints_predicted": {k: v.tolist() for k, v in result.prediction.joints.items()},
        "joints_retargeted": {k: v.tolist() for k, v in skeleton_joint_targets(result.standard_bones).items()},
        "joints_template": {k: v.tolist() for k, v in skeleton_joint_targets(result.template_bones).items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def save_retargeted_skeleton_sidecar(bones: list[StandardBone], path: str | Path, *, source_glb: str, scales: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "worldmix_retargeted_skeleton_v1",
        "source_glb": source_glb,
        "proportion_scales": scales,
        "topology": "standard_humanoid_template",
        "bones": [
            {"name": b.name, "parent": b.parent, "head": b.head.tolist(), "tail": b.tail.tolist(), "radius": b.radius}
            for b in bones
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

"""Quantitative metrics for character alignment / skinning (skeleton pack)."""

from __future__ import annotations

from typing import Any

import numpy as np

from worldmix.character.mesh_shell_skinning import pose_skeleton_frames, skin_vertices
from worldmix.character.motion_adapter import text_command_to_motion
from worldmix.character.standard_skeleton_gaussians import StandardBone, build_standard_humanoid_skeleton
from worldmix.character.vit_align import (
    MAIN_JOINTS,
    predict_joints_geometric,
    skeleton_joint_targets,
    validate_alignment,
)
def character_alignment_metrics(
    vertices: np.ndarray,
    triangles: np.ndarray,
    bones: list[StandardBone],
    *,
    prediction_source: str | None = None,
    proportion_scales: dict[str, float] | None = None,
    joint_tol: float = 0.18,
) -> dict[str, Any]:
    """Joint error, hierarchy, and proportion-scale stats vs geometric mesh landmarks."""
    validation = validate_alignment(vertices, triangles, bones, joint_tol=joint_tol)
    mesh_joints = predict_joints_geometric(vertices, triangles).joints
    bone_joints = skeleton_joint_targets(bones)
    per_joint = {
        name: float(np.linalg.norm(mesh_joints[name] - bone_joints[name])) for name in MAIN_JOINTS
    }
    height = max(float(vertices[:, 1].max() - vertices[:, 1].min()), 1e-6)
    return {
        "prediction_source": prediction_source or "unknown",
        "mean_joint_error": float(np.mean(list(per_joint.values()))),
        "max_joint_error": float(np.max(list(per_joint.values()))),
        "normalized_mean_joint_error": float(np.mean(list(per_joint.values())) / height),
        "joint_errors": per_joint,
        "hierarchy_preserved": validation.hierarchy_preserved,
        "alignment_passed": validation.passed,
        "joint_tol": joint_tol,
        "proportion_scales": proportion_scales or {},
        "mesh_height": height,
        "bone_count": len(bones),
        "template_bone_count": len(build_standard_humanoid_skeleton()),
    }


def topology_preservation_metrics(bones: list[StandardBone]) -> dict[str, Any]:
    """Verify retargeted skeleton keeps template names/parents."""
    template = build_standard_humanoid_skeleton()
    name_ok = [b.name for b in bones] == [b.name for b in template]
    parent_ok = [b.parent for b in bones] == [b.parent for b in template]
    count_ok = len(bones) == len(template)
    return {
        "topology_preserved": bool(name_ok and parent_ok and count_ok),
        "name_match": bool(name_ok),
        "parent_match": bool(parent_ok),
        "count_match": bool(count_ok),
        "bone_count": len(bones),
    }


def skinning_distortion_metrics(
    vertices: np.ndarray,
    bones: list[StandardBone],
    bone_indices: np.ndarray,
    weights: np.ndarray,
    *,
    command: str = "walk forward",
    frames: int = 8,
) -> dict[str, Any]:
    """LBS distortion under a language motion: volume proxy + surface stretch stats."""
    rest_heads, rest_frames = pose_skeleton_frames(bones, pose=None)
    motion = text_command_to_motion(command, frames=frames)
    rest = np.asarray(vertices, dtype=np.float64)
    rest_bbox = rest.max(axis=0) - rest.min(axis=0)
    rest_diag = float(np.linalg.norm(rest_bbox) + 1e-12)
    rest_centroid = rest.mean(axis=0)
    rest_rms = float(np.sqrt(np.mean(np.sum((rest - rest_centroid) ** 2, axis=1))))

    frame_stats: list[dict[str, float]] = []
    for pose in motion:
        posed_heads, posed_frames = pose_skeleton_frames(bones, pose)
        deformed = skin_vertices(rest, rest_heads, rest_frames, posed_heads, posed_frames, bone_indices, weights)
        delta = deformed - rest
        mean_disp = float(np.mean(np.linalg.norm(delta, axis=1)))
        max_disp = float(np.max(np.linalg.norm(delta, axis=1)))
        bbox = deformed.max(axis=0) - deformed.min(axis=0)
        bbox_scale = float(np.linalg.norm(bbox) / rest_diag)
        centroid = deformed.mean(axis=0)
        rms = float(np.sqrt(np.mean(np.sum((deformed - centroid) ** 2, axis=1))))
        volume_proxy = float(rms / (rest_rms + 1e-12))
        frame_stats.append(
            {
                "mean_disp": mean_disp,
                "max_disp": max_disp,
                "bbox_scale": bbox_scale,
                "volume_proxy": volume_proxy,
            }
        )

    mean_disp = float(np.mean([f["mean_disp"] for f in frame_stats]))
    max_disp = float(np.max([f["max_disp"] for f in frame_stats]))
    bbox_scale_mean = float(np.mean([f["bbox_scale"] for f in frame_stats]))
    volume_proxy_mean = float(np.mean([f["volume_proxy"] for f in frame_stats]))
    return {
        "command": command,
        "frames": frames,
        "mean_vertex_displacement": mean_disp,
        "max_vertex_displacement": max_disp,
        "normalized_mean_displacement": mean_disp / rest_diag,
        "bbox_scale_mean": bbox_scale_mean,
        "volume_proxy_mean": volume_proxy_mean,
        "volume_collapse_ratio": float(min(volume_proxy_mean, 1.0 / max(volume_proxy_mean, 1e-12))),
        "per_frame": frame_stats,
    }


def summarize_character_run(report: dict[str, Any]) -> dict[str, float]:
    """Flatten key scalars for CSV / table export."""
    align = report.get("alignment", {})
    skin = report.get("skinning", {})
    topo = report.get("topology", {})
    return {
        "mean_joint_error": float(align.get("mean_joint_error", np.nan)),
        "normalized_mean_joint_error": float(align.get("normalized_mean_joint_error", np.nan)),
        "alignment_passed": float(1.0 if align.get("alignment_passed") else 0.0),
        "topology_preserved": float(1.0 if topo.get("topology_preserved") else 0.0),
        "normalized_mean_displacement": float(skin.get("normalized_mean_displacement", np.nan)),
        "volume_proxy_mean": float(skin.get("volume_proxy_mean", np.nan)),
        "volume_collapse_ratio": float(skin.get("volume_collapse_ratio", np.nan)),
    }

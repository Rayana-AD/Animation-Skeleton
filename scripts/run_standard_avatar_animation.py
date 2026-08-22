#!/usr/bin/env python3
"""Standard skeleton + Gaussian Splat animation demo without Godot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from worldmix.character.standard_skeleton_gaussians import (
    build_rigged_gaussian_avatar,
    load_motion_json,
    pose_avatar,
    save_avatar_animation,
    walk_pose,
    write_sample_motion_json,
)
from worldmix.io.ply import export_mesh_ply


def _export_gaussian_points(avatar, output_dir: Path, frames: int, motion_sequence=None) -> None:
    """Export rest and first/last animated Gaussian centers for inspection."""
    for name, frame_idx in [("rest", None), ("frame_000", 0), ("frame_mid", frames // 2)]:
        if frame_idx is None:
            pose = {}
        elif motion_sequence:
            pose = motion_sequence[min(frame_idx, len(motion_sequence) - 1)]
        else:
            pose = walk_pose(frame_idx, frames)
        gaussians, _ = pose_avatar(avatar, pose)
        pts = np.stack([g.mean for g in gaussians])
        export_mesh_ply(pts, np.empty((0, 3), dtype=np.int32), output_dir / f"{name}_gaussians.ply")


def _export_avatar_json(avatar, output_dir: Path) -> None:
    bones = [
        {
            "name": b.name,
            "parent": b.parent,
            "head": b.head.tolist(),
            "tail": b.tail.tolist(),
            "radius": b.radius,
        }
        for b in avatar.bones
    ]
    gaussians = [
        {
            "bone_id": g.bone_id,
            "local_mean": g.local_mean.tolist(),
            "local_scale": g.local_scale.tolist(),
        }
        for g in avatar.gaussians
    ]
    payload = {
        "format": "worldmix_standard_skeleton_gaussians_v1",
        "bones": bones,
        "gaussians": gaussians,
    }
    (output_dir / "standard_avatar.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="World-Mix standard skeleton Gaussian animation")
    parser.add_argument("--points-per-bone", type=int, default=28)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/standard_avatar")
    parser.add_argument("--show-ellipsoids", action="store_true")
    parser.add_argument("--no-skin-proxy", action="store_true")
    parser.add_argument("--motion-json", default=None, help="Retargeted motion JSON to drive the skeleton")
    parser.add_argument("--write-sample-motion", default=None, help="Write sample motion JSON and exit")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.write_sample_motion:
        write_sample_motion_json(args.write_sample_motion, frames=args.frames)
        print(f"Sample motion JSON written to: {Path(args.write_sample_motion).resolve()}")
        return

    avatar = build_rigged_gaussian_avatar(points_per_bone=args.points_per_bone, seed=args.seed)
    motion_sequence = load_motion_json(args.motion_json) if args.motion_json else None
    frame_count = len(motion_sequence) if motion_sequence else args.frames
    gif_path = save_avatar_animation(
        avatar,
        output_dir=output_dir,
        frames=frame_count,
        fps=args.fps,
        show_ellipsoids=args.show_ellipsoids,
        show_skin_proxy=not args.no_skin_proxy,
        motion_sequence=motion_sequence,
    )
    _export_gaussian_points(avatar, output_dir, frame_count, motion_sequence)
    _export_avatar_json(avatar, output_dir)

    rest_gaussians, _ = pose_avatar(avatar, {})
    rest_pts = np.stack([g.mean for g in rest_gaussians])
    frame_motions = []
    for frame_idx in range(frame_count):
        pose = motion_sequence[frame_idx] if motion_sequence else walk_pose(frame_idx, frame_count)
        posed_gaussians, _ = pose_avatar(avatar, pose)
        posed_pts = np.stack([g.mean for g in posed_gaussians])
        frame_motions.append(float(np.mean(np.linalg.norm(posed_pts - rest_pts, axis=1))))
    mean_motion = float(np.mean(frame_motions))
    peak_motion = float(np.max(frame_motions))

    print("=== Standard Skeleton Gaussian Avatar ===")
    print(f"骨骼数: {len(avatar.bones)}")
    print(f"高斯数: {len(avatar.gaussians)}")
    print(f"动画帧数: {frame_count}")
    print(f"动作来源: {'motion-json' if motion_sequence else 'procedural-walk'}")
    print(f"皮肤代理: {'off' if args.no_skin_proxy else 'on'}")
    print(f"协方差椭球: {'on' if args.show_ellipsoids else 'off'}")
    print(f"平均帧位移: {mean_motion:.4f}")
    print(f"峰值帧位移: {peak_motion:.4f}")
    print(f"GIF: {gif_path.resolve()}")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

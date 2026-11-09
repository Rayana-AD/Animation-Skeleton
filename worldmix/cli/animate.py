"""CLI: GLB textured shell on standard skeleton → GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldmix.character.glb_standard_overlay import (
    build_glb_shell_on_standard_skeleton,
    save_alignment_report,
    save_dual_view_shell_animation,
    save_front_shell_animation,
    save_rest_dual_preview,
    save_rest_front_preview,
    verify_motion_path_matches_standard,
)
from worldmix.character.motion_adapter import text_command_to_motion
from worldmix.character.standard_skeleton_gaussians import load_motion_json
from worldmix.character.vit_align import save_retargeted_skeleton_sidecar


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GLB appearance shell + standard skeleton + pose_avatar (no per-GLB skeleton morph)"
    )
    parser.add_argument("glb", help="Static GLB mesh")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-vit", action="store_true", help="Disable Google ViT; geometric only")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--command", default="walk forward")
    parser.add_argument("--motion", default=None, help="Motion JSON (same format as language_avatar)")
    parser.add_argument("--panel-width", type=int, default=480)
    parser.add_argument("--panel-height", type=int, default=720)
    parser.add_argument("--max-faces", type=int, default=None, help="Triangle budget for preview raster (e.g. 48000)")
    parser.add_argument(
        "--batched-preview",
        action="store_true",
        help="Outer-shell + depth-sorted triangle batches (dense GLB homepage quality)",
    )
    parser.add_argument("--raster-batch", type=int, default=8192, help="Triangles per raster batch")
    parser.add_argument("--outer-keep", type=float, default=0.75, help="Front visible shell fraction before face cap")
    parser.add_argument("--front-only", action="store_true", help="Single front panel (faster preview)")
    args = parser.parse_args()

    glb_path = Path(args.glb)
    output = Path(args.output) if args.output else Path("outputs") / f"{glb_path.stem}_standard_shell"
    output.mkdir(parents=True, exist_ok=True)

    shell = build_glb_shell_on_standard_skeleton(glb_path, use_vit=not args.no_vit)
    save_alignment_report(shell["alignment"], output / "vit_alignment_report.json")
    save_retargeted_skeleton_sidecar(
        shell["bones"],
        output / "retargeted_skeleton.json",
        source_glb=str(glb_path.resolve()),
        scales=shell["alignment"].proportion_scales,
    )

    motion_sequence = load_motion_json(args.motion) if args.motion else text_command_to_motion(args.command, frames=args.frames)

    check = verify_motion_path_matches_standard(shell, motion_sequence)
    preview_kw = dict(
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        max_faces=args.max_faces,
        batched_raster=args.batched_preview,
        triangle_batch=args.raster_batch,
        outer_keep_fraction=args.outer_keep,
    )
    if args.front_only:
        rest_png = save_rest_front_preview(shell, output / "aligned_rest_shell_front.png", **preview_kw)
        gif_path = save_front_shell_animation(
            shell,
            output / "fitted_textured_shell_walk.gif",
            motion_sequence=motion_sequence,
            fps=args.fps,
            max_faces=args.max_faces,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )
    else:
        rest_png = save_rest_dual_preview(shell, output / "aligned_rest_shell_dual.png", **preview_kw)
        gif_path = save_dual_view_shell_animation(
            shell,
            output / "fitted_textured_shell_walk.gif",
            motion_sequence=motion_sequence,
            fps=args.fps,
            max_faces=args.max_faces,
            panel_width=args.panel_width,
            panel_height=args.panel_height,
        )

    val = shell["alignment"].validation
    print("=== GLB Standard-Skeleton Shell ===")
    print(f"glb: {glb_path.resolve()}")
    print(f"triangles: {len(shell['triangles'])} full mesh")
    if args.batched_preview:
        print(f"batched_preview: outer_keep={args.outer_keep} raster_batch={args.raster_batch}")
    if args.max_faces:
        print(f"preview_face_cap: {args.max_faces} (full mesh {len(shell['triangles'])})")
    print(f"view: {'front' if args.front_only else 'front+side'}")
    print(f"skeleton: retargeted standard template ({len(shell['bones'])} bones)")
    print(f"proportion_scales: {shell['alignment'].proportion_scales}")
    print(f"vit_source: {shell['alignment'].prediction.source}")
    print(f"alignment_passed: {val.passed} (mean joint err {val.mean_joint_error:.4f})")
    if val.notes:
        print(f"alignment_notes: {'; '.join(val.notes)}")
    print(f"motion_check: {check}")
    print(f"rest_preview: {rest_png.resolve()}")
    print(f"gif: {gif_path.resolve()}")


if __name__ == "__main__":
    main()

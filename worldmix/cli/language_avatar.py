"""CLI: language command → standard skeleton Gaussian avatar GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldmix.character.motion_adapter import select_action_template, text_command_to_motion, write_motion_json
from worldmix.character.standard_skeleton_gaussians import build_rigged_gaussian_avatar, save_avatar_animation


def main() -> None:
    parser = argparse.ArgumentParser(description="World-Mix language-command avatar smoke demo")
    parser.add_argument("command", nargs="?", default="walk forward", help="Text command, e.g. 'walk forward'")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--points-per-bone", type=int, default=16)
    parser.add_argument("--output", default="outputs/language_avatar")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    template = select_action_template(args.command)
    sequence = text_command_to_motion(args.command, frames=args.frames)
    motion_json = output / "language_motion.json"
    write_motion_json(sequence, motion_json, source=f"language:{args.command}", fps=args.fps)

    avatar = build_rigged_gaussian_avatar(points_per_bone=args.points_per_bone)
    gif = save_avatar_animation(
        avatar,
        output_dir=output,
        frames=len(sequence),
        fps=args.fps,
        motion_sequence=sequence,
        show_skin_proxy=True,
    )

    print("=== Language Avatar Smoke Demo ===")
    print(f"command: {args.command}")
    print(f"template: {template.name}")
    print(f"participating_bones: {', '.join(template.participating_bones)}")
    print(f"motion_json: {motion_json.resolve()}")
    print(f"gif: {gif.resolve()}")


if __name__ == "__main__":
    main()

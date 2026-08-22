#!/usr/bin/env python3
"""Convert BVH/Mixamo or AMASS motion into World-Mix motion JSON."""

from __future__ import annotations

import argparse
from pathlib import Path

from worldmix.character.motion_adapter import read_amass_motion, read_bvh_motion, write_motion_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert external motion to World-Mix motion JSON")
    parser.add_argument("input", help="Input .bvh or AMASS .npz")
    parser.add_argument("--output", default="outputs/motion/retargeted_motion.json")
    parser.add_argument("--format", choices=["auto", "bvh", "amass"], default="auto")
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    input_path = Path(args.input)
    fmt = args.format
    if fmt == "auto":
        fmt = "amass" if input_path.suffix.lower() == ".npz" else "bvh"

    if fmt == "amass":
        sequence = read_amass_motion(input_path, max_frames=args.max_frames, downsample=args.downsample)
    else:
        sequence = read_bvh_motion(input_path, max_frames=args.max_frames, downsample=args.downsample)

    write_motion_json(sequence, args.output, source=f"{fmt}:{input_path.name}", fps=args.fps)
    print(f"Converted {len(sequence)} frames from {input_path} -> {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()

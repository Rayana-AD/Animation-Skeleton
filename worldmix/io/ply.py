"""Minimal PLY export (no Godot / terrain dependencies)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def export_mesh_ply(
    vertices: np.ndarray,
    triangles: np.ndarray,
    path: str | Path,
) -> None:
    """Write ASCII PLY mesh."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(triangles)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for v in vertices:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in triangles:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

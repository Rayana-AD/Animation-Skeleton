"""Measure and select visible mesh shell layers by depth + color continuity."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from worldmix.character.mesh_shell import TexturedMeshShell
from worldmix.character.static_mesh_avatar import _sample_texture


def _view_depth(vertices: np.ndarray, triangles: np.ndarray, view: str) -> np.ndarray:
    axis = 2 if view == "front" else 0
    return vertices[triangles].mean(axis=1)[:, axis]


def _position_edge_faces(vertices: np.ndarray, triangles: np.ndarray) -> dict[tuple, list[int]]:
    pos_edge: dict[tuple, list[int]] = {}
    for fi, (a, b, c) in enumerate(triangles):
        for i, j in ((a, b), (b, c), (c, a)):
            pa, pb = tuple(np.round(vertices[i], 4)), tuple(np.round(vertices[j], 4))
            key = (pa, pb) if pa < pb else (pb, pa)
            pos_edge.setdefault(key, []).append(fi)
    return pos_edge


def edge_color_discontinuity(
    shell: "TexturedMeshShell",
    face_mask: np.ndarray,
    vertex_colors: np.ndarray,
) -> dict[str, float]:
    """Shared geometric edges with exactly two active faces: opposite-corner color jump."""
    diffs: list[float] = []
    pos_edge = _position_edge_faces(shell.vertices, shell.triangles)
    tris = shell.triangles
    for faces in pos_edge.values():
        active = [fi for fi in faces if face_mask[fi]]
        if len(active) != 2:
            continue
        f0, f1 = active
        e0, e1 = set(tris[f0]), set(tris[f1])
        shared = tuple(e0 & e1)
        if len(shared) != 2:
            continue
        a, b = shared
        c0 = (e0 - {a, b}).pop()
        c1 = (e1 - {a, b}).pop()
        diffs.append(float(np.linalg.norm(vertex_colors[c0] - vertex_colors[c1])))
    if not diffs:
        return {"edge_pairs": 0, "median": 0.0, "p95": 0.0, "mean": 0.0}
    arr = np.asarray(diffs)
    return {
        "edge_pairs": int(len(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "mean": float(arr.mean()),
    }


def select_visible_faces(
    shell: "TexturedMeshShell",
    *,
    view: str = "front",
    keep_fraction: float = 0.70,
    continuity_threshold: float = 0.12,
) -> tuple[np.ndarray, dict]:
    """Pick outer shell faces: depth front-layer + optional continuity prune at boundary.

    Tripo GLB exports stack two nearly-coincident shells. Keeping the frontmost
    ``keep_fraction`` of face centroids removes the inner layer without breaking
    per-vertex UV continuity. Faces still reference original vertex indices.
    """
    tris = shell.triangles
    n_faces = len(tris)
    centroid_depth = _view_depth(shell.vertices, tris, view)
    keep_fraction = float(np.clip(keep_fraction, 0.55, 0.95))
    threshold = float(np.quantile(centroid_depth, 1.0 - keep_fraction))
    mask = centroid_depth >= threshold

    vertex_colors = None
    if shell.uvs is not None and shell.texture is not None:
        vertex_colors = _sample_texture(shell.texture, shell.uvs)

    stats = {
        "view": view,
        "keep_fraction": keep_fraction,
        "depth_threshold": threshold,
        "faces_kept": int(mask.sum()),
        "faces_total": n_faces,
    }

    if vertex_colors is not None:
        stats["continuity_before"] = edge_color_discontinuity(shell, mask, vertex_colors)

        # Drop boundary faces that disagree strongly with kept neighbors across shared edges.
        pos_edge = _position_edge_faces(shell.vertices, tris)
        to_drop: set[int] = set()
        for faces in pos_edge.values():
            active = [fi for fi in faces if mask[fi]]
            if len(active) != 2:
                continue
            f0, f1 = active
            e0, e1 = set(tris[f0]), set(tris[f1])
            shared = tuple(e0 & e1)
            if len(shared) != 2:
                continue
            a, b = shared
            c0 = (e0 - {a, b}).pop()
            c1 = (e1 - {a, b}).pop()
            jump = float(np.linalg.norm(vertex_colors[c0] - vertex_colors[c1]))
            if jump <= continuity_threshold:
                continue
            # Prefer dropping the deeper (inner) face on high-discontinuity seams.
            deeper = f0 if centroid_depth[f0] < centroid_depth[f1] else f1
            to_drop.add(deeper)
        if to_drop:
            mask[list(to_drop)] = False
            stats["continuity_pruned"] = len(to_drop)
            stats["continuity_after"] = edge_color_discontinuity(shell, mask, vertex_colors)

    return mask, stats


def visible_triangles(shell: "TexturedMeshShell", *, view: str = "front", keep_fraction: float = 0.70) -> np.ndarray:
    mask, _ = select_visible_faces(shell, view=view, keep_fraction=keep_fraction)
    return shell.triangles[mask]

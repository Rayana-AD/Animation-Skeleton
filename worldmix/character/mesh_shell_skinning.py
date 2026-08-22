"""Textured GLB mesh shell skinning with fitted per-asset skeletons."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from worldmix.character.glb_skeleton_fit import (
    align_mesh_space,
    compute_vertex_skin_weights,
    fit_skeleton_to_glb_mesh,
    load_fitted_skeleton,
    save_fitted_skeleton,
    segment_mesh_vertices,
)
from worldmix.character.standard_skeleton_gaussians import (
    StandardBone,
    RiggedGaussianAvatar,
    _axis_frame,
    load_motion_json,
    pose_avatar,
    walk_pose,
)
from worldmix.character.static_mesh_avatar import _sample_texture, load_glb_mesh_with_material
from worldmix.character.mesh_rasterize import rasterize_textured_mesh, save_raster_frame


def world_to_plot_coords(points: np.ndarray) -> np.ndarray:
    """Map world coords (X, Y-up, Z) to matplotlib 3D axes (X, Z, Y)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        return np.array([pts[0], pts[2], pts[1]], dtype=np.float64)
    if pts.ndim == 2:
        return pts[:, [0, 2, 1]]
    if pts.ndim == 3:
        return pts[:, :, [0, 2, 1]]
    raise ValueError(f"Unsupported point array rank: {pts.ndim}")


def plot_bone_segment(ax, head: np.ndarray, tail: np.ndarray, **kwargs) -> None:
    """Draw a bone segment using the same Y-up -> plot mapping as scatter previews."""
    ax.plot([head[0], tail[0]], [head[2], tail[2]], [head[1], tail[1]], **kwargs)


def pose_skeleton_frames(
    bones: list[StandardBone],
    pose: dict[str, np.ndarray] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Evaluate posed bone heads and local frames (same hierarchy as pose_avatar)."""
    avatar = RiggedGaussianAvatar(bones=bones, gaussians=[])
    _, segments = pose_avatar(avatar, pose)
    heads = [head.copy() for head, _ in segments]
    frames: list[np.ndarray] = []
    for bone, (head, tail) in zip(bones, segments):
        frames.append(_axis_frame(head, tail))
    return heads, frames


def skin_vertices(
    vertices: np.ndarray,
    rest_heads: list[np.ndarray],
    rest_frames: list[np.ndarray],
    posed_heads: list[np.ndarray],
    posed_frames: list[np.ndarray],
    bone_indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Linear blend skinning in bone-local coordinates."""
    rest_heads_arr = np.asarray(rest_heads, dtype=np.float64)
    rest_frames_arr = np.asarray(rest_frames, dtype=np.float64)
    posed_heads_arr = np.asarray(posed_heads, dtype=np.float64)
    posed_frames_arr = np.asarray(posed_frames, dtype=np.float64)

    out = np.zeros_like(vertices, dtype=np.float64)
    for slot in range(bone_indices.shape[1]):
        bid = bone_indices[:, slot]
        w = weights[:, slot]
        rest_h = rest_heads_arr[bid]
        rest_f = rest_frames_arr[bid]
        pose_h = posed_heads_arr[bid]
        pose_f = posed_frames_arr[bid]
        local = np.einsum("nij,nj->ni", np.transpose(rest_f, (0, 2, 1)), vertices - rest_h)
        posed = pose_h + np.einsum("nij,nj->ni", pose_f, local)
        out += w[:, None] * posed
    return out


def skin_vertices_subset(
    vertices: np.ndarray,
    vertex_indices: np.ndarray,
    rest_heads: list[np.ndarray],
    rest_frames: list[np.ndarray],
    posed_heads: list[np.ndarray],
    posed_frames: list[np.ndarray],
    bone_indices: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Skin only selected vertices, return full vertex buffer with untouched entries copied."""
    subset = skin_vertices(
        vertices[vertex_indices],
        rest_heads,
        rest_frames,
        posed_heads,
        posed_frames,
        bone_indices[vertex_indices],
        weights[vertex_indices],
    )
    out = vertices.copy()
    out[vertex_indices] = subset
    return out


def sample_vertex_colors(
    vertices: np.ndarray,
    uvs: np.ndarray | None,
    texture: Image.Image | None,
) -> np.ndarray:
    if uvs is None or texture is None:
        return np.full((len(vertices), 3), [0.72, 0.22, 0.48], dtype=np.float64)
    return _sample_texture(texture, uvs).astype(np.float64)


def select_render_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    max_faces: int = 6000,
    seed: int = 42,
) -> np.ndarray:
    """Pick a dense, height-stratified triangle subset that stays visually connected."""
    triangles = np.asarray(triangles, dtype=np.int64)
    if len(triangles) <= max_faces:
        return triangles

    tri_verts = vertices[triangles]
    cross = np.cross(tri_verts[:, 1] - tri_verts[:, 0], tri_verts[:, 2] - tri_verts[:, 0])
    areas = np.linalg.norm(cross, axis=1)
    valid = areas > 1e-12
    triangles = triangles[valid]
    areas = areas[valid]
    centroids = tri_verts[valid].mean(axis=1)

    y = centroids[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    bins = np.linspace(y_min, y_max, 13)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    per_bin = max(1, max_faces // (len(bins) - 1))

    for i in range(len(bins) - 1):
        upper = bins[i + 1] if i < len(bins) - 2 else bins[i + 1] + 1e-9
        mask = (y >= bins[i]) & (y < upper)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        take = min(per_bin, len(idx))
        local_areas = areas[idx]
        probs = local_areas / local_areas.sum()
        pick = rng.choice(idx, size=take, replace=False, p=probs)
        chosen.extend(pick.tolist())

    if len(chosen) < max_faces:
        remaining = np.setdiff1d(np.arange(len(triangles)), np.asarray(chosen, dtype=np.int64), assume_unique=False)
        if len(remaining):
            fill = min(max_faces - len(chosen), len(remaining))
            probs = areas[remaining] / areas[remaining].sum()
            chosen.extend(rng.choice(remaining, size=fill, replace=False, p=probs).tolist())

    return triangles[np.asarray(chosen[:max_faces], dtype=np.int64)]


def face_colors_from_vertices(
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
) -> np.ndarray:
    return vertex_colors[triangles].mean(axis=1)


def plot_textured_mesh_frame(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
    bone_segments: list[tuple[np.ndarray, np.ndarray]] | None = None,
    *,
    title: str = "Textured Shell",
    view: str = "front",
    width: int = 768,
    height: int = 1024,
    uvs: np.ndarray | None = None,
    texture: Image.Image | None = None,
) -> plt.Figure:
    """Render the full mesh shell as a continuous surface (every triangle, no sampling)."""
    frame = rasterize_textured_mesh(
        vertices,
        triangles,
        uvs,
        texture,
        view=view,
        width=width,
        height=height,
        vertex_colors=vertex_colors,
    )

    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111)
    ax.imshow(frame)
    ax.axis("off")

    if bone_segments and view == "front":
        u_min, u_max = float(vertices[:, 0].min()), float(vertices[:, 0].max())
        v_min, v_max = float(vertices[:, 1].min()), float(vertices[:, 1].max())
        for head, tail in bone_segments:
            hx, hy = _project_front_xy(head, u_min, u_max, v_min, v_max, width, height)
            tx, ty = _project_front_xy(tail, u_min, u_max, v_min, v_max, width, height)
            ax.plot([hx, tx], [hy, ty], color="#1f4fd6", linewidth=1.2, alpha=0.85)

    ax.set_title(title)
    fig.tight_layout()
    return fig


def _project_front_xy(
    point: np.ndarray,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    width: int,
    height: int,
    margin: float = 0.06,
) -> tuple[float, float]:
    span_u = max(u_max - u_min, 1e-6)
    span_v = max(v_max - v_min, 1e-6)
    scale = (1.0 - 2.0 * margin) * min((width - 1) / span_u, (height - 1) / span_v)
    px = (float(point[0]) - u_min) * scale + margin * (width - 1)
    py = (v_max - float(point[1])) * scale + margin * (height - 1)
    return px, py


def _fit_front_to_image(u: np.ndarray, v: np.ndarray, width: int, height: int, margin: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    span_u = max(u_max - u_min, 1e-6)
    span_v = max(v_max - v_min, 1e-6)
    scale = (1.0 - 2.0 * margin) * min((width - 1) / span_u, (height - 1) / span_v)
    px = (u - u_min) * scale + margin * (width - 1)
    py = (v_max - v) * scale + margin * (height - 1)
    return px, py


def save_textured_shell_animation(
    vertices: np.ndarray,
    triangles: np.ndarray,
    vertex_colors: np.ndarray,
    bones: list[StandardBone],
    bone_indices: np.ndarray,
    weights: np.ndarray,
    output_path: str | Path,
    *,
    frames: int = 24,
    fps: int = 12,
    motion_sequence: list[dict[str, np.ndarray]] | None = None,
    view: str = "front",
    width: int = 640,
    height: int = 960,
    uvs: np.ndarray | None = None,
    texture: Image.Image | None = None,
    show_skeleton: bool = False,
    max_faces: int | None = None,
) -> Path:
    """Animate the full mesh shell by rasterizing every triangle each frame."""
    del max_faces  # deprecated: full mesh faces only
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rest_heads, rest_frames = pose_skeleton_frames(bones, pose=None)
    if motion_sequence is None:
        motion_sequence = [walk_pose(i, frames) for i in range(frames)]

    frame_paths: list[Path] = []
    tmp_dir = output_path.parent / "_raster_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, pose in enumerate(motion_sequence):
        posed_heads, posed_frames = pose_skeleton_frames(bones, pose)
        deformed = skin_vertices(vertices, rest_heads, rest_frames, posed_heads, posed_frames, bone_indices, weights)
        rgb = rasterize_textured_mesh(
            deformed,
            triangles,
            uvs,
            texture,
            view=view,
            width=width,
            height=height,
            vertex_colors=vertex_colors,
        )
        frame_path = tmp_dir / f"frame_{frame_idx:03d}.png"
        save_raster_frame(frame_path, rgb)
        frame_paths.append(frame_path)

    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in frame_paths]
    duration = int(1000 / fps)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    for p in frame_paths:
        p.unlink(missing_ok=True)
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    return output_path


def build_aligned_textured_shell(
    glb_path: str | Path,
    fitted_skeleton_json: str | Path | None = None,
    *,
    k_bones: int = 4,
    temperature: float = 0.08,
    write_skeleton: bool = True,
    output_dir: str | Path | None = None,
) -> dict:
    """End-to-end: align mesh, load or fit skeleton sidecar, compute shell skin weights."""
    glb_path = Path(glb_path)
    if fitted_skeleton_json and Path(fitted_skeleton_json).exists():
        vertices, triangles, uvs, texture = load_glb_mesh_with_material(glb_path)
        aligned, alignment = align_mesh_space(vertices)
        bones = load_fitted_skeleton(fitted_skeleton_json)
    else:
        fit = fit_skeleton_to_glb_mesh(glb_path)
        aligned = fit["vertices"]
        triangles = fit["triangles"]
        uvs = fit["uvs"]
        texture = fit["texture"]
        bones = fit["bones"]
        alignment = fit["alignment"]
        if write_skeleton:
            out_dir = Path(output_dir) if output_dir else Path("outputs") / f"{glb_path.stem}_fitted"
            save_fitted_skeleton(
                bones,
                out_dir / "fitted_skeleton.json",
                source_glb=str(glb_path.resolve()),
                alignment=alignment,
            )

    segments = segment_mesh_vertices(aligned, method="front_view")

    bone_indices, weights = compute_vertex_skin_weights(
        aligned,
        bones,
        segments,
        k_bones=k_bones,
        temperature=temperature,
    )
    vertex_colors = sample_vertex_colors(aligned, uvs, texture)
    return {
        "vertices": aligned,
        "triangles": triangles,
        "uvs": uvs,
        "texture": texture,
        "bones": bones,
        "segments": segments,
        "alignment": alignment,
        "bone_indices": bone_indices,
        "weights": weights,
        "vertex_colors": vertex_colors,
    }

"""GLB textured shell overlay on the standard skeleton + pose_avatar pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from worldmix.character.glb_skeleton_fit import compute_vertex_skin_weights, segment_mesh_vertices
from worldmix.character.mesh_rasterize import composite_dual_view, rasterize_textured_mesh, save_raster_frame
from worldmix.character.mesh_shell import TexturedMeshShell
from worldmix.character.mesh_shell_skinning import (
    pose_skeleton_frames,
    sample_vertex_colors,
    select_render_triangles,
    skin_vertices,
)
from worldmix.character.standard_skeleton_gaussians import (
    RiggedGaussianAvatar,
    build_rigged_gaussian_avatar_from_bones,
    build_standard_humanoid_skeleton,
    pose_avatar,
    walk_pose,
)
from worldmix.character.vit_align import (
    ViTAlignmentResult,
    align_glb_to_standard_skeleton,
    save_alignment_report,
    save_retargeted_skeleton_sidecar,
)


def build_glb_shell_on_standard_skeleton(
    glb_path: str | Path,
    *,
    use_vit: bool = True,
    k_bones: int = 4,
    temperature: float = 0.08,
) -> dict:
    """Appearance shell + ViT-retargeted standard skeleton (topology fixed, proportions adjusted)."""
    glb_path = Path(glb_path)
    alignment = align_glb_to_standard_skeleton(glb_path, use_vit=use_vit)
    bones = alignment.standard_bones
    vertices = alignment.vertices
    triangles = alignment.triangles
    uvs = alignment.uvs
    texture = alignment.texture
    mesh_shell = TexturedMeshShell(
        vertices=vertices,
        triangles=triangles,
        uvs=uvs,
        texture=texture,
        source_glb=str(glb_path.resolve()),
    )
    segments = segment_mesh_vertices(vertices, method="front_view")
    bone_indices, weights = compute_vertex_skin_weights(
        vertices, bones, segments, k_bones=k_bones, temperature=temperature
    )
    vertex_colors = sample_vertex_colors(vertices, uvs, texture)
    gaussian_avatar = build_rigged_gaussian_avatar_from_bones(bones, points_per_bone=16, seed=42)

    return {
        "source_glb": str(glb_path.resolve()),
        "vertices": vertices,
        "triangles": triangles,
        "uvs": uvs,
        "texture": texture,
        "bones": bones,
        "template_bones": alignment.template_bones,
        "segments": segments,
        "bone_indices": bone_indices,
        "weights": weights,
        "vertex_colors": vertex_colors,
        "alignment": alignment,
        "gaussian_avatar": gaussian_avatar,
        "mesh_shell": mesh_shell,
    }


def _preview_triangles(
    vertices: np.ndarray,
    triangles: np.ndarray,
    max_faces: int | None,
    *,
    seed: int = 42,
    batched_shell: bool = False,
    outer_keep_fraction: float = 0.75,
    view: str = "front",
) -> np.ndarray:
    if not batched_shell:
        if max_faces is None or len(triangles) <= max_faces:
            return triangles
        return select_render_triangles(vertices, triangles, max_faces=max_faces, seed=seed)

    shell = TexturedMeshShell(
        vertices=np.asarray(vertices, dtype=np.float64),
        triangles=np.asarray(triangles, dtype=np.int64),
        uvs=None,
        texture=None,
    )
    outer = shell.visible_triangles(view=view, keep_fraction=outer_keep_fraction)
    if max_faces is None or len(outer) <= max_faces:
        return outer
    return select_render_triangles(vertices, outer, max_faces=max_faces, seed=seed)


def save_front_shell_animation(
    shell: dict,
    output_path: str | Path,
    *,
    motion_sequence: list[dict[str, np.ndarray]] | None = None,
    frames: int = 16,
    fps: int = 8,
    panel_width: int = 480,
    panel_height: int = 720,
    max_faces: int | None = None,
    batched_raster: bool = False,
    triangle_batch: int = 8192,
    outer_keep_fraction: float = 0.75,
) -> Path:
    """Front-view GIF: LBS-deformed mesh rasterized (optional face budget for dense GLBs)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = shell["vertices"]
    triangles = shell["triangles"]
    uvs = shell["uvs"]
    texture = shell["texture"]
    bones = shell["bones"]
    bone_indices = shell["bone_indices"]
    weights = shell["weights"]
    vertex_colors = shell["vertex_colors"]
    render_tris = _preview_triangles(
        vertices,
        triangles,
        max_faces,
        batched_shell=batched_raster,
        outer_keep_fraction=outer_keep_fraction,
    )

    if motion_sequence is None:
        motion_sequence = [walk_pose(i, frames) for i in range(frames)]

    rest_heads, rest_frames = pose_skeleton_frames(bones, pose=None)
    frame_paths: list[Path] = []
    tmp_dir = output_path.parent / "_front_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, pose in enumerate(motion_sequence):
        posed_heads, posed_frames = pose_skeleton_frames(bones, pose)
        deformed = skin_vertices(vertices, rest_heads, rest_frames, posed_heads, posed_frames, bone_indices, weights)
        rgb = rasterize_textured_mesh(
            deformed,
            render_tris,
            uvs,
            texture,
            view="front",
            width=panel_width,
            height=panel_height,
            vertex_colors=vertex_colors,
            fast_outer_shell=False,
            batched=batched_raster,
            triangle_batch=triangle_batch,
            keep_fraction=outer_keep_fraction,
        )
        frame_path = tmp_dir / f"frame_{frame_idx:03d}.png"
        save_raster_frame(frame_path, rgb)
        frame_paths.append(frame_path)

    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in frame_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        disposal=2,
    )
    for p in frame_paths:
        p.unlink(missing_ok=True)
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    return output_path


def save_dual_view_shell_animation(
    shell: dict,
    output_path: str | Path,
    *,
    motion_sequence: list[dict[str, np.ndarray]] | None = None,
    frames: int = 16,
    fps: int = 8,
    panel_width: int = 480,
    panel_height: int = 720,
    max_faces: int | None = None,
) -> Path:
    """Render full-mesh front+side GIF driven by pose_avatar (same as standard_skeleton_gaussian_walk)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vertices = shell["vertices"]
    triangles = shell["triangles"]
    uvs = shell["uvs"]
    texture = shell["texture"]
    bones = shell["bones"]
    bone_indices = shell["bone_indices"]
    weights = shell["weights"]
    vertex_colors = shell["vertex_colors"]
    render_tris = _preview_triangles(vertices, triangles, max_faces)

    if motion_sequence is None:
        motion_sequence = [walk_pose(i, frames) for i in range(frames)]

    rest_heads, rest_frames = pose_skeleton_frames(bones, pose=None)
    frame_paths: list[Path] = []
    tmp_dir = output_path.parent / "_dual_view_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for frame_idx, pose in enumerate(motion_sequence):
        posed_heads, posed_frames = pose_skeleton_frames(bones, pose)
        deformed = skin_vertices(vertices, rest_heads, rest_frames, posed_heads, posed_frames, bone_indices, weights)
        dual = composite_dual_view(
            deformed,
            render_tris,
            uvs,
            texture,
            panel_width=panel_width,
            panel_height=panel_height,
            vertex_colors=vertex_colors,
            fast_outer_shell=False,
        )
        frame_path = tmp_dir / f"frame_{frame_idx:03d}.png"
        save_raster_frame(frame_path, dual)
        frame_paths.append(frame_path)

    images = [Image.open(p).convert("P", palette=Image.ADAPTIVE, colors=256) for p in frame_paths]
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
        disposal=2,
    )
    for p in frame_paths:
        p.unlink(missing_ok=True)
    if tmp_dir.exists() and not any(tmp_dir.iterdir()):
        tmp_dir.rmdir()
    return output_path


def save_rest_dual_preview(
    shell: dict,
    output_path: str | Path,
    *,
    panel_width: int = 480,
    panel_height: int = 720,
    max_faces: int | None = None,
) -> Path:
    render_tris = _preview_triangles(shell["vertices"], shell["triangles"], max_faces)
    dual = composite_dual_view(
        shell["vertices"],
        render_tris,
        shell["uvs"],
        shell["texture"],
        panel_width=panel_width,
        panel_height=panel_height,
        vertex_colors=shell["vertex_colors"],
        fast_outer_shell=False,
    )
    return save_raster_frame(output_path, dual)


def save_rest_front_preview(
    shell: dict,
    output_path: str | Path,
    *,
    panel_width: int = 480,
    panel_height: int = 720,
    max_faces: int | None = None,
    batched_raster: bool = False,
    triangle_batch: int = 8192,
    outer_keep_fraction: float = 0.75,
) -> Path:
    render_tris = _preview_triangles(
        shell["vertices"],
        shell["triangles"],
        max_faces,
        batched_shell=batched_raster,
        outer_keep_fraction=outer_keep_fraction,
    )
    rgb = rasterize_textured_mesh(
        shell["vertices"],
        render_tris,
        shell["uvs"],
        shell["texture"],
        view="front",
        width=panel_width,
        height=panel_height,
        vertex_colors=shell["vertex_colors"],
        fast_outer_shell=False,
        batched=batched_raster,
        triangle_batch=triangle_batch,
        keep_fraction=outer_keep_fraction,
    )
    return save_raster_frame(output_path, rgb)


def verify_motion_path_matches_standard(shell: dict, motion_sequence: list[dict[str, np.ndarray]]) -> dict:
    """Same pose_avatar path; skeleton topology matches template, proportions may differ."""
    gaussian_avatar = shell["gaussian_avatar"]
    bones = shell["bones"]
    template = shell.get("template_bones") or build_standard_humanoid_skeleton()
    assert len(gaussian_avatar.bones) == len(bones)
    assert [b.name for b in gaussian_avatar.bones] == [b.name for b in bones]
    assert [b.parent for b in bones] == [b.parent for b in template]

    pose = motion_sequence[0] if motion_sequence else {}
    _, g_segments = pose_avatar(gaussian_avatar, pose)
    _, b_segments = pose_avatar(RiggedGaussianAvatar(bones=bones, gaussians=[]), pose)
    max_head_delta = max(float(np.linalg.norm(a[0] - b[0])) for a, b in zip(g_segments, b_segments))
    return {
        "bone_count": len(bones),
        "max_head_delta": max_head_delta,
        "same_pipeline": max_head_delta < 1e-9,
        "topology_preserved": True,
    }

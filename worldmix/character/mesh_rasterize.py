"""Orthographic full-mesh rasterization with vertex-anchored projection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from worldmix.character.mesh_shell import TexturedMeshShell
from worldmix.character.static_mesh_avatar import _sample_texture


def _fit_to_image(
    u: np.ndarray,
    v: np.ndarray,
    width: int,
    height: int,
    margin: float = 0.06,
) -> tuple[np.ndarray, np.ndarray]:
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    span_u = max(u_max - u_min, 1e-6)
    span_v = max(v_max - v_min, 1e-6)
    scale = (1.0 - 2.0 * margin) * min((width - 1) / span_u, (height - 1) / span_v)
    px = (u - u_min) * scale + margin * (width - 1)
    py = (v_max - v) * scale + margin * (height - 1)
    return px, py


def _rasterize_triangle_vectorized(
    color: np.ndarray,
    zbuf: np.ndarray,
    tri_px: np.ndarray,
    tri_py: np.ndarray,
    tri_z: np.ndarray,
    tri_rgb: np.ndarray,
    tri_uv: np.ndarray | None,
    tex: np.ndarray | None,
) -> None:
    min_x = max(int(np.floor(tri_px.min())) - 1, 0)
    max_x = min(int(np.ceil(tri_px.max())) + 1, color.shape[1] - 1)
    min_y = max(int(np.floor(tri_py.min())) - 1, 0)
    max_y = min(int(np.ceil(tri_py.max())) + 1, color.shape[0] - 1)
    if min_x > max_x or min_y > max_y:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    gx, gy = np.meshgrid(xs, ys)
    tri2d = np.stack([tri_px, tri_py], axis=1)
    x0, y0 = tri2d[0]
    x1, y1 = tri2d[1]
    x2, y2 = tri2d[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-12:
        return
    w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
    w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
    w2 = 1.0 - w0 - w1
    mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
    if not np.any(mask):
        return
    z = w0 * tri_z[0] + w1 * tri_z[1] + w2 * tri_z[2]
    zbuf_slice = zbuf[min_y : max_y + 1, min_x : max_x + 1]
    closer = mask & (z >= zbuf_slice - 1e-9)
    if not np.any(closer):
        return
    zbuf_slice[closer] = z[closer]
    if tri_uv is not None and tex is not None:
        uv = w0[..., None] * tri_uv[0] + w1[..., None] * tri_uv[1] + w2[..., None] * tri_uv[2]
        tx = np.clip(uv[..., 0] * (tex.shape[1] - 1), 0, tex.shape[1] - 1).astype(int)
        ty = np.clip((1.0 - uv[..., 1]) * (tex.shape[0] - 1), 0, tex.shape[0] - 1).astype(int)
        rgb = tex[ty, tx]
        color_slice = color[min_y : max_y + 1, min_x : max_x + 1]
        color_slice[closer] = rgb[closer]
    else:
        rgb = w0[..., None] * tri_rgb[0] + w1[..., None] * tri_rgb[1] + w2[..., None] * tri_rgb[2]
        color[min_y : max_y + 1, min_x : max_x + 1][closer] = rgb[closer]


def rasterize_mesh_shell(
    shell: TexturedMeshShell,
    *,
    view: str = "front",
    width: int = 768,
    height: int = 1024,
    background: tuple[float, float, float] = (0.18, 0.18, 0.20),
    vertex_colors: np.ndarray | None = None,
    outer_shell_only: bool = True,
    keep_fraction: float = 0.70,
) -> np.ndarray:
    """Rasterize by vertex index — faces reuse pre-projected corner positions."""
    px, py, depth = shell.project_ortho(view=view, width=width, height=height)
    tris = shell.visible_triangles(view=view, keep_fraction=keep_fraction) if outer_shell_only else shell.triangles
    uvs = shell.uvs
    texture = shell.texture

    color = np.zeros((height, width, 3), dtype=np.float64)
    color[:] = np.asarray(background, dtype=np.float64)
    zbuf = np.full((height, width), -np.inf, dtype=np.float64)

    if vertex_colors is None:
        if uvs is not None and texture is not None:
            vertex_colors = _sample_texture(texture, uvs).astype(np.float64)
        else:
            vertex_colors = np.full((shell.vertex_count, 3), [0.72, 0.22, 0.48], dtype=np.float64)

    tri_depth = depth[tris].mean(axis=1)
    order = np.argsort(tri_depth)
    tex = np.asarray(texture.convert("RGB"), dtype=np.float64) / 255.0 if texture is not None else None

    for ti in order:
        i0, i1, i2 = tris[ti]
        _rasterize_triangle_vectorized(
            color,
            zbuf,
            np.array([px[i0], px[i1], px[i2]]),
            np.array([py[i0], py[i1], py[i2]]),
            depth[[i0, i1, i2]],
            vertex_colors[[i0, i1, i2]],
            uvs[[i0, i1, i2]] if uvs is not None else None,
            tex,
        )
    return color


def rasterize_textured_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray | None,
    texture: Image.Image | None,
    *,
    view: str = "front",
    width: int = 768,
    height: int = 1024,
    background: tuple[float, float, float] = (0.18, 0.18, 0.20),
    vertex_colors: np.ndarray | None = None,
    outer_shell_only: bool = True,
    fast_outer_shell: bool = False,
) -> np.ndarray:
    """Backward-compatible wrapper around TexturedMeshShell rasterization."""
    tris = np.asarray(triangles, dtype=np.int64)
    if fast_outer_shell:
        from worldmix.character.glb_gaussian_shell import outer_shell_face_mask

        tris = tris[outer_shell_face_mask(np.asarray(vertices, dtype=np.float64), tris)]
        outer_shell_only = False

    shell = TexturedMeshShell(
        vertices=np.asarray(vertices, dtype=np.float64),
        triangles=tris,
        uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
        texture=texture,
    )
    shell.build_adjacency()
    return rasterize_mesh_shell(
        shell,
        view=view,
        width=width,
        height=height,
        background=background,
        vertex_colors=vertex_colors,
        outer_shell_only=outer_shell_only,
    )


def composite_dual_view(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray | None,
    texture: Image.Image | None,
    *,
    panel_width: int = 480,
    panel_height: int = 720,
    vertex_colors: np.ndarray | None = None,
    gap: int = 8,
    background: tuple[float, float, float] = (0.18, 0.18, 0.20),
    outer_shell_only: bool = True,
    fast_outer_shell: bool = False,
) -> np.ndarray:
    """Side-by-side front + side orthographic panels."""
    if fast_outer_shell:
        front = rasterize_textured_mesh(
            vertices,
            triangles,
            uvs,
            texture,
            view="front",
            width=panel_width,
            height=panel_height,
            background=background,
            vertex_colors=vertex_colors,
            fast_outer_shell=True,
        )
        side = rasterize_textured_mesh(
            vertices,
            triangles,
            uvs,
            texture,
            view="side",
            width=panel_width,
            height=panel_height,
            background=background,
            vertex_colors=vertex_colors,
            fast_outer_shell=True,
        )
    else:
        shell = TexturedMeshShell(
            vertices=np.asarray(vertices, dtype=np.float64),
            triangles=np.asarray(triangles, dtype=np.int64),
            uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64),
            texture=texture,
        )
        shell.build_adjacency()
        front = rasterize_mesh_shell(
            shell,
            view="front",
            width=panel_width,
            height=panel_height,
            background=background,
            vertex_colors=vertex_colors,
            outer_shell_only=outer_shell_only,
        )
        side = rasterize_mesh_shell(
            shell,
            view="side",
            width=panel_width,
            height=panel_height,
            background=background,
            vertex_colors=vertex_colors,
            outer_shell_only=outer_shell_only,
        )
    bg = np.asarray(background, dtype=np.float64)
    out = np.zeros((panel_height, panel_width * 2 + gap, 3), dtype=np.float64)
    out[:] = bg
    out[:, :panel_width] = front
    out[:, panel_width + gap :] = side
    return out


def save_raster_frame(path: str | Path, frame_rgb: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(frame_rgb * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(path)
    return path

"""Local Google ViT (HF cache) for front-view joint saliency."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

from worldmix.character.mesh_rasterize import rasterize_textured_mesh

VIT_MODEL_ID = "google/vit-base-patch16-224"
PATCH_GRID = 14
VIT_RENDER_MAX_FACES = 4000

# Vertical bands in patch rows (ViT 224px / 16 = 14 patches).
_JOINT_BANDS = {
    "head_top": (0, 3),
    "left_shoulder": (3, 6),
    "right_shoulder": (3, 6),
    "pelvis": (6, 9),
    "left_hand": (5, 10),
    "right_hand": (5, 10),
    "left_foot": (10, 14),
    "right_foot": (10, 14),
}


@lru_cache(maxsize=1)
def _load_vit():
    import torch
    from transformers import ViTImageProcessor, ViTModel

    processor = ViTImageProcessor.from_pretrained(VIT_MODEL_ID, local_files_only=True)
    model = ViTModel.from_pretrained(VIT_MODEL_ID, output_attentions=True, local_files_only=True)
    model.eval()
    return processor, model, torch


def render_front_vit_image(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray | None,
    texture: Image.Image | None,
    size: int = 224,
    *,
    max_faces: int = VIT_RENDER_MAX_FACES,
) -> tuple[Image.Image, dict]:
    """Orthographic front RGB render packed for ViT input."""
    tris = np.asarray(triangles, dtype=np.int64)
    if len(tris) > max_faces:
        from worldmix.character.mesh_shell_skinning import select_render_triangles

        tris = select_render_triangles(vertices, tris, max_faces=max_faces, seed=0)
    rgb = rasterize_textured_mesh(
        vertices,
        tris,
        uvs,
        texture,
        view="front",
        width=size,
        height=size,
        background=(0.18, 0.18, 0.20),
        fast_outer_shell=True,
    )
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    u = vertices[:, 0]
    v = vertices[:, 1]
    meta = {
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "v_min": float(v.min()),
        "v_max": float(v.max()),
        "size": size,
    }
    return img, meta


def _attention_map(image: Image.Image) -> np.ndarray:
    processor, model, torch = _load_vit()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    # CLS token attention to patch tokens, averaged over heads.
    attn = outputs.attentions[-1][0].mean(dim=0)[0, 1:]
    grid = attn.reshape(PATCH_GRID, PATCH_GRID).cpu().numpy()
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-9)
    return grid


def _patch_centroid(grid: np.ndarray, row_lo: int, row_hi: int, col_lo: int, col_hi: int) -> tuple[float, float]:
    patch = grid[row_lo:row_hi, col_lo:col_hi]
    if patch.size == 0:
        return 0.5, 0.5
    ys, xs = np.meshgrid(
        np.linspace(row_lo + 0.5, row_hi - 0.5, patch.shape[0]),
        np.linspace(col_lo + 0.5, col_hi - 0.5, patch.shape[1]),
        indexing="ij",
    )
    w = patch / (patch.sum() + 1e-9)
    py = float((w * ys).sum() / PATCH_GRID)
    px = float((w * xs).sum() / PATCH_GRID)
    return px, py


def vit_front_joint_uv(image: Image.Image) -> dict[str, tuple[float, float]]:
    """Map ViT CLS attention peaks to normalized front (u,v) joint coordinates."""
    grid = _attention_map(image)
    mid = PATCH_GRID // 2
    out: dict[str, tuple[float, float]] = {}
    for name, (r0, r1) in _JOINT_BANDS.items():
        if "left" in name:
            px, py = _patch_centroid(grid, r0, r1, 0, mid)
        elif "right" in name:
            px, py = _patch_centroid(grid, r0, r1, mid, PATCH_GRID)
        else:
            px, py = _patch_centroid(grid, r0, r1, 0, PATCH_GRID)
        out[name] = (px, 1.0 - py)
    return out


def uv_to_front_xyz(uv: dict[str, tuple[float, float]], bounds: dict) -> dict[str, np.ndarray]:
    """Unproject ViT joint UV to front-view world XYZ (Y-up, facing -Z)."""
    u_min, u_max = bounds["u_min"], bounds["u_max"]
    v_min, v_max = bounds["v_min"], bounds["v_max"]
    joints: dict[str, np.ndarray] = {}
    for name, (nu, nv) in uv.items():
        x = u_min + nu * (u_max - u_min)
        y = v_min + nv * (v_max - v_min)
        joints[name] = np.array([x, y, 0.0], dtype=np.float64)
    return joints

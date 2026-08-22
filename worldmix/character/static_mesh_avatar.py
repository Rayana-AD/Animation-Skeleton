"""Static GLB mesh surface -> World-Mix bone-local Gaussian avatar."""

from __future__ import annotations

import json
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from worldmix.character.standard_skeleton_gaussians import (
    BoneGaussian,
    RiggedGaussianAvatar,
    SkinnedBoneGaussian,
    build_standard_humanoid_skeleton,
    _axis_frame,
)


_COMPONENT_DTYPE = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

_TYPE_WIDTH = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT4": 16,
}


def _read_glb(path: str | Path) -> tuple[dict, bytes]:
    path = Path(path)
    data = path.read_bytes()
    magic, version, _ = struct.unpack_from("<4sII", data, 0)
    if magic != b"glTF" or version != 2:
        raise ValueError(f"Not a glTF 2.0 GLB file: {path}")

    gltf = None
    binary = b""
    offset = 12
    while offset < len(data):
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_len]
        offset += chunk_len
        if chunk_type == 0x4E4F534A:
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == 0x004E4942:
            binary = chunk
    if gltf is None:
        raise ValueError(f"No JSON chunk in GLB: {path}")
    return gltf, binary


def _read_accessor(gltf: dict, binary: bytes, accessor_idx: int) -> np.ndarray:
    accessor = gltf["accessors"][accessor_idx]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = _COMPONENT_DTYPE[accessor["componentType"]]
    width = _TYPE_WIDTH[accessor["type"]]
    count = accessor["count"]
    byte_offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride")
    itemsize = np.dtype(dtype).itemsize * width

    if stride is None or stride == itemsize:
        arr = np.frombuffer(binary, dtype=dtype, count=count * width, offset=byte_offset)
        return arr.reshape(count, width).copy()

    rows = []
    for i in range(count):
        start = byte_offset + i * stride
        row = np.frombuffer(binary, dtype=dtype, count=width, offset=start)
        rows.append(row)
    return np.asarray(rows)


def _buffer_view_bytes(gltf: dict, binary: bytes, view_idx: int) -> bytes:
    view = gltf["bufferViews"][view_idx]
    offset = view.get("byteOffset", 0)
    length = view["byteLength"]
    return binary[offset : offset + length]


def _load_base_color_texture(gltf: dict, binary: bytes) -> Image.Image | None:
    materials = gltf.get("materials", [])
    textures = gltf.get("textures", [])
    images = gltf.get("images", [])
    if not materials:
        return None
    tex_info = materials[0].get("pbrMetallicRoughness", {}).get("baseColorTexture")
    if not tex_info:
        return None
    tex_idx = tex_info.get("index")
    if tex_idx is None or tex_idx >= len(textures):
        return None
    source_idx = textures[tex_idx].get("source")
    if source_idx is None or source_idx >= len(images):
        return None
    image = images[source_idx]
    if "bufferView" not in image:
        return None
    image_bytes = _buffer_view_bytes(gltf, binary, image["bufferView"])
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def load_glb_mesh_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load first mesh primitive POSITION and indices from a GLB file."""
    vertices, triangles, _, _ = load_glb_mesh_with_material(path)
    return vertices, triangles


def load_glb_mesh_with_material(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, Image.Image | None]:
    """Load first mesh primitive POSITION, indices, optional UVs, and base color texture."""
    gltf, binary = _read_glb(path)
    if not gltf.get("meshes"):
        raise ValueError(f"No meshes found in {path}")
    primitive = gltf["meshes"][0]["primitives"][0]
    attrs = primitive.get("attributes", {})
    if "POSITION" not in attrs:
        raise ValueError(f"Mesh has no POSITION attribute: {path}")
    vertices = _read_accessor(gltf, binary, attrs["POSITION"]).astype(np.float64)
    uvs = None
    if "TEXCOORD_0" in attrs:
        uvs = _read_accessor(gltf, binary, attrs["TEXCOORD_0"]).astype(np.float64)
    if "indices" in primitive:
        indices = _read_accessor(gltf, binary, primitive["indices"]).reshape(-1).astype(np.int64)
    else:
        indices = np.arange(len(vertices), dtype=np.int64)
    triangles = indices.reshape(-1, 3)
    texture = _load_base_color_texture(gltf, binary)
    return vertices, triangles, uvs, texture


def normalize_mesh_to_skeleton(
    vertices: np.ndarray,
    skeleton_height: float = 2.0,
    skeleton_floor_y: float = -0.92,
) -> np.ndarray:
    """Center and scale static mesh so it roughly covers the standard skeleton."""
    vertices = np.asarray(vertices, dtype=np.float64)
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    center = (vmin + vmax) * 0.5
    height = max(vmax[1] - vmin[1], 1e-6)
    scale = skeleton_height / height
    out = (vertices - center) * scale
    out[:, 1] -= out[:, 1].min() - skeleton_floor_y
    return out


def sample_mesh_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    count: int = 1200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted triangle surface sampling."""
    rng = np.random.default_rng(seed)
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 1e-12
    tri = tri[valid]
    areas = areas[valid]
    probs = areas / areas.sum()
    choices = rng.choice(len(tri), size=count, replace=True, p=probs)
    chosen = tri[choices]

    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    points = chosen[:, 0] + u[:, None] * (chosen[:, 1] - chosen[:, 0]) + v[:, None] * (chosen[:, 2] - chosen[:, 0])
    normals = np.cross(chosen[:, 1] - chosen[:, 0], chosen[:, 2] - chosen[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    return points, normals


def _sample_texture(texture: Image.Image, uv: np.ndarray) -> np.ndarray:
    pixels = np.asarray(texture.convert("RGB"), dtype=np.float32) / 255.0
    height, width = pixels.shape[:2]
    u = np.mod(uv[:, 0], 1.0)
    v = np.mod(uv[:, 1], 1.0)
    x = np.clip((u * (width - 1)).astype(int), 0, width - 1)
    # glTF UV origin is bottom-left; image arrays are top-left.
    y = np.clip(((1.0 - v) * (height - 1)).astype(int), 0, height - 1)
    return pixels[y, x]


def sample_textured_mesh_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray | None,
    texture: Image.Image | None,
    count: int = 6000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Area-weighted mesh sampling with interpolated texture colors."""
    rng = np.random.default_rng(seed)
    tri = vertices[triangles]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    areas = np.linalg.norm(cross, axis=1) * 0.5
    valid = areas > 1e-12
    tri = tri[valid]
    valid_triangles = triangles[valid]
    areas = areas[valid]
    probs = areas / areas.sum()
    choices = rng.choice(len(tri), size=count, replace=True, p=probs)
    chosen = tri[choices]
    chosen_indices = valid_triangles[choices]

    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]
    w = 1.0 - u - v
    points = w[:, None] * chosen[:, 0] + u[:, None] * chosen[:, 1] + v[:, None] * chosen[:, 2]
    normals = np.cross(chosen[:, 1] - chosen[:, 0], chosen[:, 2] - chosen[:, 0])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12

    if uvs is not None and texture is not None:
        chosen_uvs = uvs[chosen_indices]
        sample_uvs = w[:, None] * chosen_uvs[:, 0] + u[:, None] * chosen_uvs[:, 1] + v[:, None] * chosen_uvs[:, 2]
        colors = _sample_texture(texture, sample_uvs)
    else:
        colors = np.full((count, 3), [0.72, 0.22, 0.48], dtype=np.float64)
    return points, normals, colors


def sample_static_glb_textured_shell(
    path: str | Path,
    samples: int = 6000,
    seed: int = 42,
    skeleton_height: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample textured static GLB shell without applying skeleton deformation."""
    vertices, triangles, uvs, texture = load_glb_mesh_with_material(path)
    vertices = normalize_mesh_to_skeleton(vertices, skeleton_height=skeleton_height)
    return sample_textured_mesh_surface(vertices, triangles, uvs, texture, count=samples, seed=seed)


def _point_to_segment_distance(point: np.ndarray, head: np.ndarray, tail: np.ndarray) -> float:
    axis = tail - head
    t = np.clip(np.dot(point - head, axis) / (np.dot(axis, axis) + 1e-12), 0.0, 1.0)
    closest = head + t * axis
    return float(np.linalg.norm(point - closest))


def _capsule_distance(point: np.ndarray, bone) -> float:
    """Signed-ish distance to a bone capsule surface; negative means inside."""
    return _point_to_segment_distance(point, bone.head, bone.tail) - bone.radius


_SEMANTIC_BONE_GROUPS = {
    "head": ("neck", "head"),
    "torso": ("pelvis", "spine", "chest", "neck"),
    "left_arm": (
        "left_clavicle",
        "left_upper_arm",
        "left_lower_arm",
        "left_wrist",
        "left_hand",
        "left_thumb_01",
        "left_index_01",
        "left_middle_01",
        "left_ring_01",
        "left_pinky_01",
    ),
    "right_arm": (
        "right_clavicle",
        "right_upper_arm",
        "right_lower_arm",
        "right_wrist",
        "right_hand",
        "right_thumb_01",
        "right_index_01",
        "right_middle_01",
        "right_ring_01",
        "right_pinky_01",
    ),
    "left_leg": ("pelvis", "left_upper_leg", "left_lower_leg", "left_foot", "left_toe"),
    "right_leg": ("pelvis", "right_upper_leg", "right_lower_leg", "right_foot", "right_toe"),
}


def classify_surface_point(point: np.ndarray) -> str:
    """Heuristic semantic region for normalized humanoid-ish static meshes."""
    x, y, _ = point
    abs_x = abs(x)
    if y > 0.82:
        return "head"
    if y <= 0.08:
        return "left_leg" if x < 0.0 else "right_leg"
    if abs_x > 0.23 and y > 0.08:
        return "left_arm" if x < 0.0 else "right_arm"
    return "torso"


def _candidate_bone_indices_for_point(point: np.ndarray, bones) -> list[int]:
    region = classify_surface_point(point)
    names = set(_SEMANTIC_BONE_GROUPS[region])
    candidates = [idx for idx, bone in enumerate(bones) if bone.name in names]
    return candidates or list(range(len(bones)))


def bind_points_to_bone_distance_field(
    points: np.ndarray,
    normals: np.ndarray,
    shell_scale: float = 0.018,
    k_bones: int = 4,
    temperature: float = 0.12,
    colors: np.ndarray | None = None,
    semantic_filter: bool = True,
) -> RiggedGaussianAvatar:
    """Bind sampled static mesh points using a smooth bone capsule distance field."""
    bones = build_standard_humanoid_skeleton()
    gaussians: list[SkinnedBoneGaussian] = []
    for point, normal in zip(points, normals):
        candidates = _candidate_bone_indices_for_point(point, bones) if semantic_filter else list(range(len(bones)))
        distances = np.asarray([_capsule_distance(point, bones[idx]) for idx in candidates])
        local_order = np.argsort(distances)[: max(1, min(k_bones, len(candidates)))]
        order = np.asarray([candidates[i] for i in local_order], dtype=np.int32)
        selected_distances = distances[local_order]
        # Softmin over capsule distance. This gives smooth joint behavior and
        # falls back to the nearest bone when one capsule clearly dominates.
        logits = -selected_distances / max(temperature, 1e-6)
        logits -= logits.max()
        weights = np.exp(logits)
        weights /= weights.sum() + 1e-12

        local_means = []
        for bone_id in order:
            bone = bones[int(bone_id)]
            frame = _axis_frame(bone.head, bone.tail)
            local_means.append(frame.T @ (point - bone.head))

        primary_bone = bones[int(order[0])]
        primary_frame = _axis_frame(primary_bone.head, primary_bone.tail)
        # Surface splats are thin along normal and broader in tangent directions.
        normal_local = primary_frame.T @ normal
        dominant = int(np.argmax(np.abs(normal_local)))
        scale = np.full(3, shell_scale)
        scale[dominant] = shell_scale * 0.35
        gaussians.append(
            SkinnedBoneGaussian(
                bone_ids=order.astype(np.int32),
                weights=weights.astype(np.float64),
                local_means=np.asarray(local_means, dtype=np.float64),
                local_scale=scale,
                local_rotation=np.eye(3),
            )
        )
    return RiggedGaussianAvatar(bones=bones, gaussians=gaussians, colors=colors)


def bind_points_to_nearest_bones(
    points: np.ndarray,
    normals: np.ndarray,
    shell_scale: float = 0.018,
) -> RiggedGaussianAvatar:
    """Backward-compatible nearest-bone binding wrapper."""
    return bind_points_to_bone_distance_field(points, normals, shell_scale=shell_scale, k_bones=1)


def build_static_glb_gaussian_avatar(
    path: str | Path,
    samples: int = 1200,
    seed: int = 42,
    skeleton_height: float = 2.0,
    shell_scale: float = 0.018,
    k_bones: int = 4,
    temperature: float = 0.12,
    semantic_filter: bool = True,
) -> RiggedGaussianAvatar:
    """Convert static GLB mesh surface samples into a bone-local Gaussian avatar."""
    vertices, triangles, uvs, texture = load_glb_mesh_with_material(path)
    vertices = normalize_mesh_to_skeleton(vertices, skeleton_height=skeleton_height)
    points, normals, colors = sample_textured_mesh_surface(vertices, triangles, uvs, texture, count=samples, seed=seed)
    return bind_points_to_bone_distance_field(
        points,
        normals,
        shell_scale=shell_scale,
        k_bones=k_bones,
        temperature=temperature,
        colors=colors,
        semantic_filter=semantic_filter,
    )

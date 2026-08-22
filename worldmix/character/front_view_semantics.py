"""Front-view body semantics for static humanoid meshes (hands/feet from frontal silhouette)."""

from __future__ import annotations

import numpy as np

MESH_REGIONS = ("head", "torso", "left_arm", "right_arm", "left_leg", "right_leg")


def detect_front_depth_axis(vertices: np.ndarray) -> int:
    """Pick depth axis (0=X, 2=Z) where frontal projection separates limbs best."""
    verts = np.asarray(vertices, dtype=np.float64)
    spans = [float(verts[:, 0].max() - verts[:, 0].min()), float(verts[:, 2].max() - verts[:, 2].min())]
    # Humanoids usually have wider X (arms) than Z (front-back thickness).
    return 2 if spans[0] >= spans[1] else 0


def front_projection(vertices: np.ndarray, depth_axis: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthographic frontal coords: u=horizontal, v=vertical(Y), d=depth."""
    verts = np.asarray(vertices, dtype=np.float64)
    horizontal = 0 if depth_axis == 2 else 2
    u = verts[:, horizontal]
    v = verts[:, 1]
    d = verts[:, depth_axis]
    return u, v, d


def _normalize01(values: np.ndarray) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    return (values - vmin) / max(vmax - vmin, 1e-6)


def segment_mesh_front_view(vertices: np.ndarray, *, depth_axis: int | None = None) -> dict[str, np.ndarray]:
    """Segment mesh vertices using a frontal (u, v) silhouette instead of side-height bands."""
    verts = np.asarray(vertices, dtype=np.float64)
    if depth_axis is None:
        depth_axis = detect_front_depth_axis(verts)
    u, v, d = front_projection(verts, depth_axis)
    u_n = _normalize01(u)
    v_n = _normalize01(v)

    # Keep the front-facing shell; back-facing vertices inherit nearest front label later if needed.
    front_depth = float(np.percentile(d, 55))
    front_shell = d >= front_depth

    head = front_shell & (v_n > 0.78)
    legs = front_shell & (v_n < 0.46)
    center_u = (u_n >= 0.35) & (u_n <= 0.65)
    left_leg = legs & (u_n < 0.35)
    right_leg = legs & (u_n > 0.65)
    pelvis = legs & center_u

    left_arm = front_shell & (~head) & (~legs) & (u_n < 0.34)
    right_arm = front_shell & (~head) & (~legs) & (u_n > 0.66)
    torso = front_shell & ~(head | left_leg | right_leg | left_arm | right_arm | pelvis)

    # Back shell + side slivers: assign by nearest region centroid in (u, v).
    labels = np.full(len(verts), "torso", dtype=object)
    masks = {
        "head": head,
        "torso": torso | pelvis,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_leg": left_leg,
        "right_leg": right_leg,
    }
    for region, mask in masks.items():
        labels[mask] = region

    back = ~front_shell
    if np.any(back) and np.any(front_shell):
        centroids = {}
        for region in MESH_REGIONS:
            idx = np.where(labels == region)[0]
            if len(idx):
                centroids[region] = np.array([u_n[idx].mean(), v_n[idx].mean()])
        for vi in np.where(back)[0]:
            if v_n[vi] < 0.46 and 0.35 <= u_n[vi] <= 0.65:
                labels[vi] = "torso"
                continue
            dists = {region: np.linalg.norm(np.array([u_n[vi], v_n[vi]]) - c) for region, c in centroids.items()}
            labels[vi] = min(dists, key=dists.get)

    torso_band = (v_n >= 0.30) & (v_n <= 0.82)
    if np.any(torso_band):
        pelvis_x = max(float(np.percentile(np.abs(u[torso_band]), 40)) * 0.35, 0.04)
    else:
        pelvis_x = max(float(np.percentile(np.abs(u), 40)) * 0.35, 0.04)
    center_low = (np.abs(verts[:, 0]) < pelvis_x) & (v_n < 0.48)
    labels[center_low] = "torso"

    return {region: np.where(labels == region)[0] for region in MESH_REGIONS}


def front_extremity_landmarks(vertices: np.ndarray, segments: dict[str, np.ndarray], depth_axis: int = 2) -> dict[str, np.ndarray]:
    """Front-visible extremities: hands, feet, head top, pelvis center."""
    u, v, _ = front_projection(vertices, depth_axis)
    landmarks: dict[str, np.ndarray] = {}

    def pick_extreme(indices: np.ndarray, *, u_mode: str | None = None, v_mode: str | None = None) -> np.ndarray:
        pts = vertices[indices]
        if len(pts) == 0:
            return np.zeros(3)
        if u_mode == "min":
            return pts[int(np.argmin(u[indices]))].copy()
        if u_mode == "max":
            return pts[int(np.argmax(u[indices]))].copy()
        if v_mode == "min":
            return pts[int(np.argmin(v[indices]))].copy()
        if v_mode == "max":
            return pts[int(np.argmax(v[indices]))].copy()
        return pts.mean(axis=0)

    landmarks["head_top"] = pick_extreme(segments["head"], v_mode="max")
    landmarks["pelvis"] = pick_extreme(segments["torso"], v_mode="min")
    landmarks["chest"] = pick_extreme(segments["torso"], v_mode="max")
    landmarks["left_hand"] = pick_extreme(segments["left_arm"], u_mode="min")
    landmarks["right_hand"] = pick_extreme(segments["right_arm"], u_mode="max")
    landmarks["left_foot"] = pick_extreme(segments["left_leg"], v_mode="min")
    landmarks["right_foot"] = pick_extreme(segments["right_leg"], v_mode="min")
    landmarks["left_shoulder"] = pick_extreme(segments["left_arm"], v_mode="max")
    landmarks["right_shoulder"] = pick_extreme(segments["right_arm"], v_mode="max")
    return landmarks

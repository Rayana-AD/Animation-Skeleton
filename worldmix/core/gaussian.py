"""3D Gaussian Splat parameterization: mean, covariance, ellipsoid sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = _normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rotation_matrix_to_quaternion(rot: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    m = np.asarray(rot, dtype=np.float64)
    trace = np.trace(m)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _normalize_quaternion(np.array([w, x, y, z]))


@dataclass
class GaussianSplat:
    """Single 3D Gaussian with standard 3DGS parameterization."""

    mean: np.ndarray  # (3,)
    scale: np.ndarray  # (3,) axis-aligned scales before rotation
    rotation: np.ndarray  # (4,) quaternion [w, x, y, z]

    @property
    def covariance(self) -> np.ndarray:
        """Compute covariance matrix Sigma = R S S^T R^T."""
        rot = quaternion_to_rotation_matrix(self.rotation)
        s = np.diag(np.abs(self.scale) + 1e-8)
        return rot @ s @ s @ rot.T

    @property
    def bounding_radius(self) -> float:
        """Outer sphere radius from max eigenvalue of covariance."""
        eigvals = np.linalg.eigvalsh(self.covariance)
        return float(np.sqrt(max(float(np.max(eigvals)), 0.0)))

    def sample_ellipsoid_mesh(self, n_theta: int = 12, n_phi: int = 8) -> np.ndarray:
        """Return (n_theta*n_phi, 3) surface points on the ellipsoid."""
        verts, _ = self.ellipsoid_surface_mesh(n_u=n_theta, n_v=n_phi)
        return verts

    def ellipsoid_surface_mesh(
        self, n_u: int = 12, n_v: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Triangulated ellipsoid surface: vertices (n_v*n_u, 3), triangles (M, 3)."""
        rot = quaternion_to_rotation_matrix(self.rotation)
        s = np.abs(np.asarray(self.scale, dtype=np.float64)) + 1e-8
        us = np.linspace(0.0, 2.0 * np.pi, n_u, endpoint=False)
        vs = np.linspace(0.0, np.pi, n_v)
        verts = np.empty((n_v * n_u, 3), dtype=np.float64)
        k = 0
        for v in vs:
            sv, cv = np.sin(v), np.cos(v)
            for u in us:
                sph = np.array([sv * np.cos(u), sv * np.sin(u), cv], dtype=np.float64)
                verts[k] = self.mean + rot @ (s * sph)
                k += 1
        tris: list[list[int]] = []
        for i in range(n_v - 1):
            for j in range(n_u):
                a = i * n_u + j
                b = i * n_u + (j + 1) % n_u
                c = (i + 1) * n_u + j
                d = (i + 1) * n_u + (j + 1) % n_u
                tris.append([a, b, c])
                tris.append([b, d, c])
        return verts.astype(np.float32), np.asarray(tris, dtype=np.int64)


def build_gaussians_from_centers(
    centers: np.ndarray,
    base_scale: float = 0.08,
    scale_jitter: float = 0.02,
    rng: np.random.Generator | None = None,
) -> list[GaussianSplat]:
    """Create Gaussian splats from point centers with random orientations."""
    rng = rng or np.random.default_rng()
    centers = np.asarray(centers, dtype=np.float64)
    gaussians: list[GaussianSplat] = []
    for c in centers:
        scale = np.full(3, base_scale) + rng.uniform(-scale_jitter, scale_jitter, 3)
        scale = np.clip(scale, 1e-3, None)
        q = rng.normal(size=4)
        gaussians.append(GaussianSplat(mean=c.copy(), scale=scale, rotation=_normalize_quaternion(q)))
    return gaussians


def gaussians_to_arrays(gaussians: Iterable[GaussianSplat]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack gaussians into (N,3) means, (N,3) scales, (N,4) rotations."""
    means, scales, rots = [], [], []
    for g in gaussians:
        means.append(g.mean)
        scales.append(g.scale)
        rots.append(g.rotation)
    return np.asarray(means), np.asarray(scales), np.asarray(rots)

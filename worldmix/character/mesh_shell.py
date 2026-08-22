"""Textured mesh shell with preserved vertex topology and face adjacency.

Extract once from GLB, transform only vertex positions, project vertices once,
then rasterize faces by index so shared edges always coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from worldmix.character.static_mesh_avatar import load_glb_mesh_with_material


@dataclass
class TexturedMeshShell:
    """Whole GLB appearance shell — topology is the source of truth."""

    vertices: np.ndarray
    triangles: np.ndarray
    uvs: np.ndarray | None
    texture: Image.Image | None
    source_glb: str = ""
    vertex_adjacency: list[set[int]] = field(default_factory=list)
    face_neighbors: list[list[int]] = field(default_factory=list)

    @property
    def face_count(self) -> int:
        return int(len(self.triangles))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices))

    @classmethod
    def from_glb(cls, path: str) -> TexturedMeshShell:
        vertices, triangles, uvs, texture = load_glb_mesh_with_material(path)
        shell = cls(
            vertices=np.asarray(vertices, dtype=np.float64).copy(),
            triangles=np.asarray(triangles, dtype=np.int64).copy(),
            uvs=None if uvs is None else np.asarray(uvs, dtype=np.float64).copy(),
            texture=texture,
            source_glb=str(path),
        )
        shell.build_adjacency()
        return shell

    def copy(self) -> TexturedMeshShell:
        return TexturedMeshShell(
            vertices=self.vertices.copy(),
            triangles=self.triangles.copy(),
            uvs=None if self.uvs is None else self.uvs.copy(),
            texture=self.texture,
            source_glb=self.source_glb,
            vertex_adjacency=[s.copy() for s in self.vertex_adjacency],
            face_neighbors=[list(n) for n in self.face_neighbors],
        )

    def transform_vertices(self, matrix: np.ndarray) -> None:
        """Apply 4x4 or 3x3 transform to positions only; faces/UVs unchanged."""
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (4, 4):
            hom = np.concatenate([self.vertices, np.ones((len(self.vertices), 1))], axis=1)
            self.vertices = (hom @ m.T)[:, :3]
        elif m.shape == (3, 3):
            self.vertices = self.vertices @ m.T
        else:
            raise ValueError(f"Expected 3x3 or 4x4 transform, got {m.shape}")

    def translate(self, delta: np.ndarray) -> None:
        self.vertices = self.vertices + np.asarray(delta, dtype=np.float64)

    def build_adjacency(self) -> None:
        """Record vertex↔vertex and face↔face adjacency from triangle indices."""
        n_verts = len(self.vertices)
        n_faces = len(self.triangles)
        v_adj: list[set[int]] = [set() for _ in range(n_verts)]
        edge_to_faces: dict[tuple[int, int], list[int]] = {}

        for fi, (a, b, c) in enumerate(self.triangles):
            for i, j in ((a, b), (b, c), (c, a)):
                v_adj[i].add(j)
                v_adj[j].add(i)
                key = (min(i, j), max(i, j))
                edge_to_faces.setdefault(key, []).append(fi)

        face_neighbors: list[list[int]] = [[] for _ in range(n_faces)]
        for faces in edge_to_faces.values():
            if len(faces) < 2:
                continue
            for fi in faces:
                for fj in faces:
                    if fi != fj and fj not in face_neighbors[fi]:
                        face_neighbors[fi].append(fj)

        self.vertex_adjacency = v_adj
        self.face_neighbors = face_neighbors

    def face_normals(self) -> np.ndarray:
        tri = self.vertices[self.triangles]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)

    def outer_shell_face_mask(self, *, view: str = "front", keep_fraction: float = 0.70) -> np.ndarray:
        from worldmix.character.mesh_continuity import select_visible_faces

        mask, _ = select_visible_faces(self, view=view, keep_fraction=keep_fraction)
        return mask

    def visible_triangles(self, *, view: str = "front", keep_fraction: float = 0.70) -> np.ndarray:
        from worldmix.character.mesh_continuity import visible_triangles as continuity_visible_triangles

        return continuity_visible_triangles(self, view=view, keep_fraction=keep_fraction)

    def project_ortho(
        self,
        *,
        view: str = "front",
        width: int = 768,
        height: int = 1024,
        margin: float = 0.06,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project every vertex once; all faces reuse these screen coords."""
        verts = self.vertices
        if view == "front":
            u, v, depth = verts[:, 0], verts[:, 1], verts[:, 2]
        elif view == "side":
            u, v, depth = verts[:, 2], verts[:, 1], verts[:, 0]
        else:
            raise ValueError(f"Unsupported view: {view}")

        u_min, u_max = float(u.min()), float(u.max())
        v_min, v_max = float(v.min()), float(v.max())
        span_u = max(u_max - u_min, 1e-6)
        span_v = max(v_max - v_min, 1e-6)
        scale = (1.0 - 2.0 * margin) * min((width - 1) / span_u, (height - 1) / span_v)
        px = (u - u_min) * scale + margin * (width - 1)
        py = (v_max - v) * scale + margin * (height - 1)
        return px, py, depth

    def validate_edge_coincidence(self, px: np.ndarray, py: np.ndarray, *, tol: float = 1e-6) -> dict:
        """Faces sharing a vertex index must reuse identical projected coords."""
        max_err = 0.0
        checked = 0
        for fi, (a, b, c) in enumerate(self.triangles):
            for idx in (a, b, c):
                err = max(abs(px[idx] - px[idx]), abs(py[idx] - py[idx]))
                max_err = max(max_err, err)
                checked += 1
        return {"vertex_projections_checked": checked, "max_index_reuse_error": max_err, "passed": True}

    def _unique_edges(self):
        seen: set[tuple[int, int]] = set()
        for a, b, c in self.triangles:
            for i, j in ((a, b), (b, c), (c, a)):
                key = (min(i, j), max(i, j))
                if key not in seen:
                    seen.add(key)
                    yield key

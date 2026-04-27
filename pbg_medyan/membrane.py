"""Triangulated vesicle membrane for the MEDYAN-style engine.

The membrane is a closed icosphere mesh that responds to:

  - Edge-spring elasticity (in-plane shear / area resistance)
  - Constant osmotic pressure pushing each face outward along its normal
  - Filament plus-end "ratchet" forces from nearby filament tips

In return, the membrane feeds a reaction force back onto each plus end,
which is then projected onto the filament axis to set the Brownian-ratchet
``plus_force_proj`` of the corresponding filament. This is a deliberately
minimal version of the much richer Mem3DG / MEDYAN-vesicle treatment, but
it captures the essential chemo-mechanical feedback that makes
polymerization-driven protrusion possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


# ── icosphere mesh ─────────────────────────────────────────────────


def _icosahedron() -> Tuple[np.ndarray, np.ndarray]:
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)
    return verts, faces


def _subdivide(verts: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    new_verts = list(verts.tolist())
    cache: dict = {}

    def midpoint(a: int, b: int) -> int:
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        m = 0.5 * (np.array(new_verts[a]) + np.array(new_verts[b]))
        m = m / np.linalg.norm(m)
        new_verts.append(m.tolist())
        idx = len(new_verts) - 1
        cache[key] = idx
        return idx

    new_faces = []
    for tri in faces:
        a, b, c = tri
        ab = midpoint(a, b)
        bc = midpoint(b, c)
        ca = midpoint(c, a)
        new_faces.extend([
            [a, ab, ca],
            [b, bc, ab],
            [c, ca, bc],
            [ab, bc, ca],
        ])
    return np.array(new_verts), np.array(new_faces, dtype=int)


def icosphere(radius: float = 1.0, subdivisions: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Build a unit-icosphere mesh and scale to ``radius``.

    Returns (vertices, faces) as numpy arrays.
    """
    v, f = _icosahedron()
    for _ in range(max(0, subdivisions)):
        v, f = _subdivide(v, f)
    return v * radius, f


def _build_edges(faces: np.ndarray) -> np.ndarray:
    """Unique undirected edges from a face array."""
    edge_set = set()
    for tri in faces:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge_set.add((min(u, v), max(u, v)))
    return np.array(sorted(edge_set), dtype=int)


# ── membrane data + forces ─────────────────────────────────────────


@dataclass
class Membrane:
    """Closed triangulated membrane.

    Attributes
    ----------
    vertices : (V, 3) array of vertex positions
    faces    : (F, 3) array of triangle indices
    edges    : (E, 2) array of unique edge indices
    rest_edge_lengths : (E,) array of initial edge lengths
    center   : (3,) center of the membrane (used for outward-normal sign)
    """
    vertices: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    rest_edge_lengths: np.ndarray
    center: np.ndarray

    @classmethod
    def icosphere(cls, radius: float, subdivisions: int = 1,
                  center: np.ndarray = None) -> "Membrane":
        v, f = icosphere(radius, subdivisions)
        if center is not None:
            v = v + center.reshape(1, 3)
        else:
            center = np.zeros(3)
        edges = _build_edges(f)
        rest = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
        return cls(vertices=v.copy(), faces=f.copy(), edges=edges,
                   rest_edge_lengths=rest, center=center.astype(float).copy())

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_faces(self) -> int:
        return self.faces.shape[0]

    def face_normals_and_areas(self) -> Tuple[np.ndarray, np.ndarray]:
        """Per-face outward normals (length 1) and signed areas."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        # Outward orientation: ensure normal points away from membrane center
        face_centroids = (v0 + v1 + v2) / 3.0
        out = face_centroids - self.center.reshape(1, 3)
        n = cross / (2.0 * area.reshape(-1, 1) + 1e-12)
        sign = np.sign(np.einsum('ij,ij->i', n, out))
        sign[sign == 0] = 1.0
        n = n * sign.reshape(-1, 1)
        return n, area

    def vertex_normals(self) -> np.ndarray:
        """Area-weighted vertex normals (unit length)."""
        fn, fa = self.face_normals_and_areas()
        vn = np.zeros_like(self.vertices)
        for i, tri in enumerate(self.faces):
            vn[tri[0]] += fn[i] * fa[i]
            vn[tri[1]] += fn[i] * fa[i]
            vn[tri[2]] += fn[i] * fa[i]
        norms = np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        return vn / norms

    def total_area(self) -> float:
        _, fa = self.face_normals_and_areas()
        return float(fa.sum())

    def total_volume(self) -> float:
        """Closed-mesh volume via signed tetrahedra from origin shifted to center."""
        v = self.vertices - self.center.reshape(1, 3)
        v0 = v[self.faces[:, 0]]
        v1 = v[self.faces[:, 1]]
        v2 = v[self.faces[:, 2]]
        # Dot of v0 with (v1 cross v2) divided by 6 per tetra; sum
        return float(np.abs(np.einsum('ij,ij->i',
                                      v0,
                                      np.cross(v1, v2)).sum()) / 6.0)

    def mean_radius(self) -> float:
        d = np.linalg.norm(self.vertices - self.center.reshape(1, 3), axis=1)
        return float(d.mean())

    def update_center(self) -> None:
        """Recompute center as the centroid of vertices."""
        self.center = self.vertices.mean(axis=0)


def membrane_forces(
    mem: Membrane,
    *,
    edge_stiffness: float,
    pressure: float,
    bending_stiffness: float,
) -> np.ndarray:
    """Return per-vertex forces from edge springs, pressure, and bending."""
    f = np.zeros_like(mem.vertices)

    # Edge springs (in-plane stretch resistance ≈ surface tension)
    if edge_stiffness > 0 and len(mem.edges):
        a = mem.vertices[mem.edges[:, 0]]
        b = mem.vertices[mem.edges[:, 1]]
        d = b - a
        L = np.linalg.norm(d, axis=1)
        L_safe = L + 1e-12
        u = d / L_safe.reshape(-1, 1)
        stretch = (L - mem.rest_edge_lengths)
        f_mag = edge_stiffness * stretch.reshape(-1, 1) * u
        np.add.at(f, mem.edges[:, 0], f_mag)
        np.add.at(f, mem.edges[:, 1], -f_mag)

    # Pressure: each face contributes pressure*area*normal divided to its 3 vertices
    if pressure != 0.0:
        fn, fa = mem.face_normals_and_areas()
        face_force = fn * (pressure * fa).reshape(-1, 1) / 3.0
        for i, tri in enumerate(mem.faces):
            f[tri[0]] += face_force[i]
            f[tri[1]] += face_force[i]
            f[tri[2]] += face_force[i]

    # Bending (Laplacian smoothing approximation)
    if bending_stiffness > 0 and len(mem.edges):
        # Build vertex-neighbor sums via edges
        neigh_sum = np.zeros_like(mem.vertices)
        deg = np.zeros(mem.n_vertices)
        np.add.at(neigh_sum, mem.edges[:, 0], mem.vertices[mem.edges[:, 1]])
        np.add.at(neigh_sum, mem.edges[:, 1], mem.vertices[mem.edges[:, 0]])
        np.add.at(deg, mem.edges[:, 0], 1)
        np.add.at(deg, mem.edges[:, 1], 1)
        deg_safe = np.maximum(deg, 1).reshape(-1, 1)
        avg = neigh_sum / deg_safe
        # Force pulls vertex toward neighbor average (Laplacian)
        f += bending_stiffness * (avg - mem.vertices)

    return f


def filament_membrane_coupling(
    filaments,  # list[Filament]
    mem: Membrane,
    *,
    coupling_radius: float,
    coupling_strength: float,
) -> Tuple[np.ndarray, list]:
    """Compute mutual contact forces between filament plus-ends and membrane vertices.

    Returns
    -------
    membrane_forces : (V, 3) array of forces on each membrane vertex
    plus_end_forces : list of np.ndarray force vectors, one per filament,
                      to be added to that filament's plus-end bead
    """
    V = mem.vertices
    f_mem = np.zeros_like(V)
    plus_forces = []
    if coupling_radius <= 0 or coupling_strength <= 0 or len(filaments) == 0 or V.shape[0] == 0:
        for fil in filaments:
            plus_forces.append(np.zeros(3))
        return f_mem, plus_forces

    # Outward radial direction at each membrane vertex (from membrane center)
    radial = V - mem.center.reshape(1, 3)
    radial_norms = np.linalg.norm(radial, axis=1, keepdims=True) + 1e-12
    out_normals = radial / radial_norms

    for fil in filaments:
        if fil.n_segments == 0:
            plus_forces.append(np.zeros(3))
            continue
        tip = fil.beads[-1]
        diff = V - tip.reshape(1, 3)
        dist = np.linalg.norm(diff, axis=1)
        in_range = dist < coupling_radius
        if not np.any(in_range):
            plus_forces.append(np.zeros(3))
            continue
        idxs = np.where(in_range)[0]
        d = dist[in_range]
        # Soft contact: push vertex outward along its membrane normal,
        # push tip inward (- along normal). Magnitude ∝ (r - d).
        mag = coupling_strength * (coupling_radius - d)
        n_out = out_normals[idxs]
        per_vert = mag.reshape(-1, 1) * n_out
        f_mem[idxs] += per_vert
        plus_forces.append(-per_vert.sum(axis=0))
    return f_mem, plus_forces

"""MEDYAN-style mechanochemical cytoskeletal simulation engine.

Implements a simplified Python version of the MEDYAN model:

    - Semi-flexible actin filaments represented as polylines of cylinders
    - End-tracked polymerization (barbed/plus end) and depolymerization (pointed/minus end)
      with Brownian-ratchet force sensitivity (Peskin et al., 1993)
    - Alpha-actinin-style passive crosslinkers binding nearby filament segments
    - Myosin II minifilament motors that walk toward the plus end with a
      Hill-style force-velocity curve (stall force = Fs)
    - Force-based mechanical relaxation of bead positions (overdamped Langevin)
      under cylinder elasticity, bending stiffness, motor pulling, and crosslink
      tethering

The engine is deterministic given a numpy.random.Generator seed and is sized
for interactive simulations (~10-100 filaments, ~10-100 motors). It is not a
substitute for the production C++ MEDYAN code but reproduces the qualitative
phenomena (treadmilling, contractility, network compaction) faithfully.

References:
    Popov K, Komianos J, Papoian GA (2016) "MEDYAN: Mechanochemical
    Simulations of Contraction and Polarity Alignment in Actomyosin Networks."
    PLOS Comp Biol 12(4): e1004877.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import numpy as np


MONOMER_LENGTH = 0.0027  # micrometers per actin monomer (~2.7 nm)


@dataclass
class Filament:
    """Semi-flexible actin filament represented as a polyline of bead positions.

    The first bead is the minus (pointed) end; the last bead is the plus
    (barbed) end. Each consecutive pair of beads forms a cylinder segment
    with a rest length and an elastic spring constant.
    """
    beads: np.ndarray            # (n, 3) bead positions
    rest_lengths: np.ndarray     # (n-1,) cylinder rest lengths
    plus_force_proj: float = 0.0   # projected boundary force at plus end
    minus_force_proj: float = 0.0  # projected boundary force at minus end

    @property
    def n_beads(self) -> int:
        return self.beads.shape[0]

    @property
    def n_segments(self) -> int:
        return max(0, self.n_beads - 1)

    def total_length(self) -> float:
        if self.n_beads < 2:
            return 0.0
        d = np.diff(self.beads, axis=0)
        return float(np.linalg.norm(d, axis=1).sum())


@dataclass
class Motor:
    """Myosin II minifilament motor with two heads bound to two filaments."""
    filament_a: int
    filament_b: int
    s_a: float  # arc-length position along filament_a (from minus end)
    s_b: float
    # Cached cartesian positions of each head (recomputed each step)
    pos_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pos_b: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Crosslink:
    """Alpha-actinin-style passive crosslinker tethering two filaments."""
    filament_a: int
    filament_b: int
    s_a: float
    s_b: float
    rest_length: float
    pos_a: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pos_b: np.ndarray = field(default_factory=lambda: np.zeros(3))


# ── helper geometry ────────────────────────────────────────────────────

def _arc_to_cartesian(filament: Filament, s: float) -> Tuple[np.ndarray, int, float]:
    """Map an arc-length position s along the filament to a Cartesian point.

    Returns (position, segment_index, segment_fraction) where segment_index
    is the index of the cylinder containing s, and segment_fraction in [0, 1]
    is its location within that segment.
    """
    if filament.n_segments == 0:
        return filament.beads[0].copy(), 0, 0.0

    s = max(0.0, min(s, filament.total_length()))
    cum = 0.0
    for i in range(filament.n_segments):
        a = filament.beads[i]
        b = filament.beads[i + 1]
        seg_len = float(np.linalg.norm(b - a))
        if seg_len <= 1e-12:
            continue
        if s <= cum + seg_len:
            f = (s - cum) / seg_len
            return a + f * (b - a), i, f
        cum += seg_len
    # Fallback: end of filament
    return filament.beads[-1].copy(), filament.n_segments - 1, 1.0


def _segment_axis(filament: Filament, seg: int) -> np.ndarray:
    """Unit vector along a filament segment (minus → plus direction)."""
    a = filament.beads[seg]
    b = filament.beads[seg + 1]
    d = b - a
    n = np.linalg.norm(d)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return d / n


# ── the engine ─────────────────────────────────────────────────────────


class MedyanEngine:
    """Mechanochemical cytoskeleton simulation engine.

    Time stepping: each call to step(dt) performs one combined chemistry +
    mechanics integration. Chemistry uses tau-leaping (deterministic mean
    rates with stochastic Poisson draws); mechanics uses overdamped Euler
    relaxation of bead positions.

    Parameters
    ----------
    box_size: side length of cubic simulation domain (micrometers)
    n_filaments: initial number of filaments to seed
    initial_filament_length: rest length of seeded filaments (micrometers)
    actin_concentration: free G-actin concentration (uM)
    k_on_plus, k_off_plus: barbed-end on/off rates
    k_on_minus, k_off_minus: pointed-end on/off rates
    n_motors, n_crosslinks: initial bound motor / crosslink counts
    motor_v0: load-free walking velocity (um/s)
    motor_stall_force: stall force Fs (pN)
    motor_force: characteristic motor pulling force (pN)
    crosslink_stiffness: linear spring constant of crosslinker (pN/um)
    cylinder_stiffness: filament stretch stiffness (pN/um)
    bending_stiffness: filament bending modulus (pN um^2)
    boundary_force_scale: characteristic Brownian-ratchet force (pN)
    drag_coefficient: overdamped friction coefficient
    rng_seed: integer for numpy Generator; use 0 for default
    """

    def __init__(
        self,
        box_size: float = 2.0,
        n_filaments: int = 10,
        initial_filament_length: float = 0.4,
        actin_concentration: float = 10.0,
        k_on_plus: float = 11.6,
        k_off_plus: float = 1.4,
        k_on_minus: float = 1.3,
        k_off_minus: float = 0.8,
        n_motors: int = 0,
        n_crosslinks: int = 0,
        motor_v0: float = 0.2,
        motor_stall_force: float = 8.0,
        motor_force: float = 4.0,
        crosslink_stiffness: float = 8.0,
        crosslink_rest_length: float = 0.03,
        cylinder_stiffness: float = 20.0,
        bending_stiffness: float = 0.05,
        boundary_force_scale: float = 1.5,
        drag_coefficient: float = 40.0,
        bind_radius: float = 0.2,
        crosslink_unbind_rate: float = 0.05,
        motor_unbind_rate: float = 0.1,
        seed_region_fraction: float = 0.6,
        rng_seed: int = 0,
    ):
        self.box_size = float(box_size)
        self.actin_concentration = float(actin_concentration)
        self.k_on_plus = float(k_on_plus)
        self.k_off_plus = float(k_off_plus)
        self.k_on_minus = float(k_on_minus)
        self.k_off_minus = float(k_off_minus)
        self.motor_v0 = float(motor_v0)
        self.motor_stall_force = float(motor_stall_force)
        self.motor_force = float(motor_force)
        self.crosslink_stiffness = float(crosslink_stiffness)
        self.crosslink_rest_length = float(crosslink_rest_length)
        self.cylinder_stiffness = float(cylinder_stiffness)
        self.bending_stiffness = float(bending_stiffness)
        self.boundary_force_scale = float(boundary_force_scale)
        self.drag_coefficient = float(drag_coefficient)
        self.bind_radius = float(bind_radius)
        self.crosslink_unbind_rate = float(crosslink_unbind_rate)
        self.motor_unbind_rate = float(motor_unbind_rate)
        self.seed_region_fraction = float(seed_region_fraction)

        seed = rng_seed if rng_seed else None
        self.rng = np.random.default_rng(seed)

        self.time = 0.0
        self.filaments: List[Filament] = []
        self.motors: List[Motor] = []
        self.crosslinks: List[Crosslink] = []

        self._seed_filaments(n_filaments, initial_filament_length)
        self._seed_motors(n_motors)
        self._seed_crosslinks(n_crosslinks)

    # ── seeding ────────────────────────────────────────────────────

    def _random_unit_vec(self) -> np.ndarray:
        v = self.rng.standard_normal(3)
        v /= np.linalg.norm(v) + 1e-12
        return v

    def _seed_filaments(self, n: int, length: float) -> None:
        n_seg = max(1, int(round(length / 0.05)))
        seg_len = length / n_seg
        cx = 0.5 * self.box_size
        half_region = 0.5 * self.seed_region_fraction * self.box_size
        lo = cx - half_region
        hi = cx + half_region
        for _ in range(n):
            center = self.rng.uniform(lo, hi, size=3)
            axis = self._random_unit_vec()
            start = center - 0.5 * length * axis
            beads = np.array([start + i * seg_len * axis for i in range(n_seg + 1)])
            rest = np.full(n_seg, seg_len)
            self.filaments.append(Filament(beads=beads, rest_lengths=rest))

    def _seed_motors(self, n: int) -> None:
        if len(self.filaments) < 2 or n == 0:
            return
        for _ in range(n):
            self._try_bind_motor()

    def _seed_crosslinks(self, n: int) -> None:
        if len(self.filaments) < 2 or n == 0:
            return
        for _ in range(n):
            self._try_bind_crosslink()

    # ── chemistry: polymerization with Brownian ratchet ────────────

    def _polymerize(self, dt: float) -> None:
        for fil in self.filaments:
            # Brownian-ratchet rate suppression at plus end
            kp = self.k_on_plus * self.actin_concentration
            if fil.plus_force_proj > 0:
                kp *= math.exp(-fil.plus_force_proj / self.boundary_force_scale)
            km = self.k_off_plus
            n_add = self.rng.poisson(max(0.0, kp * dt))
            n_rem = self.rng.poisson(max(0.0, km * dt))
            net_plus = (n_add - n_rem) * MONOMER_LENGTH

            # Pointed end (slower kinetics, also force-sensitive)
            kp2 = self.k_on_minus * self.actin_concentration
            if fil.minus_force_proj > 0:
                kp2 *= math.exp(-fil.minus_force_proj / self.boundary_force_scale)
            km2 = self.k_off_minus
            n_add2 = self.rng.poisson(max(0.0, kp2 * dt))
            n_rem2 = self.rng.poisson(max(0.0, km2 * dt))
            net_minus = (n_add2 - n_rem2) * MONOMER_LENGTH

            self._extend_plus_end(fil, net_plus)
            self._extend_minus_end(fil, net_minus)

    def _extend_plus_end(self, fil: Filament, delta: float) -> None:
        if abs(delta) < 1e-12 or fil.n_segments == 0:
            return
        last = fil.n_segments - 1
        new_len = fil.rest_lengths[last] + delta
        # Cap length so filament doesn't disappear
        if new_len < 0.01:
            new_len = 0.01
        # Move last bead along axis
        axis = _segment_axis(fil, last)
        fil.beads[-1] = fil.beads[-2] + new_len * axis
        fil.rest_lengths[last] = new_len
        # If too long, split into two segments at standard size
        if new_len > 0.12:
            split_len = new_len / 2.0
            fil.beads[-1] = fil.beads[-2] + split_len * axis
            fil.rest_lengths[last] = split_len
            new_bead = fil.beads[-1] + split_len * axis
            fil.beads = np.vstack([fil.beads, new_bead[None, :]])
            fil.rest_lengths = np.append(fil.rest_lengths, split_len)
        # If too short and we have spare segments, merge with neighbor
        if new_len < 0.02 and fil.n_segments > 1:
            merge_len = fil.rest_lengths[last - 1] + new_len
            fil.beads = fil.beads[:-1].copy()
            fil.beads[-1] = fil.beads[-2] + merge_len * _segment_axis(fil, last - 1)
            fil.rest_lengths = np.append(fil.rest_lengths[:-2], merge_len)

    def _extend_minus_end(self, fil: Filament, delta: float) -> None:
        if abs(delta) < 1e-12 or fil.n_segments == 0:
            return
        new_len = fil.rest_lengths[0] + delta
        if new_len < 0.01:
            new_len = 0.01
        axis = -_segment_axis(fil, 0)  # minus-end direction
        fil.beads[0] = fil.beads[1] + new_len * axis
        fil.rest_lengths[0] = new_len
        if new_len > 0.12:
            split_len = new_len / 2.0
            fil.beads[0] = fil.beads[1] + split_len * axis
            fil.rest_lengths[0] = split_len
            new_bead = fil.beads[0] + split_len * axis
            fil.beads = np.vstack([new_bead[None, :], fil.beads])
            fil.rest_lengths = np.insert(fil.rest_lengths, 0, split_len)
        if new_len < 0.02 and fil.n_segments > 1:
            merge_len = fil.rest_lengths[1] + new_len
            fil.beads = fil.beads[1:].copy()
            fil.beads[0] = fil.beads[1] - merge_len * _segment_axis(fil, 0)
            fil.rest_lengths = np.append([merge_len], fil.rest_lengths[2:])

    # ── motor walking ──────────────────────────────────────────────

    def _walk_motors(self, dt: float) -> None:
        kept: List[Motor] = []
        for m in self.motors:
            if m.filament_a >= len(self.filaments) or m.filament_b >= len(self.filaments):
                continue
            fa = self.filaments[m.filament_a]
            fb = self.filaments[m.filament_b]
            la = fa.total_length()
            lb = fb.total_length()
            if la < 1e-6 or lb < 1e-6:
                continue
            # Force on motor head = crosslink-like spring between two heads
            pa, _, _ = _arc_to_cartesian(fa, m.s_a)
            pb, _, _ = _arc_to_cartesian(fb, m.s_b)
            r = pb - pa
            r_norm = float(np.linalg.norm(r))
            if r_norm > 2.0 * self.bind_radius:  # stretched too far → unbind
                continue
            # Force-velocity (Hill): v(F) = v0 * (1 - F/Fs)
            f_load = self.motor_force * min(1.0, r_norm / 0.2)
            fac = max(0.0, 1.0 - f_load / self.motor_stall_force)
            v = self.motor_v0 * fac
            m.s_a = min(la, m.s_a + v * dt)
            m.s_b = min(lb, m.s_b + v * dt)
            # Stochastic unbinding
            if self.rng.random() < self.motor_unbind_rate * dt:
                continue
            m.pos_a, _, _ = _arc_to_cartesian(fa, m.s_a)
            m.pos_b, _, _ = _arc_to_cartesian(fb, m.s_b)
            kept.append(m)
        self.motors = kept

    def _try_bind_motor(self) -> bool:
        if len(self.filaments) < 2:
            return False
        a, b = self.rng.choice(len(self.filaments), 2, replace=False)
        fa = self.filaments[a]
        fb = self.filaments[b]
        la = fa.total_length()
        lb = fb.total_length()
        if la < 0.05 or lb < 0.05:
            return False
        # Try a random pair of arc-length positions
        for _ in range(12):
            sa = self.rng.uniform(0, la)
            sb = self.rng.uniform(0, lb)
            pa, _, _ = _arc_to_cartesian(fa, sa)
            pb, _, _ = _arc_to_cartesian(fb, sb)
            if np.linalg.norm(pa - pb) < self.bind_radius:
                m = Motor(int(a), int(b), sa, sb, pa.copy(), pb.copy())
                self.motors.append(m)
                return True
        return False

    def _try_bind_crosslink(self) -> bool:
        if len(self.filaments) < 2:
            return False
        a, b = self.rng.choice(len(self.filaments), 2, replace=False)
        fa = self.filaments[a]
        fb = self.filaments[b]
        la = fa.total_length()
        lb = fb.total_length()
        if la < 0.05 or lb < 0.05:
            return False
        for _ in range(15):
            sa = self.rng.uniform(0, la)
            sb = self.rng.uniform(0, lb)
            pa, _, _ = _arc_to_cartesian(fa, sa)
            pb, _, _ = _arc_to_cartesian(fb, sb)
            d = float(np.linalg.norm(pa - pb))
            if d < self.bind_radius and d > 1e-6:
                cl = Crosslink(int(a), int(b), sa, sb,
                               max(self.crosslink_rest_length, d),
                               pa.copy(), pb.copy())
                self.crosslinks.append(cl)
                return True
        return False

    def _crosslink_dynamics(self, dt: float) -> None:
        kept: List[Crosslink] = []
        for cl in self.crosslinks:
            if cl.filament_a >= len(self.filaments) or cl.filament_b >= len(self.filaments):
                continue
            if self.rng.random() < self.crosslink_unbind_rate * dt:
                continue
            fa = self.filaments[cl.filament_a]
            fb = self.filaments[cl.filament_b]
            cl.pos_a, _, _ = _arc_to_cartesian(fa, cl.s_a)
            cl.pos_b, _, _ = _arc_to_cartesian(fb, cl.s_b)
            kept.append(cl)
        self.crosslinks = kept

    # ── mechanics: overdamped relaxation ───────────────────────────

    def _accumulate_forces(self) -> List[np.ndarray]:
        forces = [np.zeros_like(f.beads) for f in self.filaments]

        # Stretch (Hookean cylinders)
        for i, fil in enumerate(self.filaments):
            for s in range(fil.n_segments):
                a = fil.beads[s]
                b = fil.beads[s + 1]
                d = b - a
                L = float(np.linalg.norm(d))
                if L < 1e-9:
                    continue
                u = d / L
                stretch = (L - fil.rest_lengths[s])
                f = self.cylinder_stiffness * stretch * u
                forces[i][s] += f
                forces[i][s + 1] -= f

        # Bending: angular springs at internal beads
        for i, fil in enumerate(self.filaments):
            for s in range(1, fil.n_beads - 1):
                a = fil.beads[s - 1]
                b = fil.beads[s]
                c = fil.beads[s + 1]
                u = b - a
                v = c - b
                un = np.linalg.norm(u)
                vn = np.linalg.norm(v)
                if un < 1e-9 or vn < 1e-9:
                    continue
                u_hat = u / un
                v_hat = v / vn
                # Force pulling bead toward the average direction of neighbors
                bend = self.bending_stiffness * (v_hat - u_hat)
                forces[i][s - 1] += -0.5 * bend
                forces[i][s] += bend
                forces[i][s + 1] += -0.5 * bend

        # Crosslinkers (bilinear spring, distributed to flanking beads)
        for cl in self.crosslinks:
            fa = self.filaments[cl.filament_a]
            fb = self.filaments[cl.filament_b]
            pa, sa_idx, sa_f = _arc_to_cartesian(fa, cl.s_a)
            pb, sb_idx, sb_f = _arc_to_cartesian(fb, cl.s_b)
            r = pb - pa
            d = float(np.linalg.norm(r))
            if d < 1e-9:
                continue
            u = r / d
            f_mag = self.crosslink_stiffness * (d - cl.rest_length)
            f = f_mag * u
            # Distribute to flanking beads via barycentric weights
            forces[cl.filament_a][sa_idx] += (1 - sa_f) * f
            forces[cl.filament_a][sa_idx + 1] += sa_f * f
            forces[cl.filament_b][sb_idx] -= (1 - sb_f) * f
            forces[cl.filament_b][sb_idx + 1] -= sb_f * f

        # Motors: pulling spring (always contractile by construction)
        for m in self.motors:
            fa = self.filaments[m.filament_a]
            fb = self.filaments[m.filament_b]
            pa, sa_idx, sa_f = _arc_to_cartesian(fa, m.s_a)
            pb, sb_idx, sb_f = _arc_to_cartesian(fb, m.s_b)
            r = pb - pa
            d = float(np.linalg.norm(r))
            if d < 1e-9:
                continue
            u = r / d
            f_mag = self.motor_force  # contractile force toward each other
            f = f_mag * u
            forces[m.filament_a][sa_idx] += (1 - sa_f) * f
            forces[m.filament_a][sa_idx + 1] += sa_f * f
            forces[m.filament_b][sb_idx] -= (1 - sb_f) * f
            forces[m.filament_b][sb_idx + 1] -= sb_f * f

        return forces

    def _project_boundary_forces(self, forces: List[np.ndarray]) -> None:
        """Project net forces onto plus/minus end axes for ratchet feedback."""
        for i, fil in enumerate(self.filaments):
            if fil.n_segments == 0:
                continue
            ax_plus = _segment_axis(fil, fil.n_segments - 1)
            ax_minus = -_segment_axis(fil, 0)
            f_plus = forces[i][-1]
            f_minus = forces[i][0]
            # Force *opposing* growth is positive boundary force
            fil.plus_force_proj = max(0.0, -float(np.dot(f_plus, ax_plus)))
            fil.minus_force_proj = max(0.0, -float(np.dot(f_minus, ax_minus)))

    def _relax(self, dt: float, n_substeps: int = 4) -> None:
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
            forces = self._accumulate_forces()
            self._project_boundary_forces(forces)
            inv_drag = 1.0 / self.drag_coefficient
            for i, fil in enumerate(self.filaments):
                fil.beads = fil.beads + sub_dt * inv_drag * forces[i]

    # ── public API ─────────────────────────────────────────────────

    def step(self, dt: float, *, n_new_motors: int = 0,
             n_new_crosslinks: int = 0) -> None:
        """Advance the engine by one time step of size dt seconds."""
        self._polymerize(dt)
        self._walk_motors(dt)
        self._crosslink_dynamics(dt)
        for _ in range(n_new_motors):
            self._try_bind_motor()
        for _ in range(n_new_crosslinks):
            self._try_bind_crosslink()
        self._relax(dt)
        self.time += dt

    def network_metrics(self) -> dict:
        """Aggregate scalar metrics describing network state."""
        n_fil = len(self.filaments)
        lengths = [f.total_length() for f in self.filaments]
        total_len = float(sum(lengths))
        mean_len = float(np.mean(lengths)) if lengths else 0.0

        # Network span = bounding-box diagonal of all beads
        if n_fil:
            allb = np.vstack([f.beads for f in self.filaments])
            span = float(np.linalg.norm(allb.max(axis=0) - allb.min(axis=0)))
            radius_of_gyration = float(np.sqrt(np.mean(
                np.sum((allb - allb.mean(axis=0)) ** 2, axis=1))))
        else:
            span = 0.0
            radius_of_gyration = 0.0

        # Sum of bending energy
        bending_e = 0.0
        for fil in self.filaments:
            for s in range(1, fil.n_beads - 1):
                u = fil.beads[s] - fil.beads[s - 1]
                v = fil.beads[s + 1] - fil.beads[s]
                un = np.linalg.norm(u)
                vn = np.linalg.norm(v)
                if un > 1e-9 and vn > 1e-9:
                    cos_t = np.dot(u, v) / (un * vn)
                    cos_t = max(-1.0, min(1.0, cos_t))
                    bending_e += self.bending_stiffness * (1 - cos_t)

        # Stretch energy
        stretch_e = 0.0
        for fil in self.filaments:
            for s in range(fil.n_segments):
                d = fil.beads[s + 1] - fil.beads[s]
                L = float(np.linalg.norm(d))
                stretch_e += 0.5 * self.cylinder_stiffness * (L - fil.rest_lengths[s]) ** 2

        return {
            'n_filaments': n_fil,
            'n_motors': len(self.motors),
            'n_crosslinks': len(self.crosslinks),
            'total_length': total_len,
            'mean_filament_length': mean_len,
            'network_span': span,
            'radius_of_gyration': radius_of_gyration,
            'bending_energy': float(bending_e),
            'stretch_energy': float(stretch_e),
            'total_energy': float(bending_e + stretch_e),
        }

    def snapshot(self) -> dict:
        """Full geometric snapshot for visualization."""
        return {
            'time': self.time,
            'filaments': [f.beads.tolist() for f in self.filaments],
            'motors': [
                {'a': [m.pos_a[0], m.pos_a[1], m.pos_a[2]],
                 'b': [m.pos_b[0], m.pos_b[1], m.pos_b[2]]}
                for m in self.motors
            ],
            'crosslinks': [
                {'a': [c.pos_a[0], c.pos_a[1], c.pos_a[2]],
                 'b': [c.pos_b[0], c.pos_b[1], c.pos_b[2]]}
                for c in self.crosslinks
            ],
        }

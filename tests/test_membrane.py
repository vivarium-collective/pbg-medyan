"""Tests for the triangulated membrane and filament-membrane coupling."""

import math
import numpy as np
import pytest
from process_bigraph import allocate_core

from pbg_medyan.membrane import Membrane, icosphere
from pbg_medyan.engine import MedyanEngine
from pbg_medyan.processes import MedyanProcess


def test_icosphere_geometry():
    v, f = icosphere(radius=1.0, subdivisions=1)
    assert v.shape[0] == 42
    assert f.shape[0] == 80
    # All vertices on sphere
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_membrane_metrics_match_sphere():
    m = Membrane.icosphere(radius=1.0, subdivisions=3)
    # Subdivision 3 → 642 verts; area ≈ 4πr², volume ≈ (4/3)πr³
    assert math.isclose(m.total_area(), 4 * math.pi, rel_tol=0.02)
    assert math.isclose(m.total_volume(), (4 / 3) * math.pi, rel_tol=0.03)
    assert math.isclose(m.mean_radius(), 1.0, abs_tol=1e-6)


def test_engine_with_membrane_disabled_is_identical():
    """When enable_membrane=False, behavior matches the no-membrane engine."""
    common = dict(
        n_filaments=4, initial_filament_length=0.3,
        seed_region_fraction=0.4, rng_seed=42,
    )
    e1 = MedyanEngine(**common, enable_membrane=False)
    e2 = MedyanEngine(**common)  # default disabled
    for _ in range(5):
        e1.step(0.3)
        e2.step(0.3)
    assert e1.network_metrics()['total_length'] == pytest.approx(
        e2.network_metrics()['total_length'])


def test_filaments_push_membrane_outward():
    """A pre-grown filament tip should locally push out the nearest mem vertex."""
    e = MedyanEngine(
        box_size=2.0, n_filaments=0,
        actin_concentration=0.0,  # disable polymerization noise
        k_on_plus=0.0, k_off_plus=0.0,
        k_on_minus=0.0, k_off_minus=0.0,
        enable_membrane=True, membrane_radius=0.5,
        membrane_subdivisions=2,
        membrane_edge_stiffness=8.0,
        membrane_bending_stiffness=0.5,
        membrane_pressure=0.0,
        membrane_filament_coupling_radius=0.15,
        membrane_filament_coupling_strength=120.0,
        rng_seed=7,
    )
    # Hand-seed a single filament with its tip just inside the membrane,
    # pointing along +x. Box center is at (1,1,1); membrane radius 0.5.
    from pbg_medyan.engine import Filament
    cx = np.full(3, 1.0)
    axis = np.array([1.0, 0.0, 0.0])
    seg_len = 0.1
    n_seg = 5  # 0.5 um total → tip just at membrane surface
    beads = np.array([cx + j * seg_len * axis for j in range(n_seg + 1)])
    rest = np.full(n_seg, seg_len)
    e.filaments.append(Filament(beads=beads, rest_lengths=rest))

    # Track the membrane vertex closest to the filament tip
    tip = e.filaments[0].beads[-1]
    dists = np.linalg.norm(e.membrane.vertices - tip, axis=1)
    near_idx = int(dists.argmin())
    r0 = float(np.linalg.norm(e.membrane.vertices[near_idx] - cx))

    for _ in range(50):
        e.step(0.05)

    r1 = float(np.linalg.norm(e.membrane.vertices[near_idx] - cx))
    assert r1 > r0, (
        f'Local membrane vertex should bulge outward in response to '
        f'filament tip push: r0={r0:.4f}, r1={r1:.4f}')


def test_membrane_metrics_in_process_outputs():
    core = allocate_core()
    core.register_link('MedyanProcess', MedyanProcess)
    proc = MedyanProcess(
        config={'enable_membrane': True, 'membrane_radius': 0.5,
                'membrane_subdivisions': 1,
                'n_filaments': 3, 'rng_seed': 1},
        core=core)
    state = proc.initial_state()
    assert state['membrane_area'] > 0
    assert state['membrane_volume'] > 0
    assert state['membrane_mean_radius'] == pytest.approx(0.5, abs=1e-6)


def test_membrane_disabled_metrics_zero():
    core = allocate_core()
    core.register_link('MedyanProcess', MedyanProcess)
    proc = MedyanProcess(config={'n_filaments': 2, 'rng_seed': 1}, core=core)
    state = proc.initial_state()
    assert state['membrane_area'] == 0.0
    assert state['membrane_volume'] == 0.0
    assert state['membrane_mean_radius'] == 0.0


def test_membrane_in_snapshot():
    e = MedyanEngine(n_filaments=2, enable_membrane=True,
                     membrane_subdivisions=1, rng_seed=1)
    snap = e.snapshot()
    assert snap['membrane'] is not None
    assert len(snap['membrane']['vertices']) == 42
    assert len(snap['membrane']['faces']) == 80
    assert len(snap['membrane']['center']) == 3


def test_no_membrane_in_snapshot_when_disabled():
    e = MedyanEngine(n_filaments=2, enable_membrane=False, rng_seed=1)
    assert e.snapshot()['membrane'] is None

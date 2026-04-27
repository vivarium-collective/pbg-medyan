"""Unit tests for the MEDYAN-style simulation engine."""

import math
import numpy as np
import pytest

from pbg_medyan.engine import MedyanEngine


def test_seed_filaments():
    e = MedyanEngine(n_filaments=5, initial_filament_length=0.25,
                     rng_seed=1)
    m = e.network_metrics()
    assert m['n_filaments'] == 5
    assert math.isclose(m['mean_filament_length'], 0.25, abs_tol=1e-6)
    assert m['total_energy'] >= 0
    assert m['n_motors'] == 0
    assert m['n_crosslinks'] == 0


def test_pure_polymerization_grows_filaments():
    """With no depoly bias, filaments should net-grow."""
    e = MedyanEngine(
        n_filaments=4, initial_filament_length=0.2,
        actin_concentration=20.0,
        k_on_plus=15.0, k_off_plus=1.0,
        k_on_minus=0.5, k_off_minus=0.5,
        n_motors=0, n_crosslinks=0,
        rng_seed=42,
    )
    L0 = e.network_metrics()['total_length']
    for _ in range(20):
        e.step(0.5)
    L1 = e.network_metrics()['total_length']
    assert L1 > L0, f'Total length should grow: {L0} -> {L1}'


def test_depolymerization_shrinks():
    """When off-rate dominates on-rate, filaments should shrink."""
    e = MedyanEngine(
        n_filaments=4, initial_filament_length=0.4,
        actin_concentration=0.05,  # very low: koff dominates
        k_on_plus=11.6, k_off_plus=1.4,
        k_on_minus=1.3, k_off_minus=0.8,
        rng_seed=42,
    )
    L0 = e.network_metrics()['total_length']
    for _ in range(40):
        e.step(0.5)
    L1 = e.network_metrics()['total_length']
    assert L1 < L0, f'Total length should shrink: {L0} -> {L1}'


def test_motor_binding_succeeds():
    """With clustered seeding, motors should bind initially."""
    e = MedyanEngine(
        box_size=1.0, n_filaments=12, initial_filament_length=0.4,
        seed_region_fraction=0.4,
        n_motors=20, n_crosslinks=20, bind_radius=0.25,
        rng_seed=42,
    )
    m = e.network_metrics()
    assert m['n_motors'] > 0
    assert m['n_crosslinks'] > 0


def test_actomyosin_compaction():
    """Motors + crosslinks should constrain network span vs free polymerization."""
    common = dict(
        box_size=1.0, n_filaments=15, initial_filament_length=0.4,
        seed_region_fraction=0.4,
        actin_concentration=0.3,  # below critical concentration; little net growth
        bind_radius=0.3,
        motor_force=10.0, motor_v0=0.2,
        crosslink_stiffness=30.0,
        rng_seed=7,
    )
    # Reference: free filaments, no motors / crosslinks
    e_free = MedyanEngine(**common, n_motors=0, n_crosslinks=0)
    # Active: actomyosin contraction
    e_act = MedyanEngine(**common, n_motors=30, n_crosslinks=40)
    for _ in range(40):
        e_free.step(0.5)
        e_act.step(0.5, n_new_motors=2, n_new_crosslinks=3)
    span_free = e_free.network_metrics()['network_span']
    span_act = e_act.network_metrics()['network_span']
    assert span_act < span_free, (
        f'Active network should be more compact than free: {span_act} >= {span_free}')


def test_metrics_are_finite():
    e = MedyanEngine(n_filaments=8, n_motors=5, n_crosslinks=10,
                     bind_radius=0.3, rng_seed=42)
    for _ in range(15):
        e.step(0.4, n_new_motors=1, n_new_crosslinks=1)
    m = e.network_metrics()
    for k, v in m.items():
        assert np.isfinite(v), f'Non-finite metric: {k} = {v}'
        assert v >= 0 if k != 'time' else True


def test_snapshot_shapes():
    e = MedyanEngine(n_filaments=3, initial_filament_length=0.3,
                     n_motors=3, n_crosslinks=4, bind_radius=0.4,
                     rng_seed=1)
    e.step(0.5)
    s = e.snapshot()
    assert 'time' in s
    assert len(s['filaments']) == 3
    for f in s['filaments']:
        assert len(f) >= 2  # at least 2 beads
        for bead in f:
            assert len(bead) == 3
    for m in s['motors']:
        assert len(m['a']) == 3 and len(m['b']) == 3


def test_reproducibility_with_seed():
    e1 = MedyanEngine(n_filaments=5, n_motors=3, n_crosslinks=5,
                      bind_radius=0.3, rng_seed=123)
    e2 = MedyanEngine(n_filaments=5, n_motors=3, n_crosslinks=5,
                      bind_radius=0.3, rng_seed=123)
    for _ in range(10):
        e1.step(0.3)
        e2.step(0.3)
    m1 = e1.network_metrics()
    m2 = e2.network_metrics()
    assert math.isclose(m1['total_length'], m2['total_length'])
    assert m1['n_motors'] == m2['n_motors']

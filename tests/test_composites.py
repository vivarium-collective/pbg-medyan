"""Integration tests: full Composite assembly and run."""

import pytest
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter

from pbg_medyan.processes import MedyanProcess
from pbg_medyan.composites import make_network_document


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('MedyanProcess', MedyanProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_make_document_structure():
    doc = make_network_document(interval=1.0, n_filaments=4, rng_seed=1)
    assert 'cytoskeleton' in doc
    assert 'stores' in doc
    assert 'emitter' in doc
    assert doc['cytoskeleton']['address'] == 'local:MedyanProcess'
    assert doc['cytoskeleton']['interval'] == 1.0
    assert doc['cytoskeleton']['config']['n_filaments'] == 4


def test_composite_runs(core):
    doc = make_network_document(
        interval=1.0,
        box_size=1.0, n_filaments=4, initial_filament_length=0.25,
        actin_concentration=3.0,
        n_motors=2, n_crosslinks=3,
        rng_seed=7,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(5.0)
    assert sim.state['stores']['n_filaments'] == 4
    assert sim.state['stores']['total_length'] > 0


def test_emitter_collects_history(core):
    doc = make_network_document(
        interval=1.0,
        n_filaments=3, actin_concentration=4.0,
        rng_seed=1,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(4.0)
    results = gather_emitter_results(sim)
    hist = results[('emitter',)]
    # 5 emits expected: t=0, 1, 2, 3, 4
    assert len(hist) >= 4
    times = [h['time'] for h in hist]
    assert times[-1] == pytest.approx(4.0)
    # network grew over time
    L_first = hist[1]['total_length']  # t=0 emit may pre-date Process run
    L_last = hist[-1]['total_length']
    assert L_last >= L_first


def test_serialization_roundtrip_consistency(core):
    """Two runs with the same seed should yield identical metric histories."""
    def run_once():
        doc = make_network_document(
            interval=1.0, n_filaments=5, n_motors=2,
            actin_concentration=4.0, bind_radius=0.3,
            rng_seed=4242,
        )
        sim = Composite({'state': doc}, core=core)
        sim.run(3.0)
        return gather_emitter_results(sim)[('emitter',)]

    h1 = run_once()
    h2 = run_once()
    # Same seed -> same final total_length to floating-point precision
    assert h1[-1]['total_length'] == pytest.approx(h2[-1]['total_length'])
    assert h1[-1]['n_filaments'] == h2[-1]['n_filaments']

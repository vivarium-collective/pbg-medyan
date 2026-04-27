"""Unit tests for MedyanProcess (the PBG bridge)."""

import pytest
from process_bigraph import allocate_core

from pbg_medyan.processes import MedyanProcess


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('MedyanProcess', MedyanProcess)
    return c


def test_instantiation_uses_defaults(core):
    proc = MedyanProcess(config={}, core=core)
    assert proc.config['box_size'] == 2.0
    assert proc.config['n_filaments'] == 10
    assert proc.config['k_on_plus'] == 11.6
    assert proc.config['n_substeps'] == 8


def test_config_override(core):
    proc = MedyanProcess(
        config={'n_filaments': 20, 'actin_concentration': 5.0,
                'rng_seed': 99},
        core=core)
    assert proc.config['n_filaments'] == 20
    assert proc.config['actin_concentration'] == 5.0
    assert proc.config['rng_seed'] == 99


def test_initial_state(core):
    proc = MedyanProcess(
        config={'n_filaments': 5, 'initial_filament_length': 0.3,
                'rng_seed': 1},
        core=core)
    state = proc.initial_state()
    assert state['n_filaments'] == 5
    assert state['total_length'] > 0
    assert state['mean_filament_length'] > 0
    assert state['n_motors'] == 0
    assert state['total_energy'] >= 0


def test_single_update(core):
    proc = MedyanProcess(
        config={'n_filaments': 5, 'rng_seed': 7,
                'actin_concentration': 8.0},
        core=core)
    proc.initial_state()
    result = proc.update({}, interval=2.0)
    assert 'total_length' in result
    assert 'n_filaments' in result
    assert isinstance(result['total_length'], float)
    assert isinstance(result['n_filaments'], int)


def test_outputs_schema(core):
    proc = MedyanProcess(config={}, core=core)
    outputs = proc.outputs()
    expected = {
        'n_filaments', 'n_motors', 'n_crosslinks',
        'total_length', 'mean_filament_length',
        'network_span', 'radius_of_gyration',
        'bending_energy', 'stretch_energy', 'total_energy',
    }
    assert expected.issubset(set(outputs.keys()))


def test_no_inputs(core):
    """MedyanProcess is internally driven; no input ports."""
    proc = MedyanProcess(config={}, core=core)
    assert proc.inputs() == {}


def test_get_engine_after_init(core):
    proc = MedyanProcess(config={'n_filaments': 3, 'rng_seed': 1},
                         core=core)
    proc.initial_state()
    engine = proc.get_engine()
    assert engine is not None
    snap = engine.snapshot()
    assert len(snap['filaments']) == 3


def test_update_advances_time(core):
    proc = MedyanProcess(config={'n_filaments': 4, 'rng_seed': 1},
                         core=core)
    proc.initial_state()
    proc.update({}, interval=2.5)
    assert pytest.approx(proc.get_engine().time) == 2.5
    proc.update({}, interval=1.0)
    assert pytest.approx(proc.get_engine().time) == 3.5


def test_zero_motors_runs_cleanly(core):
    proc = MedyanProcess(
        config={'n_filaments': 3, 'n_motors': 0, 'n_crosslinks': 0,
                'rng_seed': 5},
        core=core)
    proc.initial_state()
    result = proc.update({}, interval=2.0)
    assert result['n_motors'] == 0
    assert result['n_crosslinks'] == 0

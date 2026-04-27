"""Offline tests for ``MedyanCxxProcess`` (no real MEDYAN binary needed).

These cover schema, default state, and the systeminput.txt that the
process produces for first-call vs restart-call scenarios. The actual
subprocess invocation is exercised by the gated integration tests in
``test_cxx_integration.py``.
"""

import os
import numpy as np
import pytest
from process_bigraph import allocate_core

from pbg_medyan.cxx import MedyanCxxProcess
from pbg_medyan.cxx.io import FilamentSnapshot, TrajFrame


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('MedyanCxxProcess', MedyanCxxProcess)
    return c


def test_instantiation_defaults(core):
    proc = MedyanCxxProcess(config={}, core=core)
    assert proc.config['n_filaments'] == 5
    assert proc.config['boundary_shape'] == 'CUBIC'
    assert proc.config['chemistry_preset'] == 'actin_only'
    assert proc.config['snapshot_interval'] == 0.5


def test_outputs_schema(core):
    proc = MedyanCxxProcess(config={}, core=core)
    out = proc.outputs()
    for key in ['n_filaments', 'n_linkers', 'n_motors', 'n_branchers',
                'total_filament_length', 'mean_filament_length',
                'network_span', 'cxx_runtime_seconds']:
        assert key in out, f'missing port {key}'


def test_initial_state_does_not_run(core):
    proc = MedyanCxxProcess(config={}, core=core)
    state = proc.initial_state()
    assert state['n_filaments'] == 0
    assert state['cxx_runtime_seconds'] == 0.0
    # No work dir created yet
    assert proc._work_dir is None


def test_build_keywords_first_call(core):
    proc = MedyanCxxProcess(
        config={'n_filaments': 12, 'filament_length': 3,
                'compartment_size': 250.0, 'snapshot_interval': 0.2},
        core=core)
    kw = proc._build_keywords(runtime=2.0, restart=False)
    assert kw['NUMFILAMENTS'] == 12
    assert kw['FILAMENTLENGTH'] == 3
    assert kw['PROJECTIONTYPE'] == 'STRAIGHT'
    assert 'FILAMENTFILE' not in kw
    assert kw['RUNTIME'] == 2.0
    assert kw['SNAPSHOTTIME'] == 0.2
    # All three compartment-size keys are unified
    assert kw['COMPARTMENTSIZEX'] == 250.0
    assert kw['COMPARTMENTSIZEY'] == 250.0
    assert kw['COMPARTMENTSIZEZ'] == 250.0


def test_build_keywords_restart(core):
    proc = MedyanCxxProcess(
        config={'n_filaments': 5, 'snapshot_interval': 0.5}, core=core)
    kw = proc._build_keywords(runtime=1.0, restart=True)
    assert kw['PROJECTIONTYPE'] == 'PREDEFINED'
    assert kw['FILAMENTFILE'] == 'filaments.txt'
    # Random-init keys must be cleared (None) so MEDYAN doesn't double-seed
    assert kw['NUMFILAMENTS'] is None
    assert kw['FILAMENTLENGTH'] is None
    assert kw['FILAMENTTYPE'] is None


def test_extra_keywords_override(core):
    proc = MedyanCxxProcess(
        config={}, core=core,
        extra_keywords={'FBENDINGK': 200.0, 'CALGORITHM': 'GILLESPIE'})
    kw = proc._build_keywords(runtime=1.0, restart=False)
    assert kw['FBENDINGK'] == 200.0
    assert kw['CALGORITHM'] == 'GILLESPIE'


def test_snapshot_interval_clamped_to_runtime(core):
    proc = MedyanCxxProcess(
        config={'snapshot_interval': 5.0}, core=core)
    kw = proc._build_keywords(runtime=1.0, restart=False)
    assert kw['SNAPSHOTTIME'] == 1.0


def test_frame_metrics_empty():
    empty = TrajFrame(chemstep=0, time=0.0,
                      n_filaments=0, n_linkers=0, n_motors=0, n_branchers=0)
    m = MedyanCxxProcess._frame_metrics(empty, runtime=0.1)
    assert m['n_filaments'] == 0
    assert m['total_filament_length'] == 0.0
    assert m['network_span'] == 0.0
    assert m['cxx_runtime_seconds'] == 0.1


def test_frame_metrics_with_filaments():
    fil_a = FilamentSnapshot(
        fil_id=0, fil_type=0, cyl_length=1, delta_l=0, delta_r=0,
        beads=np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]]))
    fil_b = FilamentSnapshot(
        fil_id=1, fil_type=0, cyl_length=2, delta_l=0, delta_r=0,
        beads=np.array([[0.0, 0.0, 0.0],
                        [50.0, 0.0, 0.0],
                        [50.0, 50.0, 0.0]]))
    frame = TrajFrame(chemstep=0, time=0.5,
                      n_filaments=2, n_linkers=0, n_motors=0, n_branchers=0,
                      filaments=[fil_a, fil_b])
    m = MedyanCxxProcess._frame_metrics(frame, runtime=0.2)
    assert m['n_filaments'] == 2
    assert m['total_filament_length'] == pytest.approx(100 + 50 + 50)
    assert m['mean_filament_length'] == pytest.approx(100.0)
    # Span: bbox is x:[0,100] y:[0,50] z:[0,0] → diag = sqrt(100^2 + 50^2)
    assert m['network_span'] == pytest.approx(np.hypot(100, 50))


def test_unknown_chemistry_preset_errors(core, tmp_path):
    proc = MedyanCxxProcess(
        config={'chemistry_preset': 'nonexistent'}, core=core)
    with pytest.raises(ValueError, match='Unknown chemistry_preset'):
        proc._resolve_chemistry(str(tmp_path))


def test_chemistry_text_overrides_preset(core, tmp_path):
    custom = "SPECIESDIFFUSING: ZZ 1 1 0 0 REG\n"
    proc = MedyanCxxProcess(
        config={'chemistry_text': custom,
                'chemistry_preset': 'actin_only'},
        core=core)
    name = proc._resolve_chemistry(str(tmp_path))
    written = open(os.path.join(str(tmp_path), name)).read()
    assert 'SPECIESDIFFUSING: ZZ' in written

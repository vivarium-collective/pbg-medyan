"""End-to-end integration tests against a real MEDYAN binary.

Skipped automatically unless ``$MEDYAN_BIN`` points at an executable
or ``medyan`` is on ``PATH``. To run:

    MEDYAN_BIN=/path/to/medyan pytest tests/test_cxx_integration.py -v

Or pass ``--run-medyan`` to opt in even when discovery would normally
skip (see conftest.py).
"""

import os
import shutil

import numpy as np
import pytest
from process_bigraph import allocate_core

from pbg_medyan.cxx import MedyanCxxProcess
from pbg_medyan.cxx.io import find_medyan_binary


def _binary_available() -> bool:
    try:
        find_medyan_binary()
        return True
    except FileNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _binary_available(),
    reason='MEDYAN binary not found (set MEDYAN_BIN or add medyan to PATH).')


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('MedyanCxxProcess', MedyanCxxProcess)
    return c


def test_first_call_seeds_filaments(core, tmp_path):
    proc = MedyanCxxProcess(
        config={
            'work_dir': str(tmp_path),
            'keep_workdir': True,
            'n_filaments': 4,
            'filament_length': 1,
            'snapshot_interval': 0.5,
            'minimization_interval': 0.05,
            'chemistry_preset': 'actin_only',
            'timeout': 120.0,
        },
        core=core)
    result = proc.update({}, interval=1.0)
    assert result['n_filaments'] == 4
    assert result['total_filament_length'] > 0
    assert result['cxx_runtime_seconds'] > 0
    last = proc.get_last_frame()
    assert last is not None
    assert len(last.filaments) == 4


def test_restart_preserves_filament_geometry(core, tmp_path):
    proc = MedyanCxxProcess(
        config={
            'work_dir': str(tmp_path),
            'keep_workdir': True,
            'n_filaments': 3,
            'filament_length': 1,
            'snapshot_interval': 1.0,
            'chemistry_preset': 'actin_only',
            'timeout': 120.0,
        },
        core=core)
    proc.update({}, interval=1.0)
    last_after_first = proc.get_last_frame()
    fil_lengths_first = [f.total_length_nm() for f in last_after_first.filaments]

    proc.update({}, interval=0.5)
    last_after_second = proc.get_last_frame()
    fil_lengths_second = [f.total_length_nm() for f in last_after_second.filaments]

    # Restart should preserve the filament count
    assert len(fil_lengths_second) == len(fil_lengths_first)
    # And cumulative time should advance
    assert proc.cumulative_time() == pytest.approx(1.5)

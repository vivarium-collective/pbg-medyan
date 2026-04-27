"""Offline tests for the MEDYAN C++ wrapper I/O layer.

These tests do not invoke the MEDYAN binary; they exercise the parser
and writers against synthetic fixtures so they can run in CI without
a working MEDYAN build.
"""

import os
import tempfile
import textwrap

import numpy as np
import pytest

from pbg_medyan.cxx import (
    parse_snapshot_traj,
    write_filament_file,
    write_system_input,
    find_medyan_binary,
)
from pbg_medyan.cxx.io import (
    write_chemistry_input,
    FilamentSnapshot,
    LinkerSnapshot,
    MotorSnapshot,
    BrancherSnapshot,
)


# ── synthetic snapshots ─────────────────────────────────────────────


SNAPSHOT_TWO_FRAMES = textwrap.dedent("""\
    0 0.0 2 1 1 0
    FILAMENT 0 0 0 0 0
    100.0 200.0 300.0 250.0 200.0 300.0
    FILAMENT 1 0 1 0 0
    100.0 100.0 100.0 100.0 200.0 100.0 100.0 200.0 200.0
    LINKER 0 0
    150.0 200.0 300.0 100.0 150.0 100.0
    MOTOR 0 0
    180.0 200.0 300.0 100.0 180.0 100.0

    1 0.5 1 0 0 1
    FILAMENT 0 0 0 0 0
    100.0 200.0 300.0 260.0 200.0 300.0
    BRANCHER 0 0
    100.0 200.0 300.0 105.0 200.0 305.0
""")


def test_parse_two_frames():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'snapshot.traj')
    open(p, 'w').write(SNAPSHOT_TWO_FRAMES)
    frames = parse_snapshot_traj(p)
    assert len(frames) == 2

    f0 = frames[0]
    assert f0.chemstep == 0
    assert f0.time == 0.0
    assert len(f0.filaments) == 2
    assert len(f0.linkers) == 1
    assert len(f0.motors) == 1
    assert len(f0.branchers) == 0

    fil0 = f0.filaments[0]
    assert fil0.fil_id == 0
    assert fil0.fil_type == 0
    assert fil0.cyl_length == 0
    assert fil0.beads.shape == (2, 3)
    np.testing.assert_allclose(fil0.beads[0], [100.0, 200.0, 300.0])
    assert fil0.total_length_nm() == pytest.approx(150.0)

    fil1 = f0.filaments[1]
    assert fil1.beads.shape == (3, 3)

    lk = f0.linkers[0]
    assert isinstance(lk, LinkerSnapshot)
    np.testing.assert_allclose(lk.start, [150.0, 200.0, 300.0])
    np.testing.assert_allclose(lk.end, [100.0, 150.0, 100.0])

    mo = f0.motors[0]
    assert isinstance(mo, MotorSnapshot)
    np.testing.assert_allclose(mo.end, [100.0, 180.0, 100.0])

    f1 = frames[1]
    assert f1.time == 0.5
    assert len(f1.filaments) == 1
    assert len(f1.branchers) == 1
    assert isinstance(f1.branchers[0], BrancherSnapshot)


def test_parse_skips_blank_and_comment_lines():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'snapshot.traj')
    open(p, 'w').write(textwrap.dedent("""\
        # leading comment
        0 0.0 1 0 0 0
        # mid-frame comment
        FILAMENT 0 0 0 0 0
        0.0 0.0 0.0 10.0 0.0 0.0
    """))
    frames = parse_snapshot_traj(p)
    assert len(frames) == 1
    assert frames[0].filaments[0].total_length_nm() == pytest.approx(10.0)


def test_parse_short_header():
    """Some MEDYAN versions emit fewer count fields. Parser must tolerate."""
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'snapshot.traj')
    open(p, 'w').write("0 1.5\n")
    frames = parse_snapshot_traj(p)
    assert len(frames) == 1
    assert frames[0].time == 1.5
    assert frames[0].n_filaments == 0


# ── writers ─────────────────────────────────────────────────────────


def test_filament_file_round_trip():
    fil = FilamentSnapshot(
        fil_id=0, fil_type=2, cyl_length=1, delta_l=0.0, delta_r=0.0,
        beads=np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]))
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'fil.txt')
    write_filament_file(p, [fil])
    text = open(p).read()
    assert text.startswith('FILAMENT 2 ')
    tokens = text.strip().split()
    assert tokens[0] == 'FILAMENT'
    assert tokens[1] == '2'
    coords = list(map(float, tokens[2:]))
    np.testing.assert_allclose(coords, [10, 20, 30, 40, 50, 60])


def test_system_input_skips_none_and_formats_bool():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'sys.txt')
    write_system_input(p, {
        'NX': 2,
        'BOUNDARYSHAPE': 'CUBIC',
        'allow-same-filament-pair-binding': False,
        'NUMFILAMENTS': None,         # should be skipped
        'FBENDINGK': 672.0,
    })
    text = open(p).read()
    assert 'NX: 2' in text
    assert 'BOUNDARYSHAPE: CUBIC' in text
    assert 'allow-same-filament-pair-binding: false' in text
    assert 'NUMFILAMENTS' not in text
    assert 'FBENDINGK: 672' in text


def test_chemistry_writer_appends_newline():
    d = tempfile.mkdtemp()
    p = os.path.join(d, 'chem.txt')
    write_chemistry_input(p, 'SPECIESDIFFUSING: AD 1000 20e6 0 0 REG')
    assert open(p).read().endswith('\n')


# ── binary discovery ────────────────────────────────────────────────


def test_find_binary_raises_when_missing(monkeypatch):
    monkeypatch.delenv('MEDYAN_BIN', raising=False)
    monkeypatch.setattr('shutil.which', lambda _: None)
    with pytest.raises(FileNotFoundError, match='MEDYAN binary not found'):
        find_medyan_binary()


def test_find_binary_uses_env(monkeypatch, tmp_path):
    fake = tmp_path / 'medyan'
    fake.write_text('#!/bin/sh\nexit 0\n')
    fake.chmod(0o755)
    monkeypatch.setenv('MEDYAN_BIN', str(fake))
    assert find_medyan_binary() == str(fake)

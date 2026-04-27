"""I/O utilities for the MEDYAN C++ binary.

This module is pure I/O — it knows nothing about process-bigraph.
Reading these functions in isolation is the simplest way to understand
the wire format the wrapper produces and consumes.

Format references:
  - System input keywords:
    https://raw.githubusercontent.com/medyan-dev/medyan-public/main/docs/manual/input-files.md
  - Trajectory output:
    https://raw.githubusercontent.com/medyan-dev/medyan-public/main/docs/manual/output-files.md
  - Real example used as the syntax baseline for write_system_input:
    https://github.com/medyan-dev/medyan-public/tree/main/examples/actin_only

Snapshot frames look like::

    chemstepnumber time n_filaments n_linkers n_motors n_branchers
    FILAMENT id type cyllength deltal deltar
    x0 y0 z0 x1 y1 z1 ...
    LINKER id type
    sx sy sz ex ey ez
    MOTOR id type
    sx sy sz ex ey ez
    BRANCHER id type
    sx sy sz ex ey ez

with frames separated by a blank line. Bead count for each filament
equals ``cyllength + 1``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


# ── data classes ────────────────────────────────────────────────────


@dataclass
class FilamentSnapshot:
    fil_id: int
    fil_type: int
    cyl_length: int
    delta_l: float
    delta_r: float
    beads: np.ndarray  # (n_beads, 3) — coordinates in nm

    @property
    def n_beads(self) -> int:
        return int(self.beads.shape[0])

    def total_length_nm(self) -> float:
        if self.n_beads < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.beads, axis=0), axis=1).sum())


@dataclass
class LinkerSnapshot:
    obj_id: int
    obj_type: int
    start: np.ndarray  # (3,) nm
    end: np.ndarray


@dataclass
class MotorSnapshot:
    obj_id: int
    obj_type: int
    start: np.ndarray
    end: np.ndarray


@dataclass
class BrancherSnapshot:
    obj_id: int
    obj_type: int
    start: np.ndarray
    end: np.ndarray


@dataclass
class TrajFrame:
    chemstep: int
    time: float                    # seconds
    n_filaments: int
    n_linkers: int
    n_motors: int
    n_branchers: int
    filaments: List[FilamentSnapshot] = field(default_factory=list)
    linkers: List[LinkerSnapshot] = field(default_factory=list)
    motors: List[MotorSnapshot] = field(default_factory=list)
    branchers: List[BrancherSnapshot] = field(default_factory=list)


# ── parsing ─────────────────────────────────────────────────────────


def parse_snapshot_traj(path: str) -> List[TrajFrame]:
    """Parse an entire ``snapshot.traj`` text file into a list of frames.

    Tolerates header lines with 4-7 tokens (different MEDYAN versions
    emit a varying number of count fields). Skips comment lines.
    """
    with open(path) as f:
        text = f.read()

    frames: List[TrajFrame] = []
    # Frames are separated by blank lines (one or more)
    raw_frames = re.split(r'\n\s*\n+', text.strip())
    for raw in raw_frames:
        if not raw.strip():
            continue
        lines = [ln for ln in raw.strip().split('\n')
                 if ln.strip() and not ln.lstrip().startswith('#')]
        if not lines:
            continue
        header = lines[0].split()
        if len(header) < 2:
            continue
        chemstep = int(header[0])
        time = float(header[1])
        n_filaments = int(header[2]) if len(header) > 2 else 0
        n_linkers = int(header[3]) if len(header) > 3 else 0
        n_motors = int(header[4]) if len(header) > 4 else 0
        n_branchers = int(header[5]) if len(header) > 5 else 0
        frame = TrajFrame(
            chemstep=chemstep, time=time,
            n_filaments=n_filaments, n_linkers=n_linkers,
            n_motors=n_motors, n_branchers=n_branchers)

        i = 1
        while i < len(lines):
            tok = lines[i].split()
            kind = tok[0]
            if kind == 'FILAMENT' and i + 1 < len(lines):
                fid = int(tok[1])
                ftype = int(tok[2])
                cyl_len = int(tok[3]) if len(tok) > 3 else 0
                dl = float(tok[4]) if len(tok) > 4 else 0.0
                dr = float(tok[5]) if len(tok) > 5 else 0.0
                coords = list(map(float, lines[i + 1].split()))
                beads = np.array(coords, dtype=float).reshape(-1, 3)
                frame.filaments.append(FilamentSnapshot(
                    fil_id=fid, fil_type=ftype, cyl_length=cyl_len,
                    delta_l=dl, delta_r=dr, beads=beads))
                i += 2
            elif kind in ('LINKER', 'MOTOR', 'BRANCHER') and i + 1 < len(lines):
                oid = int(tok[1])
                otype = int(tok[2])
                coords = list(map(float, lines[i + 1].split()))
                start = np.array(coords[:3], dtype=float)
                end = np.array(coords[3:6], dtype=float)
                cls = {'LINKER': LinkerSnapshot,
                       'MOTOR': MotorSnapshot,
                       'BRANCHER': BrancherSnapshot}[kind]
                target = {'LINKER': frame.linkers,
                          'MOTOR': frame.motors,
                          'BRANCHER': frame.branchers}[kind]
                target.append(cls(obj_id=oid, obj_type=otype,
                                  start=start, end=end))
                i += 2
            else:
                i += 1
        frames.append(frame)
    return frames


# ── writing ─────────────────────────────────────────────────────────


def write_filament_file(path: str, filaments: List[FilamentSnapshot]) -> None:
    """Write a ``FILAMENTFILE`` for ``PROJECTIONTYPE: PREDEFINED`` restart.

    Each line: ``FILAMENT <type> x0 y0 z0 x1 y1 z1 ...`` (nm).
    """
    with open(path, 'w') as f:
        for fil in filaments:
            coords = ' '.join(f'{c:.6f}' for c in fil.beads.ravel())
            f.write(f'FILAMENT {fil.fil_type} {coords}\n')


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return f'{v:g}'
    return str(v)


def write_system_input(path: str, params: Dict[str, Any]) -> None:
    """Write a ``systeminput.txt`` from a flat dict of MEDYAN keywords.

    Keys are written verbatim (caller is responsible for using the
    documented MEDYAN keyword spellings; see input-files.md). ``None``
    values are skipped so the dict can carry "unset" entries.
    Booleans render as ``true``/``false`` (for keys like
    ``allow-same-filament-pair-binding``); other scalars use ``str()``.
    """
    with open(path, 'w') as f:
        for key, value in params.items():
            if value is None:
                continue
            f.write(f'{key}: {_format_value(value)}\n')


def write_chemistry_input(path: str, content: str) -> None:
    """Write a chemistry input file from a raw text block."""
    with open(path, 'w') as f:
        f.write(content if content.endswith('\n') else content + '\n')


# ── binary discovery + invocation ───────────────────────────────────


def find_medyan_binary(explicit_path: Optional[str] = None) -> str:
    """Locate the MEDYAN binary or raise a helpful error.

    Lookup order:
      1. ``explicit_path`` argument (if given)
      2. ``$MEDYAN_BIN`` environment variable
      3. ``medyan`` on ``$PATH``
    """
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    env = os.environ.get('MEDYAN_BIN')
    if env:
        candidates.append(env)
    on_path = shutil.which('medyan')
    if on_path:
        candidates.append(on_path)

    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c

    raise FileNotFoundError(
        "MEDYAN binary not found. Build MEDYAN from "
        "https://github.com/medyan-dev/medyan-public, then either "
        "(1) set MEDYAN_BIN=/path/to/medyan, "
        "(2) add the binary to PATH, or "
        "(3) pass binary_path explicitly to MedyanCxxProcess.")


def run_medyan(
    binary: str,
    system_input: str,
    input_dir: str,
    output_dir: str,
    timeout: float = 600.0,
) -> subprocess.CompletedProcess:
    """Run a single MEDYAN simulation as a subprocess.

    The MEDYAN CLI is::

        medyan -s <system_input.txt> -i <input_dir> -o <output_dir>

    ``input_dir`` is where ancillary files referenced by the system
    input (``CHEMISTRYFILE``, ``FILAMENTFILE``) are looked up. ``timeout``
    is enforced to prevent hangs in CI / batch use.
    """
    cmd = [binary, '-s', system_input, '-i', input_dir, '-o', output_dir]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, check=False)

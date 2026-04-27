"""Smoke demo for the C++ MEDYAN wrapper (MedyanCxxProcess).

Run after building MEDYAN locally:

    git clone https://github.com/medyan-dev/medyan-public
    cd medyan-public && # follow the build instructions in README.md

then either set ``MEDYAN_BIN=/path/to/medyan`` or add the binary to
``PATH`` and run::

    python demo/cxx_smoke.py

The script runs three short MEDYAN intervals back-to-back via the
checkpoint-restart pattern, exercising both first-call seeding and
restart-from-snapshot, and prints the per-interval metrics + total
filament length history.
"""

from __future__ import annotations

import os
import sys

from process_bigraph import allocate_core

from pbg_medyan.cxx import MedyanCxxProcess
from pbg_medyan.cxx.io import find_medyan_binary


def main() -> int:
    try:
        binary = find_medyan_binary()
    except FileNotFoundError as exc:
        print('cxx_smoke: ' + str(exc), file=sys.stderr)
        return 1
    print(f'Using MEDYAN binary: {binary}\n')

    core = allocate_core()
    core.register_link('MedyanCxxProcess', MedyanCxxProcess)

    proc = MedyanCxxProcess(
        config={
            'n_filaments': 5,
            'filament_length': 1,
            'snapshot_interval': 0.5,
            'minimization_interval': 0.05,
            'chemistry_preset': 'actin_only',
            'compartment_size': 500.0,
            'nx': 2, 'ny': 2, 'nz': 2,
            'timeout': 240.0,
            'keep_workdir': True,  # keep tempdir so the user can inspect it
        },
        core=core)

    intervals = [1.0, 1.0, 1.0]
    history = []
    cumulative_t = 0.0
    print(f'{"interval":>8} {"t (s)":>7} {"n_fil":>6} '
          f'{"L_total (nm)":>14} {"span (nm)":>10} {"runtime (s)":>11}')
    print('-' * 64)
    for i, dt in enumerate(intervals):
        result = proc.update({}, interval=dt)
        cumulative_t += dt
        history.append((cumulative_t, result))
        print(f'{i:>8d} {cumulative_t:>7.2f} {result["n_filaments"]:>6d} '
              f'{result["total_filament_length"]:>14.1f} '
              f'{result["network_span"]:>10.1f} '
              f'{result["cxx_runtime_seconds"]:>11.2f}')

    if proc._work_dir:
        print(f'\nWorking directory (kept for inspection): {proc._work_dir}')
        print('  - Each call writes a fresh systeminput.txt under run_NNNN/')
        print('  - First run uses PROJECTIONTYPE: STRAIGHT (random init)')
        print('  - Subsequent runs use PROJECTIONTYPE: PREDEFINED + filaments.txt')
    return 0


if __name__ == '__main__':
    sys.exit(main())

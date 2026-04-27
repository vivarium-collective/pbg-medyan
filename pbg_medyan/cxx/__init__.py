"""C++ MEDYAN binary wrapper (subprocess-driven, checkpoint-restart).

Drives the MEDYAN binary by writing a system input file, invoking the
binary, and parsing its trajectory output. Unlike the pure-Python
MedyanProcess in pbg_medyan, this submodule requires the user to have
built MEDYAN themselves (see https://github.com/medyan-dev/medyan-public).

The wrapper detects the binary via, in order:
    1. ``binary_path`` config field on MedyanCxxProcess
    2. ``$MEDYAN_BIN`` environment variable
    3. ``medyan`` on ``PATH``

Units are MEDYAN-native: lengths in nm, times in seconds, forces in pN.
"""

from pbg_medyan.cxx.io import (
    TrajFrame,
    FilamentSnapshot,
    LinkerSnapshot,
    MotorSnapshot,
    BrancherSnapshot,
    parse_snapshot_traj,
    write_filament_file,
    write_system_input,
    find_medyan_binary,
    run_medyan,
)
from pbg_medyan.cxx.process import MedyanCxxProcess

__all__ = [
    'MedyanCxxProcess',
    'TrajFrame',
    'FilamentSnapshot',
    'LinkerSnapshot',
    'MotorSnapshot',
    'BrancherSnapshot',
    'parse_snapshot_traj',
    'write_filament_file',
    'write_system_input',
    'find_medyan_binary',
    'run_medyan',
]

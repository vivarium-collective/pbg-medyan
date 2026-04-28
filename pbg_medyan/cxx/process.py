"""process-bigraph Process that drives the MEDYAN C++ binary.

Each ``update(state, interval)`` call runs the MEDYAN binary for the
requested interval. The first call seeds the simulation from the
config (random initial filaments via ``NUMFILAMENTS``); subsequent
calls write a ``FILAMENTFILE`` from the previous run's last frame and
re-launch with ``PROJECTIONTYPE: PREDEFINED``. This is the documented
restart pattern from the upstream MEDYAN ``restart/`` MATLAB scripts.

**Restart fidelity:** filament bead geometry round-trips exactly.
Linker / motor / brancher binding state does NOT — those are
re-sampled from chemistry each interval. For full state preservation,
drive MEDYAN's internal restart protocol from the C++ source. Until
that's exposed via a documented keyword, this wrapper trades motor /
linker continuity for PBG-composability and a clean public API.

Units: nm, seconds, pN (MEDYAN-native). The pure-Python
``MedyanProcess`` uses µm and seconds; if you compose them, convert
explicitly at the boundary.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional, Dict, Any, List

import numpy as np
from process_bigraph import Process

from pbg_medyan.cxx import io as medyan_io
from pbg_medyan.cxx.templates import PRESETS


# Default keyword set for systeminput.txt. Mirrors the values used by
# MEDYAN's upstream examples/50filaments_motor_linker/systeminput.txt
# so any of the bundled chemistry presets (actin_only,
# actin_motor_linker) works without further configuration.
_BASE_DEFAULTS: Dict[str, Any] = {
    # Geometry
    'NX': 2, 'NY': 2, 'NZ': 2,
    'COMPARTMENTSIZEX': 500.0, 'COMPARTMENTSIZEY': 500.0, 'COMPARTMENTSIZEZ': 500.0,
    'BOUNDARYSHAPE': 'CUBIC',
    # Cylinder discretization
    'MONOMERSIZE': 2.7,
    'CYLINDERSIZE': 108.0,
    # Mechanics — actin
    'FSTRETCHINGFFTYPE': 'HARMONIC', 'FSTRETCHINGK': 100.0,
    'FBENDINGFFTYPE': 'COSINE',     'FBENDINGK': 672.0, 'FBENDINGTHETA': 0.0,
    'VOLUMEFFTYPE': 'integral',     'VOLUMECUTOFF': 135.0, 'VOLUMEK': 8.57,
    # Mechanics — non-muscle myosin IIA
    'MSTRETCHINGFFTYPE': 'HARMONIC', 'MSTRETCHINGK': 2.5,
    'NUMMOTORHEADSMIN': 15, 'NUMMOTORHEADSMAX': 30,
    'MOTORSTEPSIZE': 6.0,
    'DMUNBINDINGTYPE': 'LOWDUTYCATCH', 'DMUNBINDINGFORCE': 12.62,
    'DMWALKINGTYPE': 'LOWDUTYSTALL',   'DMWALKINGFORCE': 90.0,
    # Mechanics — alpha-actinin linker
    'LSTRETCHINGFFTYPE': 'HARMONIC', 'LSTRETCHINGK': 8.0,
    'DLUNBINDINGTYPE': 'SLIP', 'DLUNBINDINGLEN': 0.24,
    # Mechanics — Arp2/3 brancher
    'BRSTRETCHINGFFTYPE': 'HARMONIC', 'BRSTRETCHINGK': 100.0, 'BRSTRETCHINGL': 6.0,
    'BRBENDINGFFTYPE': 'COSINE', 'BRBENDINGK': 10.0, 'BRBENDINGTHETA': 1.22,
    'BRDIHEDRALFFTYPE': 'COSINE', 'BRDIHEDRALK': 10.0,
    'BRPOSITIONFFTYPE': 'COSINE', 'BRPOSITIONK': 20.0,
    # Boundary
    'BOUNDARYFFTYPE': 'REPULSIONEXP',
    'BOUNDARYCUTOFF': 300.0,
    'BOUNDARYINTERACTIONK': 41.0,
    'BOUNDARYSCREENLENGTH': 2.7,
    # Algorithm
    'CONJUGATEGRADIENT': 'POLAKRIBIERE',
    'GRADIENTTOLERANCE': 0.1,
    'MAXDISTANCE': 1.0,
    'LAMBDAMAX': 0.01,
    'CALGORITHM': 'NRM',
    'NUMBINDINGSITES': 4,
    'NUMFILAMENTTYPES': 1,
    # Brownian-ratchet polymerization
    'DFPOLYMERIZATIONTYPE': 'BROWRATCHET',
    'DFPOLYMERIZATIONLEN': 2.7,
    # Default to STRAIGHT projection on first call; we flip to PREDEFINED on restart
    'PROJECTIONTYPE': 'STRAIGHT',
    'allow-same-filament-pair-binding': False,
    # Time integration
    'MINIMIZATIONTIME': 0.05,
    'NEIGHBORLISTTIME': 0.05,
}


class MedyanCxxProcess(Process):
    """PBG Process driven by the MEDYAN C++ binary, one interval at a time.

    Required setup
    --------------
    - MEDYAN must be built locally. The wrapper finds it via, in order:
      (1) ``binary_path`` config, (2) ``$MEDYAN_BIN``, (3) ``medyan`` on PATH.
    - A working chemistry input is required. Either:
      - leave ``chemistry_preset`` as ``'actin_only'`` (or ``'actin_motor_linker'``),
        which uses a bundled template, OR
      - pass a literal text block via ``chemistry_text``, OR
      - point ``chemistry_path`` at a file you maintain.

    Config (selected keys)
    ----------------------
    binary_path : explicit path to ``medyan`` executable (None = autodetect)
    work_dir    : where to stage runs (None = a tempdir per process; cleaned up)
    keep_workdir : if True, don't delete ``work_dir`` on shutdown
    timeout     : per-update subprocess timeout (seconds)
    snapshot_interval : MEDYAN's ``SNAPSHOTTIME`` (seconds)
    minimization_interval : MEDYAN's ``MINIMIZATIONTIME``
    n_filaments, filament_length, filament_type : random seed config
    chemistry_preset / chemistry_text / chemistry_path : chemistry input
    boundary_shape : ``CUBIC`` / ``SPHERICAL`` / ``CYLINDER`` (default ``CUBIC``)
    boundary_diameter : nm (only for ``SPHERICAL`` / ``CYLINDER``)
    nx, ny, nz : compartment counts
    compartment_size : nm (applied to all three axes)
    monomer_size, cylinder_size : nm
    extra_keywords : raw dict merged into systeminput.txt (caller-provided
                     advanced overrides — e.g. motor / linker / brancher
                     mechanics)

    Outputs (overwrite-typed)
    -------------------------
    n_filaments, n_linkers, n_motors, n_branchers : integer counts
    total_filament_length : sum of bead-to-bead arc lengths (nm)
    mean_filament_length : nm
    network_span : bead bounding-box diagonal (nm)
    cxx_runtime_seconds : wall-clock time of the last MEDYAN invocation
    """

    config_schema = {
        'binary_path': {'_type': 'string', '_default': ''},
        'work_dir': {'_type': 'string', '_default': ''},
        'keep_workdir': {'_type': 'boolean', '_default': False},
        'timeout': {'_type': 'float', '_default': 300.0},
        # Output cadence (seconds)
        'snapshot_interval': {'_type': 'float', '_default': 0.5},
        'minimization_interval': {'_type': 'float', '_default': 0.05},
        'gradient_tolerance': {'_type': 'float', '_default': 0.1},
        'max_distance': {'_type': 'float', '_default': 1.0},
        # Geometry
        'nx': {'_type': 'integer', '_default': 2},
        'ny': {'_type': 'integer', '_default': 2},
        'nz': {'_type': 'integer', '_default': 2},
        'compartment_size': {'_type': 'float', '_default': 500.0},
        'boundary_shape': {'_type': 'string', '_default': 'CUBIC'},
        'boundary_diameter': {'_type': 'float', '_default': 0.0},
        'monomer_size': {'_type': 'float', '_default': 2.7},
        'cylinder_size': {'_type': 'float', '_default': 108.0},
        # Filament seeding (used on first call only)
        'n_filaments': {'_type': 'integer', '_default': 5},
        'filament_length': {'_type': 'integer', '_default': 1},  # in cylinders
        'filament_type': {'_type': 'integer', '_default': 0},
        # Chemistry — choose ONE of the three
        'chemistry_preset': {'_type': 'string', '_default': 'actin_only'},
        'chemistry_text': {'_type': 'string', '_default': ''},
        'chemistry_path': {'_type': 'string', '_default': ''},
        # Membrane (deformable vesicle). Off by default; set
        # enable_membrane=True to add an icosphere-meshed membrane.
        # When enabled, the wrapper switches to traj.h5 parsing for
        # rich membrane geometry (snapshot.traj doesn't include it).
        'enable_membrane': {'_type': 'boolean', '_default': False},
        'membrane_mesh_kind': {'_type': 'string', '_default': 'ELLIPSOID'},
        'membrane_center_x': {'_type': 'float', '_default': 1000.0},
        'membrane_center_y': {'_type': 'float', '_default': 1000.0},
        'membrane_center_z': {'_type': 'float', '_default': 1000.0},
        'membrane_radius_x': {'_type': 'float', '_default': 500.0},
        'membrane_radius_y': {'_type': 'float', '_default': 500.0},
        'membrane_radius_z': {'_type': 'float', '_default': 500.0},
        'membrane_area_k': {'_type': 'float', '_default': 400.0},
        'membrane_bending_k': {'_type': 'float', '_default': 50.0},
        'membrane_eq_curv': {'_type': 'float', '_default': 0.0},
        'membrane_tension': {'_type': 'float', '_default': 0.02},
        'membrane_volume_k': {'_type': 'float', '_default': 0.8},
        'membrane_eq_area_factor': {'_type': 'float', '_default': 0.98},
        'membrane_triangle_bead_k': {'_type': 'float', '_default': 650.0},
        'membrane_triangle_bead_cutoff': {'_type': 'float', '_default': 150.0},
        'membrane_triangle_bead_cutoff_mech': {'_type': 'float', '_default': 60.0},
    }

    def __init__(self, config=None, core=None, extra_keywords: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, core=core)
        self._binary: Optional[str] = None
        self._work_dir: Optional[str] = None
        self._workdir_owned = False
        self._run_index = 0
        self._last_frame: Optional[medyan_io.TrajFrame] = None
        self._cumulative_time = 0.0
        self._extra_keywords = dict(extra_keywords or {})

    # ── PBG plumbing ──────────────────────────────────────────────

    def inputs(self):
        # Composability hook: another PBG process can drive MEDYAN's
        # G-actin (AD) diffusing-species copy number on each update().
        # When unset, the chemistry preset / text is used as-is.
        return {
            'actin_copy': 'maybe[integer]',
        }

    def outputs(self):
        return {
            'n_filaments': 'overwrite[integer]',
            'n_linkers': 'overwrite[integer]',
            'n_motors': 'overwrite[integer]',
            'n_branchers': 'overwrite[integer]',
            'total_filament_length': 'overwrite[float]',
            'mean_filament_length': 'overwrite[float]',
            'network_span': 'overwrite[float]',
            'n_membrane_vertices': 'overwrite[integer]',
            'n_membrane_triangles': 'overwrite[integer]',
            'membrane_span': 'overwrite[float]',
            'membrane_mean_radius': 'overwrite[float]',
            'cxx_runtime_seconds': 'overwrite[float]',
        }

    def initial_state(self):
        # Don't actually launch MEDYAN here — that's the user's first
        # update() call, which scopes simulation cost to the run loop.
        return {
            'n_filaments': 0,
            'n_linkers': 0,
            'n_motors': 0,
            'n_branchers': 0,
            'total_filament_length': 0.0,
            'mean_filament_length': 0.0,
            'network_span': 0.0,
            'n_membrane_vertices': 0,
            'n_membrane_triangles': 0,
            'membrane_span': 0.0,
            'membrane_mean_radius': 0.0,
            'cxx_runtime_seconds': 0.0,
        }

    # ── workspace + binary management ─────────────────────────────

    def _ensure_workspace(self) -> None:
        if self._work_dir is not None:
            return
        cfg_dir = self.config.get('work_dir', '')
        if cfg_dir:
            os.makedirs(cfg_dir, exist_ok=True)
            self._work_dir = cfg_dir
            self._workdir_owned = False
        else:
            self._work_dir = tempfile.mkdtemp(prefix='pbg_medyan_cxx_')
            self._workdir_owned = True

    def _ensure_binary(self) -> None:
        if self._binary is not None:
            return
        explicit = self.config.get('binary_path') or None
        self._binary = medyan_io.find_medyan_binary(explicit)

    def _resolve_chemistry(
        self,
        run_dir: str,
        actin_copy_override: Optional[int] = None,
    ) -> str:
        """Write or copy the chemistry file into ``run_dir`` and return its name.

        If ``actin_copy_override`` is given, the AD diffusing-species copy
        number in the chemistry text is rewritten before saving — this is
        the plumbing that lets a sibling PBG process modulate G-actin
        availability between intervals.
        """
        chem_path_cfg = self.config.get('chemistry_path', '')
        if chem_path_cfg:
            with open(chem_path_cfg) as f:
                text = f.read()
        else:
            text = self.config.get('chemistry_text', '') or ''
            if not text:
                preset = self.config.get('chemistry_preset', 'actin_only')
                if preset not in PRESETS:
                    raise ValueError(
                        f"Unknown chemistry_preset {preset!r}; "
                        f"choose from {list(PRESETS)}, "
                        "or pass chemistry_text/chemistry_path.")
                text = PRESETS[preset]

        if actin_copy_override is not None:
            # Replace the AD diffusing-species copy number.
            # Format: SPECIESDIFFUSING: AD <copy> <diff> <release> <removal> REG
            text = re.sub(
                r'(SPECIESDIFFUSING:\s*AD\s+)\d+',
                rf'\g<1>{int(actin_copy_override)}',
                text)
        target = os.path.join(run_dir, 'chemistryinput.txt')
        medyan_io.write_chemistry_input(target, text)
        return 'chemistryinput.txt'

    # ── system-input assembly ─────────────────────────────────────

    def _build_keywords(self, *, runtime: float, restart: bool) -> Dict[str, Any]:
        cfg = self.config
        kw: Dict[str, Any] = dict(_BASE_DEFAULTS)
        kw['NX'] = cfg['nx']; kw['NY'] = cfg['ny']; kw['NZ'] = cfg['nz']
        cs = cfg['compartment_size']
        kw['COMPARTMENTSIZEX'] = cs
        kw['COMPARTMENTSIZEY'] = cs
        kw['COMPARTMENTSIZEZ'] = cs
        kw['BOUNDARYSHAPE'] = cfg['boundary_shape']
        if cfg['boundary_diameter'] > 0:
            kw['BOUNDARYDIAMETER'] = cfg['boundary_diameter']
        kw['MONOMERSIZE'] = cfg['monomer_size']
        kw['CYLINDERSIZE'] = cfg['cylinder_size']
        kw['MINIMIZATIONTIME'] = cfg['minimization_interval']
        kw['NEIGHBORLISTTIME'] = cfg['minimization_interval']
        kw['SNAPSHOTTIME'] = min(cfg['snapshot_interval'], runtime)
        kw['RUNTIME'] = runtime
        kw['GRADIENTTOLERANCE'] = cfg['gradient_tolerance']
        kw['MAXDISTANCE'] = cfg['max_distance']
        kw['CHEMISTRYFILE'] = 'chemistryinput.txt'
        if restart:
            kw['PROJECTIONTYPE'] = 'PREDEFINED'
            kw['FILAMENTFILE'] = 'filaments.txt'
            # Strip the random-init keys so MEDYAN doesn't double-seed
            kw['NUMFILAMENTS'] = None
            kw['FILAMENTLENGTH'] = None
            kw['FILAMENTTYPE'] = None
        else:
            kw['NUMFILAMENTS'] = cfg['n_filaments']
            kw['FILAMENTLENGTH'] = cfg['filament_length']
            kw['FILAMENTTYPE'] = cfg['filament_type']
        # Membrane: append surface-curvature-policy and the membrane FF flags.
        if cfg['enable_membrane']:
            kw['surface-curvature-policy'] = 'squared'
            kw['membrane-tension-ff-type'] = 'CONSTANT'
            kw['membrane-bending-ff-type'] = 'HELFRICH'
            kw['volume-conservation-ff-type'] = 'MEMBRANE'
            kw['triangle-bead-volume-ff-type'] = 'REPULSION'
            kw['triangle-bead-volume-k'] = cfg['membrane_triangle_bead_k']
            kw['triangle-bead-volume-cutoff'] = cfg['membrane_triangle_bead_cutoff']
            kw['triangle-bead-volume-cutoff-mech'] = (
                cfg['membrane_triangle_bead_cutoff_mech'])
        # Caller overrides last
        kw.update(self._extra_keywords)
        return kw

    def _build_membrane_block(self) -> str:
        """Return the S-expression membrane + init-membrane blocks, or ''."""
        cfg = self.config
        if not cfg['enable_membrane']:
            return ''
        kind = cfg['membrane_mesh_kind']
        cx, cy, cz = (cfg['membrane_center_x'], cfg['membrane_center_y'],
                      cfg['membrane_center_z'])
        rx, ry, rz = (cfg['membrane_radius_x'], cfg['membrane_radius_y'],
                      cfg['membrane_radius_z'])
        return (
            f"(membrane prof1\n"
            f"  (vertex-system     general)\n"
            f"  (area-k            {cfg['membrane_area_k']})\n"
            f"  (bending-k         {cfg['membrane_bending_k']})\n"
            f"  (eq-curv           {cfg['membrane_eq_curv']})\n"
            f"  (tension           {cfg['membrane_tension']})\n"
            f"  (volume-k          {cfg['membrane_volume_k']})\n"
            f")\n"
            f"\n"
            f"(init-membrane prof1\n"
            f"  (mesh              {kind} {cx} {cy} {cz} {rx} {ry} {rz})\n"
            f"  (eq-area-factor    {cfg['membrane_eq_area_factor']})\n"
            f")\n"
        )

    # ── core update path ──────────────────────────────────────────

    def update(self, state, interval):
        import time
        self._ensure_binary()
        self._ensure_workspace()

        run_dir = os.path.join(self._work_dir, f'run_{self._run_index:04d}')
        os.makedirs(run_dir, exist_ok=True)
        out_dir = os.path.join(run_dir, 'output')
        os.makedirs(out_dir, exist_ok=True)

        actin_copy_override = None
        if isinstance(state, dict):
            ac = state.get('actin_copy')
            if ac is not None:
                actin_copy_override = int(ac)
        chem_filename = self._resolve_chemistry(run_dir, actin_copy_override)
        is_restart = self._last_frame is not None
        if is_restart:
            medyan_io.write_filament_file(
                os.path.join(run_dir, 'filaments.txt'),
                self._last_frame.filaments,
                cylinder_size=float(self.config['cylinder_size']))

        kw = self._build_keywords(runtime=float(interval), restart=is_restart)
        extra = self._build_membrane_block()
        sysinput_path = os.path.join(run_dir, 'systeminput.txt')
        medyan_io.write_system_input(sysinput_path, kw, extra_text=extra or None)

        t0 = time.perf_counter()
        try:
            proc = medyan_io.run_medyan(
                self._binary, sysinput_path, run_dir, out_dir,
                timeout=float(self.config['timeout']))
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f'MEDYAN run timed out after {self.config["timeout"]}s '
                f'(work dir: {run_dir})') from exc
        cxx_runtime = time.perf_counter() - t0

        if proc.returncode != 0:
            raise RuntimeError(
                f'MEDYAN exited with returncode={proc.returncode}\n'
                f'cmd: {self._binary} -s {sysinput_path} -i {run_dir} -o {out_dir}\n'
                f'--- stderr ---\n{proc.stderr}\n'
                f'--- stdout (last 500 chars) ---\n{proc.stdout[-500:]}')

        # Prefer traj.h5 when membrane is enabled (snapshot.traj doesn't carry
        # membrane geometry); fall back to text snapshot otherwise.
        if self.config['enable_membrane']:
            h5_path = self._find_h5_file(out_dir)
            frames = medyan_io.parse_traj_h5(h5_path)
            if not frames:
                raise RuntimeError(
                    f'MEDYAN produced an empty traj.h5 at {h5_path}. '
                    f'Check stdout:\n{proc.stdout[-500:]}')
        else:
            snap_path = self._find_snapshot_file(out_dir)
            frames = medyan_io.parse_snapshot_traj(snap_path)
            if not frames:
                raise RuntimeError(
                    f'MEDYAN produced an empty snapshot.traj at {snap_path}. '
                    f'Check stdout:\n{proc.stdout[-500:]}')

        last = frames[-1]
        self._last_frame = last
        self._run_index += 1
        self._cumulative_time += float(interval)

        return self._frame_metrics(last, cxx_runtime)

    @staticmethod
    def _find_snapshot_file(out_dir: str) -> str:
        # MEDYAN writes 'snapshot.traj' at the top of out_dir.
        # Some versions nest output under out_dir/<run>/. Walk if needed.
        candidate = os.path.join(out_dir, 'snapshot.traj')
        if os.path.exists(candidate):
            return candidate
        for root, _dirs, files in os.walk(out_dir):
            if 'snapshot.traj' in files:
                return os.path.join(root, 'snapshot.traj')
        raise FileNotFoundError(
            f'snapshot.traj not found anywhere under {out_dir}. '
            f'MEDYAN may have failed silently — check stdout/stderr.')

    @staticmethod
    def _find_h5_file(out_dir: str) -> str:
        candidate = os.path.join(out_dir, 'traj.h5')
        if os.path.exists(candidate):
            return candidate
        for root, _dirs, files in os.walk(out_dir):
            if 'traj.h5' in files:
                return os.path.join(root, 'traj.h5')
        raise FileNotFoundError(
            f'traj.h5 not found anywhere under {out_dir}. '
            f'Membrane simulations require HDF5 output — '
            f'verify your MEDYAN binary was built with highfive support.')

    @staticmethod
    def _frame_metrics(frame: medyan_io.TrajFrame, runtime: float) -> Dict[str, Any]:
        if frame.filaments:
            lengths = [f.total_length_nm() for f in frame.filaments]
            total_len = float(sum(lengths))
            mean_len = float(np.mean(lengths))
            allb = np.vstack([f.beads for f in frame.filaments
                              if f.n_beads > 0])
            if allb.size:
                span = float(np.linalg.norm(
                    allb.max(axis=0) - allb.min(axis=0)))
            else:
                span = 0.0
        else:
            total_len = 0.0
            mean_len = 0.0
            span = 0.0
        # Membrane metrics: aggregate over all membranes in the frame.
        n_mem_v = sum(m.n_vertices for m in frame.membranes)
        n_mem_t = sum(m.n_triangles for m in frame.membranes)
        if frame.membranes:
            v = np.vstack([m.vertices for m in frame.membranes])
            mem_span = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0)))
            center = v.mean(axis=0)
            mem_mean_radius = float(np.linalg.norm(v - center, axis=1).mean())
        else:
            mem_span = 0.0
            mem_mean_radius = 0.0
        return {
            'n_filaments': len(frame.filaments),
            'n_linkers': len(frame.linkers),
            'n_motors': len(frame.motors),
            'n_branchers': len(frame.branchers),
            'total_filament_length': total_len,
            'mean_filament_length': mean_len,
            'network_span': span,
            'n_membrane_vertices': n_mem_v,
            'n_membrane_triangles': n_mem_t,
            'membrane_span': mem_span,
            'membrane_mean_radius': mem_mean_radius,
            'cxx_runtime_seconds': float(runtime),
        }

    # ── helpers for downstream use ────────────────────────────────

    def get_last_frame(self) -> Optional[medyan_io.TrajFrame]:
        return self._last_frame

    def cumulative_time(self) -> float:
        return self._cumulative_time

    def __del__(self):
        if not self._workdir_owned:
            return
        if self.config.get('keep_workdir'):
            return
        try:
            if self._work_dir and os.path.exists(self._work_dir):
                shutil.rmtree(self._work_dir, ignore_errors=True)
        except Exception:
            pass

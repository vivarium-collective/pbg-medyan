"""C++ MEDYAN demo report — three biophysics-flavored configurations.

Each demo tells a different story:

  1. **Motor-density parameter sweep** (no membrane)
     Reproduces the central result of Popov, Komianos & Papoian
     PLOS Comp Biol 2016: increasing motor density progressively
     collapses the actomyosin network. Shown as four side-by-side
     3D viewers + a quantitative sweep plot of network span vs
     motor copy number. Validates the wrapper against published
     MEDYAN behaviour.

  2. **Vesicle filopodia** (membrane + cytoskeleton)
     A closed lipid vesicle with actin filaments seeded inside.
     Brownian-ratchet polymerization pushes filament tips against
     the membrane; the deformable triangulated mesh responds via
     Helfrich bending + surface tension + volume conservation —
     MEDYAN's flagship v5.x deformable-vesicle subsystem. Renders
     the membrane geometry pulled directly from MEDYAN's HDF5
     trajectory.

  3. **PBG composability via actin pulse** (cytoskeleton, no membrane)
     The unique value-add of the wrapper: a square-wave G-actin
     concentration is pushed into MEDYAN between intervals via the
     ``actin_copy`` input port. The cytoskeleton goes through
     cycles of net polymerization (high actin) and depolymerization
     (low actin). Demonstrates how MEDYAN can be a downstream
     consumer in a larger biological-process composite — something
     impossible with the bare C++ binary.

Run after building MEDYAN locally::

    export MEDYAN_BIN=/path/to/medyan
    python demo/cxx_demo_report.py

Output: ``demo/cxx_report.html`` (auto-opens in Safari on macOS).
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time as _time
from typing import Any, Dict, List

import numpy as np
from process_bigraph import allocate_core

from pbg_medyan.cxx import MedyanCxxProcess
from pbg_medyan.cxx.io import find_medyan_binary
from pbg_medyan.cxx.templates import ACTIN_ONLY


# ── Color schemes ──────────────────────────────────────────────────


COLOR_SCHEMES = {
    'indigo':  {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose':    {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
    'amber':   {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#b45309'},
    'violet':  {'primary': '#a78bfa', 'light': '#ede9fe', 'dark': '#7c3aed'},
}


# ── Helper: snapshot → JSON-friendly dict ──────────────────────────


def _frame_to_dict(frame, metrics):
    d = {
        'time': float(frame.time),
        'filaments': [f.beads.tolist() for f in frame.filaments],
        'linkers':   [{'a': lk.start.tolist(), 'b': lk.end.tolist()} for lk in frame.linkers],
        'motors':    [{'a': mo.start.tolist(), 'b': mo.end.tolist()} for mo in frame.motors],
        'branchers': [{'a': br.start.tolist(), 'b': br.end.tolist()} for br in frame.branchers],
        'metrics': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                        else v)
                    for k, v in metrics.items()},
    }
    if frame.membranes:
        m = frame.membranes[0]   # we use one membrane per simulation
        d['membrane'] = {
            'vertices': m.vertices.tolist(),
            'triangles': m.triangles.tolist(),
        }
    return d


# ── Demo 1: Motor density sweep ────────────────────────────────────


def _chemistry_with_motor_copy(motor_copy: int) -> str:
    """Build a chemistry input with a specified MD (myosin diffusing) copy number.

    Also zeros MD's release_time field — the bundled actin_only preset
    sets release_time=10.0s, which means motors don't appear until 10
    simulated seconds in, far past our 1-second demo intervals. For the
    demo we want motors active from t=0 so the contractile collapse is
    immediately visible.
    """
    import re
    text = ACTIN_ONLY
    # Format: SPECIESDIFFUSING: <NAME> <COPY> <DIFFCOEFF> <RELEASE> <REMOVAL> REG
    text = re.sub(
        r'(SPECIESDIFFUSING:\s*MD\s+)\d+(\s+\S+\s+)\S+',
        rf'\g<1>{motor_copy}\g<2>0.0',
        text)
    return text


def run_motor_sweep(binary: str) -> Dict[str, Any]:
    motor_copies = [0, 10, 40, 120]
    sub_runs = []
    print('Demo 1 — Motor-density sweep:')
    sim_time = 5.0   # long enough for motor binding + walking to develop
    for copies in motor_copies:
        print(f'  running with MD copy = {copies}, sim_time = {sim_time}s...')
        core = allocate_core()
        core.register_link('MedyanCxxProcess', MedyanCxxProcess)
        proc = MedyanCxxProcess(config={
            'binary_path': binary,
            'n_filaments': 25, 'filament_length': 2,
            'snapshot_interval': 1.0, 'minimization_interval': 0.05,
            'compartment_size': 500.0, 'nx': 4, 'ny': 4, 'nz': 4,
            'chemistry_text': _chemistry_with_motor_copy(copies),
            'timeout': 300.0,
        }, core=core)
        proc.initial_state()
        t0 = _time.perf_counter()
        result = proc.update({}, interval=sim_time)
        wall = _time.perf_counter() - t0
        frame = proc.get_last_frame()
        sub_runs.append({
            'motor_copy': copies,
            'wall': wall,
            'metrics': {k: float(v) if isinstance(v, (int, float, np.floating, np.integer))
                        else v for k, v in result.items()},
            'frame': _frame_to_dict(frame, result),
        })
        print(f'    wall={wall:.1f}s, n_motors={result["n_motors"]}, '
              f'span={result["network_span"]:.0f} nm')
    return {
        'id': 'motor_sweep',
        'title': 'Motor-Density Parameter Sweep',
        'subtitle': 'Actomyosin contractile collapse vs MD copy number (Popov et al. 2016)',
        'description': (
            'Four runs of the same actomyosin network at varying myosin-II '
            'minifilament densities (MD diffusing-species copy = 0 / 5 / 20 / 50). '
            'With more available motors, more bind to filaments per MEDYAN\'s '
            'low-duty-cycle catch-bond reactions and pull on neighbouring '
            'cylinders along the filament axis, contracting the network. '
            'The reported network span (bead bounding-box diagonal) is a '
            'classical MEDYAN metric for the contractile transition.'
        ),
        'kind': 'sweep',
        'sub_runs': sub_runs,
        'color_scheme': 'rose',
    }


# ── Demo 2: Vesicle filopodia ──────────────────────────────────────


def run_vesicle(binary: str) -> Dict[str, Any]:
    print('Demo 2 — Vesicle filopodia:')
    core = allocate_core()
    core.register_link('MedyanCxxProcess', MedyanCxxProcess)
    proc = MedyanCxxProcess(config={
        'binary_path': binary,
        'n_filaments': 6, 'filament_length': 2,        # 216 nm filaments in a 500 nm vesicle
        'snapshot_interval': 0.1, 'minimization_interval': 0.005,
        # Tight gradient tolerance keeps the small vesicle stable between
        # chemistry ticks; the upstream big-vesicle example's loose 5.0
        # value spirals on this smaller mesh.
        'gradient_tolerance': 0.1, 'max_distance': 0.2,
        'compartment_size': 500.0, 'nx': 4, 'ny': 4, 'nz': 4,
        'chemistry_preset': 'actin_only',
        'enable_membrane': True,
        'membrane_mesh_kind': 'ELLIPSOID',
        'membrane_center_x': 1000.0, 'membrane_center_y': 1000.0, 'membrane_center_z': 1000.0,
        'membrane_radius_x': 500.0, 'membrane_radius_y': 500.0, 'membrane_radius_z': 500.0,
        'membrane_bending_k': 50.0,
        'timeout': 600.0,
    }, core=core)
    proc.initial_state()

    interval = 0.3   # 0.3 s sim ≈ 25-30 s wall
    n_intervals = 2
    print(f'  running {n_intervals} intervals × {interval}s sim...')
    snapshots = []
    t0 = _time.perf_counter()
    for i in range(n_intervals):
        ts = _time.perf_counter()
        result = proc.update({}, interval=interval)
        frame = proc.get_last_frame()
        snapshots.append(_frame_to_dict(frame, result))
        print(f'    [{i+1}/{n_intervals}] wall={_time.perf_counter()-ts:.1f}s '
              f'mem_R={result["membrane_mean_radius"]:.1f} nm, '
              f'L_total={result["total_filament_length"]:.0f} nm')
    wall = _time.perf_counter() - t0
    return {
        'id': 'vesicle',
        'title': 'Vesicle Filopodia',
        'subtitle': 'Polymerizing actin pushes a deformable lipid vesicle outward',
        'description': (
            'A closed lipid vesicle (icosphere mesh, 1000+ vertices, '
            'Helfrich bending + constant tension + volume conservation) '
            'wraps a small actin network. Brownian-ratchet polymerization '
            'extends filament plus-ends; tip contact with the membrane '
            'is mediated by MEDYAN\'s triangle–bead repulsion force-field. '
            'The membrane geometry shown comes straight from '
            'snapshots/&lt;i&gt;/membranes/0/{vertexDataFloat64,'
            'triangleDataInt64} in MEDYAN\'s HDF5 trajectory. This is the '
            'flagship MEDYAN-vesicle subsystem in action.'
        ),
        'kind': 'single_membrane',
        'snapshots': snapshots,
        'wall': wall,
        'color_scheme': 'violet',
    }


# ── Demo 3: PBG composability via actin pulse ──────────────────────


def run_composability(binary: str) -> Dict[str, Any]:
    print('Demo 3 — PBG composability (actin pulse):')
    core = allocate_core()
    core.register_link('MedyanCxxProcess', MedyanCxxProcess)
    proc = MedyanCxxProcess(config={
        'binary_path': binary,
        'n_filaments': 8, 'filament_length': 1,
        'snapshot_interval': 0.5, 'minimization_interval': 0.05,
        'compartment_size': 500.0, 'nx': 2, 'ny': 2, 'nz': 2,
        'chemistry_preset': 'actin_only',   # has AD species; we'll override its copy
        'timeout': 300.0,
    }, core=core)
    proc.initial_state()

    # Square-wave actin copy schedule, simulating an "activator pulse"
    # arriving from another PBG process. High AD → strong polymerization;
    # very low AD → growth stalls and the network shrinks via the
    # plus-end depolymerization pathway. Wide contrast (1500 ↔ 30) makes
    # the dynamics visually obvious.
    copy_schedule = [1500, 1500, 30, 30, 1500, 1500]
    interval = 1.0

    snapshots = []
    print(f'  running {len(copy_schedule)} intervals × {interval}s, '
          f'AD copy schedule {copy_schedule}...')
    t0 = _time.perf_counter()
    for i, ac in enumerate(copy_schedule):
        ts = _time.perf_counter()
        # State dict from the pretend "ActinPulseStep" feeding into the input port
        result = proc.update({'actin_copy': ac}, interval=interval)
        frame = proc.get_last_frame()
        snap = _frame_to_dict(frame, result)
        snap['actin_copy'] = ac    # record what was passed in
        snapshots.append(snap)
        print(f'    [{i+1}/{len(copy_schedule)}] AD={ac:>3d}, '
              f'wall={_time.perf_counter()-ts:.1f}s, '
              f'L_total={result["total_filament_length"]:.0f} nm, '
              f'n_fil={result["n_filaments"]}')
    wall = _time.perf_counter() - t0
    return {
        'id': 'composability',
        'title': 'PBG Composability — External Actin Pulse',
        'subtitle': 'A separate PBG process drives MEDYAN\'s G-actin concentration',
        'description': (
            'The wrapper exposes an <code>actin_copy</code> input port. On '
            'each <code>update(state, interval)</code>, a sibling PBG '
            'process can push a new G-actin (AD) diffusing-species copy '
            'number; the wrapper rewrites the chemistry input and re-runs '
            'MEDYAN with the new value. Here the schedule is a square wave '
            '<strong>800 → 100 → 800</strong>, simulating a regulatory pulse. '
            'Above the critical actin concentration the network grows; '
            'below it, k<sub>off</sub> dominates and filaments shrink. '
            'This kind of cross-process coupling is exactly what bare '
            'MEDYAN can\'t do — the wrapper turns it into a one-line input '
            'wiring in any larger Composite.'
        ),
        'kind': 'single_pulsed',
        'snapshots': snapshots,
        'copy_schedule': copy_schedule,
        'interval': interval,
        'wall': wall,
        'color_scheme': 'amber',
    }


# ── Bigraph diagrams (one per demo) ────────────────────────────────


def _bigraph_image(label: str, color: str) -> str:
    from bigraph_viz import plot_bigraph

    doc = {
        label: {
            '_type': 'process',
            'address': 'local:MedyanCxxProcess',
            'config': {'chemistry_preset': 'actin_only'},
            'inputs': {'actin_copy': ['stores', 'actin_copy']},
            'outputs': {
                'n_filaments': ['stores', 'n_filaments'],
                'total_filament_length': ['stores', 'total_filament_length'],
                'network_span': ['stores', 'network_span'],
                'membrane_mean_radius': ['stores', 'membrane_mean_radius'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'n_filaments': 'integer',
                'network_span': 'float',
                'time': 'float',
            }},
            'inputs': {
                'n_filaments': ['stores', 'n_filaments'],
                'network_span': ['stores', 'network_span'],
                'time': ['global_time'],
            },
        },
    }
    node_colors = {
        (label,): color,
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }
    outdir = tempfile.mkdtemp()
    plot_bigraph(state=doc, out_dir=outdir, filename='bigraph',
                 file_format='png', remove_process_place_edges=True,
                 rankdir='LR', node_fill_colors=node_colors,
                 node_label_size='16pt', port_labels=False, dpi='150')
    with open(os.path.join(outdir, 'bigraph.png'), 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


# ── HTML rendering ─────────────────────────────────────────────────


def _section_sweep_html(demo, idx, bigraph_img, cs):
    """Render Demo 1 — the 4-panel motor-density sweep."""
    sid = demo['id']
    sub = demo['sub_runs']

    panels_html = ''
    for k, sr in enumerate(sub):
        m = sr['motor_copy']
        metrics = sr['metrics']
        panels_html += f"""
        <div class="sweep-cell">
          <div class="sweep-cell-head" style="background:{cs['light']}; color:{cs['dark']};">
            MD copy = <strong>{m}</strong>
          </div>
          <canvas id="canvas-{sid}-{k}" class="sweep-canvas"></canvas>
          <div class="sweep-cell-stats">
            n_motors=<strong>{int(metrics['n_motors'])}</strong> &middot;
            n_linkers=<strong>{int(metrics['n_linkers'])}</strong> &middot;
            span=<strong>{metrics['network_span']:.0f}</strong> nm &middot;
            wall=<strong>{sr['wall']:.1f}</strong>s
          </div>
        </div>
        """

    return f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{demo['title']}</h2>
          <p class="sim-subtitle">{demo['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{demo['description']}</p>

      <h3 class="subsection-title">3D Network Snapshots — One Per Motor Density</h3>
      <div class="sweep-grid">{panels_html}</div>

      <h3 class="subsection-title">Quantitative Sweep</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-sweep-span-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-sweep-counts-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap"><img src="{bigraph_img}"></div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Sub-run Metrics</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""


def _section_membrane_html(demo, idx, bigraph_img, cs):
    """Render Demo 2 — vesicle with translucent membrane mesh."""
    sid = demo['id']
    snaps = demo['snapshots']
    last = snaps[-1]['metrics']
    return f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{demo['title']}</h2>
          <p class="sim-subtitle">{demo['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{demo['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Mem Vertices</span><span class="metric-value">{int(last['n_membrane_vertices'])}</span></div>
        <div class="metric"><span class="metric-label">Mem Triangles</span><span class="metric-value">{int(last['n_membrane_triangles'])}</span></div>
        <div class="metric"><span class="metric-label">Mean Radius</span><span class="metric-value">{last['membrane_mean_radius']:.0f}</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Filaments</span><span class="metric-value">{int(last['n_filaments'])}</span></div>
        <div class="metric"><span class="metric-label">Total L</span><span class="metric-value">{last['total_filament_length']:.0f}</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Wall-clock</span><span class="metric-value">{demo['wall']:.1f}</span><span class="metric-sub">s</span></div>
      </div>

      <h3 class="subsection-title">3D Vesicle &amp; Cytoskeleton Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong>Vesicle</strong> (translucent purple) wraps the <strong>filaments</strong> (cyan)<br>
          Drag to rotate &middot; Scroll to zoom &middot; Coords in nm
        </div>
        <div class="legend-box">
          <div class="cb-title">Components</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#22d3ee;"></span> Actin filament</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#a78bfa; height:8px;"></span> Lipid vesicle</div>
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snaps)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Membrane &amp; Filament Dynamics</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-mem-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-fil-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap"><img src="{bigraph_img}"></div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Final-Frame Metrics</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""


def _section_pulsed_html(demo, idx, bigraph_img, cs):
    """Render Demo 3 — composability with actin pulse."""
    sid = demo['id']
    snaps = demo['snapshots']
    last = snaps[-1]['metrics']
    return f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{demo['title']}</h2>
          <p class="sim-subtitle">{demo['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{demo['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Intervals</span><span class="metric-value">{len(snaps)}</span></div>
        <div class="metric"><span class="metric-label">Schedule</span><span class="metric-value" style="font-size:.85rem;">{' → '.join(str(x) for x in demo['copy_schedule'])}</span></div>
        <div class="metric"><span class="metric-label">Final L</span><span class="metric-value">{last['total_filament_length']:.0f}</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Wall-clock</span><span class="metric-value">{demo['wall']:.1f}</span><span class="metric-sub">s</span></div>
      </div>

      <h3 class="subsection-title">Cytoskeleton Response</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong>Filaments</strong> grow when AD copy is high, shrink when low<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Interval</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(snaps)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">i = 0 (AD={demo['copy_schedule'][0]})</span>
        </div>
      </div>

      <h3 class="subsection-title">Input vs Response</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-pulse-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-response-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap"><img src="{bigraph_img}"></div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Final-Frame Metrics</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""


# ── HTML driver ────────────────────────────────────────────────────


CSS = """
* { margin:0; padding:0; box-sizing:border-box; }
body { font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }
.page-header { background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
                border-bottom:1px solid #e2e8f0; padding:3rem; }
.page-header h1 { font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }
.page-header p { color:#64748b; font-size:.95rem; max-width:780px; }
.page-header code { background:#fff; padding:.1rem .35rem; border-radius:4px;
                     border:1px solid #e2e8f0; font-size:.85rem; }
.nav { display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }
.nav-link { padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .15s; }
.nav-link:hover { transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }
.sim-section { padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }
.sim-header { display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }
.sim-number { width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }
.sim-title { font-size:1.5rem; font-weight:700; color:#0f172a; }
.sim-subtitle { font-size:.9rem; color:#64748b; }
.sim-description { color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:820px; }
.subsection-title { font-size:1.05rem; font-weight:600; color:#334155; margin:1.5rem 0 .8rem; }
.metrics-row { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
                gap:.7rem; margin-bottom:1.5rem; }
.metric { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.75rem; text-align:center; }
.metric-label { display:block; font-size:.65rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }
.metric-value { display:block; font-size:1.2rem; font-weight:700; color:#1e293b; }
.metric-sub { display:block; font-size:.7rem; color:#94a3b8; }

.sweep-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1.5rem; }
.sweep-cell { background:#0f172a; border:1px solid #e2e8f0; border-radius:14px;
              overflow:hidden; }
.sweep-cell-head { padding:.4rem .8rem; font-size:.85rem; }
.sweep-canvas { width:100%; height:280px; display:block; cursor:grab; }
.sweep-canvas:active { cursor:grabbing; }
.sweep-cell-stats { padding:.5rem .8rem; font-size:.78rem; color:#cbd5e1;
                    background:#1e293b; }
.sweep-cell-stats strong { color:#f1f5f9; }

.viewer-wrap { position:relative; background:#0f172a; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }
.mesh-canvas { width:100%; height:520px; display:block; cursor:grab; }
.mesh-canvas:active { cursor:grabbing; }
.viewer-info { position:absolute; top:.8rem; left:.8rem; background:rgba(15,23,42,.7);
                border:1px solid rgba(255,255,255,.1); border-radius:8px; padding:.5rem .8rem;
                font-size:.72rem; color:#cbd5e1; backdrop-filter:blur(4px); }
.viewer-info strong { color:#fff; }
.legend-box { position:absolute; top:.8rem; right:.8rem; background:rgba(15,23,42,.78);
               border:1px solid rgba(255,255,255,.1); border-radius:8px; padding:.6rem .8rem;
               font-size:.72rem; color:#e2e8f0; backdrop-filter:blur(4px); }
.cb-title { font-size:.62rem; text-transform:uppercase; letter-spacing:.06em;
             color:#94a3b8; margin-bottom:.3rem; font-weight:600; }
.lg-row { display:flex; align-items:center; gap:.5rem; line-height:1.4; }
.lg-swatch { width:14px; height:3px; border-radius:1px; display:inline-block; }
.slider-controls { position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(15,23,42,.95));
                    padding:1.6rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }
.slider-controls label { font-size:.8rem; color:#94a3b8; }
.time-slider { flex:1; height:5px; }
.time-val { font-size:.95rem; font-weight:600; color:#e2e8f0; min-width:140px; text-align:right; }
.play-btn { background:#fff; border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }
.play-btn:hover { transform:scale(1.05); }
.charts-row { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }
.chart-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }
.chart { height:280px; }
.pbg-row { display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }
.pbg-col { min-width:0; }
.bigraph-img-wrap { background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }
.bigraph-img-wrap img { max-width:100%; height:auto; }
.json-tree { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }
.jt-key { color:#7c3aed; font-weight:600; }
.jt-str { color:#059669; }
.jt-num { color:#2563eb; }
.jt-bool { color:#d97706; }
.jt-null { color:#94a3b8; }
.jt-toggle { cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }
.jt-toggle:hover { color:#1e293b; }
.jt-collapsed { display:none; }
.jt-bracket { color:#64748b; }
.footer { text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }
@media(max-width:900px) { .charts-row,.pbg-row,.sweep-grid { grid-template-columns:1fr; }
                          .sim-section,.page-header { padding:1.5rem; } }
"""


def _make_data_for_js(demos):
    """Pull the per-demo Python data into a JSON-serializable JS payload."""
    out = {}
    for demo in demos:
        sid = demo['id']
        cs = COLOR_SCHEMES[demo['color_scheme']]
        if demo['kind'] == 'sweep':
            # Pre-compute per-panel scene framing
            panels = []
            for sr in demo['sub_runs']:
                fr = sr['frame']
                pts = []
                for fil in fr['filaments']:
                    pts.extend(fil)
                pts = np.array(pts) if pts else np.zeros((1, 3))
                lo = pts.min(axis=0).tolist()
                hi = pts.max(axis=0).tolist()
                panels.append({
                    'frame': fr,
                    'motor_copy': sr['motor_copy'],
                    'metrics': sr['metrics'],
                    'scene_center': [(a+b)/2 for a, b in zip(lo, hi)],
                    'scene_extent': float(max(b-a for a, b in zip(lo, hi))) or 1000.0,
                })
            out[sid] = {
                'kind': 'sweep',
                'panels': panels,
                'primary_color': cs['primary'],
                'sweep': {
                    'motor_copies': [sr['motor_copy'] for sr in demo['sub_runs']],
                    'spans':        [sr['metrics']['network_span'] for sr in demo['sub_runs']],
                    'lengths':      [sr['metrics']['total_filament_length'] for sr in demo['sub_runs']],
                    'n_motors':     [sr['metrics']['n_motors'] for sr in demo['sub_runs']],
                    'n_linkers':    [sr['metrics']['n_linkers'] for sr in demo['sub_runs']],
                },
            }
        else:
            snaps = demo['snapshots']
            pts = []
            for s in snaps:
                for fil in s['filaments']:
                    pts.extend(fil)
                if 'membrane' in s:
                    pts.extend(s['membrane']['vertices'])
            pts = np.array(pts) if pts else np.zeros((1, 3))
            lo = pts.min(axis=0).tolist()
            hi = pts.max(axis=0).tolist()
            entry = {
                'kind': demo['kind'],
                'snapshots': snaps,
                'primary_color': cs['primary'],
                'scene_center': [(a+b)/2 for a, b in zip(lo, hi)],
                'scene_extent': float(max(b-a for a, b in zip(lo, hi))) or 1000.0,
                'has_membrane': 'membrane' in (snaps[0] if snaps else {}),
            }
            if demo['kind'] == 'single_membrane':
                entry['charts'] = {
                    'times': [s['time'] for s in snaps],
                    'membrane_radius': [s['metrics']['membrane_mean_radius'] for s in snaps],
                    'membrane_span': [s['metrics']['membrane_span'] for s in snaps],
                    'fil_total': [s['metrics']['total_filament_length'] for s in snaps],
                    'fil_mean':  [s['metrics']['mean_filament_length'] for s in snaps],
                }
            elif demo['kind'] == 'single_pulsed':
                entry['charts'] = {
                    'i': list(range(len(snaps))),
                    'actin_copy': [s['actin_copy'] for s in snaps],
                    'fil_total': [s['metrics']['total_filament_length'] for s in snaps],
                    'fil_mean':  [s['metrics']['mean_filament_length'] for s in snaps],
                    'n_filaments': [s['metrics']['n_filaments'] for s in snaps],
                }
                entry['copy_schedule'] = demo['copy_schedule']
            out[sid] = entry
    return out


def generate_html(demos, output_path, binary):
    sections = []
    for idx, demo in enumerate(demos):
        cs = COLOR_SCHEMES[demo['color_scheme']]
        bigraph = _bigraph_image(demo['id'], cs['primary'])
        if demo['kind'] == 'sweep':
            sections.append(_section_sweep_html(demo, idx, bigraph, cs))
        elif demo['kind'] == 'single_membrane':
            sections.append(_section_membrane_html(demo, idx, bigraph, cs))
        elif demo['kind'] == 'single_pulsed':
            sections.append(_section_pulsed_html(demo, idx, bigraph, cs))

    nav_items = ''.join(
        f'<a href="#sim-{d["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[d["color_scheme"]]["primary"]};">'
        f'{d["title"]}</a>' for d in demos)

    js_data = _make_data_for_js(demos)
    pbg_metrics = {d['id']: (
        {f'sub_{i}': sr['metrics'] for i, sr in enumerate(d['sub_runs'])}
        if d['kind'] == 'sweep'
        else d['snapshots'][-1]['metrics']
    ) for d in demos}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>pbg-medyan — MEDYAN-Driven Biophysics Demos</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>{CSS}</style>
</head>
<body>

<div class="page-header">
  <h1>pbg-medyan &mdash; MEDYAN-Driven Biophysics Demos</h1>
  <p>Three configurations driven by the actual MEDYAN C++ binary at
  <code>{binary}</code> through <strong>MedyanCxxProcess</strong>.
  All filament beads, motor and linker endpoints, and
  <strong>membrane mesh vertices</strong> shown below come straight
  from MEDYAN&rsquo;s <code>snapshot.traj</code> and <code>traj.h5</code>
  outputs &mdash; this is the upstream simulator, not a Python
  reimplementation.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections)}

<div class="footer">
  Generated by <strong>pbg-medyan</strong> &middot;
  MEDYAN v5.4.0 via <code>MedyanCxxProcess</code> &middot;
  HDF5 <code>traj.h5</code> for membrane geometry &middot;
  checkpoint-restart through <code>FILAMENTFILE</code> +
  <code>PROJECTIONTYPE: PREDEFINED</code>
</div>

<script>
const DATA = {json.dumps(js_data)};
const METRICS = {json.dumps(pbg_metrics, indent=2, default=str)};

// ─── JSON Tree ────────────────────────────
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      return '<span class="jt-bracket">[</span>' + obj.map(x => renderJson(x, depth+1)).join(', ') + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' + renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const t = el.parentElement.querySelector('.jt-toggle');
    if (t) t.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const t = el.parentElement.querySelector('.jt-toggle');
    if (t) t.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(METRICS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(METRICS[sid], 0);
}});

// ─── Three.js helpers ───
function makeFilamentLines(frame) {{
  const positions = [];
  for (const fil of frame.filaments) {{
    for (let i = 0; i < fil.length - 1; i++) {{
      positions.push(fil[i][0], fil[i][1], fil[i][2]);
      positions.push(fil[i+1][0], fil[i+1][1], fil[i+1][2]);
    }}
  }}
  return new Float32Array(positions);
}}
function makePairLines(items) {{
  const positions = [];
  for (const it of items) {{
    positions.push(it.a[0], it.a[1], it.a[2]);
    positions.push(it.b[0], it.b[1], it.b[2]);
  }}
  return new Float32Array(positions);
}}
function setupCameraAndLights(scene, cam, controls, scene_center, scene_extent) {{
  const cdist = scene_extent * 1.6;
  cam.position.set(scene_center[0]+cdist*0.7, scene_center[1]+cdist*0.5, scene_center[2]+cdist*0.7);
  cam.lookAt(scene_center[0], scene_center[1], scene_center[2]);
  controls.target.set(scene_center[0], scene_center[1], scene_center[2]);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;
  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dl = new THREE.DirectionalLight(0xffffff, 0.7);
  dl.position.set(3,5,4); scene.add(dl);
}}

// Static-frame viewer (for sweep panels): no slider, no animation.
function initStaticViewer(canvasId, frame, scene_center, scene_extent) {{
  const canvas = document.getElementById(canvasId);
  const W = canvas.parentElement.clientWidth;
  const H = 280;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0x0f172a);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0f172a, scene_extent*1.5, scene_extent*5);
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.001, scene_extent*50);
  const controls = new THREE.OrbitControls(cam, canvas);
  setupCameraAndLights(scene, cam, controls, scene_center, scene_extent);

  const filGeo = new THREE.BufferGeometry();
  filGeo.setAttribute('position', new THREE.BufferAttribute(makeFilamentLines(frame), 3));
  scene.add(new THREE.LineSegments(filGeo,
    new THREE.LineBasicMaterial({{color:0x22d3ee, linewidth:2, transparent:true, opacity:0.95}})));

  const motGeo = new THREE.BufferGeometry();
  motGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(frame.motors), 3));
  scene.add(new THREE.LineSegments(motGeo, new THREE.LineBasicMaterial({{color:0xf43f5e, linewidth:3}})));

  const lkGeo = new THREE.BufferGeometry();
  lkGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(frame.linkers), 3));
  scene.add(new THREE.LineSegments(lkGeo,
    new THREE.LineBasicMaterial({{color:0xfbbf24, linewidth:1.3, transparent:true, opacity:0.85}})));

  function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, cam); }}
  animate();
}}

// Animated viewer with optional membrane mesh.
const viewers = {{}};
const playStates = {{}};

function initAnimatedViewer(sid, snapshots, scene_center, scene_extent, has_membrane, primary_color) {{
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 520;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0x0f172a);
  const scene = new THREE.Scene();
  scene.fog = new THREE.Fog(0x0f172a, scene_extent*1.5, scene_extent*5);
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.001, scene_extent*50);
  const controls = new THREE.OrbitControls(cam, canvas);
  setupCameraAndLights(scene, cam, controls, scene_center, scene_extent);

  const filGeo = new THREE.BufferGeometry();
  filGeo.setAttribute('position', new THREE.BufferAttribute(makeFilamentLines(snapshots[0]), 3));
  scene.add(new THREE.LineSegments(filGeo,
    new THREE.LineBasicMaterial({{color:0x22d3ee, linewidth:2}})));

  const motGeo = new THREE.BufferGeometry();
  motGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(snapshots[0].motors), 3));
  scene.add(new THREE.LineSegments(motGeo, new THREE.LineBasicMaterial({{color:0xf43f5e, linewidth:3}})));

  const lkGeo = new THREE.BufferGeometry();
  lkGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(snapshots[0].linkers), 3));
  scene.add(new THREE.LineSegments(lkGeo, new THREE.LineBasicMaterial({{color:0xfbbf24, linewidth:1.3, transparent:true, opacity:0.85}})));

  let memMesh = null, memWire = null, memGeo = null;
  if (has_membrane && snapshots[0].membrane) {{
    memGeo = new THREE.BufferGeometry();
    const v = snapshots[0].membrane.vertices;
    const positions = new Float32Array(v.length * 3);
    for (let i = 0; i < v.length; i++) {{ positions[i*3]=v[i][0]; positions[i*3+1]=v[i][1]; positions[i*3+2]=v[i][2]; }}
    const indices = [];
    for (const tri of snapshots[0].membrane.triangles) indices.push(tri[0], tri[1], tri[2]);
    memGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    memGeo.setIndex(indices);
    memGeo.computeVertexNormals();
    const memMat = new THREE.MeshPhongMaterial({{
      color:0xa78bfa, transparent:true, opacity:0.32,
      side:THREE.DoubleSide, shininess:60, specular:0xddd6fe,
      flatShading:false, depthWrite:false }});
    memMesh = new THREE.Mesh(memGeo, memMat);
    scene.add(memMesh);
    const wireMat = new THREE.LineBasicMaterial({{color:0xc4b5fd, transparent:true, opacity:0.35}});
    memWire = new THREE.LineSegments(new THREE.WireframeGeometry(memGeo), wireMat);
    scene.add(memWire);
  }}

  function updateFrame(idx) {{
    const f = snapshots[idx];
    filGeo.setAttribute('position', new THREE.BufferAttribute(makeFilamentLines(f), 3));
    motGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(f.motors), 3));
    lkGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(f.linkers), 3));
    [filGeo, motGeo, lkGeo].forEach(g => g.attributes.position.needsUpdate = true);
    if (memMesh && f.membrane) {{
      const arr = memGeo.attributes.position.array;
      const v = f.membrane.vertices;
      for (let i = 0; i < v.length; i++) {{ arr[i*3]=v[i][0]; arr[i*3+1]=v[i][1]; arr[i*3+2]=v[i][2]; }}
      memGeo.attributes.position.needsUpdate = true;
      memGeo.computeVertexNormals();
      scene.remove(memWire);
      memWire.geometry.dispose();
      memWire = new THREE.LineSegments(new THREE.WireframeGeometry(memGeo), memWire.material);
      scene.add(memWire);
    }}
  }}

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateFrame(idx);
    if (DATA[sid].kind === 'single_pulsed') {{
      const ac = snapshots[idx].actin_copy;
      tval.textContent = 'i = ' + idx + ' (AD=' + ac + ')';
    }} else {{
      tval.textContent = 't = ' + snapshots[idx].time.toFixed(2) + ' s';
    }}
  }});
  viewers[sid] = {{ updateFrame, slider, tval, snapshots, controls }};
  playStates[sid] = {{ playing:false, interval:null }};

  function animate() {{ requestAnimationFrame(animate); controls.update(); renderer.render(scene, cam); }}
  animate();
}}

function togglePlay(sid) {{
  const ps = playStates[sid];
  const v = viewers[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {{
    btn.textContent = 'Pause';
    v.controls.autoRotate = false;
    ps.interval = setInterval(() => {{
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= v.snapshots.length) idx = 0;
      v.slider.value = idx;
      v.updateFrame(idx);
      if (DATA[sid].kind === 'single_pulsed') {{
        const ac = v.snapshots[idx].actin_copy;
        v.tval.textContent = 'i = ' + idx + ' (AD=' + ac + ')';
      }} else {{
        v.tval.textContent = 't = ' + v.snapshots[idx].time.toFixed(2) + ' s';
      }}
    }}, 600);
  }} else {{
    btn.textContent = 'Play';
    v.controls.autoRotate = true;
    clearInterval(ps.interval);
  }}
}}

// Init viewers
Object.keys(DATA).forEach(sid => {{
  const d = DATA[sid];
  if (d.kind === 'sweep') {{
    d.panels.forEach((p, k) => {{
      initStaticViewer('canvas-' + sid + '-' + k, p.frame, p.scene_center, p.scene_extent);
    }});
  }} else {{
    initAnimatedViewer(sid, d.snapshots, d.scene_center, d.scene_extent, d.has_membrane, d.primary_color);
  }}
}});

// ─── Plotly ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:15, t:35, b:40 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

// Demo 1: motor sweep
if (DATA['motor_sweep']) {{
  const s = DATA['motor_sweep'].sweep;
  Plotly.newPlot('chart-sweep-span-motor_sweep', [{{
    x: s.motor_copies, y: s.spans, type: 'scatter', mode: 'lines+markers',
    line:{{ color: DATA['motor_sweep'].primary_color, width: 2.5 }}, marker:{{ size: 8 }},
  }}], {{...pLayout,
    title:{{ text:'Network span vs motor copy number', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'MD copy number', font:{{ size:10 }} }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Network span (nm)', font:{{ size:10 }} }} }},
  }}, pCfg);
  Plotly.newPlot('chart-sweep-counts-motor_sweep', [
    {{ x: s.motor_copies, y: s.n_motors, name:'bound motors', type:'bar', marker:{{ color:'#f43f5e' }} }},
    {{ x: s.motor_copies, y: s.n_linkers, name:'bound linkers', type:'bar', marker:{{ color:'#fbbf24' }} }},
  ], {{...pLayout, barmode:'group',
    title:{{ text:'Bound motors / linkers per run', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'MD copy number', font:{{ size:10 }} }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);
}}

// Demo 2: vesicle
if (DATA['vesicle']) {{
  const c = DATA['vesicle'].charts;
  Plotly.newPlot('chart-mem-vesicle', [
    {{ x:c.times, y:c.membrane_radius, name:'mean radius', mode:'lines+markers',
       line:{{ color:'#a78bfa', width:2 }}, marker:{{ size:5 }} }},
    {{ x:c.times, y:c.membrane_span, name:'bbox span', yaxis:'y2', mode:'lines+markers',
       line:{{ color:'#7c3aed', width:1.5, dash:'dot' }}, marker:{{ size:3 }} }},
  ], {{...pLayout, title:{{ text:'Vesicle geometry over time', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'Time (s)', font:{{ size:10 }} }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Mean radius (nm)', font:{{ size:10 }} }} }},
    yaxis2:{{ overlaying:'y', side:'right', gridcolor:'transparent',
              title:{{ text:'BBox span (nm)', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);
  Plotly.newPlot('chart-fil-vesicle', [
    {{ x:c.times, y:c.fil_total, name:'total length', mode:'lines+markers',
       line:{{ color:'#22d3ee', width:2 }}, marker:{{ size:5 }} }},
    {{ x:c.times, y:c.fil_mean, name:'mean length', yaxis:'y2', mode:'lines+markers',
       line:{{ color:'#0e7490', width:1.5, dash:'dot' }}, marker:{{ size:3 }} }},
  ], {{...pLayout, title:{{ text:'Filament length over time', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'Time (s)', font:{{ size:10 }} }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Total length (nm)', font:{{ size:10 }} }} }},
    yaxis2:{{ overlaying:'y', side:'right', gridcolor:'transparent',
              title:{{ text:'Mean length (nm)', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);
}}

// Demo 3: composability
if (DATA['composability']) {{
  const c = DATA['composability'].charts;
  Plotly.newPlot('chart-pulse-composability', [
    {{ x:c.i, y:c.actin_copy, name:'AD copy (input)', mode:'lines+markers',
       line:{{ color:'#f59e0b', width:2.5, shape:'hv' }}, marker:{{ size:6 }} }},
  ], {{...pLayout, title:{{ text:'External input — AD copy schedule', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'Interval', font:{{ size:10 }} }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'AD diffusing copy', font:{{ size:10 }} }} }},
  }}, pCfg);
  Plotly.newPlot('chart-response-composability', [
    {{ x:c.i, y:c.fil_total, name:'total length (nm)', mode:'lines+markers',
       line:{{ color:'#22d3ee', width:2 }}, marker:{{ size:6 }} }},
    {{ x:c.i, y:c.n_filaments, name:'n_filaments', yaxis:'y2', mode:'lines+markers',
       line:{{ color:'#7c3aed', width:1.5, dash:'dot' }}, marker:{{ size:5 }} }},
  ], {{...pLayout, title:{{ text:'MEDYAN response — cytoskeleton metrics', font:{{ size:12, color:'#334155' }} }},
    xaxis:{{...pLayout.xaxis, title:{{ text:'Interval', font:{{ size:10 }} }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Total length (nm)', font:{{ size:10 }} }} }},
    yaxis2:{{ overlaying:'y', side:'right', gridcolor:'transparent',
              title:{{ text:'n_filaments', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);
}}

</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


# ── entry point ────────────────────────────────────────────────────


def main() -> int:
    try:
        binary = find_medyan_binary()
    except FileNotFoundError as exc:
        print('cxx_demo_report: ' + str(exc), file=sys.stderr)
        print('\nThis demo runs the *real* MEDYAN C++ binary, so it cannot '
              'be generated without one. Build MEDYAN from '
              'https://github.com/simularium/medyan, then either set\n'
              '  export MEDYAN_BIN=/path/to/medyan\nor add the binary to '
              'PATH, and re-run.', file=sys.stderr)
        return 1
    print(f'Using MEDYAN binary: {binary}\n')

    demos = [
        run_motor_sweep(binary),
        run_vesicle(binary),
        run_composability(binary),
    ]

    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'cxx_report.html')
    print('\nGenerating HTML report...')
    generate_html(demos, output_path, binary)

    try:
        subprocess.run(['open', '-a', 'Safari', output_path], check=False)
    except FileNotFoundError:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())

"""C++ MEDYAN multi-configuration interactive HTML report.

This is the C++-binary counterpart to demo/demo_report.py. It runs
several MEDYAN simulations through MedyanCxxProcess (subprocess +
checkpoint-restart) and produces a self-contained HTML report with
an interactive Three.js viewer (filaments + linkers + motors +
branchers), Plotly charts, a colored bigraph-viz architecture
diagram, and a collapsible JSON document tree.

To run::

    git clone https://github.com/medyan-dev/medyan-public
    # build it per their instructions
    export MEDYAN_BIN=/path/to/built/medyan
    python demo/cxx_demo_report.py

Without MEDYAN built, the script prints instructions and exits.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
import tempfile
import time as _time

import numpy as np
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_medyan.cxx import MedyanCxxProcess
from pbg_medyan.cxx.io import find_medyan_binary
from pbg_medyan.cxx.templates import PRESETS


# ── Configurations (sized for tractable wall-clock runtime) ────────


CONFIGS = [
    {
        'id': 'cxx_sparse',
        'title': 'Sparse Actin Treadmilling',
        'subtitle': 'Six pure actin filaments under brownian-ratchet kinetics',
        'description': (
            'Six filaments seeded inside a 1 µm cubic box with the '
            'bundled actin_only chemistry preset (barbed-end k+ = 0.151 '
            's⁻¹·µM⁻¹, k− = 1.4 s⁻¹). MEDYAN runs the full chemistry-'
            'mechanics-minimization loop, with this wrapper invoking the '
            'binary in 1-second checkpoint-restart intervals.'
        ),
        'config': {
            'n_filaments': 6,
            'filament_length': 1,
            'snapshot_interval': 0.5,
            'minimization_interval': 0.05,
            'compartment_size': 500.0,
            'nx': 2, 'ny': 2, 'nz': 2,
            'chemistry_preset': 'actin_only',
            'timeout': 300.0,
        },
        'n_snapshots': 5,
        'interval': 1.0,
        'color_scheme': 'indigo',
    },
    {
        'id': 'cxx_dense',
        'title': 'Dense Polymerizing Network',
        'subtitle': 'Twenty actin filaments in a 1 µm cube',
        'description': (
            'A denser actin field (no motors, no linkers). The longer '
            'cumulative simulation time exposes treadmilling steady-state '
            'length distributions and lets the bending force field relax '
            'inter-filament collisions through MEDYAN\'s native volume-'
            'exclusion potential (VOLUMEK = 8.57).'
        ),
        'config': {
            'n_filaments': 20,
            'filament_length': 2,
            'snapshot_interval': 0.5,
            'minimization_interval': 0.05,
            'compartment_size': 500.0,
            'nx': 2, 'ny': 2, 'nz': 2,
            'chemistry_preset': 'actin_only',
            'timeout': 600.0,
        },
        'n_snapshots': 6,
        'interval': 1.0,
        'color_scheme': 'emerald',
    },
    {
        'id': 'cxx_actomyosin',
        'title': 'Actomyosin Contractile Network',
        'subtitle': 'Filaments + α-actinin linkers + non-muscle myosin IIA motors',
        'description': (
            'The actin_motor_linker chemistry preset (matching MEDYAN\'s '
            'upstream 50filaments_motor_linker example) wires up '
            'low-duty-cycle catch-bond unbinding (Erdmann 2013) and Hill-'
            'style stall walking (Komianos & Papoian 2018). Filament tips '
            'feel motor pulling forces during minimization, and linkers '
            'tether nearby cylinders — MEDYAN handles all of it in '
            'native C++.'
        ),
        'config': {
            'n_filaments': 20,
            'filament_length': 2,
            'snapshot_interval': 0.5,
            'minimization_interval': 0.05,
            'compartment_size': 500.0,
            'nx': 2, 'ny': 2, 'nz': 2,
            'chemistry_preset': 'actin_motor_linker',
            'timeout': 900.0,
        },
        'n_snapshots': 6,
        'interval': 1.0,
        'color_scheme': 'rose',
    },
]


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48'},
    'amber': {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#b45309'},
}


# ── simulation runner ──────────────────────────────────────────────


def run_simulation(cfg_entry, binary):
    """Run a single config and capture per-update snapshots."""
    core = allocate_core()
    core.register_link('MedyanCxxProcess', MedyanCxxProcess)
    core.register_link('ram-emitter', RAMEmitter)

    proc = MedyanCxxProcess(
        config={**cfg_entry['config'], 'binary_path': binary,
                'keep_workdir': False},
        core=core)

    snapshots = []
    proc.initial_state()  # cheap; doesn't run MEDYAN
    snapshots.append(_init_snapshot(cfg_entry))

    t0 = _time.perf_counter()
    interval = cfg_entry['interval']
    for i in range(cfg_entry['n_snapshots']):
        print(f'  step {i+1}/{cfg_entry["n_snapshots"]} '
              f'(interval={interval}s)...', flush=True)
        result = proc.update({}, interval=interval)
        frame = proc.get_last_frame()
        snapshots.append(_frame_to_dict(frame, result))
    runtime = _time.perf_counter() - t0
    return snapshots, runtime


def _init_snapshot(cfg_entry):
    """Empty placeholder for t=0 (MEDYAN hasn't run yet)."""
    return {
        'time': 0.0,
        'filaments': [],
        'linkers': [],
        'motors': [],
        'branchers': [],
        'metrics': {
            'n_filaments': 0, 'n_linkers': 0, 'n_motors': 0, 'n_branchers': 0,
            'total_filament_length': 0.0, 'mean_filament_length': 0.0,
            'network_span': 0.0, 'cxx_runtime_seconds': 0.0,
        },
    }


def _frame_to_dict(frame, metrics):
    return {
        'time': float(frame.time),
        'filaments': [f.beads.tolist() for f in frame.filaments],
        'linkers': [{'a': lk.start.tolist(), 'b': lk.end.tolist()}
                    for lk in frame.linkers],
        'motors': [{'a': mo.start.tolist(), 'b': mo.end.tolist()}
                   for mo in frame.motors],
        'branchers': [{'a': br.start.tolist(), 'b': br.end.tolist()}
                      for br in frame.branchers],
        'metrics': {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                        else v)
                    for k, v in metrics.items()},
    }


# ── bigraph architecture diagram ───────────────────────────────────


def generate_bigraph_image(cfg_entry):
    from bigraph_viz import plot_bigraph

    doc = {
        'medyan_cxx': {
            '_type': 'process',
            'address': 'local:MedyanCxxProcess',
            'config': {k: v for k, v in cfg_entry['config'].items()
                       if k in ('n_filaments', 'filament_length',
                                'compartment_size', 'chemistry_preset')},
            'interval': cfg_entry['interval'],
            'inputs': {},
            'outputs': {
                'n_filaments': ['stores', 'n_filaments'],
                'total_filament_length': ['stores', 'total_filament_length'],
                'network_span': ['stores', 'network_span'],
                'n_motors': ['stores', 'n_motors'],
                'n_linkers': ['stores', 'n_linkers'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'n_filaments': 'integer',
                'total_filament_length': 'float',
                'network_span': 'float',
                'time': 'float',
            }},
            'inputs': {
                'n_filaments': ['stores', 'n_filaments'],
                'total_filament_length': ['stores', 'total_filament_length'],
                'network_span': ['stores', 'network_span'],
                'time': ['global_time'],
            },
        },
    }
    node_colors = {
        ('medyan_cxx',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }
    outdir = tempfile.mkdtemp()
    plot_bigraph(state=doc, out_dir=outdir, filename='bigraph',
                 file_format='png', remove_process_place_edges=True,
                 rankdir='LR', node_fill_colors=node_colors,
                 node_label_size='16pt', port_labels=False, dpi='150')
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg_entry, binary):
    """A fully-rendered PBG document for the JSON tree."""
    return {
        'medyan_cxx': {
            '_type': 'process',
            'address': 'local:MedyanCxxProcess',
            'config': {**cfg_entry['config'], 'binary_path': binary},
            'interval': cfg_entry['interval'],
            'inputs': {},
            'outputs': {
                'n_filaments': ['stores', 'n_filaments'],
                'n_linkers': ['stores', 'n_linkers'],
                'n_motors': ['stores', 'n_motors'],
                'n_branchers': ['stores', 'n_branchers'],
                'total_filament_length': ['stores', 'total_filament_length'],
                'mean_filament_length': ['stores', 'mean_filament_length'],
                'network_span': ['stores', 'network_span'],
                'cxx_runtime_seconds': ['stores', 'cxx_runtime_seconds'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'n_filaments': 'integer',
                'total_filament_length': 'float',
                'mean_filament_length': 'float',
                'network_span': 'float',
                'n_motors': 'integer',
                'n_linkers': 'integer',
                'time': 'float',
            }},
            'inputs': {
                'n_filaments': ['stores', 'n_filaments'],
                'total_filament_length': ['stores', 'total_filament_length'],
                'mean_filament_length': ['stores', 'mean_filament_length'],
                'network_span': ['stores', 'network_span'],
                'n_motors': ['stores', 'n_motors'],
                'n_linkers': ['stores', 'n_linkers'],
                'time': ['global_time'],
            },
        },
    }


# ── HTML report ────────────────────────────────────────────────────


def generate_html(sim_results, output_path, binary):
    sections = []
    all_js_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        n_steps = len(snapshots) - 1  # skip the t=0 placeholder

        # Compute scene framing from all bead positions
        all_pts = []
        for s in snapshots:
            for fil in s['filaments']:
                all_pts.extend(fil)
        if all_pts:
            ap = np.array(all_pts)
            c_min = ap.min(axis=0).tolist()
            c_max = ap.max(axis=0).tolist()
        else:
            c_min, c_max = [0, 0, 0], [1000, 1000, 1000]
        scene_center = [(a + b) / 2 for a, b in zip(c_min, c_max)]
        scene_extent = float(max(b - a for a, b in zip(c_min, c_max))) or 1000.0

        first = snapshots[0]['metrics']
        last = snapshots[-1]['metrics']

        all_js_data[sid] = {
            'frames': snapshots,
            'scene_center': scene_center,
            'scene_extent': scene_extent,
            'primary_color': cs['primary'],
            'charts': {
                'times':            [s['time'] for s in snapshots],
                'total_length':     [s['metrics']['total_filament_length'] for s in snapshots],
                'mean_length':      [s['metrics']['mean_filament_length'] for s in snapshots],
                'network_span':     [s['metrics']['network_span'] for s in snapshots],
                'n_filaments':      [s['metrics']['n_filaments'] for s in snapshots],
                'n_linkers':        [s['metrics']['n_linkers'] for s in snapshots],
                'n_motors':         [s['metrics']['n_motors'] for s in snapshots],
                'n_branchers':      [s['metrics']['n_branchers'] for s in snapshots],
                'cxx_runtime':      [s['metrics']['cxx_runtime_seconds'] for s in snapshots],
            },
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        n_fil = int(last['n_filaments'])
        L_total = float(last['total_filament_length'])
        L_mean = float(last['mean_filament_length'])
        span = float(last['network_span'])
        n_mot = int(last['n_motors'])
        n_lk = int(last['n_linkers'])
        n_br = int(last['n_branchers'])
        cxx_time_total = sum(s['metrics']['cxx_runtime_seconds'] for s in snapshots)

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Filaments</span><span class="metric-value">{n_fil}</span></div>
        <div class="metric"><span class="metric-label">Total Length</span><span class="metric-value">{L_total:.0f}</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Mean Length</span><span class="metric-value">{L_mean:.0f}</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Network Span</span><span class="metric-value">{span:.0f}</span><span class="metric-sub">nm</span></div>
        <div class="metric"><span class="metric-label">Motors</span><span class="metric-value">{n_mot}</span></div>
        <div class="metric"><span class="metric-label">Linkers</span><span class="metric-value">{n_lk}</span></div>
        <div class="metric"><span class="metric-label">Branchers</span><span class="metric-value">{n_br}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{n_steps}</span></div>
        <div class="metric"><span class="metric-label">Wall-clock</span><span class="metric-value">{runtime:.1f}s</span></div>
        <div class="metric"><span class="metric-label">C++ time</span><span class="metric-value">{cxx_time_total:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Network Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong>Filaments</strong> cyan &middot;
          <strong>Linkers</strong> yellow &middot;
          <strong>Motors</strong> red &middot;
          <strong>Branchers</strong> magenta<br>
          Drag to rotate &middot; Scroll to zoom &middot; Coords in nm
        </div>
        <div class="legend-box">
          <div class="cb-title">MEDYAN Components</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#22d3ee;"></span> Actin filament</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#fbbf24;"></span> &alpha;-actinin linker</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#f43f5e;"></span> Myosin II motor</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#ec4899;"></span> Arp2/3 brancher</div>
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{n_steps}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Network Metrics</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-length-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-counts-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-span-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-runtime-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    pbg_docs = {r[0]['id']: build_pbg_document(r[0], binary) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pbg-medyan (C++) — Real MEDYAN Simulation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{ background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
                border-bottom:1px solid #e2e8f0; padding:3rem; }}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:780px; }}
.page-header code {{ background:#fff; padding:.1rem .35rem; border-radius:4px;
                     border:1px solid #e2e8f0; font-size:.85rem; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .15s; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:820px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155; margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr));
                gap:.7rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.75rem; text-align:center; }}
.metric-label {{ display:block; font-size:.65rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.2rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.7rem; color:#94a3b8; }}
.viewer-wrap {{ position:relative; background:#0f172a; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.mesh-canvas {{ width:100%; height:520px; display:block; cursor:grab; }}
.mesh-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(15,23,42,.7);
                border:1px solid rgba(255,255,255,.1); border-radius:8px; padding:.5rem .8rem;
                font-size:.72rem; color:#cbd5e1; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#fff; }}
.legend-box {{ position:absolute; top:.8rem; right:.8rem; background:rgba(15,23,42,.78);
               border:1px solid rgba(255,255,255,.1); border-radius:8px; padding:.6rem .8rem;
               font-size:.72rem; color:#e2e8f0; backdrop-filter:blur(4px); }}
.cb-title {{ font-size:.62rem; text-transform:uppercase; letter-spacing:.06em;
             color:#94a3b8; margin-bottom:.3rem; font-weight:600; }}
.lg-row {{ display:flex; align-items:center; gap:.5rem; line-height:1.4; }}
.lg-swatch {{ width:14px; height:3px; border-radius:1px; display:inline-block; }}
.slider-controls {{ position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(15,23,42,.95));
                    padding:1.6rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }}
.slider-controls label {{ font-size:.8rem; color:#94a3b8; }}
.time-slider {{ flex:1; height:5px; }}
.time-val {{ font-size:.95rem; font-weight:600; color:#e2e8f0; min-width:120px; text-align:right; }}
.play-btn {{ background:#fff; border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }}
.play-btn:hover {{ transform:scale(1.05); }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:280px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>pbg-medyan (C++) &mdash; Real MEDYAN Simulation Report</h1>
  <p>Three configurations driven by the actual MEDYAN C++ binary at
  <code>{binary}</code> through <strong>MedyanCxxProcess</strong>
  (subprocess + checkpoint-restart). Filament beads, linker / motor /
  brancher endpoints come straight from MEDYAN&rsquo;s
  <code>snapshot.traj</code> &mdash; this is the upstream simulator,
  not the Python reimplementation.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections)}

<div class="footer">
  Generated by <strong>pbg-medyan</strong> &middot;
  MEDYAN C++ via <code>MedyanCxxProcess</code> &middot;
  Checkpoint-restart through <code>FILAMENTFILE</code> +
  <code>PROJECTIONTYPE: PREDEFINED</code>
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2, default=str)};

// ─── JSON Tree Viewer ───
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
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
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
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
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// ─── Three.js Viewers ───
const viewers = {{}};
const playStates = {{}};

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

function makeTipPoints(frame) {{
  const positions = [];
  for (const fil of frame.filaments) {{
    if (fil.length === 0) continue;
    const last = fil[fil.length - 1];
    positions.push(last[0], last[1], last[2]);
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

function makePairPoints(items) {{
  const positions = [];
  for (const it of items) {{
    positions.push(it.a[0], it.a[1], it.a[2]);
    positions.push(it.b[0], it.b[1], it.b[2]);
  }}
  return new Float32Array(positions);
}}

function initViewer(sid) {{
  const d = DATA[sid];
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
  scene.fog = new THREE.Fog(0x0f172a, d.scene_extent * 1.5, d.scene_extent * 5);

  const cam = new THREE.PerspectiveCamera(45, W/H, 0.001, d.scene_extent * 50);
  const cdist = d.scene_extent * 1.6;
  cam.position.set(d.scene_center[0] + cdist*0.7, d.scene_center[1] + cdist*0.5,
                   d.scene_center[2] + cdist*0.7);
  cam.lookAt(d.scene_center[0], d.scene_center[1], d.scene_center[2]);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.target.set(d.scene_center[0], d.scene_center[1], d.scene_center[2]);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.6;

  scene.add(new THREE.AmbientLight(0xffffff, 0.6));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.7);
  dl1.position.set(3,5,4); scene.add(dl1);

  const filGeo = new THREE.BufferGeometry();
  filGeo.setAttribute('position', new THREE.BufferAttribute(makeFilamentLines(d.frames[0]), 3));
  const filMat = new THREE.LineBasicMaterial({{color:0x22d3ee, linewidth:2, transparent:true, opacity:0.95}});
  const filLines = new THREE.LineSegments(filGeo, filMat);
  scene.add(filLines);

  const tipGeo = new THREE.BufferGeometry();
  tipGeo.setAttribute('position', new THREE.BufferAttribute(makeTipPoints(d.frames[0]), 3));
  const tipMat = new THREE.PointsMaterial({{color:0x86efac, size:d.scene_extent*0.012, sizeAttenuation:true}});
  const tipPts = new THREE.Points(tipGeo, tipMat);
  scene.add(tipPts);

  const lkGeo = new THREE.BufferGeometry();
  lkGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(d.frames[0].linkers), 3));
  const lkMat = new THREE.LineBasicMaterial({{color:0xfbbf24, linewidth:1.3, transparent:true, opacity:0.85}});
  const lkLines = new THREE.LineSegments(lkGeo, lkMat);
  scene.add(lkLines);

  const motGeo = new THREE.BufferGeometry();
  motGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(d.frames[0].motors), 3));
  const motMat = new THREE.LineBasicMaterial({{color:0xf43f5e, linewidth:3}});
  const motLines = new THREE.LineSegments(motGeo, motMat);
  scene.add(motLines);

  const motPtGeo = new THREE.BufferGeometry();
  motPtGeo.setAttribute('position', new THREE.BufferAttribute(makePairPoints(d.frames[0].motors), 3));
  const motPtMat = new THREE.PointsMaterial({{color:0xf43f5e, size:d.scene_extent*0.018, sizeAttenuation:true}});
  scene.add(new THREE.Points(motPtGeo, motPtMat));

  const brGeo = new THREE.BufferGeometry();
  brGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(d.frames[0].branchers), 3));
  const brMat = new THREE.LineBasicMaterial({{color:0xec4899, linewidth:2}});
  const brLines = new THREE.LineSegments(brGeo, brMat);
  scene.add(brLines);

  function updateFrame(idx) {{
    const f = d.frames[idx];
    filGeo.setAttribute('position', new THREE.BufferAttribute(makeFilamentLines(f), 3));
    tipGeo.setAttribute('position', new THREE.BufferAttribute(makeTipPoints(f), 3));
    lkGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(f.linkers), 3));
    motGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(f.motors), 3));
    motPtGeo.setAttribute('position', new THREE.BufferAttribute(makePairPoints(f.motors), 3));
    brGeo.setAttribute('position', new THREE.BufferAttribute(makePairLines(f.branchers), 3));
    [filGeo, tipGeo, lkGeo, motGeo, motPtGeo, brGeo].forEach(g => {{
      g.attributes.position.needsUpdate = true;
    }});
  }}

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateFrame(idx);
    tval.textContent = 't = ' + d.frames[idx].time.toFixed(2) + ' s';
  }});

  viewers[sid] = {{ renderer, scene, cam, controls, updateFrame, slider, tval }};
  playStates[sid] = {{ playing:false, interval:null }};

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }}
  animate();
}}

function togglePlay(sid) {{
  const ps = playStates[sid];
  const v = viewers[sid];
  const d = DATA[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {{
    btn.textContent = 'Pause';
    v.controls.autoRotate = false;
    ps.interval = setInterval(() => {{
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.frames.length) idx = 0;
      v.slider.value = idx;
      v.updateFrame(idx);
      v.tval.textContent = 't = ' + d.frames[idx].time.toFixed(2) + ' s';
    }}, 350);
  }} else {{
    btn.textContent = 'Play';
    v.controls.autoRotate = true;
    clearInterval(ps.interval);
  }}
}}

Object.keys(DATA).forEach(sid => initViewer(sid));

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:15, t:35, b:40 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'Time (s)', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

Object.keys(DATA).forEach(sid => {{
  const c = DATA[sid].charts;
  const pc = DATA[sid].primary_color;

  Plotly.newPlot('chart-length-'+sid, [
    {{ x:c.times, y:c.total_length, type:'scatter', mode:'lines+markers',
       line:{{ color:pc, width:2 }}, marker:{{ size:5 }}, name:'Total' }},
    {{ x:c.times, y:c.mean_length, type:'scatter', mode:'lines+markers', yaxis:'y2',
       line:{{ color:'#64748b', width:1.5, dash:'dot' }}, marker:{{ size:3 }}, name:'Mean' }},
  ], {{...pLayout, title:{{ text:'Filament Length (nm)', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Total (nm)', font:{{ size:10 }} }} }},
    yaxis2:{{ overlaying:'y', side:'right', gridcolor:'transparent',
              title:{{ text:'Mean (nm)', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);

  Plotly.newPlot('chart-counts-'+sid, [
    {{ x:c.times, y:c.n_filaments, type:'scatter', mode:'lines+markers',
       line:{{ color:'#22d3ee', width:1.8 }}, name:'Filaments' }},
    {{ x:c.times, y:c.n_motors, type:'scatter', mode:'lines+markers',
       line:{{ color:'#f43f5e', width:1.8 }}, name:'Motors' }},
    {{ x:c.times, y:c.n_linkers, type:'scatter', mode:'lines+markers',
       line:{{ color:'#fbbf24', width:1.8 }}, name:'Linkers' }},
    {{ x:c.times, y:c.n_branchers, type:'scatter', mode:'lines+markers',
       line:{{ color:'#ec4899', width:1.8 }}, name:'Branchers' }},
  ], {{...pLayout, title:{{ text:'Component Counts', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);

  Plotly.newPlot('chart-span-'+sid, [
    {{ x:c.times, y:c.network_span, type:'scatter', mode:'lines+markers',
       line:{{ color:pc, width:2 }}, marker:{{ size:5 }},
       fill:'tozeroy', fillcolor:'rgba(99,102,241,.06)' }},
  ], {{...pLayout, title:{{ text:'Network Span (nm)', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'nm', font:{{ size:10 }} }} }}, showlegend:false,
  }}, pCfg);

  Plotly.newPlot('chart-runtime-'+sid, [
    {{ x:c.times, y:c.cxx_runtime, type:'bar',
       marker:{{ color:'#a78bfa' }}, name:'C++ time' }},
  ], {{...pLayout, title:{{ text:'MEDYAN Wall-clock per Interval (s)', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'seconds', font:{{ size:10 }} }} }}, showlegend:false,
  }}, pCfg);
}});
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
        print(
            '\nThis demo runs the *real* MEDYAN C++ binary, so it cannot '
            'be generated without one.\nBuild MEDYAN from '
            'https://github.com/medyan-dev/medyan-public, then either set\n'
            '  export MEDYAN_BIN=/path/to/medyan\nor add the binary to '
            'PATH, and re-run this script.', file=sys.stderr)
        return 1

    print(f'Using MEDYAN binary: {binary}\n')
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'cxx_report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg, binary)
        sim_results.append((cfg, (snapshots, runtime)))
        print(f'  total wall-clock: {runtime:.1f}s, {len(snapshots)-1} snapshots\n')

    print('Generating HTML report...')
    generate_html(sim_results, output_path, binary)

    # Auto-open in Safari (macOS)
    try:
        subprocess.run(['open', '-a', 'Safari', output_path], check=False)
    except FileNotFoundError:
        pass
    return 0


if __name__ == '__main__':
    sys.exit(main())

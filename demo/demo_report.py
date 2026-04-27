"""Demo: pbg-medyan multi-configuration cytoskeleton report.

Runs three distinct mechanochemical cytoskeleton simulations
(treadmilling polymerization, actomyosin contraction, dendritic
crosslinked mesh), generates an interactive 3D filament viewer with
Three.js, Plotly time-series charts, a colored bigraph-viz architecture
diagram, and a navigable PBG composite document tree — all in a single
self-contained HTML report.
"""

import json
import os
import base64
import tempfile
import time as _time

import numpy as np
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_medyan.processes import MedyanProcess
from pbg_medyan.composites import make_network_document


# ── Simulation Configs ──────────────────────────────────────────────

CONFIGS = [
    {
        'id': 'treadmill',
        'title': 'Treadmilling Polymerization',
        'subtitle': 'ATP-driven actin treadmill near steady state',
        'description': (
            'Twelve actin filaments seeded near critical G-actin concentration. '
            'Plus (barbed) ends polymerize while minus (pointed) ends slowly '
            'depolymerize, producing the characteristic ATP-actin treadmill. '
            'No motors or crosslinkers — the network is a passive sea of '
            'semi-flexible polymers under Brownian-ratchet kinetics.'
        ),
        'config': {
            'box_size': 1.5,
            'n_filaments': 12,
            'initial_filament_length': 0.25,
            'seed_region_fraction': 0.65,
            'actin_concentration': 1.5,
            'k_on_plus': 11.6, 'k_off_plus': 1.4,
            'k_on_minus': 1.3, 'k_off_minus': 0.8,
            'n_motors': 0,
            'n_crosslinks': 0,
            'rng_seed': 11,
            'n_substeps': 6,
        },
        'n_snapshots': 30,
        'total_time': 30.0,
        'camera': [2.0, 1.5, 2.0],
        'color_scheme': 'indigo',
    },
    {
        'id': 'contractile',
        'title': 'Actomyosin Contractility',
        'subtitle': 'Myosin II minifilaments compact a crosslinked actin gel',
        'description': (
            'Fifteen filaments embedded in alpha-actinin crosslinkers and '
            'driven by myosin II minifilament motors that walk toward plus '
            'ends with Hill-style force-velocity (stall force Fs = 8 pN). '
            'The Brownian-ratchet polymerization slows under load while motor '
            'pulling drives global network compaction — the hallmark mode of '
            'cellular force generation.'
        ),
        'config': {
            'box_size': 1.2,
            'n_filaments': 15,
            'initial_filament_length': 0.4,
            'seed_region_fraction': 0.4,
            'actin_concentration': 0.4,
            'n_motors': 30,
            'n_crosslinks': 40,
            'new_motors_per_step': 2,
            'new_crosslinks_per_step': 2,
            'bind_radius': 0.3,
            'motor_force': 10.0,
            'motor_v0': 0.18,
            'motor_stall_force': 8.0,
            'crosslink_stiffness': 25.0,
            'rng_seed': 23,
            'n_substeps': 8,
        },
        'n_snapshots': 30,
        'total_time': 25.0,
        'camera': [1.6, 1.2, 1.6],
        'color_scheme': 'rose',
    },
    {
        'id': 'filopodia',
        'title': 'Vesicle Filopodial Protrusion',
        'subtitle': 'Polymerizing actin pushes a deformable membrane outward',
        'description': (
            'Eighteen actin filaments seeded radially inside a closed '
            'icosphere vesicle. Plus-end polymerization drives each tip '
            'into the membrane, where contact forces transmit Brownian-'
            'ratchet feedback: the membrane locally bulges outward into '
            'filopodia-like protrusions while the lipid bilayer '
            '(modeled as edge-spring elasticity plus Laplacian bending) '
            'resists global expansion. This is the canonical mechano-'
            'chemical coupling that generates lamellipodia, filopodia, '
            'and microvilli in real cells.'
        ),
        'config': {
            'box_size': 2.0,
            'n_filaments': 18,
            'initial_filament_length': 0.25,
            'seed_mode': 'radial',
            'actin_concentration': 8.0,
            'k_on_plus': 12.0, 'k_off_plus': 1.0,
            'k_on_minus': 0.5, 'k_off_minus': 1.5,
            'n_motors': 0,
            'n_crosslinks': 0,
            'enable_membrane': True,
            'membrane_radius': 0.55,
            'membrane_subdivisions': 2,
            'membrane_edge_stiffness': 8.0,
            'membrane_bending_stiffness': 1.5,
            'membrane_pressure': 0.0,
            'membrane_drag': 25.0,
            'membrane_filament_coupling_radius': 0.12,
            'membrane_filament_coupling_strength': 120.0,
            'boundary_force_scale': 2.0,
            'rng_seed': 31,
            'n_substeps': 10,
        },
        'n_snapshots': 35,
        'total_time': 8.0,
        'camera': [1.6, 1.2, 1.6],
        'color_scheme': 'amber',
    },
    {
        'id': 'dendritic',
        'title': 'Dendritic Crosslinked Mesh',
        'subtitle': 'Dense passive crosslinker network under net polymerization',
        'description': (
            'Twenty-five short filaments grow into a densely crosslinked '
            'mesh. With high alpha-actinin density and no motors, the '
            'network behaves as a passive elastic gel — bending forces '
            'compete with crosslinker tethering and net plus-end growth. '
            'This regime models lamellipodial / dendritic networks in '
            'migrating cells.'
        ),
        'config': {
            'box_size': 1.5,
            'n_filaments': 25,
            'initial_filament_length': 0.3,
            'seed_region_fraction': 0.5,
            'actin_concentration': 1.0,
            'n_motors': 0,
            'n_crosslinks': 60,
            'new_crosslinks_per_step': 3,
            'crosslink_unbind_rate': 0.02,
            'bind_radius': 0.3,
            'crosslink_stiffness': 20.0,
            'bending_stiffness': 0.1,
            'rng_seed': 5,
            'n_substeps': 8,
        },
        'n_snapshots': 30,
        'total_time': 18.0,
        'camera': [2.2, 1.6, 2.2],
        'color_scheme': 'emerald',
    },
]


COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
    'amber': {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#b45309',
              'bg': '#fffbeb', 'accent': '#fcd34d', 'text': '#78350f'},
}


# ── Run a single simulation ────────────────────────────────────────


def run_simulation(cfg_entry):
    """Run one config, returning per-snapshot geometry + metrics + runtime."""
    core = allocate_core()
    core.register_link('MedyanProcess', MedyanProcess)
    core.register_link('ram-emitter', RAMEmitter)

    t0 = _time.perf_counter()
    proc = MedyanProcess(config=cfg_entry['config'], core=core)
    state0 = proc.initial_state()
    engine = proc.get_engine()

    n_snaps = cfg_entry['n_snapshots']
    interval = cfg_entry['total_time'] / n_snaps
    snapshots = [_snapshot(engine, state0)]

    for _ in range(n_snaps):
        metrics = proc.update({}, interval=interval)
        snapshots.append(_snapshot(engine, metrics))

    runtime = _time.perf_counter() - t0
    return snapshots, runtime


def _snapshot(engine, metrics):
    snap = engine.snapshot()
    snap['metrics'] = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer))
                       else v for k, v in metrics.items()}
    return snap


def _faces_for_frame(snapshots):
    """Return the (constant) membrane face indices, or None if no membrane."""
    s0 = snapshots[0]
    if s0.get('membrane') is None:
        return None
    return s0['membrane']['faces']


# ── Bigraph diagram ────────────────────────────────────────────────


def generate_bigraph_image(cfg_entry):
    """Render a colored bigraph-viz PNG of the simplified composite document."""
    from bigraph_viz import plot_bigraph

    doc = {
        'cytoskeleton': {
            '_type': 'process',
            'address': 'local:MedyanProcess',
            'config': {k: v for k, v in cfg_entry['config'].items()
                       if k in ('n_filaments', 'actin_concentration',
                                'n_motors', 'n_crosslinks')},
            'interval': cfg_entry['total_time'] / cfg_entry['n_snapshots'],
            'inputs': {},
            'outputs': {
                'n_filaments': ['stores', 'n_filaments'],
                'total_length': ['stores', 'total_length'],
                'network_span': ['stores', 'network_span'],
                'total_energy': ['stores', 'total_energy'],
                'n_motors': ['stores', 'n_motors'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': {
                'n_filaments': 'integer',
                'total_length': 'float',
                'network_span': 'float',
                'total_energy': 'float',
                'time': 'float',
            }},
            'inputs': {
                'n_filaments': ['stores', 'n_filaments'],
                'total_length': ['stores', 'total_length'],
                'network_span': ['stores', 'network_span'],
                'total_energy': ['stores', 'total_energy'],
                'time': ['global_time'],
            },
        },
    }

    node_colors = {
        ('cytoskeleton',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg_entry):
    return make_network_document(
        interval=cfg_entry['total_time'] / cfg_entry['n_snapshots'],
        **cfg_entry['config'])


# ── HTML generation ────────────────────────────────────────────────


def generate_html(sim_results, output_path):
    sections_html = []
    all_js_data = {}

    for idx, (cfg, (snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]

        # JS data: list of frames; each frame contains line segments + motor/cl dots
        # Membrane faces are constant — extract once; vertices vary per frame.
        mem_faces = _faces_for_frame(snapshots)
        frames = []
        for s in snapshots:
            frame = {
                'time': round(s['time'], 2),
                'filaments': s['filaments'],
                'motors': s['motors'],
                'crosslinks': s['crosslinks'],
                'metrics': s['metrics'],
            }
            if s.get('membrane') is not None:
                frame['membrane_vertices'] = s['membrane']['vertices']
            frames.append(frame)

        # Per-config metrics
        first = snapshots[0]['metrics']
        last = snapshots[-1]['metrics']
        n_fil_first = int(first.get('n_filaments', 0))
        n_fil_last = int(last.get('n_filaments', 0))
        L0 = float(first.get('total_length', 0.0))
        L1 = float(last.get('total_length', 0.0))
        s0 = float(first.get('network_span', 0.0))
        s1 = float(last.get('network_span', 0.0))
        e_last = float(last.get('total_energy', 0.0))
        m_last = int(last.get('n_motors', 0))
        x_last = int(last.get('n_crosslinks', 0))
        has_membrane = mem_faces is not None
        mem_v0 = float(first.get('membrane_volume', 0.0))
        mem_v1 = float(last.get('membrane_volume', 0.0))
        mem_a1 = float(last.get('membrane_area', 0.0))
        mem_be1 = float(last.get('membrane_bending_energy', 0.0))

        # Time-series for charts
        times = [f['time'] for f in frames]
        total_len = [f['metrics'].get('total_length', 0.0) for f in frames]
        span_series = [f['metrics'].get('network_span', 0.0) for f in frames]
        rg_series = [f['metrics'].get('radius_of_gyration', 0.0) for f in frames]
        bend_e = [f['metrics'].get('bending_energy', 0.0) for f in frames]
        stretch_e = [f['metrics'].get('stretch_energy', 0.0) for f in frames]
        n_motors_series = [f['metrics'].get('n_motors', 0) for f in frames]
        n_xl_series = [f['metrics'].get('n_crosslinks', 0) for f in frames]
        mean_len_series = [f['metrics'].get('mean_filament_length', 0.0) for f in frames]
        mem_vol_series = [f['metrics'].get('membrane_volume', 0.0) for f in frames]
        mem_area_series = [f['metrics'].get('membrane_area', 0.0) for f in frames]
        mem_be_series = [f['metrics'].get('membrane_bending_energy', 0.0) for f in frames]
        mem_r_series = [f['metrics'].get('membrane_mean_radius', 0.0) for f in frames]

        # Compute scene scale for camera framing
        all_pts = []
        for f in frames:
            for fil in f['filaments']:
                all_pts.extend(fil)
        all_pts = np.array(all_pts) if all_pts else np.zeros((1, 3))
        c_min = all_pts.min(axis=0).tolist() if all_pts.size else [0, 0, 0]
        c_max = all_pts.max(axis=0).tolist() if all_pts.size else [1, 1, 1]
        scene_center = [(a + b) / 2 for a, b in zip(c_min, c_max)]
        scene_extent = float(max(b - a for a, b in zip(c_min, c_max))) or 1.0

        all_js_data[sid] = {
            'frames': frames,
            'scene_center': scene_center,
            'scene_extent': scene_extent,
            'primary_color': cs['primary'],
            'membrane_faces': mem_faces if has_membrane else None,
            'has_membrane': has_membrane,
            'charts': {
                'times': times,
                'total_length': total_len,
                'network_span': span_series,
                'radius_of_gyration': rg_series,
                'mean_length': mean_len_series,
                'bending_energy': bend_e,
                'stretch_energy': stretch_e,
                'n_motors': n_motors_series,
                'n_crosslinks': n_xl_series,
                'membrane_volume': mem_vol_series,
                'membrane_area': mem_area_series,
                'membrane_bending_energy': mem_be_series,
                'membrane_radius': mem_r_series,
            },
        }

        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

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
        <div class="metric"><span class="metric-label">Filaments</span><span class="metric-value">{n_fil_last}</span><span class="metric-sub">{n_fil_first} &rarr; {n_fil_last}</span></div>
        <div class="metric"><span class="metric-label">Total Length</span><span class="metric-value">{L1:.2f}</span><span class="metric-sub">{L0:.2f} &rarr; {L1:.2f} &mu;m</span></div>
        <div class="metric"><span class="metric-label">Network Span</span><span class="metric-value">{s1:.2f}</span><span class="metric-sub">{s0:.2f} &rarr; {s1:.2f} &mu;m</span></div>
        <div class="metric"><span class="metric-label">Motors</span><span class="metric-value">{m_last}</span></div>
        <div class="metric"><span class="metric-label">Crosslinks</span><span class="metric-value">{x_last}</span></div>
        {('<div class="metric"><span class="metric-label">Vesicle Vol</span><span class="metric-value">' + f'{mem_v1:.3f}' + '</span><span class="metric-sub">' + f'{mem_v0:.3f} &rarr; {mem_v1:.3f}' + '</span></div>') if has_membrane else ''}
        {('<div class="metric"><span class="metric-label">Mem Bending</span><span class="metric-value">' + f'{mem_be1:.3f}' + '</span></div>') if has_membrane else ''}
        <div class="metric"><span class="metric-label">Energy</span><span class="metric-value">{e_last:.2e}</span></div>
        <div class="metric"><span class="metric-label">Snapshots</span><span class="metric-value">{len(frames)}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Network Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          {'<strong>Vesicle</strong> (translucent) wraps the network &middot; ' if has_membrane else ''}<strong>Filaments</strong> cyan &middot;
          <strong>Motors</strong> red &middot;
          <strong>Crosslinks</strong> yellow<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="legend-box">
          <div class="cb-title">Network Components</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#22d3ee;"></span> Actin filament</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#f43f5e;"></span> Myosin II motor</div>
          <div class="lg-row"><span class="lg-swatch" style="background:#fbbf24;"></span> &alpha;-actinin crosslink</div>
          {'<div class="lg-row"><span class="lg-swatch" style="background:#a78bfa; height:8px;"></span> Lipid vesicle</div>' if has_membrane else ''}
        </div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(frames)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Network Metrics &amp; Energy</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-length-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-span-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-energy-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-{'membrane' if has_membrane else 'binding'}-{sid}" class="chart"></div></div>
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
        sections_html.append(section)

    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pbg-medyan: Cytoskeletal Mechanochemistry Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:760px; }}
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
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
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
.time-val {{ font-size:.95rem; font-weight:600; color:#e2e8f0; min-width:100px; text-align:right; }}
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
  <h1>pbg-medyan &mdash; Cytoskeletal Mechanochemistry Report</h1>
  <p>Three MEDYAN-style mechanochemical simulations wrapped as
  <strong>process-bigraph</strong> Processes. Each scenario explores a
  distinct cytoskeletal regime &mdash; treadmilling polymerization,
  actomyosin contractility, and dendritic crosslinked meshes &mdash;
  with semi-flexible filaments, force-sensitive Brownian-ratchet kinetics,
  Hill-style myosin motors, and alpha-actinin tethers.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-medyan</strong> &middot;
  MEDYAN-style mechanochemistry &middot;
  Brownian-ratchet polymerization &middot;
  Hill-stall myosin motors &middot;
  Alpha-actinin crosslinkers
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

function makeMotorLines(frame) {{
  const positions = [];
  for (const m of frame.motors) {{
    positions.push(m.a[0], m.a[1], m.a[2]);
    positions.push(m.b[0], m.b[1], m.b[2]);
  }}
  return new Float32Array(positions);
}}

function makeXlLines(frame) {{
  const positions = [];
  for (const c of frame.crosslinks) {{
    positions.push(c.a[0], c.a[1], c.a[2]);
    positions.push(c.b[0], c.b[1], c.b[2]);
  }}
  return new Float32Array(positions);
}}

function makePlusEndPoints(frame) {{
  const positions = [];
  for (const fil of frame.filaments) {{
    if (fil.length === 0) continue;
    const last = fil[fil.length - 1];
    positions.push(last[0], last[1], last[2]);
  }}
  return new Float32Array(positions);
}}

function makeMotorEndpointPoints(frame) {{
  const positions = [];
  for (const m of frame.motors) {{
    positions.push(m.a[0], m.a[1], m.a[2]);
    positions.push(m.b[0], m.b[1], m.b[2]);
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

  const cam = new THREE.PerspectiveCamera(45, W/H, 0.001, 100);
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

  // Filament lines
  const filGeo = new THREE.BufferGeometry();
  filGeo.setAttribute('position', new THREE.BufferAttribute(makeFilamentLines(d.frames[0]), 3));
  const filMat = new THREE.LineBasicMaterial({{color:0x22d3ee, linewidth:2, transparent:true, opacity:0.95}});
  const filLines = new THREE.LineSegments(filGeo, filMat);
  scene.add(filLines);

  // Plus-end points (small dots at barbed ends)
  const plusGeo = new THREE.BufferGeometry();
  plusGeo.setAttribute('position', new THREE.BufferAttribute(makePlusEndPoints(d.frames[0]), 3));
  const plusMat = new THREE.PointsMaterial({{color:0x86efac, size: d.scene_extent * 0.025, sizeAttenuation:true}});
  const plusPts = new THREE.Points(plusGeo, plusMat);
  scene.add(plusPts);

  // Motors (red)
  const motGeo = new THREE.BufferGeometry();
  motGeo.setAttribute('position', new THREE.BufferAttribute(makeMotorLines(d.frames[0]), 3));
  const motMat = new THREE.LineBasicMaterial({{color:0xf43f5e, linewidth:3}});
  const motLines = new THREE.LineSegments(motGeo, motMat);
  scene.add(motLines);

  const motPtGeo = new THREE.BufferGeometry();
  motPtGeo.setAttribute('position', new THREE.BufferAttribute(makeMotorEndpointPoints(d.frames[0]), 3));
  const motPtMat = new THREE.PointsMaterial({{color:0xf43f5e, size: d.scene_extent * 0.035, sizeAttenuation:true}});
  const motPts = new THREE.Points(motPtGeo, motPtMat);
  scene.add(motPts);

  // Crosslinks (yellow tethers)
  const xlGeo = new THREE.BufferGeometry();
  xlGeo.setAttribute('position', new THREE.BufferAttribute(makeXlLines(d.frames[0]), 3));
  const xlMat = new THREE.LineBasicMaterial({{color:0xfbbf24, linewidth:1, transparent:true, opacity:0.7}});
  const xlLines = new THREE.LineSegments(xlGeo, xlMat);
  scene.add(xlLines);

  // Membrane (translucent triangulated vesicle)
  let memMesh = null, memWire = null, memGeo = null;
  if (d.has_membrane && d.membrane_faces) {{
    memGeo = new THREE.BufferGeometry();
    const verts0 = d.frames[0].membrane_vertices || [];
    const positions = new Float32Array(verts0.length * 3);
    for (let i = 0; i < verts0.length; i++) {{
      positions[i*3] = verts0[i][0];
      positions[i*3+1] = verts0[i][1];
      positions[i*3+2] = verts0[i][2];
    }}
    const indices = [];
    for (const tri of d.membrane_faces) {{
      indices.push(tri[0], tri[1], tri[2]);
    }}
    memGeo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    memGeo.setIndex(indices);
    memGeo.computeVertexNormals();
    const memMat = new THREE.MeshPhongMaterial({{
      color:0xa78bfa, transparent:true, opacity:0.32,
      side:THREE.DoubleSide, shininess:60, specular:0xddd6fe,
      flatShading:false, depthWrite:false,
    }});
    memMesh = new THREE.Mesh(memGeo, memMat);
    scene.add(memMesh);
    const wireMat = new THREE.LineBasicMaterial({{color:0xc4b5fd, transparent:true, opacity:0.35}});
    memWire = new THREE.LineSegments(new THREE.WireframeGeometry(memGeo), wireMat);
    scene.add(memWire);
  }}

  function updateFrame(idx) {{
    const f = d.frames[idx];
    filGeo.setAttribute('position', new THREE.BufferAttribute(makeFilamentLines(f), 3));
    filGeo.attributes.position.needsUpdate = true;
    plusGeo.setAttribute('position', new THREE.BufferAttribute(makePlusEndPoints(f), 3));
    plusGeo.attributes.position.needsUpdate = true;
    motGeo.setAttribute('position', new THREE.BufferAttribute(makeMotorLines(f), 3));
    motGeo.attributes.position.needsUpdate = true;
    motPtGeo.setAttribute('position', new THREE.BufferAttribute(makeMotorEndpointPoints(f), 3));
    motPtGeo.attributes.position.needsUpdate = true;
    xlGeo.setAttribute('position', new THREE.BufferAttribute(makeXlLines(f), 3));
    xlGeo.attributes.position.needsUpdate = true;
    if (memMesh && f.membrane_vertices) {{
      const arr = memGeo.attributes.position.array;
      const v = f.membrane_vertices;
      for (let i = 0; i < v.length; i++) {{
        arr[i*3] = v[i][0];
        arr[i*3+1] = v[i][1];
        arr[i*3+2] = v[i][2];
      }}
      memGeo.attributes.position.needsUpdate = true;
      memGeo.computeVertexNormals();
      // Update wireframe geometry as well
      scene.remove(memWire);
      memWire.geometry.dispose();
      memWire = new THREE.LineSegments(
        new THREE.WireframeGeometry(memGeo),
        memWire.material);
      scene.add(memWire);
    }}
  }}

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateFrame(idx);
    tval.textContent = 't = ' + d.frames[idx].time;
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
      v.tval.textContent = 't = ' + d.frames[idx].time;
    }}, 250);
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
       line:{{ color:pc, width:2 }}, marker:{{ size:4 }}, name:'Total length' }},
    {{ x:c.times, y:c.mean_length, type:'scatter', mode:'lines+markers', yaxis:'y2',
       line:{{ color:'#64748b', width:1.5, dash:'dot' }}, marker:{{ size:3 }}, name:'Mean length' }},
  ], {{...pLayout, title:{{ text:'Filament Length', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Total (μm)', font:{{ size:10 }} }} }},
    yaxis2:{{ overlaying:'y', side:'right', gridcolor:'transparent',
              title:{{ text:'Mean (μm)', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);

  Plotly.newPlot('chart-span-'+sid, [
    {{ x:c.times, y:c.network_span, type:'scatter', mode:'lines+markers',
       line:{{ color:pc, width:2 }}, marker:{{ size:4 }}, name:'Span' }},
    {{ x:c.times, y:c.radius_of_gyration, type:'scatter', mode:'lines+markers',
       line:{{ color:'#94a3b8', width:1.5, dash:'dot' }}, marker:{{ size:3 }}, name:'R_g' }},
  ], {{...pLayout, title:{{ text:'Network Compactness', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Distance (μm)', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);

  Plotly.newPlot('chart-energy-'+sid, [
    {{ x:c.times, y:c.bending_energy, type:'scatter', mode:'lines+markers',
       line:{{ color:'#6366f1', width:1.5 }}, marker:{{ size:3 }}, name:'Bending' }},
    {{ x:c.times, y:c.stretch_energy, type:'scatter', mode:'lines+markers',
       line:{{ color:'#10b981', width:1.5 }}, marker:{{ size:3 }}, name:'Stretch' }},
  ], {{...pLayout, title:{{ text:'Mechanical Energy', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }},
    showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
  }}, pCfg);

  if (DATA[sid].has_membrane) {{
    Plotly.newPlot('chart-membrane-'+sid, [
      {{ x:c.times, y:c.membrane_volume, type:'scatter', mode:'lines+markers',
         line:{{ color:'#a78bfa', width:2 }}, marker:{{ size:4 }}, name:'Volume',
         fill:'tozeroy', fillcolor:'rgba(167,139,250,.10)' }},
      {{ x:c.times, y:c.membrane_bending_energy, type:'scatter', mode:'lines+markers', yaxis:'y2',
         line:{{ color:'#f59e0b', width:1.5, dash:'dot' }}, marker:{{ size:3 }}, name:'Bend energy' }},
    ], {{...pLayout, title:{{ text:'Vesicle Volume &amp; Mem Bending', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Volume (μm³)', font:{{ size:10 }} }} }},
      yaxis2:{{ overlaying:'y', side:'right', gridcolor:'transparent',
                title:{{ text:'Bending E', font:{{ size:10 }} }} }},
      showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
    }}, pCfg);
  }} else {{
    Plotly.newPlot('chart-binding-'+sid, [
      {{ x:c.times, y:c.n_motors, type:'scatter', mode:'lines+markers',
         line:{{ color:'#f43f5e', width:1.8 }}, marker:{{ size:3 }}, name:'Motors',
         fill:'tozeroy', fillcolor:'rgba(244,63,94,.07)' }},
      {{ x:c.times, y:c.n_crosslinks, type:'scatter', mode:'lines+markers',
         line:{{ color:'#fbbf24', width:1.8 }}, marker:{{ size:3 }}, name:'Crosslinks',
         fill:'tozeroy', fillcolor:'rgba(251,191,36,.07)' }},
    ], {{...pLayout, title:{{ text:'Bound Motors &amp; Crosslinks', font:{{ size:12, color:'#334155' }} }},
      yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
      showlegend:true, legend:{{ font:{{ size:9 }}, bgcolor:'rgba(0,0,0,0)' }},
    }}, pCfg);
  }}
}});
</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        snapshots, runtime = run_simulation(cfg)
        sim_results.append((cfg, (snapshots, runtime)))
        print(f'  Runtime: {runtime:.2f}s, {len(snapshots)} snapshots')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)


if __name__ == '__main__':
    run_demo()

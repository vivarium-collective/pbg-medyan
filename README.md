# pbg-medyan

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
wrapper for the [MEDYAN](https://medyan.org/) cytoskeletal-network simulator.

**[View Interactive Demo Report](https://vivarium-collective.github.io/pbg-medyan/)** —
real-MEDYAN motor-density sweep, deformable-vesicle filopodia (membrane mesh
pulled live from MEDYAN's `traj.h5`), and a PBG-composability demo where a
sibling process drives MEDYAN's G-actin concentration through a square-wave
pulse. Three.js 3D viewers, Plotly metrics charts, and bigraph-viz
architecture diagrams. The pure-Python engine's report is at
[python_demo.html](https://vivarium-collective.github.io/pbg-medyan/python_demo.html).

Two complementary Processes ship in this package:

| | `MedyanProcess` | `MedyanCxxProcess` |
| --- | --- | --- |
| Engine | Pure-Python re-implementation | The actual MEDYAN C++ binary, driven via subprocess |
| Install cost | `pip install -e .` | Build MEDYAN locally (~30 min, see [docs/macos_arm64_build.md](docs/macos_arm64_build.md)) |
| Runtime | Real-time on a laptop (10–100 filaments) | Production MEDYAN performance |
| Membrane support | Custom icosphere + Laplacian-bending model | MEDYAN's full deformable-vesicle subsystem (Helfrich + volume conservation), parsed from `traj.h5` |
| Use it for | Prototyping, education, offline CI, testing PBG compositions | Validating against published MEDYAN behavior, real biophysics |

The Python engine captures the qualitative MEDYAN physics (Brownian-ratchet
polymerization, Hill-stall myosin walking, α-actinin tethering, overdamped
relaxation, membrane–filament coupling); the C++ bridge runs the upstream
simulator end-to-end and parses both the legacy `snapshot.traj` and the
HDF5 `traj.h5` outputs.

Public repo: <https://github.com/vivarium-collective/pbg-medyan>

## Installation

```bash
git clone https://github.com/vivarium-collective/pbg-medyan.git
cd pbg-medyan
uv venv .venv && source .venv/bin/activate
uv pip install -e .[dev]
```

Tests run offline by default:

```bash
pytest -v
```

The C++ integration tests in `tests/test_cxx_integration.py` are
auto-skipped when no MEDYAN binary is found, so the offline suite stays
green. Run them with `MEDYAN_BIN=/path/to/medyan pytest -v`.

## Quick start — Python engine

```python
from process_bigraph import Composite, allocate_core, gather_emitter_results
from process_bigraph.emitter import RAMEmitter

from pbg_medyan.processes import MedyanProcess
from pbg_medyan.composites import make_network_document

core = allocate_core()
core.register_link('MedyanProcess', MedyanProcess)
core.register_link('ram-emitter', RAMEmitter)

doc = make_network_document(
    interval=1.0,
    box_size=1.2,
    n_filaments=15,
    initial_filament_length=0.4,
    actin_concentration=0.5,
    n_motors=25, n_crosslinks=30,
    new_motors_per_step=2, new_crosslinks_per_step=3,
    bind_radius=0.3,
    rng_seed=42,
)
sim = Composite({'state': doc}, core=core)
sim.run(20.0)

results = gather_emitter_results(sim)[('emitter',)]
print(results[-1])
```

## Quick start — real MEDYAN binary

```python
from process_bigraph import allocate_core
from pbg_medyan.cxx import MedyanCxxProcess

core = allocate_core()
core.register_link('MedyanCxxProcess', MedyanCxxProcess)

proc = MedyanCxxProcess(config={
    'n_filaments': 5, 'filament_length': 1,
    'snapshot_interval': 0.1,
    'chemistry_preset': 'actin_only',   # or 'actin_motor_linker'
}, core=core)
proc.initial_state()

# Single 1-second run, snapshotted every 0.1s. Harvest all 10 frames
# for animation:
proc.update({}, interval=1.0)
for frame in proc.get_last_frames():
    print(frame.time, len(frame.filaments))

# Or stitch multiple intervals via FILAMENTFILE checkpoint-restart
# (see "Restart fidelity" caveat below):
proc.update({}, interval=1.0)
proc.update({}, interval=1.0)
```

The wrapper finds the MEDYAN binary via, in order: (1) the `binary_path`
config field, (2) the `MEDYAN_BIN` environment variable, (3) `medyan` on
`PATH`. Build instructions and the macOS arm64 patch list are in
[`docs/macos_arm64_build.md`](docs/macos_arm64_build.md).

## Demos

Two interactive HTML reports ship with the package — both are
self-contained (CDN Three.js + Plotly), open in any browser, and embed
the bigraph-viz architecture diagram + collapsible JSON document tree
for each scenario.

### `demo/demo_report.py` — Python engine, 4 scenarios

```bash
python demo/demo_report.py    # writes demo/report.html
```

1. **Treadmilling polymerization** — pure actin treadmill near critical concentration
2. **Actomyosin contractility** — myosin minifilaments + α-actinin compact a crosslinked gel
3. **Vesicle filopodial protrusion** — radial actin pushes a closed icosphere membrane outward
4. **Dendritic crosslinked mesh** — high crosslinker density, no motors

### `demo/cxx_demo_report.py` — real MEDYAN binary, 3 scenarios

```bash
MEDYAN_BIN=/path/to/medyan python demo/cxx_demo_report.py    # writes demo/cxx_report.html
```

1. **Motor-density parameter sweep** — 4 sub-runs at varying myosin
   diffusing-species copy numbers, side-by-side 3D viewers + a
   quantitative span-vs-density plot. Reproduces the central
   [Popov, Komianos & Papoian 2016](https://doi.org/10.1371/journal.pcbi.1004877)
   contractile-collapse trend.
2. **Vesicle filopodia** — closed lipid vesicle (1000+ vertices,
   2100+ triangles, Helfrich bending + constant tension + volume
   conservation) wrapping an actin network. Membrane geometry comes
   straight from `snapshots/<i>/membranes/<j>/{vertexDataFloat64,
   triangleDataInt64}` in MEDYAN's HDF5 trajectory and is rendered as
   a translucent triangulated mesh in Three.js.
3. **PBG composability via actin pulse** — a square-wave G-actin
   concentration is pushed into MEDYAN through the `actin_copy`
   input port between intervals; the network goes through cycles of
   net polymerization (high actin) and depolymerization (low actin).
   This is the demo that **can't be done** with the bare MEDYAN
   CLI — the wrapper turns it into one-line input wiring inside any
   larger PBG Composite.

`demo/cxx_smoke.py` is a minimal sanity check that runs three
back-to-back intervals and prints metrics — useful when verifying
a fresh MEDYAN build.

## API reference

### `MedyanProcess` — Python engine

Output ports (all `overwrite[T]`):

| Port | Type | Notes |
| --- | --- | --- |
| `n_filaments`, `n_motors`, `n_crosslinks` | integer | live counts |
| `total_length`, `mean_filament_length` | float | μm |
| `network_span`, `radius_of_gyration` | float | μm |
| `bending_energy`, `stretch_energy`, `total_energy` | float | network mechanics |
| `membrane_area`, `membrane_volume`, `membrane_mean_radius` | float | when `enable_membrane=True` |
| `membrane_bending_energy` | float | edge-spring deformation energy |

Selected config fields (`MedyanProcess.config_schema` for the full list):

| Field | Default | Meaning |
| --- | --- | --- |
| `box_size` | `2.0` | cubic domain side length (μm) |
| `n_filaments`, `initial_filament_length` | `10`, `0.4` | seeding |
| `actin_concentration` | `10.0` | free G-actin (μM) |
| `k_on_plus`, `k_off_plus` | `11.6, 1.4` | barbed-end on/off rates |
| `k_on_minus`, `k_off_minus` | `1.3, 0.8` | pointed-end on/off rates |
| `n_motors`, `n_crosslinks` | `0, 0` | initial bound counts |
| `new_motors_per_step`, `new_crosslinks_per_step` | `0, 0` | binding attempts per `update()` |
| `motor_v0`, `motor_stall_force`, `motor_force` | `0.2, 8.0, 4.0` | μm/s, pN, pN |
| `crosslink_stiffness`, `cylinder_stiffness`, `bending_stiffness` | `8.0, 20.0, 0.05` | pN/μm, pN/μm, pN·μm² |
| `boundary_force_scale`, `drag_coefficient` | `1.5, 40.0` | pN, friction |
| `bind_radius` | `0.2` | motor/crosslink capture radius (μm) |
| `n_substeps` | `8` | integration sub-steps per `update()` call |
| `seed_mode` | `'random'` | `'random'` or `'radial'` (outward inside membrane) |
| `enable_membrane` | `False` | wrap network in a closed triangulated vesicle |
| `membrane_radius`, `membrane_subdivisions` | `0.6, 2` | vesicle size, mesh resolution |
| `membrane_edge_stiffness`, `membrane_bending_stiffness` | `30.0, 2.0` | mesh elasticity |
| `membrane_pressure` | `0.0` | constant outward pressure (positive = inflating) |
| `membrane_filament_coupling_radius`, `..._strength` | `0.08, 60.0` | filament tip ↔ vertex contact FF |
| `rng_seed` | `0` | reproducibility seed (`0` = nondeterministic) |

`make_network_document(interval, **process_config)` returns a
ready-to-run composite document wired to a `RAMEmitter`.

### `MedyanCxxProcess` — real MEDYAN binary

Input port:

| Port | Type | Notes |
| --- | --- | --- |
| `actin_copy` | `maybe[integer]` | optional. When set on an `update(state, ...)` call, the wrapper rewrites the chemistry input's `SPECIESDIFFUSING: AD` line for that interval — a sibling PBG process can therefore drive G-actin availability into MEDYAN in real time |

Output ports (all `overwrite[T]`):

| Port | Type | Notes |
| --- | --- | --- |
| `n_filaments`, `n_linkers`, `n_motors`, `n_branchers` | integer | live counts in the last frame |
| `total_filament_length`, `mean_filament_length` | float | nm |
| `network_span` | float | bead bbox diagonal (nm) |
| `n_membrane_vertices`, `n_membrane_triangles` | integer | when `enable_membrane=True` |
| `membrane_span`, `membrane_mean_radius` | float | nm |
| `cxx_runtime_seconds` | float | wall-clock of the most recent MEDYAN invocation |

Helper accessors:
- `proc.get_last_frame()` — the final `TrajFrame` from the last `update()`.
- `proc.get_last_frames()` — **all** frames parsed during the last
  `update()` (one per `SNAPSHOTTIME`). Use this for animation: a single
  long MEDYAN run gives smooth intra-run evolution with no
  FILAMENTFILE-restart re-gridding artifacts.

Chemistry, in priority order:

```python
# 1. Bundled preset (default 'actin_only', or 'actin_motor_linker')
MedyanCxxProcess(config={'chemistry_preset': 'actin_motor_linker'})

# 2. Inline text — useful for parameter sweeps via per-line edits
MedyanCxxProcess(config={'chemistry_text': 'SPECIESDIFFUSING: AD 200 ...'})

# 3. External file — for chemistries you maintain alongside the wrapper
MedyanCxxProcess(config={'chemistry_path': '/abs/path/chemistryinput.txt'})
```

For one-off MEDYAN keyword overrides (swap the chemistry algorithm,
change a force-field constant, etc.) pass them via `extra_keywords`:

```python
proc = MedyanCxxProcess(
    config={'n_filaments': 5},
    core=core,
    extra_keywords={'CALGORITHM': 'GILLESPIE', 'FBENDINGK': 200.0},
)
```

#### Membrane (deformable vesicle)

Set `enable_membrane=True` and the wrapper emits an S-expression
`(membrane prof1 ...)` + `(init-membrane prof1 ...)` block in the
generated `systeminput.txt` and switches output parsing to `traj.h5`
(the legacy text snapshot doesn't carry membrane geometry). Selected
config:

| Field | Default | Meaning |
| --- | --- | --- |
| `membrane_mesh_kind` | `'ELLIPSOID'` | mesh primitive recognized by `init-membrane` |
| `membrane_center_x/y/z` | `1000.0` | mesh center (nm) |
| `membrane_radius_x/y/z` | `500.0` | mesh radii (nm) |
| `membrane_area_k`, `membrane_bending_k`, `membrane_tension`, `membrane_volume_k` | `400, 50, 0.02, 0.8` | Helfrich + tension + volume FFs |
| `membrane_eq_curv`, `membrane_eq_area_factor` | `0.0, 0.98` | reference geometry |
| `membrane_triangle_bead_k` / `_cutoff` / `_cutoff_mech` | `650, 150, 60` | filament–membrane repulsion FF |

Tighten `gradient_tolerance` (default `0.1`) for small vesicles — the
upstream big-vesicle examples' loose `5.0` value spirals on a
500 nm-radius mesh.

#### Restart fidelity caveat

MEDYAN does **not** expose a documented restart keyword that preserves
linker / motor / brancher binding state across separate invocations.
The pattern this wrapper uses (FILAMENTFILE + `PROJECTIONTYPE: PREDEFINED`)
is the one in MEDYAN's own MATLAB `restart/` scripts:

| state | preserved across `update()` calls? |
| --- | --- |
| filament bead positions | yes — exact, after Euclidean re-gridding |
| filament types | yes |
| diffusing-species copy numbers | no — re-seeded from chemistry each interval |
| linker / motor / brancher bindings | no — re-sampled from chemistry each interval |
| accumulated chemistry-step count | no |

For animation/visualization where this matters, do a **single long
`update()`** with a tight `snapshot_interval` and harvest every frame
via `get_last_frames()` — that's a single continuous MEDYAN run,
no checkpoint-restart, no re-gridding. Multi-`update()` driving is
still the right pattern when you need sibling PBG processes to
modulate state between intervals (the composability demo does
exactly that).

#### Units

`MedyanCxxProcess` uses MEDYAN-native units (**nm**, seconds, pN).
The pure-Python `MedyanProcess` uses **µm**, seconds, pN. If you
compose both in the same Composite, convert at the boundary.

## Architecture

```
                 Pure-Python                          Real MEDYAN
                ─────────────                        ───────────
            ┌─────────────────────┐         ┌────────────────────────┐
no inputs   │ MedyanProcess (PBG) │         │ MedyanCxxProcess (PBG) │  actin_copy
   ────────►│  ┌───────────────┐  │         │  (subprocess driver)   │  ◄─────
            │  │ MedyanEngine  │  │         │                        │
            │  │   filaments   │  │         │   write systeminput.txt│
            │  │   motors      │  │         │   write filaments.txt  │
            │  │   crosslinks  │  │         │   spawn medyan         │
            │  │   membrane    │  │         │   parse snapshot.traj  │
            │  │               │  │         │   parse traj.h5 (HDF5) │
            │  │  step(dt) =   │  │         │                        │
            │  │   poly/walk/  │  │         │  outputs: same metrics │
            │  │   xlink/relax │  │         │  + membrane geometry   │
            │  └───────────────┘  │         │                        │
            └──────────┬──────────┘         └─────────────┬──────────┘
                       │                                  │
                       └────► PBG outputs / stores ◄──────┘
```

Every output port is `overwrite[T]`, so each `update()` call replaces
the scalar/list rather than accumulating deltas.

## Caveats / scope

`MedyanProcess` is **not** an exact reproduction of MEDYAN's C++
engine. It captures the qualitative physics in a small Python
implementation suitable for prototyping PBG compositions, education,
and exploring qualitative regimes. For quantitative production work —
ATP-state hydrolysis, Arp2/3 branching nucleation, large vesicles,
or anything you'd cite in a paper — drive the upstream binary through
`MedyanCxxProcess` instead.

## References

Popov K, Komianos J, Papoian GA. (2016). MEDYAN: Mechanochemical
Simulations of Contraction and Polarity Alignment in Actomyosin Networks.
*PLOS Computational Biology* 12(4): e1004877.

Peskin CS, Odell GM, Oster GF. (1993). Cellular motions and thermal
fluctuations: the Brownian ratchet. *Biophysical Journal* 65(1): 316–24.

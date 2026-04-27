# pbg-medyan

A [process-bigraph](https://github.com/vivarium-collective/process-bigraph)
wrapper for [MEDYAN](https://medyan.org/)-style mechanochemical simulations
of cytoskeletal active networks (actin filaments, myosin II minifilament
motors, alpha-actinin crosslinkers).

## What it does

`pbg-medyan` exposes a single time-driven `MedyanProcess` that simulates
a network of semi-flexible actin filaments under combined chemistry and
mechanics:

- **Brownian-ratchet polymerization** at both filament ends with separate
  on/off rates for the barbed (plus) and pointed (minus) ends; force on
  an end exponentially suppresses growth (Peskin et al., 1993).
- **Hill-style myosin II motor walking** along filaments toward the plus
  end with stall force `Fs`; bound motors generate contractile pulling
  forces between filament pairs.
- **Alpha-actinin-style passive crosslinkers** that tether nearby filaments
  with linear springs and stochastic unbinding.
- **Overdamped force relaxation** of bead positions under cylinder
  stretch, bending, motor, and crosslink forces.
- **Optional triangulated vesicle membrane** (icosphere mesh with edge
  springs + Laplacian bending) that couples to filament plus-ends:
  polymerizing tips push membrane vertices outward and feel a reaction
  force back through the Brownian-ratchet feedback loop.

The implementation is a self-contained Python engine — no compilation of
the upstream MEDYAN C++ code is required — and is sized for interactive
demos (10–100 filaments, 0–100 motors, 0–100 crosslinks) on a laptop.

## Installation

```bash
git clone <this-repo>
cd pbg-medyan
uv venv .venv && source .venv/bin/activate
uv pip install -e .[dev]
```

## Quick start

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
# {'n_filaments': 15, 'n_motors': 22, 'n_crosslinks': 35,
#  'total_length': 8.4, 'network_span': 0.94, ...}
```

## API reference

### `MedyanProcess`

| Field | Direction | Type | Notes |
| --- | --- | --- | --- |
| (no inputs) | — | — | internally driven |
| `n_filaments` | output | `overwrite[integer]` | current filament count |
| `n_motors` | output | `overwrite[integer]` | bound motor count |
| `n_crosslinks` | output | `overwrite[integer]` | bound crosslink count |
| `total_length` | output | `overwrite[float]` | sum of filament arc lengths (μm) |
| `mean_filament_length` | output | `overwrite[float]` | mean filament length (μm) |
| `network_span` | output | `overwrite[float]` | bounding-box diagonal of all beads (μm) |
| `radius_of_gyration` | output | `overwrite[float]` | network R_g (μm) |
| `bending_energy` | output | `overwrite[float]` | sum of bending energy |
| `stretch_energy` | output | `overwrite[float]` | sum of cylinder stretch energy |
| `total_energy` | output | `overwrite[float]` | bending + stretch + membrane bending |
| `membrane_area` | output | `overwrite[float]` | total mesh surface area (μm²) |
| `membrane_volume` | output | `overwrite[float]` | enclosed vesicle volume (μm³) |
| `membrane_mean_radius` | output | `overwrite[float]` | mean vertex distance from center (μm) |
| `membrane_bending_energy` | output | `overwrite[float]` | edge-spring deformation energy |

Key config fields (see `MedyanProcess.config_schema` for the full list):

| Field | Default | Meaning |
| --- | --- | --- |
| `box_size` | `2.0` | cubic domain side length (μm) |
| `n_filaments` | `10` | initial filaments to seed |
| `initial_filament_length` | `0.4` | rest length per filament (μm) |
| `seed_region_fraction` | `0.6` | fraction of box used for clustered seeding |
| `actin_concentration` | `10.0` | free G-actin (μM) |
| `k_on_plus`, `k_off_plus` | `11.6, 1.4` | barbed-end on/off rates |
| `k_on_minus`, `k_off_minus` | `1.3, 0.8` | pointed-end on/off rates |
| `n_motors`, `n_crosslinks` | `0, 0` | initial bound counts |
| `new_motors_per_step` | `0` | binding attempts per `update()` call |
| `new_crosslinks_per_step` | `0` | binding attempts per `update()` call |
| `motor_v0` | `0.2` | unloaded walking velocity (μm/s) |
| `motor_stall_force` | `8.0` | Fs (pN) |
| `motor_force` | `4.0` | per-motor pulling force (pN) |
| `crosslink_stiffness` | `8.0` | spring constant (pN/μm) |
| `cylinder_stiffness` | `20.0` | filament stretch stiffness (pN/μm) |
| `bending_stiffness` | `0.05` | filament bending modulus (pN μm²) |
| `boundary_force_scale` | `1.5` | Brownian-ratchet characteristic force (pN) |
| `drag_coefficient` | `40.0` | overdamped friction |
| `bind_radius` | `0.2` | motor/crosslink capture radius (μm) |
| `n_substeps` | `8` | integration sub-steps per `update()` call |
| `seed_mode` | `'random'` | seed filaments at random or `'radial'` (outward inside membrane) |
| `enable_membrane` | `False` | wrap network in a closed triangulated vesicle |
| `membrane_radius` | `0.6` | initial vesicle radius (μm) |
| `membrane_subdivisions` | `2` | icosphere subdivisions (1 → 42 verts, 2 → 162, 3 → 642) |
| `membrane_edge_stiffness` | `30.0` | edge spring constant (in-plane / area resistance) |
| `membrane_bending_stiffness` | `2.0` | Laplacian-smoothing bending modulus |
| `membrane_pressure` | `0.0` | constant outward pressure (positive = inflating) |
| `membrane_filament_coupling_radius` | `0.08` | filament tip ↔ vertex contact range (μm) |
| `membrane_filament_coupling_strength` | `60.0` | contact force constant (pN/μm) |
| `rng_seed` | `0` | reproducibility seed (`0` = nondeterministic) |

### `make_network_document(interval, **process_config)`

Returns a ready-to-run composite document dict wiring `MedyanProcess`
to scalar stores and a `RAMEmitter` that records the metric time-series.

## Architecture

```
                ┌─────────────────────────┐
                │  MedyanProcess (PBG)    │
                │  (bridge wrapper)       │
                │                         │
   no inputs    │  ┌───────────────────┐  │
   ───────────► │  │   MedyanEngine    │  │
                │  │ (Python physics)  │  │
                │  │                   │  │
                │  │  filaments[]      │  │  outputs ─►  stores
                │  │  motors[]         │  │              ├─ n_filaments
                │  │  crosslinks[]     │  │              ├─ total_length
                │  │                   │  │              ├─ network_span
                │  │  step(dt) =       │  │              ├─ total_energy
                │  │   ├ polymerize    │  │              └─ ...
                │  │   ├ walk_motors   │  │
                │  │   ├ crosslinks    │  │
                │  │   └ relax(forces) │  │
                │  └───────────────────┘  │
                └─────────────────────────┘
```

Each call to `MedyanProcess.update(state, interval)` runs `n_substeps`
mechanochemical sub-steps of size `dt = interval / n_substeps`,
returning the full set of network metrics as overwrite-typed outputs.

### Mapping MEDYAN concepts to PBG

| MEDYAN concept | PBG mapping |
| --- | --- |
| C++ `System` instance | engine state held inside `MedyanProcess` |
| Reaction stepping (Gillespie) | `_polymerize`, `_walk_motors`, `_crosslink_dynamics` |
| Mechanical equilibration | `_relax` (overdamped Euler with `n_substeps`) |
| Per-filament cylinders | `Filament.beads` polyline + per-segment `rest_lengths` |
| Crosslinker / motor binding | `_try_bind_*` in continuous space, capture radius `bind_radius` |
| Brownian-ratchet feedback | `_project_boundary_forces` → `plus_force_proj`/`minus_force_proj` |
| Output observables | scalar fields on `outputs()`, all wrapped in `overwrite[T]` |

## Demo

```bash
source .venv/bin/activate
python demo/demo_report.py
open demo/report.html      # opens in default browser
```

The report runs four configurations (treadmilling polymerization,
actomyosin contractility, **vesicle filopodial protrusion**, and
dendritic crosslinked mesh) and produces a self-contained HTML page with:

- interactive Three.js 3D viewer with time slider + play/pause; in the
  vesicle config the membrane is rendered as a translucent
  triangulated mesh that deforms in response to filament tip pushing
- Plotly time-series charts (length, span, energy, binding counts; or
  vesicle volume + membrane bending energy when a membrane is active)
- color-coded bigraph-viz architecture PNG (left → right layout)
- collapsible JSON tree of the composite document

## Tests

```bash
source .venv/bin/activate
pytest -v
```

The suite covers engine kinetics (treadmilling, depoly, contractility),
PBG instantiation/update/output schema, and full Composite assembly with
RAM-emitter round-trips. All tests run offline.

## Caveats / scope

This wrapper is **not** an exact reproduction of MEDYAN's C++ engine.
It captures the qualitative physics (Brownian-ratchet polymerization,
Hill-style motor walking, alpha-actinin tethering, overdamped relaxation)
in a small Python implementation suitable for:

- prototyping process-bigraph compositions involving cytoskeletal
  components
- demos and education
- exploring qualitative regimes (treadmilling, contractility, crosslinked
  meshes)

For quantitative production simulations of large networks, ATP-state
hydrolysis dynamics, branching nucleation, vesicle membranes, or
boundaries beyond the simplified force-projection used here, run the
upstream [MEDYAN C++ code](https://medyan.org/).

## References

Popov K, Komianos J, Papoian GA. (2016). MEDYAN: Mechanochemical
Simulations of Contraction and Polarity Alignment in Actomyosin Networks.
*PLOS Computational Biology* 12(4): e1004877.

Peskin CS, Odell GM, Oster GF. (1993). Cellular motions and thermal
fluctuations: the Brownian ratchet. *Biophysical Journal* 65(1): 316–24.

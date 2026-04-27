"""MEDYAN-style cytoskeleton Process for process-bigraph.

Wraps the MedyanEngine (Python implementation of MEDYAN's mechanochemical
model) as a time-driven Process using the bridge pattern. The internal
engine is lazily initialized on first use; each update() call advances
the engine by the requested interval and returns network metrics plus
a serialized geometry snapshot.

Outputs use the ``overwrite`` wrapper so the framework replaces the
absolute scalars and lists each step rather than treating them as deltas.
"""

from __future__ import annotations

from process_bigraph import Process

from pbg_medyan.engine import MedyanEngine


class MedyanProcess(Process):
    """Bridge Process wrapping a MEDYAN-style cytoskeletal simulator.

    Simulates a network of semi-flexible actin filaments undergoing
    treadmilling polymerization, with optional myosin II minifilament
    motors walking toward plus ends and alpha-actinin-style passive
    crosslinkers tethering filament pairs. On each ``update()`` the
    engine runs ``n_substeps`` mechanochemical steps of size
    ``dt = interval / n_substeps``.

    Config:
        box_size: side length of cubic simulation domain (um)
        n_filaments: initial number of seeded filaments
        initial_filament_length: rest length of seeded filaments (um)
        actin_concentration: free G-actin concentration (uM)
        k_on_plus, k_off_plus: barbed-end on/off rates (1/uM/s, 1/s)
        k_on_minus, k_off_minus: pointed-end on/off rates
        n_motors: initial bound myosin II motor count
        n_crosslinks: initial bound alpha-actinin crosslinker count
        new_motors_per_step, new_crosslinks_per_step:
            number of new binding attempts per update() call
        motor_v0: load-free walking velocity (um/s)
        motor_stall_force: stall force Fs (pN)
        motor_force: contractile pulling force per motor (pN)
        crosslink_stiffness: linear spring constant (pN/um)
        cylinder_stiffness: filament stretch stiffness (pN/um)
        bending_stiffness: filament bending modulus (pN um^2)
        boundary_force_scale: characteristic Brownian-ratchet force (pN)
        drag_coefficient: overdamped friction coefficient
        bind_radius: capture distance for new motors/crosslinks (um)
        n_substeps: integration substeps per update() call
        rng_seed: integer seed for reproducibility (0 = nondeterministic)
    """

    config_schema = {
        'box_size': {'_type': 'float', '_default': 2.0},
        'n_filaments': {'_type': 'integer', '_default': 10},
        'initial_filament_length': {'_type': 'float', '_default': 0.4},
        'actin_concentration': {'_type': 'float', '_default': 10.0},
        'k_on_plus': {'_type': 'float', '_default': 11.6},
        'k_off_plus': {'_type': 'float', '_default': 1.4},
        'k_on_minus': {'_type': 'float', '_default': 1.3},
        'k_off_minus': {'_type': 'float', '_default': 0.8},
        'n_motors': {'_type': 'integer', '_default': 0},
        'n_crosslinks': {'_type': 'integer', '_default': 0},
        'new_motors_per_step': {'_type': 'integer', '_default': 0},
        'new_crosslinks_per_step': {'_type': 'integer', '_default': 0},
        'motor_v0': {'_type': 'float', '_default': 0.2},
        'motor_stall_force': {'_type': 'float', '_default': 8.0},
        'motor_force': {'_type': 'float', '_default': 4.0},
        'crosslink_stiffness': {'_type': 'float', '_default': 8.0},
        'crosslink_rest_length': {'_type': 'float', '_default': 0.03},
        'cylinder_stiffness': {'_type': 'float', '_default': 20.0},
        'bending_stiffness': {'_type': 'float', '_default': 0.05},
        'boundary_force_scale': {'_type': 'float', '_default': 1.5},
        'drag_coefficient': {'_type': 'float', '_default': 40.0},
        'bind_radius': {'_type': 'float', '_default': 0.2},
        'crosslink_unbind_rate': {'_type': 'float', '_default': 0.05},
        'motor_unbind_rate': {'_type': 'float', '_default': 0.1},
        'seed_region_fraction': {'_type': 'float', '_default': 0.6},
        'n_substeps': {'_type': 'integer', '_default': 8},
        'rng_seed': {'_type': 'integer', '_default': 0},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._engine = None

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'n_filaments': 'overwrite[integer]',
            'n_motors': 'overwrite[integer]',
            'n_crosslinks': 'overwrite[integer]',
            'total_length': 'overwrite[float]',
            'mean_filament_length': 'overwrite[float]',
            'network_span': 'overwrite[float]',
            'radius_of_gyration': 'overwrite[float]',
            'bending_energy': 'overwrite[float]',
            'stretch_energy': 'overwrite[float]',
            'total_energy': 'overwrite[float]',
        }

    def _build_engine(self) -> None:
        if self._engine is not None:
            return
        c = self.config
        engine_kwargs = {
            k: c[k] for k in (
                'box_size', 'n_filaments', 'initial_filament_length',
                'actin_concentration',
                'k_on_plus', 'k_off_plus', 'k_on_minus', 'k_off_minus',
                'n_motors', 'n_crosslinks',
                'motor_v0', 'motor_stall_force', 'motor_force',
                'crosslink_stiffness', 'crosslink_rest_length',
                'cylinder_stiffness', 'bending_stiffness',
                'boundary_force_scale', 'drag_coefficient',
                'bind_radius', 'crosslink_unbind_rate', 'motor_unbind_rate',
                'seed_region_fraction', 'rng_seed',
            )
        }
        self._engine = MedyanEngine(**engine_kwargs)

    def initial_state(self):
        self._build_engine()
        return self._engine.network_metrics()

    def get_engine(self) -> MedyanEngine:
        """Access the underlying engine (e.g., for snapshot()-based viz)."""
        self._build_engine()
        return self._engine

    def update(self, state, interval):
        self._build_engine()
        n_sub = max(1, int(self.config['n_substeps']))
        dt = interval / n_sub
        # Distribute new bindings across the sub-steps
        n_new_m = int(self.config['new_motors_per_step'])
        n_new_x = int(self.config['new_crosslinks_per_step'])
        for i in range(n_sub):
            add_m = (n_new_m // n_sub) + (1 if i < (n_new_m % n_sub) else 0)
            add_x = (n_new_x // n_sub) + (1 if i < (n_new_x % n_sub) else 0)
            self._engine.step(dt, n_new_motors=add_m, n_new_crosslinks=add_x)
        return self._engine.network_metrics()

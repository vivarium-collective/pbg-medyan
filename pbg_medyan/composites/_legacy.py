"""Pre-built composite document factories for MEDYAN-style simulations."""


# Output ports of MedyanProcess that can be wired to stores
_NETWORK_METRIC_PORTS = (
    'n_filaments', 'n_motors', 'n_crosslinks',
    'total_length', 'mean_filament_length',
    'network_span', 'radius_of_gyration',
    'bending_energy', 'stretch_energy', 'total_energy',
    'membrane_area', 'membrane_volume', 'membrane_mean_radius',
    'membrane_bending_energy',
)


def make_network_document(interval=1.0, **process_config):
    """Build a composite document for an actomyosin network simulation.

    Args:
        interval: time between MedyanProcess updates (seconds)
        **process_config: any MedyanProcess.config_schema field

    Returns:
        dict: composite document ready for ``Composite({'state': doc})``
              with the cytoskeleton process, scalar stores, and a
              RAM emitter wired up to record metric time-series.
    """
    outputs = {p: ['stores', p] for p in _NETWORK_METRIC_PORTS}
    emit_schema = {p: 'float' for p in _NETWORK_METRIC_PORTS}
    emit_schema['time'] = 'float'

    emit_inputs = {p: ['stores', p] for p in _NETWORK_METRIC_PORTS}
    emit_inputs['time'] = ['global_time']

    return {
        'cytoskeleton': {
            '_type': 'process',
            'address': 'local:MedyanProcess',
            'config': dict(process_config),
            'interval': interval,
            'inputs': {},
            'outputs': outputs,
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {'emit': emit_schema},
            'inputs': emit_inputs,
        },
    }

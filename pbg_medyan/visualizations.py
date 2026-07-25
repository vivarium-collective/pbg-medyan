"""Visualization Step subclasses for pbg-medyan.

Visualizations follow the pbg-superpowers convention (v0.4.15+):
each subclass overrides `update()` to consume per-step state via wires
(like an Emitter), accumulates history internally, and returns
``{'html': '<rendered figure>'}`` each step. The composite spec wires
the input ports to store paths.

See viva_superpowers.visualization for the base-class contract.
"""
from __future__ import annotations

from viva_superpowers.visualization import Visualization


class NetworkMetricsPlots(Visualization):
    """Time-series HTML plot of MedyanProcess's scalar network metrics.

    Consumes a handful of core scalars (n_filaments, total_length,
    mean_filament_length, network_span, radius_of_gyration,
    total_energy) at each step, accumulates them across calls, and
    emits a Plotly HTML figure on every update. Downstream consumers
    (dashboards, notebook viewers) read the latest 'html' from the
    wired store.
    """

    config_schema = {
        'title': {'_type': 'string', '_default': 'MEDYAN network metrics'},
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One list per consumed scalar; aligned by index across all signals.
        self.times: list[float] = []
        self.history: dict[str, list[float]] = {
            'n_filaments': [],
            'total_length': [],
            'mean_filament_length': [],
            'network_span': [],
            'radius_of_gyration': [],
            'total_energy': [],
        }

    def inputs(self):
        return {
            'n_filaments': 'integer',
            'total_length': 'float',
            'mean_filament_length': 'float',
            'network_span': 'float',
            'radius_of_gyration': 'float',
            'total_energy': 'float',
            'time': 'float',
        }

    def update(self, state, interval=1.0):
        self.times.append(float(state.get('time', len(self.times) * (interval or 1.0))))
        for key in self.history:
            v = state.get(key)
            self.history[key].append(float(v) if v is not None else 0.0)

        title = (self.config or {}).get('title', 'MEDYAN network metrics')
        traces = []
        for key, ys in self.history.items():
            traces.append(
                '{"x":' + repr(self.times) + ',"y":' + repr(ys) +
                ',"type":"scatter","mode":"lines","name":"' + key + '"}'
            )
        html = (
            f'<div id="nmp" style="height:380px"></div>'
            f'<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
            f'<script>Plotly.newPlot("nmp",[{",".join(traces)}],'
            f'{{title:"{title}",margin:{{l:55,r:15,t:35,b:40}},'
            f'xaxis:{{title:"time (s)"}},'
            f'legend:{{orientation:"h",y:-0.2}}}},'
            f'{{responsive:true,displayModeBar:false}});</script>'
        )
        return {'html': html}

"""
Validation figures: modelled capacity factors against ENTSO-E observations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

from plotlib.io import figure_size
from plotlib.palettes import hue_kwargs
from plotlib.style import despine, legend_above

__all__ = ["FIGURES"]

#: metric column -> (display label, colormap, centre for the diverging scale)
_METRICS = {
    "pearson_r": ("Pearson $r$ [-]", "viridis", None),
    "mbe_pct": ("Mean bias error [%]", "RdBu_r", 0.0),
    "nrmse": ("NRMSE [-]", "magma_r", None),
}


def _require(data):
    if data.validation is None:
        raise FileNotFoundError(
            "validation_metrics_cf.csv not found. Run validate_sensitivity_vs_entsoe.py."
        )
    return data.validation


def _heatmap(data, tech: str, metric: str):
    """Country x scenario heatmap of *metric* for one technology."""
    label, cmap, centre = _METRICS[metric]
    frame = _require(data)
    frame = frame[frame["tech"] == tech]
    if frame.empty:
        raise ValueError(f"No validation rows for tech {tech!r}.")

    grid = frame.pivot_table(
        index="country", columns="scenario", values=metric, aggfunc="mean"
    )

    fig, ax = plt.subplots(
        figsize=figure_size("FULL_WIDTH", 0.35 * len(grid) + 2.0), layout="constrained"
    )
    sns.heatmap(
        grid,
        annot=False,
        cmap=cmap,
        center=centre,
        cbar_kws={"label": label},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{tech.capitalize()} wind")
    return fig


def _bars(data, tech: str, metric: str):
    """Metric per scenario, averaged over countries."""
    label, _, _ = _METRICS[metric]
    frame = _require(data)
    frame = frame[frame["tech"] == tech]
    hue_opts = hue_kwargs(frame["scenario"].unique())

    fig, ax = plt.subplots(figsize=figure_size("HALF_WIDTH", 5.5), layout="constrained")
    sns.barplot(data=frame, x="scenario", y=metric, ax=ax, **hue_opts)
    ax.set_xlabel("")
    ax.set_ylabel(label)
    ax.set_xticks([])
    ax.set_title(f"{tech.capitalize()} wind")
    legend_above(ax, hue_opts["hue_order"])
    despine(ax=ax)
    return fig


def _register():
    figures = {}
    for tech in ("onshore", "offshore"):
        for metric in _METRICS:
            figures[f"heatmap_cf_{tech}_{metric}.png"] = (
                lambda data, summary, t=tech, m=metric: _heatmap(data, t, m)
            )
            figures[f"bars_cf_{tech}_{metric}.png"] = (
                lambda data, summary, t=tech, m=metric: _bars(data, t, m)
            )
    return figures


FIGURES = _register()

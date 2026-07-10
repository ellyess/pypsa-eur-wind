"""
Plots against spatial resolution and against capacity density.

These carry the central claim of the wake manuscript: the tiered-density
formulation gives a wake loss that does not move as the offshore regions are
refined, while the capacity-tiered formulation collapses towards zero. The
x-axis convention is always coarse -> fine (log, inverted), enforced by
:func:`plotting_style.apply_spatial_resolution_axis`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

from plotlib.io import figure_size
from plotlib.palettes import hue_kwargs
from plotlib.style import apply_spatial_resolution_axis, despine, legend_above

__all__ = ["plot_vs_density", "plot_vs_resolution"]


def _resolution_guides(ax, xvals) -> None:
    """Dashed guides at the coarsest and finest resolution, without labels."""
    values = np.asarray(xvals, dtype=float)
    for bound in (np.nanmin(values), np.nanmax(values)):
        ax.axvline(bound, linestyle="--", linewidth=0.8, color="grey", alpha=0.7)


def plot_vs_resolution(
    df,
    y: str,
    *,
    x: str = "resolution",
    hue: str = "scenario",
    ylabel: str | None = None,
    markers: bool = True,
    height_cm: float = 6.0,
    width: str = "FULL_WIDTH",
    ax=None,
):
    """Plot *y* against offshore spatial resolution, one line per scenario.

    The x-axis is log-scaled and inverted, so it reads coarse on the left and
    fine on the right. A flat line therefore means "resolution-invariant".
    """
    missing = {x, y, hue} - set(df.columns)
    if missing:
        raise KeyError(f"plot_vs_resolution needs columns {sorted(missing)}.")

    data = df.dropna(subset=[y]).sort_values(x)
    hue_opts = hue_kwargs(data[hue].unique(), hue=hue)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figure_size(width, height_cm), layout="constrained"
        )
    else:
        fig = ax.get_figure()

    sns.lineplot(data=data, x=x, y=y, marker="o", ax=ax, **hue_opts)

    # `apply_spatial_resolution_axis` already writes "Coarse"/"Fine" below the
    # axis; `add_resolution_markers` would write them again inside it, on top
    # of the data. Draw only the guide lines here.
    apply_spatial_resolution_axis(ax, annotate=True)
    if markers:
        _resolution_guides(ax, data[x])
    ax.set_ylabel(ylabel or y.replace("_", " ").capitalize())

    legend_above(ax, hue_opts["hue_order"])
    despine(ax=ax)
    return fig


def plot_vs_density(
    df,
    y: str,
    *,
    x: str = "density",
    hue: str = "scenario",
    xlabel: str = r"Capacity density [MW km$^{-2}$]",
    ylabel: str | None = None,
    marker: str | None = "o",
    height_cm: float = 6.0,
    width: str = "FULL_WIDTH",
    ax=None,
):
    """Plot *y* against offshore capacity density, one line per scenario.

    This is the companion to :func:`plot_vs_resolution`: the tiered-density
    formulation is a function of density by construction, so all its
    resolution points collapse onto a single curve here.

    Pass ``marker=None`` for densely sampled analytic curves.
    """
    missing = {x, y, hue} - set(df.columns)
    if missing:
        raise KeyError(f"plot_vs_density needs columns {sorted(missing)}.")

    data = df.dropna(subset=[y]).sort_values(x)
    hue_opts = hue_kwargs(data[hue].unique(), hue=hue)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figure_size(width, height_cm), layout="constrained"
        )
    else:
        fig = ax.get_figure()

    sns.lineplot(data=data, x=x, y=y, marker=marker, ax=ax, **hue_opts)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or y.replace("_", " ").capitalize())

    legend_above(ax, hue_opts["hue_order"])
    despine(ax=ax)
    return fig

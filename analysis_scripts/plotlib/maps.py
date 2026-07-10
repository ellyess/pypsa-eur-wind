"""
Choropleth maps of offshore regions.

Sequential quantities (capacity density) use a perceptually uniform sequential
colormap; signed quantities (differences against the baseline) use a diverging
colormap centred on zero, so that "no change" is always the neutral colour.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plotlib.io import figure_size
from plotlib.style import despine
from thesis_colors import canon, label

__all__ = ["plot_choropleth", "plot_delta_map", "plot_value_map"]


def plot_choropleth(
    gdf,
    value_col: str,
    *,
    title: str | None = None,
    cbar_label: str | None = None,
    diverging: bool = False,
    cmap: str | None = None,
    height_cm: float = 8.0,
    width: str = "MAP_WIDTH",
    ax=None,
):
    """Draw *gdf* coloured by *value_col*.

    When *diverging* is true the colour scale is centred on zero, so the sign
    of the difference is readable at a glance.
    """
    if value_col not in gdf.columns:
        raise KeyError(f"plot_choropleth: no column {value_col!r}.")

    values = gdf[value_col].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"plot_choropleth: {value_col!r} has no finite values.")

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figure_size(width, height_cm), layout="constrained"
        )
    else:
        fig = ax.get_figure()

    norm = None
    if diverging:
        bound = float(np.abs(finite).max()) or 1.0
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)

    gdf.plot(
        column=value_col,
        ax=ax,
        cmap=cmap or ("RdBu_r" if diverging else "viridis"),
        norm=norm,
        legend=True,
        legend_kwds={"label": cbar_label or value_col, "shrink": 0.7},
        edgecolor="white",
        linewidth=0.15,
    )

    ax.set_axis_off()
    if title:
        ax.set_title(title)
    despine(ax=ax)
    return fig


def plot_value_map(gdf, scenario: str, *, value_col: str = "cap_mw_per_km2", **kwargs):
    """Map *value_col* for one scenario (sequential scale)."""
    scenario = canon(scenario)
    subset = _select(gdf, scenario)
    kwargs.setdefault("title", f"Offshore capacity density: {label(scenario)}")
    kwargs.setdefault("cbar_label", r"Capacity density [MW km$^{-2}$]")
    return plot_choropleth(subset, value_col, **kwargs)


def plot_delta_map(
    gdf,
    scenario: str,
    *,
    baseline: str = "base",
    value_col: str = "cap_mw_per_km2",
    id_col: str = "region_id",
    **kwargs,
):
    """Map the difference in *value_col* between *scenario* and *baseline*."""
    scenario, baseline = canon(scenario), canon(baseline)
    for column in ("scenario", id_col, value_col):
        if column not in gdf.columns:
            raise KeyError(f"plot_delta_map: no column {column!r}.")

    frame = gdf.copy()
    frame["scenario"] = frame["scenario"].astype(str).map(canon)

    reference = frame.loc[frame["scenario"] == baseline, [id_col, value_col]].rename(
        columns={value_col: "_baseline"}
    )
    if reference.empty:
        raise ValueError(f"plot_delta_map: no rows for baseline {baseline!r}.")

    subset = frame[frame["scenario"] == scenario].merge(
        reference, on=id_col, how="left"
    )
    subset["delta"] = subset[value_col] - subset["_baseline"]

    kwargs.setdefault(
        "title", f"Δ offshore capacity density vs {label(baseline)}: {label(scenario)}"
    )
    kwargs.setdefault("cbar_label", r"Δ capacity density [MW km$^{-2}$]")
    kwargs["diverging"] = True
    return plot_choropleth(subset, "delta", **kwargs)


def _select(gdf, scenario: str):
    if "scenario" not in gdf.columns:
        return gdf
    subset = gdf[gdf["scenario"].astype(str).map(canon) == scenario]
    if subset.empty:
        raise ValueError(f"No rows for scenario {scenario!r}.")
    return subset

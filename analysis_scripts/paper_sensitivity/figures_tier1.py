"""
Tier 1 figures: the North Sea bias x wake x resolution sweep.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotlib.distributions import plot_distribution
from plotlib.io import figure_size
from plotlib.resolution import plot_vs_resolution
from plotlib.style import despine

__all__ = ["FIGURES"]

_DOMAIN = "northsea"


def _tier1(summary):
    return summary[summary["domain"] == _DOMAIN]


# ---------------------------------------------------------------------------
# Lines against resolution
# ---------------------------------------------------------------------------


def capacity_on_off_vs_resolution(data, summary):
    """Onshore and offshore build-out side by side."""
    frame = _tier1(summary)
    fig, axes = plt.subplots(
        1, 2, figsize=figure_size("FULL_WIDTH", 6.0), layout="constrained"
    )
    for ax, column, title in zip(
        axes,
        ("offwind_cap_gw", "onwind_cap_gw"),
        ("Offshore wind", "Onshore wind"),
        strict=True,
    ):
        plot_vs_resolution(frame, y=column, ylabel="Capacity [GW]", ax=ax)
        ax.set_title(title)
    axes[1].get_legend().remove()
    return fig


def objective_vs_resolution(data, summary):
    return plot_vs_resolution(
        _tier1(summary), y="objective_beur", ylabel="System cost [bn EUR]"
    )


def transmission_vs_resolution(data, summary):
    return plot_vs_resolution(
        _tier1(summary), y="trans_exp_twkm", ylabel="Transmission expansion [TW km]"
    )


def marginal_value_proxy_offwind(data, summary):
    """Cost per GW of offshore build-out, a proxy for its marginal value."""
    frame = _tier1(summary).copy()
    frame["marginal_value"] = frame["objective_beur"] / frame["offwind_cap_gw"]
    return plot_vs_resolution(
        frame,
        y="marginal_value",
        ylabel=r"System cost per offshore GW [bn EUR GW$^{-1}$]",
    )


# ---------------------------------------------------------------------------
# Capacity-factor distributions
# ---------------------------------------------------------------------------


def _cf_panel(data, column, label):
    """One ECDF per technology, coloured by scenario."""
    frame = data.tier1_cf
    techs = sorted(frame["tech"].unique())
    fig, axes = plt.subplots(
        1, len(techs), figsize=figure_size("FULL_WIDTH", 6.0), layout="constrained"
    )
    for ax, tech in zip(np.atleast_1d(axes), techs, strict=True):
        plot_distribution(
            frame[frame["tech"] == tech], value=column, kind="cdf", xlabel=label, ax=ax
        )
        ax.set_title(tech)
    for ax in np.atleast_1d(axes)[1:]:
        if ax.get_legend():
            ax.get_legend().remove()
    return fig


def cf_avail(data, summary):
    return _cf_panel(data, "avail_cf", "Available capacity factor [-]")


def cf_disp(data, summary):
    return _cf_panel(data, "disp_cf", "Dispatched capacity factor [-]")


def cf_curt(data, summary):
    return _cf_panel(data, "curt_cf", "Curtailed capacity factor [-]")


# ---------------------------------------------------------------------------
# Factor importance and interactions
# ---------------------------------------------------------------------------


def _tornado(summary, column, xlabel):
    """Spread of *column* attributable to each factor, widest first.

    For every factor (bias correction, wake model, spatial resolution) the bar
    is the range of *column* across that factor's levels, holding nothing else
    fixed. It answers "which assumption moves this number most".
    """
    frame = _tier1(summary)
    spreads = []
    for factor in ("bias", "wake", "resolution"):
        if factor not in frame.columns:
            continue
        grouped = frame.groupby(factor)[column].mean()
        spreads.append(
            {
                "factor": factor,
                "spread": float(grouped.max() - grouped.min()),
                "levels": len(grouped),
            }
        )

    spread = pd.DataFrame(spreads).sort_values("spread")

    fig, ax = plt.subplots(figsize=figure_size("HALF_WIDTH", 5.0), layout="constrained")
    ax.barh(spread["factor"], spread["spread"], color="#4D4D4D", height=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    despine(ax=ax)
    return fig


def tornado_system_cost(data, summary):
    return _tornado(summary, "objective_beur", "Spread in system cost [bn EUR]")


def tornado_offwind_cap(data, summary):
    return _tornado(summary, "offwind_cap_gw", "Spread in offshore capacity [GW]")


def tornado_transmission(data, summary):
    return _tornado(summary, "trans_exp_twkm", "Spread in transmission [TW km]")


def _heatmap(summary, resolution, column, cbar_label):
    frame = _tier1(summary)
    frame = frame[frame["resolution"] == resolution]
    if frame.empty:
        raise ValueError(f"No tier-1 rows at resolution {resolution}.")

    grid = frame.pivot_table(index="bias", columns="wake", values=column, aggfunc="mean")

    fig, ax = plt.subplots(figsize=figure_size("HALF_WIDTH", 5.5), layout="constrained")
    sns.heatmap(
        grid,
        annot=True,
        fmt=".3g",
        cmap="viridis",
        cbar_kws={"label": cbar_label},
        ax=ax,
    )
    ax.set_xlabel("Wake model")
    ax.set_ylabel("Bias correction")
    ax.set_title(f"$A_{{region}}^{{max}}$ = {resolution:,} km$^2$")
    return fig


def heatmap_cost_fine(data, summary):
    finest = _tier1(summary)["resolution"].min()
    return _heatmap(summary, finest, "objective_beur", "System cost [bn EUR]")


def heatmap_cost_coarse(data, summary):
    coarsest = _tier1(summary)["resolution"].max()
    return _heatmap(summary, coarsest, "objective_beur", "System cost [bn EUR]")


FIGURES = {
    "fig_capacity_on_off_vs_resolution.png": capacity_on_off_vs_resolution,
    "fig_objective_vs_resolution.png": objective_vs_resolution,
    "fig_trans_exp_vs_resolution.png": transmission_vs_resolution,
    "fig_marginal_value_proxy_offwind.png": marginal_value_proxy_offwind,
    "fig_cf_avail_2x2.png": cf_avail,
    "fig_cf_disp_2x2.png": cf_disp,
    "fig_cf_curt_2x2.png": cf_curt,
    "fig_tornado_system_cost.png": tornado_system_cost,
    "fig_tornado_offwind_cap.png": tornado_offwind_cap,
    "fig_tornado_transmission.png": tornado_transmission,
    "fig_heatmap_cost_fine.png": heatmap_cost_fine,
    "fig_heatmap_cost_coarse.png": heatmap_cost_coarse,
}

"""
Tier 2 figures: the Europe-wide confirmatory run, and the cross-domain panels
that show the North Sea conclusions carry to continental scale.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plotlib.distributions import plot_distribution
from plotlib.io import figure_size
from plotlib.resolution import plot_vs_resolution

__all__ = ["FIGURES"]


def _tier2(summary):
    return summary[summary["domain"] == "europe"]


def capacity_on_off_vs_resolution(data, summary):
    frame = _tier2(summary)
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
        _tier2(summary), y="objective_beur", ylabel="System cost [bn EUR]"
    )


def transmission_vs_resolution(data, summary):
    return plot_vs_resolution(
        _tier2(summary), y="trans_exp_twkm", ylabel="Transmission expansion [TW km]"
    )


def sector_coupling_h2_vs_resolution(data, summary):
    """Hydrogen production, the sector-coupled channel the North Sea run lacks."""
    frame = data.tier2
    if "h2_prod_twh" not in frame.columns:
        raise KeyError("tier2_metrics.csv has no h2_prod_twh column.")
    return plot_vs_resolution(
        frame, y="h2_prod_twh", ylabel="Hydrogen production [TWh]"
    )


def _cf_ecdf(data, column, label):
    frame = data.tier2_cf
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


def cf_ecdf_disp(data, summary):
    return _cf_ecdf(data, "disp_cf", "Dispatched capacity factor [-]")


def cf_ecdf_curt(data, summary):
    return _cf_ecdf(data, "curt_cf", "Curtailed capacity factor [-]")


# ---------------------------------------------------------------------------
# Cross-domain: does the North Sea result carry to Europe?
# ---------------------------------------------------------------------------


def _cross_domain(summary, column, ylabel):
    fig, axes = plt.subplots(
        1, 2, figsize=figure_size("FULL_WIDTH", 6.0), layout="constrained"
    )
    for ax, (domain, title) in zip(
        axes, (("northsea", "North Sea"), ("europe", "Europe")), strict=True
    ):
        frame = summary[summary["domain"] == domain]
        if frame.empty:
            raise ValueError(f"No rows for domain {domain!r}.")
        plot_vs_resolution(frame, y=column, ylabel=ylabel, ax=ax)
        ax.set_title(title)
    axes[1].get_legend().remove()
    axes[1].set_ylabel("")
    return fig


def europe_vs_northsea_offwind_cap(data, summary):
    """The panel the wake manuscript reuses for its transferability claim."""
    return _cross_domain(summary, "offwind_cap_gw", "Offshore wind capacity [GW]")


def europe_vs_northsea_objective(data, summary):
    return _cross_domain(summary, "objective_beur", "System cost [bn EUR]")


def europe_vs_northsea_trans(data, summary):
    return _cross_domain(summary, "trans_exp_twkm", "Transmission expansion [TW km]")


def europe_vs_northsea_curt(data, summary):
    return _cross_domain(summary, "offwind_curt_frac", "Offshore curtailment [-]")


FIGURES = {
    "fig_capacity_on_off_vs_resolution_tier2.png": capacity_on_off_vs_resolution,
    "fig_objective_vs_resolution_tier2.png": objective_vs_resolution,
    "fig_transmission_vs_resolution_tier2.png": transmission_vs_resolution,
    "fig_sector_coupling_h2_vs_resolution_tier2.png": sector_coupling_h2_vs_resolution,
    "fig_cf_ecdf_disp_tier2.png": cf_ecdf_disp,
    "fig_cf_ecdf_curt_tier2.png": cf_ecdf_curt,
    "fig_europe_vs_northsea_offwind_cap.png": europe_vs_northsea_offwind_cap,
    "fig_europe_vs_northsea_objective.png": europe_vs_northsea_objective,
    "fig_europe_vs_northsea_trans.png": europe_vs_northsea_trans,
    "fig_europe_vs_northsea_curt.png": europe_vs_northsea_curt,
}

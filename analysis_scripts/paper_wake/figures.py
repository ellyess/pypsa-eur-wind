"""
Every figure the wake manuscript includes, keyed by its filename.

:data:`FIGURES` maps the exact filename in ``Manuscript-PyPSA_Eur-Wake/images``
to the function that builds it, so ``make_paper.py`` can regenerate one figure
or all of them without a lookup table living in two places.

A few figures need the solved networks or the region geometries rather than the
extracted CSVs. Those are listed in :data:`EXTERNAL` together with the script
that produces them, so nothing silently goes missing from the manuscript.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plotlib.distributions import plot_distribution
from plotlib.io import figure_size
from plotlib.palettes import hue_kwargs
from plotlib.resolution import plot_vs_density, plot_vs_resolution
from plotlib.style import despine, legend_above

__all__ = ["EXTERNAL", "FIGURES", "build"]

_RESOLUTION_LABEL = (
    r"Spatial resolution ($A_{region}^{max}$) [km$^{2}$], coarse $\rightarrow$ fine"
)


# ---------------------------------------------------------------------------
# Distribution figures
# ---------------------------------------------------------------------------


def _coarse_to_fine(frame) -> list:
    """Resolution levels in the order every figure in the paper uses."""
    return sorted(frame["resolution"].unique(), reverse=True)


def wake_loss_box(data, summary):
    return plot_distribution(
        data.wake_losses,
        value="wake_loss",
        kind="box",
        split="resolution",
        split_order=_coarse_to_fine(data.wake_losses),
        split_label=_RESOLUTION_LABEL,
        xlabel="Wake loss [-]",
    )


def _cf_box(data, column, xlabel):
    return plot_distribution(
        data.cf,
        value=column,
        kind="box",
        split="resolution",
        split_order=_coarse_to_fine(data.cf),
        split_label=_RESOLUTION_LABEL,
        xlabel=xlabel,
    )


def available_cf_box(data, summary):
    return _cf_box(data, "available_cf", "Available capacity factor [-]")


def dispatch_cf_box(data, summary):
    return _cf_box(data, "dispatch_cf", "Dispatched capacity factor [-]")


def curtailment_cf_box(data, summary):
    return _cf_box(data, "curtailment_cf", "Curtailed capacity factor [-]")


# ---------------------------------------------------------------------------
# The centrepiece: invariance
# ---------------------------------------------------------------------------


def wake_loss_vs_resolution(data, summary):
    """Wake loss against offshore resolution.

    A flat line means the formulation is resolution-invariant. This is the
    figure the manuscript's central claim rests on.
    """
    return plot_vs_resolution(
        summary, y="wake_loss_pct", ylabel="Mean wake loss [%]", height_cm=6.5
    )


def wake_loss_vs_density(data, summary):
    """Wake loss against capacity density, pooled over resolutions.

    The tiered-density formulation is a function of density by construction,
    so its resolution points collapse onto one curve here.
    """
    frame = data.wake_density.copy()
    frame["wake_loss_pct"] = frame["wake_loss"] * 100.0

    # Bin the densities so the lines are readable; seaborn averages within bins.
    bins = np.linspace(0, frame["density_mw_per_km2"].quantile(0.99), 25)
    frame["density"] = pd.cut(
        frame["density_mw_per_km2"], bins=bins, labels=bins[:-1]
    ).astype(float)
    binned = (
        frame.dropna(subset=["density"])
        .groupby(["scenario", "density"], as_index=False)["wake_loss_pct"]
        .mean()
    )
    return plot_vs_density(
        binned, y="wake_loss_pct", ylabel="Mean wake loss [%]", height_cm=6.5
    )


def wake_models_density_comparison(data, summary):
    """The wake-loss formulations as functions of capacity density, for two
    reference region areas.

    This is a property of the formulations, not of a model run, so it is
    computed analytically from the wake coefficients rather than the results.
    Delegates to :func:`compare_wake_runs.plot_wake_models_density_two_areas`,
    which is the single source of truth for this figure.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for sub in ("scripts", "analysis_scripts"):
        path = str(root / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    from compare_wake_runs import (  # noqa: PLC0415
        plot_wake_models_density_two_areas,
    )

    return plot_wake_models_density_two_areas()


# ---------------------------------------------------------------------------
# System-level figures
# ---------------------------------------------------------------------------


def resolution_offshore_capacity_gw(data, summary):
    return plot_vs_resolution(
        summary,
        y="offshore_capacity_gw",
        ylabel="Offshore wind capacity [GW]",
        height_cm=6.5,
    )


def _bar(summary, y, ylabel, *, baseline_line=False):
    frame = summary.copy()
    # Bars read coarse -> fine, matching the inverted log axis of every
    # resolution line plot in the paper.
    order = [f"{value:,}" for value in sorted(frame["resolution"].unique(), reverse=True)]
    frame["resolution"] = frame["resolution"].astype(int).map("{:,}".format)
    hue_opts = hue_kwargs(frame["scenario"].unique())

    fig, ax = plt.subplots(figsize=figure_size("FULL_WIDTH", 6.0), layout="constrained")
    sns.barplot(data=frame, x="resolution", y=y, order=order, ax=ax, **hue_opts)
    ax.set_xlabel(_RESOLUTION_LABEL)
    ax.set_ylabel(ylabel)
    if baseline_line:
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
    legend_above(ax, hue_opts["hue_order"])
    despine(ax=ax)
    return fig


def system_cost_comparison(data, summary):
    return _bar(summary, "total_cost_beur", "System cost [bn EUR]")


def system_cost_delta_pct(data, summary):
    frame = summary[summary["scenario"] != "base"]
    return _bar(
        frame,
        "cost_delta_pct",
        r"$\Delta$ system cost vs baseline [%]",
        baseline_line=True,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Manuscript filename -> builder. Every builder takes (WakeData, summary).
FIGURES = {
    "wake_loss_box.pdf": wake_loss_box,
    "available_cf_box.pdf": available_cf_box,
    "dispatch_cf_box.pdf": dispatch_cf_box,
    "curtailment_cf_box.pdf": curtailment_cf_box,
    "wake_loss_vs_resolution.pdf": wake_loss_vs_resolution,
    "wake_loss_vs_density.pdf": wake_loss_vs_density,
    "wake_models_density_comparison.pdf": wake_models_density_comparison,
    "resolution_offshore_capacity_gw.pdf": resolution_offshore_capacity_gw,
    "system_cost_comparison.pdf": system_cost_comparison,
    "system_cost_delta_pct.pdf": system_cost_delta_pct,
}

#: Figures that need the solved networks, the region geometries, or a separate
#: extraction, and the command that builds them. `make_paper.py` reports these
#: rather than pretending the manuscript is complete without them.
#:
#: The keys are the filenames as the manuscript cites them, so a figure that is
#: embedded in a different format than the producing script writes, or under a
#: different name, records that extra step here rather than leaving the gap
#: silent.
EXTERNAL = {
    "capacity_mix.pdf": "wake_extra_plots.py",
    "energy_mix.pdf": "wake_extra_plots.py",
    "capacity_density_delta_maps.pdf": "compare_wake_runs.py cap-delta-map",
    "region_splits_offshore.png": "compare_spatial_runs.py --plot-region-splits --offshore-only",
    "new_more_breakpoints_fit.png": "fit_new_more_breakpoints.py",
    # tier-2 writes this panel as .png; the manuscript embeds a PDF of it.
    "fig_europe_vs_northsea_offwind_cap.pdf": "compare_sensitivity_runs_tier2.py (writes .png; converted to PDF)",
    # same builder as wake_loss_vs_resolution.pdf, run against the floored
    # extraction (the offshore-floor runs) and kept under a distinct name.
    "wake_loss_vs_resolution_floored.pdf": "make_paper.py --data-dir <floored extraction>, renamed *_floored.pdf",
}


def build(name: str, data, summary):
    """Build the figure registered under *name*."""
    if name not in FIGURES:
        raise KeyError(
            f"Unknown figure {name!r}. Known: {sorted(FIGURES)}. "
            f"External: {sorted(EXTERNAL)}."
        )
    return FIGURES[name](data, summary)


def missing_from(names) -> list[str]:
    """Return the manuscript figures *names* that this module cannot build."""
    return [name for name in names if name not in FIGURES]

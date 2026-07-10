"""
One parametrised distribution plot.

``compare_wake_runs.py`` grew twelve near-identical functions: four metrics
(wake loss, available CF, dispatched CF, curtailed CF) times three chart
kinds (PDF, CDF, box), differing only in the column they read and the axis
label they write. :func:`plot_distribution` replaces all twelve.

Every series is coloured and ordered through ``plotlib.palettes``, so the
wake formulations keep their Okabe-Ito colours and their Baseline ->
Uniform -> Tiered-capacity -> Tiered-density order in all three kinds.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from plotlib.io import figure_size
from plotlib.palettes import hue_kwargs
from plotlib.style import despine, legend_above

__all__ = ["KINDS", "plot_distribution"]

KINDS = ("pdf", "cdf", "box")

#: Line styles cycled over the split dimension of a PDF/CDF.
_DASHES = ("-", "--", ":", "-.")

#: Axis labels for the metrics the two manuscripts plot.
METRIC_LABELS = {
    "wake_loss": "Wake loss multiplier [-]",
    "cf": "Available capacity factor [-]",
    "available_cf": "Available capacity factor [-]",
    "dispatch_cf": "Dispatched capacity factor [-]",
    "curtailment_cf": "Curtailed capacity factor [-]",
}


def _value_label(value: str, override: str | None) -> str:
    if override is not None:
        return override
    return METRIC_LABELS.get(value, value.replace("_", " ").capitalize())


def plot_distribution(
    df,
    value: str,
    kind: str = "pdf",
    *,
    hue: str = "scenario",
    split: str | None = None,
    xlabel: str | None = None,
    bins: int = 40,
    height_cm: float = 6.0,
    width: str = "FULL_WIDTH",
    ax=None,
):
    """Plot the distribution of *value*, one series per *hue* level.

    Parameters
    ----------
    df : DataFrame
        Long-form data with at least the *hue* and *value* columns.
    value : str
        Column to plot, e.g. ``"wake_loss"`` or ``"dispatch_cf"``.
    kind : {"pdf", "cdf", "box"}
        ``pdf`` draws a step histogram of the density, ``cdf`` an empirical
        CDF, ``box`` a boxplot. For ``box``, *split* becomes the x-axis.
    hue : str
        Column naming the series, usually the scenario / wake formulation.
    split : str, optional
        Second grouping column, e.g. the spatial resolution. For ``box`` it
        becomes the x-axis; for ``pdf``/``cdf`` it becomes the line style.
    xlabel : str, optional
        Overrides the axis label looked up from :data:`METRIC_LABELS`.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes instead of creating a figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown kind {kind!r}. Expected one of {KINDS}.")
    missing = {hue, value} - set(df.columns)
    if missing:
        raise KeyError(f"plot_distribution needs columns {sorted(missing)}.")
    if split is not None and split not in df.columns:
        raise KeyError(f"plot_distribution: no split column {split!r}.")

    data = df.dropna(subset=[value])
    hue_opts = hue_kwargs(data[hue].unique(), hue=hue)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=figure_size(width, height_cm), layout="constrained"
        )
    else:
        fig = ax.get_figure()

    split_styles: list[tuple] = []

    if kind in ("pdf", "cdf"):
        # Neither histplot nor ecdfplot accepts a second grouping dimension,
        # so a split is drawn as a line style and gets its own legend.
        groups = (
            [(None, data)]
            if split is None
            else list(data.groupby(split, sort=True, observed=True))
        )
        for index, (split_value, subset) in enumerate(groups):
            dash = _DASHES[index % len(_DASHES)]
            line_kwargs = {} if split is None else {"linestyle": dash}
            common = dict(legend=index == 0, ax=ax, **hue_opts, **line_kwargs)

            if kind == "pdf":
                sns.histplot(
                    data=subset,
                    x=value,
                    stat="density",
                    common_norm=False,
                    common_bins=True,
                    bins=bins,
                    element="step",
                    fill=False,
                    **common,
                )
            else:
                sns.ecdfplot(data=subset, x=value, **common)

            if split is not None:
                split_styles.append((split_value, dash))

        ax.set_ylabel("Density [-]" if kind == "pdf" else "CDF [-]")
        ax.set_xlabel(_value_label(value, xlabel))
        ax.set_xlim(left=0)
        if kind == "cdf":
            ax.set_ylim(0, 1)
    else:  # box
        sns.boxplot(
            data=data,
            x=split,
            y=value,
            showfliers=False,
            linewidth=0.6,
            ax=ax,
            **hue_opts,
        )
        ax.set_ylabel(_value_label(value, xlabel))
        if split is None:
            ax.set_xlabel("")

    legend_above(ax, hue_opts["hue_order"])
    if split_styles:
        _add_split_legend(ax, split, split_styles)
    despine(ax=ax)
    return fig


def _add_split_legend(ax, split: str, split_styles) -> None:
    """Add a second legend keying the line styles to the split values.

    ``Axes.legend`` replaces any previous legend, so the hue legend has to be
    re-attached as a standalone artist for both to survive.
    """
    handles = [
        Line2D([], [], color="black", linestyle=dash, label=str(value))
        for value, dash in split_styles
    ]
    hue_legend = ax.get_legend()
    ax.legend(
        handles=handles,
        title=split.replace("_", " "),
        loc="upper right",
        frameon=False,
    )
    if hue_legend is not None:
        ax.add_artist(hue_legend)

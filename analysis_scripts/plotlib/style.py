"""
Single entry point for the figure theme.

Seaborn's ``set_theme`` overwrites font family, sizes and spine settings,
so calling it anywhere after ``thesis_plot_style()`` silently undoes the
thesis rcParams. (``compare_spatial_runs.py`` used to do exactly that.)

Here the theme is built *from* the thesis rcParams and applied once, so
seaborn and matplotlib cannot disagree.
"""

from __future__ import annotations

import seaborn as sns

from plotting_style import (
    add_resolution_markers,
    apply_spatial_resolution_axis,
    format_axes_standard,
    thesis_plot_style,
)

__all__ = [
    "add_resolution_markers",
    "apply_spatial_resolution_axis",
    "despine",
    "format_axes_standard",
    "legend_above",
    "style_constants",
    "use_style",
]

# rcParams that seaborn's own theme would otherwise clobber, and that the
# thesis style depends on. Asserted by the tests.
_PROTECTED = (
    "font.family",
    "font.size",
    "axes.labelsize",
    "xtick.labelsize",
    "ytick.labelsize",
    "axes.spines.top",
    "axes.spines.right",
    "pdf.fonttype",
    "savefig.dpi",
)

_CONSTANTS: dict | None = None


def use_style(**kwargs) -> dict:
    """Apply the thesis theme to matplotlib *and* seaborn, once.

    Returns the dict of layout constants from
    :func:`plotting_style.thesis_plot_style` (``cm``, ``FULL_WIDTH``, ...).

    Any keyword arguments are forwarded to ``thesis_plot_style``.
    """
    global _CONSTANTS

    # Seaborn goes first and supplies what the thesis style does not set
    # (its "ticks" axes style). thesis_plot_style then has the last word on
    # every rcParam it cares about, so seaborn can never clobber the fonts,
    # sizes or spine settings.
    sns.set_theme(style="ticks")
    constants = thesis_plot_style(**kwargs)

    _CONSTANTS = constants
    return constants


def style_constants() -> dict:
    """Return the layout constants, applying the style if not yet applied."""
    if _CONSTANTS is None:
        return use_style()
    return _CONSTANTS


def despine(ax=None, fig=None, **kwargs) -> None:
    """Remove the top and right spines, matching the thesis style."""
    sns.despine(ax=ax, fig=fig, top=True, right=True, **kwargs)


def legend_above(ax, hue_order=None) -> None:
    """Put the legend above the axes and swap scenario keys for labels.

    Seaborn writes the raw hue values into the legend; the thesis figures
    show the display names ("Tiered-density", not "new_more").
    """
    from plotlib.palettes import labels_for

    handles, texts = ax.get_legend_handles_labels()
    existing = ax.get_legend()

    if not handles and existing is not None:
        # `ecdfplot` builds its legend directly and leaves the artists
        # unlabelled, so `get_legend_handles_labels` comes back empty.
        handles = existing.legend_handles
        texts = [text.get_text() for text in existing.get_texts()]

    if not handles:
        return

    mapping = labels_for(hue_order if hue_order is not None else texts)
    labels = [mapping.get(text, text) for text in texts]

    if existing is not None:
        existing.remove()
    ax.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(4, len(labels)),
        frameon=False,
    )

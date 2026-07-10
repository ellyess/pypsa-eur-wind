"""
Shared seaborn plotting layer for the wake and sensitivity manuscripts.

This package is a thin plotting layer on top of the existing style
registries, not a replacement for them:

- ``plotting_style`` still owns the matplotlib rcParams, the figure widths,
  ``savefig_thesis`` and the axis formatters.
- ``thesis_colors`` still owns the palettes, the display labels and the
  canonical scenario ordering.

``plotlib`` feeds those into seaborn once, in one place, so that every figure
in both manuscripts shares a theme, a palette and an ordering.

Typical use::

    from plotlib import use_style, savefig
    from plotlib.distributions import plot_distribution

    use_style()
    fig = plot_distribution(df, value="wake_loss", kind="box")
    savefig(fig, out_dir / "wake_loss_box.pdf")
"""

from plotlib.io import figure_size, savefig
from plotlib.palettes import hue_kwargs, labels_for, order_for, palette_for
from plotlib.style import despine, use_style

__all__ = [
    "despine",
    "figure_size",
    "hue_kwargs",
    "labels_for",
    "order_for",
    "palette_for",
    "savefig",
    "use_style",
]

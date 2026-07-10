"""
Path conventions and figure saving.

The roots match ``run_all.py`` so the scripts, the orchestrator and the two
paper pipelines all agree on where results, plots and extracted data live.
Every root can be overridden by an environment variable, which is what lets
the paper pipelines be re-pointed at a new model run without editing code.
"""

from __future__ import annotations

import os
from pathlib import Path

from plotting_style import savefig_thesis
from plotlib.style import format_axes_standard, style_constants

__all__ = [
    "DATA_ROOT",
    "MANUSCRIPT_IMAGES",
    "PLOTS_ROOT",
    "REPO",
    "RESULTS_ROOT",
    "figure_size",
    "resolve_root",
    "savefig",
]

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

RESULTS_ROOT = Path(os.environ.get("PYPSA_RESULTS_ROOT", REPO / "results"))
PLOTS_ROOT = Path(os.environ.get("PYPSA_PLOTS_ROOT", REPO / "plots"))
DATA_ROOT = Path(os.environ.get("PYPSA_DATA_ROOT", REPO / "analysis_scripts" / "data"))

#: Where the wake manuscript expects its figures.
MANUSCRIPT_IMAGES = Path(
    os.environ.get(
        "WAKE_MANUSCRIPT_IMAGES",
        REPO.parent / "Manuscript-PyPSA_Eur-Wake" / "images",
    )
)


def resolve_root(override: str | Path | None, default: Path) -> Path:
    """Return *override* as a Path if given, else *default*."""
    return Path(override) if override else default


def figure_size(width: str = "FULL_WIDTH", height_cm: float = 6.0) -> tuple:
    """Return a ``(width, height)`` figure size in inches.

    *width* names one of the thesis layout constants: ``FULL_WIDTH``,
    ``HALF_WIDTH``, ``THIRD_WIDTH`` or ``MAP_WIDTH``.
    """
    constants = style_constants()
    if width not in constants:
        raise KeyError(
            f"Unknown width {width!r}. Expected one of "
            "FULL_WIDTH, HALF_WIDTH, THIRD_WIDTH, MAP_WIDTH."
        )
    return constants[width], height_cm * constants["cm"]


def savefig(fig, path, *, format_axes: bool = True, **kwargs) -> Path:
    """Save *fig* to *path* with the thesis defaults, creating parent dirs.

    Applies the standard tick formatting first unless ``format_axes=False``,
    then delegates to :func:`plotting_style.savefig_thesis`, which forces PDF
    and falls back to a 600 dpi PNG when the PDF would be oversized.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format_axes:
        format_axes_standard(fig)
    savefig_thesis(fig, path, **kwargs)
    return path

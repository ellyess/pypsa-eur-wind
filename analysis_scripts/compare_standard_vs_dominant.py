#!/usr/bin/env python3
"""Compare standard vs dominant offshore wind selection across spatial resolutions.

Reads pre-computed spatial_resolution_metrics.csv for both standard and dominant
runs, then produces:

1. A 2x2 multi-panel comparison figure (thesis-style) saved as PNG.
2. A comparison table printed to stdout.

Run from the repo root:
    python analysis_scripts/compare_standard_vs_dominant.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from analysis_scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting_style import (
    thesis_plot_style,
    apply_spatial_resolution_axis,
    add_resolution_markers,
    format_axes_standard,
)

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
STANDARD_CSV = Path("plots/spatial_diagnostics/standard/spatial_resolution_metrics.csv")
DOMINANT_CSV = Path("plots/spatial_diagnostics/dominant/spatial_resolution_metrics.csv")
OUT_PNG = Path("plots/spatial_diagnostics/standard_vs_dominant_comparison.png")

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
COLOR_STANDARD = "#235ebc"
COLOR_DOMINANT = "#d95f02"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.sort_values("amax").reset_index(drop=True)
    df["method"] = label
    return df


def _delta_pct(std_val: float, dom_val: float) -> float:
    """Percentage change from standard to dominant."""
    if std_val == 0:
        return np.nan
    return 100.0 * (dom_val - std_val) / abs(std_val)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_comparison_figure(
    std: pd.DataFrame,
    dom: pd.DataFrame,
    outpath: Path,
) -> None:
    style = thesis_plot_style()
    cm = style["cm"]
    FULL_WIDTH = style["FULL_WIDTH"]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(FULL_WIDTH, 10 * cm),
        dpi=600,
        sharex=True,
    )

    panel_specs = [
        {
            "col": "onwind_gw",
            "ylabel": "Onshore wind capacity [GW]",
            "label": "(a)",
            "scale": 1.0,
        },
        {
            "col": "offwind_gw",
            "ylabel": "Offshore wind capacity [GW]",
            "label": "(b)",
            "scale": 1.0,
        },
        {
            "col": "cf_median",
            "ylabel": "CF median [-]",
            "label": "(c)",
            "scale": 1.0,
        },
        {
            "col": "curtail_median",
            "ylabel": "Curtailment median [-]",
            "label": "(d)",
            "scale": 1.0,
        },
        {
            "col": "objective",
            "ylabel": r"System cost [B€]",
            "label": "(e)",
            "scale": 1e-9,  # EUR -> B EUR
        },
        {
            "col": "total_wind_gw",
            "ylabel": "Total wind capacity [GW]",
            "label": "(f)",
            "scale": 1.0,
        },
    ]

    amax_vals = sorted(std["amax"].unique())

    for ax, spec in zip(axes.flat, panel_specs):
        col = spec["col"]
        s = spec["scale"]

        # Standard
        ax.plot(
            std["amax"],
            std[col] * s,
            marker="o",
            color=COLOR_STANDARD,
            label="Standard",
        )
        # Dominant
        ax.plot(
            dom["amax"],
            dom[col] * s,
            marker="s",
            color=COLOR_DOMINANT,
            label="Dominant",
        )

        ax.set_ylabel(spec["ylabel"])
        ax.grid(True, alpha=0.3)

        # Panel label top-left
        ax.text(
            0.03,
            0.95,
            spec["label"],
            transform=ax.transAxes,
            fontweight="bold",
            va="top",
            ha="left",
        )

    # Apply spatial resolution axis formatting to bottom row
    for ax in axes[1, :]:
        apply_spatial_resolution_axis(ax, annotate=True)
    # Top row: just log + invert (no xlabel / annotations)
    for ax in axes[0, :]:
        ax.set_xscale("log")
        ax.invert_xaxis()

    # Add resolution markers to all panels
    for ax in axes.flat:
        add_resolution_markers(ax, amax_vals)

    # Single legend in upper-right panel
    axes[0, 1].legend(loc="best", frameon=False)

    fig.tight_layout()
    format_axes_standard(fig)

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Figure saved to {outpath}")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison_table(std: pd.DataFrame, dom: pd.DataFrame) -> None:
    merged = pd.merge(
        std, dom, on="amax", suffixes=("_std", "_dom"),
    )

    metrics = [
        ("Onshore [GW]", "onwind_gw", 1.0, ".2f"),
        ("Offshore [GW]", "offwind_gw", 1.0, ".2f"),
        ("Total wind [GW]", "total_wind_gw", 1.0, ".2f"),
        ("CF median", "cf_median", 1.0, ".4f"),
        ("System cost [B EUR]", "objective", 1e-9, ".3f"),
        ("Curtailment median", "curtail_median", 1.0, ".4f"),
    ]

    # Header
    hdr = f"{'Amax':>10s}"
    for name, _, _, _ in metrics:
        hdr += f" | {'Std':>12s} {'Dom':>12s} {'d%':>8s}"
    print()
    print("=" * len(hdr))
    print("Standard vs Dominant comparison")
    print("=" * len(hdr))

    # Column headers
    col_hdr = f"{'Amax':>10s}"
    for name, _, _, _ in metrics:
        w = len(name)
        col_hdr += f" | {name:^{max(w, 34)}s}"
    print(col_hdr)

    sub_hdr = f"{'':>10s}"
    for _ in metrics:
        sub_hdr += f" | {'Std':>12s} {'Dom':>12s} {'d%':>8s}"
    print(sub_hdr)
    print("-" * len(sub_hdr))

    for _, row in merged.iterrows():
        line = f"{row['amax']:>10.0f}"
        for name, col, scale, fmt in metrics:
            sv = row[f"{col}_std"] * scale
            dv = row[f"{col}_dom"] * scale
            dp = _delta_pct(sv, dv)
            line += f" | {sv:>12{fmt}} {dv:>12{fmt}} {dp:>+7.2f}%"
        print(line)

    print("=" * len(sub_hdr))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    std = _load(STANDARD_CSV, "standard")
    dom = _load(DOMINANT_CSV, "dominant")

    print(f"Loaded standard:  {len(std)} rows  from {STANDARD_CSV}")
    print(f"Loaded dominant:  {len(dom)} rows  from {DOMINANT_CSV}")

    make_comparison_figure(std, dom, OUT_PNG)
    print_comparison_table(std, dom)


if __name__ == "__main__":
    main()

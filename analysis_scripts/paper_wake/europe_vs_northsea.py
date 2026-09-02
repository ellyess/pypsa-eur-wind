"""Build the North Sea vs Europe offshore-capacity figure.

The manuscript figure ``fig_europe_vs_northsea_offwind_cap.pdf`` was previously
produced by ``compare_sensitivity_runs_tier2.py``, whose current output has a
different layout (Europe on the left, a shared y-axis that pushes the North Sea
density series off scale, and a six-entry legend carrying series this paper does
not use). This module rebuilds the published layout directly from the two
extracted metric sets, so the figure is reproducible from the pipeline.

Usage:
    python analysis_scripts/paper_wake/europe_vs_northsea.py \
        --northsea analysis_scripts/data/wake_extracted_sector \
        --europe   analysis_scripts/data/wake_extracted_europe_sector \
        --out      images/fig_europe_vs_northsea_offwind_cap.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

#: Scenario key -> (legend label, colour). Matches the manuscript's wake-model
#: colours: No-wake grey, Tiered density teal.
SERIES = {
    "base": ("No-wake", "#4d4d4d"),
    "new_more": ("Tiered density", "#1b9e77"),
}


def _capacities(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(Path(data_dir) / "system_metrics.csv")
    return df[["scenario", "split", "offshore_capacity_gw"]]


def _panel(ax, df: pd.DataFrame, title: str) -> None:
    # Coarse -> fine reads left to right, so the log axis is inverted.
    splits = sorted(df["split"].unique(), reverse=True)
    for scen, (label, colour) in SERIES.items():
        sub = df[df.scenario == scen].set_index("split").reindex(splits)
        ax.plot(
            splits,
            sub["offshore_capacity_gw"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=colour,
            label=label,
        )
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xticks(splits)
    ax.set_xticklabels([f"$10^{{{len(str(int(s))) - 1}}}$" for s in splits])
    for edge in (splits[0], splits[-1]):
        ax.axvline(edge, color="grey", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(r"Spatial Resolution ($A^{max}_{region}$) [km$^2$]", fontsize=9)
    ax.tick_params(labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.annotate("Coarse", xy=(0.0, -0.22), xycoords="axes fraction", fontsize=8)
    ax.annotate("Fine", xy=(0.92, -0.22), xycoords="axes fraction", fontsize=8)


def build(northsea_dir: Path, europe_dir: Path, out: Path) -> Path:
    fig, (ax_ns, ax_eu) = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    _panel(ax_ns, _capacities(northsea_dir), "North Sea (10 buses)")
    _panel(ax_eu, _capacities(europe_dir), "Europe (31 buses)")
    ax_ns.set_ylabel("Offshore wind capacity [GW]", fontsize=9)

    handles, labels = ax_ns.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.06),
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=600)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--northsea", type=Path, required=True)
    p.add_argument("--europe", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    print(f"wrote {build(a.northsea, a.europe, a.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

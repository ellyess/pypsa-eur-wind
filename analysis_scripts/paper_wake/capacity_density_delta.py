"""Build the two-resolution offshore capacity-density delta maps.

Rows are spatial resolutions, columns are wake formulations, and colour is the
change in optimal offshore capacity density relative to the no-wake baseline.

Three choices matter for legibility, because most regions do not change at all
and the few that do are small at fine resolution:

* geometry is reprojected to ETRS89-LAEA (EPSG:3035), so shapes match the
  region-split figure rather than being stretched by an equal-aspect plot of
  raw degrees;
* colour uses a symmetric log norm, linear within ``linthresh`` and logarithmic
  beyond, so moderate changes stay visible next to the largest ones;
* regions changing by more than ``outline_threshold`` are outlined, and the
  unchanged mesh is drawn in light grey so it recedes.

Usage:
    python analysis_scripts/paper_wake/capacity_density_delta.py \
        --results-dir results/paper-northsea-sector-2030-10-dominant-6h-splitfix \
        --regions-area northsea-splitfix \
        --out images/capacity_density_delta_maps.pdf
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _sub in ("scripts", "analysis_scripts"):
    if str(_ROOT / _sub) not in sys.path:
        sys.path.insert(0, str(_ROOT / _sub))

import pypsa  # noqa: E402
from compare_wake_runs import (  # noqa: E402
    build_region_capacity_density_geodf,
    label,
)

#: Wake formulations plotted, in column order. The baseline is the reference.
SCENARIOS = ["standard", "glaum", "new_more"]
BASELINE = "base"


def _delta_frames(results_dir: Path, area: str, regions_dir: Path, splits):
    """Return {(split, scenario): geodataframe} plus the per-row colour limit."""
    frames, limits = {}, {}
    for split in splits:
        density, geo = {}, {}
        for scen in [BASELINE] + SCENARIOS:
            matches = glob.glob(
                str(results_dir / f"{scen}-s{split}-biasFalse" / "networks" / "*.nc")
            )
            if not matches:
                raise FileNotFoundError(f"No network for {scen} at s{split}")
            gdf, _, _ = build_region_capacity_density_geodf(
                pypsa.Network(matches[0]),
                split=split,
                area=area,
                regions_dir=regions_dir,
                carrier_filter="offwind",
                cap_field="p_nom_opt",
            )
            geo[scen] = gdf
            density[scen] = gdf.set_index("region")["density_mw_per_km2"]

        values = []
        for scen in SCENARIOS:
            gdf = geo[scen].copy()
            gdf["delta_density"] = gdf["region"].map(
                density[scen] - density[BASELINE]
            )
            frames[(split, scen)] = gdf.to_crs(3035)
            finite = gdf["delta_density"].to_numpy(dtype=float)
            values.extend(finite[np.isfinite(finite)])
        limits[split] = float(np.max(np.abs(values)))
    return frames, limits


def build(
    results_dir: Path,
    out: Path,
    *,
    area: str,
    regions_dir: Path = Path("wake_extra"),
    splits=(10000, 1000),
    linthresh: float = 0.05,
    outline_threshold: float = 0.1,
) -> Path:
    frames, limits = _delta_frames(Path(results_dir), area, Path(regions_dir), splits)

    fig, axes = plt.subplots(
        len(splits), len(SCENARIOS), figsize=(11, 9.5), layout="constrained"
    )
    axes = np.atleast_2d(axes)

    for row, split in enumerate(splits):
        norm = mpl.colors.SymLogNorm(
            linthresh=linthresh, vmin=-limits[split], vmax=limits[split], base=10
        )
        for col, scen in enumerate(SCENARIOS):
            ax = axes[row][col]
            gdf = frames[(split, scen)]
            gdf.plot(
                column="delta_density",
                ax=ax,
                cmap="RdBu_r",
                norm=norm,
                linewidth=0.15,
                edgecolor="0.75",
            )
            changed = gdf[gdf["delta_density"].abs() > outline_threshold]
            if len(changed):
                changed.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black")
            if row == 0:
                ax.set_title(label(scen), fontsize=13)
            ax.set_aspect("equal")
            ax.axis("off")

        ticks = [t for t in (-3, -1, -0.3, -0.1, 0, 0.1, 0.3, 1, 3) if abs(t) <= limits[split]]
        bar = fig.colorbar(
            mpl.cm.ScalarMappable(cmap="RdBu_r", norm=norm),
            ax=list(axes[row]),
            orientation="vertical",
            shrink=0.8,
            pad=0.01,
            ticks=ticks,
        )
        bar.ax.set_yticklabels([f"{t:g}" for t in ticks])
        bar.set_label(r"$\Delta$ density [MW/km$^2$]", fontsize=10)

        exponent = int(round(np.log10(split)))
        axes[row][0].annotate(
            rf"$A^{{max}}_{{region}} = 10^{{{exponent}}}$ km$^2$",
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-0.05, 0.5),
            fontsize=12,
            rotation=90,
            ha="center",
            va="center",
        )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results-dir", type=Path, required=True)
    p.add_argument("--regions-area", required=True, help="Subfolder under --regions-dir")
    p.add_argument("--regions-dir", type=Path, default=Path("wake_extra"))
    p.add_argument("--splits", nargs="+", type=int, default=[10000, 1000])
    p.add_argument("--linthresh", type=float, default=0.05)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    written = build(
        a.results_dir,
        a.out,
        area=a.regions_area,
        regions_dir=a.regions_dir,
        splits=tuple(a.splits),
        linthresh=a.linthresh,
    )
    print(f"wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

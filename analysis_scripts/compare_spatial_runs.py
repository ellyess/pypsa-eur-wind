#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Spatial resolution diagnostics from PyPSA-Eur(-wind) postnetworks.

Computes, for each run/scenario network:
- total built wind capacity (onshore/offshore breakdown)
- built-capacity-weighted CF distribution (median + IQR)
- built-capacity-weighted curtailment distribution (median + IQR)
- (optional) market value of wind
- (optional) line volume and price dispersion

Then plots metrics vs Amax parsed from run-name (e.g., "base-s1000-biasFalse" -> 1000).

Usage example:
python spatial_resolution_diagnostics.py \
  --results-dir results \
  --glob "*/postnetworks/base_s_*_l*_*.nc" \
  --out-dir plots/spatial_diagnostics \
  --carrier-onwind onwind \
  --carrier-offwind offwind-ac offwind-dc offwind-float \
  --planning-horizon 2030

If your file naming differs, adjust --glob.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting_style import thesis_plot_style, apply_spatial_resolution_axis, add_resolution_markers, format_axes_standard
from thesis_colors import SCENARIO_COLORS, get_color_cycle

# Apply thesis-wide plotting style
_style = thesis_plot_style()
cm, lw, ms, dpi = _style['cm'], _style['lw'], _style['ms'], _style['dpi']
import seaborn as sns

import geopandas as gpd
import pypsa

from network_utils import (
    snapshot_weights as get_snapshot_weights,
    select_generators_by_carrier,
)

# ---- Thesis plotting settings (thesis + consistent with wake/bias figures) ----
cm = 1 / 2.54

# Default thesis-friendly figure size (can be overridden by CLI)
DEFAULT_FIG_W_CM = 14.0
DEFAULT_FIG_H_CM = 8.0

BG_COLOUR = "#f0f0f0"

# map styling (match your wake/bias style)
LW = 0.25
FACE = "#f0f0f0"
EDGE = "black"

# Seaborn/matplotlib theme (matching the thesis figure styling)
custom_params = {
    "xtick.bottom": True,
    "axes.edgecolor": "black",
    "axes.spines.right": False,
    "axes.spines.top": False,
    "mathtext.default": "regular",
}
sns.set_theme(style="ticks", rc=custom_params)

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        "mathtext.default": "it",
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "legend.title_fontsize": 6,
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)

# Muted, print-safe palette consistent with the wake/bias chapter figures
WAKE_COLORS = {
    "onshore": "#235ebc",       # onwind
    "offshore": "#6895dd",      # offwind
    "total": "#1b1b1b",
    "network": "#6c9459",
    "price": "#7a3db8",
    "cf": "#1f9e89",
    "curtailment": "#d95f02",
    "market_value": "#7570b3",
}

# -----------------------------
# Helpers
# -----------------------------

def new_fig_ax(fig_w_cm=14, fig_h_cm=8):
    """
    Create a thesis-sized figure and axis.
    """
    cm = 1 / 2.54
    fig, ax = plt.subplots(
        figsize=(fig_w_cm * cm, fig_h_cm * cm),
        constrained_layout=True,
    )
    return fig, ax

def weighted_quantile(values: np.ndarray, quantiles: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Compute weighted quantiles of *values* at given quantiles in [0,1].
    Robust to NaNs by filtering them out.
    """
    values = np.asarray(values)
    weights = np.asarray(weights)
    quantiles = np.asarray(quantiles)

    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if mask.sum() == 0:
        return np.full_like(quantiles, np.nan, dtype=float)

    v = values[mask]
    w = weights[mask]

    sorter = np.argsort(v)
    v = v[sorter]
    w = w[sorter]

    cw = np.cumsum(w)
    cw = cw / cw[-1]

    return np.interp(quantiles, cw, v)


def safe_mean_series(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    return float(np.nanmean(x.values)) if len(x) else np.nan


def discover_networks(results_dir: Path, glob_pattern: str) -> List[Path]:
    paths = sorted(results_dir.rglob(glob_pattern))
    # allow passing something like "*/postnetworks/*.nc" relative to results_dir
    return [p for p in paths if p.suffix in [".nc", ".h5"]]


def parse_amax_from_run(run_name: str) -> Optional[float]:
    """
    Extract Amax from run name segments like '...-s1000-...' or '...-s100000-...'.
    Returns float or None.
    """
    m = re.search(r"(?:^|-)s(\d+)(?:-|$)", run_name)
    if not m:
        return None
    return float(m.group(1))


def parse_run_name_from_path(p: Path) -> str:
    """
    Assumes typical PyPSA-Eur results layout:
      results/<RDIR>/<run_name>/postnetworks/...
    So run_name is the directory directly under RDIR (or results).
    We'll take the parent of 'postnetworks'.
    """
    parts = p.parts
    # find "postnetworks" index
    try:
        idx = parts.index("networks")
        # run folder is just before postnetworks
        return parts[idx - 1]
    except ValueError:
        # fallback: use parent name
        return p.parent.name


@dataclass
class WindStats:
    amax: float
    run: str
    network_path: Path
    total_wind_gw: float
    onwind_gw: float
    offwind_gw: float

    cf_median: float
    cf_q25: float
    cf_q75: float

    curtail_median: float
    curtail_q25: float
    curtail_q75: float

    wind_market_value: float  # EUR/MWh (if prices exist)
    price_dispersion: float   # std of mean nodal prices (if exists)
    line_volume: float        # sum length*capacity (if exists)
    objective: float          # system cost / objective value (EUR)
    trans_expansion: float    # transmission expansion proxy (MW-km)


# -----------------------------
# Core calculations
# -----------------------------

def built_capacity_gw(n: pypsa.Network, gens: pd.Index) -> pd.Series:
    if len(gens) == 0:
        return pd.Series(dtype=float)
    pnom = n.generators.loc[gens, "p_nom_opt"] if "p_nom_opt" in n.generators.columns else n.generators.loc[gens, "p_nom"]
    pnom = pd.to_numeric(pnom, errors="coerce").fillna(0.0)
    return pnom / 1e3  # MW -> GW


def generator_cf(n: pypsa.Network, gens: pd.Index) -> pd.Series:
    """
    CF_g = sum_t p_g,t * w_t / (p_nom_opt_g * sum_t w_t)
    """
    if len(gens) == 0:
        return pd.Series(dtype=float)

    w = get_snapshot_weights(n)
    W = float(w.sum())
    if W <= 0:
        return pd.Series(index=gens, data=np.nan)

    p = n.generators_t.p[gens].multiply(w, axis=0).sum(axis=0)

    pnom = n.generators.loc[gens, "p_nom_opt"] if "p_nom_opt" in n.generators.columns else n.generators.loc[gens, "p_nom"]
    pnom = pd.to_numeric(pnom, errors="coerce")

    cf = p / (pnom * W)
    cf = cf.replace([np.inf, -np.inf], np.nan)
    return cf


def generator_curtailment(n: pypsa.Network, gens: pd.Index) -> pd.Series:
    """
    Curtailment_g = 1 - used / available
    used = sum_t p_g,t * w_t
    available = sum_t p_max_pu_g,t * p_nom_opt_g * w_t
    """
    if len(gens) == 0:
        return pd.Series(dtype=float)

    w = get_snapshot_weights(n)

    used = n.generators_t.p[gens].multiply(w, axis=0).sum(axis=0)

    if "p_max_pu" not in n.generators_t:
        # cannot compute curtailment
        return pd.Series(index=gens, data=np.nan)

    pmax = n.generators_t.p_max_pu[gens]
    pnom = n.generators.loc[gens, "p_nom_opt"] if "p_nom_opt" in n.generators.columns else n.generators.loc[gens, "p_nom"]
    pnom = pd.to_numeric(pnom, errors="coerce")

    available = pmax.multiply(pnom, axis=1).multiply(w, axis=0).sum(axis=0)

    curtail = 1.0 - (used / available)
    curtail = curtail.replace([np.inf, -np.inf], np.nan)
    return curtail


def wind_market_value_eur_per_mwh(n: pypsa.Network, gens: pd.Index) -> float:
    """
    Market value = total revenue / total energy (EUR/MWh),
    revenue = sum_t p_g,t * price_bus(t) * w_t
    """
    if len(gens) == 0:
        return np.nan
    if not hasattr(n, "buses_t") or "marginal_price" not in n.buses_t:
        return np.nan

    w = get_snapshot_weights(n)

    gen_bus = n.generators.loc[gens, "bus"]
    prices = n.buses_t.marginal_price[gen_bus.unique()]
    # align price columns to gens by mapping
    price_for_gen = prices.reindex(columns=gen_bus.values)
    price_for_gen.columns = gens  # now columns match generator names

    p = n.generators_t.p[gens]

    # revenue in EUR: p [MW] * price [EUR/MWh] * w [h]
    revenue = (p * price_for_gen).multiply(w, axis=0).sum().sum()
    energy = p.multiply(w, axis=0).sum().sum()  # MWh because MW * h

    if energy == 0:
        return np.nan
    return float(revenue / energy)


def price_dispersion(n: pypsa.Network) -> float:
    if not hasattr(n, "buses_t") or "marginal_price" not in n.buses_t:
        return np.nan
    # std across buses of mean price
    mu = n.buses_t.marginal_price.mean(axis=0)
    return float(mu.std())


def line_volume(n: pypsa.Network) -> float:
    """
    Rough proxy: sum(length * capacity_opt) for AC lines and DC links.
    Units: MW-km (if lengths in km, capacities in MW).
    """
    total = 0.0
    if not n.lines.empty and "length" in n.lines and ("s_nom_opt" in n.lines or "s_nom" in n.lines):
        cap = n.lines["s_nom_opt"] if "s_nom_opt" in n.lines else n.lines["s_nom"]
        total += float((n.lines["length"] * cap).sum())

    if not n.links.empty and "length" in n.links and ("p_nom_opt" in n.links or "p_nom" in n.links):
        # take DC links only if carrier exists
        if "carrier" in n.links.columns:
            dc = n.links.index[n.links.carrier.eq("DC")]
        else:
            dc = n.links.index
        cap = n.links.loc[dc, "p_nom_opt"] if "p_nom_opt" in n.links else n.links.loc[dc, "p_nom"]
        total += float((n.links.loc[dc, "length"] * cap).sum())
    return total


def compute_stats_for_network(
    path: Path,
    carriers_onwind: List[str],
    carriers_offwind: List[str],
) -> Optional[WindStats]:
    run = parse_run_name_from_path(path)
    amax = parse_amax_from_run(run)
    if amax is None:
        return None

    n = pypsa.Network(path)

    on_gens = select_generators_by_carrier(n, carriers_onwind)
    off_gens = select_generators_by_carrier(n, carriers_offwind)
    wind_gens = on_gens.union(off_gens)

    on_cap = built_capacity_gw(n, on_gens).sum()
    off_cap = built_capacity_gw(n, off_gens).sum()
    tot_cap = on_cap + off_cap

    # CFs and curtailment per generator
    cf = generator_cf(n, wind_gens)
    curtail = generator_curtailment(n, wind_gens)

    # weights: built capacity in MW (not GW) to weight distributions
    w = n.generators.loc[wind_gens, "p_nom_opt"] if "p_nom_opt" in n.generators.columns else n.generators.loc[wind_gens, "p_nom"]
    w = pd.to_numeric(w, errors="coerce").fillna(0.0).values

    cf_q = weighted_quantile(cf.values, np.array([0.25, 0.5, 0.75]), w)
    cu_q = weighted_quantile(curtail.values, np.array([0.25, 0.5, 0.75]), w)

    mv = wind_market_value_eur_per_mwh(n, wind_gens)
    pdsp = price_dispersion(n)
    lv = line_volume(n)
    obj = float(getattr(n, "objective", np.nan))

    # Transmission expansion (delta line volume from initial)
    trans_exp = 0.0
    if hasattr(n, "lines") and not n.lines.empty and "s_nom_opt" in n.lines.columns:
        ln = n.lines
        base_s = ln.get("s_nom", pd.Series(0.0, index=ln.index)).fillna(0.0)
        opt_s = ln["s_nom_opt"].fillna(base_s)
        delta_s = (opt_s - base_s).clip(lower=0.0)
        length_s = ln.get("length", pd.Series(0.0, index=ln.index)).fillna(0.0)
        trans_exp += float((delta_s * length_s).sum())
    if hasattr(n, "links") and not n.links.empty and "p_nom_opt" in n.links.columns:
        lk = n.links
        base_p = lk.get("p_nom", pd.Series(0.0, index=lk.index)).fillna(0.0)
        opt_p = lk["p_nom_opt"].fillna(base_p)
        delta_p = (opt_p - base_p).clip(lower=0.0)
        length_p = lk.get("length", pd.Series(0.0, index=lk.index)).fillna(0.0)
        trans_exp += float((delta_p * length_p).sum())

    return WindStats(
        amax=amax,
        run=run,
        network_path=path,
        total_wind_gw=float(tot_cap),
        onwind_gw=float(on_cap),
        offwind_gw=float(off_cap),
        cf_q25=float(cf_q[0]),
        cf_median=float(cf_q[1]),
        cf_q75=float(cf_q[2]),
        curtail_q25=float(cu_q[0]),
        curtail_median=float(cu_q[1]),
        curtail_q75=float(cu_q[2]),
        wind_market_value=float(mv),
        price_dispersion=float(pdsp),
        line_volume=float(lv),
        objective=float(obj),
        trans_expansion=float(trans_exp),
    )
# -----------------------------
# Plotting
# -----------------------------
def plot_region_splits(
    prefix: str,
    splits: list[int],
    *,
    regions_dir: str,
    filename_template: str = "regions_offshore_s{split}.geojson",
    out: str = "plots/spatial/region_splits.png",
):
    regions_dir = Path(regions_dir)

    fig, ax = plt.subplots(
        1, len(splits),
        figsize=(17.8 * cm, 4.8 * cm),  # two-column friendly
        dpi=600,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    if len(splits) == 1:
        ax = [ax]

    for i, split in enumerate(splits):
        fp = regions_dir / prefix / filename_template.format(split=split)

        if not fp.exists():
            # Don’t crash the whole run: show a warning panel
            ax[i].set_axis_off()
            ax[i].text(
                0.5,
                0.5,
                f"Missing:\n{fp}",
                transform=ax[i].transAxes,
                ha="center",
                va="center",
                fontsize=7,
            )
        else:
            gpd.read_file(fp).plot(
                ax=ax[i],
                linewidth=LW,
                color=FACE,
                edgecolor=EDGE,
            )

            ax[i].set_title(
                rf"Spatial Resolution ($A_{{region}}^{{max}}$):" + "\n" + f"{split:,} km$^2$",
                fontsize=7,
            )
            ax[i].set_axis_off()

        # panel label
        ax[i].text(
            0.02, 0.98, f"({chr(97+i)})",
            transform=ax[i].transAxes,
            va="top", ha="left",
            fontsize=7,
        )

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    format_axes_standard(fig)
    fig.savefig(out, bbox_inches="tight", pil_kwargs={"compress_level": 1})
    plt.close(fig)


def plot_iqr_band(ax, x, median, q25, q75, ylabel, title=None):
    ax.plot(x, median, marker="o")
    ax.fill_between(x, q25, q75, alpha=0.2)  # smaller Amax = finer resolution
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", axis="both", alpha=0.3)
    sns.despine(ax=ax)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    ap.add_argument("--glob", type=str, default="*/postnetworks/*.nc",
                    help="Glob pattern relative to results-dir, used with rglob.")
    ap.add_argument("--out-dir", type=Path, default=Path("plots/spatial_diagnostics"))
    ap.add_argument("--carrier-onwind", nargs="+", default=["onwind"])
    ap.add_argument("--carrier-offwind", nargs="+", default=["offwind-ac", "offwind-dc", "offwind-float"])
    ap.add_argument("--planning-horizon", type=str, default=None,
                    help="Optional filter: keep only networks whose filename contains this string (e.g. 2030).")
    ap.add_argument("--run-include", type=str, default=None,
                    help="Optional regex filter on run name (directory).")
    ap.add_argument("--make-per-run-hists", action="store_true",
                    help="Also save CF and curtailment histograms per run.")
    ap.add_argument(
        "--fig-w-cm",
        type=float,
        default=14.0,
        help="Figure width in cm (thesis-ready default: 14 cm)",
    )
    ap.add_argument(
        "--fig-h-cm",
        type=float,
        default=8.0,
        help="Figure height in cm (thesis-ready default: 8 cm)",
    )
    ap.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Figure DPI for saved outputs (default: 600 for thesis)",
    )
    ap.add_argument(
        "--plot-region-splits",
        action="store_true",
        help="Also plot region split maps for selected Amax values.",
    )

    ap.add_argument(
        "--regions-dir",
        default="wake_extra",
        help="Base directory containing <prefix>/regions_offshore_s{split}.geojson",
    )
    ap.add_argument(
        "--regions-prefix",
        default="northsea",
        help="Subfolder name under --regions-dir containing the region geojsons (e.g. northsea).",
    )
    ap.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[100000, 50000, 10000, 5000, 1000],
        help="List of Amax (km^2) splits to plot in the region map figure.",
    )
    ap.add_argument(
        "--regions-template",
        default="regions_offshore_s{split}.geojson",
        help="Filename template inside <regions-dir>/<prefix>/, with {split} placeholder.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = discover_networks(args.results_dir, args.glob)
    if args.planning_horizon:
        paths = [p for p in paths if args.planning_horizon in p.name]

    run_re = re.compile(args.run_include) if args.run_include else None
    if run_re:
        paths = [p for p in paths if run_re.search(parse_run_name_from_path(p))]

    if not paths:
        raise SystemExit("No networks found. Adjust --results-dir / --glob / filters.")

    stats: List[WindStats] = []
    for p in paths:
        try:
            s = compute_stats_for_network(p, args.carrier_onwind, args.carrier_offwind)
            if s is not None:
                stats.append(s)
        except Exception as e:
            print(f"[WARN] Failed on {p}: {e}")

    if not stats:
        raise SystemExit("No usable networks after parsing Amax from run names.")

    df = pd.DataFrame([s.__dict__ for s in stats])

    # If multiple networks per Amax (e.g. different wake models), keep all, but
    # default to aggregating by Amax via mean (you can change this easily).
    df = df.sort_values("amax")

    df.to_csv(args.out_dir / "spatial_resolution_metrics.csv", index=False)

    # --- Plot 1: total wind built

    fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
    ax.plot(df["amax"], df["total_wind_gw"], marker="o", label="Total wind")
    ax.plot(df["amax"], df["onwind_gw"], marker="o", label="Onshore wind")
    ax.plot(df["amax"], df["offwind_gw"], marker="o", label="Offshore wind")
    apply_spatial_resolution_axis(ax, annotate=False)
    ax.set_ylabel("Built wind capacity [GW]")
    ax.grid(True, which="both", alpha=0.3)
    add_resolution_markers(ax, df['amax'].values)
    sns.despine(ax=ax)
    ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(args.out_dir / "wind_capacity_vs_amax.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 2: CF median + IQR
    fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
    plot_iqr_band(
        ax,
        df["amax"].values,
        df["cf_median"].values,
        df["cf_q25"].values,
        df["cf_q75"].values,
        ylabel="Built-capacity-weighted CF [-]",
        title="Wind CF distribution (weighted by built capacity)",
    )
    apply_spatial_resolution_axis(ax, annotate=False)
    add_resolution_markers(ax, df['amax'].values)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(args.out_dir / "wind_cf_iqr_vs_amax.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Plot 3: Curtailment median + IQR
    fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
    plot_iqr_band(
        ax,
        df["amax"].values,
        df["curtail_median"].values,
        df["curtail_q25"].values,
        df["curtail_q75"].values,
        ylabel="Built-capacity-weighted curtailment [-]",
        title="Wind curtailment distribution (weighted by built capacity)",
    )
    apply_spatial_resolution_axis(ax, annotate=False)
    add_resolution_markers(ax, df['amax'].values)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(args.out_dir / "wind_curtailment_iqr_vs_amax.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Optional: market value / line volume / price dispersion
    if df["wind_market_value"].notna().any():
        fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
        ax.plot(df["amax"], df["wind_market_value"], marker="o")
        apply_spatial_resolution_axis(ax, annotate=False)
        ax.set_ylabel("Wind market value [EUR/MWh]")
        ax.grid(True, which="both", alpha=0.3)
        add_resolution_markers(ax, df['amax'].values)
        sns.despine(ax=ax)
        fig.tight_layout()
        format_axes_standard(fig)
        fig.savefig(args.out_dir / "wind_market_value_vs_amax.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if df["line_volume"].notna().any():
        fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
        ax.plot(df["amax"], df["line_volume"], marker="o")
        apply_spatial_resolution_axis(ax, annotate=False)
        ax.set_ylabel("Line volume proxy [MW-km]")
        ax.grid(True, which="both", alpha=0.3)
        add_resolution_markers(ax, df['amax'].values)
        sns.despine(ax=ax)
        fig.tight_layout()
        format_axes_standard(fig)
        fig.savefig(args.out_dir / "line_volume_vs_amax.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    if df["price_dispersion"].notna().any():
        fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
        ax.plot(df["amax"], df["price_dispersion"], marker="o")
        apply_spatial_resolution_axis(ax, annotate=False)
        ax.set_ylabel("Price dispersion (std of mean nodal prices) [EUR/MWh]")
        ax.grid(True, which="both", alpha=0.3)
        add_resolution_markers(ax, df['amax'].values)
        sns.despine(ax=ax)
        fig.tight_layout()
        format_axes_standard(fig)
        fig.savefig(args.out_dir / "price_dispersion_vs_amax.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Plot: System cost vs Amax ---
    if df["objective"].notna().any():
        fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
        obj_bn = df["objective"] / 1e9  # Convert EUR to billion EUR
        ax.plot(df["amax"], obj_bn, marker="o", color=WAKE_COLORS["total"])
        apply_spatial_resolution_axis(ax, annotate=False)
        ax.set_ylabel(r"System cost [B€]")
        ax.grid(True, which="both", alpha=0.3)
        add_resolution_markers(ax, df['amax'].values)
        sns.despine(ax=ax)
        fig.tight_layout()
        format_axes_standard(fig)
        fig.savefig(args.out_dir / "system_cost_vs_amax.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Plot: Transmission expansion vs Amax ---
    if df["trans_expansion"].notna().any() and (df["trans_expansion"] > 0).any():
        fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
        trans_twkm = df["trans_expansion"] / 1e6  # MW-km to TW-km
        ax.plot(df["amax"], trans_twkm, marker="o", color=WAKE_COLORS["network"])
        apply_spatial_resolution_axis(ax, annotate=False)
        ax.set_ylabel(r"Transmission expansion [TW$\cdot$km]")
        ax.grid(True, which="both", alpha=0.3)
        add_resolution_markers(ax, df['amax'].values)
        sns.despine(ax=ax)
        fig.tight_layout()
        format_axes_standard(fig)
        fig.savefig(args.out_dir / "transmission_expansion_vs_amax.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Plot: CF IQR width (dispersion metric) vs Amax ---
    if df["cf_q75"].notna().any() and df["cf_q25"].notna().any():
        fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
        iqr_width = df["cf_q75"] - df["cf_q25"]
        ax.plot(df["amax"], iqr_width, marker="o", color=WAKE_COLORS["cf"])
        apply_spatial_resolution_axis(ax, annotate=False)
        ax.set_ylabel("CF IQR width (Q75 - Q25) [-]")
        ax.grid(True, which="both", alpha=0.3)
        add_resolution_markers(ax, df['amax'].values)
        sns.despine(ax=ax)
        fig.tight_layout()
        format_axes_standard(fig)
        fig.savefig(args.out_dir / "cf_iqr_width_vs_amax.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
    
    if args.plot_region_splits:
        plot_region_splits(
            args.regions_prefix,
            args.splits,
            regions_dir=args.regions_dir,
            filename_template=args.regions_template,
            out=str(Path(args.out_dir) / "region_splits.png"),
        )
        
    # --- Optional: per-run histograms
    if args.make_per_run_hists:
        for _, row in df.iterrows():
            try:
                n = pypsa.Network(row["network_path"])
                wind_gens = select_generators_by_carrier(n, args.carrier_onwind).union(
                    select_generators_by_carrier(n, args.carrier_offwind)
                )
                cf = generator_cf(n, wind_gens)
                curtail = generator_curtailment(n, wind_gens)
                w = n.generators.loc[wind_gens, "p_nom_opt"] if "p_nom_opt" in n.generators.columns else n.generators.loc[wind_gens, "p_nom"]
                w = pd.to_numeric(w, errors="coerce").fillna(0.0)

                # CF histogram (weighted)
                fig, ax = new_fig_ax(args.fig_w_cm, args.fig_h_cm)
                ax.hist(cf.dropna().values, bins=30, weights=w.loc[cf.dropna().index].values, density=True)
                ax.set_xlabel("CF [-]")
                ax.set_ylabel("Density (capacity-weighted)")
                ax.set_title(f"CF distribution (built-weighted) — {row['run']}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                format_axes_standard(fig)
                fig.savefig(args.out_dir / f"hist_cf_{row['run']}.png", dpi=200)
                plt.close(fig)

                # Curtailment histogram (weighted)
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.hist(curtail.dropna().values, bins=30, weights=w.loc[curtail.dropna().index].values, density=True)
                ax.set_xlabel("Curtailment [-]")
                ax.set_ylabel("Density (capacity-weighted)")
                ax.set_title(f"Curtailment distribution (built-weighted) — {row['run']}")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                format_axes_standard(fig)
                fig.savefig(args.out_dir / f"hist_curtail_{row['run']}.png", dpi=200)
                plt.close(fig)

            except Exception as e:
                print(f"[WARN] Histogram failed for {row['run']}: {e}")

    print(f"Done. Wrote:\n  {args.out_dir}\n  spatial_resolution_metrics.csv + plots")


if __name__ == "__main__":
    main()
    
    
# Example usage:
# python compare_spatial_runs.py \
#   --results-dir results \
#   --glob "thesis-spatial-2030-*/**/networks/*.nc" \
#   --out-dir plots/spatial \
#   --plot-region-splits \
#   --make-per-run-hists \
#   --fig-w-cm 14 --fig-h-cm 8 \
#   --regions-dir wake_extra \
#   --regions-prefix northsea \
#   --splits 100000 50000 10000 5000 1000

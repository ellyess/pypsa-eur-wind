#!/usr/bin/env python3
"""
Tier-2 plotting for thesis sensitivity (Europe-wide, sector-coupled)

- Uses your plot_style.py + thesis_colors.py (consistent thesis styling)
- Scans solved PyPSA networks from results/
- Parses scenario from folder names like: <wakeprefix>-s<RES>-biasTrue/False/Uniform
- Tier-2 intent: reduced, confirmatory set (default: base, biasUniform, bias, wake, bias+wake)
- Produces:
  1) onwind vs offwind capacity (GW) for selected resolutions (lines)
  2) transmission expansion (TW·km) for selected resolutions (lines)
  3) system objective summary (lines)
  4) offshore curtailment fraction (lines)
  5) sector-coupling lens: electrolyser capacity (GW) + H2 production (TWh) if present
  6) CF distributions (ECDF) for onwind/offwind, coarse vs fine, baseline vs bias+wake

Run (example):
python plot_tier2.py \
  --results-root results \
  --glob "thesis-sensitivity-2030-50-europe-sector-6h/**/networks/*.nc" \
  --outdir plots/sensitivity/tier2 \
  --resolutions 100000 10000 \
  --compare base bias+wake
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pypsa

from plotting_style import thesis_plot_style, apply_spatial_resolution_axis, add_resolution_markers, format_axes_standard
from thesis_colors import THESIS_COLORS, THESIS_LABELS, label as get_label

from network_utils import (
    WAKE_ALIASES,
    parse_from_path, build_manifest,
    gen_idx, scenario_key, snapshot_weights,
    wind_capacity_gw as capacity_gw_from_generators,
    wind_curtailment_frac as curtailment_frac,
    energy_twh_from_generators,
    transmission_expansion_twkm, get_objective,
    electrolyser_links, electrolyser_capacity_gw, h2_production_twh,
    cf_timeseries_system as cf_timeseries,
    load_network,
)


def get_color(key: str) -> str | None:
    return THESIS_COLORS.get(key)


def compute_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        n = load_network(r["path"])
        scen = scenario_key(r["bias"], str(r["wake"]))

        rows.append(
            {
                **r.to_dict(),
                "scenario": scen,
                # wind
                "onwind_cap_gw": capacity_gw_from_generators(n, "onwind"),
                "offwind_cap_gw": capacity_gw_from_generators(n, "offwind"),
                "onwind_gen_twh": energy_twh_from_generators(n, "onwind"),
                "offwind_gen_twh": energy_twh_from_generators(n, "offwind"),
                "onwind_curt_frac": curtailment_frac(n, "onwind"),
                "offwind_curt_frac": curtailment_frac(n, "offwind"),
                # network + cost
                "trans_exp_twkm": transmission_expansion_twkm(n),
                "objective": get_objective(n),
                # sector coupling lens
                "electrolyser_cap_gw": electrolyser_capacity_gw(n),
                "h2_prod_twh": h2_production_twh(n),
            }
        )

    df = pd.DataFrame(rows)
    return df


# ----------------------------
# Plotting
# ----------------------------
def _scenario_order(compare: list[str] | None) -> list[str]:
    base_order = ["base", "biasUniform", "bias", "wake", "bias+wake"]
    if compare:
        # keep user-specified order, but ensure valid
        return [s for s in compare if s in base_order]
    return base_order


def plot_lines_by_resolution(df: pd.DataFrame, ycol: str, ylabel: str, outpath: Path, compare: list[str], title: str):
    fig, ax = plt.subplots(figsize=(5.2, 3.0), dpi=300)

    res = sorted(df["resolution"].unique())
    # Tier-2 often uses only 1-2 resolutions; still fine on log axis
    for scen in compare:
        dd = df[df["scenario"] == scen].set_index("resolution").reindex(res)
        c = get_color(scen)
        ax.plot(res, dd[ycol].values, marker="o", linewidth=1.6, label=get_label(scen), color=c)

    apply_spatial_resolution_axis(ax, annotate=False)
    add_resolution_markers(ax, res)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=False, fontsize=8)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_on_off(df: pd.DataFrame, outpath: Path, compare: list[str]):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0), dpi=300, sharex=True)
    res = sorted(df["resolution"].unique())

    for ax, tech, col in [
        (axes[0], "onwind", "onwind_cap_gw"),
        (axes[1], "offwind", "offwind_cap_gw"),
    ]:
        for scen in compare:
            dd = df[df["scenario"] == scen].set_index("resolution").reindex(res)
            c = get_color(scen)
            ax.plot(res, dd[col].values, marker="o", linewidth=1.6, label=get_label(scen), color=c)
        apply_spatial_resolution_axis(ax, annotate=False)
        add_resolution_markers(ax, res)
        ax.set_title(get_label(tech), fontsize=9)
        ax.set_ylabel("Capacity (GW)")
        ax.grid(True, alpha=0.3)

    axes[1].legend(loc='best', frameon=False, fontsize=8)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_sector_coupling(df: pd.DataFrame, outpath: Path, compare: list[str]):
    # Two panels: electrolyser cap (GW) and H2 production (TWh)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.0), dpi=300, sharex=True)
    res = sorted(df["resolution"].unique())

    for ax, col, ylabel, title in [
        (axes[0], "electrolyser_cap_gw", "Electrolyser capacity (GW)", "Electrolyser build"),
        (axes[1], "h2_prod_twh", "H₂ production (TWh)", "Hydrogen output"),
    ]:
        for scen in compare:
            dd = df[df["scenario"] == scen].set_index("resolution").reindex(res)
            c = get_color(scen)
            ax.plot(res, dd[col].values, marker="o", linewidth=1.6, label=get_label(scen), color=c)
        apply_spatial_resolution_axis(ax, annotate=False)
        add_resolution_markers(ax, res)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[1].legend(loc='best', frameon=False, fontsize=8)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# --- ECDF for CF distributions (robust + thesis-friendly) ---

def build_cf_long_both(manifest: pd.DataFrame, resolutions: tuple[int, ...], compare: list[str]) -> pd.DataFrame:
    rows = []
    for r in manifest.itertuples(index=False):
        res = int(r.resolution)
        if res not in set(resolutions):
            continue

        scen = scenario_key(r.bias, str(r.wake))
        if scen not in compare:
            continue

        n = load_network(r.path)
        for tech in ["onwind", "offwind"]:
            ts = cf_timeseries(n, tech)
            if ts is None or ts.empty:
                continue
            ts = ts.copy()
            ts["scenario"] = scen
            ts["resolution"] = res
            ts["tech"] = tech
            rows.append(ts.reset_index(drop=True))

    if not rows:
        return pd.DataFrame()

    cf = pd.concat(rows, ignore_index=True)
    cf["scenario"] = pd.Categorical(cf["scenario"], compare, ordered=True)
    cf["tech"] = pd.Categorical(cf["tech"], ["onwind", "offwind"], ordered=True)
    return cf


def _ecdf(vals: np.ndarray):
    x = np.sort(vals)
    y = np.linspace(0, 1, len(x), endpoint=True)
    return x, y


def plot_cf_ecdf_2x2(cf_long: pd.DataFrame, outpath: Path, metric: str, ylabel: str, resolutions: tuple[int, ...], compare: list[str]):
    if cf_long.empty:
        warnings.warn(f"No CF data for {metric}; skipping {outpath.name}")
        return

    fig, axes = plt.subplots(2, len(resolutions), figsize=(8.8, 5.2), dpi=300, sharex=True, sharey=True)
    if len(resolutions) == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # ensure 2x1

    for i, tech in enumerate(["onwind", "offwind"]):
        for j, res in enumerate(resolutions):
            ax = axes[i, j]
            d = cf_long[(cf_long["tech"] == tech) & (cf_long["resolution"] == int(res))]
            if d.empty:
                ax.set_axis_off()
                continue

            for scen in compare:
                vals = d.loc[d["scenario"] == scen, metric].dropna().values
                if vals.size == 0:
                    continue
                x, y = _ecdf(vals)
                c = get_color(scen)
                ax.plot(x, y, linewidth=1.6, label=get_label(scen), color=c)

            if i == 0:
                ax.set_title(f"resolution {res}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{get_label(tech)}\n{ylabel}")
            ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel(ylabel)
    axes[1, -1].legend(loc='best', frameon=False, fontsize=8)
    fig.suptitle(f"{ylabel} ECDFs (Tier 2)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_europe_vs_northsea_panel(
    tier2_metrics: pd.DataFrame,
    tier1_csv: Path | None,
    metric: str,
    ylabel: str,
    outpath: Path,
    compare: list[str],
    title: str = "",
) -> None:
    """Side-by-side panel: same metric from Tier 1 (North Sea) and Tier 2 (Europe).

    Shows generalizability of findings across domains.
    """
    has_tier1 = tier1_csv is not None and tier1_csv.exists()
    ncols = 2 if has_tier1 else 1
    fig, axes = plt.subplots(1, ncols, figsize=(4.4 * ncols, 3.2), dpi=300, sharey=True)
    if ncols == 1:
        axes = [axes]

    def _plot_on_ax(ax, df, panel_title):
        res = sorted(df["resolution"].unique())
        for scen in compare:
            dd = df[df["scenario"] == scen].set_index("resolution").reindex(res)
            c = get_color(scen)
            ax.plot(res, dd[metric].values, marker="o", linewidth=1.4,
                   label=get_label(scen), color=c)
        apply_spatial_resolution_axis(ax, annotate=False)
        add_resolution_markers(ax, res)
        ax.set_title(panel_title, fontsize=8)
        ax.grid(True, alpha=0.3)

    # Panel 1: Tier 2 (Europe)
    _plot_on_ax(axes[0], tier2_metrics, "Europe (Tier 2)")
    axes[0].set_ylabel(ylabel)

    # Panel 2: Tier 1 (North Sea) if available
    if has_tier1:
        tier1 = pd.read_csv(tier1_csv)
        # Filter to matching scenarios
        tier1 = tier1[tier1["scenario"].isin(compare)]
        _plot_on_ax(axes[1], tier1, "North Sea (Tier 1)")

    axes[-1].legend(loc="best", frameon=False, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95] if title else [0, 0, 1, 1])
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    thesis_plot_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--glob", required=True, help="Glob relative to results-root to find networks")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--resolutions", nargs="+", type=int, default=[1000000, 10000], help="Subset for Tier-2 plots")
    ap.add_argument(
        "--compare",
        nargs="+",
        default=["base", "biasUniform", "bias", "wake", "bias+wake"],
        help="Scenarios to compare (ordered)",
    )
    ap.add_argument(
        "--tier1-metrics",
        default=None,
        help="Path to Tier 1 metrics CSV for Europe vs North Sea comparison panels",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(Path(args.results_root), args.glob)

    # Keep only the resolutions we want for Tier 2 (confirmatory)
    res_sel = tuple(int(x) for x in args.resolutions)
    manifest = manifest[manifest["resolution"].isin(res_sel)].copy()

    # Normalise scenario labels
    manifest["scenario"] = manifest.apply(lambda r: scenario_key(r["bias"], str(r["wake"])), axis=1)

    compare = _scenario_order(args.compare)
    manifest = manifest[manifest["scenario"].isin(compare)].copy()

    if args.write_manifest:
        manifest.to_csv(outdir / "manifest_tier2.csv", index=False)

    metrics = compute_metrics(manifest)
    metrics.to_csv(outdir / "tier2_metrics.csv", index=False)

    # Core plots (mirror Tier 1, but reduced)
    plot_capacity_on_off(metrics, outdir / "fig_capacity_on_off_vs_resolution_tier2.png", compare=compare)
    plot_lines_by_resolution(
        metrics, "trans_exp_twkm", "Transmission expansion (TW·km)",
        outdir / "fig_transmission_vs_resolution_tier2.png", compare=compare,
        title="Transmission reinforcement (Tier 2)"
    )
    plot_lines_by_resolution(
        metrics, "objective", "Objective (EUR)",
        outdir / "fig_objective_vs_resolution_tier2.png", compare=compare,
        title="System objective (Tier 2)"
    )
    offwind_label = get_label("offwind")
    plot_lines_by_resolution(
        metrics, "offwind_curt_frac", f"{offwind_label} curtailment (fraction)",
        outdir / "fig_offwind_curtailment_vs_resolution_tier2.png", compare=compare,
        title=f"{offwind_label} curtailment (Tier 2)"
    )

    # Sector-coupling lens (only meaningful if these assets exist)
    if (metrics["electrolyser_cap_gw"].max() > 0) or (metrics["h2_prod_twh"].max() > 0):
        plot_sector_coupling(metrics, outdir / "fig_sector_coupling_h2_vs_resolution_tier2.png", compare=compare)
    else:
        warnings.warn("No electrolyser/H2 signal detected; skipping sector-coupling figure.")

    # CF ECDFs for coarse/fine (use your selected resolutions)
    cf_long = build_cf_long_both(manifest, resolutions=res_sel, compare=compare)
    cf_long.to_csv(outdir / "tier2_cf_long.csv", index=False)

    plot_cf_ecdf_2x2(
        cf_long, outdir / "fig_cf_ecdf_disp_tier2.png",
        metric="disp_cf", ylabel="Dispatch CF",
        resolutions=res_sel, compare=compare
    )
    plot_cf_ecdf_2x2(
        cf_long, outdir / "fig_cf_ecdf_curt_tier2.png",
        metric="curt_cf", ylabel="Curtailment CF",
        resolutions=res_sel, compare=compare
    )

    # Europe vs North Sea comparison panels
    tier1_csv = Path(args.tier1_metrics) if args.tier1_metrics else None
    for metric, ylabel, fname, title in [
        ("offwind_cap_gw", "Offshore capacity (GW)", "fig_europe_vs_northsea_offwind_cap.png", "Offshore wind capacity: domain comparison"),
        ("objective", "System objective (EUR)", "fig_europe_vs_northsea_objective.png", "System cost: domain comparison"),
        ("trans_exp_twkm", "Transmission expansion (TW*km)", "fig_europe_vs_northsea_trans.png", "Transmission expansion: domain comparison"),
        ("offwind_curt_frac", "Offshore curtailment (fraction)", "fig_europe_vs_northsea_curt.png", "Offshore curtailment: domain comparison"),
    ]:
        if metric in metrics.columns:
            plot_europe_vs_northsea_panel(
                metrics, tier1_csv, metric=metric, ylabel=ylabel,
                outpath=outdir / fname, compare=compare, title=title,
            )

    print(f"[OK] Wrote Tier-2 outputs to: {outdir}")
    print(f"[OK] Metrics: {outdir / 'tier2_metrics.csv'}")
    if args.write_manifest:
        print(f"[OK] Manifest: {outdir / 'manifest_tier2.csv'}")


if __name__ == "__main__":
    main()
    
    
# EXAMPLE USAGE:
# python compare_sensitivity_runs_tier2.py \
#   --results-root ../results \
#   --glob "thesis-sensitivity-2030-30-europe-dominant-6h/**/postnetworks/*.nc" \
#   --outdir ../plots/sensitivity/tier2
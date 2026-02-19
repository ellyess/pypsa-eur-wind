#!/usr/bin/env python3
"""Compare sensitivity runs (bias × wake × spatial resolution) for sector-coupled/"lvopt" postnetworks and plot metrics.

Tailored to directory names like:

  results/thesis-sensitivity-2030-10-northsea-dominant-6h/
      new_more-s10000-biasFalse/postnetworks/base_s_10_lvopt___2030.nc

Outputs (to --outdir):
  - manifest.csv (optional)
  - tier1_metrics_all.csv
  - CF distribution figures (2×2: tech × resolution)
  - Capacity/curtailment/transmission line plots vs resolution

Notes
-----
* Wake affects offshore wind only; bias affects onshore + offshore. Therefore the script
  reports BOTH onwind and offwind metrics.
* Offshore is detected by carrier containing "offwind" (captures offwind-ac/dc/float).
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting_style import thesis_plot_style, apply_spatial_resolution_axis, add_resolution_markers, format_axes_standard
from thesis_colors import THESIS_COLORS, get_color_cycle, THESIS_LABELS

from network_utils import (
    SCEN_ORDER, TECH_ORDER,
    parse_from_path, load_network, build_manifest,
    gen_idx, scenario_key,
    wind_capacity_gw, wind_curtailment_frac,
    transmission_expansion_twkm, get_objective,
    cf_timeseries_system as cf_timeseries,
)

# Apply thesis-wide plotting style
_style = thesis_plot_style()
cm, lw, ms, dpi = _style['cm'], _style['lw'], _style['ms'], _style['dpi']

INVERT_RESOLUTION_AXIS = False  # apply_spatial_resolution_axis handles inversion for coarse->fine


def build_cf_long_both(
    manifest_df: pd.DataFrame,
    keep_resolutions=(100000, 1000),
    compare: list[str] | None = None,
) -> pd.DataFrame:
    """Long table with CF time series for onwind/offwind, tagged by scenario + resolution."""
    rows: list[pd.DataFrame] = []

    keep = set(keep_resolutions) if keep_resolutions is not None else None

    for r in manifest_df.itertuples(index=False):
        res = int(r.resolution)
        if keep is not None and res not in keep:
            continue

        n = load_network(r.path)
        scen = scenario_key(r.bias, str(r.wake))
        if compare and scen not in compare:
            continue

        for tech in TECH_ORDER:
            ts = cf_timeseries(n, tech)
            if ts is None or ts.empty:
                continue

            ts = ts.copy().reset_index(drop=True)
            ts["scenario"] = scen
            ts["resolution"] = res
            ts["tech"] = tech
            rows.append(ts)

    if not rows:
        return pd.DataFrame(columns=["avail_cf", "disp_cf", "curt_cf", "scenario", "resolution", "tech"])

    out = pd.concat(rows, ignore_index=True)
    order = compare or SCEN_ORDER
    out["scenario"] = pd.Categorical(out["scenario"], order, ordered=True)
    out["tech"] = pd.Categorical(out["tech"], TECH_ORDER, ordered=True)
    out["resolution"] = out["resolution"].astype(int)
    return out


# ----------------------------
# Plotting helpers
# ----------------------------


def plot_lines_vs_resolution(
    df: pd.DataFrame,
    ycol: str,
    ylabel: str,
    outpath: Path,
    title: str,
    compare: list[str],
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    res = sorted(df["resolution"].unique())
    if INVERT_RESOLUTION_AXIS:
        res = list(reversed(res))

    for scen in compare:
        dd = df[df["scenario"] == scen].set_index("resolution").reindex(res)
        ax.plot(res, dd[ycol].values, marker="o", linewidth=1.2, color=THESIS_COLORS.get(scen, None), label=THESIS_LABELS.get(scen, scen))
    apply_spatial_resolution_axis(ax, annotate=False)
    add_resolution_markers(ax, res)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=False, fontsize=7)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_two_panel(df: pd.DataFrame, outpath: Path, compare: list[str]) -> None:
    """Two-panel: onwind vs offwind capacity vs resolution."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), dpi=300, sharex=True)

    res = sorted(df["resolution"].unique())
    if INVERT_RESOLUTION_AXIS:
        res = list(reversed(res))

    for ax, tech in zip(axes, TECH_ORDER):
        ax.set_prop_cycle(color=get_color_cycle(compare))
        for scen in compare:
            dd = df[df["scenario"] == scen].set_index("resolution").reindex(res)
            ax.plot(res, dd[f"{tech}_cap_gw"].values, marker="o", linewidth=1.2, color=THESIS_COLORS.get(scen, None), label=THESIS_LABELS.get(scen, scen))
        ax.set_title(THESIS_LABELS.get(tech, tech), fontsize=9)
        ax.grid(True, alpha=0.3)
        apply_spatial_resolution_axis(ax, annotate=False)
        add_resolution_markers(ax, res)
        ax.set_ylabel("Capacity (GW)")

    axes[1].legend(loc='best', frameon=False, fontsize=7)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_cf_2x2(
    cf_long: pd.DataFrame,
    outpath: Path,
    metric: str = "disp_cf",
    ylabel: str = "Dispatch CF",
    resolutions: tuple[int, int] = (100000, 1000),
    compare: list[str] | None = None,
) -> None:
    """2×2 violin plot: rows=tech (onwind/offwind), cols=resolution (coarse/fine)."""
    if "tech" not in cf_long.columns:
        raise ValueError("cf_long has no 'tech' column. Build with build_cf_long_both(...).")

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 5.0), dpi=300, sharey=True)
    
    # Map resolutions to labels
    res_labels = {resolutions[0]: "Coarse", resolutions[1]: "Fine"}

    for i, tech in enumerate(TECH_ORDER):
        for j, res in enumerate(resolutions):
            ax = axes[i, j]
            d = cf_long[(cf_long["tech"] == tech) & (cf_long["resolution"] == int(res))]

            violins, labels, colors = [], [], []
            missing = []
            order = compare or SCEN_ORDER
            for scen in order:
                vals = d.loc[d["scenario"] == scen, metric].dropna().values
                if vals.size == 0:
                    missing.append(scen)
                    continue
                violins.append(vals)
                labels.append(THESIS_LABELS.get(scen, scen))
                colors.append(THESIS_COLORS.get(scen, None))

            if len(violins) == 0:
                ax.axis("off")
                continue

            if missing:
                print(f"[CF] WARNING: tech={tech} res={res} missing scenarios for {metric}: {missing}")

            parts = ax.violinplot(violins, showmeans=True, showextrema=False)
            # Apply scenario colors to violin bodies and lines
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Color the mean lines to match violin colors
            if 'cmeans' in parts:
                parts['cmeans'].set_edgecolor(colors)
                parts['cmeans'].set_linewidth(1.5)
            
            # Color the bar lines if present
            if 'cbars' in parts:
                parts['cbars'].set_edgecolor(colors)
                parts['cbars'].set_linewidth(1.0)
            
            # Color the min/max lines if present
            if 'cmaxes' in parts:
                parts['cmaxes'].set_edgecolor(colors)
            if 'cmins' in parts:
                parts['cmins'].set_edgecolor(colors)
            
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels(labels, fontsize=7, rotation=15, ha='right')
            if i == 0:
                ax.set_title(res_labels.get(res, f"resolution {res}"), fontsize=9, pad=4)
            if j == 0:
                ax.set_ylabel(f"{THESIS_LABELS.get(tech, tech)}\n{ylabel}", fontsize=9)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Wind {ylabel} distributions (Tier 1)", fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_marginal_value_proxy(
    df: pd.DataFrame,
    outpath: Path,
    cap_col: str = "offwind_cap_gw",
    compare: list[str] | None = None,
) -> None:
    """Proxy: (Δ objective)/(Δ installed capacity), relative to baseline, vs resolution."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    res = sorted(df["resolution"].unique())
    if INVERT_RESOLUTION_AXIS:
        res = list(reversed(res))

    base = df[df["scenario"] == "base"].set_index("resolution")

    scenarios = [s for s in (compare or SCEN_ORDER) if s != "base" and s in df["scenario"].unique()]
    for scen in scenarios:
        dd = df[df["scenario"] == scen].set_index("resolution")
        d_obj = (dd["objective"] - base["objective"]).reindex(res)
        d_cap = (dd[cap_col] - base[cap_col]).reindex(res)
        mv = (d_obj / d_cap).replace([np.inf, -np.inf], np.nan)
        ax.plot(res, mv.values, marker="o", linewidth=1.2, color=THESIS_COLORS.get(scen, None), label=THESIS_LABELS.get(scen, scen))

    apply_spatial_resolution_axis(ax, annotate=False)
    add_resolution_markers(ax, res)
    ax.set_ylabel(f"Δ objective / Δ {cap_col} (proxy units per GW)")
    ax.set_title("Marginal value proxy (relative to baseline)", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=False, fontsize=7)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_tornado_factor_importance(
    df: pd.DataFrame,
    outpath: Path,
    metric: str = "objective",
    metric_label: str = "System cost",
    compare: list[str] | None = None,
) -> None:
    """Tornado chart showing relative importance of each modelling choice on a metric.

    For each factor (bias, wake, resolution) the bar shows the range of the
    metric when that factor varies while all others are held at baseline.
    """
    fig, ax = plt.subplots(figsize=(5.8, 3.4), dpi=300)

    # Baseline value: base scenario at the coarsest resolution
    coarse_res = df["resolution"].max()
    baseline_val = df.loc[
        (df["scenario"] == "base") & (df["resolution"] == coarse_res), metric
    ]
    if baseline_val.empty:
        # Fallback: use the base scenario averaged across resolutions
        baseline_val = df.loc[df["scenario"] == "base", metric].mean()
    else:
        baseline_val = baseline_val.values[0]

    if baseline_val == 0 or np.isnan(baseline_val):
        print(f"[WARN] Tornado: baseline {metric} is 0 or NaN; skipping.")
        plt.close(fig)
        return

    factors = {}

    # 1) Resolution effect: base scenario across all resolutions
    res_vals = df.loc[df["scenario"] == "base", metric].dropna()
    if len(res_vals) > 1:
        delta_pct = 100 * (res_vals.max() - res_vals.min()) / abs(baseline_val)
        factors["Spatial resolution"] = delta_pct

    # 2) Wake effect: wake vs base, at each resolution, take max range
    for scen_pair in [("base", "wake")]:
        s_base, s_treat = scen_pair
        d_base = df.loc[df["scenario"] == s_base].set_index("resolution")[metric]
        d_treat = df.loc[df["scenario"] == s_treat].set_index("resolution")[metric]
        deltas = d_treat - d_base
        deltas = deltas.dropna()
        if not deltas.empty:
            delta_pct = 100 * deltas.abs().max() / abs(baseline_val)
            factors["Wake losses"] = delta_pct

    # 3) Bias effect: bias vs base
    for scen_pair in [("base", "bias")]:
        s_base, s_treat = scen_pair
        d_base = df.loc[df["scenario"] == s_base].set_index("resolution")[metric]
        d_treat = df.loc[df["scenario"] == s_treat].set_index("resolution")[metric]
        deltas = d_treat - d_base
        deltas = deltas.dropna()
        if not deltas.empty:
            delta_pct = 100 * deltas.abs().max() / abs(baseline_val)
            factors["Bias correction"] = delta_pct

    # 4) Combined bias+wake effect (interaction)
    for scen_pair in [("base", "bias+wake")]:
        s_base, s_treat = scen_pair
        d_base = df.loc[df["scenario"] == s_base].set_index("resolution")[metric]
        d_treat = df.loc[df["scenario"] == s_treat].set_index("resolution")[metric]
        deltas = d_treat - d_base
        deltas = deltas.dropna()
        if not deltas.empty:
            delta_pct = 100 * deltas.abs().max() / abs(baseline_val)
            factors["Bias + wake (combined)"] = delta_pct

    if not factors:
        print(f"[WARN] Tornado: no factor deltas computed for {metric}; skipping.")
        plt.close(fig)
        return

    # Sort by absolute magnitude
    sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
    labels = [f[0] for f in sorted_factors]
    values = [f[1] for f in sorted_factors]

    colors = {
        "Spatial resolution": "#E69F00",
        "Wake losses": THESIS_COLORS.get("wake", "#2F4B7C"),
        "Bias correction": THESIS_COLORS.get("bias", "#5DAE8B"),
        "Bias + wake (combined)": THESIS_COLORS.get("bias+wake", "#8172B2"),
    }
    bar_colors = [colors.get(l, "#999999") for l in labels]

    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color=bar_colors, edgecolor="black", linewidth=0.6, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(f"Max |delta {metric_label}| relative to baseline (%)")
    ax.set_title(f"Factor importance: {metric_label}", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(values):
        ax.text(v + 0.1, i, f"{v:.1f}%", va="center", fontsize=7)

    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_interaction_heatmap(
    df: pd.DataFrame,
    outpath: Path,
    metric: str = "objective",
    metric_label: str = "System cost (EUR)",
    resolution: int | None = None,
) -> None:
    """2x2 heatmap: rows=bias (off/on), cols=wake (off/on), cells=metric value.

    If resolution is None, uses the finest available resolution.
    """
    if resolution is None:
        resolution = df["resolution"].min()

    sub = df[df["resolution"] == resolution].copy()
    if sub.empty:
        print(f"[WARN] Heatmap: no data at resolution {resolution}")
        return

    # Build 2x2 matrix
    scenarios = {
        (False, False): "base",
        (True, False): "bias",
        (False, True): "wake",
        (True, True): "bias+wake",
    }

    matrix = np.full((2, 2), np.nan)
    for (bias_on, wake_on), scen in scenarios.items():
        val = sub.loc[sub["scenario"] == scen, metric]
        if not val.empty:
            matrix[int(bias_on), int(wake_on)] = val.values[0]

    # Check if we have enough data
    if np.all(np.isnan(matrix)):
        print(f"[WARN] Heatmap: all NaN for {metric} at resolution {resolution}")
        return

    fig, ax = plt.subplots(figsize=(4.2, 3.5), dpi=300)

    # Normalize to baseline for percentage display
    baseline = matrix[0, 0]
    if not np.isnan(baseline) and baseline != 0:
        pct_matrix = 100 * (matrix - baseline) / abs(baseline)
    else:
        pct_matrix = matrix

    im = ax.imshow(pct_matrix, cmap="RdYlGn_r", aspect="auto")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(f"Delta {metric_label} vs baseline (%)", fontsize=7)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Wake off", "Wake on"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Bias off", "Bias on"])

    # Annotate cells
    for i in range(2):
        for j in range(2):
            val = pct_matrix[i, j]
            if not np.isnan(val):
                text_color = "white" if abs(val) > np.nanmax(np.abs(pct_matrix)) * 0.6 else "black"
                ax.text(j, i, f"{val:+.2f}%", ha="center", va="center",
                       fontsize=8, fontweight="bold", color=text_color)

    ax.set_title(f"Interaction: bias x wake\n(resolution={resolution:,} km²)", fontsize=9)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_interaction_wake_effect(df: pd.DataFrame, ycol: str, ylabel: str, outpath: Path, title: str) -> None:
    """Δ(wake) = metric(wake) - metric(off), split by bias on/off."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    res = sorted(df["resolution"].unique())
    if INVERT_RESOLUTION_AXIS:
        res = list(reversed(res))

    for bias in ["false", "true"]:
        d_off = df[(df["bias"] == bias) & (df["wake"] == "off")].set_index("resolution")
        d_wk = df[(df["bias"] == bias) & (df["wake"] != "off")].set_index("resolution")
        dd = (d_wk[ycol] - d_off[ycol]).reindex(res)
        label_key = "bias" if bias == "true" else "base"
        label_text = "With Bias" if bias == "true" else "Without Bias"
        ax.plot(res, dd.values, marker="o", linewidth=1.2, color=THESIS_COLORS.get(label_key, None), label=label_text)
    
    apply_spatial_resolution_axis(ax, annotate=False)
    add_resolution_markers(ax, res)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=False, fontsize=7)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main workflow
# ----------------------------


def compute_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        n = load_network(r["path"])
        rows.append(
            {
                **r.to_dict(),
                "scenario": scenario_key(r["bias"], str(r["wake"])),
                # capacities
                "offwind_cap_gw": wind_capacity_gw(n, "offwind"),
                "onwind_cap_gw": wind_capacity_gw(n, "onwind"),
                # curtailment
                "offwind_curt_frac": wind_curtailment_frac(n, "offwind"),
                "onwind_curt_frac": wind_curtailment_frac(n, "onwind"),
                # network/system
                "trans_exp_twkm": transmission_expansion_twkm(n),
                "objective": get_objective(n),
            }
        )
    return pd.DataFrame(rows)


def write_summary_table(df: pd.DataFrame, outpath: Path, coarse: int, fine: int) -> None:
    """Small anchor table for baseline vs bias+wake at coarse/fine."""
    sel = df[(df["resolution"].isin([coarse, fine])) & (df["scenario"].isin(["base", "bias+wake"]))].copy()
    table = sel.pivot_table(
        index="scenario",
        columns="resolution",
        values=[
            "offwind_cap_gw",
            "onwind_cap_gw",
            "offwind_curt_frac",
            "onwind_curt_frac",
            "trans_exp_twkm",
            "objective",
        ],
        aggfunc="first",
    )
    table.to_csv(outpath)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results", help="Root directory containing results/")
    ap.add_argument(
        "--glob",
        default="thesis-sensitivity-*/**/postnetworks/*.nc",
        help="Glob relative to results-root to find networks",
    )
    ap.add_argument("--outdir", default="plots/sensitivity/tier1", help="Output directory")
    ap.add_argument("--coarse", type=int, default=100000, help="Coarse resolution (for CF plots/table)")
    ap.add_argument("--fine", type=int, default=1000, help="Fine resolution (for CF plots/table)")
    ap.add_argument("--write-manifest", action="store_true", help="Write manifest.csv to outdir")
    ap.add_argument(
        "--compare",
        nargs="+",
        default=["base", "biasUniform", "bias", "wake", "bias+wake"],
        help="Scenarios to plot (ordered)",
    )
    args = ap.parse_args()

    results_root = Path(args.results_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(results_root, args.glob)
    # Your Tier-1 set
    manifest = manifest[manifest["wake"].isin(["off", "density"])].copy()

    # Validate completeness (nice warning, not fatal)
    expected_res = sorted(manifest["resolution"].unique())
    expected = {(res, bias, wake) for res in expected_res for bias in ["false", "true"] for wake in ["off", "density"]}
    found = {(int(r.resolution), str(r.bias), str(r.wake)) for r in manifest.itertuples()}
    missing = sorted(expected - found)
    if missing:
        warnings.warn(f"Missing {len(missing)} combinations (resolution, bias, wake). First few: {missing[:10]}")

    if args.write_manifest:
        manifest.to_csv(outdir / "manifest.csv", index=False)

    compare = [s for s in args.compare if s in SCEN_ORDER]

    # CF distributions (both techs)
    cf_long = build_cf_long_both(manifest, keep_resolutions=(args.coarse, args.fine), compare=compare)
    cf_long.to_csv(outdir / "tier1_cf_long_on_off.csv", index=False)
    print("[CF] counts:\n", cf_long.groupby(["tech", "resolution", "scenario"]).size())

    plot_cf_2x2(cf_long, outdir / "fig_cf_disp_2x2.png", metric="disp_cf", ylabel="Dispatch CF", resolutions=(args.coarse, args.fine), compare=compare)
    plot_cf_2x2(cf_long, outdir / "fig_cf_avail_2x2.png", metric="avail_cf", ylabel="Availability CF", resolutions=(args.coarse, args.fine), compare=compare)
    plot_cf_2x2(cf_long, outdir / "fig_cf_curt_2x2.png", metric="curt_cf", ylabel="Curtailment CF", resolutions=(args.coarse, args.fine), compare=compare)

    # Core scalar metrics
    metrics = compute_metrics(manifest)
    metrics.to_csv(outdir / "tier1_metrics_all.csv", index=False)

    # Capacity plots (on + off)
    plot_capacity_two_panel(metrics, outdir / "fig_capacity_on_off_vs_resolution.png", compare=compare)

    # Curtailment plots (separate for on/off)
    plot_lines_vs_resolution(metrics, "offwind_curt_frac", "Offshore wind curtailment (fraction)", outdir / "fig_curt_offwind_vs_resolution.png", "Offshore curtailment sensitivity (Tier 1)", compare=compare)
    plot_lines_vs_resolution(metrics, "onwind_curt_frac", "Onshore wind curtailment (fraction)", outdir / "fig_curt_onwind_vs_resolution.png", "Onshore curtailment sensitivity (Tier 1)", compare=compare)

    # Transmission + objective-derived plots
    plot_lines_vs_resolution(metrics, "trans_exp_twkm", "Transmission expansion (TW·km)", outdir / "fig_trans_exp_vs_resolution.png", "Network reinforcement sensitivity (Tier 1)", compare=compare)
    plot_marginal_value_proxy(metrics, outdir / "fig_marginal_value_proxy_offwind.png", cap_col="offwind_cap_gw", compare=compare)

    # Interaction plots (wake effect) - do this mainly for offshore metrics
    plot_interaction_wake_effect(metrics, "offwind_cap_gw", "Δ offshore capacity due to wakes (GW)", outdir / "fig_interaction_wake_offwind_cap.png", "Wake impact on offshore capacity, split by bias")
    plot_interaction_wake_effect(metrics, "trans_exp_twkm", "Δ transmission expansion due to wakes (TW·km)", outdir / "fig_interaction_wake_trans.png", "Wake impact on grid reinforcement, split by bias")

    write_summary_table(metrics, outdir / "table_summary_coarse_fine.csv", coarse=args.coarse, fine=args.fine)

    # Tornado chart: factor importance for system cost
    plot_tornado_factor_importance(metrics, outdir / "fig_tornado_system_cost.png", metric="objective", metric_label="System cost", compare=compare)
    # Tornado for offshore capacity
    plot_tornado_factor_importance(metrics, outdir / "fig_tornado_offwind_cap.png", metric="offwind_cap_gw", metric_label="Offshore capacity (GW)", compare=compare)
    # Tornado for transmission
    plot_tornado_factor_importance(metrics, outdir / "fig_tornado_transmission.png", metric="trans_exp_twkm", metric_label="Transmission expansion (TW*km)", compare=compare)

    # Interaction heatmaps at coarse and fine resolution
    plot_interaction_heatmap(metrics, outdir / "fig_heatmap_cost_coarse.png", metric="objective", metric_label="System cost", resolution=args.coarse)
    plot_interaction_heatmap(metrics, outdir / "fig_heatmap_cost_fine.png", metric="objective", metric_label="System cost", resolution=args.fine)
    plot_interaction_heatmap(metrics, outdir / "fig_heatmap_offwind_cap_fine.png", metric="offwind_cap_gw", metric_label="Offshore capacity (GW)", resolution=args.fine)

    # System cost vs resolution (was missing as a dedicated plot)
    plot_lines_vs_resolution(metrics, "objective", "System objective (EUR)", outdir / "fig_objective_vs_resolution.png", "System cost sensitivity (Tier 1)", compare=compare)

    print(f"[OK] outdir: {outdir}")
    print(f"[OK] metrics: {outdir / 'tier1_metrics_all.csv'}")
    if args.write_manifest:
        print(f"[OK] manifest: {outdir / 'manifest.csv'}")


if __name__ == "__main__":
    main()


# Example usage:
# python compare_sensitivity_runs_sector_styled.py \
#   --results-root results \
#   --glob "thesis-sensitivity-2030-10-northsea-dominant-6h/**/postnetworks/*.nc" \
#   --outdir plots/sensitivity/tier1

# python compare_sensitivity_runs_sector_styled.py \
#   --results-root ../results \
#   --glob "thesis-sensitivity-2030-30-europe-dominant-6h/**/postnetworks/*.nc" \
#   --outdir ../plots/sensitivity/tier2 \
#   --coarse 1000000 \
#   --fine 10000
#!/usr/bin/env python3
"""
Compare optimised 2030 wind capacities from tier-2 sensitivity runs against
TYNDP 2024 National Trends "Best Estimate" 2030 targets.

This is a *consistency check* (not strict validation): the model optimises
capacity endogenously under cost assumptions, while TYNDP provides
policy-informed national deployment targets.

Produces:
  1. Scatter plot: model vs TYNDP capacity per country (onshore + offshore)
  2. Bar chart: country-level comparison (grouped by scenario)
  3. Ratio heatmap: model / TYNDP by scenario x country
  4. Summary CSV with per-country, per-scenario model and TYNDP values
  5. Aggregate statistics printed to console

Run from the repo root:
    python analysis_scripts/validate_capacity_vs_tyndp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plotting_style import thesis_plot_style, format_axes_standard
from thesis_colors import THESIS_COLORS, label as get_label
from network_utils import (
    load_network,
    build_manifest,
    scenario_key,
    bus_country,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("results")
RESULTS_PREFIX = "thesis-sensitivity-2030-30-europe-dominant-6h"
RESULTS_GLOB = f"{RESULTS_PREFIX}/*/postnetworks/*.nc"
OUTDIR = Path("plots/sensitivity/validation_tyndp")

TIER2_COUNTRIES = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "ES",
    "FR", "GB", "GR", "HR", "IE", "IT", "LU", "ME", "MK", "NL",
    "NO", "PT", "RS", "SE", "SI",
]

# TYNDP 2024 National Trends — Best Estimate 2030 (MW)
# Source: "Final Supply Inputs for TYNDP 2024 Scenarios", sheets 1.2. & 1.3.
TYNDP_ONSHORE_MW = {
    "AL": 280.0, "AT": 9000.0, "BA": 786.6, "BE": 5248.0,
    "BG": 948.0, "CH": 310.0, "CZ": 958.3, "DE": 115000.7,
    "DK": 7305.6, "ES": 48317.5, "FR": 31300.0, "GR": 7100.0,
    "HR": 1442.0, "IE": 8952.8, "IT": 18410.8, "LU": 400.0,
    "ME": 254.2, "MK": 410.0, "NL": 9100.0, "NO": 5773.2,
    "PT": 8699.5, "RS": 4812.2, "SE": 17952.2, "SI": 122.4,
}

TYNDP_OFFSHORE_MW = {
    "BE": 5760.0, "DE": 30521.3, "DK": 8250.6, "ES": 2800.0,
    "FR": 3875.0, "GR": 2700.0, "HR": 510.0, "IE": 5025.2,
    "IT": 8500.0, "NL": 16542.5, "NO": 3003.5, "PT": 2000.0,
    "SE": 600.0,
}

TECHS = {
    "onshore": {"model_carriers": ["onwind"], "tyndp": TYNDP_ONSHORE_MW},
    "offshore": {
        "model_carriers": ["offwind-ac", "offwind-dc", "offwind-float"],
        "tyndp": TYNDP_OFFSHORE_MW,
    },
}


# ---------------------------------------------------------------------------
# Model extraction — optimised capacity by country
# ---------------------------------------------------------------------------

def model_capacity_by_country(n, countries: list[str]) -> dict[tuple[str, str], float]:
    """Extract optimised capacity (MW) per (country, tech) from a network."""
    result = {}
    g = n.generators

    for tech_label, cfg in TECHS.items():
        carriers = cfg["model_carriers"]
        mask = g.carrier.isin(carriers)
        if not mask.any():
            continue
        idx = g.index[mask]
        gen_cc = g.loc[idx, "bus"].map(bus_country)

        if "p_nom_opt" in g.columns:
            cap = g.loc[idx, "p_nom_opt"].fillna(g.loc[idx, "p_nom"])
        else:
            cap = g.loc[idx, "p_nom"]
        cap = cap.fillna(0.0)

        for cc in countries:
            cc_cap = cap[gen_cc == cc].sum()
            result[(cc, tech_label)] = float(cc_cap)

    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_scatter_model_vs_tyndp(
    df: pd.DataFrame,
    tech: str,
    outpath: Path,
) -> None:
    """Scatter plot: model capacity vs TYNDP target for each country."""
    style = thesis_plot_style()
    cm = style["cm"]

    sub = df[(df["tech"] == tech) & (df["tyndp_gw"] > 0)].copy()
    if sub.empty:
        return

    scenarios = sorted(sub["scenario"].unique())
    colors = {s: THESIS_COLORS.get(s, "#666666") for s in scenarios}

    fig, ax = plt.subplots(figsize=(10 * cm, 10 * cm), dpi=600)

    max_val = max(sub["tyndp_gw"].max(), sub["model_gw"].max()) * 1.15

    # 1:1 line
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=0.6, alpha=0.5, zorder=0)

    for scen in scenarios:
        s = sub[sub["scenario"] == scen]
        ax.scatter(
            s["tyndp_gw"], s["model_gw"],
            s=12, alpha=0.7, color=colors[scen],
            label=get_label(scen), edgecolors="none", zorder=2,
        )
        # Label points for the largest countries
        for _, row in s.iterrows():
            if row["tyndp_gw"] > max_val * 0.15 or row["model_gw"] > max_val * 0.15:
                ax.annotate(
                    row["country"], (row["tyndp_gw"], row["model_gw"]),
                    fontsize=4, alpha=0.7, textcoords="offset points",
                    xytext=(3, 3),
                )

    ax.set_xlabel("TYNDP 2024 Best Estimate [GW]")
    ax.set_ylabel("Model optimised capacity [GW]")
    ax.set_title(f"{tech.capitalize()} wind: model vs TYNDP 2030")
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect("equal")
    ax.legend(fontsize=5, frameon=False, loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_bar_country_comparison(
    df: pd.DataFrame,
    tech: str,
    ref_scenario: str,
    outpath: Path,
) -> None:
    """Bar chart comparing model capacity and TYNDP target per country."""
    style = thesis_plot_style()
    cm = style["cm"]

    # Use fine resolution only to avoid duplicate country bars
    sub = df[
        (df["tech"] == tech)
        & (df["scenario"] == ref_scenario)
        & (df["resolution"] == 10000)
    ].copy()
    sub = sub[sub["tyndp_gw"] > 0].sort_values("tyndp_gw", ascending=False)
    if sub.empty:
        return

    countries = sub["country"].values
    x = np.arange(len(countries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(17.8 * cm, 6 * cm), dpi=600)
    ax.bar(x - width / 2, sub["tyndp_gw"].values, width, label="TYNDP 2024",
           color="#4D4D4D", alpha=0.8)
    ax.bar(x + width / 2, sub["model_gw"].values, width,
           label=f"Model ({get_label(ref_scenario)})",
           color=THESIS_COLORS.get(ref_scenario, "#2F4B7C"), alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(countries, fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("Capacity [GW]")
    ax.set_title(f"{tech.capitalize()} wind capacity: model vs TYNDP 2030")
    ax.legend(fontsize=5, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_ratio_heatmap(
    df: pd.DataFrame,
    tech: str,
    outpath: Path,
) -> None:
    """Heatmap of model/TYNDP ratio across scenarios and countries."""
    style = thesis_plot_style()
    cm = style["cm"]

    sub = df[(df["tech"] == tech) & (df["tyndp_gw"] > 0)].copy()
    if sub.empty:
        return

    pivot = sub.pivot_table(
        index="scenario_label", columns="country", values="ratio", aggfunc="mean",
    )
    col_order = [c for c in TIER2_COUNTRIES if c in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(
        figsize=(max(8, len(col_order) * 0.8) * cm, max(4, len(pivot) * 0.6) * cm),
        dpi=600,
    )

    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=2.0,
    )
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=5, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=5)

    for i in range(len(pivot)):
        for j in range(len(col_order)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4)

    ax.set_title(f"{tech.capitalize()} wind: model / TYNDP ratio")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Model / TYNDP", fontsize=6)

    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_total_capacity_bars(
    df: pd.DataFrame,
    outpath: Path,
) -> None:
    """Stacked bar chart: total onshore + offshore capacity by scenario vs TYNDP."""
    style = thesis_plot_style()
    cm = style["cm"]

    # Aggregate: total GW per scenario+tech
    agg = df.groupby(["scenario_label", "tech"]).agg(
        model_gw=("model_gw", "sum"),
        tyndp_gw=("tyndp_gw", "sum"),
    ).reset_index()

    scenarios = sorted(agg["scenario_label"].unique())
    # Add TYNDP as a separate bar
    labels = ["TYNDP 2024"] + scenarios

    on_model = []
    off_model = []
    on_tyndp = []
    off_tyndp = []

    for scen in scenarios:
        s = agg[agg["scenario_label"] == scen]
        on_row = s[s["tech"] == "onshore"]
        off_row = s[s["tech"] == "offshore"]
        on_model.append(float(on_row["model_gw"].sum()) if not on_row.empty else 0)
        off_model.append(float(off_row["model_gw"].sum()) if not off_row.empty else 0)
        on_tyndp.append(float(on_row["tyndp_gw"].sum()) if not on_row.empty else 0)
        off_tyndp.append(float(off_row["tyndp_gw"].sum()) if not off_row.empty else 0)

    # TYNDP total (same for all scenarios)
    tyndp_on_total = on_tyndp[0] if on_tyndp else 0
    tyndp_off_total = off_tyndp[0] if off_tyndp else 0

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(17.8 * cm, 7 * cm), dpi=600)

    # TYNDP bar
    on_vals = [tyndp_on_total] + on_model
    off_vals = [tyndp_off_total] + off_model

    bars_on = ax.bar(x, on_vals, width, label="Onshore", color="#5DAE8B", alpha=0.85)
    bars_off = ax.bar(x, off_vals, width, bottom=on_vals, label="Offshore",
                      color="#2F4B7C", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("Total capacity [GW]")
    ax.set_title("Total wind capacity: model scenarios vs TYNDP 2024")
    ax.legend(fontsize=5, frameon=False)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate totals on bars
    for i, (on, off) in enumerate(zip(on_vals, off_vals)):
        total = on + off
        ax.text(i, total + 2, f"{total:.0f}", ha="center", va="bottom", fontsize=5)

    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    thesis_plot_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    # --- Build manifest ---
    print(f"Building manifest from {RESULTS_ROOT / RESULTS_GLOB}")
    manifest = build_manifest(RESULTS_ROOT, RESULTS_GLOB)
    print(f"  Found {len(manifest)} networks")

    # --- Extract model capacities ---
    all_rows = []

    for _, mrow in manifest.iterrows():
        nc_path = Path(mrow["path"])
        res = int(mrow["resolution"])
        bias = mrow["bias"]
        wake = mrow["wake"]
        scen = scenario_key(bias, wake)
        scenario_label = f"{scen} (s={res:,})"

        print(f"  Processing: {mrow['scenario_folder']} -> {scenario_label}")
        n = load_network(nc_path)

        cap_by_cc = model_capacity_by_country(n, TIER2_COUNTRIES)

        for cc in TIER2_COUNTRIES:
            for tech_label, cfg in TECHS.items():
                tyndp_mw = cfg["tyndp"].get(cc, 0.0)
                model_mw = cap_by_cc.get((cc, tech_label), 0.0)

                ratio = model_mw / tyndp_mw if tyndp_mw > 0 else np.nan

                all_rows.append({
                    "country": cc,
                    "tech": tech_label,
                    "scenario": scen,
                    "scenario_label": scenario_label,
                    "resolution": res,
                    "bias": bias,
                    "wake": wake,
                    "model_mw": model_mw,
                    "model_gw": model_mw / 1e3,
                    "tyndp_mw": tyndp_mw,
                    "tyndp_gw": tyndp_mw / 1e3,
                    "ratio": ratio,
                    "diff_gw": (model_mw - tyndp_mw) / 1e3,
                    "scenario_folder": mrow["scenario_folder"],
                })

    results = pd.DataFrame(all_rows)
    results.to_csv(OUTDIR / "capacity_vs_tyndp.csv", index=False)
    print(f"\n  Saved metrics to {OUTDIR / 'capacity_vs_tyndp.csv'}")
    print(f"  Total rows: {len(results)}")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("CAPACITY vs TYNDP 2024 SUMMARY")
    print("=" * 80)
    for tech_label in ["onshore", "offshore"]:
        sub = results[(results["tech"] == tech_label) & (results["tyndp_mw"] > 0)]
        if sub.empty:
            continue
        print(f"\n--- {tech_label.upper()} ---")
        grp = sub.groupby(["scenario", "resolution"]).agg(
            model_total_gw=("model_gw", "sum"),
            tyndp_total_gw=("tyndp_gw", "sum"),
            mean_ratio=("ratio", "mean"),
            median_ratio=("ratio", "median"),
        )
        grp["total_ratio"] = grp["model_total_gw"] / grp["tyndp_total_gw"]
        print(grp.to_string(float_format=lambda x: f"{x:.3f}"))

    # --- Plots ---

    # 1. Scatter plots (model vs TYNDP per country)
    for tech_label in ["onshore", "offshore"]:
        plot_scatter_model_vs_tyndp(
            results, tech_label,
            OUTDIR / f"scatter_{tech_label}_model_vs_tyndp.png",
        )

    # 2. Country-level bar charts for baseline scenario
    ref_scenario = "base"
    for tech_label in ["onshore", "offshore"]:
        plot_bar_country_comparison(
            results, tech_label, ref_scenario,
            OUTDIR / f"bars_{tech_label}_vs_tyndp_{ref_scenario}.png",
        )

    # 3. Ratio heatmaps
    for tech_label in ["onshore", "offshore"]:
        plot_ratio_heatmap(
            results, tech_label,
            OUTDIR / f"heatmap_{tech_label}_ratio.png",
        )

    # 4. Total capacity stacked bars
    plot_total_capacity_bars(
        results,
        OUTDIR / "total_capacity_vs_tyndp.png",
    )

    print(f"\nAll outputs saved to {OUTDIR}")


if __name__ == "__main__":
    main()

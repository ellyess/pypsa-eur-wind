#!/usr/bin/env python3
"""
wake_extra_plots.py

Additional thesis-quality plots for the wake modelling chapter.

Generates:
1. capacity_mix.pdf       -- Stacked bar: installed capacity (GW) by carrier
2. energy_mix.pdf         -- Stacked bar: annual generation (TWh) by carrier
3. offshore_cap_vs_resolution.pdf -- Offshore wind capacity vs spatial resolution
4. system_cost_delta.pdf  -- % change in system cost vs baseline
5. curtailment_rate.pdf   -- Offshore wind curtailment rate (%)
6. cf_heatmap.pdf         -- Heatmap of mean available CF per generator region
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import pypsa

# ---------------------------------------------------------------------------
# Import thesis styling (robust to being run from anywhere)
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
for p in [HERE, HERE.parent, Path.cwd()]:
    if (p / "thesis_colors.py").exists() and (p / "plotting_style.py").exists():
        sys.path.insert(0, str(p))
        break

from thesis_colors import (
    canon,
    label,
    WAKE_ORDER,
    WAKE_MODEL_COLORS,
)
from plotting_style import thesis_plot_style, apply_spatial_resolution_axis, savefig_thesis
from plotlib.io import PLOTS_ROOT as _PLOTS_ROOT, RESULTS_ROOT as _RESULTS_ROOT

from network_utils import (
    load_network,
    capacity_by_carrier,
    energy_by_carrier_twh,
    wind_capacity_gw,
    wind_curtailment_frac,
    snapshot_weights,
    gen_idx,
    get_objective,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_RUN = "thesis-wake-2030-10-northsea-dominant-6h"

# Overridden by --results-dir / --out-dir. They follow run_all.py's roots, which
# honour the PYPSA_RESULTS_ROOT and PYPSA_PLOTS_ROOT environment variables.
RESULTS_ROOT = _RESULTS_ROOT / DEFAULT_RUN
OUT_DIR = _PLOTS_ROOT / "wake_analysis"
NC_TEMPLATE = "{scenario}-s{split}-biasFalse/networks/base_s_10_elec_lvopt_.nc"

SCENARIOS = ["base", "standard", "glaum", "new_more"]
SPLITS = [1000, 5000, 10000, 50000, 100000]
DEFAULT_SPLIT = 1000

# Technology display names for stacked bars (from PyPSA-Eur nice_names)
TECH_LABELS = {
    "onwind": "Onshore Wind",
    "offwind-ac": "Offshore Wind (AC)",
    "offwind-dc": "Offshore Wind (DC)",
    "offwind-float": "Offshore Wind (Floating)",
    "solar": "Solar",
    "OCGT": "Open-Cycle Gas",
    "CCGT": "Combined-Cycle Gas",
    "nuclear": "Nuclear",
    "coal": "Coal",
    "lignite": "Lignite",
    "biomass": "Biomass",
    "solid biomass": "Solid Biomass",
    "biogas": "Biogas",
    "oil": "Oil",
    "gas": "Gas",
    "battery": "Battery Storage",
    "H2 Electrolysis": "H\u2082 Electrolysis",
    "H2 Fuel Cell": "H\u2082 Fuel Cell",
    "hydro": "Reservoir & Dam",
    "ror": "Run of River",
    "PHS": "Pumped Hydro Storage",
    "DC": "HVDC Links",
    "AC": "AC",
    "load": "Load Shedding",
}

# Technology colors for stacked bars (from PyPSA-Eur tech_colors)
TECH_COLORS = {
    # wind
    "onwind": "#235ebc",
    "offwind-ac": "#6895dd",
    "offwind-dc": "#74c6f2",
    "offwind-float": "#b5e2fa",
    # solar
    "solar": "#f9d002",
    # gas
    "OCGT": "#e0986c",
    "CCGT": "#a85522",
    "gas": "#e05b09",
    # coal / oil / nuclear
    "coal": "#545454",
    "lignite": "#826837",
    "oil": "#c9c9c9",
    "nuclear": "#ff8c00",
    # biomass
    "biomass": "#baa741",
    "solid biomass": "#baa741",
    "biogas": "#e3d37d",
    # hydro
    "hydro": "#298c81",
    "ror": "#3dbfb0",
    "PHS": "#51dbcc",
    # storage / conversion
    "battery": "#ace37f",
    "H2 Electrolysis": "#ff29d9",
    "H2 Fuel Cell": "#c251ae",
    "H2 Store": "#bf13a0",
    # transmission
    "DC": "#8a1caf",
    "AC": "#70af1d",
    "lines": "#6c9459",
    # other
    "load": "#dd2e23",
    "Other": "#aaaaaa",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _color_for(scenario_key: str) -> str:
    k = canon(scenario_key)
    return WAKE_MODEL_COLORS.get(k, "#4D4D4D")


def _sorted_scenarios(keys: list[str]) -> list[str]:
    c = [canon(k) for k in keys]
    order = {k: i for i, k in enumerate(WAKE_ORDER)}
    return sorted(set(c), key=lambda k: order.get(k, 9999))


_module_style = thesis_plot_style()
_cm = _module_style['cm']
FULL_WIDTH = _module_style['FULL_WIDTH']
HALF_WIDTH = _module_style['HALF_WIDTH']


def _savefig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"  Saved: {out}")
    plt.close(fig)


def _nc_path(scenario: str, split: int) -> Path | None:
    p = RESULTS_ROOT / NC_TEMPLATE.format(scenario=scenario, split=split)
    if p.exists():
        return p
    # Electricity-only and sector-coupled runs name their solved network
    # differently (base_s_10_elec_lvopt_.nc vs base_s_10___2030.nc); fall back
    # to whatever single .nc the run wrote.
    candidates = sorted(p.parent.glob("*.nc")) if p.parent.exists() else []
    return candidates[0] if candidates else p


def _load_if_exists(scenario: str, split: int) -> pypsa.Network | None:
    p = _nc_path(scenario, split)
    if p is not None and p.exists():
        return load_network(p)
    else:
        print(f"  [WARN] Network not found: {p}")
        return None


def _tech_color(carrier: str) -> str:
    return TECH_COLORS.get(carrier, "#888888")


def _tech_label(carrier: str) -> str:
    return TECH_LABELS.get(carrier, carrier)


# ---------------------------------------------------------------------------
# 1) Capacity mix bar chart
# ---------------------------------------------------------------------------

def plot_capacity_mix(
    networks: dict[str, pypsa.Network],
    *,
    out: Path,
    top_n: int = 8,
) -> None:
    """Stacked bar chart: installed capacity (GW) by carrier for each wake scenario."""
    style = thesis_plot_style()
    cm = style["cm"]

    # Collect capacity data
    all_cap = {}
    for scen, n in networks.items():
        cap = capacity_by_carrier(n) / 1e3  # MW -> GW
        all_cap[scen] = cap

    df = pd.DataFrame(all_cap).fillna(0.0)

    # Select top N carriers by total capacity across all scenarios
    total = df.sum(axis=1).sort_values(ascending=False)
    top_carriers = total.head(top_n).index.tolist()

    # Add "Other" for the rest if needed
    other = total.index.difference(top_carriers)
    if len(other) > 0:
        df_plot = df.loc[top_carriers].copy()
        df_plot.loc["Other"] = df.loc[other].sum()
        top_carriers = top_carriers + ["Other"]
    else:
        df_plot = df.loc[top_carriers].copy()

    # Reorder scenarios
    ordered_scen = _sorted_scenarios(list(networks.keys()))
    df_plot = df_plot[[s for s in ordered_scen if s in df_plot.columns]]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 8.0 * cm), layout="constrained")

    x = np.arange(len(df_plot.columns))
    bar_width = 0.6
    bottom = np.zeros(len(x))

    for carrier in top_carriers:
        vals = df_plot.loc[carrier].values.astype(float)
        color = _tech_color(carrier)
        lbl = _tech_label(carrier)
        ax.bar(
            x, vals, bar_width,
            bottom=bottom,
            label=lbl,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([label(s) for s in df_plot.columns], rotation=15, ha="right")
    ax.set_ylabel("Installed capacity [GW]")
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        fontsize=6,
    )

    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 2) Energy mix bar chart
# ---------------------------------------------------------------------------

def plot_energy_mix(
    networks: dict[str, pypsa.Network],
    *,
    out: Path,
    top_n: int = 8,
) -> None:
    """Stacked bar chart: annual generation (TWh) by carrier for each wake scenario."""
    style = thesis_plot_style()
    cm = style["cm"]

    all_en = {}
    for scen, n in networks.items():
        en = energy_by_carrier_twh(n)
        all_en[scen] = en

    df = pd.DataFrame(all_en).fillna(0.0)

    # Only keep positive generation (ignore negative/consumption)
    df = df.clip(lower=0.0)

    total = df.sum(axis=1).sort_values(ascending=False)
    top_carriers = total.head(top_n).index.tolist()

    other = total.index.difference(top_carriers)
    if len(other) > 0:
        df_plot = df.loc[top_carriers].copy()
        df_plot.loc["Other"] = df.loc[other].sum()
        top_carriers = top_carriers + ["Other"]
    else:
        df_plot = df.loc[top_carriers].copy()

    ordered_scen = _sorted_scenarios(list(networks.keys()))
    df_plot = df_plot[[s for s in ordered_scen if s in df_plot.columns]]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 8.0 * cm), layout="constrained")

    x = np.arange(len(df_plot.columns))
    bar_width = 0.6
    bottom = np.zeros(len(x))

    for carrier in top_carriers:
        vals = df_plot.loc[carrier].values.astype(float)
        color = _tech_color(carrier)
        lbl = _tech_label(carrier)
        ax.bar(
            x, vals, bar_width,
            bottom=bottom,
            label=lbl,
            color=color,
            edgecolor="white",
            linewidth=0.4,
        )
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([label(s) for s in df_plot.columns], rotation=15, ha="right")
    ax.set_ylabel("Annual generation [TWh]")
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        fontsize=6,
    )

    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 3) Offshore capacity vs spatial resolution
# ---------------------------------------------------------------------------

def plot_offshore_cap_vs_resolution(
    *,
    out: Path,
    scenarios: list[str] | None = None,
    splits: list[int] | None = None,
) -> None:
    """Line plot: total offshore wind capacity (GW) across spatial resolutions."""
    if scenarios is None:
        scenarios = SCENARIOS
    if splits is None:
        splits = SPLITS

    style = thesis_plot_style()
    cm = style["cm"]
    lw = style["lw"]
    ms = style["ms"]

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 7.0 * cm), layout="constrained")

    ordered = _sorted_scenarios(scenarios)
    markers = ["o", "s", "^", "D"]

    for i, scen in enumerate(ordered):
        x_vals, y_vals = [], []
        for split in sorted(splits):
            n = _load_if_exists(scen, split)
            if n is None:
                continue
            cap = wind_capacity_gw(n, "offwind")
            x_vals.append(split)
            y_vals.append(cap)
            del n  # free memory

        if x_vals:
            ax.plot(
                x_vals, y_vals,
                marker=markers[i % len(markers)],
                linewidth=lw,
                markersize=ms,
                label=label(scen),
                color=_color_for(scen),
            )

    apply_spatial_resolution_axis(ax)
    ax.set_ylabel("Offshore wind capacity [GW]")
    ax.legend(frameon=False, ncol=2)

    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 4) System cost delta bar chart
# ---------------------------------------------------------------------------

def plot_system_cost_delta(
    networks: dict[str, pypsa.Network],
    *,
    out: Path,
    baseline: str = "base",
) -> None:
    """Bar chart: % change in total system cost relative to baseline."""
    style = thesis_plot_style()
    cm = style["cm"]

    if baseline not in networks:
        print(f"  [WARN] Baseline '{baseline}' not loaded. Skipping cost delta plot.")
        return

    base_cost = get_objective(networks[baseline])
    if np.isnan(base_cost) or base_cost == 0:
        print("  [WARN] Baseline cost is NaN or zero. Skipping.")
        return

    ordered = _sorted_scenarios(list(networks.keys()))
    # Exclude baseline itself from bars
    scen_list = [s for s in ordered if s != baseline]

    deltas = []
    for s in scen_list:
        cost = get_objective(networks[s])
        pct = (cost - base_cost) / abs(base_cost) * 100.0
        deltas.append(pct)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 6.5 * cm), layout="constrained")

    x = np.arange(len(scen_list))
    colors = [_color_for(s) for s in scen_list]
    bars = ax.bar(x, deltas, 0.55, color=colors, edgecolor="black", linewidth=0.6)

    # Label values on bars
    for bar, val in zip(bars, deltas):
        va = "bottom" if val >= 0 else "top"
        yoff = 0.15 if val >= 0 else -0.15
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + yoff,
            f"{val:+.2f}%",
            ha="center",
            va=va,
            fontsize=6,
            fontweight="bold",
        )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels([label(s) for s in scen_list], rotation=15, ha="right")
    ax.set_ylabel(f"System cost change vs {label(baseline)} [%]")

    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 5) Offshore wind curtailment rate
# ---------------------------------------------------------------------------

def plot_curtailment_rate(
    *,
    out: Path,
    scenarios: list[str] | None = None,
    splits: list[int] | None = None,
) -> None:
    """Grouped bar chart: offshore wind curtailment rate (%) across scenarios and splits."""
    if scenarios is None:
        scenarios = SCENARIOS
    if splits is None:
        splits = SPLITS

    style = thesis_plot_style()
    cm = style["cm"]

    ordered = _sorted_scenarios(scenarios)

    # Build data matrix [scenario x split]
    data = {}
    for scen in ordered:
        row = {}
        for split in splits:
            n = _load_if_exists(scen, split)
            if n is None:
                row[split] = np.nan
                continue
            frac = wind_curtailment_frac(n, "offwind")
            row[split] = frac * 100.0 if not np.isnan(frac) else np.nan
            del n
        data[scen] = row

    n_scenarios = len(ordered)
    n_splits = len(splits)

    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 7.0 * cm), layout="constrained")

    group_width = n_scenarios + 1.0
    bar_width = 0.75

    for i, split in enumerate(splits):
        for j, scen in enumerate(ordered):
            pos = i * group_width + j
            val = data[scen].get(split, np.nan)
            if np.isnan(val):
                continue
            ax.bar(
                pos, val, bar_width,
                color=_color_for(scen),
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
            )

    # X-axis labels = split groups
    split_labels = [f"{s:,}" for s in splits]
    group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(split_labels)
    ax.set_xlabel(r"Spatial resolution $A_{region}^{max}$ [km$^2$]")
    ax.set_ylabel("Offshore wind curtailment rate [%]")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_color_for(s), alpha=0.85, edgecolor="black",
              linewidth=0.5, label=label(s))
        for s in ordered
    ]
    ax.legend(
        handles=legend_elements,
        title="Wake model",
        frameon=False,
        fontsize=6,
        loc="upper right",
    )

    _savefig(fig, out)


# ---------------------------------------------------------------------------
# 6) Capacity factor heatmap
# ---------------------------------------------------------------------------

def plot_cf_heatmap(
    networks: dict[str, pypsa.Network],
    *,
    out: Path,
) -> None:
    """Heatmap: mean available CF per offshore generator region, rows = scenarios."""
    style = thesis_plot_style()
    cm = style["cm"]

    ordered = _sorted_scenarios(list(networks.keys()))

    # Collect per-generator mean available CF for offshore wind,
    # aggregated by bus (capacity-weighted mean CF per bus).
    cf_dict = {}
    for scen in ordered:
        n = networks[scen]
        idx = gen_idx(n, "offwind")
        if len(idx) == 0:
            continue

        if (
            hasattr(n.generators_t, "p_max_pu")
            and not n.generators_t.p_max_pu.empty
        ):
            valid_idx = idx.intersection(n.generators_t.p_max_pu.columns)
            if len(valid_idx) == 0:
                continue
            cf_per_gen = n.generators_t.p_max_pu[valid_idx].mean(axis=0)
        else:
            continue

        # Capacity-weighted mean CF per bus
        bus_names = n.generators.loc[valid_idx, "bus"]
        p_nom = n.generators.loc[valid_idx]
        if "p_nom_opt" in p_nom.columns and p_nom["p_nom_opt"].notna().any():
            weights = p_nom["p_nom_opt"].fillna(p_nom["p_nom"]).fillna(1.0)
        else:
            weights = p_nom["p_nom"].fillna(1.0)

        tmp = pd.DataFrame({
            "bus": bus_names.values,
            "cf": cf_per_gen.values,
            "w": weights.values,
        })
        tmp["wcf"] = tmp["cf"] * tmp["w"]
        agg = tmp.groupby("bus").agg({"wcf": "sum", "w": "sum"})
        cf_by_bus = agg["wcf"] / agg["w"]
        cf_dict[scen] = cf_by_bus

    if not cf_dict:
        print("  [WARN] No CF data available for heatmap. Skipping.")
        return

    df = pd.DataFrame(cf_dict).T  # rows = scenarios, cols = bus names
    df = df.reindex(ordered)

    # Sort columns (regions) alphabetically
    df = df[sorted(df.columns)]

    # Drop columns that are all NaN
    df = df.dropna(axis=1, how="all")

    if df.empty:
        print("  [WARN] Heatmap data empty after processing. Skipping.")
        return

    # Rename row labels for display
    row_labels = [label(s) for s in df.index]

    # Figure size: adjust width based on number of regions
    n_cols = len(df.columns)
    fig_width = max(FULL_WIDTH, n_cols * 0.18)
    fig_height = max(4.0 * cm, len(df) * 1.0 * cm + 2.0 * cm)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), layout="constrained")

    im = ax.imshow(
        df.values.astype(float),
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=df.values[np.isfinite(df.values)].max() if np.any(np.isfinite(df.values)) else 1.0,
    )

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    # X-axis: show every N-th label to avoid clutter
    step = max(1, n_cols // 25)
    col_labels = list(df.columns)
    ax.set_xticks(range(0, n_cols, step))
    ax.set_xticklabels(
        [col_labels[i] for i in range(0, n_cols, step)],
        rotation=90,
        fontsize=4.5,
    )
    ax.set_xlabel("Generator region (bus)")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Mean available CF [-]")

    _savefig(fig, out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Additional wake-chapter plots built from solved networks."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_ROOT,
        help=f"Run directory holding the solved networks (default: {RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help=f"Where to write the figures (default: {OUT_DIR}).",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    global RESULTS_ROOT, OUT_DIR

    args = _parse_args(argv)
    RESULTS_ROOT = args.results_dir
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("wake_extra_plots.py -- Additional thesis wake chapter plots")
    print("=" * 60)
    print(f"results: {RESULTS_ROOT}")

    # ------------------------------------------------------------------
    # Load networks for default split (1000 km^2)
    # ------------------------------------------------------------------
    print(f"\nLoading networks for split={DEFAULT_SPLIT} ...")
    nets_default: dict[str, pypsa.Network] = {}
    for scen in SCENARIOS:
        n = _load_if_exists(scen, DEFAULT_SPLIT)
        if n is not None:
            nets_default[scen] = n
            print(f"  Loaded {scen}: {len(n.generators)} generators, "
                  f"{len(n.buses)} buses")

    if not nets_default:
        print("[ERROR] No networks loaded. Aborting.")
        return

    # ------------------------------------------------------------------
    # 1) Capacity mix
    # ------------------------------------------------------------------
    print("\n[1/6] Capacity mix bar chart ...")
    try:
        plot_capacity_mix(nets_default, out=OUT_DIR / "capacity_mix.pdf")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ------------------------------------------------------------------
    # 2) Energy mix
    # ------------------------------------------------------------------
    print("\n[2/6] Energy mix bar chart ...")
    try:
        plot_energy_mix(nets_default, out=OUT_DIR / "energy_mix.pdf")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ------------------------------------------------------------------
    # 3) Offshore cap vs resolution  (needs all splits)
    # ------------------------------------------------------------------
    print("\n[3/6] Offshore capacity vs spatial resolution ...")
    try:
        plot_offshore_cap_vs_resolution(out=OUT_DIR / "offshore_cap_vs_resolution.pdf")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # Free memory from default networks before heavy multi-split loads
    plt.close("all")

    # ------------------------------------------------------------------
    # 4) System cost delta
    # ------------------------------------------------------------------
    print("\n[4/6] System cost delta bar chart ...")
    try:
        plot_system_cost_delta(nets_default, out=OUT_DIR / "system_cost_delta.pdf")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ------------------------------------------------------------------
    # 5) Curtailment rate (all splits)
    # ------------------------------------------------------------------
    print("\n[5/6] Offshore wind curtailment rate ...")
    try:
        plot_curtailment_rate(out=OUT_DIR / "curtailment_rate.pdf")
    except Exception as e:
        print(f"  [ERROR] {e}")

    plt.close("all")

    # ------------------------------------------------------------------
    # 6) CF heatmap
    # ------------------------------------------------------------------
    print("\n[6/6] Capacity factor heatmap ...")
    try:
        plot_cf_heatmap(nets_default, out=OUT_DIR / "cf_heatmap.pdf")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    plt.close("all")
    del nets_default

    print("\n" + "=" * 60)
    print("Done. All plots saved to:")
    print(f"  {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()

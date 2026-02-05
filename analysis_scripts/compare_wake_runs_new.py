#!/usr/bin/env python3
"""
compare_wake_runs_new.py

Thesis-ready plotting script for the *wake chapter results*, using your shared:
- thesis_colors.py
- plotting_style.py

This script provides a set of plot functions that match the “results from scratch”
structure we discussed:
1) Wake loss distributions (PDF/CDF + boxplots)
2) Wake loss vs capacity density
3) ΔCF maps (wake impact on capacity factor)
4) Optimal capacity density maps + Δ maps vs baseline
5) Spatial resolution interaction (metrics vs A_region^max)
6) System-level outcomes (cost, curtailment, transmission, etc.)
7) Robustness summary (metric table + optional radar)

It’s designed to be “data-interface agnostic”: you plug in CSV/Parquet/GeoJSON
exports from your pipeline (PyPSA results + wake regions), and it makes the figures.

Usage examples:
    python wake_results_plots.py dist --in results/wake_losses.csv --out plots/wake/dist.pdf
    python wake_results_plots.py cdf  --in results/wake_losses.csv --out plots/wake/cdf.pdf
    python wake_results_plots.py box  --in results/wake_losses.csv --out plots/wake/box.pdf
    python wake_results_plots.py loss_vs_density --in results/wake_density.csv --out plots/wake/loss_vs_density.pdf
    python wake_results_plots.py delta_cf_map --in results/delta_cf.geojson --scenario new_more --out plots/wake/delta_cf_new_more.pdf
    python wake_results_plots.py cap_map --in results/cap_density.geojson --scenario new_more --out plots/wake/cap_density_new_more.pdf
    python wake_results_plots.py cap_delta_map --in results/cap_density.geojson --scenario new_more --baseline base --out plots/wake/cap_delta_new_more.pdf
    python wake_results_plots.py resolution_lines --in results/resolution_metrics.csv --y total_offwind_cap_mw --out plots/wake/offwindcap_vs_res.pdf
    python wake_results_plots.py system_bars --in results/system_metrics.csv --y system_cost_eur_per_mwh --out plots/wake/system_cost.pdf

Data conventions (recommended):
- Scenario keys: base, standard, glaum, new_more  (aliases supported via thesis_colors.canon)
- CSV columns:
    dist/cdf/box:     scenario, wake_loss
    loss_vs_density:  scenario, density_mw_per_km2, wake_loss
    resolution_lines: scenario, split_km2, <metric>
    system_bars:      scenario, <metric>
- GeoJSON columns:
    maps: geometry + scenario + value (or separate columns per scenario)

NOTE:
- This script intentionally avoids seaborn to keep styling deterministic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import pypsa
except Exception:
    pypsa = None

# -----------------------------------------------------------------------------
# Import thesis style + colors (robust to being run from anywhere)
# -----------------------------------------------------------------------------

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
from plotting_style import thesis_plot_style, apply_spatial_resolution_axis


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _ensure_outdir(out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)

def _color_for(scenario_key: str) -> str:
    k = canon(scenario_key)
    return WAKE_MODEL_COLORS.get(k, "#4D4D4D")

def _sorted_scenarios(keys: list[str]) -> list[str]:
    """Sort scenarios using WAKE_ORDER when possible, otherwise stable."""
    c = [canon(k) for k in keys]
    order = {k: i for i, k in enumerate(WAKE_ORDER)}
    return sorted(set(c), key=lambda k: order.get(k, 9999))

def _get_split_style(split: int) -> dict:
    """Get line style and marker for a given split size."""
    split_styles = {
        1000: {'linestyle': '-', 'marker': 'o', 'alpha': 1.0},
        5000: {'linestyle': '--', 'marker': 's', 'alpha': 0.9},
        10000: {'linestyle': '-.', 'marker': '^', 'alpha': 0.85},
        50000: {'linestyle': ':', 'marker': 'D', 'alpha': 0.8},
        100000: {'linestyle': '-', 'marker': 'v', 'alpha': 0.75},
    }
    return split_styles.get(split, {'linestyle': '-', 'marker': 'o', 'alpha': 1.0})

def _format_split_label(split: int) -> str:
    """Format split size for legend."""
    if split >= 100000:
        return f"{split//1000}k"
    elif split >= 1000:
        return f"{split//1000}k"
    else:
        return f"{split}"

def _savefig(fig: plt.Figure, out: Path) -> None:
    _ensure_outdir(out)
    fig.savefig(out)
    plt.close(fig)

def _read_table(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in [".csv"]:
        return pd.read_csv(path)
    if suf in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if suf in [".feather"]:
        return pd.read_feather(path)
    raise ValueError(f"Unsupported table format: {path}")

def _read_geo(path: Path):
    if gpd is None:
        raise RuntimeError("geopandas is not available in this environment.")
    return gpd.read_file(path)

def _require_cols(df: pd.DataFrame, cols: list[str], *, where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing columns {missing}. Available: {list(df.columns)}")

def _coerce_scenario_col(df: pd.DataFrame) -> pd.DataFrame:
    if "scenario" not in df.columns:
        return df
    df = df.copy()
    df["scenario"] = df["scenario"].astype(str).map(canon)
    return df


# -----------------------------------------------------------------------------
# 1) Wake loss distributions (PDF / CDF / Box)
# -----------------------------------------------------------------------------

def plot_wake_loss_pdf(df: pd.DataFrame, *, out: Path, bins: int = 40) -> None:
    """
    Histogram density (PDF) of wake_loss per scenario.
    Expects columns: scenario, wake_loss
    Optional: split (will style differently per split)
    """
    _require_cols(df, ["scenario", "wake_loss"], where="plot_wake_loss_pdf")
    df = _coerce_scenario_col(df)

    style = thesis_plot_style()
    cm = style["cm"]
    
    has_splits = 'split' in df.columns
    splits = sorted(df['split'].unique()) if has_splits else [None]

    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())

    for s in scenarios:
        for split in splits:
            if has_splits:
                mask = (df["scenario"] == s) & (df["split"] == split)
                x = df.loc[mask, "wake_loss"].dropna().to_numpy()
                split_style = _get_split_style(split)
                lbl = f"{label(s)} ({_format_split_label(split)})"
            else:
                x = df.loc[df["scenario"] == s, "wake_loss"].dropna().to_numpy()
                split_style = {'linestyle': '-', 'alpha': 1.0}
                lbl = label(s)
            
            if len(x) == 0:
                continue
            
            ax.hist(
                x,
                bins=bins,
                density=True,
                histtype="step",
                label=lbl,
                color=_color_for(s),
                linestyle=split_style['linestyle'],
                alpha=split_style['alpha'],
            )

    ax.set_xlabel("Wake loss multiplier [-]")
    ax.set_ylabel("Density [-]")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, ncol=2, fontsize=7 if has_splits else 9)
    _savefig(fig, out)

def plot_wake_loss_cdf(df: pd.DataFrame, *, out: Path) -> None:
    """
    Empirical CDF of wake_loss per scenario.
    Expects columns: scenario, wake_loss
    Optional: split (will style differently per split)
    """
    _require_cols(df, ["scenario", "wake_loss"], where="plot_wake_loss_cdf")
    df = _coerce_scenario_col(df)

    style = thesis_plot_style()
    cm = style["cm"]
    lw = style["lw"]
    
    has_splits = 'split' in df.columns
    splits = sorted(df['split'].unique()) if has_splits else [None]

    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())

    for s in scenarios:
        for split in splits:
            if has_splits:
                mask = (df["scenario"] == s) & (df["split"] == split)
                x = df.loc[mask, "wake_loss"].dropna().to_numpy()
                split_style = _get_split_style(split)
                lbl = f"{label(s)} ({_format_split_label(split)})"
            else:
                x = df.loc[df["scenario"] == s, "wake_loss"].dropna().to_numpy()
                split_style = {'linestyle': '-', 'marker': None, 'alpha': 1.0}
                lbl = label(s)
            
            if len(x) == 0:
                continue
            
            x = np.sort(x)
            y = np.linspace(0, 1, len(x), endpoint=True)
            ax.plot(x, y, label=lbl, color=_color_for(s),
                   linestyle=split_style['linestyle'],
                   alpha=split_style['alpha'],
                   linewidth=lw)

    ax.set_xlabel("Wake loss multiplier [-]")
    ax.set_ylabel("CDF [-]")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, fontsize=7 if has_splits else 9)
    _savefig(fig, out)

def plot_wake_loss_box(df: pd.DataFrame, *, out: Path) -> None:
    """
    Boxplot of wake_loss with splits on x-axis and scenarios as groups.
    Expects columns: scenario, wake_loss
    Optional: split (will create grouped boxplots)
    """
    _require_cols(df, ["scenario", "wake_loss"], where="plot_wake_loss_box")
    df = _coerce_scenario_col(df)

    style = thesis_plot_style()
    cm = style["cm"]
    
    has_splits = 'split' in df.columns
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    if has_splits:
        splits = sorted(df['split'].unique())
        n_scenarios = len(scenarios)
        n_splits = len(splits)
        
        # Create grouped box plot with splits on x-axis
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.5 * cm), layout="constrained")
        
        positions = []
        data_all = []
        colors = []
        
        group_width = n_scenarios + 0.5
        for i, split in enumerate(splits):
            for j, scenario in enumerate(scenarios):
                mask = (df['scenario'] == scenario) & (df['split'] == split)
                data = df.loc[mask, 'wake_loss'].dropna().to_numpy()
                if len(data) > 0:
                    pos = i * group_width + j
                    positions.append(pos)
                    data_all.append(data)
                    colors.append(_color_for(scenario))
        
        bp = ax.boxplot(
            data_all,
            positions=positions,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(linewidth=1.2),
            widths=0.6,
        )
        
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
            patch.set_alpha(0.7)
        
        # Set x-axis labels for splits
        group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([_format_split_label(s) for s in splits])
        
        # Add legend for scenarios
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=_color_for(s), alpha=0.7, label=label(s)) 
                          for s in scenarios]
        ax.legend(handles=legend_elements, title="Wake model", frameon=False, 
                 loc='upper right', fontsize=7)
        
    else:
        # Single split or no split column
        data = [df.loc[df["scenario"] == s, "wake_loss"].dropna().to_numpy() for s in scenarios]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        bp = ax.boxplot(
            data,
            tick_labels=[label(s) for s in scenarios],
            patch_artist=True,
            showfliers=False,
            medianprops=dict(linewidth=1.2),
        )
        for patch, s in zip(bp["boxes"], scenarios):
            patch.set_facecolor(_color_for(s))
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        
        ax.tick_params(axis="x", rotation=15)

    ax.set_xlabel(r"Spatial Resolution ($A_{region}^{max}$) [km$^2$]")
    ax.set_ylabel("Wake loss multiplier [-]")
    _savefig(fig, out)


# -----------------------------------------------------------------------------
# 1b) Capacity Factor distributions (PDF / CDF / Box)
# -----------------------------------------------------------------------------

def plot_cf_pdf(df: pd.DataFrame, *, out: Path, bins: int = 40) -> None:
    """
    Histogram density (PDF) of available_cf per scenario.
    Expects columns: scenario, available_cf
    Optional: split (will style differently per split)
    """
    _require_cols(df, ["scenario", "available_cf"], where="plot_cf_pdf")
    df = _coerce_scenario_col(df)

    style = thesis_plot_style()
    cm = style["cm"]
    
    has_splits = 'split' in df.columns
    splits = sorted(df['split'].unique()) if has_splits else [None]

    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())

    for s in scenarios:
        for split in splits:
            if has_splits:
                mask = (df["scenario"] == s) & (df["split"] == split)
                x = df.loc[mask, "available_cf"].dropna().to_numpy()
                split_style = _get_split_style(split)
                lbl = f"{label(s)} ({_format_split_label(split)})"
            else:
                x = df.loc[df["scenario"] == s, "available_cf"].dropna().to_numpy()
                split_style = {'linestyle': '-', 'alpha': 1.0}
                lbl = label(s)
            
            if len(x) == 0:
                continue
            
            ax.hist(
                x,
                bins=bins,
                density=True,
                histtype="step",
                label=lbl,
                color=_color_for(s),
                linestyle=split_style['linestyle'],
                alpha=split_style['alpha'],
            )

    ax.set_xlabel("Capacity factor [-]")
    ax.set_ylabel("Density [-]")
    ax.set_xlim(left=0, right=1)
    ax.legend(frameon=False, ncol=2, fontsize=7 if has_splits else 9)
    _savefig(fig, out)


def plot_cf_cdf(df: pd.DataFrame, *, out: Path) -> None:
    """
    Empirical CDF of available_cf per scenario.
    Expects columns: scenario, available_cf
    Optional: split (will style differently per split)
    """
    _require_cols(df, ["scenario", "available_cf"], where="plot_cf_cdf")
    df = _coerce_scenario_col(df)

    style = thesis_plot_style()
    cm = style["cm"]
    lw = style["lw"]
    
    has_splits = 'split' in df.columns
    splits = sorted(df['split'].unique()) if has_splits else [None]

    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())

    for s in scenarios:
        for split in splits:
            if has_splits:
                mask = (df["scenario"] == s) & (df["split"] == split)
                x = df.loc[mask, "available_cf"].dropna().to_numpy()
                split_style = _get_split_style(split)
                lbl = f"{label(s)} ({_format_split_label(split)})"
            else:
                x = df.loc[df["scenario"] == s, "available_cf"].dropna().to_numpy()
                split_style = {'linestyle': '-', 'marker': None, 'alpha': 1.0}
                lbl = label(s)
            
            if len(x) == 0:
                continue
            
            x = np.sort(x)
            y = np.linspace(0, 1, len(x), endpoint=True)
            ax.plot(x, y, label=lbl, color=_color_for(s),
                   linestyle=split_style['linestyle'],
                   alpha=split_style['alpha'],
                   linewidth=lw)

    ax.set_xlabel("Capacity factor [-]")
    ax.set_ylabel("CDF [-]")
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, fontsize=7 if has_splits else 9)
    _savefig(fig, out)


def plot_cf_box(df: pd.DataFrame, *, out: Path) -> None:
    """
    Boxplot of available_cf with splits on x-axis and scenarios as groups.
    Expects columns: scenario, available_cf
    Optional: split (will create grouped boxplots)
    """
    _require_cols(df, ["scenario", "available_cf"], where="plot_cf_box")
    df = _coerce_scenario_col(df)

    style = thesis_plot_style()
    cm = style["cm"]
    
    has_splits = 'split' in df.columns
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    if has_splits:
        splits = sorted(df['split'].unique())
        n_splits = len(splits)
        n_scenarios = len(scenarios)
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        
        # Create grouped boxplots with splits on x-axis
        positions = []
        all_data = []
        colors = []
        
        group_width = n_scenarios * 1.0
        for i, split in enumerate(splits):
            for j, scenario in enumerate(scenarios):
                mask = (df['scenario'] == scenario) & (df['split'] == split)
                data = df.loc[mask, 'available_cf'].dropna().to_numpy()
                if len(data) > 0:
                    pos = i * group_width + j
                    positions.append(pos)
                    all_data.append(data)
                    colors.append(_color_for(scenario))
        
        bp = ax.boxplot(
            all_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(linewidth=1.2),
        )
        
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
            patch.set_alpha(0.7)
        
        # Set x-axis labels for splits
        group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([_format_split_label(s) for s in splits])
        
        # Add legend for scenarios
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=_color_for(s), alpha=0.7, label=label(s)) 
                          for s in scenarios]
        ax.legend(handles=legend_elements, title="Wake model", frameon=False, 
                 loc='upper right', fontsize=7)
        
    else:
        # Single split or no split column
        data = [df.loc[df["scenario"] == s, "available_cf"].dropna().to_numpy() for s in scenarios]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        bp = ax.boxplot(
            data,
            tick_labels=[label(s) for s in scenarios],
            patch_artist=True,
            showfliers=False,
            medianprops=dict(linewidth=1.2),
        )
        for patch, s in zip(bp["boxes"], scenarios):
            patch.set_facecolor(_color_for(s))
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        
        ax.tick_params(axis="x", rotation=15)

    ax.set_xlabel(r"Spatial Resolution ($A_{region}^{max}$) [km$^2$]")
    ax.set_ylabel("Capacity factor [-]")
    _savefig(fig, out)


def plot_dispatch_cf_pdf(df: pd.DataFrame, *, out: Path, bins: int = 40) -> None:
    """Histogram density (PDF) of dispatch_cf per scenario."""
    _require_cols(df, ["scenario", "dispatch_cf"], where="plot_dispatch_cf_pdf")
    df = _coerce_scenario_col(df)
    
    style = thesis_plot_style()
    cm = style["cm"]
    has_splits = 'split' in df.columns
    splits = sorted(df['split'].unique()) if has_splits else [None]
    
    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    for s in scenarios:
        for split in splits:
            if has_splits:
                mask = (df["scenario"] == s) & (df["split"] == split)
                x = df.loc[mask, "dispatch_cf"].dropna().to_numpy()
                split_style = _get_split_style(split)
                lbl = f"{label(s)} ({_format_split_label(split)})"
            else:
                x = df.loc[df["scenario"] == s, "dispatch_cf"].dropna().to_numpy()
                split_style = {'linestyle': '-', 'alpha': 1.0}
                lbl = label(s)
            
            if len(x) == 0:
                continue
            
            ax.hist(x, bins=bins, density=True, histtype="step", label=lbl,
                   color=_color_for(s), linestyle=split_style['linestyle'], 
                   alpha=split_style['alpha'])
    
    ax.set_xlabel("Dispatch capacity factor [-]")
    ax.set_ylabel("Density [-]")
    ax.set_xlim(left=0, right=1)
    ax.legend(frameon=False, ncol=2, fontsize=7 if has_splits else 9)
    _savefig(fig, out)


def plot_dispatch_cf_cdf(df: pd.DataFrame, *, out: Path) -> None:
    """Empirical CDF of dispatch_cf per scenario."""
    _require_cols(df, ["scenario", "dispatch_cf"], where="plot_dispatch_cf_cdf")
    df = _coerce_scenario_col(df)
    
    style = thesis_plot_style()
    cm = style["cm"]
    lw = style["lw"]
    has_splits = 'split' in df.columns
    splits = sorted(df['split'].unique()) if has_splits else [None]
    
    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    for s in scenarios:
        for split in splits:
            if has_splits:
                mask = (df["scenario"] == s) & (df["split"] == split)
                x = df.loc[mask, "dispatch_cf"].dropna().to_numpy()
                split_style = _get_split_style(split)
                lbl = f"{label(s)} ({_format_split_label(split)})"
            else:
                x = df.loc[df["scenario"] == s, "dispatch_cf"].dropna().to_numpy()
                split_style = {'linestyle': '-', 'marker': None, 'alpha': 1.0}
                lbl = label(s)
            
            if len(x) == 0:
                continue
            
            x = np.sort(x)
            y = np.linspace(0, 1, len(x), endpoint=True)
            ax.plot(x, y, label=lbl, color=_color_for(s),
                   linestyle=split_style['linestyle'], alpha=split_style['alpha'], 
                   linewidth=lw)
    
    ax.set_xlabel("Dispatch capacity factor [-]")
    ax.set_ylabel("CDF [-]")
    ax.set_xlim(left=0, right=1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, fontsize=7 if has_splits else 9)
    _savefig(fig, out)


def plot_dispatch_cf_box(df: pd.DataFrame, *, out: Path) -> None:
    """Boxplot of dispatch_cf with splits on x-axis and scenarios as groups."""
    _require_cols(df, ["scenario", "dispatch_cf"], where="plot_dispatch_cf_box")
    df = _coerce_scenario_col(df)
    
    style = thesis_plot_style()
    cm = style["cm"]
    has_splits = 'split' in df.columns
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    if has_splits:
        splits = sorted(df['split'].unique())
        n_splits = len(splits)
        n_scenarios = len(scenarios)
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        positions = []
        all_data = []
        colors = []
        
        group_width = n_scenarios * 1.0
        for i, split in enumerate(splits):
            for j, scenario in enumerate(scenarios):
                mask = (df['scenario'] == scenario) & (df['split'] == split)
                data = df.loc[mask, 'dispatch_cf'].dropna().to_numpy()
                if len(data) > 0:
                    pos = i * group_width + j
                    positions.append(pos)
                    all_data.append(data)
                    colors.append(_color_for(scenario))
        
        bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(linewidth=1.2))
        
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
            patch.set_alpha(0.7)
        
        group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([_format_split_label(s) for s in splits])
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=_color_for(s), alpha=0.7, label=label(s)) 
                          for s in scenarios]
        ax.legend(handles=legend_elements, title="Wake model", frameon=False, 
                 loc='upper right', fontsize=7)
    else:
        data = [df.loc[df["scenario"] == s, "dispatch_cf"].dropna().to_numpy() for s in scenarios]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        bp = ax.boxplot(data, tick_labels=[label(s) for s in scenarios], patch_artist=True,
                       showfliers=False, medianprops=dict(linewidth=1.2))
        for patch, s in zip(bp["boxes"], scenarios):
            patch.set_facecolor(_color_for(s))
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        
        ax.tick_params(axis="x", rotation=15)
    
    ax.set_xlabel(r"Spatial Resolution ($A_{region}^{max}$) [km$^2$]")
    ax.set_ylabel("Dispatch capacity factor [-]")
    _savefig(fig, out)


def plot_curtailment_cf_pdf(df: pd.DataFrame, *, out: Path, bins: int = 40) -> None:
    """Histogram density (PDF) of curtailment_cf per scenario."""
    _require_cols(df, ["scenario", "curtailment_cf"], where="plot_curtailment_cf_pdf")
    df = _coerce_scenario_col(df)
    
    style = thesis_plot_style()
    cm = style["cm"]
    has_splits = 'split' in df.columns
    splits = sorted(df['split'].unique()) if has_splits else [None]
    
    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    for s in scenarios:
        for split in splits:
            if has_splits:
                mask = (df["scenario"] == s) & (df["split"] == split)
                x = df.loc[mask, "curtailment_cf"].dropna().to_numpy()
                split_style = _get_split_style(split)
                lbl = f"{label(s)} ({_format_split_label(split)})"
            else:
                x = df.loc[df["scenario"] == s, "curtailment_cf"].dropna().to_numpy()
                split_style = {'linestyle': '-', 'alpha': 1.0}
                lbl = label(s)
            
            if len(x) == 0:
                continue
            
            ax.hist(x, bins=bins, density=True, histtype="step", label=lbl,
                   color=_color_for(s), linestyle=split_style['linestyle'], 
                   alpha=split_style['alpha'])
    
    ax.set_xlabel("Curtailment capacity factor [-]")
    ax.set_ylabel("Density [-]")
    ax.set_xlim(left=0)
    ax.legend(frameon=False, ncol=2, fontsize=7 if has_splits else 9)
    _savefig(fig, out)


def plot_curtailment_cf_cdf(df: pd.DataFrame, *, out: Path) -> None:
    """Empirical CDF of curtailment_cf per scenario."""
    _require_cols(df, ["scenario", "curtailment_cf"], where="plot_curtailment_cf_cdf")
    df = _coerce_scenario_col(df)
    
    style = thesis_plot_style()
    cm = style["cm"]
    lw = style["lw"]
    has_splits = 'split' in df.columns
    splits = sorted(df['split'].unique()) if has_splits else [None]
    
    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    for s in scenarios:
        for split in splits:
            if has_splits:
                mask = (df["scenario"] == s) & (df["split"] == split)
                x = df.loc[mask, "curtailment_cf"].dropna().to_numpy()
                split_style = _get_split_style(split)
                lbl = f"{label(s)} ({_format_split_label(split)})"
            else:
                x = df.loc[df["scenario"] == s, "curtailment_cf"].dropna().to_numpy()
                split_style = {'linestyle': '-', 'marker': None, 'alpha': 1.0}
                lbl = label(s)
            
            if len(x) == 0:
                continue
            
            x = np.sort(x)
            y = np.linspace(0, 1, len(x), endpoint=True)
            ax.plot(x, y, label=lbl, color=_color_for(s),
                   linestyle=split_style['linestyle'], alpha=split_style['alpha'], 
                   linewidth=lw)
    
    ax.set_xlabel("Curtailment capacity factor [-]")
    ax.set_ylabel("CDF [-]")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, ncol=2, fontsize=7 if has_splits else 9)
    _savefig(fig, out)


def plot_curtailment_cf_box(df: pd.DataFrame, *, out: Path) -> None:
    """Boxplot of curtailment_cf with splits on x-axis and scenarios as groups."""
    _require_cols(df, ["scenario", "curtailment_cf"], where="plot_curtailment_cf_box")
    df = _coerce_scenario_col(df)
    
    style = thesis_plot_style()
    cm = style["cm"]
    has_splits = 'split' in df.columns
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    if has_splits:
        splits = sorted(df['split'].unique())
        n_splits = len(splits)
        n_scenarios = len(scenarios)
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        positions = []
        all_data = []
        colors = []
        
        group_width = n_scenarios * 1.0
        for i, split in enumerate(splits):
            for j, scenario in enumerate(scenarios):
                mask = (df['scenario'] == scenario) & (df['split'] == split)
                data = df.loc[mask, 'curtailment_cf'].dropna().to_numpy()
                if len(data) > 0:
                    pos = i * group_width + j
                    positions.append(pos)
                    all_data.append(data)
                    colors.append(_color_for(scenario))
        
        bp = ax.boxplot(all_data, positions=positions, widths=0.6, patch_artist=True,
                       showfliers=False, medianprops=dict(linewidth=1.2))
        
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
            patch.set_alpha(0.7)
        
        group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([_format_split_label(s) for s in splits])
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=_color_for(s), alpha=0.7, label=label(s)) 
                          for s in scenarios]
        ax.legend(handles=legend_elements, title="Wake model", frameon=False, 
                 loc='upper right', fontsize=7)
    else:
        data = [df.loc[df["scenario"] == s, "curtailment_cf"].dropna().to_numpy() for s in scenarios]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        bp = ax.boxplot(data, tick_labels=[label(s) for s in scenarios], patch_artist=True,
                       showfliers=False, medianprops=dict(linewidth=1.2))
        for patch, s in zip(bp["boxes"], scenarios):
            patch.set_facecolor(_color_for(s))
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)
        
        ax.tick_params(axis="x", rotation=15)
    
    ax.set_xlabel(r"Spatial Resolution ($A_{region}^{max}$) [km$^2$]")
    ax.set_ylabel("Curtailment capacity factor [-]")
    _savefig(fig, out)


# -----------------------------------------------------------------------------
# 1c) Capacity density maps from networks
# -----------------------------------------------------------------------------

def _default_gen_region_parser(gen_index: pd.Index) -> pd.Series:
    """
    Default region parser matching PyPSA generator naming:
      generators.index like "NO4 0_00022 offwind-ac ..." -> region "NO4 0"
    i.e. take first two whitespace-separated tokens.
    """
    s = gen_index.to_series().astype(str)
    return s.str.split().str[:2].str.join(" ")


def build_region_capacity_density_geodf(
    n,
    *,
    split: int,
    area: str,
    regions_dir: Path = Path("wake_extra"),
    carrier_filter: str = "offwind",
    regions_name_col: str = "name",
    regions_region_col: str = "region",
    cap_field: str = "p_nom_opt",
    region_parser=None,
    target_crs_for_area: int = 3035,
):
    """
    Build a GeoDataFrame with per-region capacity and capacity density.

    Returns a GeoDataFrame with at least:
        - geometry
        - area_km2
        - <cap_field> (summed by region)
        - density_mw_per_km2

    Notes:
    - Density is computed as: (capacity_MW)/area_km2.
      PyPSA stores p_nom_opt/p_nom_max in MW by default, so we use MW/km².
    """
    if pypsa is None:
        raise RuntimeError("pypsa not available, cannot calculate capacity density")
    if gpd is None:
        raise RuntimeError("geopandas not available, cannot create maps")
    
    if region_parser is None:
        region_parser = _default_gen_region_parser

    regions_dir = Path(regions_dir)

    gens = n.generators.copy()

    # Filter to carrier (offshore wind)
    if "carrier" in gens.columns:
        gens = gens[gens["carrier"].str.startswith(carrier_filter)]

    if gens.empty:
        raise ValueError(f"No generators matched carrier='{carrier_filter}'.")

    if cap_field not in gens.columns:
        raise KeyError(f"Generator table does not contain '{cap_field}'. Available: {list(gens.columns)}")

    # Assign region
    gens[regions_region_col] = region_parser(gens.index)

    # Aggregate capacity per region
    cap_by_region = (
        gens.groupby(regions_region_col, dropna=False)[cap_field]
        .sum()
        .rename(cap_field)
        .to_frame()
    )

    # Load regions geometry
    regions_path = regions_dir / area / f"regions_offshore_s{split}.geojson"
    
    if not regions_path.exists():
        raise FileNotFoundError(f"Region file not found: {regions_path}")

    regions = gpd.read_file(regions_path)

    if regions_name_col not in regions.columns:
        raise KeyError(
            f"Regions file {regions_path} has no '{regions_name_col}' column. "
            f"Available: {list(regions.columns)}"
        )

    regions = regions.copy()
    regions[regions_region_col] = regions[regions_name_col].astype(str)

    # Join capacity into regions
    geodf = regions.merge(cap_by_region, on=regions_region_col, how="left")
    
    # Fill NaN capacity values with 0 (regions with no generators)
    geodf[cap_field] = geodf[cap_field].fillna(0.0)

    # Compute area_km2 (robustly)
    if "area_km2" not in geodf.columns:
        try:
            geodf["area_km2"] = geodf.geometry.to_crs(target_crs_for_area).area / 1e6
        except Exception as e:
            raise RuntimeError(
                "Failed to compute area_km2. Ensure geometries are valid and have a CRS."
            ) from e

    # Compute density: MW/km² (PyPSA p_nom_* is in MW)
    density_col = "density_mw_per_km2"
    geodf[density_col] = geodf[cap_field] / geodf["area_km2"]

    return geodf, cap_field, density_col


def plot_capacity_density_maps(
    networks_by_scenario: dict,
    *,
    split: int,
    area: str,
    out: Path,
    baseline_scenario: str = "base",
    regions_dir: Path = Path("wake_extra"),
) -> None:
    """
    Plot optimal capacity density difference (Δ) per region across wake models as spatial maps.
    
    Creates a multi-panel figure showing capacity density delta (MW/km²) relative to baseline,
    one panel per wake model (excluding baseline itself).
    Uses diverging colormap (RdBu_r) centered at 0 to show increases/decreases.
    
    Args:
        networks_by_scenario: Dict mapping scenario name to PyPSA network
        split: Split size (e.g., 1000, 10000)
        area: Area name (e.g., 'northsea')
        out: Output file path
        baseline_scenario: Baseline scenario name (default: 'base')
        regions_dir: Directory containing region GeoJSON files
    """
    if pypsa is None:
        print("[WARN] pypsa not available, skipping capacity density maps")
        return
    if gpd is None:
        print("[WARN] geopandas not available, skipping capacity density maps")
        return
    
    if not networks_by_scenario:
        return
    
    # Check baseline exists
    if baseline_scenario not in networks_by_scenario:
        print(f"[WARN] Baseline scenario '{baseline_scenario}' not found in networks. Skipping density map.")
        return
    
    style = thesis_plot_style()
    cm = style["cm"]
    
    # Calculate capacity density for each scenario
    geodfs = {}
    for scenario, network in networks_by_scenario.items():
        try:
            geodf, cap_col, density_col = build_region_capacity_density_geodf(
                network,
                split=split,
                area=area,
                regions_dir=regions_dir,
                carrier_filter="offwind",
                cap_field="p_nom_opt",
            )
            geodfs[scenario] = geodf
        except Exception as e:
            print(f"[WARN] Failed to calculate density for {scenario}: {e}")
            continue
    
    if baseline_scenario not in geodfs:
        print(f"[WARN] Could not calculate baseline density. Skipping map.")
        return
    
    # Get scenarios to plot (exclude baseline)
    scenarios = _sorted_scenarios([s for s in geodfs.keys() if s != baseline_scenario])
    if not scenarios:
        return
    
    # Get baseline density
    baseline_gdf = geodfs[baseline_scenario]
    baseline_density = baseline_gdf.set_index("region")["density_mw_per_km2"]
    
    # Compute deltas for each scenario
    delta_gdfs = {}
    all_delta_values = []
    
    for scenario in scenarios:
        gdf = geodfs[scenario].copy()
        gdf_density = gdf.set_index("region")["density_mw_per_km2"]
        
        # Align and compute delta
        delta = gdf_density - baseline_density
        gdf["delta_density"] = gdf["region"].map(delta)
        
        delta_gdfs[scenario] = gdf
        
        # Collect values for colorbar limits
        vals = gdf["delta_density"].values
        all_delta_values.extend(vals[np.isfinite(vals)])
    
    if not delta_gdfs or not all_delta_values:
        return
    
    # Symmetric colorbar limits around 0
    vlim = float(np.max(np.abs(all_delta_values)))
    
    # Create multi-panel figure
    n_panels = len(delta_gdfs)
    fig, axes = plt.subplots(
        1, n_panels, 
        figsize=(16.4 * cm, 4.5 * cm), 
        sharex=True, 
        sharey=True, 
        layout="constrained"
    )
    
    if n_panels == 1:
        axes = [axes]
    
    for i, (ax, scenario) in enumerate(zip(axes, delta_gdfs.keys())):
        gdf = delta_gdfs[scenario]
        
        # Only show colorbar on last panel
        show_legend = (i == n_panels - 1)
        
        gdf.plot(
            column="delta_density",
            ax=ax,
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            legend=show_legend,
            linewidth=0.1,
            edgecolor='black',
            legend_kwds={
                "label": r"$\Delta$ density [MW/km$^2$]",
                "orientation": "vertical",
                "pad": 0.02,
                "shrink": 0.8,
            } if show_legend else None,
        )
        
        # Title with nice label
        ax.set_title(label(scenario), fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    
    _savefig(fig, out)


# -----------------------------------------------------------------------------
# 2) Wake loss vs capacity density
# -----------------------------------------------------------------------------

def plot_wake_loss_vs_density(df: pd.DataFrame, *, out: Path) -> None:
    """
    Scatter + binned mean curve: wake_loss vs density.
    Expects columns: scenario, density_mw_per_km2, wake_loss
    """
    _require_cols(df, ["scenario", "density_mw_per_km2", "wake_loss"], where="plot_wake_loss_vs_density")
    df = _coerce_scenario_col(df)

    style = thesis_plot_style()
    cm = style["cm"]
    lw = style["lw"]
    ms = style["ms"]

    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.5 * cm), layout="constrained")

    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    for s in scenarios:
        d = df[df["scenario"] == s].dropna(subset=["density_mw_per_km2", "wake_loss"])
        if d.empty:
            continue

        x = d["density_mw_per_km2"].to_numpy()
        y = d["wake_loss"].to_numpy()

        ax.plot(x, y, linestyle="none", marker="o", markersize=ms, alpha=0.25, color=_color_for(s))

        # binned mean
        bins = np.quantile(x, np.linspace(0, 1, 12))
        bins = np.unique(bins)
        if len(bins) >= 4:
            idx = np.digitize(x, bins[1:-1], right=True)
            xb, yb = [], []
            for b in np.unique(idx):
                m = idx == b
                xb.append(np.nanmean(x[m]))
                yb.append(np.nanmean(y[m]))
            ax.plot(xb, yb, linewidth=lw, label=label(s), color=_color_for(s))

    ax.set_xlabel(r"Installed density [MW km$^{-2}$]")
    ax.set_ylabel("Wake loss multiplier [-]")
    ax.legend(frameon=False, ncol=2)
    _savefig(fig, out)


# -----------------------------------------------------------------------------
# 2b) Capacity density maps from PyPSA networks
# -----------------------------------------------------------------------------

def _default_gen_region_parser(gen_index: pd.Index) -> pd.Series:
    """
    Default region parser matching PyPSA generator naming:
      generators.index like "NO4 0_00022 offwind-ac ..." -> region "NO4 0"
    i.e. take first two whitespace-separated tokens.
    """
    s = gen_index.to_series().astype(str)
    return s.str.split().str[:2].str.join(" ")


def build_region_capacity_density_geodf(
    n,
    *,
    split: int,
    area: str,
    regions_dir: Path = Path("wake_extra"),
    carrier_filter: str = "offwind",
    regions_name_col: str = "name",
    regions_region_col: str = "region",
    cap_field: str = "p_nom_opt",
    region_parser=None,
    target_crs_for_area: int = 3035,
):
    """
    Build a GeoDataFrame with per-region capacity and capacity density.

    Returns tuple: (geodf, cap_col, density_col) where geodf has:
        - geometry
        - area_km2
        - <cap_field> (summed by region)
        - density_mw_per_km2

    Notes:
    - Density is computed as: (capacity_MW)/area_km2.
      PyPSA stores p_nom_opt/p_nom_max in MW by default, so we use MW/km².
    """
    if pypsa is None:
        raise RuntimeError("pypsa not available, cannot calculate capacity density")
    if gpd is None:
        raise RuntimeError("geopandas not available, cannot create maps")
    
    if region_parser is None:
        region_parser = _default_gen_region_parser

    regions_dir = Path(regions_dir)

    gens = n.generators.copy()

    # Filter to carrier (offshore wind)
    if "carrier" in gens.columns:
        gens = gens[gens["carrier"].str.startswith(carrier_filter)]

    if gens.empty:
        raise ValueError(f"No generators matched carrier='{carrier_filter}'.")

    if cap_field not in gens.columns:
        raise KeyError(f"Generator table does not contain '{cap_field}'. Available: {list(gens.columns)}")

    # Assign region
    gens[regions_region_col] = region_parser(gens.index)

    # Aggregate capacity per region
    cap_by_region = (
        gens.groupby(regions_region_col, dropna=False)[cap_field]
        .sum()
        .rename(cap_field)
        .to_frame()
    )

    # Load regions geometry
    regions_path = regions_dir / area / f"regions_offshore_s{split}.geojson"
    
    if not regions_path.exists():
        raise FileNotFoundError(f"Region file not found: {regions_path}")

    regions = gpd.read_file(regions_path)

    if regions_name_col not in regions.columns:
        raise KeyError(
            f"Regions file {regions_path} has no '{regions_name_col}' column. "
            f"Available: {list(regions.columns)}"
        )

    regions = regions.copy()
    regions[regions_region_col] = regions[regions_name_col].astype(str)

    # Join capacity into regions
    geodf = regions.merge(cap_by_region, on=regions_region_col, how="left")
    
    # Fill NaN capacity values with 0 (regions with no generators)
    geodf[cap_field] = geodf[cap_field].fillna(0.0)

    # Compute area_km2 (robustly)
    if "area_km2" not in geodf.columns:
        try:
            geodf["area_km2"] = geodf.geometry.to_crs(target_crs_for_area).area / 1e6
        except Exception as e:
            raise RuntimeError(
                "Failed to compute area_km2. Ensure geometries are valid and have a CRS."
            ) from e

    # Compute density: MW/km² (PyPSA p_nom_* is in MW)
    density_col = "density_mw_per_km2"
    geodf[density_col] = geodf[cap_field] / geodf["area_km2"]

    return geodf, cap_field, density_col


def plot_capacity_density_maps(
    networks_by_scenario: dict,
    *,
    split: int,
    area: str,
    out: Path,
    baseline_scenario: str = "base",
    regions_dir: Path = Path("wake_extra"),
) -> None:
    """
    Plot optimal capacity density difference (Δ) per region across wake models as spatial maps.
    
    Creates a multi-panel figure showing capacity density delta (MW/km²) relative to baseline,
    one panel per wake model (excluding baseline itself).
    Uses diverging colormap (RdBu_r) centered at 0 to show increases/decreases.
    
    Args:
        networks_by_scenario: Dict mapping scenario name to PyPSA network
        split: Split size (e.g., 1000, 10000)
        area: Area name (e.g., 'northsea')
        out: Output file path
        baseline_scenario: Baseline scenario name (default: 'base')
        regions_dir: Directory containing region GeoJSON files
    """
    if pypsa is None:
        print("[WARN] pypsa not available, skipping capacity density maps")
        return
    if gpd is None:
        print("[WARN] geopandas not available, skipping capacity density maps")
        return
    
    if not networks_by_scenario:
        return
    
    # Check baseline exists
    if baseline_scenario not in networks_by_scenario:
        print(f"[WARN] Baseline scenario '{baseline_scenario}' not found in networks. Skipping density map.")
        return
    
    style = thesis_plot_style()
    cm = style["cm"]
    
    # Calculate capacity density for each scenario
    geodfs = {}
    for scenario, network in networks_by_scenario.items():
        try:
            geodf, cap_col, density_col = build_region_capacity_density_geodf(
                network,
                split=split,
                area=area,
                regions_dir=regions_dir,
                carrier_filter="offwind",
                cap_field="p_nom_opt",
            )
            geodfs[scenario] = geodf
        except Exception as e:
            print(f"[WARN] Failed to calculate density for {scenario}: {e}")
            continue
    
    if baseline_scenario not in geodfs:
        print(f"[WARN] Could not calculate baseline density. Skipping map.")
        return
    
    # Get scenarios to plot (exclude baseline)
    scenarios = _sorted_scenarios([s for s in geodfs.keys() if s != baseline_scenario])
    if not scenarios:
        return
    
    # Get baseline density
    baseline_gdf = geodfs[baseline_scenario]
    baseline_density = baseline_gdf.set_index("region")["density_mw_per_km2"]
    
    # Compute deltas for each scenario
    delta_gdfs = {}
    all_delta_values = []
    
    for scenario in scenarios:
        gdf = geodfs[scenario].copy()
        gdf_density = gdf.set_index("region")["density_mw_per_km2"]
        
        # Align and compute delta
        delta = gdf_density - baseline_density
        gdf["delta_density"] = gdf["region"].map(delta)
        
        delta_gdfs[scenario] = gdf
        
        # Collect values for colorbar limits
        vals = gdf["delta_density"].values
        all_delta_values.extend(vals[np.isfinite(vals)])
    
    if not delta_gdfs or not all_delta_values:
        return
    
    # Symmetric colorbar limits around 0
    vlim = float(np.max(np.abs(all_delta_values)))
    
    # Create multi-panel figure
    n_panels = len(delta_gdfs)
    fig, axes = plt.subplots(
        1, n_panels, 
        figsize=(16.4 * cm, 4.5 * cm), 
        sharex=True, 
        sharey=True, 
        layout="constrained"
    )
    
    if n_panels == 1:
        axes = [axes]
    
    for i, (ax, scenario) in enumerate(zip(axes, delta_gdfs.keys())):
        gdf = delta_gdfs[scenario]
        
        # Only show colorbar on last panel
        show_legend = (i == n_panels - 1)
        
        gdf.plot(
            column="delta_density",
            ax=ax,
            cmap="RdBu_r",
            vmin=-vlim,
            vmax=vlim,
            legend=show_legend,
            linewidth=0.1,
            edgecolor='black',
            legend_kwds={
                "label": r"$\Delta$ density [MW/km$^2$]",
                "orientation": "vertical",
                "pad": 0.02,
                "shrink": 0.8,
            } if show_legend else None,
        )
        
        # Title with nice label
        ax.set_title(label(scenario), fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    
    _savefig(fig, out)


# -----------------------------------------------------------------------------
# 3–4) Maps: ΔCF and capacity density
# -----------------------------------------------------------------------------

def _plot_choropleth(
    gdf,
    *,
    value_col: str,
    out: Path,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if gpd is None:
        raise RuntimeError("geopandas is required for map plots.")
    style = thesis_plot_style()
    cm = style["cm"]

    fig, ax = plt.subplots(figsize=(16.4 * cm, 8.0 * cm), layout="constrained")
    gdf.plot(
        ax=ax,
        column=value_col,
        linewidth=0.2,
        edgecolor="black",
        legend=True,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    _savefig(fig, out)

def plot_delta_cf_map(gdf, *, scenario: str, out: Path, value_col: str = "delta_cf") -> None:
    """
    ΔCF map for one scenario. Expects gdf has columns: scenario, value_col
    """
    scenario = canon(scenario)
    if "scenario" in gdf.columns:
        gg = gdf[gdf["scenario"].astype(str).map(canon) == scenario].copy()
    else:
        gg = gdf.copy()
    if value_col not in gg.columns:
        raise ValueError(f"delta_cf_map: missing '{value_col}' in GeoDataFrame columns.")

    _plot_choropleth(
        gg,
        value_col=value_col,
        out=out,
        title=f"Δ Capacity factor (wake impact): {label(scenario)}",
    )

def plot_capacity_density_map(gdf, *, scenario: str, out: Path, value_col: str = "cap_mw_per_km2") -> None:
    """
    Capacity density map for one scenario. Expects columns: scenario, value_col
    """
    scenario = canon(scenario)
    if "scenario" in gdf.columns:
        gg = gdf[gdf["scenario"].astype(str).map(canon) == scenario].copy()
    else:
        gg = gdf.copy()
    if value_col not in gg.columns:
        raise ValueError(f"cap_map: missing '{value_col}' in GeoDataFrame columns.")

    _plot_choropleth(
        gg,
        value_col=value_col,
        out=out,
        title=f"Optimal offshore capacity density: {label(scenario)}",
    )

def plot_capacity_density_delta_map(
    gdf,
    *,
    scenario: str,
    baseline: str = "base",
    out: Path,
    value_col: str = "cap_mw_per_km2",
    id_col: str = "region_id",
) -> None:
    """
    Δ capacity density vs baseline map.
    Expects: rows for multiple scenarios, and a region identifier to align.
    Required columns: scenario, id_col, value_col
    """
    scenario = canon(scenario)
    baseline = canon(baseline)

    if "scenario" not in gdf.columns:
        raise ValueError("cap_delta_map: GeoDataFrame must include a 'scenario' column.")
    for c in [id_col, value_col]:
        if c not in gdf.columns:
            raise ValueError(f"cap_delta_map: missing '{c}'.")

    gg = gdf.copy()
    gg["scenario"] = gg["scenario"].astype(str).map(canon)

    base = gg[gg["scenario"] == baseline][[id_col, value_col]].rename(columns={value_col: "base_val"})
    sc = gg[gg["scenario"] == scenario].merge(base, on=id_col, how="left")
    sc["delta"] = sc[value_col] - sc["base_val"]

    _plot_choropleth(
        sc,
        value_col="delta",
        out=out,
        title=f"Δ offshore capacity density vs {label(baseline)}: {label(scenario)}",
    )


# -----------------------------------------------------------------------------
# 5) Spatial resolution interactions (lines)
# -----------------------------------------------------------------------------

def plot_resolution_lines(df: pd.DataFrame, *, y: str, out: Path) -> None:
    """
    Metric vs spatial resolution (split_km2), one line per scenario.
    Expects columns: scenario, split_km2, y
    Applies your standard coarse->fine log axis.
    """
    _require_cols(df, ["scenario", "split_km2", y], where="plot_resolution_lines")
    df = _coerce_scenario_col(df)

    style = thesis_plot_style()
    cm = style["cm"]
    lw = style["lw"]
    ms = style["ms"]

    fig, ax = plt.subplots(figsize=(16.4 * cm, 6.3 * cm), layout="constrained")

    for s in _sorted_scenarios(df["scenario"].unique().tolist()):
        d = df[df["scenario"] == s].sort_values("split_km2")
        ax.plot(
            d["split_km2"],
            d[y],
            marker="o",
            linewidth=lw,
            markersize=ms,
            label=label(s),
            color=_color_for(s),
        )

    apply_spatial_resolution_axis(ax)
    ax.set_ylabel(y.replace("_", r"\_"))
    ax.legend(frameon=False, ncol=2)
    _savefig(fig, out)


# -----------------------------------------------------------------------------
# 6) System-level outcomes (bars and combined)
# -----------------------------------------------------------------------------

def plot_transmission_expansion(df: pd.DataFrame, *, out: Path) -> None:
    """
    Combined plot showing AC and DC line volume expansion across scenarios and splits.
    Expects columns: scenario, split, line_volume_ac, line_volume_dc
    """
    _require_cols(df, ["scenario", "line_volume_ac", "line_volume_dc"], 
                  where="plot_transmission_expansion")
    df = _coerce_scenario_col(df)
    
    has_splits = 'split' in df.columns
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    style = thesis_plot_style()
    cm = style["cm"]
    
    if has_splits:
        splits = sorted(df['split'].unique())
        n_scenarios = len(scenarios)
        n_splits = len(splits)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.4 * cm, 6.5 * cm), layout="constrained")
        
        bar_width = 0.8 / n_splits
        group_positions = np.arange(n_scenarios)
        
        # AC line volume
        for j, split in enumerate(splits):
            values = []
            for s in scenarios:
                mask = (df["scenario"] == s) & (df["split"] == split)
                val = df.loc[mask, "line_volume_ac"].mean() if mask.any() else 0
                values.append(val)
            
            positions = group_positions + j * bar_width - (n_splits - 1) * bar_width / 2
            colors_list = [_color_for(s) for s in scenarios]
            
            ax1.bar(positions, values, bar_width, 
                   color=colors_list,
                   edgecolor="black", 
                   linewidth=0.6,
                   alpha=0.8 - j * 0.15,
                   label=_format_split_label(split))
        
        ax1.set_xticks(group_positions)
        ax1.set_xticklabels([label(s) for s in scenarios], rotation=15, ha="right")
        ax1.set_ylabel("AC line volume [MV·km]")
        ax1.legend(title=r"$A_{region}^{max}$", frameon=False, fontsize=7, loc='upper left')
        
        # DC line volume
        for j, split in enumerate(splits):
            values = []
            for s in scenarios:
                mask = (df["scenario"] == s) & (df["split"] == split)
                val = df.loc[mask, "line_volume_dc"].mean() if mask.any() else 0
                values.append(val)
            
            positions = group_positions + j * bar_width - (n_splits - 1) * bar_width / 2
            colors_list = [_color_for(s) for s in scenarios]
            
            ax2.bar(positions, values, bar_width, 
                   color=colors_list,
                   edgecolor="black", 
                   linewidth=0.6,
                   alpha=0.8 - j * 0.15,
                   label=_format_split_label(split))
        
        ax2.set_xticks(group_positions)
        ax2.set_xticklabels([label(s) for s in scenarios], rotation=15, ha="right")
        ax2.set_ylabel("DC line volume [MW·km]")
        
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        
        x = np.arange(len(scenarios))
        colors = [_color_for(s) for s in scenarios]
        
        # AC line volume
        ac_values = [df.loc[df["scenario"] == s, "line_volume_ac"].mean() for s in scenarios]
        ax1.bar(x, ac_values, color=colors, edgecolor="black", linewidth=0.6)
        ax1.set_xticks(x, [label(s) for s in scenarios], rotation=15, ha="right")
        ax1.set_ylabel("AC line volume [MV·km]")
        
        # DC line volume
        dc_values = [df.loc[df["scenario"] == s, "line_volume_dc"].mean() for s in scenarios]
        ax2.bar(x, dc_values, color=colors, edgecolor="black", linewidth=0.6)
        ax2.set_xticks(x, [label(s) for s in scenarios], rotation=15, ha="right")
        ax2.set_ylabel("DC line volume [MW·km]")
    
    _savefig(fig, out)


def plot_system_cost_comparison(df: pd.DataFrame, *, out: Path) -> None:
    """
    Bar chart showing total system cost across scenarios and splits.
    Expects columns: scenario, split, total_cost_eur
    """
    _require_cols(df, ["scenario", "total_cost_eur"], where="plot_system_cost_comparison")
    df = _coerce_scenario_col(df)
    
    has_splits = 'split' in df.columns
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    style = thesis_plot_style()
    cm = style["cm"]
    
    if has_splits:
        splits = sorted(df['split'].unique())
        n_scenarios = len(scenarios)
        n_splits = len(splits)
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.5 * cm), layout="constrained")
        
        bar_width = 0.8 / n_splits
        group_positions = np.arange(n_scenarios)
        
        for j, split in enumerate(splits):
            values = []
            for s in scenarios:
                mask = (df["scenario"] == s) & (df["split"] == split)
                val = df.loc[mask, "total_cost_eur"].mean() if mask.any() else 0
                values.append(val / 1e9)  # Convert to billion EUR
            
            positions = group_positions + j * bar_width - (n_splits - 1) * bar_width / 2
            colors_list = [_color_for(s) for s in scenarios]
            
            ax.bar(positions, values, bar_width, 
                  color=colors_list,
                  edgecolor="black", 
                  linewidth=0.6,
                  alpha=0.8 - j * 0.15,
                  label=_format_split_label(split))
        
        ax.set_xticks(group_positions)
        ax.set_xticklabels([label(s) for s in scenarios], rotation=15, ha="right")
        ax.set_ylabel("Total system cost [B€]")
        ax.legend(title=r"$A_{region}^{max}$", frameon=False, fontsize=7, loc='upper left')
        
    else:
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        
        x = np.arange(len(scenarios))
        colors = [_color_for(s) for s in scenarios]
        values = [df.loc[df["scenario"] == s, "total_cost_eur"].mean() / 1e9 for s in scenarios]
        
        ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x, [label(s) for s in scenarios], rotation=15, ha="right")
        ax.set_ylabel("Total system cost [B€]")
    
    _savefig(fig, out)


def plot_curtailment_analysis(df: pd.DataFrame, *, out: Path) -> None:
    """
    Combined plot showing curtailment and system cost trade-off.
    Expects columns: scenario, split, curtailment_twh, total_cost_eur
    """
    _require_cols(df, ["scenario", "curtailment_twh", "total_cost_eur"], 
                  where="plot_curtailment_analysis")
    df = _coerce_scenario_col(df)
    
    has_splits = 'split' in df.columns
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    style = thesis_plot_style()
    cm = style["cm"]
    
    if has_splits:
        splits = sorted(df['split'].unique())
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.5 * cm), layout="constrained")
        
        # Scatter plot with splits as marker types
        markers = ['o', 's', '^']
        for j, split in enumerate(splits):
            for s in scenarios:
                mask = (df["scenario"] == s) & (df["split"] == split)
                if mask.any():
                    cost = df.loc[mask, "total_cost_eur"].mean() / 1e9
                    curtail = df.loc[mask, "curtailment_twh"].mean()
                    ax.scatter(curtail, cost, 
                             color=_color_for(s),
                             marker=markers[j % len(markers)],
                             s=100,
                             edgecolor='black',
                             linewidth=0.8,
                             label=f"{label(s)} ({_format_split_label(split)})" if j == 0 else "",
                             alpha=0.8)
        
        ax.set_xlabel("Curtailment [TWh]")
        ax.set_ylabel("Total system cost [B€]")
        
        # Create legend handles
        from matplotlib.lines import Line2D
        legend_elements = []
        for s in scenarios:
            legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=_color_for(s), markersize=8,
                                        markeredgecolor='black', markeredgewidth=0.8,
                                        label=label(s)))
        for j, split in enumerate(splits):
            legend_elements.append(Line2D([0], [0], marker=markers[j % len(markers)], 
                                        color='w', markerfacecolor='gray', markersize=8,
                                        markeredgecolor='black', markeredgewidth=0.8,
                                        label=_format_split_label(split)))
        
        ax.legend(handles=legend_elements, frameon=False, fontsize=6, ncol=2)
        
    else:
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        
        for s in scenarios:
            mask = df["scenario"] == s
            if mask.any():
                cost = df.loc[mask, "total_cost_eur"].mean() / 1e9
                curtail = df.loc[mask, "curtailment_twh"].mean()
                ax.scatter(curtail, cost, 
                         color=_color_for(s),
                         marker='o',
                         s=100,
                         edgecolor='black',
                         linewidth=0.8,
                         label=label(s))
        
        ax.set_xlabel("Curtailment [TWh]")
        ax.set_ylabel("Total system cost [B€]")
        ax.legend(frameon=False)
    
    _savefig(fig, out)


def plot_system_bars(df: pd.DataFrame, *, y: str, out: Path) -> None:
    """
    Bar chart of a system metric by scenario.
    Expects columns: scenario, y
    Optional: split (will create grouped bars per split)
    """
    _require_cols(df, ["scenario", y], where="plot_system_bars")
    df = _coerce_scenario_col(df)
    
    has_splits = 'split' in df.columns
    scenarios = _sorted_scenarios(df["scenario"].unique().tolist())
    
    style = thesis_plot_style()
    cm = style["cm"]
    
    if has_splits:
        splits = sorted(df['split'].unique())
        n_scenarios = len(scenarios)
        n_splits = len(splits)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.5 * cm), layout="constrained")
        
        bar_width = 0.8 / n_splits
        group_positions = np.arange(n_scenarios)
        
        for j, split in enumerate(splits):
            values = []
            for s in scenarios:
                mask = (df["scenario"] == s) & (df["split"] == split)
                val = df.loc[mask, y].mean() if mask.any() else 0
                values.append(val)
            
            positions = group_positions + j * bar_width - (n_splits - 1) * bar_width / 2
            colors_list = [_color_for(s) for s in scenarios]
            
            ax.bar(positions, values, bar_width, 
                  color=colors_list,
                  edgecolor="black", 
                  linewidth=0.6,
                  alpha=0.8 - j * 0.15,
                  label=_format_split_label(split))
        
        ax.set_xticks(group_positions)
        ax.set_xticklabels([label(s) for s in scenarios], rotation=15, ha="right")
        ax.legend(title="Split", frameon=False, fontsize=7)
        
    else:
        # Single split or no split column - aggregate if needed
        if 'split' in df.columns:
            df_agg = df.groupby('scenario', as_index=False)[y].mean()
        else:
            df_agg = df
        
        d = df_agg.set_index("scenario").reindex(scenarios)[y]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        x = np.arange(len(scenarios))
        ax.bar(x, d.to_numpy(), color=[_color_for(s) for s in scenarios], 
               edgecolor="black", linewidth=0.6)
        
        ax.set_xticks(x, [label(s) for s in scenarios], rotation=15, ha="right")

    ax.set_ylabel(y.replace("_", r"\_"))
    _savefig(fig, out)


# -----------------------------------------------------------------------------
# 7) Run all plots
# -----------------------------------------------------------------------------

def run_all_plots(
    *,
    wake_losses_csv: Path | None = None,
    wake_density_csv: Path | None = None,
    cf_metrics_csv: Path | None = None,
    resolution_csv: Path | None = None,
    system_csv: Path | None = None,
    capacity_geojson: Path | None = None,
    delta_cf_geojson: Path | None = None,
    scenarios: list[str] | None = None,
    baseline: str = "base",
    out_dir: Path,
    networks_dir: Path | None = None,
    split: int = 1000,
    area: str = "northsea",
    regions_dir: Path = Path("wake_extra"),
) -> None:
    """
    Generate all available plots based on provided input files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if scenarios is None:
        scenarios = ["base", "standard", "glaum", "new_more"]
    
    print("Generating all plots...")
    
    # Wake loss distribution plots
    if wake_losses_csv and wake_losses_csv.exists():
        print(f"  - Wake loss PDF from {wake_losses_csv.name}")
        df = _read_table(wake_losses_csv)
        plot_wake_loss_pdf(df, out=out_dir / "wake_loss_pdf.pdf")
        
        print(f"  - Wake loss CDF from {wake_losses_csv.name}")
        plot_wake_loss_cdf(df, out=out_dir / "wake_loss_cdf.pdf")
        
        print(f"  - Wake loss boxplot from {wake_losses_csv.name}")
        plot_wake_loss_box(df, out=out_dir / "wake_loss_box.pdf")
    
    # Capacity factor distribution plots
    if cf_metrics_csv and cf_metrics_csv.exists():
        df = _read_table(cf_metrics_csv)
        
        # Available CF plots
        print(f"  - Available CF PDF from {cf_metrics_csv.name}")
        plot_cf_pdf(df, out=out_dir / "available_cf_pdf.pdf")
        
        print(f"  - Available CF CDF from {cf_metrics_csv.name}")
        plot_cf_cdf(df, out=out_dir / "available_cf_cdf.pdf")
        
        print(f"  - Available CF boxplot from {cf_metrics_csv.name}")
        plot_cf_box(df, out=out_dir / "available_cf_box.pdf")
        
        # Dispatch CF plots
        print(f"  - Dispatch CF PDF from {cf_metrics_csv.name}")
        plot_dispatch_cf_pdf(df, out=out_dir / "dispatch_cf_pdf.pdf")
        
        print(f"  - Dispatch CF CDF from {cf_metrics_csv.name}")
        plot_dispatch_cf_cdf(df, out=out_dir / "dispatch_cf_cdf.pdf")
        
        print(f"  - Dispatch CF boxplot from {cf_metrics_csv.name}")
        plot_dispatch_cf_box(df, out=out_dir / "dispatch_cf_box.pdf")
        
        # Curtailment CF plots
        print(f"  - Curtailment CF PDF from {cf_metrics_csv.name}")
        plot_curtailment_cf_pdf(df, out=out_dir / "curtailment_cf_pdf.pdf")
        
        print(f"  - Curtailment CF CDF from {cf_metrics_csv.name}")
        plot_curtailment_cf_cdf(df, out=out_dir / "curtailment_cf_cdf.pdf")
        
        print(f"  - Curtailment CF boxplot from {cf_metrics_csv.name}")
        plot_curtailment_cf_box(df, out=out_dir / "curtailment_cf_box.pdf")
    
    # Wake loss vs density
    if wake_density_csv and wake_density_csv.exists():
        print(f"  - Wake loss vs density from {wake_density_csv.name}")
        df = _read_table(wake_density_csv)
        plot_wake_loss_vs_density(df, out=out_dir / "wake_loss_vs_density.pdf")
    
    # Delta CF maps
    if delta_cf_geojson and delta_cf_geojson.exists():
        gdf = _read_geo(delta_cf_geojson)
        for s in scenarios:
            try:
                print(f"  - ΔCF map for {s}")
                plot_delta_cf_map(gdf, scenario=s, out=out_dir / f"delta_cf_{canon(s)}.pdf")
            except Exception as e:
                print(f"    Warning: Could not generate ΔCF map for {s}: {e}")
    
    # Capacity density maps
    if capacity_geojson and capacity_geojson.exists():
        gdf = _read_geo(capacity_geojson)
        for s in scenarios:
            try:
                print(f"  - Capacity density map for {s}")
                plot_capacity_density_map(gdf, scenario=s, out=out_dir / f"cap_density_{canon(s)}.pdf")
            except Exception as e:
                print(f"    Warning: Could not generate capacity map for {s}: {e}")
        
        # Delta capacity maps
        for s in [sc for sc in scenarios if canon(sc) != canon(baseline)]:
            try:
                print(f"  - Δ Capacity density map for {s} vs {baseline}")
                plot_capacity_density_delta_map(
                    gdf, scenario=s, baseline=baseline, 
                    out=out_dir / f"cap_delta_{canon(s)}_vs_{canon(baseline)}.pdf"
                )
            except Exception as e:
                print(f"    Warning: Could not generate Δ capacity map for {s}: {e}")
    
    # Resolution lines
    if resolution_csv and resolution_csv.exists():
        df = _read_table(resolution_csv)
        metrics = [col for col in df.columns if col not in ["scenario", "split_km2"]]
        for m in metrics:
            try:
                print(f"  - Resolution plot for {m}")
                plot_resolution_lines(df, y=m, out=out_dir / f"resolution_{m}.pdf")
            except Exception as e:
                print(f"    Warning: Could not generate resolution plot for {m}: {e}")
    
    # System metric bars
    if system_csv and system_csv.exists():
        df = _read_table(system_csv)
        
        # Combined plots for key system metrics
        try:
            print(f"  - Transmission expansion (AC/DC)")
            plot_transmission_expansion(df, out=out_dir / "transmission_expansion.pdf")
        except Exception as e:
            print(f"    Warning: Could not generate transmission expansion plot: {e}")
        
        try:
            print(f"  - System cost comparison")
            plot_system_cost_comparison(df, out=out_dir / "system_cost_comparison.pdf")
        except Exception as e:
            print(f"    Warning: Could not generate system cost plot: {e}")
        
        try:
            print(f"  - Curtailment vs cost analysis")
            plot_curtailment_analysis(df, out=out_dir / "curtailment_cost_analysis.pdf")
        except Exception as e:
            print(f"    Warning: Could not generate curtailment analysis plot: {e}")
        
        # Individual metric bars
        metrics = [col for col in df.columns if col not in ["scenario", "split"]]
        for m in metrics:
            try:
                print(f"  - System bar chart for {m}")
                plot_system_bars(df, y=m, out=out_dir / f"system_{m}.pdf")
            except Exception as e:
                print(f"    Warning: Could not generate system bar chart for {m}: {e}")
    
    # Capacity density maps from networks
    if networks_dir and networks_dir.exists() and pypsa is not None:
        print("  - Capacity density delta maps from networks")
        try:
            # Load networks for each scenario
            networks = {}
            for scenario in scenarios:
                nc_pattern = f"{scenario}-s{split}-biasFalse"
                nc_dir = networks_dir / nc_pattern / "postnetworks"
                if nc_dir.exists():
                    nc_files = list(nc_dir.glob("*.nc"))
                    if nc_files:
                        print(f"    Loading network for {scenario}: {nc_files[0].name}")
                        networks[scenario] = pypsa.Network(str(nc_files[0]))
            
            if networks:
                plot_capacity_density_maps(
                    networks,
                    split=split,
                    area=area,
                    out=out_dir / "capacity_density_delta_maps.pdf",
                    baseline_scenario=baseline,
                    regions_dir=regions_dir,
                )
        except Exception as e:
            print(f"    Warning: Failed to create capacity density maps: {e}")
    
    print(f"\nAll plots saved to {out_dir}/")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Wake chapter thesis plots (styled).")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dist", help="Wake loss PDF (hist density).")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--bins", type=int, default=40)

    p = sub.add_parser("cdf", help="Wake loss CDF.")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("box", help="Wake loss boxplot.")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("loss_vs_density", help="Wake loss vs density scatter + binned mean.")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("delta_cf_map", help="ΔCF map for a scenario.")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--scenario", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--value-col", default="delta_cf")

    p = sub.add_parser("cap_map", help="Capacity density map for a scenario.")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--scenario", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--value-col", default="cap_mw_per_km2")

    p = sub.add_parser("cap_delta_map", help="Δ capacity density map vs baseline.")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--scenario", required=True, type=str)
    p.add_argument("--baseline", default="base")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--value-col", default="cap_mw_per_km2")
    p.add_argument("--id-col", default="region_id")

    p = sub.add_parser("resolution_lines", help="Metric vs spatial resolution (log coarse->fine).")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--y", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("system_bars", help="Bar chart of system metric by scenario.")
    p.add_argument("--in", dest="inp", required=True, type=Path)
    p.add_argument("--y", required=True, type=str)
    p.add_argument("--out", required=True, type=Path)

    p = sub.add_parser("all", help="Generate all plots from input files.")
    p.add_argument("--wake-losses", type=Path, help="CSV with wake loss data")
    p.add_argument("--wake-density", type=Path, help="CSV with wake loss vs density data")
    p.add_argument("--cf-metrics", type=Path, help="CSV with capacity factor metrics")
    p.add_argument("--resolution", type=Path, help="CSV with resolution metrics")
    p.add_argument("--system", type=Path, help="CSV with system metrics")
    p.add_argument("--capacity-geo", type=Path, help="GeoJSON with capacity density data")
    p.add_argument("--delta-cf-geo", type=Path, help="GeoJSON with delta CF data")
    p.add_argument("--scenarios", nargs="+", help="List of scenarios to plot")
    p.add_argument("--baseline", default="base", help="Baseline scenario for delta plots")
    p.add_argument("--out-dir", required=True, type=Path, help="Output directory for all plots")
    p.add_argument("--networks-dir", type=Path, help="Directory containing network .nc files for capacity density maps")
    p.add_argument("--split", type=int, default=1000, help="Split size for region files (default: 1000)")
    p.add_argument("--area", default="northsea", help="Area name for region files (default: northsea)")
    p.add_argument("--regions-dir", type=Path, default=Path("wake_extra"), help="Directory containing region GeoJSON files")

    args = parser.parse_args()

    if args.cmd in {"dist", "cdf", "box", "loss_vs_density", "resolution_lines", "system_bars"}:
        df = _read_table(args.inp)

    if args.cmd == "dist":
        plot_wake_loss_pdf(df, out=args.out, bins=args.bins)
    elif args.cmd == "cdf":
        plot_wake_loss_cdf(df, out=args.out)
    elif args.cmd == "box":
        plot_wake_loss_box(df, out=args.out)
    elif args.cmd == "loss_vs_density":
        plot_wake_loss_vs_density(df, out=args.out)
    elif args.cmd == "delta_cf_map":
        gdf = _read_geo(args.inp)
        plot_delta_cf_map(gdf, scenario=args.scenario, out=args.out, value_col=args.value_col)
    elif args.cmd == "cap_map":
        gdf = _read_geo(args.inp)
        plot_capacity_density_map(gdf, scenario=args.scenario, out=args.out, value_col=args.value_col)
    elif args.cmd == "cap_delta_map":
        gdf = _read_geo(args.inp)
        plot_capacity_density_delta_map(
            gdf,
            scenario=args.scenario,
            baseline=args.baseline,
            out=args.out,
            value_col=args.value_col,
            id_col=args.id_col,
        )
    elif args.cmd == "resolution_lines":
        plot_resolution_lines(df, y=args.y, out=args.out)
    elif args.cmd == "system_bars":
        plot_system_bars(df, y=args.y, out=args.out)
    elif args.cmd == "all":
        run_all_plots(
            wake_losses_csv=args.wake_losses,
            wake_density_csv=args.wake_density,
            cf_metrics_csv=args.cf_metrics,
            resolution_csv=args.resolution,
            system_csv=args.system,
            capacity_geojson=args.capacity_geo,
            delta_cf_geojson=args.delta_cf_geo,
            scenarios=args.scenarios,
            baseline=args.baseline,
            out_dir=args.out_dir,
            networks_dir=args.networks_dir,
            split=args.split,
            area=args.area,
            regions_dir=args.regions_dir,
        )
    else:
        raise RuntimeError(f"Unknown command: {args.cmd}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        main()
        
# # EXAMPLE USAGE:
# python compare_wake_runs_new.py all \
#     --wake-losses data/wake_extracted/wake_losses.csv \
#     --system data/wake_extracted/system_metrics.csv \
#     --resolution data/wake_extracted/resolution_metrics.csv \
#     --scenarios base standard glaum new_more \
#     --out-dir ../plots/wake_analysis/
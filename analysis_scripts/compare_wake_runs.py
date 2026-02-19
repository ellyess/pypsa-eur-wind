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
from typing import Optional, Tuple
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:
    sns = None

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
from plotting_style import thesis_plot_style, apply_spatial_resolution_axis, add_resolution_markers, format_axes_standard

# Try importing wake_helpers for wake model plotting
WAKE_HELPERS_AVAILABLE = False
try:
    _scripts_dir = HERE.parent / "scripts"
    if _scripts_dir.exists():
        sys.path.insert(0, str(_scripts_dir))
    from wake_helpers import WakeSplitSpec, _glaum_spec, _new_more_spec
    WAKE_HELPERS_AVAILABLE = True
except Exception:
    pass


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

def _style_boxplot(bp, colors: list[str], *, alpha: float = 0.7) -> None:
    """Color box faces, edges, and median lines to match per-scenario colors."""
    for patch, median, color in zip(bp["boxes"], bp["medians"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_alpha(alpha)
        median.set_color(color)
        median.set_linewidth(1.2)

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
    format_axes_standard(fig)
    fig.savefig(out, bbox_inches="tight")
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
    if has_splits:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=6)
    else:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
    if has_splits:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=6)
    else:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
            medianprops=dict(color="black"),
            widths=0.6,
        )

        _style_boxplot(bp, colors)
        
        # Set x-axis labels for splits
        group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([_format_split_label(s) for s in splits])
        
        # Add legend for scenarios
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=_color_for(s), alpha=0.7, label=label(s)) 
                          for s in scenarios]
        ax.legend(handles=legend_elements, title="Wake model", frameon=False,
                 loc='best', fontsize=7)
        
    else:
        # Single split or no split column
        data = [df.loc[df["scenario"] == s, "wake_loss"].dropna().to_numpy() for s in scenarios]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        bp = ax.boxplot(
            data,
            tick_labels=[label(s) for s in scenarios],
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black"),
        )
        _style_boxplot(bp, [_color_for(s) for s in scenarios], alpha=1.0)
        
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
    if has_splits:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=6)
    else:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
    if has_splits:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=6)
    else:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
            medianprops=dict(color="black"),
        )

        _style_boxplot(bp, colors)
        
        # Set x-axis labels for splits
        group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([_format_split_label(s) for s in splits])
        
        # Add legend for scenarios
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=_color_for(s), alpha=0.7, label=label(s)) 
                          for s in scenarios]
        ax.legend(handles=legend_elements, title="Wake model", frameon=False,
                 loc='best', fontsize=7)
        
    else:
        # Single split or no split column
        data = [df.loc[df["scenario"] == s, "available_cf"].dropna().to_numpy() for s in scenarios]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        bp = ax.boxplot(
            data,
            tick_labels=[label(s) for s in scenarios],
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="black"),
        )
        _style_boxplot(bp, [_color_for(s) for s in scenarios], alpha=1.0)
        
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
    if has_splits:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=6)
    else:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
    if has_splits:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=6)
    else:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
                       showfliers=False, medianprops=dict(color="black"))

        _style_boxplot(bp, colors)
        
        group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([_format_split_label(s) for s in splits])
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=_color_for(s), alpha=0.7, label=label(s)) 
                          for s in scenarios]
        ax.legend(handles=legend_elements, title="Wake model", frameon=False,
                 loc='best', fontsize=7)
    else:
        data = [df.loc[df["scenario"] == s, "dispatch_cf"].dropna().to_numpy() for s in scenarios]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        bp = ax.boxplot(data, tick_labels=[label(s) for s in scenarios], patch_artist=True,
                       showfliers=False, medianprops=dict(color="black"))
        _style_boxplot(bp, [_color_for(s) for s in scenarios], alpha=1.0)
        
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
    if has_splits:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=6)
    else:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
    if has_splits:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=6)
    else:
        ax.legend(loc='best', frameon=False, fontsize=9)
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
                       showfliers=False, medianprops=dict(color="black"))

        _style_boxplot(bp, colors)
        
        group_centers = [i * group_width + (n_scenarios - 1) / 2 for i in range(n_splits)]
        ax.set_xticks(group_centers)
        ax.set_xticklabels([_format_split_label(s) for s in splits])
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=_color_for(s), alpha=0.7, label=label(s)) 
                          for s in scenarios]
        ax.legend(handles=legend_elements, title="Wake model", frameon=False,
                 loc='best', fontsize=7)
    else:
        data = [df.loc[df["scenario"] == s, "curtailment_cf"].dropna().to_numpy() for s in scenarios]
        
        fig, ax = plt.subplots(figsize=(16.4 * cm, 6.0 * cm), layout="constrained")
        bp = ax.boxplot(data, tick_labels=[label(s) for s in scenarios], patch_artist=True,
                       showfliers=False, medianprops=dict(color="black"))
        _style_boxplot(bp, [_color_for(s) for s in scenarios], alpha=1.0)
        
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
    ax.legend(loc='best', frameon=False)
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

    apply_spatial_resolution_axis(ax, annotate=False)
    add_resolution_markers(ax, df["split_km2"].dropna().values)
    ax.set_ylabel(y.replace("_", r"\_"))
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=6)
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
        ax1.legend(title=r"$A_{region}^{max}$", frameon=False, fontsize=6,
                   loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=n_splits)
        
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
        ax.legend(title=r"$A_{region}^{max}$", frameon=False, fontsize=6,
                  loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=n_splits)
        
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
        
        ax.legend(handles=legend_elements, frameon=False, fontsize=6,
                  loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=4)
        
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
        ax.legend(loc='best', frameon=False)

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
        ax.legend(title="Split", frameon=False, fontsize=6,
                  loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=n_splits)
        
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
# Wake methods plotting helpers
# -----------------------------------------------------------------------------

def _step_xy(breaks: np.ndarray, M: np.ndarray, x_max: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build x,y arrays for matplotlib step plot (where='post'),
    ensuring the step reaches x_max and y has same length as x.
    """
    breaks = np.asarray(breaks, dtype=float)
    M = np.asarray(M, dtype=float)

    # ensure starts at 0
    if breaks[0] > 0:
        breaks = np.r_[0.0, breaks]

    # enforce monotonic (remove duplicates)
    breaks = np.unique(breaks)

    # ensure ends at x_max
    if breaks[-1] < x_max:
        breaks = np.r_[breaks, x_max]
    else:
        breaks[-1] = x_max

    # M should be per-interval => len = len(breaks)-1
    need = len(breaks) - 1
    if len(M) >= need:
        M_use = M[:need]
    else:
        M_use = np.r_[M, np.full(need - len(M), M[-1])]

    # matplotlib step expects y same length as x
    y = np.r_[M_use, M_use[-1]]
    return breaks, y


def _cum_total_loss_from_piecewise_marginal(
    q: np.ndarray,
    breaks: np.ndarray,
    M: np.ndarray,
) -> np.ndarray:
    """
    Compute total loss T(q) for a piecewise-constant marginal loss M over breaks.

    q: query points (same units as breaks)
    breaks: length K+1, tier boundaries
    M: length K, marginal loss per tier (fraction)
    Returns T(q) (fraction): (1/q) * integral_0^q M(s) ds, with T(0)=0
    """
    q = np.asarray(q, dtype=float)
    breaks = np.asarray(breaks, dtype=float)
    M = np.asarray(M, dtype=float)

    # Precompute cumulative integral at tier boundaries
    K = len(M)
    widths = np.diff(breaks)  # length K
    cum_int = np.zeros(K + 1, dtype=float)
    cum_int[1:] = np.cumsum(M * widths)

    T = np.zeros_like(q, dtype=float)
    for i, qi in enumerate(q):
        if qi <= 0:
            T[i] = 0.0
            continue

        # find tier index t where breaks[t] <= qi < breaks[t+1]
        t = np.searchsorted(breaks, qi, side="right") - 1
        t = int(np.clip(t, 0, K))  # K means beyond last finite break

        if t >= K:
            # beyond last breakpoint: hold last M constant
            extra = qi - breaks[-1]
            integ = cum_int[-1] + M[-1] * extra
        else:
            extra = qi - breaks[t]
            integ = cum_int[t] + M[t] * extra

        T[i] = integ / qi

    return T


def _extend_density_total_loss_to_capacity_axis(
    P_GW: np.ndarray,
    *,
    A_ref_km2: float,
    x_breaks: np.ndarray,
    M_den: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Option A: represent density-tier model on the capacity axis P (GW),
    extending beyond x_max by holding last marginal tier constant.

    Returns:
        (T_den_on_P, M_den_on_P_step)
    """
    P_GW = np.asarray(P_GW, dtype=float)
    x_breaks = np.asarray(x_breaks, dtype=float)
    M_den = np.asarray(M_den, dtype=float)

    # map P <-> x for reference area:
    # x [MW/km2] = (P [GW] * 1000 [MW/GW]) / A_ref_km2
    def P_to_x(Pgw: np.ndarray) -> np.ndarray:
        return (Pgw * 1000.0) / float(A_ref_km2)

    x_of_P = P_to_x(P_GW)

    # total loss computed in x-space with constant extension beyond x_max
    T_den = _cum_total_loss_from_piecewise_marginal(
        q=x_of_P,
        breaks=x_breaks,
        M=M_den,
    )

    # Marginal loss step function on P-axis: same tiers but mapped
    # Breaks in P:
    P_breaks = (x_breaks * float(A_ref_km2)) / 1000.0  # GW
    # Build step arrays for plotting
    P_step = np.r_[P_breaks[:-1], P_GW[-1]]
    M_step = np.r_[M_den, M_den[-1]]

    return T_den, (P_breaks, M_den)


def _capacity_tier_total_loss_on_capacity_axis(
    P_GW: np.ndarray,
    spec: WakeSplitSpec,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute capacity-tier model total loss T(P) for P-axis (GW),
    with constant extension beyond last finite breakpoint by holding last marginal constant.
    """
    P_GW = np.asarray(P_GW, dtype=float)

    # Convert max_caps (MW segment sizes) into breakpoints in GW
    # spec.max_caps is segment capacity "amount" per tier (except last inf)
    # For glaum: [2e3, 10e3, inf] MW => breakpoints at 2GW, 12GW, inf
    seg_caps_MW = np.asarray(spec.max_caps, dtype=float)
    seg_caps_MW_finite = seg_caps_MW[np.isfinite(seg_caps_MW)]
    breaks_GW = np.r_[0.0, np.cumsum(seg_caps_MW_finite) / 1000.0]  # GW

    M_cap = np.asarray(spec.factors, dtype=float)

    breaks2 = np.array([0.0, breaks_GW[1], breaks_GW[2]])  # [0,2,12]
    M1, M2, M3 = M_cap

    T = np.zeros_like(P_GW, dtype=float)
    for i, P in enumerate(P_GW):
        if P <= 0:
            T[i] = 0.0
            continue
        if P <= breaks2[1]:
            integ = M1 * (P - 0.0)
        elif P <= breaks2[2]:
            integ = M1 * (breaks2[1] - 0.0) + M2 * (P - breaks2[1])
        else:
            integ = (
                M1 * (breaks2[1] - 0.0)
                + M2 * (breaks2[2] - breaks2[1])
                + M3 * (P - breaks2[2])
            )
        T[i] = integ / P

    # For marginal step plotting:
    P_breaks = np.array([0.0, breaks2[1], breaks2[2]])  # boundaries (last tier extends)
    return T, (P_breaks, np.array([M1, M2, M3], dtype=float))


def _capacity_tier_total_loss_on_density_axis(
    x_grid: np.ndarray,
    spec: WakeSplitSpec,
    *,
    A_ref_km2: float,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Show capacity-tier model as a function of density x by mapping x -> P (for a reference area).
    """
    x_grid = np.asarray(x_grid, dtype=float)
    P_GW = (x_grid * float(A_ref_km2)) / 1000.0
    T_P, (P_breaks, M) = _capacity_tier_total_loss_on_capacity_axis(P_GW, spec)
    # Map P breaks to x breaks:
    x_breaks = (P_breaks * 1000.0) / float(A_ref_km2)
    return T_P, (x_breaks, M)


def _density_tier_total_loss_on_density_axis(
    x_grid: np.ndarray,
    x_breaks: np.ndarray,
    M_den: np.ndarray,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    x_grid = np.asarray(x_grid, dtype=float)
    x_breaks = np.asarray(x_breaks, dtype=float)
    M_den = np.asarray(M_den, dtype=float)
    T = _cum_total_loss_from_piecewise_marginal(x_grid, x_breaks, M_den)
    return T, (x_breaks, M_den)


def plot_wake_models_density_two_areas(
    *,
    A_left_km2: float = 1000.0,
    A_right_km2: float = 10000.0,
    x_max: float = 4.0,
    alpha_uniform: float = 0.8855,
    savepath: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    2x2 review-proof wake figure where ALL panels use capacity density x (MW/km2).

    Columns: different reference region areas (A_left_km2, A_right_km2)
    Rows:
        - Top: total loss T(x)
        - Bottom: marginal loss M(x) (tier step)

    Overlays in each panel:
        - capacity-tier (mapped to x using A_ref): blue
        - density-tier (native in x): red
        - uniform scaling correction (constant): dashed grey
        - no-wake baseline: dotted grey
    """
    if not WAKE_HELPERS_AVAILABLE:
        raise ImportError("wake_helpers not available; cannot plot wake models.")

    style = thesis_plot_style()
    cm = style["cm"]
    lw = style["lw"]

    if sns is not None:
        sns.set_theme(style="ticks")
        sns.despine()

    # --- specs from wake methods ---
    spec_cap = _glaum_spec()
    spec_den, x_breaks_den = _new_more_spec()
    M_den = np.asarray(spec_den.factors, dtype=float)

    # --- grids ---
    x_grid = np.geomspace(1e-3, x_max, 1200)

    _WAKE_COLORS = WAKE_MODEL_COLORS

    # constants
    loss_uniform = 1.0 - float(alpha_uniform)

    fig, axes = plt.subplots(2, 2, figsize=(18 * cm, 14 * cm), dpi=dpi, sharey="row")
    ax_TL, ax_TR = axes[0, 0], axes[0, 1]
    ax_ML, ax_MR = axes[1, 0], axes[1, 1]

    def _plot_column(ax_T, ax_M, A_ref_km2: float, col_title: str) -> None:
        # --- TOTAL LOSS (density axis) ---
        T_cap_x, (x_breaks_cap, M_cap) = _capacity_tier_total_loss_on_density_axis(
            x_grid, spec_cap, A_ref_km2=A_ref_km2
        )
        T_den_x, (x_breaks_den_arr, M_den_arr) = _density_tier_total_loss_on_density_axis(
            x_grid, x_breaks_den, M_den
        )

        ax_T.plot(x_grid, T_cap_x, color=_WAKE_COLORS["glaum"], lw=lw, label="Tiered capacity (total)")
        ax_T.plot(x_grid, T_den_x, color=_WAKE_COLORS["new_more"], lw=lw, label="Tiered density (total)")
        ax_T.axhline(loss_uniform, color=_WAKE_COLORS["standard"], ls="--", lw=lw*0.7, label="Uniform scaling")
        ax_T.axhline(0.0, color=_WAKE_COLORS["base"], ls=":", lw=lw*0.7, label="No-wake")

        # threshold markers
        for xb in np.asarray(x_breaks_cap, dtype=float)[1:]:
            if 0 < xb < x_max:
                ax_T.axvline(xb, color=_WAKE_COLORS["glaum"], alpha=0.18, lw=1.4)
        for xb in np.asarray(x_breaks_den_arr, dtype=float)[1:]:
            if 0 < xb < x_max:
                ax_T.axvline(xb, color=_WAKE_COLORS["new_more"], alpha=0.18, lw=1.4)

        ax_T.set_title(f"Total loss vs density\n{col_title}")
        ax_T.set_xlabel(r"Installed Capacity Density [MW/km$^2$]")
        ax_T.set_ylabel("Loss (fraction)")

        # --- MARGINAL LOSS (density axis) ---
        x_edges_cap, y_cap = _step_xy(np.asarray(x_breaks_cap, float), np.asarray(M_cap, float), x_max=x_max)
        x_edges_den_step, y_den = _step_xy(np.asarray(x_breaks_den_arr, float), np.asarray(M_den_arr, float), x_max=x_max)

        ax_M.step(x_edges_cap, y_cap, where="post", color=_WAKE_COLORS["glaum"], lw=lw, label="Tiered capacity (marginal)")
        ax_M.step(x_edges_den_step, y_den, where="post", color=_WAKE_COLORS["new_more"], lw=lw, label="Tiered density (marginal)")
        ax_M.axhline(loss_uniform, color=_WAKE_COLORS["standard"], ls="--", lw=lw*0.7, label="Uniform scaling")
        ax_M.axhline(0.0, color=_WAKE_COLORS["base"], ls=":", lw=lw*0.7, label="No-wake")

        for xb in x_edges_cap[1:-1]:
            ax_M.axvline(xb, color=_WAKE_COLORS["glaum"], alpha=0.18, lw=1.4)
        for xb in x_edges_den_step[1:-1]:
            ax_M.axvline(xb, color=_WAKE_COLORS["new_more"], alpha=0.18, lw=1.4)

        ax_M.set_title(f"Marginal loss (tier step) vs density\n{col_title}")
        ax_M.set_xlabel(r"Installed Capacity Density [MW/km$^2$]")
        ax_M.set_ylabel("Marginal loss (fraction)")

        # --- formatting ---
        for ax in (ax_T, ax_M):
            ax.set_xscale("log")
            ax.set_xlim(1e-3, x_max)
            ax.set_xticks([1e-3, 1e-2, 1e-1, 0.25, 1.0, 2.5, 4.0])
            ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_major_formatter(lambda v, pos: f"{100*v:.0f}%")
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    _plot_column(ax_TL, ax_ML, A_left_km2, col_title=rf"$A_{{\mathrm{{region}}}}$ = {A_left_km2:,.0f} km$^2$")
    _plot_column(ax_TR, ax_MR, A_right_km2, col_title=rf"$A_{{\mathrm{{region}}}}$ = {A_right_km2:,.0f} km$^2$")

    # One shared legend
    handles, labels = ax_TL.get_legend_handles_labels()
    h2, l2 = ax_ML.get_legend_handles_labels()
    handles += h2
    labels += l2

    seen = set()
    H, L = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            H.append(h)
            L.append(l)

    fig.legend(H, L, loc="best", ncol=1, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig


# -----------------------------------------------------------------------------
# 7a) Wake loss vs resolution per model + System cost delta
# -----------------------------------------------------------------------------

def plot_wake_loss_vs_resolution(
    df: pd.DataFrame,
    *,
    out: Path,
) -> None:
    """Line plot: mean wake loss (y) vs spatial resolution (x), one line per wake model.

    Expects a DataFrame with columns: scenario, split_km2, and a wake-loss metric
    (e.g. mean_wake_loss, median_available_cf, or similar).
    """
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    # Detect the best available y-column
    y_col = None
    for candidate in ["mean_wake_loss", "wake_loss_mean", "available_cf_mean", "mean_available_cf"]:
        if candidate in df.columns:
            y_col = candidate
            break
    if y_col is None:
        # Fallback: use first numeric column that's not scenario/split
        numeric_cols = [c for c in df.columns if c not in ("scenario", "split_km2") and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            y_col = numeric_cols[0]
        else:
            print("[WARN] No suitable y column found in resolution CSV; skipping.")
            plt.close(fig)
            return

    scenarios_present = df["scenario"].unique()
    for scen in WAKE_ORDER:
        if scen not in scenarios_present:
            continue
        sub = df[df["scenario"] == scen].sort_values("split_km2")
        color = WAKE_MODEL_COLORS.get(canon(scen), None)
        ax.plot(sub["split_km2"], sub[y_col], marker="o", linewidth=1.4,
               label=label(canon(scen)), color=color)

    apply_spatial_resolution_axis(ax, annotate=False)
    add_resolution_markers(ax, df["split_km2"].dropna().values)
    ylabel_nice = y_col.replace("_", " ").title()
    ax.set_ylabel(ylabel_nice)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def plot_system_cost_delta_pct(
    df: pd.DataFrame,
    *,
    baseline: str = "base",
    out: Path,
) -> None:
    """Bar chart: system cost delta (%) relative to baseline, per wake model.

    Expects a DataFrame with columns: scenario, and one of objective/system_cost.
    """
    # Detect cost column
    cost_col = None
    for candidate in ["objective", "system_cost", "total_cost"]:
        if candidate in df.columns:
            cost_col = candidate
            break
    if cost_col is None:
        print("[WARN] No cost column found in system metrics CSV; skipping.")
        return

    baseline_val = df.loc[df["scenario"] == baseline, cost_col]
    if baseline_val.empty:
        print(f"[WARN] No baseline ({baseline}) found in system metrics.")
        return
    baseline_val = baseline_val.values[0]

    if baseline_val == 0 or np.isnan(baseline_val):
        print("[WARN] Baseline cost is 0 or NaN; skipping.")
        return

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    scens = [s for s in WAKE_ORDER if s in df["scenario"].values and s != baseline]
    deltas = []
    colors = []
    labels_list = []
    for scen in scens:
        val = df.loc[df["scenario"] == scen, cost_col].values[0]
        delta_pct = 100 * (val - baseline_val) / abs(baseline_val)
        deltas.append(delta_pct)
        colors.append(WAKE_MODEL_COLORS.get(canon(scen), "#999999"))
        labels_list.append(label(canon(scen)))

    x = np.arange(len(labels_list))
    bars = ax.bar(x, deltas, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, rotation=15, ha="right")
    ax.set_ylabel("System cost change vs baseline (%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate bars
    for i, v in enumerate(deltas):
        sign = "+" if v >= 0 else ""
        ax.text(i, v + (0.1 if v >= 0 else -0.15), f"{sign}{v:.2f}%",
               ha="center", va="bottom" if v >= 0 else "top", fontsize=7)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# 7b) Run all plots
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

    # Wake model comparison (2x2 density figure)
    if WAKE_HELPERS_AVAILABLE:
        print("  - Wake model density comparison (2x2)")
        try:
            plot_wake_models_density_two_areas(
                savepath=str(out_dir / "wake_models_density_comparison.pdf"),
            )
        except Exception as e:
            print(f"    Warning: Failed to create wake model comparison: {e}")

    # Wake loss vs resolution per model
    if resolution_csv and resolution_csv.exists():
        print("  - Wake loss vs resolution per model")
        try:
            df_res = _read_table(resolution_csv)
            plot_wake_loss_vs_resolution(df_res, out=out_dir / "wake_loss_vs_resolution.pdf")
        except Exception as e:
            print(f"    Warning: Failed to create wake loss vs resolution plot: {e}")

    # System cost delta (%) per wake model
    if system_csv and system_csv.exists():
        print("  - System cost delta (%) per wake model")
        try:
            df_sys = _read_table(system_csv)
            plot_system_cost_delta_pct(df_sys, baseline=baseline, out=out_dir / "system_cost_delta_pct.pdf")
        except Exception as e:
            print(f"    Warning: Failed to create system cost delta plot: {e}")

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
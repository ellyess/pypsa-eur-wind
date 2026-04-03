#!/usr/bin/env python3
"""
Tier 1 system-level validation: compare modelled wind capacity factors
(from atlite profiles) against ENTSO-E observed wind generation.

Supports N labeled scenarios for comparing different bias correction methods
(e.g., IDW vs Kriging vs uncorrected).

Approach:
  1. Load pre-built atlite wind profiles for each scenario
  2. Aggregate bus-level CFs to country level (weighted by p_nom_max)
  3. Scale by IRENASTAT actual installed capacity per country (2023)
  4. Compare against ENTSO-E observed wind generation

Usage:
    # Three-way comparison: uncorrected vs IDW vs Kriging
    python validate_wind_generation.py \
        --entsoe-csv data/entsoe_generation_2023.csv \
        --scenario "Uncorrected (ERA5)" resources/.../base-s100000-biasFalse \
        --scenario "IDW corrected" resources/.../base-s100000-biasidw \
        --scenario "Kriging corrected" resources/.../base-s100000-biaskriging \
        --baseline "Uncorrected (ERA5)" \
        --out plots/validation_idw_vs_kriging

    # Legacy two-scenario comparison still works:
    python validate_wind_generation.py \
        --entsoe-csv data/entsoe_generation_2023.csv \
        --scenario "Corrected" resources/.../biasTrue \
        --scenario "Uncorrected" resources/.../biasFalse \
        --out plots/validation
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml

# Add analysis_scripts dir to path for shared modules
sys.path.insert(0, str(Path(__file__).parent))

from plotting_style import thesis_plot_style, format_axes_standard
from thesis_colors import THESIS_COLORS

# Apply thesis-wide plotting style
_style = thesis_plot_style()
cm = _style["cm"]
HALF_WIDTH = _style["HALF_WIDTH"]
FULL_WIDTH = _style["FULL_WIDTH"]

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

COUNTRIES = ["BE", "DK", "NL", "GB", "FR", "DE", "NO"]

# IRENASTAT installed wind capacities for 2023 (MW)
# Source: IRENA Renewable Capacity Statistics 2024
INSTALLED_CAPACITY_2023 = {
    "onshore": {
        "BE": 3163, "DK": 4845, "NL": 6204, "GB": 15006,
        "FR": 22137, "DE": 61111, "NO": 5281,
    },
    "offshore": {
        "BE": 2262, "DK": 2308, "NL": 3721, "GB": 14738,
        "FR": 482, "DE": 8384, "NO": 0,
    },
}

WIND_CARRIERS = {
    "onshore": ["onwind"],
    "offshore": ["offwind-ac", "offwind-dc", "offwind-float"],
}

# Mapping from ENTSO-E carrier names to thesis wind types
ENTSOE_NICE_NAMES = {
    "Wind Onshore": "onshore",
    "Onshore Wind": "onshore",
    "Wind Offshore": "offshore",
    "Offshore Wind": "offshore",
}

# Plot colours — observed is always near-black
COL_OBS = "#1a1a1a"

# Scenario colour palette (colorblind-safe, Okabe-Ito based)
SCENARIO_PALETTE = [
    "#4D4D4D",  # charcoal (baseline/uncorrected)
    "#5DAE8B",  # muted green (IDW / PyVWF)
    "#0072B2",  # blue (Kriging)
    "#D55E00",  # orange (spare)
    "#8172B2",  # purple (spare)
]


# ─────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────

def bus_country(bus_name: str) -> str:
    """Extract ISO-2 country code from a PyPSA-Eur bus name."""
    return str(bus_name)[:2]


def load_profiles(resource_dir: Path) -> dict[str, xr.Dataset]:
    """Load wind profile NetCDFs from a scenario resource directory."""
    profiles = {}
    for carrier in ["onwind", "offwind-ac", "offwind-dc", "offwind-float"]:
        path = resource_dir / f"profile_10_{carrier}.nc"
        if path.exists():
            profiles[carrier] = xr.open_dataset(path)
    return profiles


def aggregate_profiles_to_country(
    profiles: dict[str, xr.Dataset],
    wind_type: str,
) -> pd.DataFrame:
    """
    Aggregate bus-level profiles to country-level capacity factors.

    CF is weighted by p_nom_max (potential installable capacity per bus).
    Returns DataFrame: index=time, columns=country (ISO2).
    """
    carriers = WIND_CARRIERS[wind_type]

    all_cf = []
    all_weights = []
    all_countries = []

    for carrier in carriers:
        if carrier not in profiles:
            continue
        ds = profiles[carrier]
        cf = ds["profile"].squeeze("year")  # (time, bus)
        p_nom_max = ds["p_nom_max"]  # (bus,)
        buses = ds["bus"].values
        countries = [bus_country(b) for b in buses]

        cf_df = cf.to_pandas()  # index=time, columns=bus
        weights = pd.Series(p_nom_max.values, index=buses)
        country_map = pd.Series(countries, index=buses)

        all_cf.append(cf_df)
        all_weights.append(weights)
        all_countries.append(country_map)

    if not all_cf:
        return pd.DataFrame()

    cf_combined = pd.concat(all_cf, axis=1)
    weights_combined = pd.concat(all_weights)
    countries_combined = pd.concat(all_countries)

    result = {}
    for country in sorted(countries_combined.unique()):
        mask = countries_combined == country
        bus_cols = mask[mask].index
        w = weights_combined[bus_cols]
        if w.sum() == 0:
            continue
        cf_country = (cf_combined[bus_cols] * w).sum(axis=1) / w.sum()
        result[country] = cf_country

    return pd.DataFrame(result)


def load_entsoe_generation(
    csv_path: Path | None = None,
    api_key: str | None = None,
    countries: list[str] | None = None,
    start: str = "2023-01-01",
    end: str = "2024-01-01",
) -> pd.DataFrame:
    """
    Load ENTSO-E wind generation data.

    Returns DataFrame with MultiIndex columns (country, wind_type).
    Values are in MW (instantaneous power).
    """
    if countries is None:
        countries = COUNTRIES

    if csv_path is not None and csv_path.exists():
        gen = pd.read_csv(csv_path, index_col=0, header=[0, 1], parse_dates=True)
        result = {}
        for country in countries:
            if country not in gen.columns.get_level_values(0):
                continue
            country_gen = gen[country]
            for col_name in country_gen.columns:
                wt = ENTSOE_NICE_NAMES.get(col_name)
                if wt is not None:
                    result[(country, wt)] = country_gen[col_name]
        df = pd.DataFrame(result)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["country", "wind_type"])
        return df

    if api_key:
        from entsoe import EntsoePandasClient
        from entsoe.exceptions import NoMatchingDataError

        client = EntsoePandasClient(api_key=api_key)
        start_ts = pd.Timestamp(start, tz="Europe/Brussels")
        end_ts = pd.Timestamp(end, tz="Europe/Brussels")

        result = {}
        for country in countries:
            try:
                gen = client.query_generation(country, start=start_ts, end=end_ts, nett=True)
                gen = gen.tz_localize(None).resample("1h").mean()
                for entsoe_name, wind_type in ENTSOE_NICE_NAMES.items():
                    if entsoe_name in gen.columns:
                        result[(country, wind_type)] = gen[entsoe_name]
            except NoMatchingDataError:
                print(f"  [WARN] No ENTSO-E data for {country}")

        df = pd.DataFrame(result)
        df.columns = pd.MultiIndex.from_tuples(df.columns, names=["country", "wind_type"])
        return df

    raise ValueError(
        "Provide either --entsoe-csv with an existing CSV or "
        "--config with a valid ENTSO-E API key"
    )


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────

def compute_metrics(modelled: pd.Series, observed: pd.Series, name: str = "") -> dict:
    """Compute comparison metrics between modelled and observed generation."""
    aligned = pd.concat(
        [modelled.rename("mod"), observed.rename("obs")], axis=1
    ).dropna()
    if len(aligned) < 10:
        return {"name": name, "n": len(aligned)}

    mod = aligned["mod"]
    obs = aligned["obs"]

    corr = mod.corr(obs)
    mae = (mod - obs).abs().mean()
    rmse = np.sqrt(((mod - obs) ** 2).mean())
    mbe = (mod - obs).mean()
    nrmse = rmse / obs.mean() if obs.mean() != 0 else np.nan

    mod_monthly = mod.resample("ME").mean()
    obs_monthly = obs.resample("ME").mean()
    monthly_corr = mod_monthly.corr(obs_monthly) if len(mod_monthly) > 2 else np.nan

    return {
        "name": name,
        "n": len(aligned),
        "pearson_r": corr,
        "mae_mw": mae,
        "rmse_mw": rmse,
        "mbe_mw": mbe,
        "nrmse": nrmse,
        "monthly_r": monthly_corr,
        "obs_mean_mw": obs.mean(),
        "mod_mean_mw": mod.mean(),
        "obs_annual_twh": obs.sum() / 1e6,
        "mod_annual_twh": mod.sum() / 1e6,
    }


# ─────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────

def plot_timeseries_comparison(
    scenario_series: dict[str, pd.Series],
    obs: pd.Series,
    *,
    country: str,
    wind_type: str,
    outpath: Path,
    scenario_colors: dict[str, str],
    resample: str = "1W",
):
    """Weekly-smoothed time series comparison for N scenarios."""
    fig, ax = plt.subplots(figsize=(FULL_WIDTH, 4.5 * cm))

    obs_r = obs.resample(resample).mean() / 1e3
    ax.plot(obs_r.index, obs_r.values, label="ENTSO-E observed",
            color=COL_OBS, linewidth=1.2)

    for label, series in scenario_series.items():
        sr = series.resample(resample).mean() / 1e3
        ax.plot(sr.index, sr.values, label=label,
                color=scenario_colors[label], linewidth=1.0)

    ax.set_ylabel("Generation [GW]")
    ax.set_title(f"{country} \u2014 {wind_type} wind ({resample} mean)")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=6,
        ncol=len(scenario_series) + 1,
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    format_axes_standard(fig)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_bars(
    scenario_series: dict[str, pd.Series],
    obs: pd.Series,
    *,
    country: str,
    wind_type: str,
    outpath: Path,
    scenario_colors: dict[str, str],
):
    """Monthly generation bar chart for N scenarios."""
    obs_m = obs.resample("ME").sum() / 1e6
    months = obs_m.index.month
    x = np.arange(len(months))

    n_bars = 1 + len(scenario_series)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(x - (n_bars - 1) * width / 2, obs_m.values, width,
           label="Observed", color=COL_OBS, alpha=0.7)

    for i, (label, series) in enumerate(scenario_series.items(), start=1):
        m = series.resample("ME").sum() / 1e6
        offset = x - (n_bars - 1) * width / 2 + i * width
        ax.bar(offset, m.values, width, label=label,
               color=scenario_colors[label])

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in months])
    ax.set_xlabel("Month")
    ax.set_ylabel("Generation [TWh]")
    ax.set_title(f"{country} \u2014 {wind_type} wind monthly generation")
    ax.legend(frameon=False, fontsize=6)

    format_axes_standard(fig)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_scatter_monthly(
    all_monthly: dict,
    scenario_labels: list[str],
    *,
    wind_type: str,
    outpath: Path,
    scenario_colors: dict[str, str],
):
    """Scatter plot of monthly generation — one panel per scenario."""
    n_cols = len(scenario_labels)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4),
                              sharex=True, sharey=True)
    if n_cols == 1:
        axes = [axes]

    for ax, label in zip(axes, scenario_labels):
        obs_vals, mod_vals = [], []
        for key, data in all_monthly.items():
            if key[1] != wind_type:
                continue
            if label not in data or "obs" not in data:
                continue
            obs_vals.extend(data["obs"].values)
            mod_vals.extend(data[label].values)

        if obs_vals:
            obs_arr = np.array(obs_vals)
            mod_arr = np.array(mod_vals)
            ax.scatter(obs_arr, mod_arr, s=12, alpha=0.6,
                       color=scenario_colors[label], edgecolors="none")

            vmin = min(obs_arr.min(), mod_arr.min()) * 0.9
            vmax = max(obs_arr.max(), mod_arr.max()) * 1.1
            ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)

            corr = np.corrcoef(obs_arr, mod_arr)[0, 1]
            ax.set_title(f"{label} (r={corr:.3f})")

        ax.set_xlabel("Observed [TWh/month]")
        ax.set_ylabel("Modelled [TWh/month]")

    fig.suptitle(f"{wind_type.capitalize()} wind \u2014 monthly generation scatter", y=1.02)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(
    metrics_df: pd.DataFrame,
    scenario_labels: list[str],
    *,
    outpath: Path,
    scenario_colors: dict[str, str],
):
    """Summary bar chart of correlation and NRMSE by country for N scenarios."""
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 6.0 * cm))

    legend_handles = []
    legend_labels_list = []

    for ax, metric, ylabel, title in [
        (axes[0], "pearson_r", "Pearson r", "Temporal correlation"),
        (axes[1], "nrmse", "NRMSE", "Normalised RMSE"),
    ]:
        n_scenarios = len(scenario_labels)
        width = 0.8 / n_scenarios

        for wt_idx, wind_type in enumerate(["onshore", "offshore"]):
            subset = metrics_df[metrics_df["wind_type"] == wind_type]
            if subset.empty:
                continue

            countries = sorted(subset["country"].unique())
            x = np.arange(len(countries))

            for s_idx, label in enumerate(scenario_labels):
                vals = []
                for c in countries:
                    row = subset[(subset["country"] == c) & (subset["scenario"] == label)]
                    vals.append(row[metric].values[0] if len(row) > 0 else 0)

                base_x = x + wt_idx * (len(countries) + 1)
                offset = base_x - (n_scenarios - 1) * width / 2 + s_idx * width
                bars = ax.bar(
                    offset, vals, width,
                    color=scenario_colors[label],
                    alpha=0.8 if wt_idx == 0 else 0.5,
                )
                # Collect legend handles from first panel only
                if ax is axes[0] and len(countries) > 0:
                    legend_handles.append(bars)
                    legend_labels_list.append(
                        f"{label} ({'on' if wt_idx == 0 else 'off'}shore)"
                    )

            # Country labels
            tick_positions = list(x) + list(x + len(countries) + 1)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(countries + countries, rotation=0)

        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Shared legend below both panels
    fig.legend(
        legend_handles, legend_labels_list,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
        fontsize=5,
        ncol=min(len(legend_labels_list), 3),
    )

    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bias_improvement(
    metrics_df: pd.DataFrame,
    scenario_labels: list[str],
    baseline_label: str,
    *,
    outpath: Path,
    scenario_colors: dict[str, str],
):
    """NRMSE improvement scatter: each non-baseline scenario vs baseline."""
    non_baseline = [l for l in scenario_labels if l != baseline_label]
    if not non_baseline:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    markers_wt = {"onshore": "o", "offshore": "s"}

    for label in non_baseline:
        for wind_type, marker in markers_wt.items():
            subset = metrics_df[metrics_df["wind_type"] == wind_type]
            if subset.empty:
                continue

            countries = sorted(subset["country"].unique())
            for country in countries:
                row_base = subset[
                    (subset["country"] == country) & (subset["scenario"] == baseline_label)
                ]
                row_scen = subset[
                    (subset["country"] == country) & (subset["scenario"] == label)
                ]
                if row_base.empty or row_scen.empty:
                    continue

                nrmse_base = row_base["nrmse"].values[0]
                nrmse_scen = row_scen["nrmse"].values[0]

                ax.scatter(
                    nrmse_base, nrmse_scen, s=40, marker=marker,
                    color=scenario_colors[label],
                    edgecolors="black", linewidth=0.5, zorder=3,
                )
                ax.annotate(
                    country, (nrmse_base, nrmse_scen),
                    textcoords="offset points", xytext=(4, 4), fontsize=6,
                )

    lims = ax.get_xlim()
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel(f"NRMSE \u2014 {baseline_label}")
    ax.set_ylabel("NRMSE \u2014 Corrected scenario")
    ax.set_title("Improvement from bias correction")

    # Build legend
    handles = []
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    handles.append(mlines.Line2D([], [], color="k", linestyle="--", linewidth=0.8, label="1:1 line"))
    for label in non_baseline:
        handles.append(mpatches.Patch(color=scenario_colors[label], label=label))
    for wind_type, marker in markers_wt.items():
        handles.append(mlines.Line2D([], [], color="gray", marker=marker, linestyle="None",
                                      markersize=6, label=wind_type.capitalize()))
    ax.legend(handles=handles, frameon=False, fontsize=6)

    format_axes_standard(fig)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", default=None,
                    help="PyPSA-Eur config YAML (for ENTSO-E API key)")
    ap.add_argument("--entsoe-csv", default=None,
                    help="Pre-built ENTSO-E generation CSV")
    ap.add_argument(
        "--scenario", action="append", nargs=2,
        metavar=("LABEL", "RESOURCE_DIR"),
        help="A labelled scenario: --scenario 'IDW corrected' path/to/biasidw "
             "(repeat for each scenario)",
    )
    ap.add_argument(
        "--baseline", default=None,
        help="Label of the baseline scenario (for delta calculations). "
             "Defaults to the first --scenario.",
    )
    ap.add_argument("--out", default="plots/validation", help="Output directory")
    ap.add_argument("--countries", default=",".join(COUNTRIES))
    args = ap.parse_args()

    if not args.scenario or len(args.scenario) < 1:
        ap.error("At least one --scenario LABEL DIR is required")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    countries = [c.strip() for c in args.countries.split(",")]

    # Parse scenarios
    scenarios = [(label, Path(res_dir)) for label, res_dir in args.scenario]
    scenario_labels = [label for label, _ in scenarios]
    baseline_label = args.baseline or scenario_labels[0]

    # Assign colors
    scenario_colors = {
        label: SCENARIO_PALETTE[i % len(SCENARIO_PALETTE)]
        for i, label in enumerate(scenario_labels)
    }

    print("=" * 60)
    print("Wind generation validation: modelled vs ENTSO-E observed")
    print(f"Scenarios: {scenario_labels}")
    print(f"Baseline:  {baseline_label}")
    print("=" * 60)

    # ── 1. Load profiles ──
    print("\n[1] Loading wind profiles...")
    all_profiles = {}
    for label, res_dir in scenarios:
        all_profiles[label] = load_profiles(res_dir)
        print(f"  {label:>25s}: {list(all_profiles[label].keys())} from {res_dir.name}")

    cf_country: dict[tuple[str, str], pd.DataFrame] = {}
    for label in scenario_labels:
        for wind_type in ["onshore", "offshore"]:
            cf_country[(label, wind_type)] = aggregate_profiles_to_country(
                all_profiles[label], wind_type
            )
    print(f"  Country-level CFs aggregated for {len(cf_country)} (scenario, wind_type) combos")

    # ── 2. Load ENTSO-E data ──
    print("\n[2] Loading ENTSO-E generation data...")
    api_key = None
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        api_key = cfg.get("private", {}).get("keys", {}).get("entsoe_api")

    entsoe_csv = Path(args.entsoe_csv) if args.entsoe_csv else None
    entsoe = load_entsoe_generation(
        csv_path=entsoe_csv, api_key=api_key, countries=countries
    )
    print(f"  Available: {list(entsoe.columns)}")
    print(f"  Time range: {entsoe.index[0]} \u2014 {entsoe.index[-1]}")

    # ── 3. Compute metrics ──
    print("\n[3] Computing modelled generation and metrics...")
    all_metrics = []
    all_monthly: dict[tuple[str, str], dict] = {}

    for wind_type in ["onshore", "offshore"]:
        cap_dict = INSTALLED_CAPACITY_2023[wind_type]

        for country in countries:
            if country not in cap_dict or cap_dict[country] == 0:
                continue

            installed_mw = cap_dict[country]

            if (country, wind_type) not in entsoe.columns:
                print(f"  [SKIP] No ENTSO-E data for {country} {wind_type}")
                continue
            gen_obs = entsoe[(country, wind_type)]

            # Compute for each scenario
            scenario_gen = {}
            for label in scenario_labels:
                cf = cf_country.get((label, wind_type))
                if cf is None or country not in cf.columns:
                    continue
                scenario_gen[label] = cf[country] * installed_mw

            if not scenario_gen:
                continue

            # Find common timestamps across all scenarios and observations
            common_idx = gen_obs.dropna().index
            for series in scenario_gen.values():
                common_idx = common_idx.intersection(series.dropna().index)

            if len(common_idx) < 100:
                print(
                    f"  [SKIP] Too few overlapping timestamps for "
                    f"{country} {wind_type}: {len(common_idx)}"
                )
                continue

            gen_obs_a = gen_obs.loc[common_idx]

            # Monthly aggregates for scatter plots
            obs_monthly = gen_obs_a.resample("ME").sum() / 1e6
            all_monthly.setdefault((country, wind_type), {})["obs"] = obs_monthly

            # Metrics and aligned series for each scenario
            scenario_gen_aligned = {}
            metric_strs = []
            for label in scenario_labels:
                if label not in scenario_gen:
                    continue
                gen_a = scenario_gen[label].loc[common_idx]
                scenario_gen_aligned[label] = gen_a

                m = compute_metrics(gen_a, gen_obs_a, f"{country}_{wind_type}_{label}")
                m.update({
                    "country": country, "wind_type": wind_type,
                    "scenario": label, "installed_mw": installed_mw,
                })
                all_metrics.append(m)

                mod_monthly = gen_a.resample("ME").sum() / 1e6
                all_monthly[(country, wind_type)][label] = mod_monthly

                metric_strs.append(
                    f"r={m.get('pearson_r', 0):.3f} NRMSE={m.get('nrmse', 0):.3f}"
                )

            labels_str = "  ".join(
                f"{label}: {s}" for label, s in zip(scenario_gen_aligned.keys(), metric_strs)
            )
            print(f"  {country} {wind_type:>8s}: {labels_str}")

            # Per-country plots
            plot_timeseries_comparison(
                scenario_gen_aligned, gen_obs_a,
                country=country, wind_type=wind_type,
                outpath=outdir / f"ts_{country}_{wind_type}.png",
                scenario_colors=scenario_colors,
            )
            plot_monthly_bars(
                scenario_gen_aligned, gen_obs_a,
                country=country, wind_type=wind_type,
                outpath=outdir / f"monthly_{country}_{wind_type}.png",
                scenario_colors=scenario_colors,
            )

    # ── 4. Summary outputs ──
    print("\n[4] Generating summary outputs...")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(outdir / "validation_metrics.csv", index=False)
    print(f"  Metrics saved to {outdir / 'validation_metrics.csv'}")

    if not metrics_df.empty:
        print("\n  === SUMMARY ===")
        for wt in ["onshore", "offshore"]:
            subset = metrics_df[metrics_df["wind_type"] == wt]
            if subset.empty:
                continue
            print(f"\n  {wt.upper()} WIND:")
            for _, row in subset.iterrows():
                print(
                    f"    {row['country']:>2s} ({row['scenario']:>25s}): "
                    f"r={row.get('pearson_r', 0):.3f}  "
                    f"NRMSE={row.get('nrmse', 0):.3f}  "
                    f"MBE={row.get('mbe_mw', 0):+.0f} MW  "
                    f"obs={row.get('obs_annual_twh', 0):.1f} TWh  "
                    f"mod={row.get('mod_annual_twh', 0):.1f} TWh"
                )

        # Summary plots
        plot_metrics_summary(
            metrics_df, scenario_labels,
            outpath=outdir / "metrics_summary.png",
            scenario_colors=scenario_colors,
        )
        plot_bias_improvement(
            metrics_df, scenario_labels, baseline_label,
            outpath=outdir / "bias_improvement.png",
            scenario_colors=scenario_colors,
        )

        for wind_type in ["onshore", "offshore"]:
            if any(k[1] == wind_type for k in all_monthly):
                plot_scatter_monthly(
                    all_monthly, scenario_labels,
                    wind_type=wind_type,
                    outpath=outdir / f"scatter_monthly_{wind_type}.png",
                    scenario_colors=scenario_colors,
                )

    print(f"\nAll outputs saved to: {outdir}")
    print("Done.")


if __name__ == "__main__":
    main()

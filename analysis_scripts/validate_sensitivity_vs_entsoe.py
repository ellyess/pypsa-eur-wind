#!/usr/bin/env python3
"""
Validate tier-2 sensitivity runs against ENTSO-E 2023 observed wind generation.

Compares **capacity factors** (not absolute MW) to isolate wind resource model
quality from the 2030-vs-2023 capacity mismatch.

- Model CF: capacity-weighted mean p_max_pu per country (available resource,
  before curtailment).
- Observed CF: ENTSO-E generation / ENTSO-E installed capacity (mid-2023).

Both are resampled to 6 h to match model timesteps.

Produces:
  1. Per-country validation metrics (Pearson r, NRMSE, MBE) as CSV
  2. Multi-panel heatmaps: metric vs scenario x country
  3. Time-series CF comparison plots for a reference scenario
  4. Summary bar charts: metrics by bias correction, wake model, resolution
  5. Monthly mean CF comparison

Run from the repo root:
    python analysis_scripts/validate_sensitivity_vs_entsoe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from plotting_style import thesis_plot_style, format_axes_standard
from thesis_colors import THESIS_COLORS, THESIS_LABELS, label as get_label
from network_utils import (
    load_network,
    build_manifest,
    scenario_key,
    snapshot_weights,
    bus_country,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_ROOT = Path("results")
RESULTS_PREFIX = "thesis-sensitivity-2030-30-europe-dominant-6h"
RESULTS_GLOB = f"{RESULTS_PREFIX}/*/postnetworks/*.nc"
ENTSOE_CSV = Path("data/entsoe_generation_2023.csv")
OUTDIR = Path("plots/sensitivity/validation_entsoe")

TIER2_COUNTRIES = [
    "AL", "AT", "BA", "BE", "BG", "CH", "CZ", "DE", "DK", "ES",
    "FR", "GB", "GR", "HR", "IE", "IT", "LU", "ME", "MK", "NL",
    "NO", "PT", "RS", "SE", "SI",
]

TECHS = {
    "onshore": {"model_carriers": ["onwind"], "entsoe_carrier": "Wind Onshore"},
    "offshore": {
        "model_carriers": ["offwind-ac", "offwind-dc", "offwind-float"],
        "entsoe_carrier": "Wind Offshore",
    },
}

# ENTSO-E installed capacity mid-2023 (MW), queried via entsoe-py
# Source: ENTSO-E Transparency Platform, query_installed_generation_capacity()
INSTALLED_CAPACITY_2023 = {
    "onshore": {
        "AT": 3568.8, "BA": 134.6, "BE": 3053.2, "BG": 705.0,
        "CZ": 339.0, "DE": 57589.6, "DK": 4710.5, "ES": 29320.4,
        "FR": 20841.9, "GR": 4547.0, "HR": 981.0, "IE": 1918.7,
        "IT": 11204.0, "LU": 152.0, "ME": 118.0, "MK": 37.0,
        "NL": 6757.0, "NO": 5129.8, "PT": 5328.4, "RS": 533.3,
        "SE": 14700.0, "SI": 2.4,
    },
    "offshore": {
        "BE": 2262.1, "DE": 8128.8, "DK": 2305.7, "FR": 493.5,
        "IT": 30.0, "NL": 4739.0, "PT": 25.0,
    },
}


# ---------------------------------------------------------------------------
# ENTSO-E loading
# ---------------------------------------------------------------------------

def load_entsoe(path: Path) -> pd.DataFrame:
    """Load ENTSO-E generation CSV with multi-row header."""
    raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    raw.index.name = "timestamp"
    raw.columns = pd.MultiIndex.from_tuples(
        [(c.strip(), k.strip()) for c, k in raw.columns],
        names=["country", "carrier"],
    )
    return raw.apply(pd.to_numeric, errors="coerce")


def entsoe_cf(
    entsoe: pd.DataFrame, country: str, carrier: str, capacity_mw: float,
) -> pd.Series:
    """Observed capacity factor: ENTSO-E generation / installed capacity."""
    if (country, carrier) not in entsoe.columns:
        return pd.Series(dtype=float)
    gen = entsoe[(country, carrier)]
    return gen / capacity_mw


# ---------------------------------------------------------------------------
# Model extraction — capacity-weighted available CF
# ---------------------------------------------------------------------------

def model_cf_by_country_tech(
    n, countries: list[str],
) -> dict[tuple[str, str], pd.Series]:
    """Capacity-weighted mean p_max_pu (available CF) per (country, tech).

    This reflects the wind resource model output (ERA5 + corrections),
    independent of optimiser curtailment decisions.
    """
    result = {}
    g = n.generators

    for tech_label, cfg in TECHS.items():
        carriers = cfg["model_carriers"]
        mask = g.carrier.isin(carriers)
        if not mask.any():
            continue
        idx = g.index[mask]
        gen_cc = g.loc[idx, "bus"].map(bus_country)

        # Get capacity for weighting (p_nom_opt if available, else p_nom)
        if "p_nom_opt" in g.columns:
            cap = g.loc[idx, "p_nom_opt"].fillna(g.loc[idx, "p_nom"])
        else:
            cap = g.loc[idx, "p_nom"]
        cap = cap.fillna(0.0)

        # Get p_max_pu time series
        try:
            pmax_pu = n.generators_t.p_max_pu[idx]
        except KeyError:
            continue

        for cc in countries:
            cc_mask = gen_cc == cc
            cc_gens = idx[cc_mask]
            cc_cap = cap[cc_mask]

            if len(cc_gens) == 0 or cc_cap.sum() <= 0:
                result[(cc, tech_label)] = pd.Series(np.nan, index=n.snapshots)
                continue

            # Capacity-weighted mean CF
            weighted = pmax_pu[cc_gens].multiply(cc_cap, axis=1)
            cf = weighted.sum(axis=1) / cc_cap.sum()
            result[(cc, tech_label)] = cf

    return result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_cf_metrics(obs_cf: pd.Series, mod_cf: pd.Series) -> dict:
    """Compute validation metrics between observed and modelled CF time series."""
    mask = obs_cf.notna() & mod_cf.notna() & (obs_cf >= 0) & (mod_cf >= 0)
    if mask.sum() < 10:
        return {k: np.nan for k in [
            "pearson_r", "nrmse", "mbe_cf", "mbe_pct", "mae_cf",
            "obs_mean_cf", "mod_mean_cf", "n_points",
        ]}

    o = obs_cf[mask].values.astype(float)
    m = mod_cf[mask].values.astype(float)

    r = float(np.corrcoef(o, m)[0, 1])
    diff = m - o
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    o_mean = float(np.mean(o))
    nrmse = rmse / o_mean if o_mean > 0 else np.nan
    mbe = float(np.mean(diff))
    mae = float(np.mean(np.abs(diff)))
    mbe_pct = 100.0 * mbe / o_mean if o_mean > 0 else np.nan

    return {
        "pearson_r": r,
        "nrmse": nrmse,
        "mbe_cf": mbe,
        "mbe_pct": mbe_pct,
        "mae_cf": mae,
        "obs_mean_cf": o_mean,
        "mod_mean_cf": float(np.mean(m)),
        "n_points": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    outpath: Path,
    cmap: str = "RdYlGn",
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = ".2f",
    higher_better: bool = True,
    valid_countries: list[str] | None = None,
) -> None:
    """Heatmap of a metric across scenarios (rows) and countries (cols)."""
    style = thesis_plot_style()
    cm = style["cm"]

    pivot = df.pivot_table(
        index="scenario_label", columns="country", values=metric_col, aggfunc="mean",
    )
    if valid_countries is not None:
        col_order = [c for c in valid_countries if c in pivot.columns]
    else:
        col_order = sorted(pivot.columns)
    pivot = pivot[col_order]

    fig, ax = plt.subplots(
        figsize=(max(12, len(col_order) * 1.2) * cm, max(5, len(pivot) * 0.8) * cm),
        dpi=600,
    )

    if not higher_better:
        cmap = cmap + "_r" if not cmap.endswith("_r") else cmap.replace("_r", "")

    im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, fontsize=7, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=6)

    for i in range(len(pivot)):
        for j in range(len(col_order)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=5.5)

    ax.set_title(title, fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_bar_by_dimension(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    outpath: Path,
) -> None:
    """Bar chart of mean metric grouped by bias correction, wake model, resolution."""
    style = thesis_plot_style()
    cm = style["cm"]

    fig, axes = plt.subplots(1, 3, figsize=(17.8 * cm, 6 * cm), dpi=600)

    # (a) By bias correction
    ax = axes[0]
    grp = df.groupby("bias")[metric_col].mean().sort_index()
    colors = ["#4D4D4D", "#D55E00", "#5DAE8B"]
    ax.bar(range(len(grp)), grp.values, color=colors[: len(grp)])
    ax.set_xticks(range(len(grp)))
    bias_labels = {"false": "None", "idw": "IDW", "uniform": "Uniform"}
    ax.set_xticklabels([bias_labels.get(k, k) for k in grp.index], fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_title("(a) By bias correction")
    ax.grid(True, axis="y", alpha=0.3)

    # (b) By wake model
    ax = axes[1]
    grp = df.groupby("wake")[metric_col].mean().sort_index()
    wake_labels = {"off": "No wake", "density": "Density-based"}
    ax.bar(range(len(grp)), grp.values, color=["#4D4D4D", "#2F4B7C"][: len(grp)])
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([wake_labels.get(k, k) for k in grp.index], fontsize=6)
    ax.set_title("(b) By wake model")
    ax.grid(True, axis="y", alpha=0.3)

    # (c) By resolution
    ax = axes[2]
    grp = df.groupby("resolution")[metric_col].mean().sort_values()
    ax.bar(range(len(grp)), grp.values, color="#666666")
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels([f"{int(r):,}" for r in grp.index], fontsize=6)
    ax.set_xlabel(r"$A_{region}^{max}$ [km$^2$]")
    ax.set_title("(c) By resolution")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_cf_timeseries(
    obs_cf: pd.Series,
    mod_cf: pd.Series,
    country: str,
    tech: str,
    scenario_label: str,
    outpath: Path,
) -> None:
    """Plot observed vs modelled CF time series."""
    style = thesis_plot_style()
    cm = style["cm"]

    fig, ax = plt.subplots(figsize=(17.8 * cm, 5 * cm), dpi=600)
    ax.plot(obs_cf.index, obs_cf.values, linewidth=0.4, alpha=0.7,
            label="ENTSO-E CF", color="#4D4D4D")
    ax.plot(mod_cf.index, mod_cf.values, linewidth=0.4, alpha=0.7,
            label="Model CF", color="#2F4B7C")
    ax.set_ylabel("Capacity factor")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"{country} {tech} - {scenario_label}")
    ax.legend(loc="upper right", frameon=False, fontsize=6)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_cf_comparison(
    monthly_records: list[dict],
    outpath: Path,
    valid_countries: list[str] | None = None,
) -> None:
    """Multi-panel monthly mean CF comparison for a reference scenario."""
    if not monthly_records:
        return

    style = thesis_plot_style()
    cm = style["cm"]
    mdf = pd.DataFrame(monthly_records)

    if valid_countries is not None:
        countries = [c for c in valid_countries if c in mdf["country"].unique()]
    else:
        countries = sorted(mdf["country"].unique())
    techs = [t for t in ["onshore", "offshore"] if t in mdf["tech"].unique()]

    ncols = len(countries)
    nrows = len(techs)
    if ncols == 0 or nrows == 0:
        return

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(min(17.8, 2.5 * ncols) * cm, 4 * nrows * cm),
        dpi=600, sharex=True,
    )
    if nrows == 1:
        axes = axes[np.newaxis, :]
    if ncols == 1:
        axes = axes[:, np.newaxis]

    for i, tech in enumerate(techs):
        for j, cc in enumerate(countries):
            ax = axes[i, j]
            sub = mdf[(mdf["country"] == cc) & (mdf["tech"] == tech)]
            if sub.empty:
                ax.set_axis_off()
                continue
            months = sub["month"].values
            ax.bar(months - 0.15, sub["obs_mean_cf"].values, width=0.3,
                   color="#4D4D4D", alpha=0.7, label="ENTSO-E")
            ax.bar(months + 0.15, sub["mod_mean_cf"].values, width=0.3,
                   color="#2F4B7C", alpha=0.7, label="Model")
            ax.set_ylim(0, 0.8)
            if i == 0:
                ax.set_title(cc, fontsize=6)
            if j == 0:
                ax.set_ylabel(f"{tech.capitalize()}\nCF", fontsize=6)
            ax.set_xlim(0.5, 12.5)
            ax.grid(True, axis="y", alpha=0.3)
            ax.tick_params(labelsize=5)

    axes[0, -1].legend(loc="upper right", frameon=False, fontsize=5)
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

    # --- Load ENTSO-E ---
    print(f"Loading ENTSO-E data from {ENTSOE_CSV}")
    entsoe = load_entsoe(ENTSOE_CSV)
    print(f"  ENTSO-E shape: {entsoe.shape}, period: {entsoe.index[0]} to {entsoe.index[-1]}")

    entsoe_countries = sorted(entsoe.columns.get_level_values("country").unique())
    VALID_COUNTRIES = [c for c in TIER2_COUNTRIES if c in entsoe_countries]
    print(f"  ENTSO-E countries: {entsoe_countries}")
    print(f"  Tier-2 countries with ENTSO-E data: {VALID_COUNTRIES}")

    # --- Build manifest ---
    print(f"\nBuilding manifest from {RESULTS_ROOT / RESULTS_GLOB}")
    manifest = build_manifest(RESULTS_ROOT, RESULTS_GLOB)
    print(f"  Found {len(manifest)} networks")
    print(f"  Resolutions: {sorted(manifest['resolution'].unique())}")
    print(f"  Bias types: {sorted(manifest['bias'].unique())}")
    print(f"  Wake models: {sorted(manifest['wake'].unique())}")

    # --- Compute CF validation metrics ---
    all_rows = []
    monthly_records_ref = []
    ref_scenario = None

    for _, mrow in manifest.iterrows():
        nc_path = Path(mrow["path"])
        res = int(mrow["resolution"])
        bias = mrow["bias"]
        wake = mrow["wake"]
        scen = scenario_key(bias, wake)
        scenario_label = f"{scen} (s={res:,})"

        print(f"\n  Processing: {mrow['scenario_folder']} -> {scenario_label}")
        n = load_network(nc_path)

        # Model available CF by country
        mod_cf = model_cf_by_country_tech(n, VALID_COUNTRIES)

        for cc in VALID_COUNTRIES:
            for tech_label, cfg in TECHS.items():
                entsoe_carrier = cfg["entsoe_carrier"]

                # Check installed capacity
                cap_mw = INSTALLED_CAPACITY_2023.get(tech_label, {}).get(cc, 0.0)
                if cap_mw <= 0:
                    continue

                # Observed CF
                obs_cf_full = entsoe_cf(entsoe, cc, entsoe_carrier, cap_mw)
                if obs_cf_full.empty or not isinstance(obs_cf_full.index, pd.DatetimeIndex):
                    continue

                # Resample observed to 6h
                obs_cf_6h = obs_cf_full.resample("6h").mean()

                # Model CF
                mod_cf_series = mod_cf.get((cc, tech_label))
                if mod_cf_series is None:
                    continue

                # Align timestamps
                common_idx = obs_cf_6h.index.intersection(mod_cf_series.index)
                if len(common_idx) < 10:
                    continue

                obs_aligned = obs_cf_6h.loc[common_idx]
                mod_aligned = mod_cf_series.loc[common_idx]

                metrics = compute_cf_metrics(obs_aligned, mod_aligned)
                metrics.update({
                    "country": cc,
                    "tech": tech_label,
                    "resolution": res,
                    "bias": bias,
                    "wake": wake,
                    "scenario": scen,
                    "scenario_label": scenario_label,
                    "scenario_folder": mrow["scenario_folder"],
                    "installed_capacity_mw": cap_mw,
                })
                all_rows.append(metrics)

                # Monthly records for reference scenario
                if scen == "base" and res == 10000 and ref_scenario is None:
                    for month in range(1, 13):
                        m_mask = common_idx.month == month
                        if m_mask.sum() == 0:
                            continue
                        monthly_records_ref.append({
                            "country": cc,
                            "tech": tech_label,
                            "month": month,
                            "obs_mean_cf": float(obs_aligned[m_mask].mean()),
                            "mod_mean_cf": float(mod_aligned[m_mask].mean()),
                        })

        if scen == "base" and res == 10000:
            ref_scenario = scenario_label

    # --- Results ---
    results = pd.DataFrame(all_rows)
    results.to_csv(OUTDIR / "validation_metrics_cf.csv", index=False)
    print(f"\n  Saved metrics to {OUTDIR / 'validation_metrics_cf.csv'}")
    print(f"  Total rows: {len(results)}")

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("CAPACITY FACTOR VALIDATION SUMMARY (mean across countries)")
    print("=" * 80)
    for tech_label in ["onshore", "offshore"]:
        sub = results[results["tech"] == tech_label]
        if sub.empty:
            continue
        print(f"\n--- {tech_label.upper()} ---")
        grp = sub.groupby(["scenario", "resolution"])[
            ["pearson_r", "nrmse", "mbe_pct", "obs_mean_cf", "mod_mean_cf"]
        ].mean()
        print(grp.to_string(float_format=lambda x: f"{x:.3f}"))

    # --- Plots ---

    # 1. Heatmaps
    for tech_label in ["onshore", "offshore"]:
        sub = results[results["tech"] == tech_label]
        if sub.empty:
            continue
        for metric, title_sfx, cmap, hb, vmin, vmax, fmt in [
            ("pearson_r", "Pearson r", "RdYlGn", True, 0.5, 1.0, ".2f"),
            ("nrmse", "NRMSE", "RdYlGn_r", False, None, None, ".2f"),
            ("mbe_pct", "MBE [%]", "RdBu_r", True, -50, 50, ".0f"),
        ]:
            plot_metric_heatmap(
                sub, metric,
                title=f"{tech_label.capitalize()} wind CF: {title_sfx}",
                outpath=OUTDIR / f"heatmap_cf_{tech_label}_{metric}.png",
                cmap=cmap, vmin=vmin, vmax=vmax, fmt=fmt,
                higher_better=hb, valid_countries=VALID_COUNTRIES,
            )

    # 2. Bar charts by dimension
    for tech_label in ["onshore", "offshore"]:
        sub = results[results["tech"] == tech_label]
        if sub.empty:
            continue
        for metric, ylabel in [
            ("pearson_r", "Pearson r (mean)"),
            ("nrmse", "NRMSE (mean)"),
            ("mbe_pct", "MBE [%] (mean)"),
        ]:
            plot_bar_by_dimension(
                sub, metric, ylabel,
                outpath=OUTDIR / f"bars_cf_{tech_label}_{metric}.png",
            )

    # 3. Time-series CF comparison for reference scenario
    ref_rows = results[(results["scenario"] == "base") & (results["resolution"] == 10000)]
    if not ref_rows.empty:
        ts_dir = OUTDIR / "timeseries_cf"
        ts_dir.mkdir(exist_ok=True)

        ref_folder = ref_rows.iloc[0]["scenario_folder"]
        ref_path = list(
            RESULTS_ROOT.glob(f"{RESULTS_PREFIX}/{ref_folder}/postnetworks/*.nc")
        )
        if ref_path:
            n_ref = load_network(ref_path[0])
            mod_cf_ref = model_cf_by_country_tech(n_ref, VALID_COUNTRIES)

            for cc in VALID_COUNTRIES:
                for tech_label, cfg in TECHS.items():
                    cap_mw = INSTALLED_CAPACITY_2023.get(tech_label, {}).get(cc, 0.0)
                    if cap_mw <= 0:
                        continue
                    obs_series = entsoe_cf(entsoe, cc, cfg["entsoe_carrier"], cap_mw)
                    if obs_series.empty or not isinstance(obs_series.index, pd.DatetimeIndex):
                        continue
                    obs_6h = obs_series.resample("6h").mean()
                    mod_s = mod_cf_ref.get((cc, tech_label))
                    if mod_s is None:
                        continue
                    common = obs_6h.index.intersection(mod_s.index)
                    if len(common) < 10:
                        continue
                    plot_cf_timeseries(
                        obs_6h.loc[common], mod_s.loc[common],
                        cc, tech_label, "Baseline (s=10,000)",
                        ts_dir / f"ts_cf_{cc}_{tech_label}.png",
                    )

    # 4. Monthly CF comparison
    plot_monthly_cf_comparison(
        monthly_records_ref,
        OUTDIR / "monthly_cf_comparison_baseline.png",
        valid_countries=VALID_COUNTRIES,
    )

    print(f"\nAll outputs saved to {OUTDIR}")


if __name__ == "__main__":
    main()

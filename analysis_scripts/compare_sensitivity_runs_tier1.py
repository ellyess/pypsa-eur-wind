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
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting_style import thesis_plot_style, apply_spatial_resolution_axis
from thesis_colors import THESIS_COLORS, get_color_cycle, THESIS_LABELS

# Apply thesis-wide plotting style
_style = thesis_plot_style()
cm, lw, ms, dpi = _style['cm'], _style['lw'], _style['ms'], _style['dpi']

try:
    import pypsa
except ImportError as e:
    raise SystemExit("pypsa is required: pip install pypsa") from e


# ----------------------------
# Configuration
# ----------------------------

WAKE_ALIASES = {
    # folder prefix -> normalized wake label
    "base": "off",
    # your density-based variant(s)
    "new_more": "density",
}

INVERT_RESOLUTION_AXIS = False  # apply_spatial_resolution_axis handles inversion for coarse->fine

SCEN_ORDER = ["base", "bias", "wake", "bias+wake"]
TECH_ORDER = ["onwind", "offwind"]


# ----------------------------
# Parsing utilities
# ----------------------------

_RE_SCENARIO = re.compile(
    r"(?P<wakeprefix>[^/]+)-s(?P<res>\d+)-bias(?P<bias>True|False)",
    re.IGNORECASE,
)


def parse_from_path(nc_path: Path) -> dict:
    """Parse scenario metadata from a network file path."""
    parts = nc_path.parts
    scenario_folder = None
    for p in reversed(parts):
        if "-s" in p and "bias" in p:
            scenario_folder = p
            break
    if scenario_folder is None:
        raise ValueError(f"Could not find scenario folder in path: {nc_path}")

    m = _RE_SCENARIO.search(scenario_folder)
    if not m:
        raise ValueError(f"Scenario folder didn't match pattern: {scenario_folder}")

    wakeprefix = m.group("wakeprefix")
    res = int(m.group("res"))
    bias = m.group("bias").lower() == "true"
    wake = WAKE_ALIASES.get(wakeprefix.lower(), wakeprefix.lower())

    return {
        "path": str(nc_path),
        "resolution": res,
        "bias": bias,
        "wake": wake,  # expected "off" or "density"
        "wakeprefix": wakeprefix,
        "scenario_folder": scenario_folder,
    }


# ----------------------------
# Network loading & selectors
# ----------------------------


def load_network(path: str) -> "pypsa.Network":
    return pypsa.Network(path)


def gen_idx(n: "pypsa.Network", tech: str) -> pd.Index:
    carr = n.generators.carrier.astype(str).str.lower()
    if tech == "offwind":
        return n.generators.index[carr.str.contains("offwind")]
    if tech == "onwind":
        return n.generators.index[carr.eq("onwind")]
    raise ValueError(f"Unknown tech: {tech}")


def scenario_key(bias: bool, wake: str) -> str:
    wake_is_off = (wake == "off")
    if (not bias) and wake_is_off:
        return "base"
    if bias and wake_is_off:
        return "bias"
    if (not bias) and (not wake_is_off):
        return "wake"
    return "bias+wake"


# ----------------------------
# Metric extraction
# ----------------------------


def wind_capacity_gw(n: "pypsa.Network", tech: str) -> float:
    idx = gen_idx(n, tech)
    if len(idx) == 0:
        return 0.0

    g = n.generators.loc[idx]
    p_nom_opt = g["p_nom_opt"] if "p_nom_opt" in g.columns else pd.Series(index=g.index, dtype=float)
    p_nom = g["p_nom"] if "p_nom" in g.columns else pd.Series(index=g.index, dtype=float)

    # Use p_nom_opt where present; fall back to p_nom
    cap_mw = p_nom_opt.where(p_nom_opt.notna(), p_nom).fillna(0.0).sum()
    return float(cap_mw) / 1e3  # MW -> GW


def wind_curtailment_frac(n: "pypsa.Network", tech: str) -> float:
    idx = gen_idx(n, tech)
    if len(idx) == 0:
        return float("nan")

    try:
        p = n.generators_t.p[idx]
        p_max_pu = n.generators_t.p_max_pu[idx]
    except Exception:
        return float("nan")

    g = n.generators.loc[idx]
    if "p_nom_opt" in g.columns and g["p_nom_opt"].notna().any():
        p_nom = g["p_nom_opt"].fillna(g.get("p_nom", 0.0))
    else:
        p_nom = g.get("p_nom", 0.0)

    denom = float(p_nom.sum())
    if denom <= 0:
        return float("nan")

    potential = p_max_pu.multiply(p_nom, axis=1)
    curtailed = (potential - p).clip(lower=0.0).sum().sum()
    pot = potential.sum().sum()
    return float(curtailed / pot) if pot > 0 else float("nan")


def transmission_expansion_twkm(n: "pypsa.Network") -> float:
    """Length-weighted expansion proxy: (MW*km)/1e6 = TW*km."""
    total_mw_km = 0.0

    if hasattr(n, "lines") and (not n.lines.empty) and ("s_nom_opt" in n.lines.columns):
        ln = n.lines
        base = ln.get("s_nom", pd.Series(0.0, index=ln.index)).fillna(0.0)
        opt = ln["s_nom_opt"].fillna(base)
        delta = (opt - base).clip(lower=0.0)
        length = ln.get("length", pd.Series(0.0, index=ln.index)).fillna(0.0)
        total_mw_km += float((delta * length).sum())

    if hasattr(n, "links") and (not n.links.empty) and ("p_nom_opt" in n.links.columns):
        lk = n.links
        base = lk.get("p_nom", pd.Series(0.0, index=lk.index)).fillna(0.0)
        opt = lk["p_nom_opt"].fillna(base)
        delta = (opt - base).clip(lower=0.0)
        length = lk.get("length", pd.Series(0.0, index=lk.index)).fillna(0.0)
        total_mw_km += float((delta * length).sum())

    return total_mw_km / 1e6


def get_objective(n: "pypsa.Network") -> float:
    if hasattr(n, "objective") and (n.objective is not None):
        try:
            return float(n.objective)
        except Exception:
            pass
    warnings.warn("Objective not found; returning NaN.")
    return float("nan")


# ----------------------------
# CF time series + long table
# ----------------------------


def cf_timeseries(n: "pypsa.Network", tech: str) -> pd.DataFrame | None:
    """Return availability/dispatch/curtailment CF time series (system-aggregated) for a tech."""
    idx = gen_idx(n, tech)
    if len(idx) == 0:
        return None

    g = n.generators.loc[idx]
    if "p_nom_opt" in g.columns and g["p_nom_opt"].notna().any():
        p_nom = g["p_nom_opt"].fillna(g.get("p_nom", 0.0))
    else:
        p_nom = g.get("p_nom", 0.0)

    denom = float(p_nom.sum())
    if denom <= 0:
        return None

    try:
        p = n.generators_t.p[idx]
        p_max_pu = n.generators_t.p_max_pu[idx]
    except Exception:
        return None

    potential = p_max_pu.multiply(p_nom, axis=1)
    avail_cf = potential.sum(axis=1) / denom
    disp_cf = p.sum(axis=1) / denom
    curt_cf = (potential.sum(axis=1) - p.sum(axis=1)).clip(lower=0.0) / denom
    return pd.DataFrame({"avail_cf": avail_cf, "disp_cf": disp_cf, "curt_cf": curt_cf})


def build_cf_long_both(manifest_df: pd.DataFrame, keep_resolutions=(100000, 1000)) -> pd.DataFrame:
    """Long table with CF time series for onwind/offwind, tagged by scenario + resolution."""
    rows: list[pd.DataFrame] = []

    keep = set(keep_resolutions) if keep_resolutions is not None else None

    for r in manifest_df.itertuples(index=False):
        res = int(r.resolution)
        if keep is not None and res not in keep:
            continue

        n = load_network(r.path)
        scen = scenario_key(bool(r.bias), str(r.wake))

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
    out["scenario"] = pd.Categorical(out["scenario"], SCEN_ORDER, ordered=True)
    out["tech"] = pd.Categorical(out["tech"], TECH_ORDER, ordered=True)
    out["resolution"] = out["resolution"].astype(int)
    return out


# ----------------------------
# Plotting helpers
# ----------------------------


def plot_lines_vs_resolution(df: pd.DataFrame, ycol: str, ylabel: str, outpath: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    res = sorted(df["resolution"].unique())
    if INVERT_RESOLUTION_AXIS:
        res = list(reversed(res))

    for scen in SCEN_ORDER:
        dd = df[df["scenario"] == scen].set_index("resolution").reindex(res)
        ax.plot(res, dd[ycol].values, marker="o", linewidth=1.2, color=THESIS_COLORS.get(scen, None), label=THESIS_LABELS.get(scen, scen))
        apply_spatial_resolution_axis(ax)
        ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_capacity_two_panel(df: pd.DataFrame, outpath: Path) -> None:
    """Two-panel: onwind vs offwind capacity vs resolution."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.1), dpi=300, sharex=True)

    res = sorted(df["resolution"].unique())
    if INVERT_RESOLUTION_AXIS:
        res = list(reversed(res))

    for ax, tech in zip(axes, TECH_ORDER):
        ax.set_prop_cycle(color=get_color_cycle(SCEN_ORDER))
        for scen in SCEN_ORDER:
            dd = df[df["scenario"] == scen].set_index("resolution").reindex(res)
            ax.plot(res, dd[f"{tech}_cap_gw"].values, marker="o", linewidth=1.2, color=THESIS_COLORS.get(scen, None), label=THESIS_LABELS.get(scen, scen))
        ax.set_title(THESIS_LABELS.get(tech, tech), fontsize=9)
        ax.grid(True, alpha=0.3)
        apply_spatial_resolution_axis(ax)
        ax.set_ylabel("Capacity (GW)")

    axes[1].legend(frameon=False, fontsize=7)
    axes[0].invert_xaxis()
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_cf_2x2(
    cf_long: pd.DataFrame,
    outpath: Path,
    metric: str = "disp_cf",
    ylabel: str = "Dispatch CF",
    resolutions: tuple[int, int] = (100000, 1000),
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
            for scen in SCEN_ORDER:
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
    fig.savefig(outpath)
    plt.close(fig)


def plot_marginal_value_proxy(df: pd.DataFrame, outpath: Path, cap_col: str = "offwind_cap_gw") -> None:
    """Proxy: (Δ objective)/(Δ installed capacity), relative to baseline, vs resolution."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    res = sorted(df["resolution"].unique())
    if INVERT_RESOLUTION_AXIS:
        res = list(reversed(res))

    base = df[df["scenario"] == "base"].set_index("resolution")

    for scen in ["bias", "wake", "bias+wake"]:
        dd = df[df["scenario"] == scen].set_index("resolution")
        d_obj = (dd["objective"] - base["objective"]).reindex(res)
        d_cap = (dd[cap_col] - base[cap_col]).reindex(res)
        mv = (d_obj / d_cap).replace([np.inf, -np.inf], np.nan)
        ax.plot(res, mv.values, marker="o", linewidth=1.2, color=THESIS_COLORS.get(scen, None), label=THESIS_LABELS.get(scen, scen))
        apply_spatial_resolution_axis(ax)

        ax.set_ylabel(f"Δ objective / Δ {cap_col} (proxy units per GW)")
        ax.set_title("Marginal value proxy (relative to baseline)", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


def plot_interaction_wake_effect(df: pd.DataFrame, ycol: str, ylabel: str, outpath: Path, title: str) -> None:
    """Δ(wake) = metric(wake) - metric(off), split by bias on/off."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    res = sorted(df["resolution"].unique())
    if INVERT_RESOLUTION_AXIS:
        res = list(reversed(res))

    for bias in [False, True]:
        d_off = df[(df["bias"] == bias) & (df["wake"] == "off")].set_index("resolution")
        d_wk = df[(df["bias"] == bias) & (df["wake"] != "off")].set_index("resolution")
        dd = (d_wk[ycol] - d_off[ycol]).reindex(res)
        label_key = "bias" if bias else "base"
        label_text = "With Bias" if bias else "Without Bias"
        ax.plot(res, dd.values, marker="o", linewidth=1.2, color=THESIS_COLORS.get(label_key, None), label=label_text)
    
    apply_spatial_resolution_axis(ax)
    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# ----------------------------
# Main workflow
# ----------------------------


def build_manifest(results_root: Path, pattern: str) -> pd.DataFrame:
    """Build a manifest of network files to compare.

    For sector runs, solved networks are typically written under ``postnetworks/``.
    If the provided glob finds nothing, we also try a sensible fallback that swaps
    ``postnetworks`` <-> ``networks`` (useful if you re-use the script across runs).
    """
    nc_files = sorted(results_root.glob(pattern))
    if not nc_files:
        # fallback: swap common subdirs
        alt = None
        if "postnetworks" in pattern:
            alt = pattern.replace("postnetworks", "networks")
        elif "networks" in pattern:
            alt = pattern.replace("networks", "postnetworks")
        if alt:
            nc_files = sorted(results_root.glob(alt))
            if nc_files:
                warnings.warn(f"No files for glob {pattern!r}; using fallback {alt!r} instead.")
                pattern = alt
    if not nc_files:
        raise SystemExit(f"No networks found under {results_root} with pattern {pattern!r}")
    recs = [parse_from_path(p) for p in nc_files]
    return pd.DataFrame(recs)


def compute_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        n = load_network(r["path"])
        rows.append(
            {
                **r.to_dict(),
                "scenario": scenario_key(bool(r["bias"]), str(r["wake"])),
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
    args = ap.parse_args()

    results_root = Path(args.results_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(results_root, args.glob)
    # Your Tier-1 set
    manifest = manifest[manifest["wake"].isin(["off", "density"])].copy()

    # Validate completeness (nice warning, not fatal)
    expected_res = sorted(manifest["resolution"].unique())
    expected = {(res, bias, wake) for res in expected_res for bias in [False, True] for wake in ["off", "density"]}
    found = {(int(r.resolution), bool(r.bias), str(r.wake)) for r in manifest.itertuples()}
    missing = sorted(expected - found)
    if missing:
        warnings.warn(f"Missing {len(missing)} combinations (resolution, bias, wake). First few: {missing[:10]}")

    if args.write_manifest:
        manifest.to_csv(outdir / "manifest.csv", index=False)

    # CF distributions (both techs)
    cf_long = build_cf_long_both(manifest, keep_resolutions=(args.coarse, args.fine))
    cf_long.to_csv(outdir / "tier1_cf_long_on_off.csv", index=False)
    print("[CF] counts:\n", cf_long.groupby(["tech", "resolution", "scenario"]).size())

    plot_cf_2x2(cf_long, outdir / "fig_cf_disp_2x2.png", metric="disp_cf", ylabel="Dispatch CF", resolutions=(args.coarse, args.fine))
    plot_cf_2x2(cf_long, outdir / "fig_cf_avail_2x2.png", metric="avail_cf", ylabel="Availability CF", resolutions=(args.coarse, args.fine))
    plot_cf_2x2(cf_long, outdir / "fig_cf_curt_2x2.png", metric="curt_cf", ylabel="Curtailment CF", resolutions=(args.coarse, args.fine))

    # Core scalar metrics
    metrics = compute_metrics(manifest)
    metrics.to_csv(outdir / "tier1_metrics_all.csv", index=False)

    # Capacity plots (on + off)
    plot_capacity_two_panel(metrics, outdir / "fig_capacity_on_off_vs_resolution.png")

    # Curtailment plots (separate for on/off)
    plot_lines_vs_resolution(metrics, "offwind_curt_frac", "Offshore wind curtailment (fraction)", outdir / "fig_curt_offwind_vs_resolution.png", "Offshore curtailment sensitivity (Tier 1)")
    plot_lines_vs_resolution(metrics, "onwind_curt_frac", "Onshore wind curtailment (fraction)", outdir / "fig_curt_onwind_vs_resolution.png", "Onshore curtailment sensitivity (Tier 1)")

    # Transmission + objective-derived plots
    plot_lines_vs_resolution(metrics, "trans_exp_twkm", "Transmission expansion (TW·km)", outdir / "fig_trans_exp_vs_resolution.png", "Network reinforcement sensitivity (Tier 1)")
    plot_marginal_value_proxy(metrics, outdir / "fig_marginal_value_proxy_offwind.png", cap_col="offwind_cap_gw")

    # Interaction plots (wake effect) - do this mainly for offshore metrics
    plot_interaction_wake_effect(metrics, "offwind_cap_gw", "Δ offshore capacity due to wakes (GW)", outdir / "fig_interaction_wake_offwind_cap.png", "Wake impact on offshore capacity, split by bias")
    plot_interaction_wake_effect(metrics, "trans_exp_twkm", "Δ transmission expansion due to wakes (TW·km)", outdir / "fig_interaction_wake_trans.png", "Wake impact on grid reinforcement, split by bias")

    write_summary_table(metrics, outdir / "table_summary_coarse_fine.csv", coarse=args.coarse, fine=args.fine)

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
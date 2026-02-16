#!/usr/bin/env python3
"""
Tier-2 plotting for thesis sensitivity (Europe-wide, sector-coupled)

- Uses your plot_style.py + thesis_colors.py (consistent thesis styling)
- Scans solved PyPSA networks from results/
- Parses scenario from folder names like: <wakeprefix>-s<RES>-biasTrue/False
- Tier-2 intent: reduced, confirmatory set (default: baseline vs bias+wake)
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
import re
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pypsa

# ----------------------------
# Thesis styling
# ----------------------------
# Your repo should provide these.
# We keep defensive fallbacks so the script never hard-crashes.
try:
    import plotting_style as ps  # expected to define thesis_plot_style() + apply_spatial_resolution_axis()
except Exception:  # pragma: no cover
    ps = None

try:
    import thesis_colors as tc  # expected to define THESIS_COLORS + label()
except Exception:  # pragma: no cover
    tc = None


def apply_thesis_style():
    if ps is not None and hasattr(ps, "thesis_plot_style"):
        ps.thesis_plot_style()
    else:
        # Minimal sane defaults (will be overridden if your style exists)
        plt.rcParams.update(
            {
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "legend.fontsize": 8,
            }
        )


def get_color(key: str) -> str | None:
    """Pull consistent colors from thesis_colors.py if present."""
    if tc is not None and hasattr(tc, "THESIS_COLORS"):
        d = tc.THESIS_COLORS
        if isinstance(d, dict) and key in d:
            return d[key]
    return None


def get_label(key: str) -> str:
    """Pull consistent labels from thesis_colors.py if present."""
    if tc is not None and hasattr(tc, "label"):
        try:
            return tc.label(key)
        except Exception:
            pass
    return key


# ----------------------------
# Scenario parsing
# ----------------------------
_RE_SCENARIO = re.compile(r"(?P<wakeprefix>[^/]+)-s(?P<res>\d+)-bias(?P<bias>True|False)", re.IGNORECASE)

WAKE_ALIASES = {
    "base": "off",
    "standard": "off",
    "no_wake": "off",
    "wakeoff": "off",
    "off": "off",
    # density-based wake
    "new_more": "density",
    "density": "density",
    "density_based": "density",
    "density-based": "density",
}


def scenario_key(bias: bool, wake: str) -> str:
    wake_is_off = (wake == "off")
    if (not bias) and wake_is_off:
        return "base"
    if bias and wake_is_off:
        return "bias"
    if (not bias) and (not wake_is_off):
        return "wake"
    return "bias+wake"


def parse_from_path(nc_path: Path) -> dict:
    # Find parent folder with "-sNNN-biasX"
    scenario_folder = None
    for p in reversed(nc_path.parts):
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
        "wake": wake,
        "wakeprefix": wakeprefix,
        "scenario_folder": scenario_folder,
    }


def build_manifest(results_root: Path, pattern: str) -> pd.DataFrame:
    files = sorted(results_root.glob(pattern))
    if not files:
        raise SystemExit(f"No networks found under {results_root} with pattern {pattern!r}")
    recs = [parse_from_path(p) for p in files]
    df = pd.DataFrame(recs)
    return df


# ----------------------------
# Tech selection + metrics
# ----------------------------
def gen_idx(n: pypsa.Network, tech: str) -> pd.Index:
    carr = n.generators.carrier.astype(str).str.lower()
    if tech == "offwind":
        return n.generators.index[carr.str.contains("offwind")]
    if tech == "onwind":
        return n.generators.index[carr.eq("onwind")]
    raise ValueError(tech)


def capacity_gw_from_generators(n: pypsa.Network, tech: str) -> float:
    idx = gen_idx(n, tech)
    if len(idx) == 0:
        return 0.0
    g = n.generators.loc[idx]

    p_nom_opt = g["p_nom_opt"] if "p_nom_opt" in g.columns else pd.Series(index=g.index, dtype=float)
    p_nom = g["p_nom"] if "p_nom" in g.columns else pd.Series(index=g.index, dtype=float)

    cap_mw = p_nom_opt.where(p_nom_opt.notna(), p_nom).fillna(0.0).sum()
    return float(cap_mw) / 1e3


def curtailment_frac(n: pypsa.Network, tech: str) -> float:
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


def snapshot_weights(n: pypsa.Network) -> pd.Series:
    # PyPSA-Eur commonly uses snapshot_weightings["generators"]
    if hasattr(n, "snapshot_weightings") and isinstance(n.snapshot_weightings, pd.DataFrame):
        if "generators" in n.snapshot_weightings.columns:
            return n.snapshot_weightings["generators"]
    # fallback: assume 1 hour per snapshot
    return pd.Series(1.0, index=n.snapshots)


def energy_twh_from_generators(n: pypsa.Network, tech: str) -> float:
    idx = gen_idx(n, tech)
    if len(idx) == 0:
        return 0.0
    try:
        p = n.generators_t.p[idx].sum(axis=1)  # MW
    except Exception:
        return 0.0
    w = snapshot_weights(n)
    mwh = float((p * w).sum())
    return mwh / 1e6  # TWh


def transmission_expansion_twkm(n: pypsa.Network) -> float:
    total_mw_km = 0.0
    if hasattr(n, "lines") and not n.lines.empty and "s_nom_opt" in n.lines.columns:
        ln = n.lines
        base = ln.get("s_nom", pd.Series(0.0, index=ln.index)).fillna(0.0)
        opt = ln["s_nom_opt"].fillna(base)
        delta = (opt - base).clip(lower=0.0)
        length = ln.get("length", pd.Series(0.0, index=ln.index)).fillna(0.0)
        total_mw_km += float((delta * length).sum())

    if hasattr(n, "links") and not n.links.empty and "p_nom_opt" in n.links.columns:
        lk = n.links
        base = lk.get("p_nom", pd.Series(0.0, index=lk.index)).fillna(0.0)
        opt = lk["p_nom_opt"].fillna(base)
        delta = (opt - base).clip(lower=0.0)
        length = lk.get("length", pd.Series(0.0, index=lk.index)).fillna(0.0)
        total_mw_km += float((delta * length).sum())

    return total_mw_km / 1e6  # TW·km


def get_objective(n: pypsa.Network) -> float:
    if hasattr(n, "objective") and n.objective is not None:
        try:
            return float(n.objective)
        except Exception:
            pass
    return float("nan")


# --- sector coupling lens: electrolysers / H2 production (robust heuristics) ---
_ELEC_KEYWORDS = ("electroly", "electrolysis", "h2 electro", "pem", "alkaline")

def electrolyser_links(n: pypsa.Network) -> pd.Index:
    if not hasattr(n, "links") or n.links.empty:
        return pd.Index([])
    carr = n.links.carrier.astype(str).str.lower()
    mask = np.zeros(len(carr), dtype=bool)
    for k in _ELEC_KEYWORDS:
        mask |= carr.str.contains(k)
    return n.links.index[mask]

def electrolyser_capacity_gw(n: pypsa.Network) -> float:
    idx = electrolyser_links(n)
    if len(idx) == 0:
        return 0.0
    lk = n.links.loc[idx]
    if "p_nom_opt" in lk.columns and lk["p_nom_opt"].notna().any():
        cap_mw = lk["p_nom_opt"].fillna(lk.get("p_nom", 0.0)).sum()
    else:
        cap_mw = lk.get("p_nom", pd.Series(0.0, index=lk.index)).sum()
    return float(cap_mw) / 1e3

def h2_production_twh(n: pypsa.Network) -> float:
    """
    Heuristic: use link power on the output side if present.
    Many PyPSA-Eur sector models put electricity on bus0 and hydrogen on bus1.
    We'll compute energy on p1 (MW) if available; otherwise p0 with sign flip.
    """
    idx = electrolyser_links(n)
    if len(idx) == 0:
        return 0.0
    w = snapshot_weights(n)

    try:
        if hasattr(n.links_t, "p1"):
            p = n.links_t.p1[idx].sum(axis=1)  # MW (H2 out, often positive)
        else:
            p = -n.links_t.p0[idx].sum(axis=1)  # MW (elec in negative), flip to "production"
    except Exception:
        return 0.0

    mwh = float((p * w).sum())
    return mwh / 1e6


def compute_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in manifest.iterrows():
        n = pypsa.Network(r["path"])
        scen = scenario_key(bool(r["bias"]), str(r["wake"]))

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
    base_order = ["base", "bias", "wake", "bias+wake"]
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

    if ps is not None and hasattr(ps, "apply_spatial_resolution_axis"):
        ps.apply_spatial_resolution_axis(ax)
    else:
        ax.set_xscale("log")
        ax.set_xlabel("Spatial resolution (log)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath)
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
        if ps is not None and hasattr(ps, "apply_spatial_resolution_axis"):
            ps.apply_spatial_resolution_axis(ax)
        else:
            ax.set_xscale("log")
            ax.set_xlabel("Spatial resolution (log)")
        ax.set_title(get_label(tech), fontsize=9)
        ax.set_ylabel("Capacity (GW)")
        ax.grid(True, alpha=0.3)

    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath)
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
        if ps is not None and hasattr(ps, "apply_spatial_resolution_axis"):
            ps.apply_spatial_resolution_axis(ax)
        else:
            ax.set_xscale("log")
            ax.set_xlabel("Spatial resolution (log)")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)


# --- ECDF for CF distributions (robust + thesis-friendly) ---
def cf_timeseries(n: pypsa.Network, tech: str) -> pd.DataFrame | None:
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


def build_cf_long_both(manifest: pd.DataFrame, resolutions: tuple[int, ...], compare: list[str]) -> pd.DataFrame:
    rows = []
    for r in manifest.itertuples(index=False):
        res = int(r.resolution)
        if res not in set(resolutions):
            continue

        scen = scenario_key(bool(r.bias), str(r.wake))
        if scen not in compare:
            continue

        n = pypsa.Network(r.path)
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
    axes[1, -1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(f"{ylabel} ECDFs (Tier 2)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outpath)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    apply_thesis_style()

    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results")
    ap.add_argument("--glob", required=True, help="Glob relative to results-root to find networks")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--write-manifest", action="store_true")
    ap.add_argument("--resolutions", nargs="+", type=int, default=[1000000, 10000], help="Subset for Tier-2 plots")
    ap.add_argument("--compare", nargs="+", default=["base", "bias+wake"], help="Scenarios to compare (ordered)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(Path(args.results_root), args.glob)

    # Keep only the resolutions we want for Tier 2 (confirmatory)
    res_sel = tuple(int(x) for x in args.resolutions)
    manifest = manifest[manifest["resolution"].isin(res_sel)].copy()

    # Normalise scenario labels
    manifest["scenario"] = manifest.apply(lambda r: scenario_key(bool(r["bias"]), str(r["wake"])), axis=1)

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
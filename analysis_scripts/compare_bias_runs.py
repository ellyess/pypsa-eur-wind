#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pypsa

from plotting_style import thesis_plot_style, format_axes_standard
from thesis_colors import label, THESIS_LABELS, THESIS_COLORS

from network_utils import (
    snapshot_weights as snap_weights,
    get_objective as objective,
    bus_country,
    capacity_by_carrier,
    energy_by_carrier_twh,
    cf_timeseries_per_gen as cf_timeseries,
)

try:
    import xarray as xr
except ImportError:
    xr = None

# Apply thesis-wide plotting style
_style = thesis_plot_style()
cm, lw, ms, dpi = _style['cm'], _style['lw'], _style['ms'], _style['dpi']

# -----------------------------
# Nice names (applied to plots/tables where relevant)
# -----------------------------
# Extended nice names that complement thesis_colors.THESIS_LABELS
NICE_NAMES = {
    "OCGT": "Open-Cycle Gas",
    "CCGT": "Combined-Cycle Gas",
    "offwind-ac": "Offshore Wind (AC)",
    "offwind-dc": "Offshore Wind (DC)",
    "offwind-float": "Offshore Wind (Floating)",
    "onwind": "Onshore Wind",
    "solar": "Solar",
    "PHS": "Pumped Hydro Storage",
    "hydro": "Reservoir & Dam",
    "battery": "Battery Storage",
    "H2": "Hydrogen Storage",
    "lines": "Transmission Lines",
    "ror": "Run of River",
    "load": "Load Shedding",
    "ac": "AC",
    "dc": "DC",
    "bias_off": "Baseline",
    "bias_uniform": "Uniform",
    "bias_on": "PyVWF",

}


def nice(x: str) -> str:
    """Get nice label, checking both THESIS_LABELS and local NICE_NAMES."""
    return label(x, default=NICE_NAMES.get(x, x))


# -----------------------------
# Aggregations
# -----------------------------

def wind_breakdown_by_country(
    n: pypsa.Network,
    carriers: tuple[str, ...],
    *,
    label: str,
) -> tuple[pd.Series, pd.Series]:
    """
    Return (capacity_MW_by_country, energy_TWh_by_country) for specified wind carriers.
    """
    if not len(n.generators):
        return pd.Series(dtype=float), pd.Series(dtype=float)

    mask = n.generators.carrier.isin(carriers)
    if not mask.any():
        return pd.Series(dtype=float), pd.Series(dtype=float)

    gens = n.generators.loc[mask].copy()
    gens["country"] = gens.bus.map(bus_country)

    # Capacity
    p_nom = gens.p_nom_opt if "p_nom_opt" in gens else gens.p_nom
    cap = p_nom.groupby(gens["country"]).sum().sort_values(ascending=False)
    cap.name = f"{label}_capacity_MW"

    # Energy
    if not hasattr(n.generators_t, "p") or n.generators_t.p.empty:
        en = pd.Series(dtype=float, name=f"{label}_energy_TWh")
    else:
        w = snap_weights(n)
        mwh = (n.generators_t.p[gens.index].mul(w, axis=0)).sum(axis=0)
        en = (mwh.groupby(gens["country"]).sum().sort_values(ascending=False) / 1e6)
        en.name = f"{label}_energy_TWh"

    return cap, en


def align_delta(a: pd.Series, b: pd.Series, name_a="bias_off", name_b="bias_on") -> pd.DataFrame:
    df = pd.concat([a.rename(name_a), b.rename(name_b)], axis=1).fillna(0.0)
    df["delta"] = df[name_b] - df[name_a]
    df["delta_pct"] = np.where(df[name_a] != 0, 100 * df["delta"] / df[name_a], np.nan)
    return df



def align_multi(series_by_run: dict[str, pd.Series], baseline: str = "bias_off") -> pd.DataFrame:
    """Align multiple Series on index and compute deltas vs baseline."""
    df = pd.concat({k: v for k, v in series_by_run.items()}, axis=1).fillna(0.0)
    if baseline not in df.columns:
        raise KeyError(f"baseline {baseline!r} not in series_by_run keys {list(df.columns)}")
    for k in df.columns:
        if k == baseline:
            continue
        df[f"delta_{k}"] = df[k] - df[baseline]
        df[f"delta_pct_{k}"] = np.where(df[baseline] != 0, 100 * df[f"delta_{k}"] / df[baseline], np.nan)
    return df


# -----------------------------
# CF distributions
# -----------------------------

def _weighted_sample_values(
    cf: pd.DataFrame,
    w: pd.Series,
    *,
    max_assets: int,
    weight_scale: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Produce a reproducible, memory-safe sample of CF values for CDF/quantiles.

    - Optionally subsamples columns to at most max_assets.
    - Repeats each snapshot row proportional to snapshot weight * weight_scale.
      This is an approximation but stable and fast for large networks.
    """
    if cf.empty:
        return np.array([])

    cf = cf.copy()

    # Subsample assets if huge
    if max_assets and cf.shape[1] > max_assets:
        rng = np.random.default_rng(seed)
        cols = rng.choice(cf.columns.to_numpy(), size=max_assets, replace=False)
        cf = cf.loc[:, cols]

    w = w.reindex(cf.index).fillna(1.0)

    # Clamp to [0,1] after cleaning
    cf = cf.replace([np.inf, -np.inf], np.nan)
    vals = []
    reps = np.maximum(1, np.round(w.to_numpy() * max(1, weight_scale)).astype(int))

    for i, r in enumerate(reps):
        row = cf.iloc[i].to_numpy()
        row = row[np.isfinite(row)]
        if row.size == 0:
            continue
        row = np.clip(row, 0.0, 1.0)
        vals.append(np.tile(row, r))

    if not vals:
        return np.array([])
    return np.concatenate(vals)


def _empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([]), np.array([])
    x = np.sort(values)
    y = np.linspace(0, 1, len(x), endpoint=True)
    return x, y



def plot_cdf_and_delta(
    n0: pypsa.Network,
    n1: pypsa.Network,
    carriers: tuple[str, ...],
    *,
    label: str,
    kind: str,
    out_cdf: Path,
    out_delta: Path,
    max_assets: int,
    weight_scale: int,
    n2: pypsa.Network | None = None,
):
    """
    Save:
      - CDF plot (Raw vs PyVWF [+ Uniform if provided])
      - Delta CDF plot(s) vs Raw on a common x-grid:
            CDF(PyVWF) - CDF(Raw)
            CDF(Uniform) - CDF(Raw)  (optional)
    """
    w0 = snap_weights(n0)
    w1 = snap_weights(n1)

    cf0 = cf_timeseries(n0, carriers, kind=kind)
    cf1 = cf_timeseries(n1, carriers, kind=kind)

    cf2 = None
    w2 = None
    if n2 is not None:
        w2 = snap_weights(n2)
        cf2 = cf_timeseries(n2, carriers, kind=kind)

    if cf0.empty or cf1.empty or (n2 is not None and (cf2 is None or cf2.empty)):
        out_cdf.with_suffix(".txt").write_text(
            f"Skipped {label} {kind}: missing data (p_max_pu or p) in one/both networks.\n"
        )
        out_delta.with_suffix(".txt").write_text(
            f"Skipped {label} {kind}: missing data (p_max_pu or p) in one/both networks.\n"
        )
        return

    v0 = _weighted_sample_values(cf0, w0, max_assets=max_assets, weight_scale=weight_scale, seed=0)
    v1 = _weighted_sample_values(cf1, w1, max_assets=max_assets, weight_scale=weight_scale, seed=0)
    if v0.size == 0 or v1.size == 0:
        out_cdf.with_suffix(".txt").write_text(f"Skipped {label} {kind}: no finite values after cleaning.\n")
        out_delta.with_suffix(".txt").write_text(f"Skipped {label} {kind}: no finite values after cleaning.\n")
        return

    x0, y0 = _empirical_cdf(v0)
    x1, y1 = _empirical_cdf(v1)

    x2 = y2 = None
    if n2 is not None and cf2 is not None and w2 is not None:
        v2 = _weighted_sample_values(cf2, w2, max_assets=max_assets, weight_scale=weight_scale, seed=0)
        if v2.size != 0:
            x2, y2 = _empirical_cdf(v2)

    # CDF plot
    plt.figure(figsize=(5.2, 3.2))
    plt.plot(x0, y0, label=nice("bias_off"), linewidth=2, color=THESIS_COLORS.get("base", None))
    plt.plot(x1, y1, label=nice("bias_on"), linewidth=2, color=THESIS_COLORS.get("bias", None))
    if x2 is not None and y2 is not None and len(x2) > 0:
        plt.plot(x2, y2, label=nice("bias_uniform"), linewidth=2, color=THESIS_COLORS.get("standard", None))
    plt.xlabel("Capacity factor")
    plt.ylabel("CDF")
    plt.title(f"{label} CF distribution ({kind})")
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    format_axes_standard(plt.gcf())
    plt.savefig(out_cdf, dpi=300, bbox_inches="tight")
    plt.close()

    # Delta CDF plot on common x-grid
    x = np.linspace(0, 1, 600)
    c0 = np.interp(x, x0, y0)
    c1 = np.interp(x, x1, y1)
    dc1 = c1 - c0

    plt.figure(figsize=(5.2, 3.2))
    plt.plot(x, dc1, label=f"{nice('bias_on')} − {nice('bias_off')}", linewidth=3, color=THESIS_COLORS.get("bias", None))
    if x2 is not None and y2 is not None and len(x2) > 0:
        c2 = np.interp(x, x2, y2)
        dc2 = c2 - c0
        plt.plot(x, dc2, label=f"{nice('bias_uniform')} − {nice('bias_off')}", linewidth=3, color=THESIS_COLORS.get("standard", None))
    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
    plt.xlabel("Capacity factor")
    plt.ylabel("ΔCDF")
    plt.title(f"{label} ΔCDF ({kind})")
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    format_axes_standard(plt.gcf())
    plt.savefig(out_delta, dpi=300, bbox_inches="tight")
    plt.close()



def quantile_shifts(
    n0: pypsa.Network,
    n1: pypsa.Network,
    carriers: tuple[str, ...],
    *,
    label: str,
    kind: str,
    max_assets: int,
    weight_scale: int,
    n2: pypsa.Network | None = None,
) -> pd.Series:
    """
    Compute p10/p50/p90 for Raw and PyVWF (and optionally Uniform) and the shifts vs Raw.
    Uses the same sampling as the CDF plots for consistency and scalability.
    """
    w0 = snap_weights(n0)
    w1 = snap_weights(n1)
    cf0 = cf_timeseries(n0, carriers, kind=kind)
    cf1 = cf_timeseries(n1, carriers, kind=kind)

    if cf0.empty or cf1.empty:
        base = {
            "label": label,
            "kind": kind,
            "p10_raw": np.nan, "p50_raw": np.nan, "p90_raw": np.nan,
            "p10_pyvwf": np.nan, "p50_pyvwf": np.nan, "p90_pyvwf": np.nan,
            "p10_shift_pyvwf": np.nan, "p50_shift_pyvwf": np.nan, "p90_shift_pyvwf": np.nan,
        }
        if n2 is not None:
            base.update({
                "p10_uniform": np.nan, "p50_uniform": np.nan, "p90_uniform": np.nan,
                "p10_shift_uniform": np.nan, "p50_shift_uniform": np.nan, "p90_shift_uniform": np.nan,
            })
        return pd.Series(base)

    v0 = _weighted_sample_values(cf0, w0, max_assets=max_assets, weight_scale=weight_scale, seed=0)
    v1 = _weighted_sample_values(cf1, w1, max_assets=max_assets, weight_scale=weight_scale, seed=0)

    def _q(v: np.ndarray) -> tuple[float, float, float]:
        v = v[np.isfinite(v)]
        if v.size == 0:
            return (np.nan, np.nan, np.nan)
        v = np.clip(v, 0.0, 1.0)
        return (float(np.quantile(v, 0.10)), float(np.quantile(v, 0.50)), float(np.quantile(v, 0.90)))

    p10_0, p50_0, p90_0 = _q(v0)
    p10_1, p50_1, p90_1 = _q(v1)

    out = {
        "label": label,
        "kind": kind,
        "p10_raw": p10_0, "p50_raw": p50_0, "p90_raw": p90_0,
        "p10_pyvwf": p10_1, "p50_pyvwf": p50_1, "p90_pyvwf": p90_1,
        "p10_shift_pyvwf": p10_1 - p10_0,
        "p50_shift_pyvwf": p50_1 - p50_0,
        "p90_shift_pyvwf": p90_1 - p90_0,
    }

    if n2 is not None:
        w2 = snap_weights(n2)
        cf2 = cf_timeseries(n2, carriers, kind=kind)
        if cf2.empty:
            out.update({
                "p10_uniform": np.nan, "p50_uniform": np.nan, "p90_uniform": np.nan,
                "p10_shift_uniform": np.nan, "p50_shift_uniform": np.nan, "p90_shift_uniform": np.nan,
            })
        else:
            v2 = _weighted_sample_values(cf2, w2, max_assets=max_assets, weight_scale=weight_scale, seed=0)
            p10_2, p50_2, p90_2 = _q(v2)
            out.update({
                "p10_uniform": p10_2, "p50_uniform": p50_2, "p90_uniform": p90_2,
                "p10_shift_uniform": p10_2 - p10_0,
                "p50_shift_uniform": p50_2 - p50_0,
                "p90_shift_uniform": p90_2 - p90_0,
            })

    return pd.Series(out)



def plot_grouped_deltas(
    base: pd.Series,
    a: pd.Series,
    b: pd.Series | None,
    *,
    title: str,
    ylabel: str,
    outpath: Path,
    nice_index: dict[str, str] | None = None,
    label_a: str = "PyVWF − Raw",
    label_b: str = "Uniform − Raw",
    top: int = 20,
    sort_by_abs: bool = True,
):
    """
    Grouped bar plot of deltas relative to base for series a and optionally b.

    base: Raw values
    a: PyVWF values
    b: Uniform values (optional)
    """
    # Align indices
    idx = base.index.union(a.index)
    if b is not None:
        idx = idx.union(b.index)

    df = pd.DataFrame(index=idx)
    df["raw"] = base.reindex(idx).fillna(0.0)
    df["pyvwf"] = a.reindex(idx).fillna(0.0)
    df["d_pyvwf"] = df["pyvwf"] - df["raw"]

    if b is not None:
        df["uniform"] = b.reindex(idx).fillna(0.0)
        df["d_uniform"] = df["uniform"] - df["raw"]
    else:
        df["d_uniform"] = np.nan

    # Sorting
    if sort_by_abs:
        if b is not None:
            score = np.maximum(df["d_pyvwf"].abs(), df["d_uniform"].abs())
        else:
            score = df["d_pyvwf"].abs()
        df = df.loc[score.sort_values(ascending=False).index]
    else:
        df = df.sort_values("d_pyvwf", ascending=False)

    df = df.head(top)

    # Nice labels
    plot_index = [nice_index.get(str(k), str(k)) for k in df.index] if nice_index else [str(k) for k in df.index]

    # Plot
    x = np.arange(len(df))
    width = 0.40 if b is not None else 0.60

    plt.figure(figsize=(max(8, 0.45 * len(df)), 5))
    plt.axhline(0.0, linewidth=1, color="black", linestyle="--")

    plt.bar(x - (width / 2 if b is not None else 0), df["d_pyvwf"].to_numpy(), width=width, label=label_a, color=THESIS_COLORS.get("bias", None))

    if b is not None:
        plt.bar(x + width / 2, df["d_uniform"].to_numpy(), width=width, label=label_b, color=THESIS_COLORS.get("standard", None))

    plt.xticks(x, plot_index, rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend(loc='best', frameon=False)
    plt.tight_layout()
    format_axes_standard(plt.gcf())
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def apply_nice_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = [nice(str(i)) for i in out.index]
    return out


def plot_system_cost_bar(
    n0: pypsa.Network,
    n1: pypsa.Network,
    n2: pypsa.Network | None,
    *,
    outpath: Path,
) -> None:
    """Bar chart of system cost across bias correction scenarios."""
    costs = {"Baseline": objective(n0), "PyVWF": objective(n1)}
    if n2 is not None:
        costs["Uniform"] = objective(n2)

    labels_list = list(costs.keys())
    values = [v / 1e9 for v in costs.values()]  # EUR -> B EUR
    color_map = {"Baseline": THESIS_COLORS.get("base"), "PyVWF": THESIS_COLORS.get("bias"), "Uniform": THESIS_COLORS.get("standard")}
    colors = [color_map.get(l, "#999999") for l in labels_list]

    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)
    x = np.arange(len(labels_list))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_list)
    ax.set_ylabel(r"System cost [B€]")

    # Add delta annotations relative to baseline
    base_cost = values[0]
    for i, (lbl, v) in enumerate(zip(labels_list, values)):
        if i == 0:
            continue
        delta_pct = 100 * (v - base_cost) / abs(base_cost) if base_cost != 0 else 0
        sign = "+" if delta_pct >= 0 else ""
        ax.text(i, v + 0.002 * abs(base_cost), f"{sign}{delta_pct:.2f}%",
               ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_duration_curve(
    n0: pypsa.Network,
    n1: pypsa.Network,
    n2: pypsa.Network | None,
    *,
    carriers: tuple[str, ...],
    label_text: str,
    outpath: Path,
) -> None:
    """Wind output duration curve (sorted dispatch) for each bias scenario."""
    fig, ax = plt.subplots(figsize=(5.2, 3.2), dpi=300)

    for n, name, color_key in [
        (n0, "Baseline", "base"),
        (n1, "PyVWF", "bias"),
        (n2, "Uniform", "standard"),
    ]:
        if n is None:
            continue
        mask = n.generators.carrier.isin(carriers)
        if not mask.any():
            continue
        idx = n.generators.index[mask]
        if not hasattr(n.generators_t, "p") or n.generators_t.p.empty:
            continue
        total_gw = n.generators_t.p[idx].sum(axis=1) / 1e3  # MW -> GW
        sorted_gw = total_gw.sort_values(ascending=False).values
        hours = np.arange(1, len(sorted_gw) + 1)
        ax.plot(hours, sorted_gw, linewidth=1.2, label=name,
               color=THESIS_COLORS.get(color_key))

    ax.set_xlabel("Hours")
    ax.set_ylabel(f"{label_text} output [GW]")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_bias_correction_map(
    bias_path: str | Path,
    *,
    outpath: Path,
) -> None:
    """Plot spatial maps of the bias correction scalar and offset fields."""
    if xr is None:
        print("[WARN] xarray not available; skipping bias correction map.")
        return

    bias_path = Path(bias_path)
    if not bias_path.exists():
        print(f"[WARN] Bias correction file not found: {bias_path}")
        return

    ds = xr.open_dataset(bias_path)

    has_scalar = "scalar" in ds.data_vars
    has_offset = "offset" in ds.data_vars
    n_panels = int(has_scalar) + int(has_offset)
    if n_panels == 0:
        print("[WARN] Bias correction file has neither 'scalar' nor 'offset' variable.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.0), dpi=300)
    if n_panels == 1:
        axes = [axes]

    panel_idx = 0
    if has_scalar:
        ax = axes[panel_idx]
        scalar = ds["scalar"].squeeze()
        if scalar.ndim == 2:
            im = ax.pcolormesh(scalar.x, scalar.y, scalar.values, cmap="RdBu_r",
                              vmin=0.85, vmax=1.15, shading="auto")
        else:
            im = ax.scatter(scalar.x, scalar.y, c=scalar.values, cmap="RdBu_r",
                          vmin=0.85, vmax=1.15, s=2)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Scalar [-]")
        ax.set_title("Bias correction: scalar")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        panel_idx += 1

    if has_offset:
        ax = axes[panel_idx]
        offset = ds["offset"].squeeze()
        vlim = float(np.nanmax(np.abs(offset.values))) if np.any(np.isfinite(offset.values)) else 1.0
        if offset.ndim == 2:
            im = ax.pcolormesh(offset.x, offset.y, offset.values, cmap="RdBu_r",
                              vmin=-vlim, vmax=vlim, shading="auto")
        else:
            im = ax.scatter(offset.x, offset.y, c=offset.values, cmap="RdBu_r",
                          vmin=-vlim, vmax=vlim, s=2)
        fig.colorbar(im, ax=ax, shrink=0.8, label="Offset [m/s]")
        ax.set_title("Bias correction: offset")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    ds.close()
    fig.tight_layout()
    format_axes_standard(fig)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="bias OFF .nc")
    ap.add_argument("--corr", required=True, help="bias ON .nc")
    ap.add_argument("--uniform", default=None, help="bias UNIFORM .nc (optional)")
    ap.add_argument("--out", required=True, help="output dir")
    ap.add_argument("--offshore-carriers", default="offwind,offwind-ac,offwind-dc,offwind-float")
    ap.add_argument("--onshore-carriers", default="onwind")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--cf-max-assets", type=int, default=4000, help="Max generators sampled for CF distributions/quantiles (per group).")
    ap.add_argument("--cf-weight-scale", type=int, default=10, help="Snapshot weight scaling for CF sampling (higher = smoother CDF, slower).")
    ap.add_argument("--bias-file", default=None, help="Path to atlite_bias.nc for map plot (auto-detected if omitted).")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    offshore_carriers = tuple(c.strip() for c in args.offshore_carriers.split(",") if c.strip())
    onshore_carriers = tuple(c.strip() for c in args.onshore_carriers.split(",") if c.strip())

    n0 = pypsa.Network(args.raw)   # bias OFF
    n1 = pypsa.Network(args.corr)  # bias ON
    n2 = pypsa.Network(args.uniform) if args.uniform else None  # bias UNIFORM
    # Carrier-level totals
    cap_series = {"bias_off": capacity_by_carrier(n0), "bias_on": capacity_by_carrier(n1)}
    en_series = {"bias_off": energy_by_carrier_twh(n0), "bias_on": energy_by_carrier_twh(n1)}
    if n2 is not None:
        cap_series["bias_uniform"] = capacity_by_carrier(n2)
        en_series["bias_uniform"] = energy_by_carrier_twh(n2)

    cap_cmp = align_multi(cap_series, baseline="bias_off")
    en_cmp = align_multi(en_series, baseline="bias_off")

    cap_cmp.to_csv(outdir / "capacity_by_carrier.csv")
    en_cmp.to_csv(outdir / "energy_by_carrier_TWh.csv")

    
    # Offshore + onshore breakdowns by country
    offcap0, offen0 = wind_breakdown_by_country(n0, offshore_carriers, label="offshore")
    offcap1, offen1 = wind_breakdown_by_country(n1, offshore_carriers, label="offshore")
    offcapU, offenU = wind_breakdown_by_country(n2, offshore_carriers, label="offshore") if n2 is not None else (None, None)
    oncap0, onen0 = wind_breakdown_by_country(n0, onshore_carriers, label="onshore")
    oncap1, onen1 = wind_breakdown_by_country(n1, onshore_carriers, label="onshore")
    oncapU, onenU = wind_breakdown_by_country(n2, onshore_carriers, label="onshore") if n2 is not None else (None, None)
    

    offcap_series = {"bias_off": offcap0, "bias_on": offcap1}
    offen_series = {"bias_off": offen0, "bias_on": offen1}
    oncap_series = {"bias_off": oncap0, "bias_on": oncap1}
    onen_series = {"bias_off": onen0, "bias_on": onen1}

    if n2 is not None:
        offcap2, offen2 = wind_breakdown_by_country(n2, offshore_carriers, label="offshore")
        oncap2, onen2 = wind_breakdown_by_country(n2, onshore_carriers, label="onshore")
        offcap_series["bias_uniform"] = offcap2
        offen_series["bias_uniform"] = offen2
        oncap_series["bias_uniform"] = oncap2
        onen_series["bias_uniform"] = onen2

    offcap_cmp = align_multi(offcap_series, baseline="bias_off")
    offen_cmp = align_multi(offen_series, baseline="bias_off")
    oncap_cmp = align_multi(oncap_series, baseline="bias_off")
    onen_cmp = align_multi(onen_series, baseline="bias_off")

    offcap_cmp.to_csv(outdir / "offshore_capacity_by_country.csv")
    offen_cmp.to_csv(outdir / "offshore_energy_by_country_TWh.csv")
    oncap_cmp.to_csv(outdir / "onshore_capacity_by_country.csv")
    onen_cmp.to_csv(outdir / "onshore_energy_by_country_TWh.csv")


    cap0 = capacity_by_carrier(n0)
    cap1 = capacity_by_carrier(n1)
    capU = capacity_by_carrier(n2) if n2 is not None else None

    en0 = energy_by_carrier_twh(n0)
    en1 = energy_by_carrier_twh(n1)
    enU = energy_by_carrier_twh(n2) if n2 is not None else None
    
    
    plot_grouped_deltas(
        cap0, cap1, capU,
        title="Δ Capacity by carrier (relative to Raw)",
        ylabel="MW",
        outpath=outdir / "delta_capacity_by_carrier_grouped.png",
        nice_index=NICE_NAMES,
        label_a="PyVWF − Raw",
        label_b="Uniform − Raw",
        top=args.top,
    )

    plot_grouped_deltas(
        en0, en1, enU,
        title="Δ Energy by carrier (relative to Raw)",
        ylabel="TWh",
        outpath=outdir / "delta_energy_by_carrier_grouped.png",
        nice_index=NICE_NAMES,
        label_a="PyVWF − Raw",
        label_b="Uniform − Raw",
        top=args.top,
    )
    
    plot_grouped_deltas(
    oncap0, oncap1, oncapU,
    title="Δ Onshore wind capacity by country (relative to Raw)",
    ylabel="MW",
    outpath=outdir / "delta_onshore_capacity_by_country_grouped.png",
    nice_index=None,  # ISO2 already fine
    top=args.top,
)

    plot_grouped_deltas(
        onen0, onen1, onenU,
        title="Δ Onshore wind energy by country (relative to Raw)",
        ylabel="TWh",
        outpath=outdir / "delta_onshore_energy_by_country_grouped.png",
        nice_index=None,
        top=args.top,
    )

    plot_grouped_deltas(
        offcap0, offcap1, offcapU,
        title="Δ Offshore wind capacity by country (relative to Raw)",
        ylabel="MW",
        outpath=outdir / "delta_offshore_capacity_by_country_grouped.png",
        nice_index=None,
        top=args.top,
    )

    plot_grouped_deltas(
        offen0, offen1, offenU,
        title="Δ Offshore wind energy by country (relative to Raw)",
        ylabel="TWh",
        outpath=outdir / "delta_offshore_energy_by_country_grouped.png",
        nice_index=None,
        top=args.top,
    )

    # CF distributions + ΔCDF
    for group_label, carriers in [
        ("Onshore wind", onshore_carriers),
        ("Offshore wind", offshore_carriers),
    ]:
        for kind in ["availability", "dispatch"]:
            out_cdf = outdir / f"cdf_cf_{group_label.split()[0].lower()}_{kind}.png"
            out_delta = outdir / f"delta_cdf_cf_{group_label.split()[0].lower()}_{kind}.png"
            plot_cdf_and_delta(
                n0,
                n1,
                carriers,
                label=group_label,
                kind=kind,
                out_cdf=out_cdf,
                out_delta=out_delta,
                max_assets=args.cf_max_assets,
                weight_scale=args.cf_weight_scale,
                n2=n2,
            )

    # Quantile shift table (p10/p50/p90)
    rows = []
    for group_label, carriers in [
        ("Onshore wind", onshore_carriers),
        ("Offshore wind", offshore_carriers),
    ]:
        for kind in ["availability", "dispatch"]:
            rows.append(
                quantile_shifts(
                    n0,
                    n1,
                    carriers,
                    label=group_label,
                    kind=kind,
                    max_assets=args.cf_max_assets,
                    weight_scale=args.cf_weight_scale,
                    n2=n2,
                )
            )

    qtab = pd.DataFrame(rows)
    qtab.to_csv(outdir / "cf_quantile_shifts.csv", index=False)

    # Summary
    summary = pd.DataFrame([{
        "objective_bias_off": objective(n0),
        "objective_bias_on": objective(n1),
        "objective_delta_pyvwf": objective(n1) - objective(n0),
        "objective_bias_uniform": (objective(n2) if n2 is not None else np.nan),
        "objective_delta_uniform": ((objective(n2) - objective(n0)) if n2 is not None else np.nan),
        "onshore_carriers": ",".join(onshore_carriers),
        "offshore_carriers": ",".join(offshore_carriers),
        "cf_max_assets": args.cf_max_assets,
        "cf_weight_scale": args.cf_weight_scale,
        "bias_off_name": nice("bias_off"),
        "bias_on_name": nice("bias_on"),
    }])
    summary.to_csv(outdir / "summary.csv", index=False)

    # --- New plots ---
    # System cost bar chart
    plot_system_cost_bar(n0, n1, n2, outpath=outdir / "system_cost_bar.png")

    # Duration curves: offshore and onshore
    plot_duration_curve(n0, n1, n2, carriers=offshore_carriers, label_text="Offshore wind", outpath=outdir / "duration_curve_offshore.png")
    plot_duration_curve(n0, n1, n2, carriers=onshore_carriers, label_text="Onshore wind", outpath=outdir / "duration_curve_onshore.png")
    # Combined wind
    all_wind_carriers = onshore_carriers + offshore_carriers
    plot_duration_curve(n0, n1, n2, carriers=all_wind_carriers, label_text="Total wind", outpath=outdir / "duration_curve_total_wind.png")

    # Bias correction spatial map
    if args.bias_file and Path(args.bias_file).exists():
        plot_bias_correction_map(args.bias_file, outpath=outdir / "bias_correction_map.png")
    else:
        for bias_file in ["bias-extra/atlite_bias.nc", "../bias-extra/atlite_bias.nc", "data/atlite_bias.nc"]:
            if Path(bias_file).exists():
                plot_bias_correction_map(bias_file, outpath=outdir / "bias_correction_map.png")
                break
        else:
            print("[INFO] No bias correction netCDF found; skipping spatial map.")

    print("Wrote outputs to:", outdir)
    print(summary.T)


if __name__ == "__main__":
    main()

# EXAMPLE USAGE:
# python compare_bias_runs_styled.py \
#   --raw     results/thesis-bias-2030-10-northsea-standard-6h/base-s100000-biasFalse/networks/base_s_10_elec_lvopt_.nc \
#   --corr    results/thesis-bias-2030-10-northsea-standard-6h/base-s100000-biasTrue/networks/base_s_10_elec_lvopt_.nc \
#   --uniform results/thesis-bias-2030-10-northsea-standard-6h/base-s100000-biasUniform/networks/base_s_10_elec_lvopt_.nc \
#   --out     plots/bias \
#   --top     30
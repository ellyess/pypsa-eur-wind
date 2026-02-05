#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, NamedTuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pypsa
import geopandas as gpd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from plotting_style import thesis_plot_style, apply_spatial_resolution_axis
from thesis_colors import WAKE_MODEL_COLORS, label, THESIS_LABELS, WAKE_ORDER
import seaborn as sns

try:
    from scripts.wake_helpers import (
        WakeSplitSpec,
        _glaum_spec,
        _new_more_spec,
    )
    WAKE_HELPERS_AVAILABLE = True
except ImportError:
    WAKE_HELPERS_AVAILABLE = False
    print("[WARN] wake_helpers not available; plot_wake_models_density_two_areas will not work.")

# Apply thesis-wide plotting style
_style = thesis_plot_style()
cm, lw, ms, dpi = _style['cm'], _style['lw'], _style['ms'], _style['dpi']
import pypsa


# -----------------------------
# Offshore carriers (PyPSA-Eur)
# -----------------------------
OFFSHORE_CARRIERS = ("offwind", "offwind-ac", "offwind-dc", "offwind-float")


# -----------------------------
# Scenario parsing: "<wake>-s<split>-bias<True/False>"
# e.g. "new_more-s1000-biasFalse"
# -----------------------------
_SCEN_RE = re.compile(r"^(?P<wake>.+)-s(?P<s>\d+)-bias(?P<bias>(True|False))$")


def parse_scenario_name(name: str) -> Dict[str, Any]:
    m = _SCEN_RE.match(name)
    if not m:
        # soft fallback
        s = None
        bias = None
        ms = re.search(r"-s(\d+)", name)
        if ms:
            s = int(ms.group(1))
        mb = re.search(r"-bias(True|False)", name)
        if mb:
            bias = (mb.group(1) == "True")
        return {"wake": name, "s": s, "bias": bias, "scenario": name}

    return {
        "wake": m.group("wake"),
        "s": int(m.group("s")),
        "bias": (m.group("bias") == "True"),
        "scenario": name,
    }


def find_network_files(prefix_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    Find sector-coupled networks in:
      results/<prefix>/<scenario>/postnetworks/*_lvopt_*.nc
    """
    out: List[Tuple[Path, Dict[str, Any]]] = []
    for scen_dir in sorted(prefix_dir.iterdir()):
        if not scen_dir.is_dir():
            continue
        meta = parse_scenario_name(scen_dir.name)
        net_dir = scen_dir / "postnetworks"
        if not net_dir.exists():
            continue
        for f in sorted(net_dir.glob("*_lvopt_*.nc")):
            out.append((f, meta))
    return out


# -----------------------------
# Color mapping
# -----------------------------
def wake_color_map(wakes: List[str], baseline: str = "base") -> Dict[str, Any]:
    """
    Use thesis-consistent wake colors from WAKE_MODEL_COLORS.
    Falls back to matplotlib colors if wake model not in palette.
    """
    prop_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    wakes_sorted = sorted(set(wakes))
    cmap: Dict[str, Any] = {}
    k = 0
    for w in wakes_sorted:
        # Try to get color from thesis palette first
        if w in WAKE_MODEL_COLORS:
            cmap[w] = WAKE_MODEL_COLORS[w]
        else:
            # Fallback to color cycle for unknown wake models
            cmap[w] = prop_cycle[k % len(prop_cycle)]
            k += 1
    return cmap


# -----------------------------
# Network helpers
# -----------------------------
def snap_weights(n: pypsa.Network) -> pd.Series:
    if hasattr(n, "snapshot_weightings") and isinstance(n.snapshot_weightings, pd.DataFrame):
        for col in ("generators", "objective"):
            if col in n.snapshot_weightings.columns:
                return n.snapshot_weightings[col].reindex(n.snapshots).fillna(1.0)
    return pd.Series(1.0, index=n.snapshots)


def objective(n: pypsa.Network) -> float:
    return float(getattr(n, "objective", np.nan))


def bus_country(bus_name: str) -> str:
    s = str(bus_name)
    return s[:2] if len(s) >= 2 else "??"


def offshore_generator_index(n: pypsa.Network) -> pd.Index:
    if not len(n.generators) or "carrier" not in n.generators.columns:
        return pd.Index([])
    car = n.generators["carrier"].fillna("").astype(str)
    return n.generators.index[car.str.startswith(OFFSHORE_CARRIERS)]


# -----------------------------
# Offshore metrics (capacity/energy/by-country)
# -----------------------------
def offshore_capacity_mw(n: pypsa.Network) -> float:
    idx = offshore_generator_index(n)
    if len(idx) == 0:
        return 0.0
    gens = n.generators.loc[idx]
    p_nom = gens["p_nom_opt"] if "p_nom_opt" in gens.columns else gens["p_nom"]
    return float(p_nom.sum())


def offshore_energy_twh(n: pypsa.Network) -> float:
    idx = offshore_generator_index(n)
    if len(idx) == 0 or not hasattr(n, "generators_t") or not hasattr(n.generators_t, "p"):
        return 0.0
    w = snap_weights(n)
    mwh = (n.generators_t.p[idx].mul(w, axis=0)).sum().sum()
    return float(mwh / 1e6)  # TWh


def offshore_energy_by_country_twh(n: pypsa.Network) -> pd.Series:
    idx = offshore_generator_index(n)
    if len(idx) == 0 or not hasattr(n.generators_t, "p"):
        return pd.Series(dtype=float, name="offshore_energy_TWh")

    gens = n.generators.loc[idx].copy()
    gens["country"] = gens["bus"].map(bus_country)

    w = snap_weights(n)
    mwh = (n.generators_t.p[idx].mul(w, axis=0)).sum(axis=0)  # per generator
    twh = (mwh.groupby(gens["country"]).sum() / 1e6).sort_values(ascending=False)
    twh.name = "offshore_energy_TWh"
    return twh


def offshore_capacity_by_country_mw(n: pypsa.Network) -> pd.Series:
    idx = offshore_generator_index(n)
    if len(idx) == 0:
        return pd.Series(dtype=float, name="offshore_capacity_MW")
    gens = n.generators.loc[idx].copy()
    gens["country"] = gens["bus"].map(bus_country)
    p_nom = gens["p_nom_opt"] if "p_nom_opt" in gens.columns else gens["p_nom"]
    s = p_nom.groupby(gens["country"]).sum().sort_values(ascending=False)
    s.name = "offshore_capacity_MW"
    return s


# -----------------------------
# CF time series (like your bias script)
# -----------------------------
def cf_timeseries(n: pypsa.Network, *, kind: str) -> pd.DataFrame:
    """
    Offshore CF time series:
      availability: generators_t.p_max_pu
      dispatch:     generators_t.p / p_nom_opt
    """
    idx = offshore_generator_index(n)
    if len(idx) == 0:
        return pd.DataFrame(index=n.snapshots)

    if kind == "availability":
        if not hasattr(n.generators_t, "p_max_pu") or n.generators_t.p_max_pu.empty:
            return pd.DataFrame(index=n.snapshots)
        return n.generators_t.p_max_pu[idx].copy()

    if kind == "dispatch":
        if not hasattr(n.generators_t, "p") or n.generators_t.p.empty:
            return pd.DataFrame(index=n.snapshots)
        p = n.generators_t.p[idx]
        gens = n.generators.loc[idx]
        p_nom = (gens["p_nom_opt"] if "p_nom_opt" in gens.columns else gens["p_nom"]).astype(float)
        cf = p.div(p_nom, axis=1)
        return cf

    raise ValueError(f"Unknown kind={kind!r}")


def curtailment_timeseries(n: pypsa.Network) -> pd.DataFrame:
    """
    Offshore curtailment fraction time series (per unit of available power):
        curtail = max(0, (p_max_pu - p/p_nom) / p_max_pu)
    where p_max_pu is the availability and p/p_nom is dispatch capacity factor.
    Values are clipped to [0,1] and defined as 0 when p_max_pu is ~0.
    """
    avail = cf_timeseries(n, kind="availability")
    disp = cf_timeseries(n, kind="dispatch")
    if avail.empty or disp.empty:
        return pd.DataFrame(index=n.snapshots)

    # Align columns (offshore generators) and index (snapshots)
    common = avail.columns.intersection(disp.columns)
    if len(common) == 0:
        return pd.DataFrame(index=n.snapshots)
    avail = avail[common]
    disp = disp[common]

    eps = 1e-6
    curtail = (avail - disp).clip(lower=0.0)
    curtail = curtail.divide(avail.where(avail > eps), axis=0)
    curtail = curtail.fillna(0.0).clip(lower=0.0, upper=1.0)
    return curtail



def _weighted_sample_values(
    cf: pd.DataFrame,
    w: pd.Series,
    *,
    max_assets: int,
    weight_scale: int,
    seed: int = 0,
) -> np.ndarray:
    """
    Same approach as your bias script:
      - optional generator subsample
      - approximate snapshot weighting via repetition (fast, stable)
    """
    if cf.empty:
        return np.array([])

    cf = cf.copy()

    if max_assets and cf.shape[1] > max_assets:
        rng = np.random.default_rng(seed)
        cols = rng.choice(cf.columns.to_numpy(), size=max_assets, replace=False)
        cf = cf.loc[:, cols]

    w = w.reindex(cf.index).fillna(1.0)
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


# -----------------------------
# Plotting: CDF + ΔCDF (vs baseline)
# -----------------------------
def plot_cdf_and_delta_multi(
    series: Dict[str, np.ndarray],
    *,
    title: str,
    baseline: str,
    out_cdf: Path,
    out_delta: Path,
    colors: Dict[str, Any],
):
    """
    series[wake] = sampled values in [0,1]
    Produces:
      - CDF plot (all wakes)
      - ΔCDF plot (each wake vs baseline)
    """
    # compute cdfs
    cdfs: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for wake, v in series.items():
        x, y = _empirical_cdf(v)
        if x.size:
            cdfs[wake] = (x, y)

    if baseline not in cdfs:
        out_delta.with_suffix(".txt").write_text(
            f"Skipped ΔCDF: baseline {baseline!r} missing.\n"
        )
        return

    # CDF (all)
    plt.figure(figsize=(5.2, 3.2))
    # Plot in defined order for consistent legend
    for wake in WAKE_ORDER:
        if wake not in cdfs:
            continue
        x, y = cdfs[wake]
        wake_label = label(wake, default=wake)
        plt.plot(x, y, label=wake_label, linewidth=2, color=colors.get(wake, None))
    plt.xlabel("Capacity factor")
    plt.ylabel("CDF")
    plt.title(title)
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    out_cdf.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_cdf, dpi=300)
    plt.close()

    # ΔCDF vs baseline on common grid
    xgrid = np.linspace(0, 1, 600)
    xb, yb = cdfs[baseline]
    cb = np.interp(xgrid, xb, yb)

    plt.figure(figsize=(5.2, 3.2))
    # Plot in defined order for consistent legend
    for wake in WAKE_ORDER:
        if wake == baseline or wake not in cdfs:
            continue
        x, y = cdfs[wake]
        cw = np.interp(xgrid, x, y)
        wake_label = label(wake, default=wake)
        plt.plot(xgrid, cw - cb, label=wake_label, linewidth=3, color=colors.get(wake, None))

    baseline_label = label(baseline, default=baseline)
    plt.axhline(0.0, linestyle="--", linewidth=1.0, color="black")
    plt.xlabel("Capacity factor")
    plt.ylabel("ΔCDF (vs. baseline)")
    plt.title(title.replace("CDF", "ΔCDF") + f" (baseline: {baseline_label})")
    plt.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    out_delta.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_delta, dpi=300)
    plt.close()


def quantile_table(values_by_wake: Dict[str, np.ndarray], baseline: str) -> pd.DataFrame:
    """
    p10/p50/p90 per wake and shifts vs baseline.
    """
    qs = [0.10, 0.50, 0.90]
    rows = []
    for wake, v in values_by_wake.items():
        v = v[np.isfinite(v)]
        if v.size == 0:
            rows.append({"wake": wake, "p10": np.nan, "p50": np.nan, "p90": np.nan})
            continue
        p = np.quantile(v, qs)
        rows.append({"wake": wake, "p10": p[0], "p50": p[1], "p90": p[2]})
    df = pd.DataFrame(rows).set_index("wake").sort_index()

    if baseline in df.index:
        for col in ["p10", "p50", "p90"]:
            df[f"{col}_shift_vs_{baseline}"] = df[col] - df.loc[baseline, col]
    return df


# -----------------------------
# Region capacity density calculation
# -----------------------------
class RegionDensityResult(NamedTuple):
    geodf: gpd.GeoDataFrame
    cap_col: str
    density_col: str


def _default_gen_region_parser(gen_index: pd.Index) -> pd.Series:
    """
    Default region parser matching PyPSA generator naming:
      generators.index like "NO4 0_00022 offwind-ac ..." -> region "NO4 0"
    i.e. take first two whitespace-separated tokens.
    """
    s = gen_index.to_series().astype(str)
    return s.str.split().str[:2].str.join(" ")


def build_region_capacity_density_geodf(
    n: pypsa.Network,
    *,
    split: int,
    area: str,
    regions_dir: Union[str, Path] = "wake_extra",
    carrier_filter: str = "offwind",
    regions_name_col: str = "name",
    regions_region_col: str = "region",
    cap_field: str = "p_nom_opt",
    region_parser=None,
    target_crs_for_area: int = 3035,
) -> RegionDensityResult:
    """
    Build a GeoDataFrame with per-region capacity and capacity density.

    Returns a GeoDataFrame with at least:
        - geometry
        - area_km2
        - <cap_field> (summed by region)
        - density_mw_per_km2

    Notes
    -----
    - Density is computed as: (capacity_MW)/area_km2.
      PyPSA stores p_nom_opt/p_nom_max in MW by default, so we use MW/km².
    """
    if region_parser is None:
        region_parser = _default_gen_region_parser

    regions_dir = Path(regions_dir)

    gens = n.generators.copy()

    # --- filter to carrier (offshore wind) ---
    if "carrier" in gens.columns:
        gens = gens[gens["carrier"].str.startswith(carrier_filter)]

    if gens.empty:
        raise ValueError(
            f"No generators matched carrier='{carrier_filter}'."
        )

    if cap_field not in gens.columns:
        raise KeyError(f"Generator table does not contain '{cap_field}'. Available: {list(gens.columns)}")

    # --- assign region ---
    gens[regions_region_col] = region_parser(gens.index)

    # --- aggregate capacity per region ---
    cap_by_region = (
        gens.groupby(regions_region_col, dropna=False)[cap_field]
        .sum()
        .rename(cap_field)
        .to_frame()
    )

    # --- load regions geometry ---
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

    # --- join capacity into regions ---
    geodf = regions.merge(cap_by_region, on=regions_region_col, how="left")
    
    # Fill NaN capacity values with 0 (regions with no generators)
    geodf[cap_field] = geodf[cap_field].fillna(0.0)

    # --- compute area_km2 (robustly) ---
    if "area_km2" not in geodf.columns:
        try:
            geodf["area_km2"] = geodf.geometry.to_crs(target_crs_for_area).area / 1e6
        except Exception as e:
            raise RuntimeError(
                "Failed to compute area_km2. Ensure geometries are valid and have a CRS."
            ) from e

    # --- compute density: MW/km² (PyPSA p_nom_* is in MW) ---
    density_col = "density_mw_per_km2"
    geodf[density_col] = geodf[cap_field] / geodf["area_km2"]

    return RegionDensityResult(geodf=geodf, cap_col=cap_field, density_col=density_col)


def offshore_capacity_density_by_region(
    n: pypsa.Network,
    *,
    split: int,
    area: str,
    wake_extra_dir: Path = Path("wake_extra"),
) -> tuple[pd.Series, gpd.GeoDataFrame | None]:
    """
    Extract optimal capacity density (MW/km²) per offshore region using robust implementation.
    
    Args:
        n: PyPSA network
        split: Split value (e.g., 1000, 5000) for region file
        area: Area name (e.g., 'northsea')
        wake_extra_dir: Base directory for wake_extra files
    
    Returns:
        - Series indexed by region name with capacity density values
        - GeoDataFrame with region geometries and density data
    """
    try:
        result = build_region_capacity_density_geodf(
            n,
            split=split,
            area=area,
            regions_dir=wake_extra_dir,
            carrier_filter="offwind",
            cap_field="p_nom_opt",
        )
        
        # Extract density series from geodf
        density = result.geodf.set_index("region")[result.density_col]
        density.name = "capacity_density_MW_per_km2"
        
        return density.sort_values(ascending=False), result.geodf
        
    except Exception as e:
        print(f"[WARN] Failed to calculate regional capacity density: {e}")
        return pd.Series(dtype=float, name="capacity_density_MW_per_km2"), None




# -----------------------------
# Plotting: distributions vs spatial resolution (Figure-style summary)
# -----------------------------
def _summary_quantiles(v: np.ndarray, qs=(0.05, 0.25, 0.50, 0.75, 0.95)) -> dict:
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {f"q{int(q*100):02d}": np.nan for q in qs}
    out = np.quantile(v, qs)
    return {f"q{int(q*100):02d}": float(out[i]) for i, q in enumerate(qs)}


def plot_output_distributions(
    df: pd.DataFrame,
    *,
    outpath: Path,
    baseline_wake: str,
    s_values: list[int] | None = None,
    wakes: list[str] | None = None,
):
    """
    Produce a two-panel summary similar to fig:distributions:

      (left) dispatch capacity-factor distribution by wake model and spatial resolution
      (right) curtailment fraction distribution by wake model and spatial resolution

    Each glyph shows:
      - whiskers: 5th–95th percentiles
      - thick bar: 25th–75th percentiles
      - marker: median (50th percentile)

    Uses thesis colors and axis formatting helpers.
    """
    if s_values is None:
        s_values = sorted([int(v) for v in df["s"].dropna().unique()])
    if not s_values:
        return

    if wakes is None:
        # Use defined wake order when possible; otherwise fall back to unique wakes.
        present = list(df["wake"].dropna().unique())
        wakes = [w for w in WAKE_ORDER if w in present] + [w for w in present if w not in WAKE_ORDER]
    if not wakes:
        return

    # Color + labels
    colors = wake_color_map(wakes, baseline=baseline_wake)

    # x positions per spatial resolution
    x = np.arange(len(s_values), dtype=float)
    n_w = len(wakes)
    # keep groups compact and readable
    group_width = 0.72
    step = group_width / max(n_w, 1)
    offsets = (np.arange(n_w) - (n_w - 1) / 2.0) * step

    fig, (ax_cf, ax_curt) = plt.subplots(1, 2, figsize=(12.0 * cm, 4.5 * cm), dpi=dpi, layout="constrained")

    def _draw(ax, xpos, stats, color):
        # whisker (q05–q95)
        ax.vlines(xpos, stats["q05"], stats["q95"], color=color, linewidth=2.0, alpha=0.95, zorder=2)
        # IQR (q25–q75)
        ax.vlines(xpos, stats["q25"], stats["q75"], color=color, linewidth=8.0, alpha=0.70, zorder=3)
        # median marker
        ax.plot([xpos], [stats["q50"]], marker="o", markersize=ms, color=color, zorder=4)

    # Build and plot stats
    for i_s, s in enumerate(s_values):
        sub_s = df[df["s"] == s]
        for i_w, w in enumerate(wakes):
            sub = sub_s[sub_s["wake"] == w]
            # concatenate all sampled arrays for this (s, wake)
            v_cf = []
            v_curt = []
            for arr in sub.get("_cf_dispatch", []):
                if isinstance(arr, np.ndarray) and arr.size:
                    v_cf.append(arr)
            for arr in sub.get("_curtailment", []):
                if isinstance(arr, np.ndarray) and arr.size:
                    v_curt.append(arr)

            v_cf = np.concatenate(v_cf) if v_cf else np.array([])
            v_curt = np.concatenate(v_curt) if v_curt else np.array([])

            stats_cf = _summary_quantiles(v_cf)
            stats_curt = _summary_quantiles(v_curt)

            xpos = x[i_s] + offsets[i_w]
            _draw(ax_cf, xpos, stats_cf, colors.get(w, None))
            _draw(ax_curt, xpos, stats_curt, colors.get(w, None))

    # Axes formatting
    ax_cf.set_ylabel("Capacity factor")
    ax_curt.set_ylabel("Curtailment fraction")
    for ax in (ax_cf, ax_curt):
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s:,}" for s in s_values])
        ax.set_xlabel(r"Spatial Resolution ($A_{region}^{max}$) [km$^2$]")

    # Legend across top
    handles = []
    for w in wakes:
        handles.append(Line2D([0], [0], color=colors.get(w, None), linewidth=8, solid_capstyle="round", label=label(w, default=w)))
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 4), frameon=False, title="Wake model", bbox_to_anchor=(0.5, 1.05))

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_capacity_density_by_region(
    geodf_by_wake: Dict[str, gpd.GeoDataFrame],
    *,
    outpath: Path,
    baseline_wake: str,
):
    """
    Plot optimal capacity density difference (Δ) per region across wake models as spatial maps.
    
    Creates a multi-panel figure showing capacity density delta (MW/km²) relative to baseline,
    one panel per wake model (excluding baseline itself).
    Uses diverging colormap (RdBu_r) centered at 0 to show increases/decreases.
    """
    if not geodf_by_wake:
        return
    
    # Check baseline exists
    if baseline_wake not in geodf_by_wake:
        print(f"[WARN] Baseline wake '{baseline_wake}' not found in data. Skipping density plot.")
        return
    
    # Order wakes consistently, excluding baseline
    wakes = [w for w in WAKE_ORDER if w in geodf_by_wake and w != baseline_wake]
    if not wakes:
        return
    
    # Get baseline density
    baseline_gdf = geodf_by_wake[baseline_wake]
    if "density_mw_per_km2" not in baseline_gdf.columns:
        print(f"[WARN] density_mw_per_km2 column missing in baseline GeoDataFrame")
        return
    
    # Compute deltas for each wake model
    delta_gdfs = {}
    all_delta_values = []
    
    for wake in wakes:
        gdf = geodf_by_wake[wake].copy()
        if "density_mw_per_km2" not in gdf.columns:
            continue
        
        # Compute delta (assumes regions are in same order)
        gdf["delta_density"] = gdf["density_mw_per_km2"] - baseline_gdf["density_mw_per_km2"]
        delta_gdfs[wake] = gdf
        
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
        dpi=dpi, 
        sharex=True, 
        sharey=True, 
        layout="constrained"
    )
    
    if n_panels == 1:
        axes = [axes]
    
    for i, (ax, wake) in enumerate(zip(axes, delta_gdfs.keys())):
        gdf = delta_gdfs[wake]
        
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
            legend_kwds={
                "label": r"$\Delta\rho_{A_{region}}^{opt}$ [MW/km$^2$]",
                "orientation": "vertical",
                "pad": 0.2,
                "shrink": 1.0,
            } if show_legend else None,
        )
        
        # Title with nice label
        ax.set_title(label(wake, default=wake), fontsize=9)
        ax.set_aspect("equal")
        ax.axis("off")
    
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Wake methods plotting helpers
# -----------------------------
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


def wake_losses(*args, **kwargs):
    """
    Placeholder for your wake-loss specific analysis function.

    Your original repository likely had a custom implementation here.
    Keep using it if you already have it elsewhere; otherwise adapt.
    """
    raise NotImplementedError("wake_losses() is project-specific; plug in your existing implementation.")


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
    2×2 review-proof wake figure where ALL panels use capacity density x (MW/km²).

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

    sns.set_theme(style="ticks")
    sns.despine()

    # --- specs from your wake methods ---
    spec_cap = _glaum_spec()
    spec_den, x_breaks_den = _new_more_spec()  # x_breaks_den ends at 4 in your method
    M_den = np.asarray(spec_den.factors, dtype=float)

    # --- grids ---
    # avoid 0 for log scale, but still show up to x_max
    x_grid = np.geomspace(1e-3, x_max, 1200)

    # Use thesis color scheme from thesis_colors.py
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
        ax_T.set_xlabel("Installed Capacity Density [MW/km²]")
        ax_T.set_ylabel("Loss (fraction)")

        # --- MARGINAL LOSS (density axis) ---
        # Build robust step arrays that reach x_max
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
        ax_M.set_xlabel("Installed Capacity Density [MW/km²]")
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

    _plot_column(ax_TL, ax_ML, A_left_km2, col_title=rf"$A_{{region}}$ = {A_left_km2:,.0f} km²")
    _plot_column(ax_TR, ax_MR, A_right_km2, col_title=rf"$A_{{region}}$ = {A_right_km2:,.0f} km²")

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

    fig.legend(H, L, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, required=True)
    p.add_argument("--prefix", type=str, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--only-bias", type=str, default=None, help="Filter bias True/False (e.g. 'False')")
    p.add_argument("--baseline-wake", type=str, default="base", help="Wake model used as Δ baseline")
    p.add_argument("--max-networks", type=int, default=0, help="Debug limit (0 = no limit)")
    p.add_argument("--wake-extra-dir", type=Path, default="wake_extra", help="Directory containing wake_extra region files")

    # CF sampling controls (same spirit as your bias script)
    p.add_argument("--max-assets", type=int, default=2500, help="Max offshore generators sampled per network")
    p.add_argument("--weight-scale", type=int, default=3, help="Snapshot weight repetition scale")

    args = p.parse_args()

    prefix_dir = args.results_root / args.prefix
    if not prefix_dir.exists():
        raise SystemExit(f"Prefix dir not found: {prefix_dir}")
    
    # Extract area from prefix (e.g., "thesis-wake-2030-10-northsea-dominant-6h" -> "northsea")
    prefix_parts = args.prefix.split("-")
    area = None
    for part in prefix_parts:
        if part in ["northsea", "europe", "baltic", "combined", "standard", "dominant"]:
            if part not in ["combined", "standard", "dominant"]:
                area = part
                break
    
    if area is None:
        # Fallback: try to find "northsea" or "europe" anywhere in prefix
        if "northsea" in args.prefix:
            area = "northsea"
        elif "europe" in args.prefix:
            area = "europe"
        else:
            print(f"[WARN] Could not extract area from prefix '{args.prefix}', using 'northsea' as default")
            area = "northsea"

    nets = find_network_files(prefix_dir)
    if not nets:
        raise SystemExit(f"No *_elec_*.nc found under: {prefix_dir}")

    if args.max_networks and args.max_networks > 0:
        nets = nets[: args.max_networks]

    rows = []
    for nc_path, meta in nets:
        try:
            n = pypsa.Network(nc_path)
        except Exception as e:
            print(f"[WARN] Failed to load {nc_path}: {e}")
            continue

        row = dict(meta)
        row["network_file"] = str(nc_path)

        # offshore metrics
        row["objective"] = objective(n)
        row["offshore_capacity_MW"] = offshore_capacity_mw(n)
        row["offshore_energy_TWh"] = offshore_energy_twh(n)

        # optional: by-country breakdowns (saved as separate CSVs later)
        row["_offshore_cap_by_country"] = offshore_capacity_by_country_mw(n)
        row["_offshore_en_by_country"] = offshore_energy_by_country_twh(n)
        
        # Capacity density by region (returns both density and geometry)
        split = meta.get("s")
        
        if split is not None and area is not None:
            density, gdf = offshore_capacity_density_by_region(
                n,
                split=split,
                area=area,
                wake_extra_dir=args.wake_extra_dir,
            )
            row["_capacity_density_by_region"] = density
            row["_regions_gdf"] = gdf
            
            # Debug: check if gdf has the density column
            if gdf is not None:
                if "density_mw_per_km2" in gdf.columns:
                    n_non_zero = (gdf["density_mw_per_km2"] > 0).sum()
                    print(f"[DEBUG] {meta['scenario']}: GeoDataFrame has {len(gdf)} regions, {n_non_zero} with non-zero density")
                else:
                    print(f"[WARN] {meta['scenario']}: GeoDataFrame missing density_mw_per_km2 column! Columns: {gdf.columns.tolist()}")
        else:
            print(f"[WARN] Missing split or area for capacity density calculation")
            row["_capacity_density_by_region"] = pd.Series(dtype=float)
            row["_regions_gdf"] = None

        # CF sampled values (availability + dispatch)
        w = snap_weights(n)
        for kind in ("availability", "dispatch"):
            cf = cf_timeseries(n, kind=kind)
            v = _weighted_sample_values(
                cf,
                w,
                max_assets=args.max_assets,
                weight_scale=args.weight_scale,
                seed=0,
            )
            row[f"_cf_{kind}"] = v

        # Curtailment fraction sampled values (per unit of availability)
        curtail = curtailment_timeseries(n)
        v_curt = _weighted_sample_values(
            curtail,
            w,
            max_assets=args.max_assets,
            weight_scale=args.weight_scale,
            seed=0,
        )
        row["_curtailment"] = v_curt


        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No networks loaded successfully.")

    if args.only_bias is not None:
        want = args.only_bias.strip()
        if want in ("True", "False"):
            df = df[df["bias"] == (want == "True")].copy()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # # Save index CSV (no big arrays)
    # idx_cols = [c for c in df.columns if not c.startswith("_")]
    # df[idx_cols].to_csv(out_dir / "wake_offshore_index.csv", index=False)

    # # ---- Metric summary + deltas vs baseline per s
    # metric_cols = ["objective", "offshore_capacity_MW", "offshore_energy_TWh"]
    # all_metric_rows = []

    # for s in sorted([int(v) for v in df["s"].dropna().unique()]):
    #     sub = df[df["s"] == s].copy()
    #     if sub.empty:
    #         continue

    #     # metrics wide by wake
    #     metrics = (
    #         sub.groupby("wake")[metric_cols]
    #         .mean(numeric_only=True)
    #         .sort_index()
    #     )

    #     # deltas vs baseline wake
    #     if args.baseline_wake in metrics.index:
    #         base = metrics.loc[args.baseline_wake]
    #         delta = metrics.sub(base, axis=1)
    #         delta.columns = [f"delta_{c}" for c in delta.columns]
    #         metrics_out = pd.concat([metrics, delta], axis=1)
    #     else:
    #         metrics_out = metrics

    #     metrics_out.insert(0, "s", s)
    #     metrics_out.to_csv(out_dir / f"metrics_offshore_s{s}.csv")
    #     all_metric_rows.append(metrics_out.reset_index().rename(columns={"index": "wake"}))

    # if all_metric_rows:
    #     pd.concat(all_metric_rows, ignore_index=True).to_csv(out_dir / "metrics_offshore_all_s.csv", index=False)

    # # ---- By-country deltas (capacity + energy) per s (optional but useful)
    # for s in sorted([int(v) for v in df["s"].dropna().unique()]):
    #     sub = df[df["s"] == s].copy()
    #     if sub.empty:
    #         continue

    #     cap_by_wake = {r["wake"]: r["_offshore_cap_by_country"] for _, r in sub.iterrows()}
    #     en_by_wake = {r["wake"]: r["_offshore_en_by_country"] for _, r in sub.iterrows()}

    #     # average if multiple networks per wake (rare)
    #     def _mean_series_dict(dct):
    #         out = {}
    #         for k, v in dct.items():
    #             out.setdefault(k, []).append(v)
    #         return {k: pd.concat(vs, axis=1).mean(axis=1) for k, vs in out.items()}

    #     cap_by_wake = _mean_series_dict(cap_by_wake)
    #     en_by_wake = _mean_series_dict(en_by_wake)

    #     cap_df = pd.concat(cap_by_wake, axis=1).fillna(0.0)
    #     en_df = pd.concat(en_by_wake, axis=1).fillna(0.0)

    #     if args.baseline_wake in cap_df.columns:
    #         for k in cap_df.columns:
    #             if k == args.baseline_wake:
    #                 continue
    #             cap_df[f"delta_{k}"] = cap_df[k] - cap_df[args.baseline_wake]
    #     if args.baseline_wake in en_df.columns:
    #         for k in en_df.columns:
    #             if k == args.baseline_wake:
    #                 continue
    #             en_df[f"delta_{k}"] = en_df[k] - en_df[args.baseline_wake]

    #     cap_df.to_csv(out_dir / f"offshore_capacity_by_country_s{s}.csv")
    #     en_df.to_csv(out_dir / f"offshore_energy_by_country_s{s}.csv")

    # # ---- CF CDF + ΔCDF plots (availability + dispatch), one figure per s
    # for s in sorted([int(v) for v in df["s"].dropna().unique()]):
    #     sub = df[df["s"] == s].copy()
    #     if sub.empty:
    #         continue

    #     wakes = sorted(sub["wake"].unique())
    #     colors = wake_color_map(wakes, baseline=args.baseline_wake)

    #     for kind in ("availability", "dispatch"):
    #         values_by_wake = {}
    #         for wake, grp in sub.groupby("wake"):
    #             vals = [v for v in grp[f"_cf_{kind}"].values if isinstance(v, np.ndarray) and v.size]
    #             if vals:
    #                 values_by_wake[wake] = np.concatenate(vals)
    #             else:
    #                 values_by_wake[wake] = np.array([])

    #         # Save quantiles table (like your bias script)
    #         q = quantile_table(values_by_wake, baseline=args.baseline_wake)
    #         q.to_csv(out_dir / f"quantiles_offshore_cf_{kind}_s{s}.csv")

    #         # Use label function for nice titles
    #         kind_label = "Availability" if kind == "availability" else "Dispatch"
    #         baseline_label = label(args.baseline_wake, default=args.baseline_wake)
            
    #         # Plots
    #         title = f"Offshore wind CF {kind_label} (s{s})"
    #         out_cdf = out_dir / f"cdf_offshore_cf_{kind}_s{s}.png"
    #         out_delta = out_dir / f"delta_cdf_offshore_cf_{kind}_s{s}.png"

    #         plot_cdf_and_delta_multi(
    #             values_by_wake,
    #             title=title,
    #             baseline=args.baseline_wake,
    #             out_cdf=out_cdf,
    #             out_delta=out_delta,
    #             colors=colors,
    #         )

    #         print(f"[OK] s{s} {kind}: {out_delta}")

    # # ---- Output distributions summary plot (CF + curtailment vs resolution)
    # s_values = sorted([int(v) for v in df["s"].dropna().unique()])
    # plot_output_distributions(
    #     df,
    #     outpath=out_dir / "fig_distributions_summary.png",
    #     baseline_wake=args.baseline_wake,
    #     s_values=s_values,
    # )
    # print(f"[OK] Distributions summary: {out_dir / 'fig_distributions_summary.png'}")
    
    # # ---- Capacity density by region plots (one per s)
    # for s in s_values:
    #     sub = df[df["s"] == s].copy()
    #     if sub.empty:
    #         continue
        
    #     geodf_by_wake = {}
        
    #     for wake, grp in sub.groupby("wake"):
    #         gdfs = [g for g in grp["_regions_gdf"].values 
    #                 if g is not None]
            
    #         # Use geometry from first available network
    #         if gdfs:
    #             geodf_by_wake[wake] = gdfs[0]
        
    #     if geodf_by_wake:
    #         # Debug: print columns from first geodf
    #         first_wake = list(geodf_by_wake.keys())[0]
    #         first_gdf = geodf_by_wake[first_wake]
    #         print(f"[DEBUG] s{s} GeoDataFrame columns: {first_gdf.columns.tolist()}")
    #         if "density_mw_per_km2" in first_gdf.columns:
    #             print(f"[DEBUG] density_mw_per_km2 range: {first_gdf['density_mw_per_km2'].min():.2f} to {first_gdf['density_mw_per_km2'].max():.2f}")
    #         else:
    #             print(f"[WARN] density_mw_per_km2 column missing!")
            
    #         outpath = out_dir / f"capacity_density_by_region_s{s}.png"
    #         plot_capacity_density_by_region(
    #             geodf_by_wake,
    #             outpath=outpath,
    #             baseline_wake=args.baseline_wake,
    #         )
    #         print(f"[OK] s{s} capacity density delta: {outpath}")
            
    #         # Save density values as CSV (extract from geodfs)
    #         csv_data = {}
    #         index_col = None
    #         for wake, gdf in geodf_by_wake.items():
    #             if "density_mw_per_km2" in gdf.columns:
    #                 # Try to find the region identifier column
    #                 for col in ["region", "name"]:
    #                     if col in gdf.columns:
    #                         if index_col is None:
    #                             index_col = col
    #                         csv_data[wake] = gdf.set_index(col)["density_mw_per_km2"]
    #                         break
            
    #         if csv_data and index_col:
    #             csv_path = out_dir / f"capacity_density_by_region_s{s}.csv"
    #             pd.DataFrame(csv_data).to_csv(csv_path)
    #             print(f"[OK] s{s} capacity density CSV (indexed by '{index_col}'): {csv_path}")
    #         else:
    #             print(f"[WARN] s{s} could not save density CSV - missing density_mw_per_km2 or region column")

    # ---- Wake models comparison plot (methodology figure)
    if WAKE_HELPERS_AVAILABLE:
        try:
            wake_methods_plot = out_dir / "wake_models_density_comparison.png"
            plot_wake_models_density_two_areas(
                A_left_km2=1000.0,
                A_right_km2=10000.0,
                x_max=4.0,
                alpha_uniform=0.8855,
                savepath=str(wake_methods_plot),
                dpi=600,
            )
            print(f"[OK] Wake models comparison: {wake_methods_plot}")
        except Exception as e:
            print(f"[WARN] Failed to generate wake models comparison plot: {e}")
    else:
        print("[INFO] Skipping wake models comparison plot (wake_helpers not available)")

    print(f"[DONE] Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
    
# Example usage:
# python compare_wake_runs_styled.py \
#   --results-root results \
#   --prefix thesis-wake-2030-10-northsea-dominant-6h \
#   --out-dir plots/wakes \
#   --only-bias False

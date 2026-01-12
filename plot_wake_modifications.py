# plot_wake_modifications.py
from __future__ import annotations

import itertools
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Optional dependencies used by some functions
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # noqa: F401
except Exception:  # pragma: no cover
    sns = None  # type: ignore

try:
    import geopandas as gpd  # noqa: F401
except Exception:  # pragma: no cover
    gpd = None  # type: ignore

try:
    import pypsa  # noqa: F401
except Exception as e:  # pragma: no cover
    raise ImportError("This module requires pypsa to be installed.") from e

try:
    from shapely.errors import ShapelyDeprecationWarning  # noqa: F401
except Exception:  # pragma: no cover
    ShapelyDeprecationWarning = Warning  # type: ignore
    
from scripts.wake_helpers import (
    WakeSplitSpec,
    _glaum_spec,
    _new_more_spec,
)


# -----------------------------------------------------------------------------
# Plot styling (optional; call from notebook/script)
# -----------------------------------------------------------------------------

def setup_plot_style() -> None:
    """Set consistent matplotlib/seaborn style for thesis figures."""
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

    # Avoid global seaborn dependency if not installed
    if sns is not None:
        custom_params = {
            "xtick.bottom": True,
            "axes.edgecolor": "black",
            "axes.spines.right": False,
            "axes.spines.top": False,
            "mathtext.default": "regular",
        }
        sns.set_theme(style="ticks", rc=custom_params)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "mathtext.default": "it",
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
        }
    )
    plt.rc("xtick", labelsize=9)
    plt.rc("ytick", labelsize=9)


# -----------------------------------------------------------------------------
# Scenario parsing / naming
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ScenarioSpec:
    wake_model: str
    bias: str
    region_max: float
    raw: str


# Expected format: <wake>-s<region_max>-<bias>
_SCENARIO_RE = re.compile(
    r"^(?P<wake>[^-]+)-s(?P<region>\d+(\.\d+)?)-(?P<bias>[^-]+)$"
)



def parse_scenario(s: str) -> ScenarioSpec:
    """Parse scenario name '<wake>-s<region_max>-<bias>'."""
    m = _SCENARIO_RE.match(s)
    if not m:
        raise ValueError(f"Scenario '{s}' must match '<wake>-s<region_max>-<bias>'")
    return ScenarioSpec(
        wake_model=m.group("wake"),
        bias=m.group("bias"),
        region_max=float(m.group("region")),
        raw=s,
    )


def scenario_list(
    models: Sequence[str],
    splits: Sequence[float],
    biases: Sequence[str],
) -> list[str]:
    """All '<wake>-s<region_max>-<bias>' combinations."""
    out: list[str] = []
    for m, s, b in itertools.product(models, splits, biases):
        s_txt = str(int(s)) if float(s).is_integer() else str(float(s))
        out.append(f"{m}-s{s_txt}-{b}")
    return out



def network_path(
    *,
    results_dir: Path,
    prefix: str,
    scenario: str,
    clusters: int,
    year: int = 2030,
    variant: str = "base",
) -> Path:
    """Build standard postnetwork file path."""
    return (
        results_dir
        / prefix
        / scenario
        / "postnetworks"
        / f"{variant}_s_{clusters}_lvopt___{year}.nc"
    )


def filter_by_bias(results: pd.DataFrame, bias: str) -> pd.DataFrame:
    """Convenience helper to subset a results table for a single bias."""
    if "bias" not in results.columns:
        raise KeyError("results does not contain a 'bias' column. Did you run clean_results()?")
    return results.loc[results["bias"] == bias]


# -----------------------------------------------------------------------------
# Names / labels
# -----------------------------------------------------------------------------

_WAKE_LABELS = {
    "base": r"No-wake",
    "standard": r"Uniform scaling",
    "glaum": r"Tiered capacity",
    "new_more": r"Tiered density",
}

_BIAS_LABELS = {
    "biasTrue": "Bias corrected",   # change to r"\nobias{}" if you make a macro
    "biasFalse": "No bias correction",
}


def nice_names_for_plotting(scenario: str) -> str:
    """
    Human-readable label for '<wake>-s<region_max>-<bias>' scenarios.
    """
    spec = parse_scenario(scenario)

    wake_label = _WAKE_LABELS.get(spec.wake_model, spec.wake_model)
    bias_label = _BIAS_LABELS.get(spec.bias, spec.bias)

    # Keep your original (a)(b)(c)(d) ordering where possible
    tag = ""
    if spec.wake_model == "base":
        tag = "(a) "
    elif spec.wake_model == "standard":
        tag = "(b) "
    elif spec.wake_model == "glaum":
        tag = "(c) "
    elif spec.wake_model == "new_more":
        tag = "(d) "

    area = int(spec.region_max) if float(spec.region_max).is_integer() else spec.region_max
    return (
        f"{tag}{wake_label}\n"
        # f"{bias_label}\n"
        # + r"$A_{region}^{max}:$ " + f"{area:,}" + r" km$\mathrm{^{2}}$"
    )


def nice_names_for_plotting_label(models: Sequence[str]) -> list[str]:
    """Wake-model label list (for secondary axis)."""
    labels = []
    for m in models:
        labels.append(_WAKE_LABELS.get(m, m))
    return labels


# -----------------------------------------------------------------------------
# Results table construction
# -----------------------------------------------------------------------------

def clean_results(
    n: "pypsa.Network",
    dfs: list[pd.DataFrame],
    scenarios: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean and pivot PyPSA statistics output.

    Assumes PyPSA statistics index structure (common across versions):
      level 0 → component (e.g. 'Generator')
      level 1 → carrier
    """
    results = pd.concat(dfs, axis=0)

    # Explicitly name index levels by position
    if isinstance(results.index, pd.MultiIndex) and results.index.nlevels >= 2:
        results.index = results.index.set_names(
            ["component", "carrier"] + list(results.index.names[2:])
        )
    else:
        raise ValueError(
            "Unexpected PyPSA statistics format:\n"
            f"Index type: {type(results.index)}"
        )

    # Pivot to (scenario, scenario_nice) × (metric, carrier)
    results = (
        results
        .reset_index("carrier")
        .pivot(index=["scenario", "scenario_nice"], columns="carrier")
    )

    # Order scenarios explicitly
    order = {sc: i for i, sc in enumerate(scenarios)}
    order_key = results.index.get_level_values("scenario").map(order)
    results = results.iloc[order_key.argsort()[::-1]]

    # Align carrier colours safely
    carriers = results.columns.get_level_values("carrier")
    carrier_colors = (
        n.carriers
        .set_index("nice_name")["color"]
        .reindex(carriers)
    )

    # sanitize invalid / missing colours
    carrier_colors = (
        carrier_colors
        .fillna("lightgrey")
        .replace("", "lightgrey")
        .replace(" ", "lightgrey")
    )


    # Parse scenario metadata (wake, bias, resolution)
    specs = results.index.get_level_values("scenario").map(parse_scenario)

    results = results.copy()
    results["wake_model"] = specs.map(lambda x: x.wake_model)
    results["bias"] = specs.map(lambda x: x.bias)
    results["region_max"] = specs.map(lambda x: x.region_max)

    return results, carrier_colors


def results_dataframe(
    clusters: int,
    models: Sequence[str],
    splits: Sequence[float],
    biases: Sequence[str],
    prefix: str,
    results_dir: Path = Path("results"),
    year: int = 2030,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load PyPSA networks for each scenario and return:
      - results: a pivoted DataFrame (indexed by scenario/scenario_nice)
      - colors: a Series of carrier colors aligned to columns
    """
    scenarios = scenario_list(models, splits, biases)
    dfs: list[pd.DataFrame] = []

    reference_network: Optional["pypsa.Network"] = None

    for sc in scenarios:
        npath = network_path(
            results_dir=results_dir, prefix=prefix, scenario=sc, clusters=clusters, year=year
        )
        n = pypsa.Network(str(npath))
        reference_network = reference_network or n

        df = (
            n.statistics()
            .filter(regex="Generator", axis=0)[["Energy Balance", "Optimal Capacity", "Capacity Factor"]]
            .copy()
        )
        df["Energy Balance"] = df["Energy Balance"] / 1e6  # -> TWh
        df["Optimal Capacity"] = df["Optimal Capacity"] / 1e3  # -> GW
        df["scenario"] = sc
        df["scenario_nice"] = nice_names_for_plotting(sc)
        dfs.append(df)

    if reference_network is None:
        raise ValueError("No scenarios generated; check models/splits/biases inputs.")

    results, colors = clean_results(reference_network, dfs, scenarios)
    return results, colors


# -----------------------------------------------------------------------------
# Plotting helpers (used by all plotting functions)
# -----------------------------------------------------------------------------

cm = 1 / 2.54


def _var_meta(var: str) -> tuple[str, str]:
    """Return (save_stub, xlabel) for a given variable name."""
    if var == "Optimal Capacity":
        return "p_opt", f"{var} [GW]"
    if var == "Energy Balance":
        return "e_opt", f"{var} [TWh]"
    if var == "Capacity Factor":
        return "cf_opt", f"{var}"
    return var.lower().replace(" ", "_"), var


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: plt.Figure, savepath: Optional[Union[str, Path]], dpi: int = 600) -> None:
    if savepath is None:
        return
    sp = Path(savepath)
    _ensure_parent(sp)
    fig.savefig(sp, bbox_inches="tight", dpi=dpi)


def _metric_slice(results: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Slice a results table for a given metric.

    Works with MultiIndex columns: (metric, carrier).
    """
    if not isinstance(results.columns, pd.MultiIndex):
        # Fallback: maybe already flat columns
        if var in results.columns:
            out = results[[var]].copy()
            return out
        raise KeyError(f"Expected MultiIndex columns with metrics, but got {type(results.columns)}")
    # metric is level 0 in our pivot (because pivot columns='carrier' keeps original columns at top level)
    return results.xs(var, axis=1, level=0)


def _apply_scenario_yaxis_styling(
    ax: plt.Axes,
    results: pd.DataFrame,
) -> None:
    """
    Apply y-axis labeling for scenarios.

    If multiple biases exist, group by (wake_model, bias); otherwise group by wake_model only.
    """
    # Basic labels: region max (and bias if multiple)
    if "bias" in results.columns and results["bias"].nunique() > 1:
        labels_y = (
            results[["bias", "region_max"]]
            .assign(region_max=lambda d: d["region_max"].map("{:,.0f}".format))
            .astype(str)
            .agg(" | ".join, axis=1)
        )
        sec_ylabel = r"Wake Model | Bias | Spatial Resolution ($A_{region}^{max}$) [km$\mathrm{^{2}}$]"
    else:
        labels_y = results["region_max"].map("{:,.0f}".format)
        sec_ylabel = r"Wake Model | Spatial Resolution ($A_{region}^{max}$) [km$\mathrm{^{2}}$]"

    ax.set_yticklabels(list(labels_y))

    # Grouping
    # Determine group order as it appears in the table
    group_cols = ["wake_model"] + (["bias"] if ("bias" in results.columns and results["bias"].nunique() > 1) else [])
    groups = results[group_cols].apply(tuple, axis=1).tolist()

    # Find boundaries where group changes (for separator lines)
    boundaries = [-0.5]
    centers = []
    labels = []
    start = 0
    for i in range(1, len(groups) + 1):
        if i == len(groups) or groups[i] != groups[i - 1]:
            end = i - 1
            centers.append((start + end) / 2)
            if len(group_cols) == 2:
                w, b = groups[i - 1]
                labels.append(f"{_WAKE_LABELS.get(w, w)}\n{_BIAS_LABELS.get(b, b)}")
            else:
                (w,) = groups[i - 1]
                labels.append(_WAKE_LABELS.get(w, w))
            boundaries.append(i - 0.5)
            start = i

    # Secondary axis for group labels
    sec = ax.secondary_yaxis(location=0)
    sec.set_yticks(centers, labels=labels)
    sec.tick_params("y", length=60, width=0)
    sec.set_ylabel(sec_ylabel)

    # Secondary axis for separator ticks
    sec2 = ax.secondary_yaxis(location=0)
    sec2.set_yticks(boundaries, labels=[])
    sec2.tick_params("y", length=120, width=1.5)

    # Horizontal separator lines spanning plot width (skip last boundary)
    xmin, xmax = ax.get_xlim()
    for y in boundaries[1:-1]:
        ax.hlines(y=y, xmin=xmin, xmax=xmax, linewidth=1.0, color="r", linestyles="--")


def _legend_from_axis(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    title: str = "Carrier",
    ncol: int = 3,
    anchor: tuple[float, float] = (0.5, 0.0),
    labels_override: Optional[Sequence[str]] = None,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if labels_override is not None:
        labels = list(labels_override)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=anchor,
        ncol=ncol,
        title=title,
        frameon=False,
    )
    if ax.get_legend() is not None:
        ax.get_legend().remove()


def _set_title_from_prefix(ax: plt.Axes, prefix: str) -> None:
    if "combined" in prefix:
        ax.set_title("Combined")
    elif "offwind-ac" in prefix:
        ax.set_title("AC")
    elif "offwind-dc" in prefix:
        ax.set_title("DC")
    elif "offwind-float" in prefix:
        ax.set_title("Floating")
    else:
        ax.set_title(prefix)


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------

def plot_stacked(
    var: str,
    results: pd.DataFrame,
    filter: str,
    colours: pd.Series,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Horizontal stacked bar for a single results table."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5, 7), dpi=dpi)

    save_stub, xlabel = _var_meta(var)

    data = _metric_slice(results, var).filter(regex=filter, axis=1)

    if data.empty:
        raise ValueError(
            f"No numeric data to plot for var='{var}', filter='{filter}'. "
            f"Available carriers: {list(results.columns.get_level_values('carrier').unique())}"
        )

    data.plot(
        kind="barh",
        stacked=True,
        legend=False,
        color=getattr(colours, "values", colours),
        ylabel="",
        xlabel=xlabel,
        ax=ax,
    )

    _apply_scenario_yaxis_styling(ax, results)
    _legend_from_axis(
        fig,
        ax,
        title="Carrier",
        ncol=3,
        anchor=(0.5, 0.0),
        labels_override=["Offshore Wind (Combined)"] if ("singlewind" in name) else None,
    )

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_{save_stub}_stacked.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_stacked_multi(
    clusters: int,
    models: Sequence[str],
    splits: Sequence[float],
    biases: Sequence[str],
    var: str,
    filter: str,
    prefix_list: Sequence[str],
    *,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 600,
) -> tuple[plt.Figure, np.ndarray]:
    """Multi-panel stacked bars for multiple prefixes."""
    fig, axes = plt.subplots(
        1, len(prefix_list),
        figsize=(16.4 * cm, 18 * cm),
        dpi=dpi,
        sharey=True,
        sharex=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    save_stub, xlabel = _var_meta(var)

    last_results = None
    last_ax = None
    for i, prefix in enumerate(prefix_list):
        ax = axes[i]
        results, colours = results_dataframe(clusters, models, splits, biases, prefix)

        data = _metric_slice(results, var).filter(regex=filter, axis=1)
        if data.empty:
            raise ValueError(f"No data for prefix='{prefix}', var='{var}', filter='{filter}'")

        data.plot(
            kind="barh",
            stacked=True,
            legend=False,
            color=getattr(colours, "values", colours),
            ylabel="",
            xlabel=xlabel if i == (len(prefix_list) // 2) else "",
            ax=ax,
        )

        _set_title_from_prefix(ax, prefix)
        if i == 0:
            _apply_scenario_yaxis_styling(ax, results)
        else:
            ax.set_yticklabels([])

        last_results = results
        last_ax = ax

    if last_ax is not None:
        _legend_from_axis(fig, last_ax, title="Carrier", ncol=3, anchor=(0.5, 0.0))

    if savepath is None:
        savepath = Path(plots_dir) / f"{save_stub}_stacked_multi.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, axes


def plot_capacities_split(
    results: pd.DataFrame,
    colours: pd.Series,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Stacked barh of optimal capacity split across carriers."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5, 7), dpi=dpi)

    save_stub, xlabel = _var_meta("Optimal Capacity")

    data = _metric_slice(results, "Optimal Capacity")
    if data.empty:
        raise ValueError("No 'Optimal Capacity' data found to plot.")

    data.plot(
        kind="barh",
        stacked=True,
        legend=False,
        color=getattr(colours, "values", colours),
        ylabel="",
        xlabel=xlabel,
        ax=ax,
    )

    _apply_scenario_yaxis_styling(ax, results)
    _legend_from_axis(fig, ax, title="Carrier", ncol=3, anchor=(0.5, 0.0))

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_{save_stub}_split.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_generation_series(
    series: pd.Series,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    ylabel: str = "Generation [p.u.]",
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Simple time series plot for a generation profile / capacity factor series."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)

    series.plot(ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    ax.set_title(name)

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_generation_series.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_regional_carrier_percentage(
    df: pd.DataFrame,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot carrier share (%) per region (expects columns as carriers, index as regions)."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=dpi)

    pct = df.div(df.sum(axis=1), axis=0) * 100.0
    pct.plot(kind="bar", stacked=True, ax=ax, legend=False)

    ax.set_ylabel("Share [%]")
    ax.set_xlabel("")
    ax.set_title(name)

    _legend_from_axis(fig, ax, title="Carrier", ncol=3, anchor=(0.5, 0.0))

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_regional_carrier_percentage.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_region_optimal_capacity(
    geodf,
    carrier: str,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    cmap: str = "viridis",
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Choropleth of optimal capacity for a given carrier."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    col_candidates = [
        f"Optimal Capacity|{carrier}",
        f"Optimal Capacity {carrier}",
        carrier,
    ]
    col = next((c for c in col_candidates if c in geodf.columns), None)
    if col is None:
        raise KeyError(f"Could not find capacity column for carrier='{carrier}' in geodf. Tried {col_candidates}")

    geodf.plot(column=col, ax=ax, legend=True, cmap=cmap)
    ax.set_axis_off()
    ax.set_title(f"{name} – {carrier}")

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_region_optimal_capacity_{carrier}.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_region_optimal_density(
    geodf,
    carrier: str,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    area_km2_col: str = "area_km2",
    cmap: str = "viridis",
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Choropleth of capacity density derived from a given carrier column."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    if area_km2_col not in geodf.columns:
        raise KeyError(f"geodf must contain '{area_km2_col}' to compute density.")

    cap_candidates = [f"Optimal Capacity|{carrier}", f"Optimal Capacity {carrier}", carrier]
    cap_col = next((c for c in cap_candidates if c in geodf.columns), None)
    if cap_col is None:
        raise KeyError(f"Could not find capacity column for carrier='{carrier}' in geodf. Tried {cap_candidates}")

    density = geodf[cap_col] * 1e3 / geodf[area_km2_col]  # GW -> MW/km^2 if cap in GW
    plot_gdf = geodf.copy()
    plot_gdf["density"] = density

    plot_gdf.plot(column="density", ax=ax, legend=True, cmap=cmap)
    ax.set_axis_off()
    ax.set_title(f"{name} – {carrier} density")

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_region_optimal_density_{carrier}.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_distribution(
    dist_series: pd.Series,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    xlabel: str = "Value",
    ylabel: str = "Count",
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram distribution plot (wake losses / runtime / etc.)."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(5, 3), dpi=dpi)

    dist_series.dropna().plot(kind="hist", ax=ax, bins=30)
    ax.set_title(name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_distribution.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_region_capacity_density(
    geodf,
    cap_col: str,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    area_km2_col: str = "area_km2",
    cmap: str = "viridis",
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Generic choropleth of density computed from a chosen capacity column."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    if cap_col not in geodf.columns:
        raise KeyError(f"geodf does not contain '{cap_col}'")
    if area_km2_col not in geodf.columns:
        raise KeyError(f"geodf must contain '{area_km2_col}' to compute density.")

    plot_gdf = geodf.copy()
    plot_gdf["density"] = plot_gdf[cap_col] * 1e3 / plot_gdf[area_km2_col]

    plot_gdf.plot(column="density", ax=ax, legend=True, cmap=cmap)
    ax.set_axis_off()
    ax.set_title(f"{name} – density")

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_region_capacity_density.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_temporal_comp_data(
    clusters: int,
    models: Sequence[str],
    splits: Sequence[float],
    biases: Sequence[str],
    prefix: str,
    name: str,
    *,
    ax: Optional[plt.Axes] = None,
    fig: Optional[plt.Figure] = None,
    plots_dir: Union[str, Path] = "plots",
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 600,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the temporal comparison curve from temporal_comp_data()."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=dpi)

    df = temporal_comp_data(clusters, models, splits, biases, prefix)
    df.plot(ax=ax, legend=False)

    ax.set_title(name)
    ax.set_xlabel("")
    ax.set_ylabel("Value")

    _legend_from_axis(fig, ax, title="Scenario", ncol=2, anchor=(0.5, 0.0))

    if savepath is None:
        savepath = Path(plots_dir) / f"{name}_temporal_comp.png"
    _save_fig(fig, savepath, dpi=dpi)
    return fig, ax


def plot_region_optimal_density_new(
    *,
    carrier: str,
    clusters: int,
    scenarios: list[str],
    prefix: str,
    vmin: float | None = None,
    vmax: float | None = None,
    results_dir: str | Path = "results",
    regions_dir: str | Path = "wake_extra",
    year: int = 2030,
    savepath: str | Path | None = None,
    dpi: int = 600,
):
    fig, ax = plt.subplots(
        1, len(scenarios),
        figsize=(16.4 * cm, 4.5 * cm),
        dpi=dpi,
        sharex=True, sharey=True,
        layout="constrained",
    )

    if len(scenarios) == 1:
        ax = [ax]

    for i, scenario in enumerate(scenarios):
        out = build_region_capacity_density_geodf(
            carrier=carrier,
            clusters=clusters,
            scenario=scenario,
            prefix=prefix,
            results_dir=results_dir,
            regions_dir=regions_dir,
            year=year,
            cap_field="p_nom_opt",
        )
        geodf = out.geodf

        legend_value = (i == (len(scenarios) - 1))
        geodf.plot(
            column=out.density_col,
            ax=ax[i],
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            legend=legend_value,
            linewidth=0.1,
            legend_kwds={
                "label": r"$\rho_{A_{region}}^{opt}$ [MW/km$\mathrm{^{2}}$]",
                "orientation": "vertical",
                "pad": 0.2,
                "shrink": 1,
            },
        )
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()

    if savepath is None:
        savepath = Path("plots") / f"{prefix}_region_optimal_density.png"
    Path(savepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savepath, bbox_inches="tight")
    return fig, ax


# -----------------------------------------------------------------------------
# Downstream analysis helpers (kept close to your original structure)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RegionDensityResult:
    geodf: "gpd.GeoDataFrame"
    cap_col: str
    density_col: str


def _default_gen_region_parser(gen_index: pd.Index) -> pd.Series:
    """
    Default region parser matching your old logic:
      generators.index like "NO4 0_00022 offwind-ac ..." -> region "NO4 0"
    i.e. take first two whitespace-separated tokens.
    """
    s = gen_index.to_series().astype(str)
    return s.str.split().str[:2].str.join(" ")


def build_region_capacity_density_geodf(
    *,
    carrier: str,
    clusters: int,
    scenario: str,
    prefix: str,
    results_dir: Union[str, Path] = "results",
    year: int = 2030,
    variant: str = "base",
    regions_dir: Union[str, Path] = "wake_extra",
    regions_subdir_from_prefix: bool = True,
    regions_name_col: str = "name",
    regions_region_col: str = "region",
    cap_field: str = "p_nom_opt",              # "p_nom_opt" or "p_nom_max"
    carrier_filter_regex: Optional[str] = None, # overrides `carrier` if provided
    region_parser=_default_gen_region_parser,
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
    - Density is computed as: (capacity_GW*1000)/area_km2 if capacity is in GW,
        or (capacity_MW)/area_km2 if capacity is in MW.
        PyPSA stores p_nom_opt/p_nom_max in MW by default, so we use MW/km².
    """
    if gpd is None:
        raise ImportError("geopandas is required to build region density GeoDataFrames.")

    import pypsa  # local import to keep module import light

    results_dir = Path(results_dir)
    regions_dir = Path(regions_dir)

    # --- load network ---
    npath = (
        results_dir
        / prefix
        / scenario
        / "postnetworks"
        / f"{variant}_s_{clusters}_lvopt___{year}.nc"
    )
    n = pypsa.Network(str(npath))

    gens = n.generators.copy()

    # --- filter to carrier (your old code used regex on index) ---
    regex = carrier_filter_regex or carrier
    gens = gens.filter(regex=regex, axis=0)

    if gens.empty:
        raise ValueError(
            f"No generators matched regex='{regex}' in {npath}.\n"
            f"Tip: check generator naming and the regex you pass as `carrier`."
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
    # your old path logic:
    # wake_extra/<prefix.split('-')[2]>/regions_offshore_<scenario.split('-')[1]>.geojson
    scenario_split_key = scenario.split("-")[1]  # e.g. "s50000" or "s10000"
    if regions_subdir_from_prefix:
        prefix_key = prefix.split("-")[2] if len(prefix.split("-")) >= 3 else prefix
        regions_path = regions_dir / f"northsea/regions_offshore_{scenario_split_key}.geojson"
    else:
        regions_path = regions_dir / prefix / f"regions_offshore_{scenario_split_key}.geojson"

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


def nice_names_dist(scenario: str) -> str:
    """Short label for distribution plots."""
    spec = parse_scenario(scenario)
    wake_label = _WAKE_LABELS.get(spec.wake_model, spec.wake_model)
    bias_label = _BIAS_LABELS.get(spec.bias, spec.bias)
    return f"{wake_label} | {bias_label} | s{spec.region_max:g}"


def prepare_dist_series(
    dist_dict: dict[str, Sequence[float]],
    models: Sequence[str],
    splits: Sequence[float],
    biases: Sequence[str],
) -> pd.Series:
    """Flatten a dict of scenario->values into a labelled Series."""
    scenarios = scenario_list(models, splits, biases)
    rows = []
    for sc in scenarios:
        vals = dist_dict.get(sc, [])
        for v in vals:
            rows.append((sc, v))
    s = pd.Series({(sc, i): v for i, (sc, v) in enumerate(rows)})
    s.index = pd.MultiIndex.from_tuples(s.index, names=["scenario", "i"])
    return s


def prepare_runtime_data(runtime_dict: dict[str, float]) -> pd.Series:
    """Prepare a series of runtime values keyed by scenario."""
    return pd.Series(runtime_dict).sort_index()


def temporal_comp_data(
    clusters: int,
    models: Sequence[str],
    splits: Sequence[float],
    biases: Sequence[str],
    prefix: str,
    results_dir: Path = Path("results"),
    year: int = 2030,
) -> pd.DataFrame:
    """
    Load and compute a temporal comparison DataFrame across scenarios.

    NOTE: This is a placeholder-style implementation because the exact
    comparison metric depends on your original logic. Replace the body
    with your project-specific temporal comparison code if needed.
    """
    scenarios = scenario_list(models, splits, biases)
    series = {}
    for sc in scenarios:
        npath = network_path(results_dir=results_dir, prefix=prefix, scenario=sc, clusters=clusters, year=year)
        n = pypsa.Network(str(npath))
        # Example: mean offshore wind CF over time (adjust to your needs)
        gens = n.generators_t.p.filter(like="offwind")
        if gens.empty:
            continue
        series[sc] = gens.sum(axis=1)
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series)


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
    x_breaks: Sequence[float],
    M_den: Sequence[float],
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
    # If last is infinite, we still want a final breakpoint for plotting
    # We'll just use P_max as "end" when plotting; for computation we use constant extension

    M_cap = np.asarray(spec.factors, dtype=float)
    # Need breaks length K+1. If M has K tiers and last is infinite,
    # we supply a last breakpoint at breaks_GW[-1] (last finite), and the code extends.
    breaks_for_compute = np.asarray(list(breaks_GW) + [breaks_GW[-1]], dtype=float)  # placeholder last
    # But breaks must be strictly increasing; so we provide only finite breaks, and allow extension:
    breaks_for_compute = breaks_GW  # finite breaks only

    # For capacity-based, we want breaks length K+1 where last break is last finite boundary.
    # K = len(M_cap) must equal len(breaks_for_compute)-1? Not necessarily if last tier infinite.
    # We'll construct breaks explicitly:
    # Example glaum: tiers = 3, boundaries at [0,2,12], last tier extends beyond 12.
    breaks = breaks_GW  # [0,2,12]
    # so K should be 3, but breaks length is 3 => K mismatch. Fix by adding last boundary same as last finite?
    # Instead, treat tiers as:
    # Tier1 over [0,2], Tier2 over [2,12], Tier3 over [12, inf].
    # breaks must be [0,2,12] and M length 3, and our helper can handle "t>=K" only if breaks length K+1.
    # Therefore, use breaks length 4: [0,2,12,12] would not be increasing.
    # We'll implement capacity total loss directly (clearer/safer).

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
    x_breaks: Sequence[float],
    M_den: Sequence[float],
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    x_grid = np.asarray(x_grid, dtype=float)
    x_breaks = np.asarray(x_breaks, dtype=float)
    M_den = np.asarray(M_den, dtype=float)
    T = _cum_total_loss_from_piecewise_marginal(x_grid, x_breaks, M_den)
    return T, (x_breaks, M_den)


def plot_wake_models_capacity_and_density(
    *,
    A_ref_km2: float = 1000.0,
    P_max_GW: float = 30.0,
    x_max: float = 4.0,
    alpha_uniform: float = 0.8855,  # uniform scaling factor => constant loss = 1-alpha
    savepath: Optional[str] = None,
    dpi: int = 600,
) -> plt.Figure:
    """
    Review-proof wake-model figure (Option A).

    Layout: 2x2
        - Top-left: Total loss vs capacity P (GW)
        - Top-right: Total loss vs density x (MW/km^2)
        - Bottom-left: Marginal loss vs capacity P (step)
        - Bottom-right: Marginal loss vs density x (step)

    Both panels overlay BOTH tiered methods:
        - capacity-tier (blue)
        - density-tier (red)
    plus:
        - uniform scaling correction (dashed horizontal)
        - no-wake baseline (dotted at 0)

    Option A:
        density-tier is extended on the P-axis beyond x_max via constant last marginal tier.
    """
    if sns is not None:
        sns.set_theme(style="ticks")
        sns.despine()

    # --- specs ---
    spec_cap = _glaum_spec()
    spec_den, x_breaks = _new_more_spec()
    M_den = np.asarray(spec_den.factors, dtype=float)  # len=6 for your x_breaks

    # --- grids ---
    P_grid = np.linspace(0.0, P_max_GW, 800)
    x_grid = np.linspace(0.0, x_max, 800)

    # --- compute total losses on both axes ---
    # capacity-tier
    T_cap_on_P, (Pbreak_cap, Mcap) = _capacity_tier_total_loss_on_capacity_axis(P_grid, spec_cap)
    T_cap_on_x, (xbreak_cap, Mcap_x) = _capacity_tier_total_loss_on_density_axis(
        x_grid, spec_cap, A_ref_km2=A_ref_km2
    )

    # density-tier
    T_den_on_x, (xbreak_den, Mden_x) = _density_tier_total_loss_on_density_axis(x_grid, x_breaks, M_den)
    T_den_on_P, (Pbreak_den, Mden_P) = _extend_density_total_loss_to_capacity_axis(
        P_grid,
        A_ref_km2=A_ref_km2,
        x_breaks=x_breaks,
        M_den=M_den,
    )

    # --- constants ---
    loss_uniform = 1.0 - float(alpha_uniform)

    # --- colors ---
    blue = "#1f77b4"
    red = "#d62728"
    grey = "0.3"

    # --- figure ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), dpi=dpi, sharey="row")
    ax_TP, ax_Tx = axes[0, 0], axes[0, 1]
    ax_MP, ax_Mx = axes[1, 0], axes[1, 1]

    # -------------------------
    # TOP: Total loss
    # -------------------------
    ax_TP.plot(P_grid, T_cap_on_P, color=blue, lw=2.5, label="Tiered capacity-based (total)")
    ax_TP.plot(P_grid, T_den_on_P, color=red, lw=2.5, label="Tiered density-based (total)")
    ax_TP.axhline(loss_uniform, color=grey, ls="--", lw=1.8, label="Uniform scaling correction (constant)")
    ax_TP.axhline(0.0, color=grey, ls=":", lw=1.8, label="No-wake baseline")

    ax_TP.set_title("Total loss vs installed capacity")
    ax_TP.set_xlabel("Installed capacity, P (GW)")
    ax_TP.set_ylabel("Loss (fraction)")

    ax_Tx.plot(x_grid, T_cap_on_x, color=blue, lw=2.5, label="Tiered capacity-based (total)")
    ax_Tx.plot(x_grid, T_den_on_x, color=red, lw=2.5, label="Tiered density-based (total)")
    ax_Tx.axhline(loss_uniform, color=grey, ls="--", lw=1.8, label="Uniform scaling correction (constant)")
    ax_Tx.axhline(0.0, color=grey, ls=":", lw=1.8, label="No-wake baseline")

    ax_Tx.set_title("Total loss vs installed capacity density")
    ax_Tx.set_xlabel("Installed capacity density, x (MW/km²)")

    # Force axis limits (density MUST go to 4)
    ax_TP.set_xlim(0.0, P_max_GW)
    ax_Tx.set_xlim(0.0, x_max)
    ax_MP.set_xlim(0.0, P_max_GW)
    ax_Mx.set_xlim(0.0, x_max)
    
    # --- Log scale ONLY on density axis panels ---
    for ax in (ax_Tx, ax_Mx):
        ax.set_xscale("log")
        ax.set_xlim(1e-3, x_max)  # avoid 0 on log-scale; keep max at 4
        ax.set_xticks([1e-3, 1e-2, 1e-1, 0.25, 1.0, 2.5, 4.0])
        ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))


    # Secondary x-axes (equivalence for A_ref)
    def P_to_x(Pgw): return (np.asarray(Pgw) * 1000.0) / float(A_ref_km2)
    def x_to_P(xmwkm2): return (np.asarray(xmwkm2) * float(A_ref_km2)) / 1000.0


    # Threshold markers (light verticals)
    for pb in Pbreak_cap[1:]:
        ax_TP.axvline(pb, color=blue, alpha=0.18, lw=1.5)
    for pb in Pbreak_den[1:]:
        ax_TP.axvline(pb, color=red, alpha=0.18, lw=1.5)

    for xb in xbreak_cap[1:]:
        ax_Tx.axvline(xb, color=blue, alpha=0.18, lw=1.5)
    for xb in xbreak_den[1:]:
        ax_Tx.axvline(xb, color=red, alpha=0.18, lw=1.5)

    # -------------------------
    # BOTTOM: Marginal loss (step) — ROBUST CONSTRUCTION
    # -------------------------

    # ---- Capacity-tier marginal vs P
    # spec_cap max_caps are [2e3, 10e3, inf] MW; cumulative boundaries are [2,12] GW
    P_bnds_cap = np.array([0.0, 2.0, 12.0], dtype=float)
    P_bnds_cap = P_bnds_cap[P_bnds_cap < P_max_GW]
    P_edges_cap = np.r_[P_bnds_cap, P_max_GW]

    # Mcap should correspond to tiers; if helper already gives it, use it; else derive:
    # expected marginal tiers length = len(P_edges_cap)-1
    Mcap_plot = np.asarray(Mcap, dtype=float).copy()
    # If helper returned more than needed, truncate; if fewer, pad with last
    need = len(P_edges_cap) - 1
    if len(Mcap_plot) >= need:
        Mcap_plot = Mcap_plot[:need]
    else:
        Mcap_plot = np.r_[Mcap_plot, np.full(need - len(Mcap_plot), Mcap_plot[-1])]

    y_cap = np.r_[Mcap_plot, Mcap_plot[-1]]  # y must match x length
    ax_MP.step(P_edges_cap, y_cap, where="post", color=blue, lw=2.5, label="Tiered capacity-based (marginal)")

    # ---- Density-tier marginal vs x  (domain is exactly 0..x_max)
    x_edges_den = np.asarray(x_breaks, dtype=float)
    # ensure ends at x_max (4.0)
    if x_edges_den[-1] < x_max:
        x_edges_den = np.r_[x_edges_den, x_max]
    else:
        x_edges_den[-1] = x_max

    # M_den is per-interval (len = len(x_edges_den)-1 expected)
    Mden_plot_x = np.asarray(M_den, dtype=float).copy()
    need = len(x_edges_den) - 1
    if len(Mden_plot_x) >= need:
        Mden_plot_x = Mden_plot_x[:need]
    else:
        Mden_plot_x = np.r_[Mden_plot_x, np.full(need - len(Mden_plot_x), Mden_plot_x[-1])]

    y_den_x = np.r_[Mden_plot_x, Mden_plot_x[-1]]  # y length = x length
    ax_Mx.step(x_edges_den, y_den_x, where="post", color=red, lw=2.5, label="Tiered density-based (marginal)")

    # ---- Density-tier marginal mapped to P with Option A extension
    # map x edges -> P edges
    P_edges_den_core = x_to_P(x_edges_den)  # GW
    if P_edges_den_core[-1] < P_max_GW:
        # Option A: extend beyond x_max by holding last marginal constant
        P_edges_den = np.r_[P_edges_den_core, P_max_GW]
        Mden_plot_P = np.r_[Mden_plot_x, Mden_plot_x[-1]]  # add one extra segment for extension
    else:
        P_edges_den = P_edges_den_core
        Mden_plot_P = Mden_plot_x

    y_den_P = np.r_[Mden_plot_P, Mden_plot_P[-1]]
    ax_MP.step(P_edges_den, y_den_P, where="post", color=red, lw=2.5, label="Tiered density-based (marginal)")

    # ---- Capacity-tier marginal mapped to x (for completeness on right-bottom)
    # Map capacity boundaries to density: x = P*1000/Aref
    x_edges_cap_core = P_to_x(P_edges_cap)
    x_edges_cap_core = np.clip(x_edges_cap_core, 0.0, x_max)
    # remove duplicates after clipping (important!)
    x_edges_cap = np.unique(x_edges_cap_core)

    # Determine which tiers apply up to x_max (given A_ref)
    # For A_ref=1000 and x_max=4 => P=4GW, so only Tier1 (0-2GW) and Tier2 (2-4GW) appear.
    # Use the first two marginal values from Mcap_plot.
    if len(x_edges_cap) >= 2:
        # build y to match x_edges_cap length
        # intervals count = len(x_edges_cap)-1
        nint = len(x_edges_cap) - 1
        Mcap_x_plot = Mcap_plot[:max(1, min(len(Mcap_plot), nint))]
        if len(Mcap_x_plot) < nint:
            Mcap_x_plot = np.r_[Mcap_x_plot, np.full(nint - len(Mcap_x_plot), Mcap_x_plot[-1])]
        y_cap_x = np.r_[Mcap_x_plot, Mcap_x_plot[-1]]
        ax_Mx.step(x_edges_cap, y_cap_x, where="post", color=blue, lw=2.5, label="Tiered capacity-based (marginal)")

    # Reference lines + labels for bottom panels
    for ax in [ax_MP, ax_Mx]:
        ax.axhline(loss_uniform, color=grey, ls="--", lw=1.8, label="Uniform scaling correction (constant)")
        ax.axhline(0.0, color=grey, ls=":", lw=1.8, label="No-wake baseline")

    ax_MP.set_title("Marginal loss (tier step) vs installed capacity")
    ax_MP.set_xlabel("Installed capacity, P (GW)")
    ax_MP.set_ylabel("Marginal loss, M(·) (fraction)")

    ax_Mx.set_title("Marginal loss (tier step) vs installed capacity density")
    ax_Mx.set_xlabel("Installed capacity density, x (MW/km²)")


    # Formatting: percentage ticks + grid
    for ax in [ax_TP, ax_Tx, ax_MP, ax_Mx]:
        ax.yaxis.set_major_formatter(lambda v, pos: f"{100*v:.0f}%")
        ax.grid(True, alpha=0.25)

    # One clean legend for entire figure (bottom center)
    handles, labels = ax_TP.get_legend_handles_labels()
    # add bottom panel handles too (for the marginal labels)
    h_b, l_b = ax_MP.get_legend_handles_labels()
    handles += h_b
    labels += l_b

    seen = set()
    h2, l2 = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            h2.append(h)
            l2.append(l)

    fig.legend(h2, l2, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
    return fig



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
    if sns is not None:
        sns.set_theme(style="ticks")
        sns.despine()

    # --- specs from your wake methods ---
    spec_cap = _glaum_spec()
    spec_den, x_breaks_den = _new_more_spec()  # x_breaks_den ends at 4 in your method
    M_den = np.asarray(spec_den.factors, dtype=float)

    # --- grids ---
    # avoid 0 for log scale, but still show up to x_max
    x_grid = np.geomspace(1e-3, x_max, 1200)

    # constants
    loss_uniform = 1.0 - float(alpha_uniform)
    blue = "#1f77b4"
    red = "#d62728"
    grey = "0.30"

    fig, axes = plt.subplots(2, 2, figsize=(14, 7.5), dpi=dpi, sharey="row")
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

        ax_T.plot(x_grid, T_cap_x, color=blue, lw=2.6, label="Tiered capacity-based (total)")
        ax_T.plot(x_grid, T_den_x, color=red, lw=2.6, label="Tiered density-based (total)")
        ax_T.axhline(loss_uniform, color=grey, ls="--", lw=1.8, label="Uniform scaling correction")
        ax_T.axhline(0.0, color=grey, ls=":", lw=1.8, label="No-wake baseline")

        # threshold markers
        for xb in np.asarray(x_breaks_cap, dtype=float)[1:]:
            if 0 < xb < x_max:
                ax_T.axvline(xb, color=blue, alpha=0.18, lw=1.4)
        for xb in np.asarray(x_breaks_den_arr, dtype=float)[1:]:
            if 0 < xb < x_max:
                ax_T.axvline(xb, color=red, alpha=0.18, lw=1.4)

        ax_T.set_title(f"Total loss vs density\n{col_title}")
        ax_T.set_xlabel("Installed capacity density, x (MW/km²)")
        ax_T.set_ylabel("Loss (fraction)")

        # --- MARGINAL LOSS (density axis) ---
        # Build robust step arrays that reach x_max
        x_edges_cap, y_cap = _step_xy(np.asarray(x_breaks_cap, float), np.asarray(M_cap, float), x_max=x_max)
        x_edges_den_step, y_den = _step_xy(np.asarray(x_breaks_den_arr, float), np.asarray(M_den_arr, float), x_max=x_max)

        ax_M.step(x_edges_cap, y_cap, where="post", color=blue, lw=2.6, label="Tiered capacity-based (marginal)")
        ax_M.step(x_edges_den_step, y_den, where="post", color=red, lw=2.6, label="Tiered density-based (marginal)")
        ax_M.axhline(loss_uniform, color=grey, ls="--", lw=1.8, label="Uniform scaling correction")
        ax_M.axhline(0.0, color=grey, ls=":", lw=1.8, label="No-wake baseline")

        for xb in x_edges_cap[1:-1]:
            ax_M.axvline(xb, color=blue, alpha=0.18, lw=1.4)
        for xb in x_edges_den_step[1:-1]:
            ax_M.axvline(xb, color=red, alpha=0.18, lw=1.4)

        ax_M.set_title(f"Marginal loss (tier step) vs density\n{col_title}")
        ax_M.set_xlabel("Installed capacity density, x (MW/km²)")
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

    _plot_column(ax_TL, ax_ML, A_left_km2, col_title=fr"$A_{{region}}$ = {A_left_km2:,.0f} km²")
    _plot_column(ax_TR, ax_MR, A_right_km2, col_title=fr"$A_{{region}}$ = {A_right_km2:,.0f} km²")

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

    fig.legend(H, L, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    if savepath is not None:
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    return fig


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
    """Parse scenario name '<wake>-<bias>-s<region_max>'."""
    m = _SCENARIO_RE.match(s)
    if not m:
        raise ValueError(f"Scenario '{s}' must match '<wake>-<bias>-s<region_max>'")
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
    "base": r"$\mathrm{WC}: \alpha = 1.0$",
    "standard": r"$\mathrm{WC}: \alpha = 0.8855$",
    "glaum": r"$\mathrm{WC}: \mathrm{T}({P^{max}_{nom}})$",
    "new_more": r"$\mathrm{WC}:\mathrm{T}(\rho_{A_{region}})$",
}

# You can expand these with your bias method names
_BIAS_LABELS = {
    "biasTrue": r"$\mathrm{BC}: \mathrm{none}$",
    "biasFalse": r"$\mathrm{BC}: \mathrm{none}$",
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
        f"{bias_label}\n"
        + r"$A_{region}^{max}:$ " + f"{area:,}" + r" km$\mathrm{^{2}}$"
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


# -----------------------------------------------------------------------------
# Downstream analysis helpers (kept close to your original structure)
# -----------------------------------------------------------------------------

def df_to_geodf(df: pd.DataFrame, shapes_gdf):
    """
    Convert a per-region DataFrame to a GeoDataFrame by joining with shapes.
    Expects shapes_gdf index compatible with df index.
    """
    if gpd is None:
        raise ImportError("geopandas is required for df_to_geodf()")
    geodf = shapes_gdf.join(df, how="left")
    # optional area column
    if "area_km2" not in geodf.columns:
        try:
            geodf["area_km2"] = geodf.geometry.to_crs(3035).area / 1e6
        except Exception:
            pass
    return geodf


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

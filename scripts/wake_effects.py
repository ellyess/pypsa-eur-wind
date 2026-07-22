# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Configurable wake-effect models for offshore wind.

PyPSA-Eur represents offshore wake losses with a flat ``correction_factor``
of 0.8855 applied to the capacity factors of every offshore wind carrier.
That proxy is independent of how much capacity is actually built, so it
cannot capture the central feature of wake losses: they grow as a wind farm
is packed more densely.

This module adds a ``tiered_density`` model, selected via ``electricity:
wake_model: method``. It is off by default, so the flat proxy remains in
effect unless the model is explicitly requested.

``tiered_density``
    Marginal losses are a function of capacity *density* (MW/km²) rather
    than absolute capacity, using the fitted loss curve

        T(x) = alpha * exp(-x / beta) + gamma * x + delta

    with x in MW/km². Because density is intensive, the resulting wake loss
    is invariant to the offshore spatial resolution.

The model is implemented by splitting each offshore generator into
capacity-band sub-generators, each carrying the marginal wake loss of its
band. The optimiser then fills the cheap, low-loss bands first, which is
what makes the losses respond endogenously to deployment.

References
----------
The loss-density relationship T(x) is taken from
https://doi.org/10.1016/j.weer.2026.100025.

Benmoufok, E. et al. (in preparation) — for its implementation as capacity
bands in a capacity-expansion model.

See also https://github.com/PyPSA/pypsa-eur/issues/153.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Equal-area CRS used to derive offshore region areas.
AREA_CRS = "EPSG:6933"

#: Wake models that replace, rather than scale, the offshore profiles.
METHODS = ("tiered_density",)

DEFAULT_TIERED_DENSITY_COEFFICIENTS: dict = {
    "alpha": 7.3,
    "beta": 0.05,
    "gamma": -0.7,
    "delta": -14.6,
    "breakpoints": [0, 0.0370257, 0.826982, 1.51092, 2.29324, 3.17241, 4],
}

_DEFAULTS = {
    "tiered_density": DEFAULT_TIERED_DENSITY_COEFFICIENTS,
}


@dataclass(frozen=True)
class WakeSplitSpec:
    """
    Specification of segment capacities and wake factors.

    Attributes
    ----------
    factors : list of float
        Marginal wake loss fraction per segment (length = n_segments).
    max_caps : list of float
        Maximum capacity per segment in MW (last can be np.inf).
    """

    factors: list[float]
    max_caps: list[float]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def get_wake_coefficients(config: dict, method: str) -> dict:
    """
    Return wake coefficients for *method*, merging config overrides.

    Reads ``config["electricity"]["wake_model"][method]`` and merges it over
    the built-in defaults, so a config need only name the coefficients it
    wants to change.
    """
    coeffs = dict(_DEFAULTS.get(method, {}))
    coeffs.update(config.get("electricity", {}).get("wake_model", {}).get(method, {}))
    return coeffs


def check_no_double_counting(config: dict) -> None:
    """
    Raise if a wake model would compound with the ``correction_factor``.

    The wake models supersede the flat 0.8855 proxy that PyPSA-Eur applies
    to offshore profiles via ``renewable: <carrier>: correction_factor``.
    Leaving that factor in place while enabling a model would apply the
    losses twice.
    """
    offending = {
        carrier: cfg["correction_factor"]
        for carrier, cfg in config.get("renewable", {}).items()
        if carrier.startswith("offwind") and cfg.get("correction_factor", 1.0) != 1.0
    }
    if offending:
        raise ValueError(
            "A wake model is enabled, but these offshore carriers still apply "
            f"a correction_factor: {offending}. The wake model replaces that "
            "flat proxy; set `correction_factor: 1` for them to avoid "
            "double-counting the wake losses."
        )


# ---------------------------------------------------------------------------
# Segment specifications
# ---------------------------------------------------------------------------


def tiered_density_wake_spec(
    coeffs: dict | None = None,
) -> tuple[WakeSplitSpec, list[float]]:
    """
    Build a density-tier wake specification.

    Marginal wake losses per density tier follow the fitted curve
    ``T(x) = alpha * exp(-x / beta) + gamma * x + delta`` with x the
    capacity density in MW/km².

    Returns
    -------
    tuple of (WakeSplitSpec, list of float)
        The spec (with empty ``max_caps``, computed later from region areas)
        and the density breakpoints.
    """
    if coeffs is None:
        coeffs = DEFAULT_TIERED_DENSITY_COEFFICIENTS

    alpha = coeffs["alpha"]
    beta = coeffs["beta"]
    gamma = coeffs["gamma"]
    delta = coeffs["delta"]

    def total_loss(x: float) -> float:
        """Total wake loss T(x) in percent at density x MW/km²."""
        return alpha * np.exp(-x / beta) + gamma * x + delta

    def marginal_loss(x0: float, x1: float) -> float:
        """Average marginal wake loss over density interval [x0, x1]."""
        return (total_loss(x1) * x1 - total_loss(x0) * x0) / (x1 - x0)

    breakpoints = list(coeffs["breakpoints"])
    factors = [
        -(marginal_loss(breakpoints[i], breakpoints[i + 1])) / 100.0
        for i in range(len(breakpoints) - 1)
    ]
    return WakeSplitSpec(factors=factors, max_caps=[]), breakpoints


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _offwind_region_mapping(n) -> pd.Series:
    """
    Map offwind generator names to their region key.

    With resource classes, offshore generators are named
    ``"{region} {resource_class} {carrier}"``, e.g.
    ``"BE2 0AC_00000 0 offwind-ac"``. Only the carrier is stripped here: a bus
    may legitimately end in a number ("GB0 0"), so ``"GB0 0 offwind-ac"`` is
    ambiguous on its own. :func:`_resolve_region_keys` settles it against the
    region names.
    """
    gen_idx = n.generators.filter(like="offwind", axis=0).index
    return gen_idx.to_series().str.replace(r"\s+offwind-[\w-]+$", "", regex=True)


def _resolve_region_keys(keys: pd.Series, known) -> pd.Series:
    """
    Strip the resource-class index from the keys that carry one.

    A key that already names a region is left alone; otherwise its final
    whitespace-separated token is dropped, which is where the resource class
    lives.
    """
    known = set(known)
    return keys.map(lambda key: key if key in known else key.rsplit(" ", 1)[0])


def _ensure_region_area(regions_gdf):
    """
    Return *regions_gdf* with an ``area`` column in km².

    The offshore regions written by ``cluster_network`` carry no area
    column, so derive it in an equal-area CRS when absent.
    """
    if "area" in regions_gdf.columns:
        return regions_gdf

    regions_gdf = regions_gdf.copy()
    regions_gdf["area"] = regions_gdf.geometry.to_crs(AREA_CRS).area / 1e6
    return regions_gdf


def _split_profile_by_capacity(
    n,
    df: pd.DataFrame,
    num_splits: int,
    label_prefix: str = " w",
) -> tuple[list[pd.Series], list[pd.Series], list[str], list[str]]:
    """
    Split each generator into wake loss segments with modified p_max_pu.

    Each original generator is replaced by up to *num_splits* sub-generators,
    each with a capacity band and the wake loss factor of that band applied
    to its capacity factor time series.

    Returns
    -------
    tuple
        (generators_to_add, pmax_to_add, labels, to_drop)
    """
    generators_to_add: list[pd.Series] = []
    pmax_to_add: list[pd.Series] = []
    labels: list[str] = []
    to_drop: list[str] = []

    for gen_name in df.index:
        base = df.loc[gen_name]
        base_pmax = n.generators_t.p_max_pu.loc[:, gen_name]

        to_drop.append(gen_name)

        remaining_p_nom_max = float(base.p_nom_max)
        remaining_p_nom = float(base.p_nom)
        remaining_p_nom_min = float(base.p_nom_min)

        for i in range(1, num_splits + 1):
            if remaining_p_nom_max <= 0:
                break

            seg_max = float(base.get(f"max_capacity_{i}", 0.0))
            if not np.isfinite(seg_max) or seg_max <= 0:
                seg_p_nom_max = remaining_p_nom_max
            else:
                seg_p_nom_max = min(seg_max, remaining_p_nom_max)

            seg = base.copy()
            seg["p_nom_max"] = seg_p_nom_max
            remaining_p_nom_max -= seg_p_nom_max

            seg_p_nom = (
                min(seg_p_nom_max, remaining_p_nom) if remaining_p_nom > 0 else 0.0
            )
            seg_p_nom_min = (
                min(seg_p_nom_max, remaining_p_nom_min)
                if remaining_p_nom_min > 0
                else 0.0
            )

            seg["p_nom"] = seg_p_nom
            seg["p_nom_min"] = seg_p_nom_min
            remaining_p_nom -= seg_p_nom
            remaining_p_nom_min -= seg_p_nom_min

            wake_factor = float(base.get(f"factor_wake_{i}", 0.0))
            seg_pmax = base_pmax * (1.0 - wake_factor)

            labels.append(f"{gen_name}{label_prefix}{i}")
            generators_to_add.append(seg)
            pmax_to_add.append(seg_pmax)

    return generators_to_add, pmax_to_add, labels, to_drop


def _assign_segment_count(p_nom_max: np.ndarray, cumcaps: np.ndarray) -> np.ndarray:
    """Return number of segments needed for each generator."""
    return 1 + (cumcaps <= p_nom_max[:, None]).sum(axis=1)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def add_wake_generators(n, config: dict, regions_gdf) -> None:
    """
    Apply the density-tiered wake model by splitting offshore generators.

    Modifies *n* in place, replacing each offshore wind generator with
    capacity-band sub-generators whose ``p_max_pu`` carries the marginal
    wake loss of that band.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify.
    config : dict
        Full snakemake config dictionary.
    regions_gdf : GeoDataFrame
        Offshore regions with a ``name`` column. An ``area`` column in km² is
        used if present, otherwise derived from the geometries.
    """
    mapping = _offwind_region_mapping(n)
    if mapping.empty:
        return

    wake_generators = n.generators.loc[mapping.index].copy()
    coeffs = get_wake_coefficients(config, "tiered_density")

    regions_gdf = _ensure_region_area(regions_gdf)
    offshore_reg = regions_gdf[["name", "area"]].set_index("name")

    mapping = _resolve_region_keys(mapping, offshore_reg.index)
    wake_generators = wake_generators.assign(region=mapping.values)
    wake_generators = wake_generators.join(offshore_reg, on="region")

    if wake_generators["area"].isna().any():
        missing = wake_generators.loc[wake_generators["area"].isna(), "region"].unique()
        raise ValueError(
            "Missing offshore region areas for some generators. "
            f"First missing regions: {missing[:10]!r}"
        )

    spec, breakpoints = tiered_density_wake_spec(coeffs)
    factors = spec.factors

    dx = np.diff(np.asarray(breakpoints, dtype=float))
    area = wake_generators["area"].to_numpy(dtype=float)
    max_caps = np.column_stack([area * dx[i] for i in range(len(dx))])
    max_caps[:, -1] = np.inf

    for i, f in enumerate(factors, start=1):
        wake_generators[f"factor_wake_{i}"] = f
    for i in range(1, len(factors)):
        wake_generators[f"max_capacity_{i}"] = max_caps[:, i - 1]
    wake_generators[f"max_capacity_{len(factors)}"] = np.inf

    cumcaps = np.cumsum(max_caps[:, : len(factors) - 1], axis=1)
    pmax = wake_generators["p_nom_max"].to_numpy(dtype=float)
    seg_count = _assign_segment_count(pmax, cumcaps)

    split_generators: dict[int, pd.DataFrame] = {
        k: wake_generators.loc[seg_count == k] for k in range(1, len(factors) + 1)
    }

    if not any(len(df) for df in split_generators.values()):
        return

    gens_to_add: list[pd.Series] = []
    pmax_to_add: list[pd.Series] = []
    labels_all: list[str] = []
    to_drop: list[str] = []

    for num_splits, df in split_generators.items():
        if df.empty:
            continue
        g_add, t_add, lab, drop = _split_profile_by_capacity(
            n, df, num_splits=num_splits
        )
        gens_to_add.extend(g_add)
        pmax_to_add.extend(t_add)
        labels_all.extend(lab)
        to_drop.extend(drop)

    if not labels_all:
        return

    # `pd.concat` discards the index name, which PyPSA's exporter looks the
    # index up by. Restore whatever the network already used.
    generators_index_name = n.generators.index.name
    p_max_pu_columns_name = n.generators_t.p_max_pu.columns.name

    n.generators.drop(index=to_drop, inplace=True)
    n.generators_t.p_max_pu.drop(columns=to_drop, inplace=True)

    add_df = pd.concat(gens_to_add, axis=1, keys=labels_all).T.infer_objects()
    add_t = pd.concat(pmax_to_add, axis=1, keys=labels_all)

    n.generators = pd.concat([n.generators, add_df], axis=0)
    n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, add_t], axis=1)

    n.generators.index.name = generators_index_name
    n.generators_t.p_max_pu.columns.name = p_max_pu_columns_name

    logger.info(
        "Applied tiered_density wake model: split %d generators into %d segments.",
        len(to_drop),
        len(labels_all),
    )


def apply_wake_model(n, config: dict, regions_offshore=None) -> None:
    """
    Apply the wake model selected in ``electricity: wake_model: method``.

    This is the entry point used by :mod:`scripts.add_electricity`. When the
    method is ``"none"`` (the default) the network is left untouched and
    offshore wake losses remain represented by the flat ``correction_factor``.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify in place.
    config : dict
        Full snakemake config dictionary.
    regions_offshore : path-like, optional
        Offshore regions file, needed for the region areas.
    """
    method = config.get("electricity", {}).get("wake_model", {}).get("method", "none")

    if method in ("none", None, ""):
        logger.info("No wake model applied; using the offshore correction_factor.")
        return

    if method not in METHODS:
        raise ValueError(
            f"Unknown wake model method {method!r}. Expected one of "
            f"{('none', *METHODS)}."
        )

    check_no_double_counting(config)

    if regions_offshore is None:
        raise ValueError("The tiered_density wake model needs the offshore regions.")

    add_wake_generators(n, config, gpd.read_file(regions_offshore))

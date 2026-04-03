# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Wake effect modeling for offshore wind generators.

Replaces the flat correction_factor approach with configurable wake models
that account for wind farm interaction effects at the system level.

Three models are supported:

- **flat**: Uniform derate factor (default 0.8855, the current PyPSA-Eur
  default). Applied as a simple multiplier on p_max_pu.

- **capacity_tiered**: Capacity-tiered wake model (after Glaum et al.).
  Applies a global derate followed by additional marginal losses for
  generators exceeding MW capacity thresholds.

- **tiered_density**: Density-dependent wake model. Marginal wake losses
  depend on capacity density (MW/km²), making the model consistent across
  different spatial resolutions. Uses a fitted exponential loss curve:
  T(x) = alpha * exp(-x/beta) + gamma * x + delta, where x is capacity
  density in MW/km².

References
----------
Glaum et al. (2023), for the capacity-tiered approach.
Benmoufok et al. (in preparation), for the tiered-density model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default wake coefficients
# ---------------------------------------------------------------------------

DEFAULT_FLAT_COEFFICIENTS: dict = {
    "derate_factor": 0.8855,
}

DEFAULT_CAPACITY_TIERED_COEFFICIENTS: dict = {
    "global_derate": 0.906,
    "f2": 0.1279732,
    "f3_extra": 0.13902848,
    "max_caps": [2000, 10000],
}

DEFAULT_TIERED_DENSITY_COEFFICIENTS: dict = {
    "alpha": 7.3,
    "beta": 0.05,
    "gamma": -0.7,
    "delta": -14.6,
    "breakpoints": [0, 0.0370257, 0.826982, 1.51092, 2.29324, 3.17241, 4],
}

_DEFAULTS = {
    "flat": DEFAULT_FLAT_COEFFICIENTS,
    "capacity_tiered": DEFAULT_CAPACITY_TIERED_COEFFICIENTS,
    "tiered_density": DEFAULT_TIERED_DENSITY_COEFFICIENTS,
    # Legacy aliases
    "base": DEFAULT_FLAT_COEFFICIENTS,
    "standard": DEFAULT_FLAT_COEFFICIENTS,
    "glaum": DEFAULT_CAPACITY_TIERED_COEFFICIENTS,
    "new_more": DEFAULT_TIERED_DENSITY_COEFFICIENTS,
}


@dataclass(frozen=True)
class WakeSplitSpec:
    """Specification of segment capacities and wake factors.

    Attributes
    ----------
    factors : list of float
        Marginal wake loss fraction per segment (length = n_segments).
    max_caps : list of float
        Maximum capacity per segment in MW (last can be np.inf).
    """

    factors: List[float]
    max_caps: List[float]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_wake_coefficients(config: dict, method: str) -> dict:
    """Return wake coefficients for *method*, merging config overrides.

    Reads from ``config["electricity"]["wake_model"][method]`` and merges
    with built-in defaults. If the config has no overrides, all defaults
    are used.

    Parameters
    ----------
    config : dict
        Full snakemake config dictionary.
    method : str
        Wake model name: "flat", "capacity_tiered", or "tiered_density".
        Legacy aliases ("base", "standard", "glaum", "new_more") are also
        accepted.

    Returns
    -------
    dict
        Merged coefficient dictionary.
    """
    base = dict(_DEFAULTS.get(method, {}))
    overrides = config.get("electricity", {}).get("wake_model", {}).get(method, {})
    base.update(overrides)
    return base


def tiered_density_wake_spec(
    coeffs: Optional[dict] = None,
) -> Tuple[WakeSplitSpec, List[float]]:
    """Build a density-tier wake specification.

    The tiered-density model computes marginal wake losses per density
    tier based on the fitted curve T(x) = alpha * exp(-x/beta) + gamma*x + delta,
    where x is capacity density in MW/km².

    Parameters
    ----------
    coeffs : dict, optional
        Coefficient dictionary with keys: alpha, beta, gamma, delta, breakpoints.
        If None, uses DEFAULT_TIERED_DENSITY_COEFFICIENTS.

    Returns
    -------
    tuple of (WakeSplitSpec, list of float)
        The spec (with empty max_caps, computed later from region areas)
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


def capacity_tiered_wake_spec(
    coeffs: Optional[dict] = None,
) -> WakeSplitSpec:
    """Build a capacity-tiered wake specification.

    After Glaum et al.: applies a global derate to all offshore generators,
    then additional marginal losses for generators exceeding capacity
    thresholds.

    Parameters
    ----------
    coeffs : dict, optional
        Coefficient dictionary with keys: global_derate, f2, f3_extra, max_caps.
        If None, uses DEFAULT_CAPACITY_TIERED_COEFFICIENTS.

    Returns
    -------
    WakeSplitSpec
        Segment definition with three tiers.
    """
    if coeffs is None:
        coeffs = DEFAULT_CAPACITY_TIERED_COEFFICIENTS

    f2 = coeffs["f2"]
    f3_extra = coeffs["f3_extra"]
    f3 = 1.0 - ((1.0 - f2) * (1.0 - f3_extra))
    max_caps_cfg = list(coeffs["max_caps"]) + [np.inf]
    return WakeSplitSpec(
        factors=[0.0, f2, f3],
        max_caps=max_caps_cfg,
    )


def _offwind_region_mapping(n) -> pd.Series:
    """Map offwind generator names to region names (without carrier suffix)."""
    gen_idx = n.generators.filter(like="offwind", axis=0).index
    return gen_idx.to_series().str.replace(r" offwind-\w+", "", regex=True)


def _split_profile_by_capacity(
    n,
    df: pd.DataFrame,
    num_splits: int,
    label_prefix: str = " w",
) -> Tuple[List[pd.Series], List[pd.Series], List[str], List[str]]:
    """Split each generator into wake loss segments with modified p_max_pu.

    Each original generator is replaced by up to *num_splits* sub-generators,
    each with a different capacity band and wake loss factor applied to its
    capacity factor time series.

    Parameters
    ----------
    n : pypsa.Network
        Network containing generator data and time series.
    df : DataFrame
        Subset of generators to split (must have factor_wake_i and
        max_capacity_i columns).
    num_splits : int
        Maximum number of segments per generator.
    label_prefix : str
        Suffix format for segment labels.

    Returns
    -------
    tuple
        (generators_to_add, pmax_to_add, labels, to_drop)
    """
    generators_to_add: List[pd.Series] = []
    pmax_to_add: List[pd.Series] = []
    labels: List[str] = []
    to_drop: List[str] = []

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


def _assign_segment_count(
    p_nom_max: np.ndarray,
    cumcaps: np.ndarray,
) -> np.ndarray:
    """Return number of segments needed for each generator."""
    return 1 + (cumcaps <= p_nom_max[:, None]).sum(axis=1)


def add_wake_generators(
    n,
    config: dict,
    method: str,
    regions_gdf=None,
) -> None:
    """Apply wake effects to offshore wind by splitting generators into segments.

    This is the main entry point for wake modeling. It modifies the network
    in-place by replacing offshore wind generators with wake-adjusted
    sub-generators.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify.
    config : dict
        Full snakemake config dictionary.
    method : str
        Wake model: "tiered_density" or "capacity_tiered".
    regions_gdf : GeoDataFrame, optional
        Offshore regions with "name" and "area" columns. Required for
        "tiered_density" method.
    """
    if method not in {"tiered_density", "capacity_tiered", "new_more", "glaum"}:
        raise ValueError(f"Unknown wake method: {method!r}")

    # Normalize legacy names
    canonical = {"new_more": "tiered_density", "glaum": "capacity_tiered"}
    method = canonical.get(method, method)

    mapping = _offwind_region_mapping(n)
    if mapping.empty:
        return

    wake_generators = n.generators.loc[mapping.index].copy()
    coeffs = get_wake_coefficients(config, method)

    if method == "tiered_density":
        if regions_gdf is None:
            raise ValueError(
                "regions_gdf is required for the tiered_density wake model."
            )

        offshore_reg = regions_gdf[["name", "area"]].set_index("name")

        wake_generators = wake_generators.assign(region=mapping.values)
        wake_generators = wake_generators.join(offshore_reg, on="region")

        if wake_generators["area"].isna().any():
            missing = wake_generators.loc[
                wake_generators["area"].isna(), "region"
            ].unique()
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

        split_generators: Dict[int, pd.DataFrame] = {
            k: wake_generators.loc[seg_count == k] for k in range(1, len(factors) + 1)
        }

    else:  # capacity_tiered
        global_derate = coeffs.get("global_derate", 0.906)
        n.generators_t.p_max_pu.loc[:, mapping.index] *= global_derate

        min_cap = coeffs.get("max_caps", [2000, 10000])[0]
        big = wake_generators[wake_generators.p_nom_max > min_cap].copy()
        if big.empty:
            return

        spec = capacity_tiered_wake_spec(coeffs)
        for i, f in enumerate(spec.factors, start=1):
            big[f"factor_wake_{i}"] = f
        for i, cap in enumerate(spec.max_caps, start=1):
            big[f"max_capacity_{i}"] = cap

        pmax = big["p_nom_max"].to_numpy(dtype=float)
        seg_count = np.where(pmax <= (spec.max_caps[0] + spec.max_caps[1]), 2, 3)
        split_generators = {
            2: big.loc[seg_count == 2],
            3: big.loc[seg_count == 3],
        }

    if not any(len(df) for df in split_generators.values()):
        return

    gens_to_add: List[pd.Series] = []
    pmax_to_add: List[pd.Series] = []
    labels_all: List[str] = []
    to_drop: List[str] = []

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

    n.generators.drop(index=to_drop, inplace=True)
    n.generators_t.p_max_pu.drop(columns=to_drop, inplace=True)

    add_df = pd.concat(gens_to_add, axis=1, keys=labels_all).T.infer_objects()
    add_t = pd.concat(pmax_to_add, axis=1, keys=labels_all)

    n.generators = pd.concat([n.generators, add_df], axis=0)
    n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, add_t], axis=1)
    n.generators_t.p_max_pu.columns.names = ["Generator"]

    logger.info(
        "Applied %s wake model: split %d generators into %d segments.",
        method,
        len(to_drop),
        len(labels_all),
    )


def drop_non_dominant_offwind_generators(n) -> None:
    """Drop non-dominant offshore wind generators per region.

    When multiple offshore wind carriers (offwind-ac, offwind-dc, offwind-float)
    are present in the same region, keeps only the one with the largest
    p_nom_max. Ties are broken by p_nom, then by name.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify in-place.
    """
    mapping = _offwind_region_mapping(n)
    if mapping.empty:
        return

    gens = n.generators.loc[mapping.index].copy()
    gens["region"] = mapping.values

    gens["p_nom_max"] = pd.to_numeric(gens["p_nom_max"], errors="coerce").fillna(0.0)
    gens["p_nom"] = pd.to_numeric(gens.get("p_nom", 0.0), errors="coerce").fillna(0.0)

    gens = gens.assign(_name=gens.index.astype(str))
    gens_sorted = gens.sort_values(
        by=["region", "p_nom_max", "p_nom", "_name"],
        ascending=[True, True, True, True],
    )
    keep_idx = gens_sorted.groupby("region", sort=False).tail(1).index
    drop_idx = mapping.index.difference(keep_idx)

    if drop_idx.empty:
        return

    n.generators.drop(index=drop_idx, inplace=True)

    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p_max_pu"):
        cols_to_drop = [c for c in drop_idx if c in n.generators_t.p_max_pu.columns]
        if cols_to_drop:
            n.generators_t.p_max_pu.drop(columns=cols_to_drop, inplace=True)

    logger.info("Dropped %d non-dominant offwind generators.", len(drop_idx))

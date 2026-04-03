"""
Shared utilities for querying PyPSA networks in the thesis analysis pipeline.

Consolidates duplicated functions from compare_sensitivity_runs_tier{1,2}.py,
compare_bias_runs.py, compare_spatial_runs.py, and extract_wake_data.py.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pypsa


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RE_SCENARIO = re.compile(
    r"(?P<wakeprefix>[^/]+)-s(?P<res>\d+)-bias(?P<bias>True|False|Uniform|idw|kriging)",
    re.IGNORECASE,
)

WAKE_ALIASES = {
    "base": "off",
    "standard": "off",
    "no_wake": "off",
    "wakeoff": "off",
    "off": "off",
    "new_more": "density",
    "density": "density",
    "density_based": "density",
    "density-based": "density",
}

SCEN_ORDER = ["base", "biasUniform", "bias", "wake", "bias+wake"]
TECH_ORDER = ["onwind", "offwind"]

_ELEC_KEYWORDS = ("electroly", "electrolysis", "h2 electro", "pem", "alkaline")


# ---------------------------------------------------------------------------
# Network loading & scenario parsing
# ---------------------------------------------------------------------------


def load_network(path: str | Path) -> pypsa.Network:
    """Load a PyPSA network from a .nc file."""
    return pypsa.Network(str(path))


def _normalize_bias(bias: bool | str) -> str:
    """Normalize bias flag to one of: "true", "false", "uniform", "idw", "kriging"."""
    if isinstance(bias, bool):
        return "true" if bias else "false"
    key = str(bias).strip().lower()
    if key in {"true", "false", "idw", "kriging"}:
        return key
    if key in {"uniform", "biasuniform", "bias_uniform"}:
        return "uniform"
    return key


def parse_from_path(nc_path: Path) -> dict:
    """Parse scenario metadata from a network file path.

    Expects a parent folder matching ``<wakeprefix>-s<RES>-bias<True|False>``.
    """
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
    bias = _normalize_bias(m.group("bias"))
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
    """Build a manifest of network files matching *pattern* under *results_root*.

    If the glob finds nothing, a fallback swapping ``postnetworks`` <->
    ``networks`` is attempted automatically.
    """
    nc_files = sorted(results_root.glob(pattern))
    if not nc_files:
        alt = None
        if "postnetworks" in pattern:
            alt = pattern.replace("postnetworks", "networks")
        elif "networks" in pattern:
            alt = pattern.replace("networks", "postnetworks")
        if alt:
            nc_files = sorted(results_root.glob(alt))
            if nc_files:
                warnings.warn(
                    f"No files for glob {pattern!r}; using fallback {alt!r} instead."
                )
                pattern = alt
    if not nc_files:
        raise SystemExit(
            f"No networks found under {results_root} with pattern {pattern!r}"
        )
    recs = [parse_from_path(p) for p in nc_files]
    return pd.DataFrame(recs)


def find_scenario_dirs(
    results_dir: Path, scenarios: list[str], splits: list[int]
) -> dict[tuple[str, int], Path]:
    """Find scenario directories matching ``{scenario}-s{split}-biasFalse/``."""
    scenario_dirs: dict[tuple[str, int], Path] = {}
    for split in splits:
        for scenario in scenarios:
            pattern = f"{scenario}-s{split}-biasFalse"
            candidate = results_dir / pattern
            if candidate.exists():
                scenario_dirs[(scenario, split)] = candidate
                print(f"Found {scenario} (split={split}): {candidate.name}")
            else:
                print(f"Warning: {pattern} not found in {results_dir}")
    return scenario_dirs


def scenario_key(bias: bool | str, wake: str) -> str:
    """Map (bias flag, wake label) to a canonical scenario key."""
    wake_is_off = wake == "off"
    bias_key = _normalize_bias(bias)
    if bias_key == "uniform":
        return "biasUniform" if wake_is_off else "biasUniform+wake"
    if (bias_key == "false") and wake_is_off:
        return "base"
    if bias_key in ("true", "idw", "kriging") and wake_is_off:
        return "bias"
    if (bias_key == "false") and (not wake_is_off):
        return "wake"
    return "bias+wake"


# ---------------------------------------------------------------------------
# Generator selection
# ---------------------------------------------------------------------------


def gen_idx(n: pypsa.Network, tech: str) -> pd.Index:
    """Return generator index for *tech* (``"onwind"`` or ``"offwind"``)."""
    carr = n.generators.carrier.astype(str).str.lower()
    if tech == "offwind":
        return n.generators.index[carr.str.contains("offwind")]
    if tech == "onwind":
        return n.generators.index[carr.eq("onwind")]
    raise ValueError(f"Unknown tech: {tech}")


def select_generators_by_carrier(
    n: pypsa.Network, carriers: Iterable[str]
) -> pd.Index:
    """Return generator index for an explicit set of *carriers*."""
    carriers = set(carriers)
    if n.generators.empty:
        return pd.Index([])
    return n.generators.index[n.generators.carrier.isin(carriers)]


# ---------------------------------------------------------------------------
# Snapshot weights
# ---------------------------------------------------------------------------


def snapshot_weights(n: pypsa.Network) -> pd.Series:
    """Snapshot weights (hours per timestep). Falls back to 1.0 per snapshot."""
    if (
        hasattr(n, "snapshot_weightings")
        and isinstance(n.snapshot_weightings, pd.DataFrame)
        and "generators" in n.snapshot_weightings.columns
    ):
        return (
            n.snapshot_weightings["generators"]
            .reindex(n.snapshots)
            .fillna(1.0)
        )
    return pd.Series(1.0, index=n.snapshots)


# ---------------------------------------------------------------------------
# Scalar metric extraction
# ---------------------------------------------------------------------------


def wind_capacity_gw(n: pypsa.Network, tech: str) -> float:
    """Optimised wind capacity in GW for *tech*."""
    idx = gen_idx(n, tech)
    if len(idx) == 0:
        return 0.0
    g = n.generators.loc[idx]
    p_nom_opt = (
        g["p_nom_opt"]
        if "p_nom_opt" in g.columns
        else pd.Series(index=g.index, dtype=float)
    )
    p_nom = (
        g["p_nom"]
        if "p_nom" in g.columns
        else pd.Series(index=g.index, dtype=float)
    )
    cap_mw = p_nom_opt.where(p_nom_opt.notna(), p_nom).fillna(0.0).sum()
    return float(cap_mw) / 1e3


def wind_curtailment_frac(n: pypsa.Network, tech: str) -> float:
    """Curtailment fraction (0-1) for *tech*."""
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


def transmission_expansion_twkm(n: pypsa.Network) -> float:
    """Length-weighted transmission expansion proxy (TW*km)."""
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

    return total_mw_km / 1e6


def get_objective(n: pypsa.Network) -> float:
    """System objective value (EUR). Returns NaN if unavailable."""
    return float(getattr(n, "objective", np.nan))


# ---------------------------------------------------------------------------
# Carrier-level aggregations
# ---------------------------------------------------------------------------


def bus_country(bus_name: str) -> str:
    """Extract ISO-2 country code from a PyPSA-Eur bus name."""
    s = str(bus_name)
    return s[:2] if len(s) >= 2 else "??"


def capacity_by_carrier(n: pypsa.Network) -> pd.Series:
    """Total optimised capacity (MW) grouped by carrier."""
    parts = []

    if len(n.generators):
        p_nom = (
            n.generators.p_nom_opt
            if "p_nom_opt" in n.generators
            else n.generators.p_nom
        )
        parts.append(p_nom.groupby(n.generators.carrier).sum())

    if len(n.storage_units):
        p_nom = (
            n.storage_units.p_nom_opt
            if "p_nom_opt" in n.storage_units
            else n.storage_units.p_nom
        )
        parts.append(p_nom.groupby(n.storage_units.carrier).sum())

    if len(n.links):
        p_nom = (
            n.links.p_nom_opt
            if "p_nom_opt" in n.links
            else n.links.p_nom
        )
        parts.append(p_nom.groupby(n.links.carrier).sum())

    if not parts:
        return pd.Series(dtype=float, name="capacity_MW")

    s = pd.concat(parts).groupby(level=0).sum()
    s.name = "capacity_MW"
    return s.sort_values(ascending=False)


def energy_by_carrier_twh(n: pypsa.Network) -> pd.Series:
    """Total generation (TWh) grouped by carrier."""
    if (
        not hasattr(n, "generators_t")
        or not hasattr(n.generators_t, "p")
        or n.generators_t.p.empty
    ):
        return pd.Series(dtype=float, name="energy_TWh")

    w = snapshot_weights(n)
    mwh = (n.generators_t.p.mul(w, axis=0)).sum(axis=0)
    twh = mwh.groupby(n.generators.carrier).sum() / 1e6
    twh.name = "energy_TWh"
    return twh.sort_values(ascending=False)


def energy_twh_from_generators(n: pypsa.Network, tech: str) -> float:
    """Total generation (TWh) for *tech* (via ``gen_idx``)."""
    idx = gen_idx(n, tech)
    if len(idx) == 0:
        return 0.0
    try:
        p = n.generators_t.p[idx].sum(axis=1)
    except Exception:
        return 0.0
    w = snapshot_weights(n)
    mwh = float((p * w).sum())
    return mwh / 1e6


# ---------------------------------------------------------------------------
# CF time series
# ---------------------------------------------------------------------------


def cf_timeseries_system(n: pypsa.Network, tech: str) -> pd.DataFrame | None:
    """System-aggregated CF time series (avail_cf, disp_cf, curt_cf) for *tech*."""
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


def cf_timeseries_per_gen(
    n: pypsa.Network,
    carriers: tuple[str, ...],
    *,
    kind: str = "availability",
) -> pd.DataFrame:
    """Per-generator CF time series for selected *carriers*.

    *kind* is ``"availability"`` (p_max_pu) or ``"dispatch"`` (p / p_nom).
    """
    gens = n.generators
    mask = gens.carrier.isin(carriers)
    if not mask.any():
        return pd.DataFrame(index=n.snapshots)

    idx = gens.index[mask]

    if kind == "availability":
        if not hasattr(n.generators_t, "p_max_pu") or n.generators_t.p_max_pu.empty:
            return pd.DataFrame(index=n.snapshots)
        return n.generators_t.p_max_pu[idx].copy()

    if kind == "dispatch":
        if not hasattr(n.generators_t, "p") or n.generators_t.p.empty:
            return pd.DataFrame(index=n.snapshots)
        p = n.generators_t.p[idx]
        p_nom = (
            gens.loc[idx].p_nom_opt
            if "p_nom_opt" in gens.loc[idx]
            else gens.loc[idx].p_nom
        ).astype(float)
        return p.div(p_nom, axis=1)

    raise ValueError(f"Unknown kind={kind!r}")


# ---------------------------------------------------------------------------
# Sector coupling (electrolyser / H2)
# ---------------------------------------------------------------------------


def electrolyser_links(n: pypsa.Network) -> pd.Index:
    """Return link index for electrolyser-like components."""
    if not hasattr(n, "links") or n.links.empty:
        return pd.Index([])
    carr = n.links.carrier.astype(str).str.lower()
    mask = np.zeros(len(carr), dtype=bool)
    for k in _ELEC_KEYWORDS:
        mask |= carr.str.contains(k)
    return n.links.index[mask]


def electrolyser_capacity_gw(n: pypsa.Network) -> float:
    """Total electrolyser capacity in GW."""
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
    """Heuristic H2 production (TWh) from electrolyser links."""
    idx = electrolyser_links(n)
    if len(idx) == 0:
        return 0.0
    w = snapshot_weights(n)
    try:
        if hasattr(n.links_t, "p1"):
            p = n.links_t.p1[idx].sum(axis=1)
        else:
            p = -n.links_t.p0[idx].sum(axis=1)
    except Exception:
        return 0.0
    mwh = float((p * w).sum())
    return mwh / 1e6

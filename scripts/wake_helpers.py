# scripts/wake_helpers.py
"""
Backward-compatibility layer and research-specific caching helpers.

This module re-exports the upstream-bound functions from their new locations
(split_regions, wake_effects) and keeps the research-specific caching and
config helpers that are only needed in this fork.

For upstream contributions, use:
    - scripts.split_regions   (variable spatial resolution)
    - scripts.wake_effects    (wake effect modeling)
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional, Union

import geopandas as gpd

_logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def atomic_write(write: Callable[[Path], None], target: PathLike) -> Path:
    """Write to *target* atomically, via a temporary file in the same directory.

    The caches under ``wake_extra/`` are keyed on the technology and the area
    threshold only, so every run that shares those writes the same path. With
    several runs in flight, two jobs writing the file directly will interleave
    and corrupt it, or trip netCDF4's file cache. ``os.replace`` is atomic on
    the same filesystem, so a reader sees either the previous file or the
    complete new one, and the last writer simply wins.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    handle, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.")
    os.close(handle)
    tmp = Path(tmp_name)
    try:
        write(tmp)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    return target

# ---------------------------------------------------------------------------
# Re-exports from upstream-bound modules
# ---------------------------------------------------------------------------

# Region splitting (upstream: scripts/split_regions.py)
from split_regions import (  # noqa: F401
    cluster_points,
    fill_shape_with_points,
    mesh_region,
    split_regions,
    voronoi_partition,
)

# Wake effects (upstream: scripts/wake_effects.py)
from wake_effects import (  # noqa: F401
    WakeSplitSpec,
    add_wake_generators,
    capacity_tiered_wake_spec,
    drop_non_dominant_offwind_generators,
    tiered_density_wake_spec,
)

# Legacy aliases for wake_effects functions
_glaum_spec = capacity_tiered_wake_spec
_new_more_spec = tiered_density_wake_spec


# ---------------------------------------------------------------------------
# Research-specific: config helpers
# ---------------------------------------------------------------------------


def get_spatial_mods(config: dict) -> dict:
    """Return the ``spatial_mods`` sub-dictionary with a safe default."""
    return config.get("spatial_mods", {})


def _wind_threshold_key(technology: str) -> Optional[str]:
    """Return threshold key for wind technologies, otherwise None."""
    if technology.startswith("onwind"):
        return "onshore_threshold"
    if technology.startswith("offwind"):
        return "offshore_threshold"
    return None


def get_threshold(mods: dict, technology: str) -> Optional[int]:
    """Return the configured area threshold (km2) for wind technologies.

    If the corresponding threshold is set to False/None/0 in the config,
    splitting is disabled and this returns None.
    """
    key = _wind_threshold_key(technology)
    if key is None:
        raise ValueError(
            "get_threshold() is only defined for wind technologies. "
            f"Got technology={technology!r}."
        )

    val = mods.get(key)
    if val is False or val is None:
        return None

    try:
        ival = int(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid config spatial_mods.{key}={val!r}") from e

    if ival <= 0:
        return None

    return ival


# Legacy alias used by add_electricity.py (old API)
def get_wake_coefficients(mods: dict, method: str) -> dict:
    """Return wake coefficients for *method* (legacy interface).

    Reads from ``spatial_mods.wake_coefficients.<method>`` and merges
    with built-in defaults.
    """
    from wake_effects import _DEFAULTS

    base = dict(_DEFAULTS.get(method, {}))
    overrides = mods.get("wake_coefficients", {}).get(method, {})
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Research-specific: paths & caching
# ---------------------------------------------------------------------------


def get_wake_dir(mods: dict) -> Path:
    """Return the wake cache directory and ensure it exists."""
    shared = str(mods.get("shared_files", "default"))
    d = Path("wake_extra") / shared
    d.mkdir(parents=True, exist_ok=True)
    return d


def regions_file(
    wake_dir: Path,
    technology: str,
    threshold: Optional[int],
) -> Optional[Path]:
    """Return the cached regions path for wind technologies."""
    if technology.startswith("offwind"):
        tag = f"s{threshold}" if threshold is not None else "nosplit"
        return wake_dir / f"regions_offshore_{tag}.geojson"

    if technology.startswith("onwind"):
        tag = f"s{threshold}" if threshold is not None else "nosplit"
        return wake_dir / f"regions_onshore_{tag}.geojson"

    return None


def _threshold_token(threshold: Optional[int]) -> str:
    """Return a stable filename token for split / no-split cases."""
    return "nosplit" if threshold is None else str(int(threshold))


def availability_cache_path(
    wake_dir: Path,
    clusters,
    technology: str,
    threshold: Optional[int],
) -> Path:
    """Return cache path for availability matrices."""
    cache_root = wake_dir / "availability_matrices"
    cache_root.mkdir(parents=True, exist_ok=True)
    thr = _threshold_token(threshold)
    return cache_root / f"availability_matrix_{clusters}_{technology}_{thr}.nc"


def profile_cache_path(
    wake_dir: Path,
    clusters,
    technology: str,
    threshold: Optional[int],
    bias: Optional[str] = None,
    correction_factor: Optional[float] = None,
) -> Path:
    """Return cache path for renewable profiles."""
    cache_root = wake_dir / "renewable_profiles"
    cache_root.mkdir(parents=True, exist_ok=True)
    thr = _threshold_token(threshold)
    suffix = f"_bias{bias}" if bias is not None else ""
    if correction_factor is not None and correction_factor != 1.0:
        suffix += f"_cf{correction_factor}"
    return cache_root / f"profile_{clusters}_{technology}_{thr}{suffix}.nc"


def solar_thermal_cache_path(
    *,
    wake_dir: Path,
    clusters: int,
) -> Path:
    """Return cache path for solar thermal profiles."""
    cache_root = wake_dir / "solar_thermal_profiles"
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"solar_thermal_{clusters}.nc"


def temperature_cache_paths(
    *,
    wake_dir: Path,
    clusters: int,
) -> tuple[Path, Path]:
    """Return cache paths for (air, soil) temperature."""
    cache_root = wake_dir / "temperature_profiles"
    cache_root.mkdir(parents=True, exist_ok=True)
    air = cache_root / f"temp_air_{clusters}.nc"
    soil = cache_root / f"temp_soil_{clusters}.nc"
    return air, soil


# ---------------------------------------------------------------------------
# Research-specific: region loading
# ---------------------------------------------------------------------------


def load_regions(
    technology: str,
    threshold: Optional[int],
    wake_dir: Path,
    fallback_path: PathLike,
) -> gpd.GeoDataFrame:
    """Load cached wind regions (split or nosplit); otherwise load fallback."""
    p = regions_file(wake_dir, technology, threshold)
    if p is None:
        return gpd.read_file(fallback_path)

    if p.is_file():
        return gpd.read_file(p)

    _logger.warning(
        "Split-region cache not found at %s; falling back to unsplit "
        "regions from %s. Run region splitting first if splitting "
        "is intended for threshold=%s.",
        p,
        fallback_path,
        threshold,
    )
    return gpd.read_file(fallback_path)

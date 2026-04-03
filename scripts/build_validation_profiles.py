#!/usr/bin/env python3
"""
Standalone profile builder for IDW and Kriging bias correction scenarios.

Bypasses snakemake to build wind profiles directly using cached availability
matrices and the existing cutout. Uses the same logic as build_renewable_profiles.py.
"""
import logging
import sys
import time
from pathlib import Path

import atlite
import geopandas as gpd
import numpy as np
import xarray as xr

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("build_validation_profiles")

# ── Paths ──
PROJECT = Path("/Users/ellyess/Library/CloudStorage/OneDrive-ImperialCollegeLondon/PhD/pypsa-eur-wind")
CUTOUT_PATH = PROJECT / "cutouts" / "europe-2023-sarah3-era5.nc"
WAKE_DIR = PROJECT / "wake_extra" / "northsea"
AVAIL_DIR = WAKE_DIR / "availability_matrices"
PROFILE_CACHE = WAKE_DIR / "renewable_profiles"
REGION_DIR = WAKE_DIR / "regions"

# Resource directory prefix (matching validation config)
RESOURCE_PREFIX = PROJECT / "resources" / "thesis-validation-2023-northsea-2030-10-northsea-standard-6h"

# ── Config matching config.northsea_validation_dispatch.yaml ──
CLUSTERS = "10"
THRESHOLD = 100000

CARRIERS = {
    "onwind": {
        "turbine": "Vestas_V112_3MW",
        "capacity_per_sqkm": 3,
        "correction_factor": 1.0,
        "clip_p_max_pu": 1e-2,
    },
    "offwind-ac": {
        "turbine": "NREL_ReferenceTurbine_2020ATB_12MW_offshore",
        "capacity_per_sqkm": 4,
        "correction_factor": 1.0,
        "clip_p_max_pu": 1e-2,
    },
    "offwind-dc": {
        "turbine": "NREL_ReferenceTurbine_2020ATB_12MW_offshore",
        "capacity_per_sqkm": 4,
        "correction_factor": 1.0,
        "clip_p_max_pu": 1e-2,
    },
    "offwind-float": {
        "turbine": "NREL_ReferenceTurbine_2020ATB_12MW_offshore",
        "capacity_per_sqkm": 4,
        "correction_factor": 1.0,
        "clip_p_max_pu": 1e-2,
    },
}

BIAS_SCENARIOS = {
    "idw": "idw",
    "kriging": "kriging",
    "False": False,
}


def _simplify_polys(geom, minarea=1):
    """Simplify offshore polygons (from build_shapes.py)."""
    from shapely.geometry import MultiPolygon, Polygon
    if isinstance(geom, MultiPolygon):
        polys = [p for p in geom.geoms if p.area >= minarea]
        if not polys:
            return geom
        return MultiPolygon(polys) if len(polys) > 1 else polys[0]
    return geom


def load_regions(technology):
    """Load bus regions from wake_extra cache."""
    # Wake_extra stores split regions as regions_{onshore/offshore}_s{threshold}.geojson
    if technology == "onwind":
        kind = "onshore"
    else:
        kind = "offshore"
    split_path = WAKE_DIR / f"regions_{kind}_s{THRESHOLD}.geojson"
    if split_path.exists():
        return gpd.read_file(split_path)
    # Fallback: check existing resource directories
    for d in sorted(PROJECT.glob("resources/*/base-s*")):
        p = d / f"regions_{kind}_base_s_{CLUSTERS}.geojson"
        if p.exists():
            return gpd.read_file(p)
    raise FileNotFoundError(f"No regions file found for {technology} (looked for {split_path})")


def build_profile(carrier, bias_key, bias_corr_value, cutout):
    """Build a single carrier × bias scenario profile."""
    cache_name = f"profile_{CLUSTERS}_{carrier}_{THRESHOLD}_bias{bias_key}.nc"
    cache_path = PROFILE_CACHE / cache_name

    if cache_path.exists():
        logger.info(f"Cache hit: {cache_name}")
        return xr.open_dataset(cache_path)

    logger.info(f"Building {cache_name}...")
    cfg = CARRIERS[carrier]

    # Load availability matrix
    avail_path = AVAIL_DIR / f"availability_matrix_{CLUSTERS}_{carrier}_{THRESHOLD}.nc"
    if not avail_path.exists():
        logger.error(f"Missing availability matrix: {avail_path}")
        return None
    availability = xr.open_dataarray(avail_path)

    # Load regions
    regions = load_regions(carrier)
    regions = regions.set_index("name").rename_axis("bus")

    if carrier.startswith("offwind"):
        offshore_regions = availability.coords["bus"].values
        regions = regions.loc[offshore_regions]
        regions = regions.map(lambda g: _simplify_polys(g, minarea=1)).set_crs(regions.crs)
    else:
        regions = regions.representative_point()
    regions = regions.geometry.to_crs(3035)
    buses = regions.index

    area = cutout.grid.to_crs(3035).area / 1e6
    area = xr.DataArray(
        area.values.reshape(cutout.shape), [cutout.coords["y"], cutout.coords["x"]]
    )

    correction_factor = cfg["correction_factor"]

    # Build resource kwargs
    resource = {
        "turbine": cfg["turbine"],
        "smooth": False,
        "add_cutout_windspeed": True,
        "bias_corr": bias_corr_value,
    }

    # Capacity factor for layout
    t0 = time.time()
    logger.info(f"  Computing capacity factor for {carrier}...")
    capacity_factor = correction_factor * cutout.wind(capacity_factor=True, **resource)
    layout = capacity_factor * area * cfg["capacity_per_sqkm"]
    logger.info(f"  Capacity factor done ({time.time() - t0:.1f}s)")

    # Weighted profile time series
    t0 = time.time()
    logger.info(f"  Computing profile time series for {carrier}...")
    profile = cutout.wind(
        matrix=availability.stack(spatial=["y", "x"]),
        layout=layout,
        index=buses,
        per_unit=True,
        return_capacity=False,
        **resource,
    )
    profile = profile.expand_dims({"year": [0]}).rename("profile")
    logger.info(f"  Profile done ({time.time() - t0:.1f}s)")

    # p_nom_max
    p_nom_max = cfg["capacity_per_sqkm"] * availability @ area

    # Average distance
    layoutmatrix = (layout * availability).stack(spatial=["y", "x"])
    coords = cutout.grid.representative_point().to_crs(3035)
    average_distance = []
    for bus in buses:
        row = layoutmatrix.sel(bus=bus).data
        nz_b = row != 0
        row = row[nz_b]
        co = coords[nz_b]
        distances = co.distance(regions[bus]).div(1e3)
        average_distance.append((distances * (row / row.sum())).sum())
    average_distance = xr.DataArray(average_distance, [buses])

    ds = xr.merge([
        correction_factor * profile,
        p_nom_max.rename("p_nom_max"),
        average_distance.rename("average_distance"),
    ])

    # Filter low-quality buses
    mean_profile = ds["profile"].mean("time")
    if "year" in ds.indexes:
        mean_profile = mean_profile.max("year")
    ds = ds.sel(bus=(mean_profile > cfg["clip_p_max_pu"]) & (ds["p_nom_max"] > 0.0))

    if "clip_p_max_pu" in cfg:
        ds["profile"] = ds["profile"].where(ds["profile"] >= cfg["clip_p_max_pu"], 0)

    # Save to cache
    ds.to_netcdf(cache_path)
    logger.info(f"  Saved to {cache_path}")
    return ds


def main():
    logger.info("=" * 60)
    logger.info("Standalone validation profile builder")
    logger.info("=" * 60)

    # Load cutout once
    logger.info(f"Loading cutout: {CUTOUT_PATH.name}")
    cutout = atlite.Cutout(str(CUTOUT_PATH))
    # Select 2023 time range
    cutout = cutout.sel(time=slice("2023-01-01", "2023-12-31"))
    logger.info(f"  Time range: {cutout.data.time.values[0]} to {cutout.data.time.values[-1]}")
    logger.info(f"  Shape: {cutout.shape}")

    for scenario_name, bias_value in BIAS_SCENARIOS.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Scenario: {scenario_name} (bias_corr={bias_value})")
        logger.info(f"{'='*40}")

        # Create resource directory — match scenario naming from scenarios-validation.yaml
        scenario_dir_name = f"base-s{THRESHOLD}-bias{scenario_name}"
        res_dir = RESOURCE_PREFIX / scenario_dir_name
        res_dir.mkdir(parents=True, exist_ok=True)

        for carrier in CARRIERS:
            ds = build_profile(carrier, scenario_name, bias_value, cutout)
            if ds is not None:
                out_path = res_dir / f"profile_{CLUSTERS}_{carrier}.nc"
                if not out_path.exists():
                    ds.to_netcdf(out_path)
                    logger.info(f"  -> {out_path}")
                else:
                    logger.info(f"  -> {out_path} (already exists)")

    logger.info("\nAll profiles built.")
    logger.info(f"Resource directories created at: {RESOURCE_PREFIX}")


if __name__ == "__main__":
    main()

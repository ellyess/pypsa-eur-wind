#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build time series for air and soil temperatures per clustered model region.

Uses ``atlite.Cutout.temperature`` and ``atlite.Cutout.soil_temperature``.
Executed in ``build_sector.smk``.

This version adds wake-shared caching under:
    wake_extra/<shared_files>/temperature/
"""

import atlite
import geopandas as gpd
import numpy as np
import xarray as xr
from pathlib import Path
from _helpers import get_snapshots, set_scenario_config
from dask.distributed import Client, LocalCluster

from wake_helpers import (
    get_offshore_mods,
    get_wake_dir,
    temperature_cache_paths
)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("build_temperature_profiles", clusters=48)

    set_scenario_config(snakemake)

    # ---------------- basic config ----------------
    nprocesses = int(snakemake.threads)

    clusters = int(getattr(snakemake.wildcards, "clusters", 0))
    scope = getattr(snakemake.wildcards, "scope", "total")

    time = get_snapshots(
        snakemake.params.snapshots,
        snakemake.params.drop_leap_day,
    )

    # ---------------- wake-shared cache root ----------------
    mods = get_offshore_mods(snakemake.config)
    wake_dir = get_wake_dir(mods)

    cache_air, cache_soil = temperature_cache_paths(
        wake_dir=wake_dir,
        clusters=clusters,
    )

    # ---------------- CACHE HIT ----------------
    if cache_air.is_file() and cache_soil.is_file():
        xr.open_dataarray(cache_air).to_netcdf(snakemake.output.temp_air)
        xr.open_dataarray(cache_soil).to_netcdf(snakemake.output.temp_soil)
        raise SystemExit(0)

    # ---------------- CACHE MISS: COMPUTE ----------------
    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster)

    try:
        cutout = atlite.Cutout(snakemake.input.cutout).sel(time=time)

        clustered_regions = (
            gpd.read_file(snakemake.input.regions_onshore)
            .set_index("name")
            .buffer(0)
        )

        I = cutout.indicatormatrix(clustered_regions)

        pop_layout = xr.open_dataarray(snakemake.input.pop_layout)
        stacked_pop = pop_layout.stack(spatial=("y", "x"))

        M = I.T.dot(np.diag(I.dot(stacked_pop)))
        nonzero_sum = M.sum(axis=0, keepdims=True)
        nonzero_sum[nonzero_sum == 0.0] = 1.0
        M_tilde = M / nonzero_sum

        # ---- air temperature
        temp_air = cutout.temperature(
            matrix=M_tilde.T,
            index=clustered_regions.index,
            dask_kwargs=dict(scheduler=client),
            show_progress=False,
        )

        # ---- soil temperature
        temp_soil = cutout.soil_temperature(
            matrix=M_tilde.T,
            index=clustered_regions.index,
            dask_kwargs=dict(scheduler=client),
            show_progress=False,
        )

        # ---- write cache
        cache_air.parent.mkdir(parents=True, exist_ok=True)
        temp_air.to_netcdf(cache_air)
        temp_soil.to_netcdf(cache_soil)

        # ---- write rule outputs
        temp_air.to_netcdf(snakemake.output.temp_air)
        temp_soil.to_netcdf(snakemake.output.temp_soil)

    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            cluster.close()
        except Exception:
            pass

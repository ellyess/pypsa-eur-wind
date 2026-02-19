#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Build solar thermal collector profile time series.

Uses ``atlite.Cutout.solar_thermal`` to compute heat generation for clustered onshore
regions from population layout and weather data cutout. The rule is executed in
``build_sector.smk``.

This version adds wake-shared caching under:
    wake_extra/<shared_files>/solar_thermal/

so repeated runs with identical effective inputs reuse the cached NetCDF.
"""

import os
import shutil
from pathlib import Path

import atlite
import geopandas as gpd
import numpy as np
import xarray as xr
from _helpers import get_snapshots, set_scenario_config
from dask.distributed import Client, LocalCluster

from wake_helpers import (
    get_offshore_mods,
    get_wake_dir,
    solar_thermal_cache_path,
)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("build_solar_thermal_profiles", clusters=48)

    set_scenario_config(snakemake)

    # ---- config / inputs
    nprocesses = int(snakemake.threads)

    config = dict(snakemake.params.solar_thermal)
    config.pop("cutout", None)

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)

    # ---- wake-shared cache root (wake_extra/<shared_files>/...)
    mods = get_offshore_mods(snakemake.config)
    wake_dir = get_wake_dir(mods)

    clusters = int(getattr(snakemake.wildcards, "clusters", 0))

    cache_path = solar_thermal_cache_path(
        wake_dir=wake_dir,
        clusters=clusters
    )

    out_path = Path(snakemake.output.solar_thermal)

    # ---------------- CACHE HIT ----------------
    if cache_path.is_file():
        xr.open_dataarray(cache_path).to_netcdf(snakemake.output.solar_thermal)
        raise SystemExit(0)

    # ---------------- CACHE MISS: COMPUTE ----------------
    cluster = LocalCluster(n_workers=nprocesses, threads_per_worker=1)
    client = Client(cluster)

    try:
        cutout = atlite.Cutout(snakemake.input.cutout).sel(time=time)

        clustered_regions = (
            gpd.read_file(snakemake.input.regions_onshore).set_index("name").buffer(0)
        )

        I = cutout.indicatormatrix(clustered_regions)

        pop_layout = xr.open_dataarray(snakemake.input.pop_layout)

        stacked_pop = pop_layout.stack(spatial=("y", "x"))
        M = I.T.dot(np.diag(I.dot(stacked_pop)))

        nonzero_sum = M.sum(axis=0, keepdims=True)
        nonzero_sum[nonzero_sum == 0.0] = 1.0
        M_tilde = M / nonzero_sum

        solar_thermal = cutout.solar_thermal(
            **config,
            matrix=M_tilde.T,
            index=clustered_regions.index,
            dask_kwargs=dict(scheduler=client),
            show_progress=False,
        )

        # Write cache atomically
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        solar_thermal.to_netcdf(cache_path)

        # ---- write rule outputs
        solar_thermal.to_netcdf(snakemake.output.solar_thermal)
    finally:
        try:
            client.close()
        except Exception:
            pass
        try:
            cluster.close()
        except Exception:
            pass

# -*- coding: utf-8 -*-

# SPDX-FileCopyrightText: 2016 - 2023 The Atlite Authors
#
# SPDX-License-Identifier: MIT
"""
Functions for use in conjunction with wind data generation.
"""

import logging
import re
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def extrapolate_wind_speed(ds, to_height, from_height=None, bias_corr=False):
    """
    Extrapolate the wind speed from a given height above ground to another.

    If ds already contains a key refering to wind speeds at the desired to_height,
    no conversion is done and the wind speeds are directly returned.

    Extrapolation of the wind speed follows the logarithmic law as desribed in [1].

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing the wind speed time-series at 'from_height' with key
        'wnd{height:d}m' and the surface orography with key 'roughness' at the
        geographic locations of the wind speeds.
    from_height : int
        (Optional)
        Height (m) from which the wind speeds are interpolated to 'to_height'.
        If not provided, the closest height to 'to_height' is selected.
    to_height : int|float
        Height (m) to which the wind speeds are extrapolated to.
    bias_corr : bool or str or Path, optional
        If False (default), no bias correction is applied.
        If True, bias correction is applied using 'bias-extra/atlite_bias.nc'.
        If a string or Path, it is used as the path to the bias correction
        dataset (must contain 'scalar' and 'offset' variables).

    Returns
    -------
    da : xarray.DataArray
        DataArray containing the extrapolated wind speeds. Name of the DataArray
        is 'wnd{to_height:d}'.

    References
    ----------
    [1] Equation (2) in Andresen, G. et al (2015): 'Validation of Danish wind
    time series from a new global renewable energy atlas for energy system
    analysis'.

    [2] https://en.wikipedia.org/w/index.php?title=Roughness_length&oldid=862127433,
    Retrieved 2019-02-15.
    """
    # Fast lane
    to_name = "wnd{h:0d}m".format(h=int(to_height))
    if to_name in ds:
        return ds[to_name]

    if from_height is None:
        # Determine closest height to to_name
        heights = np.asarray([int(s[3:-1]) for s in ds if re.match(r"wnd\d+m", s)])

        if len(heights) == 0:
            raise AssertionError("Wind speed is not in dataset")

        from_height = heights[np.argmin(np.abs(heights - to_height))]

    from_name = "wnd{h:0d}m".format(h=int(from_height))

    # Wind speed extrapolation
    wnd_spd = ds[from_name] * (
        np.log(to_height / ds["roughness"]) / np.log(from_height / ds["roughness"])
    )

    # Bias correction based on Ellyess et al. (10.1016/j.energy.2024.133759)
    if bias_corr:
        from pathlib import Path

        if isinstance(bias_corr, (str, Path)) and str(bias_corr) not in ("True", "true"):
            bias_path = str(bias_corr)
        else:
            bias_path = "bias-extra/atlite_bias.nc"

        logger.info("Applying wind speed bias correction from %s", bias_path)
        bias_fac = xr.open_dataset(bias_path)
        # # Handle both x/y and lon/lat coordinate conventions
        # if "lon" in bias_fac.dims and "lat" in bias_fac.dims:
        #     interp_kw = dict(lon=wnd_spd.x.values, lat=wnd_spd.y.values)
        # else:
        interp_kw = dict(x=wnd_spd.x.values, y=wnd_spd.y.values)
        scalar = bias_fac.scalar.interp(
            method="nearest", **interp_kw,
            kwargs={"fill_value": 1.0},
        )
        offset = bias_fac.offset.interp(
            method="nearest", **interp_kw,
            kwargs={"fill_value": 0.0},
        )
        wnd_spd = (wnd_spd * scalar) + offset

    wnd_spd.attrs.update(
        {
            "long name": "extrapolated {ht} m wind speed using logarithmic "
            "method with roughness and {hf} m wind speed"
            "".format(ht=to_height, hf=from_height),
            "units": "m s**-1",
        }
    )
    
    return wnd_spd.rename(to_name)

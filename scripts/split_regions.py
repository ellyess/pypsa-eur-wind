# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Variable spatial resolution for wind resources via Voronoi region splitting.

Allows offshore (and optionally onshore) wind regions to be split into
sub-regions using K-means/Voronoi partitioning, decoupling wind resource
resolution from network clustering resolution.

This enables sensitivity analysis of how spatial aggregation affects
capacity factors, capacity allocation, and system costs.

References
----------
Benmoufok et al. (in preparation), "Sensitivity of European power system
  optimisation to wind resource spatial resolution, wake losses, and bias
  correction."
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Polygon
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)

#: Ceiling on how far mesh_region may raise the part count above
#: ceil(area / threshold) while trying to bring every part under the cap.
#: Only a backstop. Measured need on test geometry: 1.2x for a plain rectangle,
#: 1.25x for a notched multipolygon with a detached island, 1.4x for three
#: separate islands, where each component needs its own parts.
_MAX_PARTS_FACTOR = 4

# Type aliases
Geometry = Union[Polygon, MultiPolygon]
ArrayLike2D = Union[np.ndarray, Sequence[Sequence[float]]]
CRSLike = Union[int, str, dict]


def _as_2d_points(points: ArrayLike2D) -> np.ndarray:
    """Validate and coerce points to an (N, 2) float ndarray."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("`points` must have shape (N, 2).")
    if arr.shape[0] == 0:
        raise ValueError("`points` must contain at least one point.")
    return arr


def cluster_points(
    points: ArrayLike2D,
    n_clusters: int,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster 2D points using K-means and return cluster centres.

    Parameters
    ----------
    points : array-like of shape (N, 2)
        Input points to cluster.
    n_clusters : int
        Number of clusters.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_clusters, 2)
        Cluster centre coordinates.
    """
    pts = _as_2d_points(points)
    if n_clusters < 1:
        raise ValueError("`n_clusters` must be >= 1.")
    if n_clusters > len(pts):
        raise ValueError("`n_clusters` cannot exceed number of points.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(
        pts
    )
    return np.asarray(kmeans.cluster_centers_, dtype=float)


def fill_shape_with_points(
    shape: Geometry,
    min_points: int,
    initial_num: int = 50,
    grow_step: int = 10,
    shrink_frac: float = 0.01,
    max_iter: int = 60,
) -> np.ndarray:
    """Generate interior points by grid-sampling a polygon.

    Iteratively increases grid density until at least *min_points* unique
    points lie inside *shape*.

    Parameters
    ----------
    shape : Polygon or MultiPolygon
        Region to fill with points.
    min_points : int
        Minimum number of interior points required.
    initial_num : int
        Initial grid resolution per axis.
    grow_step : int
        Grid resolution increment per iteration.
    shrink_frac : float
        Fractional inward shrinkage of bounds per iteration.
    max_iter : int
        Maximum iterations before raising an error.

    Returns
    -------
    np.ndarray of shape (M, 2)
        Unique interior points (M >= min_points).
    """
    if min_points < 1:
        raise ValueError("`min_points` must be >= 1.")
    if initial_num < 2:
        raise ValueError("`initial_num` must be >= 2.")

    from shapely import contains_xy

    x_min, y_min, x_max, y_max = shape.bounds
    collected = np.empty((0, 2), dtype=float)
    num = int(initial_num)

    for _ in range(max_iter):
        xs = np.linspace(x_min, x_max, num=num)
        ys = np.linspace(y_min, y_max, num=num)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")

        mask = contains_xy(shape, xx.ravel(), yy.ravel())
        new_pts = np.column_stack([xx.ravel()[mask], yy.ravel()[mask]])

        if new_pts.size:
            collected = np.vstack([collected, new_pts])
            uniq = np.unique(collected, axis=0)
            if len(uniq) >= min_points:
                return uniq

        num += int(grow_step)
        dx = (x_max - x_min) * shrink_frac
        dy = (y_max - y_min) * shrink_frac
        x_min += dx
        x_max -= dx
        y_min += dy
        y_max -= dy

        if x_max <= x_min or y_max <= y_min:
            break

    raise RuntimeError(
        f"Could not generate at least {min_points} interior points "
        f"within max_iter={max_iter}."
    )


def voronoi_partition(
    points: ArrayLike2D,
    outline: Geometry,
) -> List[Polygon]:
    """Compute a Voronoi partition of *points* clipped to *outline*.

    Parameters
    ----------
    points : array-like of shape (N, 2)
        Seed points for the partition.
    outline : Polygon or MultiPolygon
        Boundary to clip the Voronoi cells to.

    Returns
    -------
    list of Polygon
        Clipped Voronoi cells.
    """
    pts = _as_2d_points(points)

    if len(pts) == 1:
        if isinstance(outline, Polygon):
            return [outline]
        return list(outline.geoms)

    from shapely import MultiPoint
    from shapely.ops import voronoi_diagram
    from shapely.validation import make_valid

    if not outline.is_valid:
        outline = make_valid(outline)

    mp = MultiPoint([tuple(p) for p in pts])
    vd = voronoi_diagram(mp, envelope=outline)

    cells: List[Polygon] = []
    for cell in vd.geoms:
        if not cell.is_valid:
            cell = make_valid(cell)
        try:
            clipped = cell.intersection(outline)
        except Exception:
            clipped = cell.buffer(0).intersection(outline.buffer(0))
        if clipped.is_empty:
            continue
        if not clipped.is_valid:
            clipped = make_valid(clipped)
        if isinstance(clipped, Polygon):
            cells.append(clipped)
        elif isinstance(clipped, MultiPolygon):
            cells.extend(g for g in clipped.geoms if isinstance(g, Polygon))

    return cells


def mesh_region(
    geometry: Geometry,
    area_km2: float,
    threshold_km2: float,
    random_state: int = 0,
    min_points_factor: int = 5,
) -> List[Polygon]:
    """Split a region into Voronoi cells if it exceeds an area threshold.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon
        Region geometry (in an equal-area CRS).
    area_km2 : float
        Area of the region in km².
    threshold_km2 : float
        Maximum area per sub-region in km².
    random_state : int
        Random seed for K-means.
    min_points_factor : int
        Multiplier for minimum interior points (n_parts * factor).

    Returns
    -------
    list of Polygon
        Original geometry if below threshold, otherwise Voronoi sub-cells.
    """
    if not geometry.is_valid:
        from shapely.validation import make_valid

        geometry = make_valid(geometry)

    if area_km2 <= threshold_km2:
        if isinstance(geometry, Polygon):
            return [geometry]
        return list(geometry.geoms)

    # ceil(area / threshold) is the count you would need if every part came out
    # at exactly the cap. Voronoi cells from K-means centres are unequal in area,
    # so at that count some parts overflow and the threshold this function
    # documents as a maximum is not honoured. It also understates the
    # requirement whenever the region has disconnected components, since a part
    # cannot span a gap. Raise the count until no part exceeds the cap.
    n_parts = int(np.ceil(area_km2 / threshold_km2))
    max_parts = max(n_parts * _MAX_PARTS_FACTOR, n_parts + 1)

    # Convert part areas to km2 using the caller's own area_km2 rather than
    # assuming the CRS is in metres, so the check engages whatever the units.
    to_km2 = area_km2 / geometry.area if geometry.area > 0 else 0.0

    while True:
        inner_pts = fill_shape_with_points(
            geometry, min_points=max(n_parts * min_points_factor, n_parts)
        )
        centers = cluster_points(
            inner_pts, n_clusters=n_parts, random_state=random_state
        )
        parts = voronoi_partition(centers, geometry)

        largest_km2 = max((p.area for p in parts), default=0.0) * to_km2
        if largest_km2 <= threshold_km2:
            return parts

        if n_parts >= max_parts:
            logger.warning(
                "mesh_region: could not bring every part under %.1f km2 within "
                "%d parts; largest is %.1f km2 (%.1f%% over). Returning anyway.",
                threshold_km2,
                max_parts,
                largest_km2,
                100.0 * (largest_km2 / threshold_km2 - 1.0),
            )
            return parts

        n_parts += 1


def split_regions(
    regions: gpd.GeoDataFrame,
    threshold_km2: float,
    bus_main_col: str = "bus_main",
    out_crs: CRSLike = 4326,
    area_crs: CRSLike = "EPSG:6933",
    random_state: int = 0,
) -> gpd.GeoDataFrame:
    """Split all regions so each part is at most *threshold_km2* in area.

    Large offshore or onshore wind regions are partitioned into smaller
    sub-regions using K-means seeded Voronoi diagrams. This enables
    higher-resolution wind resource modelling without changing the
    network clustering level.

    Parameters
    ----------
    regions : GeoDataFrame
        Input regions with a geometry column and bus assignment.
    threshold_km2 : float
        Maximum area per sub-region in km².
    bus_main_col : str
        Column identifying the network bus each region belongs to.
    out_crs : CRS-like
        Output coordinate reference system.
    area_crs : CRS-like
        Equal-area CRS used for area calculations and splitting.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    GeoDataFrame
        Split regions with columns: name, bus_main, country, geometry, area.
    """
    if threshold_km2 <= 0:
        raise ValueError("`threshold_km2` must be positive.")
    if regions.geometry is None:
        raise ValueError("`regions` must have a geometry column.")

    from pyproj import Transformer
    from shapely.ops import transform as shapely_transform

    reg = regions.copy().to_crs(out_crs)
    reg_ea = reg.to_crs(area_crs)
    reg["_area_km2"] = reg_ea.area / 1e6

    to_out = Transformer.from_crs(area_crs, out_crs, always_xy=True)

    parts: List[gpd.GeoDataFrame] = []
    for idx, row in reg.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        bus_main = row.iloc[0]
        area_km2 = float(row["_area_km2"])
        geom_ea = reg_ea.loc[idx, "geometry"]

        sub_geoms_ea = mesh_region(
            geom_ea, area_km2, threshold_km2, random_state=random_state
        )
        sub_geoms = [shapely_transform(to_out.transform, g) for g in sub_geoms_ea]

        parts.append(
            gpd.GeoDataFrame(
                {bus_main_col: [bus_main] * len(sub_geoms), "geometry": sub_geoms},
                crs=out_crs,
            )
        )

    if not parts:
        out = gpd.GeoDataFrame(
            {
                bus_main_col: pd.Series(dtype=object),
                "geometry": gpd.GeoSeries([], crs=out_crs),
            },
            crs=out_crs,
        )
    else:
        out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=out_crs)

    out["region"] = out.groupby(bus_main_col).cumcount().astype(str).str.zfill(5)
    out["name"] = out[bus_main_col].astype(str) + "_" + out["region"]
    out["country"] = out[bus_main_col].astype(str).str[:2]
    out["area"] = out.to_crs(area_crs).area / 1e6

    return out[["name", bus_main_col, "country", "geometry", "area"]].rename(
        columns={bus_main_col: "bus_main"}
    )

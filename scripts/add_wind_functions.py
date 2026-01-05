import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod

from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, Point

def cluster_points(n_clusters, point_list):
    """
    Clusters the inner points of a region into n_clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
        point_list
    )
    cluster_centers = np.array(kmeans.cluster_centers_)
    return cluster_centers

def fill_shape_with_points(shape, oversize_factor, num=50): # pre 11/04/2025
# def fill_shape_with_points(shape, oversize_factor, num=100):
    """
    Fills the shape of the offshore region with points. This is needed for
    splitting the regions into smaller regions.
    """

    inner_points = list()
    x_min, y_min, x_max, y_max = shape.bounds
    iteration = 0
    while True:
        for x in np.linspace(x_min, x_max, num=num):
            for y in np.linspace(y_min, y_max, num=num):
                if Point(x, y).within(shape):
                    inner_points.append((x, y))
        if len(inner_points) > oversize_factor:
            break
        else:
            # perturb bounds that not the same points are added again
            num += 1
            x_min += abs(x_max - x_min) * 0.01
            x_max -= abs(x_max - x_min) * 0.01
            y_min += abs(y_max - y_min) * 0.01
            y_max -= abs(y_max - y_min) * 0.01
    return inner_points

def voronoi_partition_pts(points, outline):
    """
    Compute the polygons of a voronoi partition of `points` within the polygon
    `outline`. Taken from
    https://github.com/FRESNA/vresutils/blob/master/vresutils/graph.py.

    Attributes
    ----------
    points : Nx2 - ndarray[dtype=float]
    outline : Polygon
    Returns
    -------
    polygons : N - ndarray[dtype=Polygon|MultiPolygon]
    """
    # Convert shapes to equidistant projection shapes
    outline = gpd.GeoSeries(outline, crs="4326")[0]

    if len(points) == 1:
        polygons = [outline]
    else:
        xmin, ymin = np.amin(points, axis=0)
        xmax, ymax = np.amax(points, axis=0)
        xspan = xmax - xmin
        yspan = ymax - ymin

        # to avoid any network positions outside all Voronoi cells, append
        # the corners of a rectangle framing these points
        vor = Voronoi(
            np.vstack(
                (
                    points,
                    [
                        [xmin - 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmin - 3.0 * xspan, ymax + 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymin - 3.0 * yspan],
                        [xmax + 3.0 * xspan, ymax + 3.0 * yspan],
                    ],
                )
            )
        )

        polygons = []
        for i in range(len(points)):
            poly = Polygon(vor.vertices[vor.regions[vor.point_region[i]]])

            if not poly.is_valid:
                poly = poly.buffer(0)

            with np.errstate(invalid="ignore"):
                poly = poly.intersection(outline)

            polygons.append(poly)
            
    final_result = []
    for g in polygons:
        if isinstance(g, MultiPolygon):
            final_result.extend(g.geoms)
        else:
            final_result.append(g)
    return final_result


def mesh_region(region, threshold):
    """Split a regional polygon into multiple parts across it's shortest dimension"""
    geometry = region.geometry
    area = region["area"]
    if area < threshold:
        return [geometry]
    else: 
        n_regions = int(np.ceil(area / threshold))
        inner_points = fill_shape_with_points(geometry, n_regions)
        cluster_centers = cluster_points(n_regions, inner_points)
        inner_regions = voronoi_partition_pts(cluster_centers, geometry)
        return inner_regions

def split_regions(regions,threshold):
    """
    Split all regional polygons into multiple parts across it's shortest dimension
    Parameters
    
    Arguments:
    - regions: GeoDataFrame
        A GeoDataFrame containing the regions to be split.
    - threshold: float
        The maximum area of each split region in square kilometers. Regions larger than this
        will be split into smaller regions.
    Returns:
    - regions_split: GeoDataFrame
        A GeoDataFrame containing the split regions with columns:
        'name', 'bus_main', 'country', 'geometry', 'area'.
    """
    regions["area"] = regions.to_crs({'proj':'cea'}).area / 10**6
    splits = []
    for i, region in regions.iterrows():
        inner_regions = gpd.GeoDataFrame({
                'bus_main':region.iloc[0],
                'geometry':gpd.GeoSeries(
                    mesh_region(region, threshold),
                    crs=4326
                    ),
            })
        splits.append(inner_regions)
        
    regions_split = gpd.pd.concat(splits,ignore_index=True)
    regions_split["region"] = regions_split.groupby("bus_main").cumcount().astype(str).str.zfill(5)
    
    regions_split["name"] = regions_split["bus_main"].astype(str) +  "_" + regions_split["region"]
    regions_split['country'] = regions_split['bus_main'].str[:2]
    regions_split["area"] = regions_split.to_crs({'proj':'cea'}).area / 10**6
    return regions_split[['name','bus_main','country','geometry','area']]

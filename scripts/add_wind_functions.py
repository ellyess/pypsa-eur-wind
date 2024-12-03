import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Geod

from scipy.spatial import Voronoi
from sklearn.cluster import KMeans

from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection, Point

def calculate_area(shape, ellipsoid="WGS84"):
    """
    Calculates area in km².
    """
    geod = Geod(ellps=ellipsoid)
    return abs(geod.geometry_area_perimeter(shape)[0]) / 1e6

def cluster_points(n_clusters, point_list):
    """
    Clusters the inner points of a region into n_clusters.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(
        point_list
    )
    cluster_centers = np.array(kmeans.cluster_centers_)
    return cluster_centers

def fill_shape_with_points(shape, oversize_factor, num=10):
    """
    Fills the shape of the offshore region with points. This is needed for
    splitting the regions into smaller regions.
    """

    inner_points = list()
    x_min, y_min, x_max, y_max = shape.bounds
    iteration = 0
    while True:
        for x in np.linspace(x_min, x_max, num=num*int(np.ceil(oversize_factor/4))):
            for y in np.linspace(y_min, y_max, num=num*int(np.ceil(oversize_factor/4))):
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


def mesh_region(geometry, threshold):
    area = calculate_area(geometry)
    if area < threshold:
        return [geometry]
    else: 
        n_regions = int(np.ceil(area / threshold))
        inner_points = fill_shape_with_points(geometry, n_regions)
        cluster_centers = cluster_points(n_regions, inner_points)
        inner_regions = voronoi_partition_pts(cluster_centers, geometry)
        return inner_regions


# def split_regions(regions,threshold):
#     def prep_old_region(regions):
#         regions.columns = ['bus_main', 'geometry']
#         regions['country'] = regions.bus_main.str[:2]
#         regions_old = gpd.read_file('resources/regions_offshore_base_s.geojson')
#         regions_old.columns = ["region","country","geometry"]
#         regions_c = regions_old.sjoin(
#             regions, 
#             on_attribute="country", 
#             how="left", 
#             predicate='within')[['region','bus_main','country', 'geometry']]
#         regions_c['name'] = regions_c.bus_main+'_'+regions_c.region
#         return regions_c
#     # regions = prep_old_region(regions)
#     """Split all regional polygons into multiple parts across it's shortest dimension"""
#     splits = []
#     for i, region in regions.iterrows():
#         inner_regions = gpd.GeoDataFrame({
#             'geometry':gpd.GeoSeries(
#                 mesh_region(region.geometry, threshold),
#                 crs=4326
#                 ),
#             'bus_main':region.iloc[0]
#         })
#         splits.append(inner_regions)
#     regions_split = gpd.pd.concat(splits,ignore_index=True)
#     regions_split["name"] = regions_split["bus_main"] +  "_" + regions_split.index.astype(str) + regions_split.groupby("bus_main").cumcount().astype(str)
#     regions_split['country'] = regions_split['bus_main'].str[:2]
#     return regions_split



# def split_regions(regions,threshold,max_area):
#     """Split all regional polygons into multiple parts across it's shortest dimension"""

#     sea_shape = gpd.read_file('data/north_sea_shape_updated.geojson')
#     splits = []
#     for i, region in regions.iterrows():
#         if (threshold < max_area):
#             if region.geometry.intersects(sea_shape.geometry.union_all()):
#                 inner_regions = gpd.GeoDataFrame({
#                     'name':region.iloc[0],
#                     'bus_main':region.bus_main,
#                     'geometry':gpd.GeoSeries(
#                         mesh_region(region.geometry, threshold),
#                         crs=4326
#                         ),
#                 })
#             else:
#                 inner_regions = gpd.GeoDataFrame({
#                     'name':region.iloc[0],
#                     'bus_main':region.bus_main,
#                     'geometry':gpd.GeoSeries(
#                         [region.geometry],
#                         crs=4326
#                         ),
#                 })
#         else:
#             inner_regions = gpd.GeoDataFrame({
#                 'name':region.iloc[0],
#                 'bus_main':region.iloc[0],
#                 'geometry':gpd.GeoSeries(
#                     mesh_region(region.geometry, threshold),
#                     crs=4326
#                     ),
#             })
#         splits.append(inner_regions)

    
#     regions_split = gpd.pd.concat(splits,ignore_index=True)
#     if (threshold < max_area):
#         regions_split["name"] = regions_split["name"].astype(str) + regions_split.index.astype(str) + regions_split.groupby("bus_main").cumcount().astype(str)
#     else:
#         regions_split["name"] = regions_split["bus_main"].astype(str) +  "_" + regions_split.index.astype(str) + regions_split.groupby("bus_main").cumcount().astype(str)
#     regions_split['country'] = regions_split['bus_main'].str[:2]
#     return regions_split


def split_regions(regions,threshold,max_area):
    """Split all regional polygons into multiple parts across it's shortest dimension"""
    
    if threshold >= max_area:
        max_area = threshold

    splits = []
    for i, region in regions.iterrows():
        inner_regions = gpd.GeoDataFrame({
                'bus_main':region.iloc[0],
                'geometry':gpd.GeoSeries(
                    mesh_region(region.geometry, max_area),
                    crs=4326
                    ),
            })
        splits.append(inner_regions)
        
    regions_split = gpd.pd.concat(splits,ignore_index=True)
    regions_split["region_main"] = regions_split.groupby("bus_main").cumcount().astype(str) #+ regions_split.index.astype(str)
    regions_split["region"] = regions_split["region_main"].str.zfill(5)
    
    splits = []
    if (threshold < max_area):
        sea_shape = gpd.read_file('data/north_sea_shape_updated.geojson')
        splits = []
        for i, region in regions_split.iterrows():
            if region.geometry.intersects(sea_shape.geometry.union_all()):
                inner_regions = gpd.GeoDataFrame({
                    'bus_main':region.bus_main,
                    'region_main':region.region_main,
                    'geometry':gpd.GeoSeries(
                        mesh_region(region.geometry, threshold),
                        crs=4326
                        ),
                })
            else:
                inner_regions = gpd.GeoDataFrame({
                    'bus_main':region.bus_main,
                    'region_main':region.region_main,
                    'geometry':gpd.GeoSeries(
                        [region.geometry],
                        crs=4326
                        ),
                })
            splits.append(inner_regions)
            
        regions_split = gpd.pd.concat(splits,ignore_index=True)
        regions_split["region"] = (regions_split["region_main"] + regions_split.groupby(["bus_main","region_main"]).cumcount().astype(str)).str.zfill(5)
        
    regions_split["name"] = regions_split["bus_main"].astype(str) +  "_" + regions_split["region"]
    regions_split['country'] = regions_split['bus_main'].str[:2]
    regions_split["region_main"] = regions_split["bus_main"].astype(str) +  "_" + regions_split["region_main"].str.zfill(5)

    return regions_split[['name','region_main','bus_main','country','geometry']]



# def more_regions(regions):
#     regions.columns = ['bus_main', 'geometry']
#     regions['country'] = regions.bus_main.str[:2]
#     regions_old = gpd.read_file(snakemake.input.regions_more)
#     regions_old.columns = ["region","country","geometry"]
#     regions_c = regions_old.sjoin(
#         regions, 
#         on_attribute="country", 
#         how="left", 
#         predicate='within')[['bus_main', 'name', 'country', 'geometry']]
#     regions_c['name'] = regions_c.bus_main+'_'+regions_c.region
#     return regions_c
    
    
def bus_region_pnom(scenario_list, prefix_list):
  total_df =[]
  regions = gpd.read_file('../ellyess_extra/regions_offwind-ac_False.geojson')
  regions.columns = ['bus','area','geometry']
  fig, ax = plt.subplots(int(len(scenario_list)/2),2,figsize=(10,12),sharex=True, sharey=True)
  for prefix in prefix_list:
    i = 0
    j = 0
    for scenario in scenario_list:
        if i == int(len(scenario_list)/2):
          i = 0
          j = 1
          
        n = pypsa.Network("../results/"+prefix+"/"+scenario+"/postnetworks/base_s_10_lv1.5___2030.nc")

        df = n.generators.filter(regex="offwind", axis=0).groupby("bus").agg({
          "p_nom_max": np.sum, 
          "p_nom_min": np.sum,
          "p_nom": np.sum,
          "p_nom_opt": np.sum,
          # }).reset_index().merge(regions, on="region")
          }).merge(regions, on="bus")
  
        df['installed'] = (abs(df['p_nom_opt']-df['p_nom']))/df['area']
        df_to_geodf(df, geom_col="geometry", crs="3035").plot(
            column='installed', 
            ax=ax[i,j],
            cmap='hot',
            vmin = 0, vmax=1,
            legend=True,
            legend_kwds={
                'label': scenario,
                'shrink':0.5,
                "orientation": "vertical"
            })
        
        if i != int(len(scenario_list)/2):
          i += 1
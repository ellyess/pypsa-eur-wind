# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Tests for scripts/split_regions.py — variable spatial resolution.
"""

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from split_regions import (
    cluster_points,
    fill_shape_with_points,
    mesh_region,
    split_regions,
    voronoi_partition,
)

# ---------------------------------------------------------------------------
# cluster_points
# ---------------------------------------------------------------------------


class TestClusterPoints:
    def test_basic(self):
        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        centers = cluster_points(pts, n_clusters=2, random_state=42)
        assert centers.shape == (2, 2)

    def test_single_cluster(self):
        pts = np.array([[0, 0], [1, 0], [0, 1]])
        centers = cluster_points(pts, n_clusters=1)
        assert centers.shape == (1, 2)
        # Center should be near the centroid
        centroid = pts.mean(axis=0)
        np.testing.assert_allclose(centers[0], centroid, atol=0.5)

    def test_n_clusters_equals_n_points(self):
        pts = np.array([[0, 0], [10, 10], [20, 20]])
        centers = cluster_points(pts, n_clusters=3, random_state=0)
        assert centers.shape == (3, 2)

    def test_invalid_n_clusters_zero(self):
        pts = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="n_clusters"):
            cluster_points(pts, n_clusters=0)

    def test_invalid_n_clusters_exceeds_points(self):
        pts = np.array([[0, 0], [1, 1]])
        with pytest.raises(ValueError, match="n_clusters"):
            cluster_points(pts, n_clusters=5)

    def test_invalid_points_1d(self):
        with pytest.raises(ValueError, match="shape"):
            cluster_points(np.array([1, 2, 3]), n_clusters=1)

    def test_invalid_points_empty(self):
        with pytest.raises(ValueError, match="at least one"):
            cluster_points(np.empty((0, 2)), n_clusters=1)


# ---------------------------------------------------------------------------
# fill_shape_with_points
# ---------------------------------------------------------------------------


class TestFillShapeWithPoints:
    def test_rectangle(self):
        shape = box(0, 0, 10, 10)
        pts = fill_shape_with_points(shape, min_points=5)
        assert len(pts) >= 5
        # All points should be inside the shape
        for p in pts:
            assert shape.contains(
                __import__("shapely.geometry", fromlist=["Point"]).Point(p)
            )

    def test_small_shape_still_works(self):
        shape = box(0, 0, 0.1, 0.1)
        pts = fill_shape_with_points(shape, min_points=1, initial_num=100)
        assert len(pts) >= 1

    def test_invalid_min_points(self):
        with pytest.raises(ValueError, match="min_points"):
            fill_shape_with_points(box(0, 0, 1, 1), min_points=0)

    def test_invalid_initial_num(self):
        with pytest.raises(ValueError, match="initial_num"):
            fill_shape_with_points(box(0, 0, 1, 1), min_points=1, initial_num=1)


# ---------------------------------------------------------------------------
# voronoi_partition
# ---------------------------------------------------------------------------


class TestVoronoiPartition:
    def test_single_point(self):
        outline = box(0, 0, 10, 10)
        pts = np.array([[5, 5]])
        cells = voronoi_partition(pts, outline)
        assert len(cells) == 1
        # Single cell should be the entire outline
        assert cells[0].equals(outline)

    def test_two_points(self):
        outline = box(0, 0, 10, 10)
        pts = np.array([[2.5, 5], [7.5, 5]])
        cells = voronoi_partition(pts, outline)
        assert len(cells) == 2
        # Total area should approximately equal outline area
        total_area = sum(c.area for c in cells)
        np.testing.assert_allclose(total_area, outline.area, rtol=0.01)

    def test_no_empty_cells(self):
        outline = box(0, 0, 10, 10)
        pts = np.array([[2, 2], [8, 2], [5, 8]])
        cells = voronoi_partition(pts, outline)
        for c in cells:
            assert not c.is_empty
            assert c.area > 0

    def test_multipolygon_outline(self):
        p1 = box(0, 0, 5, 5)
        p2 = box(10, 10, 15, 15)
        outline = MultiPolygon([p1, p2])
        pts = np.array([[2.5, 2.5]])
        cells = voronoi_partition(pts, outline)
        assert len(cells) >= 1


# ---------------------------------------------------------------------------
# mesh_region
# ---------------------------------------------------------------------------


class TestMeshRegion:
    def test_below_threshold_returns_original(self):
        geom = box(0, 0, 1, 1)  # area = 1
        result = mesh_region(geom, area_km2=500, threshold_km2=1000)
        assert len(result) == 1
        assert result[0].equals(geom)

    def test_above_threshold_splits(self):
        geom = box(0, 0, 100, 100)  # large region in projected coords
        result = mesh_region(geom, area_km2=20000, threshold_km2=5000)
        assert len(result) >= 2
        # Total area should be close to original
        total_area = sum(r.area for r in result)
        np.testing.assert_allclose(total_area, geom.area, rtol=0.01)

    def test_exact_threshold(self):
        geom = box(0, 0, 10, 10)
        result = mesh_region(geom, area_km2=5000, threshold_km2=5000)
        assert len(result) == 1  # Should not split at exact threshold

    def test_multipolygon_below_threshold(self):
        p1 = box(0, 0, 5, 5)
        p2 = box(10, 10, 15, 15)
        geom = MultiPolygon([p1, p2])
        result = mesh_region(geom, area_km2=500, threshold_km2=1000)
        assert len(result) == 2  # Returns individual polygons

    @pytest.mark.parametrize(
        "name, geom",
        [
            ("rectangle", box(0, 0, 300e3, 150e3)),
            (
                "notch_arm_island",
                unary_union(
                    [
                        Polygon(
                            [
                                (0, 0),
                                (260e3, 0),
                                (260e3, 120e3),
                                (150e3, 120e3),
                                (150e3, 60e3),
                                (110e3, 60e3),
                                (110e3, 120e3),
                                (0, 120e3),
                            ]
                        ),
                        Polygon(
                            [
                                (260e3, 40e3),
                                (400e3, 55e3),
                                (400e3, 75e3),
                                (260e3, 70e3),
                            ]
                        ),
                        Point(470e3, 30e3).buffer(28e3),
                    ]
                ),
            ),
            (
                "three_islands",
                unary_union(
                    [
                        Point(0, 0).buffer(40e3),
                        Point(200e3, 0).buffer(55e3),
                        Point(400e3, 0).buffer(30e3),
                    ]
                ),
            ),
        ],
    )
    def test_no_part_exceeds_threshold(self, name, geom):
        """threshold_km2 is documented as a maximum, so honour it.

        ceil(area / threshold) is the count needed only if every part came out
        at exactly the cap. Voronoi cells from K-means centres are unequal in
        area, so at that count some parts overflow; the count has to rise until
        none do. Before this was fixed, the notched case returned 9 parts of
        which 5 exceeded the cap, the largest by 20.6%.
        """
        threshold_km2 = 4000.0
        area_km2 = geom.area / 1e6
        parts = mesh_region(geom, area_km2, threshold_km2)

        areas_km2 = np.array([p.area for p in parts]) / 1e6
        assert areas_km2.max() <= threshold_km2 * (1 + 1e-6), (
            f"{name}: largest part {areas_km2.max():,.0f} km2 exceeds the "
            f"{threshold_km2:,.0f} km2 threshold"
        )
        # the partition must still tile the original
        np.testing.assert_allclose(
            unary_union(parts).area, geom.area, rtol=1e-6
        )


# ---------------------------------------------------------------------------
# split_regions (integration test)
# ---------------------------------------------------------------------------


class TestSplitRegions:
    @pytest.fixture
    def sample_regions(self):
        """Create a simple GeoDataFrame with two regions."""
        return gpd.GeoDataFrame(
            {
                "name": ["bus0", "bus1"],
                "geometry": [
                    box(0, 50, 5, 55),  # ~500 km wide region in WGS84
                    box(5, 50, 6, 51),  # small region
                ],
            },
            crs=4326,
        )

    def test_basic_split(self, sample_regions):
        result = split_regions(sample_regions, threshold_km2=10000)
        assert isinstance(result, gpd.GeoDataFrame)
        assert "name" in result.columns
        assert "bus_main" in result.columns
        assert "country" in result.columns
        assert "area" in result.columns
        assert "geometry" in result.columns
        assert len(result) >= 2  # At least the two original regions

    def test_small_threshold_creates_more_parts(self, sample_regions):
        coarse = split_regions(sample_regions, threshold_km2=100000)
        fine = split_regions(sample_regions, threshold_km2=1000)
        assert len(fine) >= len(coarse)

    def test_invalid_threshold(self, sample_regions):
        with pytest.raises(ValueError, match="positive"):
            split_regions(sample_regions, threshold_km2=0)

        with pytest.raises(ValueError, match="positive"):
            split_regions(sample_regions, threshold_km2=-100)

    def test_empty_regions(self):
        empty = gpd.GeoDataFrame(
            {"name": [], "geometry": []},
            crs=4326,
        )
        result = split_regions(empty, threshold_km2=10000)
        assert len(result) == 0

    def test_country_extraction(self, sample_regions):
        # bus_main_col default is first column = "name"
        result = split_regions(sample_regions, threshold_km2=100000)
        # country is derived from bus_main[:2]
        assert all(len(c) == 2 for c in result["country"])

    def test_crs_preserved(self, sample_regions):
        result = split_regions(sample_regions, threshold_km2=10000)
        assert result.crs is not None
        assert result.crs.to_epsg() == 4326

..
  SPDX-FileCopyrightText: 2024 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _region_splitting:

####################################
Variable Spatial Resolution
####################################

By default, PyPSA-Eur represents wind resources at the same spatial resolution
as the network (determined by the ``clusters`` wildcard). This means a
10-cluster model has at most 10 distinct offshore wind capacity factor profiles.

This extension allows decoupling wind resource resolution from network
resolution: large offshore or onshore regions can be split into smaller
sub-regions, each with its own capacity factor profile and resource potential.

Motivation
==========

Spatial aggregation of wind resources has significant effects on power system
modelling results:

- **Capacity factor smoothing**: Large regions average out local wind
  variability, artificially inflating capacity factors in low-wind areas and
  deflating them in high-wind areas.
- **Capacity allocation**: Coarser resolution leads to higher total offshore
  capacity (+3x in extreme cases) because the optimiser cannot distinguish
  between high-quality and low-quality sites within a region.
- **System cost**: Coarsening from 1,000 km² to 100,000 km² per region can
  increase total system cost by up to 8.5%.

Method
======

Regions exceeding the configured area threshold are split using:

1. **Grid sampling**: Interior points are generated within the region geometry.
2. **K-means clustering**: Points are clustered into *N* groups, where
   *N* = ceil(region_area / threshold).
3. **Voronoi partitioning**: Cluster centres seed a Voronoi diagram, clipped to
   the original region boundary.

The splitting is performed in an equal-area CRS (EPSG:6933) to ensure uniform
sub-region sizes, then results are projected back to WGS84.

Each sub-region inherits the bus assignment of its parent region, receives a
unique name (``{bus}_{index}``), and has its area computed for use by the
:ref:`wake_effects` model.

Configuration
=============

.. code-block:: yaml

   clustering:
     region_split:
       enable: false
       offshore_threshold_km2: 10000
       onshore_threshold_km2: false

- ``enable``: Set to ``true`` to activate region splitting.
- ``offshore_threshold_km2``: Maximum area per offshore sub-region in km².
  Set to ``false`` to disable offshore splitting.
- ``onshore_threshold_km2``: Maximum area per onshore sub-region in km².
  Set to ``false`` to disable onshore splitting (recommended for most uses).

Typical threshold values:

.. list-table::
   :header-rows: 1

   * - Threshold
     - Description
     - Approximate regions (North Sea)
   * - 1,000 km²
     - Very fine
     - ~300
   * - 5,000 km²
     - Fine
     - ~60
   * - 10,000 km²
     - Medium (recommended)
     - ~30
   * - 50,000 km²
     - Coarse
     - ~6
   * - 100,000 km²
     - Very coarse
     - ~3

Implementation
==============

Region splitting is implemented in ``scripts/split_regions.py`` and integrated
into ``scripts/cluster_network.py``. It runs after network clustering and
produces split region GeoJSON files that are used by downstream rules
(``determine_availability_matrix``, ``build_renewable_profiles``).

Key functions:

- ``split_regions(regions, threshold_km2)``: Main entry point. Splits a
  GeoDataFrame of regions into sub-regions.
- ``mesh_region(geometry, area_km2, threshold_km2)``: Splits a single region
  geometry if it exceeds the threshold.
- ``voronoi_partition(points, outline)``: Computes Voronoi cells clipped to a
  boundary.
- ``cluster_points(points, n_clusters)``: K-means clustering of seed points.
- ``fill_shape_with_points(shape, min_points)``: Generates interior points for
  seeding.

Interaction with Wake Modelling
================================

The tiered-density wake model (:ref:`wake_effects`) uses region areas to
convert between absolute capacity and capacity density. When region splitting
is enabled, each sub-region has a smaller area, resulting in density tiers
mapped to smaller absolute capacities. This makes the wake model's behaviour
consistent across resolutions.

When using the ``tiered_density`` wake model, region splitting should typically
be enabled to provide physically meaningful sub-region areas.

References
==========

- Benmoufok, E. et al. (in preparation). Sensitivity of European power system
  optimisation to spatial resolution of offshore wind resources.

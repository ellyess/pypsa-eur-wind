# SPDX-FileCopyrightText: 2026 Ellyess Benmoufok
#
# SPDX-License-Identifier: MIT
"""
Tests for wake_helpers.reconstruct_split_availability.

atlite undercounts land-use availability on sub-grid Voronoi regions, so the
fork computes availability on the unsplit regions and redistributes it to the
split sub-regions by exact geometric overlap. The redistribution must conserve
availability: summing the split regions of a parent must return the parent's
availability.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from wake_helpers import reconstruct_split_availability


def _da(data, buses):
    """(bus, y, x) DataArray from a {bus: 2x2 array} dict."""
    arr = np.stack([np.asarray(data[b], dtype=float) for b in buses])
    return xr.DataArray(
        arr,
        dims=("bus", "y", "x"),
        coords={"bus": buses, "y": [0, 1], "x": [0, 1]},
    )


def test_conserves_availability_when_splits_partition_parents():
    # Parent A occupies cells (0,0) and (0,1); B occupies (1,0) and (1,1).
    geom_unsplit = _da(
        {"A": [[0.8, 0.4], [0, 0]], "B": [[0, 0], [1.0, 0.5]]}, ["A", "B"]
    )
    # Availability (with exclusions) is a fraction of the geometry.
    avail_unsplit = _da(
        {"A": [[0.4, 0.1], [0, 0]], "B": [[0, 0], [0.5, 0.5]]}, ["A", "B"]
    )
    # Two splits partition A per cell, one split is all of B.
    geom_split = _da(
        {
            "A1": [[0.5, 0.1], [0, 0]],
            "A2": [[0.3, 0.3], [0, 0]],
            "B1": [[0, 0], [1.0, 0.5]],
        },
        ["A1", "A2", "B1"],
    )
    parent_of = {"A1": "A", "A2": "A", "B1": "B"}

    m_split = reconstruct_split_availability(
        avail_unsplit, geom_unsplit, geom_split, parent_of
    )

    # Conservation: the splits of A sum back to A, B1 equals B.
    a_sum = m_split.sel(bus=["A1", "A2"]).sum("bus")
    np.testing.assert_allclose(
        a_sum.values, avail_unsplit.sel(bus="A").values, atol=1e-12
    )
    np.testing.assert_allclose(
        m_split.sel(bus="B1").values, avail_unsplit.sel(bus="B").values, atol=1e-12
    )


def test_never_exceeds_geometry():
    geom_unsplit = _da({"A": [[0.5, 0.5], [0, 0]]}, ["A"])
    # availability equals geometry -> fraction 1 everywhere in A
    avail_unsplit = _da({"A": [[0.5, 0.5], [0, 0]]}, ["A"])
    geom_split = _da({"A1": [[0.3, 0.2], [0, 0]]}, ["A1"])

    m = reconstruct_split_availability(
        avail_unsplit, geom_unsplit, geom_split, {"A1": "A"}
    )
    assert bool((m <= geom_split + 1e-12).all())
    # fraction was 1, so split availability equals its geometry
    xr.testing.assert_allclose(m.sel(bus="A1"), geom_split.sel(bus="A1"))


def test_zero_availability_parent_gives_zero():
    geom_unsplit = _da({"A": [[1.0, 1.0], [0, 0]]}, ["A"])
    avail_unsplit = _da({"A": [[0.0, 0.0], [0, 0]]}, ["A"])  # entirely excluded
    geom_split = _da({"A1": [[0.6, 0.4], [0, 0]]}, ["A1"])

    m = reconstruct_split_availability(
        avail_unsplit, geom_unsplit, geom_split, {"A1": "A"}
    )
    assert float(m.sum()) == 0.0


def test_handles_cell_with_no_parent_geometry():
    # A parent that does not cover a cell must not divide by zero there.
    geom_unsplit = _da({"A": [[0.5, 0.0], [0, 0]]}, ["A"])
    avail_unsplit = _da({"A": [[0.25, 0.0], [0, 0]]}, ["A"])
    geom_split = _da({"A1": [[0.5, 0.0], [0, 0]]}, ["A1"])

    m = reconstruct_split_availability(
        avail_unsplit, geom_unsplit, geom_split, {"A1": "A"}
    )
    assert np.isfinite(m.values).all()
    assert float(m.sel(bus="A1").isel(y=0, x=1)) == 0.0


def test_output_bus_order_matches_geom_split():
    geom_unsplit = _da({"A": [[1.0, 0], [0, 0]], "B": [[0, 1.0], [0, 0]]}, ["A", "B"])
    avail_unsplit = _da({"A": [[0.5, 0], [0, 0]], "B": [[0, 0.5], [0, 0]]}, ["A", "B"])
    geom_split = _da({"B1": [[0, 1.0], [0, 0]], "A1": [[1.0, 0], [0, 0]]}, ["B1", "A1"])
    m = reconstruct_split_availability(
        avail_unsplit, geom_unsplit, geom_split, {"A1": "A", "B1": "B"}
    )
    assert list(m.coords["bus"].values) == ["B1", "A1"]

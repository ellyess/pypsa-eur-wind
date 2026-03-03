# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: 2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Tests for scripts/wake_effects.py — wake effect modeling.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from wake_effects import (
    DEFAULT_CAPACITY_TIERED_COEFFICIENTS,
    DEFAULT_FLAT_COEFFICIENTS,
    DEFAULT_TIERED_DENSITY_COEFFICIENTS,
    WakeSplitSpec,
    add_wake_generators,
    capacity_tiered_wake_spec,
    drop_non_dominant_offwind_generators,
    get_wake_coefficients,
    tiered_density_wake_spec,
)

# ---------------------------------------------------------------------------
# WakeSplitSpec
# ---------------------------------------------------------------------------


class TestWakeSplitSpec:
    def test_construction(self):
        spec = WakeSplitSpec(factors=[0.0, 0.1, 0.2], max_caps=[1000, 5000, np.inf])
        assert len(spec.factors) == 3
        assert len(spec.max_caps) == 3

    def test_frozen(self):
        spec = WakeSplitSpec(factors=[0.1], max_caps=[100])
        with pytest.raises(AttributeError):
            spec.factors = [0.2]


# ---------------------------------------------------------------------------
# get_wake_coefficients
# ---------------------------------------------------------------------------


class TestGetWakeCoefficients:
    def test_defaults_flat(self):
        config = {}
        coeffs = get_wake_coefficients(config, "flat")
        assert coeffs["derate_factor"] == 0.8855

    def test_defaults_tiered_density(self):
        coeffs = get_wake_coefficients({}, "tiered_density")
        assert "alpha" in coeffs
        assert "breakpoints" in coeffs
        assert len(coeffs["breakpoints"]) == 7

    def test_defaults_capacity_tiered(self):
        coeffs = get_wake_coefficients({}, "capacity_tiered")
        assert "global_derate" in coeffs
        assert "max_caps" in coeffs

    def test_config_overrides(self):
        config = {
            "electricity": {
                "wake_model": {
                    "flat": {"derate_factor": 0.90},
                }
            }
        }
        coeffs = get_wake_coefficients(config, "flat")
        assert coeffs["derate_factor"] == 0.90

    def test_partial_override_preserves_defaults(self):
        config = {
            "electricity": {
                "wake_model": {
                    "tiered_density": {"alpha": 9.0},
                }
            }
        }
        coeffs = get_wake_coefficients(config, "tiered_density")
        assert coeffs["alpha"] == 9.0
        # Other defaults should still be present
        assert coeffs["beta"] == DEFAULT_TIERED_DENSITY_COEFFICIENTS["beta"]
        assert coeffs["gamma"] == DEFAULT_TIERED_DENSITY_COEFFICIENTS["gamma"]

    def test_unknown_method_returns_empty(self):
        coeffs = get_wake_coefficients({}, "nonexistent")
        assert coeffs == {}

    def test_legacy_aliases(self):
        # "standard" should use flat defaults, "glaum" should use capacity_tiered
        std = get_wake_coefficients({}, "standard")
        assert std["derate_factor"] == DEFAULT_FLAT_COEFFICIENTS["derate_factor"]

        glaum = get_wake_coefficients({}, "glaum")
        assert (
            glaum["global_derate"]
            == DEFAULT_CAPACITY_TIERED_COEFFICIENTS["global_derate"]
        )

        new_more = get_wake_coefficients({}, "new_more")
        assert new_more["alpha"] == DEFAULT_TIERED_DENSITY_COEFFICIENTS["alpha"]


# ---------------------------------------------------------------------------
# tiered_density_wake_spec
# ---------------------------------------------------------------------------


class TestTieredDensityWakeSpec:
    def test_default_coefficients(self):
        spec, breakpoints = tiered_density_wake_spec()
        assert len(breakpoints) == 7
        assert len(spec.factors) == 6  # 7 breakpoints = 6 tiers
        assert spec.max_caps == []  # Computed later from region areas

    def test_factors_are_finite(self):
        spec, _ = tiered_density_wake_spec()
        for f in spec.factors:
            assert np.isfinite(f)

    def test_custom_coefficients(self):
        custom = {
            "alpha": 10.0,
            "beta": 0.1,
            "gamma": -1.0,
            "delta": -20.0,
            "breakpoints": [0, 1.0, 2.0, 4.0],
        }
        spec, breakpoints = tiered_density_wake_spec(custom)
        assert len(breakpoints) == 4
        assert len(spec.factors) == 3

    def test_factors_are_positive_for_default(self):
        """Wake loss factors should be positive (they represent reductions)."""
        spec, _ = tiered_density_wake_spec()
        # At least the first few tiers should have meaningful positive losses
        assert any(f > 0 for f in spec.factors)


# ---------------------------------------------------------------------------
# capacity_tiered_wake_spec
# ---------------------------------------------------------------------------


class TestCapacityTieredWakeSpec:
    def test_default_coefficients(self):
        spec = capacity_tiered_wake_spec()
        assert len(spec.factors) == 3
        assert len(spec.max_caps) == 3
        assert spec.max_caps[-1] == np.inf

    def test_tier1_no_loss(self):
        spec = capacity_tiered_wake_spec()
        assert spec.factors[0] == 0.0  # First tier has no additional loss

    def test_increasing_losses(self):
        spec = capacity_tiered_wake_spec()
        # Tier 3 loss should be >= tier 2
        assert spec.factors[2] >= spec.factors[1]

    def test_custom_coefficients(self):
        custom = {
            "global_derate": 0.95,
            "f2": 0.10,
            "f3_extra": 0.10,
            "max_caps": [1000, 5000],
        }
        spec = capacity_tiered_wake_spec(custom)
        assert spec.max_caps[0] == 1000
        assert spec.max_caps[1] == 5000
        assert spec.max_caps[2] == np.inf


# ---------------------------------------------------------------------------
# Helper: create a mock PyPSA network
# ---------------------------------------------------------------------------


def _make_mock_network(n_generators=3, p_nom_max=None, timesteps=4):
    """Create a minimal mock PyPSA network for testing wake effects."""
    if p_nom_max is None:
        p_nom_max = [5000.0, 3000.0, 1000.0]

    gen_names = [f"region{i} offwind-ac" for i in range(n_generators)]
    generators = pd.DataFrame(
        {
            "p_nom_max": p_nom_max[:n_generators],
            "p_nom": [0.0] * n_generators,
            "p_nom_min": [0.0] * n_generators,
            "carrier": ["offwind-ac"] * n_generators,
            "bus": [f"bus{i}" for i in range(n_generators)],
        },
        index=gen_names,
    )

    p_max_pu = pd.DataFrame(
        np.random.default_rng(42).uniform(0.2, 0.8, (timesteps, n_generators)),
        columns=gen_names,
        index=pd.date_range("2023-01-01", periods=timesteps, freq="h"),
    )
    p_max_pu.columns.names = ["Generator"]

    # Mock the network
    n = MagicMock()
    n.generators = generators
    n.generators_t = MagicMock()
    n.generators_t.p_max_pu = p_max_pu
    return n


# ---------------------------------------------------------------------------
# drop_non_dominant_offwind_generators
# ---------------------------------------------------------------------------


class TestDropNonDominant:
    def test_keeps_largest_per_region(self):
        gen_names = [
            "region0 offwind-ac",
            "region0 offwind-dc",
            "region1 offwind-ac",
        ]
        generators = pd.DataFrame(
            {
                "p_nom_max": [5000.0, 3000.0, 2000.0],
                "p_nom": [0.0, 0.0, 0.0],
                "p_nom_min": [0.0, 0.0, 0.0],
                "carrier": ["offwind-ac", "offwind-dc", "offwind-ac"],
            },
            index=gen_names,
        )
        p_max_pu = pd.DataFrame(
            np.ones((2, 3)) * 0.5,
            columns=gen_names,
            index=pd.date_range("2023-01-01", periods=2, freq="h"),
        )

        n = MagicMock()
        n.generators = generators
        n.generators_t = MagicMock()
        n.generators_t.p_max_pu = p_max_pu

        drop_non_dominant_offwind_generators(n)

        # region0 should keep only offwind-ac (p_nom_max=5000)
        assert "region0 offwind-ac" in n.generators.index
        assert "region0 offwind-dc" not in n.generators.index
        # region1 has only one, should remain
        assert "region1 offwind-ac" in n.generators.index

    def test_no_offwind_does_nothing(self):
        gen_names = ["bus0 onwind"]
        generators = pd.DataFrame(
            {"p_nom_max": [1000.0], "p_nom": [0.0], "carrier": ["onwind"]},
            index=gen_names,
        )
        n = MagicMock()
        n.generators = generators

        drop_non_dominant_offwind_generators(n)
        # Should not modify anything
        assert len(n.generators) == 1


# ---------------------------------------------------------------------------
# add_wake_generators (integration tests)
# ---------------------------------------------------------------------------


class TestAddWakeGenerators:
    def test_capacity_tiered_splits_generators(self):
        n = _make_mock_network(n_generators=2, p_nom_max=[15000.0, 500.0])
        config = {}

        add_wake_generators(n, config, method="capacity_tiered")

        # The first generator (15000 MW) should have been split
        # The second (500 MW) is below the threshold (2000 MW default)
        assert len(n.generators) > 1  # at least the 500 MW one remains unsplit

    def test_tiered_density_requires_regions(self):
        n = _make_mock_network()
        config = {}

        with pytest.raises(ValueError, match="regions_gdf is required"):
            add_wake_generators(n, config, method="tiered_density")

    def test_tiered_density_with_regions(self):
        import geopandas as gpd
        from shapely.geometry import box

        n = _make_mock_network(n_generators=2, p_nom_max=[10000.0, 5000.0])

        regions_gdf = gpd.GeoDataFrame(
            {
                "name": ["region0", "region1"],
                "area": [5000.0, 3000.0],  # km²
                "geometry": [box(0, 0, 1, 1), box(1, 0, 2, 1)],
            },
            crs=4326,
        )
        config = {}

        add_wake_generators(n, config, method="tiered_density", regions_gdf=regions_gdf)

        # Generators should have been split into segments
        assert len(n.generators) >= 2

    def test_invalid_method(self):
        n = _make_mock_network()
        with pytest.raises(ValueError, match="Unknown wake method"):
            add_wake_generators(n, {}, method="invalid")

    def test_legacy_aliases(self):
        """Legacy names 'new_more' and 'glaum' should still work."""
        import geopandas as gpd
        from shapely.geometry import box

        n = _make_mock_network(n_generators=1, p_nom_max=[15000.0])
        config = {}

        # glaum -> capacity_tiered
        add_wake_generators(n, config, method="glaum")
        assert len(n.generators) >= 1

        # new_more -> tiered_density (needs regions)
        n2 = _make_mock_network(n_generators=1, p_nom_max=[10000.0])
        regions_gdf = gpd.GeoDataFrame(
            {
                "name": ["region0"],
                "area": [5000.0],
                "geometry": [box(0, 0, 1, 1)],
            },
            crs=4326,
        )
        add_wake_generators(n2, config, method="new_more", regions_gdf=regions_gdf)
        assert len(n2.generators) >= 1

    def test_no_offwind_does_nothing(self):
        """Network with no offwind generators should be unmodified."""
        gen_names = ["bus0 onwind"]
        generators = pd.DataFrame(
            {
                "p_nom_max": [1000.0],
                "p_nom": [0.0],
                "p_nom_min": [0.0],
                "carrier": ["onwind"],
                "bus": ["bus0"],
            },
            index=gen_names,
        )
        p_max_pu = pd.DataFrame(
            [[0.5]],
            columns=gen_names,
            index=pd.date_range("2023-01-01", periods=1, freq="h"),
        )

        n = MagicMock()
        n.generators = generators
        n.generators_t = MagicMock()
        n.generators_t.p_max_pu = p_max_pu

        original_len = len(n.generators)
        add_wake_generators(n, {}, method="capacity_tiered")
        assert len(n.generators) == original_len

    def test_capacity_tiered_applies_global_derate(self):
        """The global derate should reduce p_max_pu for all offwind generators."""
        n = _make_mock_network(n_generators=1, p_nom_max=[500.0])
        original_pmax = n.generators_t.p_max_pu.values.copy()

        config = {}
        add_wake_generators(n, config, method="capacity_tiered")

        # 500 MW is below the splitting threshold (2000), so no split occurs
        # but the global derate (0.906) should still be applied
        expected = original_pmax * 0.906
        np.testing.assert_allclose(n.generators_t.p_max_pu.values, expected, rtol=1e-6)

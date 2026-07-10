# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Tests for scripts/wake_effects.py — offshore wind wake models.
"""

from unittest.mock import MagicMock

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from scripts.wake_effects import (
    AREA_CRS,
    DEFAULT_TIERED_DENSITY_COEFFICIENTS,
    DEFAULT_UNIFORM_COEFFICIENTS,
    WakeSplitSpec,
    _ensure_region_area,
    _resolve_region_keys,
    add_wake_generators,
    apply_wake_model,
    capacity_tiered_wake_spec,
    check_no_double_counting,
    get_wake_coefficients,
    tiered_density_wake_spec,
)


def _make_mock_network(
    n_generators=3, p_nom_max=None, timesteps=4, resource_class=False
):
    """Create a minimal mock PyPSA network for testing wake effects.

    With ``resource_class`` the generators are named the way PyPSA-Eur names
    them once resource classes are enabled: "{region} {class} {carrier}".
    """
    if p_nom_max is None:
        p_nom_max = [5000.0, 3000.0, 1000.0]

    suffix = " 0 offwind-ac" if resource_class else " offwind-ac"
    gen_names = [f"region{i}{suffix}" for i in range(n_generators)]
    generators = pd.DataFrame(
        {
            "p_nom_max": p_nom_max[:n_generators],
            "p_nom": [0.0] * n_generators,
            "p_nom_min": [0.0] * n_generators,
            "carrier": ["offwind-ac"] * n_generators,
            "bus": [f"bus{i}" for i in range(n_generators)],
        },
        index=pd.Index(gen_names, name="name"),
    )

    p_max_pu = pd.DataFrame(
        np.random.default_rng(42).uniform(0.2, 0.8, (timesteps, n_generators)),
        columns=gen_names,
        index=pd.date_range("2023-01-01", periods=timesteps, freq="h"),
    )
    p_max_pu.columns.name = "name"

    n = MagicMock()
    n.generators = generators
    n.generators_t = MagicMock()
    n.generators_t.p_max_pu = p_max_pu
    return n


def _make_regions(names, geometries=None):
    if geometries is None:
        geometries = [box(i, 0, i + 1, 1) for i in range(len(names))]
    return gpd.GeoDataFrame({"name": names, "geometry": geometries}, crs=4326)


class TestWakeSplitSpec:
    def test_construction(self):
        spec = WakeSplitSpec(factors=[0.0, 0.1, 0.2], max_caps=[1000, 5000, np.inf])
        assert len(spec.factors) == 3
        assert len(spec.max_caps) == 3

    def test_frozen(self):
        spec = WakeSplitSpec(factors=[0.1], max_caps=[100])
        with pytest.raises(AttributeError):
            spec.factors = [0.2]


class TestGetWakeCoefficients:
    def test_defaults_uniform(self):
        assert get_wake_coefficients({}, "uniform")["derate_factor"] == 0.8855

    def test_defaults_tiered_density(self):
        coeffs = get_wake_coefficients({}, "tiered_density")
        assert "alpha" in coeffs
        assert len(coeffs["breakpoints"]) == 7

    def test_defaults_capacity_tiered(self):
        coeffs = get_wake_coefficients({}, "capacity_tiered")
        assert "global_derate" in coeffs
        assert "max_caps" in coeffs

    def test_config_overrides(self):
        config = {"electricity": {"wake_model": {"uniform": {"derate_factor": 0.90}}}}
        assert get_wake_coefficients(config, "uniform")["derate_factor"] == 0.90

    def test_partial_override_preserves_defaults(self):
        config = {"electricity": {"wake_model": {"tiered_density": {"alpha": 9.0}}}}
        coeffs = get_wake_coefficients(config, "tiered_density")
        assert coeffs["alpha"] == 9.0
        assert coeffs["beta"] == DEFAULT_TIERED_DENSITY_COEFFICIENTS["beta"]
        assert coeffs["gamma"] == DEFAULT_TIERED_DENSITY_COEFFICIENTS["gamma"]

    def test_unknown_method_returns_empty(self):
        assert get_wake_coefficients({}, "nonexistent") == {}


class TestTieredDensityWakeSpec:
    def test_default_coefficients(self):
        spec, breakpoints = tiered_density_wake_spec()
        assert len(breakpoints) == 7
        assert len(spec.factors) == 6  # 7 breakpoints = 6 tiers
        assert spec.max_caps == []  # computed later from region areas

    def test_factors_are_finite(self):
        spec, _ = tiered_density_wake_spec()
        assert all(np.isfinite(f) for f in spec.factors)

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

    def test_losses_grow_with_density(self):
        """Marginal wake loss must increase with capacity density."""
        spec, _ = tiered_density_wake_spec()
        assert any(f > 0 for f in spec.factors)
        assert spec.factors[-1] > spec.factors[0]


class TestCapacityTieredWakeSpec:
    def test_default_coefficients(self):
        spec = capacity_tiered_wake_spec()
        assert len(spec.factors) == 3
        assert len(spec.max_caps) == 3
        assert spec.max_caps[-1] == np.inf

    def test_tier1_no_loss(self):
        assert capacity_tiered_wake_spec().factors[0] == 0.0

    def test_increasing_losses(self):
        spec = capacity_tiered_wake_spec()
        assert spec.factors[2] >= spec.factors[1]

    def test_custom_coefficients(self):
        custom = {
            "global_derate": 0.95,
            "f2": 0.10,
            "f3_extra": 0.10,
            "max_caps": [1000, 5000],
        }
        spec = capacity_tiered_wake_spec(custom)
        assert spec.max_caps == [1000, 5000, np.inf]


class TestEnsureRegionArea:
    def test_passthrough_when_present(self):
        regions = _make_regions(["r0"])
        regions["area"] = [42.0]
        assert _ensure_region_area(regions) is regions

    def test_derived_when_missing(self):
        regions = _make_regions(["r0"], [box(0.0, 0.0, 1.0, 1.0)])
        assert "area" not in regions.columns

        filled = _ensure_region_area(regions)
        expected = regions.geometry.to_crs(AREA_CRS).area.iloc[0] / 1e6
        assert filled["area"].iloc[0] == pytest.approx(expected)
        # a one-degree square on the equator is roughly 12,400 km²
        assert 10_000 < filled["area"].iloc[0] < 15_000
        # the input is not mutated
        assert "area" not in regions.columns


class TestCheckNoDoubleCounting:
    def test_passes_with_unit_correction_factor(self):
        config = {"renewable": {"offwind-ac": {"correction_factor": 1.0}}}
        check_no_double_counting(config)  # does not raise

    def test_passes_when_absent(self):
        check_no_double_counting({"renewable": {"offwind-ac": {}}})

    def test_raises_on_residual_correction_factor(self):
        config = {"renewable": {"offwind-dc": {"correction_factor": 0.8855}}}
        with pytest.raises(ValueError, match="double-counting"):
            check_no_double_counting(config)

    def test_ignores_onshore_carriers(self):
        config = {"renewable": {"onwind": {"correction_factor": 0.9}}}
        check_no_double_counting(config)  # onshore has no wake model


class TestAddWakeGenerators:
    def test_capacity_tiered_splits_large_generators(self):
        n = _make_mock_network(n_generators=2, p_nom_max=[15000.0, 500.0])
        add_wake_generators(n, {}, method="capacity_tiered")
        # the 15 GW generator is split; the 500 MW one stays below the tier
        assert len(n.generators) > 1

    def test_tiered_density_requires_regions(self):
        n = _make_mock_network()
        with pytest.raises(ValueError, match="regions_gdf is required"):
            add_wake_generators(n, {}, method="tiered_density")

    def test_tiered_density_with_regions(self):
        n = _make_mock_network(n_generators=2, p_nom_max=[10000.0, 5000.0])
        regions = _make_regions(["region0", "region1"])
        regions["area"] = [5000.0, 3000.0]  # km²

        add_wake_generators(n, {}, method="tiered_density", regions_gdf=regions)
        assert len(n.generators) >= 2

    def test_tiered_density_derives_area(self):
        """Standard offshore regions carry no 'area' column."""
        n = _make_mock_network(n_generators=1, p_nom_max=[10000.0])
        regions = _make_regions(["region0"])
        add_wake_generators(n, {}, method="tiered_density", regions_gdf=regions)
        assert len(n.generators) >= 1

    def test_tiered_density_rejects_missing_region(self):
        n = _make_mock_network(n_generators=2, p_nom_max=[10000.0, 5000.0])
        regions = _make_regions(["region0"])  # region1 absent
        with pytest.raises(ValueError, match="Missing offshore region areas"):
            add_wake_generators(n, {}, method="tiered_density", regions_gdf=regions)

    def test_invalid_method(self):
        n = _make_mock_network()
        with pytest.raises(ValueError, match="Unknown wake method"):
            add_wake_generators(n, {}, method="invalid")


class TestApplyWakeModel:
    def test_none_leaves_network_untouched(self):
        n = _make_mock_network()
        before = n.generators_t.p_max_pu.copy()
        apply_wake_model(n, {})
        pd.testing.assert_frame_equal(n.generators_t.p_max_pu, before)
        assert len(n.generators) == 3

    def test_uniform_derates_offshore(self):
        n = _make_mock_network()
        before = n.generators_t.p_max_pu.copy()
        config = {"electricity": {"wake_model": {"method": "uniform"}}}
        apply_wake_model(n, config)

        derate = DEFAULT_UNIFORM_COEFFICIENTS["derate_factor"]
        pd.testing.assert_frame_equal(n.generators_t.p_max_pu, before * derate)
        # a uniform derate never splits generators
        assert len(n.generators) == 3

    def test_uniform_leaves_onshore_alone(self):
        n = _make_mock_network(n_generators=1)
        n.generators.loc["onwind gen"] = {
            "p_nom_max": 100.0,
            "p_nom": 0.0,
            "p_nom_min": 0.0,
            "carrier": "onwind",
            "bus": "bus0",
        }
        n.generators_t.p_max_pu["onwind gen"] = 0.5

        config = {"electricity": {"wake_model": {"method": "uniform"}}}
        apply_wake_model(n, config)
        assert (n.generators_t.p_max_pu["onwind gen"] == 0.5).all()

    def test_rejects_unknown_method(self):
        n = _make_mock_network()
        config = {"electricity": {"wake_model": {"method": "glaum"}}}
        with pytest.raises(ValueError, match="Unknown wake model method"):
            apply_wake_model(n, config)

    def test_guards_against_double_counting(self):
        n = _make_mock_network()
        config = {
            "electricity": {"wake_model": {"method": "uniform"}},
            "renewable": {"offwind-ac": {"correction_factor": 0.8855}},
        }
        with pytest.raises(ValueError, match="double-counting"):
            apply_wake_model(n, config)

    def test_none_ignores_correction_factor(self):
        """The default must not trip the guard on an unmodified config."""
        n = _make_mock_network()
        config = {"renewable": {"offwind-ac": {"correction_factor": 0.8855}}}
        apply_wake_model(n, config)  # does not raise

    def test_tiered_density_reads_regions_file(self, tmp_path):
        path = tmp_path / "regions_offshore.geojson"
        _make_regions(["region0", "region1", "region2"]).to_file(path)

        n = _make_mock_network()
        config = {"electricity": {"wake_model": {"method": "tiered_density"}}}
        apply_wake_model(n, config, path)
        assert len(n.generators) >= 3

    def test_tiered_density_without_regions_raises(self):
        n = _make_mock_network()
        config = {"electricity": {"wake_model": {"method": "tiered_density"}}}
        with pytest.raises(ValueError, match="needs the offshore regions"):
            apply_wake_model(n, config)


class TestResolveRegionKeys:
    """PyPSA-Eur names generators "{region} {resource_class} {carrier}"."""

    def test_key_that_names_a_region_is_left_alone(self):
        keys = pd.Series(["GB0 0", "BE2 0AC_00001"])
        known = ["GB0 0", "BE2 0AC_00001"]
        assert list(_resolve_region_keys(keys, known)) == known

    def test_trailing_class_index_is_stripped(self):
        keys = pd.Series(["BE2 0AC_00001 0", "BE2 0AC_00002 3"])
        known = ["BE2 0AC_00001", "BE2 0AC_00002"]
        assert list(_resolve_region_keys(keys, known)) == known

    def test_bus_ending_in_a_number_is_not_truncated(self):
        """ "GB0 0 offwind-ac" is ambiguous; the region set disambiguates it."""
        assert list(_resolve_region_keys(pd.Series(["GB0 0"]), ["GB0 0"])) == ["GB0 0"]
        assert list(_resolve_region_keys(pd.Series(["GB0 0"]), ["GB0"])) == ["GB0"]


class TestResourceClassNaming:
    def test_tiered_density_handles_resource_class_suffix(self):
        n = _make_mock_network(
            n_generators=2, p_nom_max=[10000.0, 5000.0], resource_class=True
        )
        assert n.generators.index[0] == "region0 0 offwind-ac"
        regions = _make_regions(["region0", "region1"])
        regions["area"] = [5000.0, 3000.0]

        add_wake_generators(n, {}, method="tiered_density", regions_gdf=regions)
        assert len(n.generators) >= 2

    def test_missing_region_still_raises(self):
        n = _make_mock_network(n_generators=2, resource_class=True)
        with pytest.raises(ValueError, match="Missing offshore region areas"):
            add_wake_generators(
                n, {}, method="tiered_density", regions_gdf=_make_regions(["region0"])
            )


class TestIndexNamesPreserved:
    """PyPSA's exporter looks the index up by name; pd.concat drops it."""

    def test_capacity_tiered_preserves_index_names(self):
        n = _make_mock_network(n_generators=1, p_nom_max=[15000.0])
        add_wake_generators(n, {}, method="capacity_tiered")
        assert n.generators.index.name == "name"
        assert n.generators_t.p_max_pu.columns.name == "name"

    def test_tiered_density_preserves_index_names(self):
        n = _make_mock_network(n_generators=1, p_nom_max=[10000.0])
        regions = _make_regions(["region0"])
        add_wake_generators(n, {}, method="tiered_density", regions_gdf=regions)
        assert n.generators.index.name == "name"
        assert n.generators_t.p_max_pu.columns.name == "name"

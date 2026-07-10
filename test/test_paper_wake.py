# SPDX-FileCopyrightText: 2026 Ellyess Benmoufok
#
# SPDX-License-Identifier: MIT
"""
Tests for analysis_scripts/paper_wake — the wake manuscript figure pipeline.
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis_scripts"))

from paper_wake import figures as figs  # noqa: E402
from paper_wake.loader import RESOLUTION, load, summarise  # noqa: E402
from plotlib import use_style  # noqa: E402

MANUSCRIPT = REPO.parent / "Manuscript-PyPSA_Eur-Wake" / "main-updated.tex"


@pytest.fixture(scope="module")
def data():
    return load()


@pytest.fixture(scope="module")
def summary(data):
    return summarise(data)


class TestLoader:
    def test_harmonises_resolution_column(self, data):
        """`split` and `split_km2` both become `resolution`."""
        for frame in (data.wake_losses, data.cf, data.system, data.resolution):
            assert RESOLUTION in frame.columns
            assert "split" not in frame.columns
            assert "split_km2" not in frame.columns

    def test_scenarios_are_canonical(self, data):
        assert set(data.scenarios) <= {"base", "standard", "glaum", "new_more"}

    def test_missing_dir_raises_pointed_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="extract_wake_data.py"):
            load(tmp_path)


class TestSummary:
    def test_one_row_per_scenario_and_resolution(self, data, summary):
        assert len(summary) == len(data.scenarios) * len(data.resolutions)

    def test_baseline_cost_delta_is_zero(self, summary):
        baseline = summary[summary["scenario"] == "base"]["cost_delta_pct"]
        assert (baseline.abs() < 1e-9).all()

    def test_cost_delta_positive_for_wake_models(self, summary):
        """Wake losses can only make the system more expensive."""
        wake = summary[summary["scenario"] != "base"]["cost_delta_pct"]
        assert (wake > 0).all()

    def test_baseline_has_no_wake_loss(self, summary):
        baseline = summary[summary["scenario"] == "base"]["wake_loss_pct"]
        assert (baseline.abs() < 1e-9).all()

    def test_cost_delta_without_baseline_is_null(self, data):
        stripped = data.system[data.system["scenario"] != "base"]
        partial = summarise(
            type(data)(
                wake_losses=data.wake_losses,
                wake_density=data.wake_density,
                cf=data.cf,
                system=stripped,
                resolution=data.resolution,
            )
        )
        assert partial["cost_delta_pct"].isna().all()

    def test_tiered_density_offshore_capacity_is_resolution_invariant(self, summary):
        """The manuscript's central quantitative claim: ~13-14 GW, flat."""
        capacity = summary[summary["scenario"] == "new_more"]["offshore_capacity_gw"]
        assert capacity.max() - capacity.min() < 2.0


class TestFigureRegistry:
    def test_registry_and_external_are_disjoint(self):
        assert not set(figs.FIGURES) & set(figs.EXTERNAL)

    def test_every_manuscript_figure_is_accounted_for(self):
        """No figure in the .tex may be silently unbuildable."""
        if not MANUSCRIPT.is_file():
            pytest.skip("manuscript not available")
        included = set(
            re.findall(r"\\includegraphics\[[^\]]*\]\{images/([^}]+)\}", MANUSCRIPT.read_text())
        )
        assert included, "no figures found in the manuscript"
        unaccounted = included - set(figs.FIGURES) - set(figs.EXTERNAL)
        assert not unaccounted, f"figures nothing knows how to build: {unaccounted}"

    def test_unknown_figure_raises(self, data, summary):
        with pytest.raises(KeyError, match="Unknown figure"):
            figs.build("nope.pdf", data, summary)

    @pytest.mark.parametrize("name", sorted(figs.FIGURES))
    def test_every_registered_figure_builds(self, name, data, summary):
        use_style()
        fig = figs.build(name, data, summary)
        assert fig.get_axes()


class TestSeries:
    def test_summary_is_serialisable(self, summary, tmp_path):
        path = tmp_path / "paper_metrics.csv"
        summary.to_csv(path, index=False)
        assert pd.read_csv(path).shape == summary.shape

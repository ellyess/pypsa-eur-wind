# SPDX-FileCopyrightText: 2026 Ellyess Benmoufok
#
# SPDX-License-Identifier: MIT
"""
Tests for analysis_scripts/plotlib — the shared seaborn plotting layer.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis_scripts"))

from plotlib import order_for, palette_for, savefig, use_style  # noqa: E402
from plotlib.distributions import plot_distribution  # noqa: E402
from plotlib.resolution import plot_vs_density, plot_vs_resolution  # noqa: E402
from plotlib.style import legend_above  # noqa: E402


@pytest.fixture
def styled():
    return use_style()


@pytest.fixture
def wake_df():
    rng = np.random.default_rng(0)
    rows = [
        {"scenario": scenario, "resolution": resolution, "wake_loss": abs(value)}
        for scenario in ("base", "standard", "glaum", "new_more")
        for resolution in (1000, 10000, 100000)
        for value in rng.normal(0.12, 0.02, 50)
    ]
    return pd.DataFrame(rows)


class TestStyle:
    def test_seaborn_does_not_clobber_thesis_rcparams(self, styled):
        """The whole point of use_style: sns.set_theme must not win."""
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["font.size"] == 7.0
        assert mpl.rcParams["xtick.labelsize"] == 6.0
        assert mpl.rcParams["axes.spines.top"] is False
        assert mpl.rcParams["axes.spines.right"] is False
        assert mpl.rcParams["pdf.fonttype"] == 42
        assert mpl.rcParams["savefig.dpi"] == 600
        assert mpl.rcParams["axes.grid"] is False

    def test_returns_layout_constants(self, styled):
        for key in ("cm", "FULL_WIDTH", "HALF_WIDTH", "THIRD_WIDTH", "MAP_WIDTH"):
            assert key in styled
        assert styled["FULL_WIDTH"] > styled["HALF_WIDTH"] > styled["THIRD_WIDTH"]

    def test_idempotent(self):
        first = use_style()
        second = use_style()
        assert first == second
        assert mpl.rcParams["font.size"] == 7.0


class TestPalettes:
    def test_wake_order_is_canonical(self):
        assert order_for(["new_more", "base", "glaum", "standard"]) == [
            "base",
            "standard",
            "glaum",
            "new_more",
        ]

    def test_aliases_resolve(self):
        # "uniform" is an alias of "standard", "tiered-density" of "new_more"
        assert order_for(["tiered-density", "uniform"]) == ["standard", "new_more"]
        assert palette_for(["uniform"]) == palette_for(["standard"])

    def test_wake_palette_selected_for_wake_keys(self):
        colours = palette_for(["base", "standard", "glaum", "new_more"])
        assert colours["glaum"] == "#0072B2"
        assert colours["new_more"] == "#009E73"

    def test_thesis_palette_selected_for_scenario_keys(self):
        colours = palette_for(["base", "bias", "wake", "bias+wake"])
        assert colours["bias"] == "#5DAE8B"

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="No colour defined"):
            palette_for(["not_a_scenario"])

    def test_order_keeps_unknown_keys(self):
        # never silently drop a series
        assert set(order_for(["base", "zzz"])) == {"base", "zzz"}


class TestPlotDistribution:
    @pytest.mark.parametrize("kind", ["pdf", "cdf", "box"])
    def test_each_kind_renders(self, styled, wake_df, kind):
        fig = plot_distribution(wake_df, value="wake_loss", kind=kind)
        assert fig.get_axes()

    @pytest.mark.parametrize("kind", ["pdf", "cdf", "box"])
    def test_split_renders(self, styled, wake_df, kind):
        fig = plot_distribution(
            wake_df, value="wake_loss", kind=kind, split="resolution"
        )
        assert fig.get_axes()

    def test_pdf_split_keeps_both_legends(self, styled, wake_df):
        """The hue legend must survive the split legend being added."""
        fig = plot_distribution(
            wake_df, value="wake_loss", kind="pdf", split="resolution"
        )
        ax = fig.get_axes()[0]
        legends = [ax.get_legend()] + [
            child for child in ax.get_children() if isinstance(child, mpl.legend.Legend)
        ]
        assert len({id(legend) for legend in legends if legend is not None}) == 2

    def test_legend_uses_display_labels(self, styled, wake_df):
        fig = plot_distribution(wake_df, value="wake_loss", kind="cdf")
        texts = [t.get_text() for t in fig.get_axes()[0].get_legend().get_texts()]
        assert "Tiered density" in texts
        assert "new_more" not in texts

    def test_bad_kind_raises(self, styled, wake_df):
        with pytest.raises(ValueError, match="Unknown kind"):
            plot_distribution(wake_df, value="wake_loss", kind="violin")

    def test_missing_column_raises(self, styled, wake_df):
        with pytest.raises(KeyError):
            plot_distribution(wake_df, value="nope", kind="pdf")

    def test_missing_split_raises(self, styled, wake_df):
        with pytest.raises(KeyError, match="no split column"):
            plot_distribution(wake_df, value="wake_loss", split="nope")


class TestResolution:
    @pytest.fixture
    def agg(self, wake_df):
        frame = wake_df.groupby(["scenario", "resolution"], as_index=False)[
            "wake_loss"
        ].mean()
        frame["density"] = frame["resolution"] / 1e5
        return frame

    def test_x_axis_is_inverted_and_log(self, styled, agg):
        """Coarse on the left, fine on the right."""
        ax = plot_vs_resolution(agg, y="wake_loss").get_axes()[0]
        assert ax.get_xscale() == "log"
        left, right = ax.get_xlim()
        assert left > right

    def test_coarse_fine_annotated_once(self, styled, agg):
        """`add_resolution_markers` used to duplicate these on top of the data."""
        ax = plot_vs_resolution(agg, y="wake_loss").get_axes()[0]
        texts = [t.get_text() for t in ax.texts]
        assert texts.count("Coarse") == 1
        assert texts.count("Fine") == 1

    def test_guides_drawn_at_extremes(self, styled, agg):
        ax = plot_vs_resolution(agg, y="wake_loss").get_axes()[0]
        guides = [line for line in ax.lines if line.get_linestyle() == "--"]
        assert len(guides) == 2

    def test_vs_density_renders(self, styled, agg):
        ax = plot_vs_density(agg, y="wake_loss").get_axes()[0]
        assert ax.get_xlabel().startswith("Capacity density")

    def test_missing_column_raises(self, styled, agg):
        with pytest.raises(KeyError):
            plot_vs_resolution(agg, y="nope")


class TestLegendAbove:
    def test_no_handles_is_a_noop(self, styled):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        legend_above(ax)
        assert ax.get_legend() is None


class TestSavefig:
    """Manuscripts reference figures by name, so the extension must survive."""

    def _figure(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        return fig

    def test_png_stays_png(self, styled, tmp_path):
        written = savefig(self._figure(), tmp_path / "fig.png")
        assert written.suffix == ".png"
        assert written.is_file()
        assert not (tmp_path / "fig.pdf").exists()

    def test_pdf_stays_pdf(self, styled, tmp_path):
        written = savefig(self._figure(), tmp_path / "fig.pdf")
        assert written.suffix == ".pdf"
        assert written.is_file()

    def test_creates_parent_directories(self, styled, tmp_path):
        written = savefig(self._figure(), tmp_path / "a" / "b" / "fig.png")
        assert written.is_file()

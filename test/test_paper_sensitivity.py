# SPDX-FileCopyrightText: 2026 Ellyess Benmoufok
#
# SPDX-License-Identifier: MIT
"""
Tests for analysis_scripts/paper_sensitivity — the sensitivity figure pipeline.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "analysis_scripts"))

from paper_sensitivity import figures_tier1, figures_tier2, figures_validation  # noqa: E402
from paper_sensitivity.loader import load, summarise  # noqa: E402
from plotlib import use_style  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (REPO / "plots" / "sensitivity" / "tier1" / "tier1_metrics_all.csv").is_file(),
    reason="sensitivity CSVs not extracted",
)


@pytest.fixture(scope="module")
def data():
    return load()


@pytest.fixture(scope="module")
def summary(data):
    return summarise(data)


class TestLoader:
    def test_drops_machine_specific_paths(self, data):
        """Absolute OneDrive paths must never reach a figure or a table."""
        for frame in (data.tier1, data.tier2):
            assert "path" not in frame.columns
            assert "scenario_folder" not in frame.columns

    def test_scenarios_are_canonical(self, data):
        assert set(data.scenarios) <= {
            "base",
            "standard",
            "biasUniform",
            "bias",
            "wake",
            "bias+wake",
        }

    def test_missing_dir_raises_pointed_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="compare_sensitivity_runs_tier1"):
            load(tmp_path)


class TestSummary:
    def test_both_domains_present(self, summary):
        assert set(summary["domain"]) == {"northsea", "europe"}

    def test_baseline_delta_is_zero_in_each_domain(self, summary):
        baseline = summary[summary["scenario"] == "base"]["objective_delta_pct"]
        assert (baseline.abs() < 1e-9).all()

    def test_delta_is_computed_within_domain(self, summary):
        """A europe row must not be normalised against a north-sea baseline."""
        europe = summary[(summary["domain"] == "europe") & (summary["scenario"] == "base")]
        assert (europe["objective_delta_pct"].abs() < 1e-9).all()

    def test_no_machine_specific_columns(self, summary):
        assert "path" not in summary.columns


class TestFigures:
    def test_groups_are_disjoint_except_shared_panel(self):
        tier1, tier2 = set(figures_tier1.FIGURES), set(figures_tier2.FIGURES)
        assert not tier1 & tier2

    @pytest.mark.parametrize("name", sorted(figures_tier1.FIGURES))
    def test_tier1_figures_build(self, name, data, summary):
        use_style()
        assert figures_tier1.FIGURES[name](data, summary).get_axes()

    @pytest.mark.parametrize("name", sorted(figures_tier2.FIGURES))
    def test_tier2_figures_build(self, name, data, summary):
        use_style()
        assert figures_tier2.FIGURES[name](data, summary).get_axes()

    @pytest.mark.parametrize("name", sorted(figures_validation.FIGURES))
    def test_validation_figures_build(self, name, data, summary):
        if data.validation is None:
            pytest.skip("validation CSV absent")
        use_style()
        assert figures_validation.FIGURES[name](data, summary).get_axes()

    def test_validation_without_data_raises_pointed_error(self, data, summary):
        stripped = type(data)(
            tier1=data.tier1,
            tier1_cf=data.tier1_cf,
            tier2=data.tier2,
            tier2_cf=data.tier2_cf,
            validation=None,
        )
        builder = next(iter(figures_validation.FIGURES.values()))
        with pytest.raises(FileNotFoundError, match="validate_sensitivity_vs_entsoe"):
            builder(stripped, summary)

    def test_wake_manuscript_reuses_the_cross_domain_panel(self):
        """The wake paper's EXTERNAL registry points here for this figure.

        tier-2 writes the panel as .png, while the wake manuscript embeds a PDF
        of it. paper_wake.EXTERNAL is keyed by the filenames the manuscript
        cites, so the two registries agree on the stem, not the extension.
        """
        from paper_wake.figures import EXTERNAL

        stem = "fig_europe_vs_northsea_offwind_cap"
        assert f"{stem}.png" in figures_tier2.FIGURES
        wake_entries = [
            value for name, value in EXTERNAL.items() if name.rsplit(".", 1)[0] == stem
        ]
        assert len(wake_entries) == 1, f"expected one wake entry for {stem}"
        assert "tier2" in wake_entries[0]

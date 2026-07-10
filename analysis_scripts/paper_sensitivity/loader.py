"""
Load the two-tier sensitivity results into tidy tables.

``compare_sensitivity_runs_tier1.py`` and ``_tier2.py`` currently both extract
metrics *and* plot them. This module owns the extraction half, so the figures
in :mod:`paper_sensitivity.figures_tier1` and ``figures_tier2`` read one
harmonised table instead of re-globbing the solved networks.

Tier 1 is the North Sea sweep (bias x wake x three resolutions); tier 2 is the
Europe-wide confirmatory run at two resolutions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from plotlib.io import PLOTS_ROOT
from thesis_colors import canon

__all__ = ["SensitivityData", "load", "summarise"]

_TIER1 = {
    "metrics": "tier1/tier1_metrics_all.csv",
    "cf": "tier1/tier1_cf_long_on_off.csv",
}
_TIER2 = {
    "metrics": "tier2/tier2_metrics.csv",
    "cf": "tier2/tier2_cf_long.csv",
}
_VALIDATION = {"cf": "validation_entsoe/validation_metrics_cf.csv"}

#: Columns carrying an absolute path to the run that produced the row. They
#: are machine-specific, so they never reach a figure or a metrics table.
_DROP = ("path", "scenario_folder")


def _harmonise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.drop(columns=[c for c in _DROP if c in frame.columns])
    if "scenario" in frame.columns:
        frame["scenario"] = frame["scenario"].astype(str).map(canon)
    if "resolution" in frame.columns:
        frame["resolution"] = pd.to_numeric(frame["resolution"], errors="coerce")
    return frame


@dataclass(frozen=True)
class SensitivityData:
    """Tier 1, tier 2 and the ENTSO-E validation, harmonised."""

    tier1: pd.DataFrame
    tier1_cf: pd.DataFrame
    tier2: pd.DataFrame
    tier2_cf: pd.DataFrame
    validation: pd.DataFrame | None

    @property
    def scenarios(self) -> list[str]:
        return sorted(self.tier1["scenario"].unique())

    @property
    def resolutions(self) -> list[int]:
        return sorted(self.tier1["resolution"].unique())


def load(plots_dir: str | Path | None = None) -> SensitivityData:
    """Read the tier CSVs from *plots_dir* (default ``plots/sensitivity``)."""
    root = Path(plots_dir) if plots_dir else PLOTS_ROOT / "sensitivity"

    required = {**_TIER1, **_TIER2}
    missing = [name for name in required.values() if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {root}. Run compare_sensitivity_runs_tier1.py "
            "and _tier2.py first (see run_all.py)."
        )

    validation_path = root / _VALIDATION["cf"]
    validation = (
        _harmonise(pd.read_csv(validation_path)) if validation_path.is_file() else None
    )

    return SensitivityData(
        tier1=_harmonise(pd.read_csv(root / _TIER1["metrics"])),
        tier1_cf=_harmonise(pd.read_csv(root / _TIER1["cf"])),
        tier2=_harmonise(pd.read_csv(root / _TIER2["metrics"])),
        tier2_cf=_harmonise(pd.read_csv(root / _TIER2["cf"])),
        validation=validation,
    )


def summarise(data: SensitivityData) -> pd.DataFrame:
    """One row per tier x scenario x resolution, with the headline metrics."""
    shared = [
        "scenario",
        "resolution",
        "bias",
        "wake",
        "offwind_cap_gw",
        "onwind_cap_gw",
        "offwind_curt_frac",
        "trans_exp_twkm",
        "objective",
    ]

    frames = []
    for tier, frame in (("northsea", data.tier1), ("europe", data.tier2)):
        columns = [column for column in shared if column in frame.columns]
        subset = frame[columns].copy()
        subset.insert(0, "domain", tier)
        subset["objective_beur"] = subset["objective"] / 1e9
        frames.append(subset)

    summary = pd.concat(frames, ignore_index=True)
    summary["objective_delta_pct"] = _objective_delta_pct(summary)
    return summary.sort_values(["domain", "scenario", "resolution"]).reset_index(
        drop=True
    )


def _objective_delta_pct(summary: pd.DataFrame, baseline: str = "base") -> pd.Series:
    """Percent change in objective against *baseline* within each domain."""
    reference = (
        summary[summary["scenario"] == baseline]
        .set_index(["domain", "resolution"])["objective_beur"]
        .to_dict()
    )
    keys = list(zip(summary["domain"], summary["resolution"], strict=True))
    base_objective = pd.Series([reference.get(key) for key in keys], index=summary.index)
    return (summary["objective_beur"] / base_objective - 1.0) * 100.0


def write_metrics(data: SensitivityData, out: str | Path) -> Path:
    """Write ``sensitivity_metrics.csv`` and return its path."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summarise(data).to_csv(out, index=False)
    return out

"""
Load the wake results into one tidy table.

``extract_wake_data.py`` turns solved networks into five CSVs under
``analysis_scripts/data/wake_extracted``. This module reads those, harmonises
their column names, and derives the scalar summary that the manuscript quotes
(``paper_metrics.csv``): one row per scenario x offshore resolution.

Re-pointing the paper at a new model run is a matter of changing
``--data-dir``; nothing here hard-codes a run name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from plotlib.io import DATA_ROOT
from thesis_colors import canon

__all__ = ["WakeData", "load", "summarise"]

#: Canonical name of the resolution column, in km².
RESOLUTION = "resolution"

_FILES = {
    "wake_losses": "wake_losses.csv",
    "wake_density": "wake_density.csv",
    "cf": "cf_metrics.csv",
    "system": "system_metrics.csv",
    "resolution": "resolution_metrics.csv",
}

# extract_wake_data.py names the resolution column `split`, except in
# resolution_metrics.csv where it is `split_km2`.
_RESOLUTION_ALIASES = ("split", "split_km2")


@dataclass(frozen=True)
class WakeData:
    """The five extracted tables, harmonised."""

    wake_losses: pd.DataFrame
    wake_density: pd.DataFrame
    cf: pd.DataFrame
    system: pd.DataFrame
    resolution: pd.DataFrame

    @property
    def scenarios(self) -> list[str]:
        return sorted(self.wake_losses["scenario"].unique())

    @property
    def resolutions(self) -> list[int]:
        return sorted(self.wake_losses[RESOLUTION].unique())


def _harmonise(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every table a `scenario` and a `resolution` column."""
    frame = frame.copy()
    for alias in _RESOLUTION_ALIASES:
        if alias in frame.columns:
            frame = frame.rename(columns={alias: RESOLUTION})
            break
    if "scenario" in frame.columns:
        frame["scenario"] = frame["scenario"].astype(str).map(canon)
    if RESOLUTION in frame.columns:
        frame[RESOLUTION] = pd.to_numeric(frame[RESOLUTION], errors="coerce")
    return frame


def load(data_dir: str | Path | None = None) -> WakeData:
    """Read the extracted CSVs from *data_dir*.

    Raises a pointed error if the extraction has not been run, rather than
    failing later with a missing-column traceback.
    """
    root = Path(data_dir) if data_dir else DATA_ROOT / "wake_extracted"

    missing = [name for name in _FILES.values() if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing} under {root}. Run extract_wake_data.py first "
            "(see run_all.py, chapter `wake_extract`)."
        )

    tables = {
        key: _harmonise(pd.read_csv(root / name)) for key, name in _FILES.items()
    }
    return WakeData(**tables)


def summarise(data: WakeData) -> pd.DataFrame:
    """Return one row per scenario x resolution with the headline numbers.

    These are the values the manuscript quotes inline: the mean wake loss,
    the offshore build-out, the system cost and its relative change against
    the wake-free baseline, and the median capacity factors.
    """
    system = data.system.copy()
    system["total_cost_beur"] = system["total_cost_eur"] / 1e9

    summary = system[
        [
            "scenario",
            RESOLUTION,
            "total_cost_beur",
            "offshore_capacity_gw",
            "transmission_capacity_gw",
            "curtailment_twh",
        ]
    ].copy()

    wake = _weighted_wake_loss(data.wake_losses)
    summary = summary.merge(wake, on=["scenario", RESOLUTION], how="left")

    medians = data.cf.groupby(["scenario", RESOLUTION], as_index=False)[
        ["available_cf", "dispatch_cf", "curtailment_cf"]
    ].median()
    medians = medians.rename(
        columns={column: f"{column}_median" for column in medians.columns[2:]}
    )
    summary = summary.merge(medians, on=["scenario", RESOLUTION], how="left")

    summary["cost_delta_pct"] = _cost_delta_pct(summary)
    summary["wake_loss_pct"] = summary["wake_loss_mean"] * 100.0

    return summary.sort_values(["scenario", RESOLUTION]).reset_index(drop=True)


def _weighted_wake_loss(losses: pd.DataFrame) -> pd.DataFrame:
    """Capacity-weighted mean wake loss per scenario x resolution.

    ``extract_wake_data`` emits one row per region with a ``weight`` (built
    offshore MW). The headline the manuscript quotes is the capacity-weighted
    mean; where nothing is built (all weights zero) it degrades to a plain mean
    so the number is still defined.
    """
    frame = losses.copy()
    if "weight" not in frame.columns:
        frame["weight"] = 1.0
    frame["weight"] = frame["weight"].fillna(0.0)
    frame["_wl_w"] = frame["wake_loss"] * frame["weight"]

    grouped = frame.groupby(["scenario", RESOLUTION], as_index=False).agg(
        _num=("_wl_w", "sum"), _den=("weight", "sum"), _plain=("wake_loss", "mean")
    )
    grouped["wake_loss_mean"] = (
        (grouped["_num"] / grouped["_den"]).where(grouped["_den"] > 0, grouped["_plain"])
    )
    return grouped[["scenario", RESOLUTION, "wake_loss_mean"]]


def _cost_delta_pct(summary: pd.DataFrame, baseline: str = "base") -> pd.Series:
    """Percent change in system cost against *baseline* at the same resolution."""
    if baseline not in set(summary["scenario"]):
        return pd.Series(pd.NA, index=summary.index, dtype="Float64")

    reference = (
        summary.loc[summary["scenario"] == baseline]
        .set_index(RESOLUTION)["total_cost_beur"]
        .to_dict()
    )
    base_cost = summary[RESOLUTION].map(reference)
    return (summary["total_cost_beur"] / base_cost - 1.0) * 100.0


def write_metrics(data: WakeData, out: str | Path) -> Path:
    """Write ``paper_metrics.csv`` and return its path."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summarise(data).to_csv(out, index=False)
    return out

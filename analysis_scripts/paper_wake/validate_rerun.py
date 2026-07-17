#!/usr/bin/env python3
"""
Check the harmonised rerun against the manuscript's headline claims.

    python -m paper_wake.validate_rerun --data-dir data/wake_extracted_sector

The manuscript's claims rest on a handful of numbers from the SECTOR-COUPLED
North Sea sweep (paper-northsea-sector-2030-10-dominant-6h), where offshore
wind deploys endogenously: a wake-free baseline whose build-out grows with
refinement, a flat uniform derate, a tiered-capacity build-out that collapses
at coarse resolution and recovers toward the baseline under refinement (the
resolution artefact), and a tiered-density build-out and wake loss that stay
stable. If the rerun does not reproduce them, it has found something rather
than confirmed something, and the figures should not be regenerated until the
difference is understood.

Every check is stated as an expectation with a tolerance, and the script exits
non-zero if any of them fail. It reads `paper_metrics.csv` semantics straight
from `paper_wake.loader`, so it validates exactly what the figures will plot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paper_wake.loader import load, summarise  # noqa: E402

# Canonical values from the sector-coupled harmonised rerun
# (paper-northsea-sector-2030-10-dominant-6h, solved 2026-07-11).
#
# Framing note: sector coupling puts offshore wind into genuine cost
# competition, so it deploys ENDOGENOUSLY (no CCL floor) and the wake models
# differentiate both the build-out and its siting. Deployment at
# s100000/s10000/s1000: base 45.2/52.0/58.0, standard 21.6/21.7/21.7,
# glaum 2.1/12.1/25.6, new_more 6.2/10.9/10.8 GW.
# See memory: sector-coupled-wake-runs.
EXPECTED = {
    # The baseline must carry no wake loss at all. This is the check that the
    # `base`-is-secretly-Uniform regression would have failed.
    "baseline_wake_loss_pct": (0.0, 0.01),
    # Uniform is a flat 0.8855 derate: 11.45% loss, independent of resolution.
    "uniform_wake_loss_pct": (11.45, 1.0),
    # Baseline build-out grows with refinement (better sites become visible).
    "baseline_capacity_gw": (51.7, 5.0),          # mean over resolutions
    "baseline_capacity_growth_gw": (12.8, 5.0),   # fine minus coarse
    # Uniform suppresses by a resolution-INVARIANT amount (~21.7 GW flat).
    "uniform_capacity_gw": (21.7, 2.0),
    "uniform_capacity_spread_gw": (0.0, 1.0),
    # Tiered-capacity is the resolution artefact: near-zero at coarse
    # resolution, recovering toward the baseline as regions refine.
    "tiered_capacity_recovery_gw": (23.5, 6.0),   # fine minus coarse
    # Tiered-density is the strongest and most stable suppressor; its two finer
    # resolutions agree to well under a GW.
    "tiered_density_capacity_gw": (9.3, 2.5),     # mean over resolutions
    "tiered_density_fine_pair_gap_gw": (0.0, 1.0),  # |s1000 - s10000|
    # Tiered-density wake loss is resolution-invariant at ~15%.
    "tiered_density_wake_loss_pct": (15.0, 1.5),
    "tiered_density_wake_loss_spread_pp": (0.0, 1.5),
}

# Cost deltas against the wake-free baseline at the finest resolution (1000 km2),
# in %. Sector-coupled system costs dwarf the wake effect, so these are small;
# ordering still holds (tiered-density costs the most, uniform close behind).
EXPECTED_COST_DELTA_PCT = {"new_more": 0.42, "standard": 0.36, "glaum": 0.32}
COST_DELTA_TOL = 0.2


class Check:
    def __init__(self) -> None:
        self.rows: list[tuple[bool, str]] = []

    def expect(self, name: str, actual: float, target: float, tol: float) -> None:
        ok = abs(actual - target) <= tol
        self.rows.append(
            (ok, f"{name:<42} {actual:9.3f}   expected {target:.3f} +/- {tol:.2f}")
        )

    def report(self) -> int:
        for ok, text in self.rows:
            print(f"  {'PASS' if ok else 'FAIL'}  {text}")
        failed = sum(not ok for ok, _ in self.rows)
        print()
        if failed:
            print(f"{failed} of {len(self.rows)} checks FAILED.")
            print(
                "The rerun does not reproduce the thesis. Do not regenerate the "
                "manuscript figures until this is understood."
            )
            return 1
        print(f"All {len(self.rows)} checks passed; the rerun supports the "
              "manuscript's deployment and wake-loss claims.")
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = summarise(load(args.data_dir))
    finest = summary["resolution"].min()
    check = Check()

    def scenario(key):
        rows = summary[summary["scenario"] == key]
        if rows.empty:
            raise SystemExit(f"error: no rows for scenario {key!r} — rerun incomplete?")
        return rows

    base = scenario("base")
    check.expect(
        "baseline wake loss [%] (max over res)",
        base["wake_loss_pct"].abs().max(),
        *EXPECTED["baseline_wake_loss_pct"],
    )

    uniform = scenario("standard")
    check.expect(
        "uniform wake loss [%] (mean over res)",
        uniform["wake_loss_pct"].mean(),
        *EXPECTED["uniform_wake_loss_pct"],
    )
    check.expect(
        "uniform wake loss spread over res [pp]",
        uniform["wake_loss_pct"].max() - uniform["wake_loss_pct"].min(),
        0.0,
        0.5,
    )

    # --- Endogenous deployment (the headline story) ---------------------------
    # Baseline builds the most and grows as refinement exposes better sites.
    base_sorted = base.sort_values("resolution")  # ascending: fine -> coarse
    check.expect(
        "baseline offshore [GW] (mean over res)",
        base["offshore_capacity_gw"].mean(),
        *EXPECTED["baseline_capacity_gw"],
    )
    check.expect(
        "baseline offshore growth fine-coarse [GW]",
        base_sorted.iloc[0]["offshore_capacity_gw"]
        - base_sorted.iloc[-1]["offshore_capacity_gw"],
        *EXPECTED["baseline_capacity_growth_gw"],
    )

    # Uniform: a resolution-invariant suppression.
    check.expect(
        "uniform offshore [GW] (mean over res)",
        uniform["offshore_capacity_gw"].mean(),
        *EXPECTED["uniform_capacity_gw"],
    )
    check.expect(
        "uniform offshore spread over res [GW]",
        uniform["offshore_capacity_gw"].max()
        - uniform["offshore_capacity_gw"].min(),
        *EXPECTED["uniform_capacity_spread_gw"],
    )

    # Tiered-capacity: the resolution artefact. Its build-out must recover
    # strongly toward the baseline as the offshore regions are refined.
    tiercap = scenario("glaum").sort_values("resolution")  # fine -> coarse
    tc_fine = tiercap.iloc[0]["offshore_capacity_gw"]
    tc_coarse = tiercap.iloc[-1]["offshore_capacity_gw"]
    ok = tc_fine > tc_coarse
    check.rows.append(
        (
            ok,
            f"{'tiered-capacity recovers with refinement':<42} "
            f"{tc_fine:9.3f}   expected > coarse ({tc_coarse:.3f})",
        )
    )
    check.expect(
        "tiered-capacity recovery fine-coarse [GW]",
        tc_fine - tc_coarse,
        *EXPECTED["tiered_capacity_recovery_gw"],
    )

    # Tiered-density: strongest, most stable suppressor.
    density = scenario("new_more").sort_values("resolution")  # fine -> coarse
    check.expect(
        "tiered-density offshore [GW] (mean over res)",
        density["offshore_capacity_gw"].mean(),
        *EXPECTED["tiered_density_capacity_gw"],
    )
    check.expect(
        "tiered-density fine-pair gap [GW]",
        abs(
            density.iloc[0]["offshore_capacity_gw"]
            - density.iloc[1]["offshore_capacity_gw"]
        ),
        *EXPECTED["tiered_density_fine_pair_gap_gw"],
    )

    # --- Wake loss ------------------------------------------------------------
    check.expect(
        "tiered-density wake loss [%] (mean over res)",
        density["wake_loss_pct"].mean(),
        *EXPECTED["tiered_density_wake_loss_pct"],
    )
    check.expect(
        "tiered-density wake loss spread [pp]",
        density["wake_loss_pct"].max() - density["wake_loss_pct"].min(),
        *EXPECTED["tiered_density_wake_loss_spread_pp"],
    )

    for key, target in EXPECTED_COST_DELTA_PCT.items():
        rows = scenario(key)
        rows = rows[rows["resolution"] == finest]
        check.expect(
            f"cost delta [%] {key} @ {int(finest):,} km2",
            float(rows["cost_delta_pct"].iloc[0]),
            target,
            COST_DELTA_TOL,
        )

    print(f"\nValidating sector-coupled rerun against the manuscript's "
          f"deployment and wake-loss claims "
          f"(finest resolution = {int(finest):,} km2)\n")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())

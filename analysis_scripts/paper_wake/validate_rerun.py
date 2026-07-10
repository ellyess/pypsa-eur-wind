#!/usr/bin/env python3
"""
Check the harmonised rerun against the canonical thesis results.

    python -m paper_wake.validate_rerun

The manuscript's claims rest on a handful of numbers from thesis Fig. 7.9-7.14.
If the rerun does not reproduce them, it has found something rather than
confirmed something, and the figures should not be regenerated until the
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

# Canonical values from the thesis. See AUDIT_and_PLAN.md and Fig. 7.9-7.14.
EXPECTED = {
    # The baseline must carry no wake loss at all. This is the check that the
    # `base`-is-secretly-Uniform regression would have failed.
    "baseline_wake_loss_pct": (0.0, 0.01),
    # Uniform is a flat 0.8855 derate: 11.45% loss, independent of resolution.
    "uniform_wake_loss_pct": (11.45, 1.0),
    # Tiered-density offshore build-out is resolution-invariant at ~13-14 GW.
    "tiered_density_capacity_gw": (13.5, 2.0),
    # ... and its spread across resolutions is small.
    "tiered_density_capacity_spread_gw": (0.0, 2.0),
}

# Cost deltas against the wake-free baseline at the finest resolution, in %.
EXPECTED_COST_DELTA_PCT = {"new_more": 1.77, "standard": 1.49, "glaum": 1.29}
COST_DELTA_TOL = 0.35


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
        print(f"All {len(self.rows)} checks passed; the rerun reproduces the thesis.")
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

    density = scenario("new_more")
    check.expect(
        "tiered-density offshore capacity [GW]",
        density["offshore_capacity_gw"].mean(),
        *EXPECTED["tiered_density_capacity_gw"],
    )
    check.expect(
        "tiered-density capacity spread [GW]",
        density["offshore_capacity_gw"].max() - density["offshore_capacity_gw"].min(),
        *EXPECTED["tiered_density_capacity_spread_gw"],
    )

    # Tiered-capacity must collapse as the offshore regions are refined: its
    # wake loss at the finest resolution is far below its loss at the coarsest.
    tiercap = scenario("glaum").sort_values("resolution")
    coarse = tiercap.iloc[-1]["wake_loss_pct"]
    fine = tiercap.iloc[0]["wake_loss_pct"]
    ok = fine < coarse
    check.rows.append(
        (
            ok,
            f"{'tiered-capacity collapses with refinement':<42} "
            f"{fine:9.3f}   expected < coarse ({coarse:.3f})",
        )
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

    print(f"\nValidating rerun against thesis Fig. 7.9-7.14 "
          f"(finest resolution = {int(finest):,} km2)\n")
    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())

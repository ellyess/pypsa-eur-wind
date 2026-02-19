#!/usr/bin/env python3
"""Generate scenario YAML files from compact profile specifications.

Instead of maintaining hundreds of lines of repetitive YAML, this script
expresses each scenario set as a compact specification (wake models,
thresholds, bias variants, technologies) and generates the full YAML
that PyPSA-Eur's scenario system expects.

Usage:
    python config/generate_scenarios.py --profile wake
    python config/generate_scenarios.py --profile bias
    python config/generate_scenarios.py --profile sensitivity
    python config/generate_scenarios.py --profile all          # generate all profiles
    python config/generate_scenarios.py --list                 # show available profiles

Each profile produces a file at config/scenarios-<profile>.yaml.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Profile definitions
#
# Each profile specifies the Cartesian product to expand:
#   wake_models       - list of wake model names
#   thresholds        - list of offshore threshold values (km^2)
#   bias_values       - list of bias_corr values (bool or str)
#   technologies      - list of renewable carriers to include
#   onshore_threshold - "same" (= offshore_threshold) or "disabled" (= False)
#   extras            - per-(wake, threshold, bias) overrides (e.g. correction_factor)
# ---------------------------------------------------------------------------

STANDARD_TECHS = ["onwind", "offwind-ac", "offwind-dc", "offwind-float"]
COMBINED_TECHS = ["onwind", "offwind-combined"]

PROFILES: dict[str, dict[str, Any]] = {
    "wake": dict(
        wake_models=["base", "standard", "glaum", "new_more"],
        thresholds=[100_000, 50_000, 10_000, 5_000, 1_000],
        bias_values=[False],
        technologies=STANDARD_TECHS,
        onshore_threshold="disabled",
    ),
    "wake-combined": dict(
        wake_models=["base", "standard", "glaum", "new_more"],
        thresholds=[100_000, 50_000, 10_000, 5_000, 1_000],
        bias_values=[False],
        technologies=COMBINED_TECHS,
        onshore_threshold="disabled",
    ),
    "sensitivity": dict(
        wake_models=["base", "new_more"],
        thresholds=[100_000, 50_000, 10_000, 5_000, 1_000],
        bias_values=[False, True],
        technologies=STANDARD_TECHS,
        onshore_threshold="same",
    ),
    "sensitivity-europe": dict(
        wake_models=["base", "new_more"],
        thresholds=[1_000_000, 10_000],
        bias_values=[False, True],
        technologies=STANDARD_TECHS,
        onshore_threshold="same",
    ),
    "spatial": dict(
        wake_models=["base"],
        thresholds=[100_000, 50_000, 10_000, 5_000, 1_000],
        bias_values=[False],
        technologies=STANDARD_TECHS,
        onshore_threshold="same",
    ),
    "bias": dict(
        wake_models=["base"],
        thresholds=[100_000],
        bias_values=[True, False, "Uniform"],
        technologies=STANDARD_TECHS,
        onshore_threshold="same",
        extras={
            ("base", 100_000, "Uniform"): {
                "onwind": {"correction_factor": 0.93},
                "offwind-ac": {"correction_factor": 0.8855},
                "offwind-dc": {"correction_factor": 0.8855},
                "offwind-float": {"correction_factor": 0.8855},
            }
        },
    ),
}


# ---------------------------------------------------------------------------
# Generation logic
# ---------------------------------------------------------------------------


def _bias_label(bias) -> str:
    """Scenario-name label for a bias value."""
    if isinstance(bias, bool):
        return str(bias)
    return str(bias)


def _bias_corr_value(bias):
    """Config value for ``resource.bias_corr``."""
    if bias == "Uniform":
        return False
    return bias


def generate_scenarios(profile_name: str) -> dict:
    """Return the full scenario dictionary for a named profile."""
    if profile_name not in PROFILES:
        raise ValueError(
            f"Unknown profile {profile_name!r}. "
            f"Available: {', '.join(sorted(PROFILES))}"
        )

    prof = PROFILES[profile_name]
    extras = prof.get("extras", {})
    scenarios: dict[str, Any] = {}

    for wake_model, threshold, bias in itertools.product(
        prof["wake_models"], prof["thresholds"], prof["bias_values"]
    ):
        name = f"{wake_model}-s{threshold}-bias{_bias_label(bias)}"

        onshore_thr = threshold if prof["onshore_threshold"] == "same" else False

        renewable: dict[str, Any] = {}
        for tech in prof["technologies"]:
            tech_block: dict[str, Any] = {
                "resource": {"bias_corr": _bias_corr_value(bias)}
            }
            key = (wake_model, threshold, bias)
            if key in extras and tech in extras[key]:
                tech_block.update(extras[key][tech])
            renewable[tech] = tech_block

        scenarios[name] = {
            "offshore_mods": {
                "wake_model": wake_model,
                "offshore_threshold": threshold,
                "onshore_threshold": onshore_thr,
            },
            "renewable": renewable,
        }

    return scenarios


def write_scenarios(profile_name: str, output: Path | None = None) -> Path:
    """Generate and write scenarios for a profile."""
    scenarios = generate_scenarios(profile_name)
    if output is None:
        output = Path(__file__).parent / f"scenarios-{profile_name}.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(scenarios, f, default_flow_style=False, sort_keys=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES) + ["all"],
        help="Which profile to generate (or 'all').",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path (default: config/scenarios-<profile>.yaml).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="list_profiles",
        help="List available profiles and exit.",
    )
    args = parser.parse_args()

    if args.list_profiles:
        print("Available profiles:")
        for name, prof in PROFILES.items():
            n = (
                len(prof["wake_models"])
                * len(prof["thresholds"])
                * len(prof["bias_values"])
            )
            print(f"  {name:25s}  {n:3d} scenarios")
        return

    if args.profile is None:
        parser.error("--profile is required (or use --list)")

    if args.profile == "all":
        for name in PROFILES:
            out = write_scenarios(name)
            n = len(generate_scenarios(name))
            print(f"  {name:25s}  {n:3d} scenarios -> {out}")
    else:
        out = write_scenarios(args.profile, args.output)
        n = len(generate_scenarios(args.profile))
        print(f"Wrote {n} scenarios to {out}")


if __name__ == "__main__":
    main()

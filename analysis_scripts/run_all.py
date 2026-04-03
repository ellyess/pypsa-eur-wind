#!/usr/bin/env python3
"""
Orchestrator for thesis analysis pipeline.

Runs all (or selected) chapter analysis scripts with consistent paths.
Edit the CHAPTERS config below to match your results directory layout.

Usage:
    python run_all.py                          # run everything
    python run_all.py --chapters wake bias     # selected chapters
    python run_all.py --dry-run                # print commands, don't execute
    python run_all.py --list                   # show available chapters
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

RESULTS_ROOT = REPO / "results"
PLOTS_ROOT = REPO / "plots"
DATA_ROOT = HERE / "data"

# ---------------------------------------------------------------------------
# Chapter definitions
#
# Each entry maps a chapter key to:
#   script   - Python script to run (relative to analysis_scripts/)
#   args     - CLI arguments as a list of strings
#   depends  - list of chapter keys that must run first (optional)
#
# Edit paths below to match your actual results layout.
# ---------------------------------------------------------------------------

CHAPTERS = {
    # ---- Wake (Chapter 6) ----
    "wake_extract": {
        "script": "extract_wake_data.py",
        "args": [
            "--results-dir", str(RESULTS_ROOT / "thesis-wake-2030-10-northsea-dominant-6h"),
            "--out-dir", str(DATA_ROOT / "wake_extracted"),
            "--scenarios", "base", "standard", "glaum", "new_more",
            "--splits", "1000", "10000", "100000",
            "--resolution-analysis",
        ],
    },
    "wake_plot": {
        "script": "compare_wake_runs.py",
        "args": [
            "all",
            "--wake-losses", str(DATA_ROOT / "wake_extracted" / "wake_losses.csv"),
            "--wake-density", str(DATA_ROOT / "wake_extracted" / "wake_density.csv"),
            "--cf-metrics", str(DATA_ROOT / "wake_extracted" / "cf_metrics.csv"),
            "--system", str(DATA_ROOT / "wake_extracted" / "system_metrics.csv"),
            "--resolution", str(DATA_ROOT / "wake_extracted" / "resolution_metrics.csv"),
            "--scenarios", "base", "standard", "glaum", "new_more",
            "--networks-dir", str(RESULTS_ROOT / "thesis-wake-2030-10-northsea-dominant-6h"),
            "--out-dir", str(PLOTS_ROOT / "wake_analysis"),
        ],
        "depends": ["wake_extract"],
    },

    # ---- Bias (Chapter 7) ----
    "bias": {
        "script": "compare_bias_runs.py",
        "args": [
            "--raw", str(
                RESULTS_ROOT
                / "thesis-bias-2030-10-northsea-standard-6h"
                / "base-s100000-biasFalse"
                / "networks"
                / "base_s_10_elec_lvopt_.nc"
            ),
            "--corr", str(
                RESULTS_ROOT
                / "thesis-bias-2030-10-northsea-standard-6h"
                / "base-s100000-biasidw"
                / "networks"
                / "base_s_10_elec_lvopt_.nc"
            ),
            "--uniform", str(
                RESULTS_ROOT
                / "thesis-bias-2030-10-northsea-standard-6h"
                / "base-s100000-biasUniform"
                / "networks"
                / "base_s_10_elec_lvopt_.nc"
            ),
            "--out", str(PLOTS_ROOT / "bias"),
        ],
    },

    # ---- Spatial resolution (Chapter 5) ----
    "spatial": {
        "script": "compare_spatial_runs.py",
        "args": [
            "--results-dir", str(RESULTS_ROOT / "thesis-spatial-2030-10-northsea-standard-6h"),
            "--glob", "*/networks/*.nc",
            "--out-dir", str(PLOTS_ROOT / "spatial_diagnostics"),
        ],
    },

    # ---- Sensitivity (Chapter 8 - Tier 1) ----
    "sensitivity_tier1": {
        "script": "compare_sensitivity_runs_tier1.py",
        "args": [
            "--results-root", str(RESULTS_ROOT),
            "--glob", "thesis-sensitivity-2030-10-northsea-dominant-6h/**/postnetworks/*.nc",
            "--outdir", str(PLOTS_ROOT / "sensitivity" / "tier1"),
        ],
    },

    # ---- Sensitivity (Chapter 8 - Tier 2) ----
    "sensitivity_tier2": {
        "script": "compare_sensitivity_runs_tier2.py",
        "args": [
            "--results-root", str(RESULTS_ROOT),
            "--glob", "thesis-sensitivity-2030-30-europe-dominant-6h/**/postnetworks/*.nc",
            "--outdir", str(PLOTS_ROOT / "sensitivity" / "tier2"),
            "--compare", "base", "biasUniform", "bias", "wake", "bias+wake",
        ],
    },

    # ---- Breakpoint fitting (utility) ----
    "breakpoints": {
        "script": "fit_new_more_breakpoints.py",
        "args": [
            "--outdir", str(REPO / "wake_extra" / "new_more_fit"),
        ],
    },
}

# Convenience aliases: --chapters wake  ->  wake_extract + wake_plot
ALIASES = {
    "wake": ["wake_extract", "wake_plot"],
    "sensitivity": ["sensitivity_tier1", "sensitivity_tier2"],
}


def resolve_chapters(names: list[str] | None) -> list[str]:
    """Resolve chapter names (expanding aliases) in dependency order."""
    if names is None:
        return _topo_sort(CHAPTERS.keys())

    expanded: list[str] = []
    for name in names:
        if name in ALIASES:
            expanded.extend(ALIASES[name])
        elif name in CHAPTERS:
            expanded.append(name)
        else:
            print(f"Unknown chapter: {name!r}")
            print(f"Available: {', '.join(sorted(CHAPTERS))} + aliases: {', '.join(sorted(ALIASES))}")
            sys.exit(1)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for ch in expanded:
        if ch not in seen:
            seen.add(ch)
            unique.append(ch)

    return _topo_sort(unique)


def _topo_sort(keys) -> list[str]:
    """Simple topological sort respecting depends_on."""
    keys = list(keys)
    result: list[str] = []
    visited: set[str] = set()

    def visit(k: str):
        if k in visited:
            return
        visited.add(k)
        for dep in CHAPTERS.get(k, {}).get("depends", []):
            if dep in keys:
                visit(dep)
        result.append(k)

    for k in keys:
        visit(k)
    return result


def run_chapter(name: str, *, dry_run: bool = False) -> bool:
    """Run a single chapter. Returns True on success."""
    ch = CHAPTERS[name]
    script = HERE / ch["script"]
    cmd = [sys.executable, str(script)] + ch["args"]

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 60}")

    if dry_run:
        return True

    result = subprocess.run(cmd, cwd=str(HERE))
    if result.returncode != 0:
        print(f"\n[FAIL] {name} exited with code {result.returncode}")
        return False

    print(f"\n[OK] {name}")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Run thesis analysis pipeline (all or selected chapters)."
    )
    ap.add_argument(
        "--chapters", nargs="+", default=None,
        help="Chapter(s) to run. Aliases: wake, sensitivity. Default: all.",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing.",
    )
    ap.add_argument(
        "--list", action="store_true", dest="list_chapters",
        help="List available chapters and exit.",
    )
    args = ap.parse_args()

    if args.list_chapters:
        print("Available chapters:")
        for name, ch in CHAPTERS.items():
            deps = ch.get("depends", [])
            dep_str = f"  (depends: {', '.join(deps)})" if deps else ""
            print(f"  {name:25s} -> {ch['script']}{dep_str}")
        print("\nAliases:")
        for alias, targets in ALIASES.items():
            print(f"  {alias:25s} -> {', '.join(targets)}")
        return

    chapters = resolve_chapters(args.chapters)

    if args.dry_run:
        print("[DRY RUN] Would execute the following:\n")

    failed: list[str] = []
    for name in chapters:
        ok = run_chapter(name, dry_run=args.dry_run)
        if not ok:
            failed.append(name)

    print(f"\n{'=' * 60}")
    if failed:
        print(f"[DONE] {len(chapters) - len(failed)}/{len(chapters)} succeeded. Failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        mode = "DRY RUN" if args.dry_run else "DONE"
        print(f"[{mode}] {len(chapters)}/{len(chapters)} chapters completed.")


if __name__ == "__main__":
    main()

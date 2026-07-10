#!/usr/bin/env python3
"""
Regenerate the wake manuscript's figures and metrics table.

    python -m paper_wake.make_paper                    # into plots/paper_wake
    python -m paper_wake.make_paper --to-manuscript    # into the manuscript
    python -m paper_wake.make_paper --only wake_loss_vs_resolution.pdf

Writing into the manuscript is opt-in: those figures are the canonical ones
copied from the thesis, and regenerating them from a stale extraction would
silently replace the results the paper reports.

Pointing the paper at a different model run is a matter of ``--data-dir``;
no run name is hard-coded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python paper_wake/make_paper.py` as well as `-m paper_wake.make_paper`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from paper_wake import figures as figs  # noqa: E402
from paper_wake.loader import load, summarise  # noqa: E402
from plotlib import savefig, use_style  # noqa: E402
from plotlib.io import DATA_ROOT, MANUSCRIPT_IMAGES, PLOTS_ROOT  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_ROOT / "wake_extracted",
        help="Directory of CSVs written by extract_wake_data.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write figures into (default: plots/paper_wake).",
    )
    parser.add_argument(
        "--to-manuscript",
        action="store_true",
        help=f"Write into the manuscript instead ({MANUSCRIPT_IMAGES}). "
        "This overwrites the canonical figures, so it is opt-in.",
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=None,
        help="Where to write paper_metrics.csv (default: alongside --out).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="FIGURE",
        help="Build only these figures (by manuscript filename).",
    )
    parser.add_argument(
        "--list", action="store_true", help="List the figures and exit."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Report what would be written."
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.out and args.to_manuscript:
        print("error: pass either --out or --to-manuscript, not both", file=sys.stderr)
        return 2
    args.out = args.out or (
        MANUSCRIPT_IMAGES if args.to_manuscript else PLOTS_ROOT / "paper_wake"
    )

    if args.list:
        print("Built from the extracted CSVs:")
        for name in sorted(figs.FIGURES):
            print(f"  {name}")
        print("\nBuilt elsewhere (need networks or geometries):")
        for name, script in sorted(figs.EXTERNAL.items()):
            print(f"  {name:<45} <- {script}")
        return 0

    selected = args.only or sorted(figs.FIGURES)
    unknown = [name for name in selected if name not in figs.FIGURES]
    if unknown:
        external = [name for name in unknown if name in figs.EXTERNAL]
        for name in external:
            print(f"error: {name} is built by {figs.EXTERNAL[name]}", file=sys.stderr)
        truly_unknown = [name for name in unknown if name not in figs.EXTERNAL]
        if truly_unknown:
            print(f"error: unknown figures {truly_unknown}", file=sys.stderr)
        return 2

    use_style()
    data = load(args.data_dir)
    summary = summarise(data)

    print(f"Loaded {len(data.scenarios)} scenarios x {len(data.resolutions)} resolutions")

    metrics_path = args.metrics or (args.out / "paper_metrics.csv")
    if args.dry_run:
        print(f"[dry-run] would write {metrics_path}")
    else:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(metrics_path, index=False)
        print(f"wrote {metrics_path}")

    written, failed = [], []
    for name in selected:
        if args.dry_run:
            print(f"[dry-run] would write {args.out / name}")
            continue
        try:
            fig = figs.build(name, data, summary)
        except Exception as error:  # keep going; report at the end
            failed.append((name, error))
            print(f"FAILED {name}: {error}", file=sys.stderr)
            continue
        path = savefig(fig, args.out / name)
        written.append(path)
        print(f"wrote {path}")

    if not args.dry_run:
        print(f"\n{len(written)} figure(s) written to {args.out}")

    external = sorted(figs.EXTERNAL)
    if external and not args.only:
        print("\nStill produced by other scripts (not regenerated here):")
        for name in external:
            print(f"  {name:<45} <- {figs.EXTERNAL[name]}")

    if failed:
        print(f"\n{len(failed)} figure(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Regenerate the sensitivity manuscript's figures and metrics table.

    python -m paper_sensitivity.make_paper --out /tmp/figs
    python -m paper_sensitivity.make_paper --group tier1
    python -m paper_sensitivity.make_paper --list

Until the sensitivity manuscript exists, the default output is the thesis
chapter's figure directory, so the pipeline stays exercised.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

from paper_sensitivity import figures_tier1, figures_tier2, figures_validation  # noqa: E402
from paper_sensitivity.loader import load, summarise  # noqa: E402
from plotlib import savefig, use_style  # noqa: E402
from plotlib.io import PLOTS_ROOT  # noqa: E402

GROUPS = {
    "tier1": figures_tier1.FIGURES,
    "tier2": figures_tier2.FIGURES,
    "validation": figures_validation.FIGURES,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory holding tier1/, tier2/ and validation_entsoe/ CSVs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PLOTS_ROOT / "sensitivity",
        help="Directory to write figures into.",
    )
    parser.add_argument(
        "--group",
        nargs="+",
        choices=sorted(GROUPS),
        default=sorted(GROUPS),
        help="Which figure groups to build.",
    )
    parser.add_argument("--only", nargs="+", metavar="FIGURE")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list:
        for group in sorted(GROUPS):
            print(f"{group}:")
            for name in sorted(GROUPS[group]):
                print(f"  {name}")
        return 0

    selected = {}
    for group in args.group:
        selected.update({name: (group, fn) for name, fn in GROUPS[group].items()})
    if args.only:
        unknown = [name for name in args.only if name not in selected]
        if unknown:
            print(f"error: unknown figures {unknown}", file=sys.stderr)
            return 2
        selected = {name: selected[name] for name in args.only}

    use_style()
    data = load(args.plots_dir)
    summary = summarise(data)
    print(f"Loaded {len(data.scenarios)} scenarios; validation: {data.validation is not None}")

    metrics_path = args.out / "sensitivity_metrics.csv"
    if args.dry_run:
        print(f"[dry-run] would write {metrics_path}")
    else:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(metrics_path, index=False)
        print(f"wrote {metrics_path}")

    written, failed = [], []
    for name, (group, builder) in sorted(selected.items()):
        target = args.out / group / name
        if args.dry_run:
            print(f"[dry-run] would write {target}")
            continue
        try:
            fig = builder(data, summary)
        except Exception as error:
            failed.append((name, error))
            print(f"FAILED {name}: {error}", file=sys.stderr)
            continue
        path = savefig(fig, target)
        written.append(path)
        print(f"wrote {path}")

    if not args.dry_run:
        print(f"\n{len(written)} figure(s) written to {args.out}")
    if failed:
        print(f"{len(failed)} figure(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

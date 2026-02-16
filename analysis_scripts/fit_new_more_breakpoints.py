#!/usr/bin/env python3
"""
Fit piecewise breakpoints for the new_more wake spec.

Goal:
- Choose breakpoints that best approximate the analytic loss curve.
- Output a CSV with breakpoints and factors, plus a diagnostic plot.

The piecewise approximation is applied to y(x) * x, where y(x) is the
percent total loss and x is MW/km^2. This matches how factors are
computed in scripts.wake_helpers._new_more_spec().

Methods (breakpoint selection):
1) Define the analytic loss curve y(x) and transform it to y(x)*x, which
    is the quantity used to compute marginal losses per density interval.
2) Discretize the domain [0, xmax] into a candidate grid (linear or log).
3) For any interval [x_i, x_j] on the grid, approximate y(x)*x by a
    straight line between the endpoints and compute its squared error
    against the analytic curve on that interval.
4) Use dynamic programming to find the set of n_breaks breakpoints that
    minimizes the total squared error across all intervals.
5) Convert the resulting breakpoints into marginal loss factors by
    taking the secant slope of y(x)*x over each interval and applying the
    same transformation used in _new_more_spec().
6) Export the breakpoints and factors, plus a diagnostic plot comparing
    the analytic curve to the piecewise approximation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Optional thesis styling
try:
    import plotting_style as ps
except Exception:  # pragma: no cover
    ps = None


@dataclass(frozen=True)
class FitResult:
    breakpoints: List[float]
    factors: List[float]
    sse_yx: float
    sse_y: float


def y_loss_percent(x: np.ndarray) -> np.ndarray:
    """Analytic total loss (percent)."""
    alpha = 7.3
    beta = 0.05
    gamma = -0.7
    delta = -14.6
    return alpha * np.exp(-x / beta) + gamma * x + delta


def piecewise_factors(x_breaks: List[float]) -> List[float]:
    """Compute marginal loss fractions per interval from breakpoints."""
    x = np.asarray(x_breaks, dtype=float)
    y = y_loss_percent(x)
    # Secant slope of y(x) * x over each interval
    num = (y[1:] * x[1:]) - (y[:-1] * x[:-1])
    den = (x[1:] - x[:-1])
    slope = num / den
    return (-(slope) / 100.0).tolist()


def _interval_sse(
    xs: np.ndarray,
    yx: np.ndarray,
    i: int,
    j: int,
    weights: np.ndarray | None = None,
) -> float:
    """SSE of linear interpolation of yx between xs[i] and xs[j]."""
    x0 = xs[i]
    x1 = xs[j]
    y0 = yx[i]
    y1 = yx[j]
    if x1 <= x0:
        return float("inf")

    x_seg = xs[i : j + 1]
    yx_true = yx[i : j + 1]
    yx_hat = y0 + (y1 - y0) * (x_seg - x0) / (x1 - x0)

    err = yx_true - yx_hat
    if weights is not None:
        w = weights[i : j + 1]
        return float(np.sum(w * err * err))
    return float(np.sum(err * err))


def fit_breakpoints(
    *,
    xmax: float,
    n_breaks: int,
    grid_size: int,
    grid: str,
    weight: str,
) -> FitResult:
    if n_breaks < 2:
        raise ValueError("n_breaks must be >= 2")

    if grid not in {"linear", "log"}:
        raise ValueError("grid must be 'linear' or 'log'")

    if weight not in {"none", "x"}:
        raise ValueError("weight must be 'none' or 'x'")

    if grid == "linear":
        xs = np.linspace(0.0, xmax, grid_size)
    else:
        # Avoid log(0) by starting at a small epsilon
        eps = max(1e-6, xmax / 1e6)
        xs = np.concatenate([[0.0], np.geomspace(eps, xmax, grid_size - 1)])

    y = y_loss_percent(xs)
    yx = y * xs

    weights = None
    if weight == "x":
        weights = xs / max(xs.max(), 1.0)

    n = len(xs)
    segments = n_breaks - 1

    # Precompute interval SSE
    sse = np.full((n, n), np.inf)
    for i in range(n - 1):
        for j in range(i + 1, n):
            sse[i, j] = _interval_sse(xs, yx, i, j, weights=weights)

    # DP: dp[k, j] = min SSE to reach j with k segments
    dp = np.full((segments + 1, n), np.inf)
    prev = np.full((segments + 1, n), -1, dtype=int)

    # First segment from 0 -> j
    for j in range(1, n):
        dp[1, j] = sse[0, j]
        prev[1, j] = 0

    for k in range(2, segments + 1):
        for j in range(k, n):
            best_val = float("inf")
            best_i = -1
            for i in range(k - 1, j):
                val = dp[k - 1, i] + sse[i, j]
                if val < best_val:
                    best_val = val
                    best_i = i
            dp[k, j] = best_val
            prev[k, j] = best_i

    # Backtrack from end
    idx = n - 1
    bp_idx = [idx]
    for k in range(segments, 0, -1):
        idx = prev[k, idx]
        if idx < 0:
            raise RuntimeError("Failed to reconstruct breakpoints")
        bp_idx.append(idx)

    bp_idx = sorted(bp_idx)
    breakpoints = xs[bp_idx].tolist()

    factors = piecewise_factors(breakpoints)

    # Evaluate SSE in y(x) and y(x)*x for reporting
    yx_hat = np.interp(xs, breakpoints, y_loss_percent(np.array(breakpoints)) * np.array(breakpoints))
    y_hat = np.where(xs > 0, yx_hat / xs, y_loss_percent(xs))

    sse_yx = float(np.sum((yx - yx_hat) ** 2))
    sse_y = float(np.sum((y - y_hat) ** 2))

    return FitResult(breakpoints=breakpoints, factors=factors, sse_yx=sse_yx, sse_y=sse_y)


def _write_outputs(outdir: Path, result: FitResult) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    # CSV with interval factors
    rows = []
    for i in range(len(result.breakpoints) - 1):
        rows.append(
            {
                "x0": result.breakpoints[i],
                "x1": result.breakpoints[i + 1],
                "factor": result.factors[i],
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "new_more_breakpoints.csv", index=False)

    # Text list for quick copy/paste
    list_txt = ", ".join(f"{x:.6g}" for x in result.breakpoints)
    (outdir / "new_more_breakpoints.txt").write_text(list_txt + "\n")


def _plot_fit(outdir: Path, result: FitResult, xmax: float) -> None:
    if ps is not None and hasattr(ps, "thesis_plot_style"):
        ps.thesis_plot_style()

    xs = np.linspace(0.0, xmax, 1000)
    y = y_loss_percent(xs)
    yx = y * xs

    bp = np.array(result.breakpoints)
    y_bp = y_loss_percent(bp)
    yx_bp = y_bp * bp

    yx_hat = np.interp(xs, bp, yx_bp)
    y_hat = np.where(xs > 0, yx_hat / xs, y_loss_percent(xs))

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.5), dpi=300, sharex=True)

    axes[0].plot(xs, y, label="true", linewidth=1.6)
    axes[0].plot(xs, y_hat, label="fit", linewidth=1.6)
    axes[0].scatter(bp, y_bp, s=12, zorder=3, label="fit breaks")
    axes[0].set_ylabel("Loss (percent)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(xs, yx, label="true", linewidth=1.6)
    axes[1].plot(xs, yx_hat, label="fit", linewidth=1.6)
    axes[1].scatter(bp, yx_bp, s=12, zorder=3, label="fit breaks")
    axes[1].set_xlabel(r"Density (MW/km$^2$)")
    axes[1].set_ylabel("Loss * density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "new_more_breakpoints_fit.png")
    plt.close(fig)


def _parse_breaks(breaks: Optional[str]) -> Optional[List[float]]:
    if not breaks:
        return None
    vals = [float(x.strip()) for x in breaks.split(",") if x.strip()]
    if len(vals) < 2:
        return None
    return vals


def _plot_fit_with_existing(
    outdir: Path,
    result: FitResult,
    xmax: float,
    existing_breaks: Optional[List[float]],
) -> None:
    if existing_breaks is None:
        _plot_fit(outdir, result, xmax=xmax)
        return

    if ps is not None and hasattr(ps, "thesis_plot_style"):
        ps.thesis_plot_style()

    xs = np.linspace(0.0, xmax, 1000)
    y = y_loss_percent(xs)
    yx = y * xs

    bp_fit = np.array(result.breakpoints)
    y_bp_fit = y_loss_percent(bp_fit)
    yx_bp_fit = y_bp_fit * bp_fit
    yx_hat_fit = np.interp(xs, bp_fit, yx_bp_fit)
    y_hat_fit = np.where(xs > 0, yx_hat_fit / xs, y_loss_percent(xs))

    bp_cur = np.array(existing_breaks, dtype=float)
    y_bp_cur = y_loss_percent(bp_cur)
    yx_bp_cur = y_bp_cur * bp_cur
    yx_hat_cur = np.interp(xs, bp_cur, yx_bp_cur)
    y_hat_cur = np.where(xs > 0, yx_hat_cur / xs, y_loss_percent(xs))

    fig, axes = plt.subplots(2, 1, figsize=(5.0, 5.5), dpi=300, sharex=True)

    axes[0].plot(xs, y, label="true", linewidth=1.6)
    axes[0].plot(xs, y_hat_fit, label="fit", linewidth=1.6)
    axes[0].plot(xs, y_hat_cur, label="current", linewidth=1.6, linestyle="--")
    axes[0].scatter(bp_fit, y_bp_fit, s=12, zorder=3, label="fit breaks")
    axes[0].scatter(bp_cur, y_bp_cur, s=12, zorder=3, label="current breaks")
    axes[0].set_ylabel("Loss (percent)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(xs, yx, label="true", linewidth=1.6)
    axes[1].plot(xs, yx_hat_fit, label="fit", linewidth=1.6)
    axes[1].plot(xs, yx_hat_cur, label="current", linewidth=1.6, linestyle="--")
    axes[1].scatter(bp_fit, yx_bp_fit, s=12, zorder=3, label="fit breaks")
    axes[1].scatter(bp_cur, yx_bp_cur, s=12, zorder=3, label="current breaks")
    axes[1].set_xlabel(r"Density (MW/km$^2$)")
    axes[1].set_ylabel("Loss * density")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "new_more_breakpoints_fit.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmax", type=float, default=4.0, help="Max density (MW/km^2)")
    ap.add_argument("--n-breaks", type=int, default=7, help="Number of breakpoints (including 0 and xmax)")
    ap.add_argument("--grid-size", type=int, default=300, help="Candidate grid size")
    ap.add_argument("--grid", choices=["linear", "log"], default="log")
    ap.add_argument("--weight", choices=["none", "x"], default="none")
    ap.add_argument(
        "--existing-breaks",
        default="0,0.025,0.05,0.25,1,2.5,4",
        help="Comma-separated breakpoints to plot as the current spec",
    )
    ap.add_argument("--outdir", default="wake_extra/new_more_fit", help="Output directory")
    args = ap.parse_args()

    result = fit_breakpoints(
        xmax=float(args.xmax),
        n_breaks=int(args.n_breaks),
        grid_size=int(args.grid_size),
        grid=str(args.grid),
        weight=str(args.weight),
    )

    outdir = Path(args.outdir)
    _write_outputs(outdir, result)
    existing_breaks = _parse_breaks(args.existing_breaks)
    _plot_fit_with_existing(outdir, result, xmax=float(args.xmax), existing_breaks=existing_breaks)

    print("[OK] Breakpoints:", result.breakpoints)
    print("[OK] Factors:", result.factors)
    print(f"[OK] SSE(y*x)={result.sse_yx:.6g} | SSE(y)={result.sse_y:.6g}")
    print(f"[OK] Wrote outputs to: {outdir}")


if __name__ == "__main__":
    main()

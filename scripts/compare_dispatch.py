#!/usr/bin/env python3
"""
compare_dispatch_offshore.py

Ex-post dispatch comparison for PyPSA(-Eur) solved networks, tailored to:
  - OFFSHORE-ONLY analysis (filter generators by carrier/name)
  - Slightly different generator indices between runs (e.g. wake run splits base generators)

Key idea for "different generators":
  We aggregate both networks' generator dispatch and limits onto a common "GROUP" id,
  e.g. base generator name, by stripping wake split suffixes via regex.
  Then we compare aggregated group-level dispatch:

    P_ref_group(t)  vs  Pmax_alt_group(t)

and redispatch:

    dP_group(t) = P_alt_group(t) - P_ref_group(t)

Outputs:
  - CSV summaries
  - Region-level violation heatmap (time x region)
  - Redispatch summaries by region/carrier (offshore only)

Usage:
  python compare_dispatch_offshore.py --ref ref.nc --alt alt.nc --out outdir \
      --offshore-like offwind --group-regex " offwind-\\w+$" --time-agg D

Notes:
  - Requires snapshots to match exactly (same hours).
  - Region is inferred from bus country / bus name prefix.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pypsa
import matplotlib.pyplot as plt


# ---------------------------
# Helpers
# ---------------------------

def _infer_bus_region(n: pypsa.Network) -> pd.Series:
    """
    Returns region label per bus.

    Preference:
      1) buses['country'] if exists
      2) first two letters of bus name (ISO2-like)
    """
    buses = n.buses.index.to_series()
    if "country" in n.buses.columns:
        c = n.buses["country"].reindex(n.buses.index)
        prefix = buses.str.slice(0, 2).where(buses.str.len() >= 2, "UN")
        return c.fillna(prefix).fillna("UNKNOWN").astype(str)
    return buses.str.slice(0, 2).where(buses.str.len() >= 2, "UN").fillna("UNKNOWN").astype(str)


def _get_dt_hours(n: pypsa.Network) -> pd.Series:
    if hasattr(n, "snapshot_weightings") and "generators" in n.snapshot_weightings.columns:
        dt = n.snapshot_weightings["generators"].reindex(n.snapshots).fillna(1.0)
        return dt.astype(float)
    return pd.Series(1.0, index=n.snapshots, name="dt_hours")


def _assert_snapshots_match(n_ref: pypsa.Network, n_alt: pypsa.Network) -> None:
    if not n_ref.snapshots.equals(n_alt.snapshots):
        raise ValueError(
            "Snapshots do not match between REF and ALT networks.\n"
            "This script compares hour-by-hour; ensure identical snapshots."
        )


def _select_offshore_generators(n: pypsa.Network, offshore_like: str) -> pd.Index:
    """
    Select offshore generators by (carrier contains offshore_like) OR (name contains offshore_like).
    Examples:
      offshore_like='offwind' matches carriers like 'offwind-ac', 'offwind-combined'
    """
    like = str(offshore_like)
    idx = n.generators.index
    carrier = n.generators["carrier"].astype(str)

    mask = carrier.str.contains(like, case=False, regex=False)
    # also allow fallback by name
    mask = mask | idx.to_series().str.contains(like, case=False, regex=False)
    return idx[mask]


def _group_id_from_name(names: pd.Index, group_regex: str) -> pd.Index:
    """
    Build grouping IDs from generator names by applying regex substitution.

    Default intended use:
      If wake split names look like "DE0 0_12345 offwind-3" and base is "DE0 0_12345",
      use group_regex=r" offwind-\\w+$" to strip the suffix -> base group.

    If you need a different mapping, adjust regex.
    """
    pattern = re.compile(group_regex)
    grouped = [pattern.sub("", str(x)) for x in names]
    return pd.Index(grouped, name="group")


def _aggregate_ts_by_group(ts: pd.DataFrame, group: pd.Index) -> pd.DataFrame:
    """
    Aggregate a snapshot x generator table to snapshot x group by summing across generators in each group.
    """
    # Align columns and group index
    if len(ts.columns) != len(group):
        raise ValueError("Group index length must match number of columns in ts.")
    out = ts.copy()
    out.columns = group
    # Sum split generators that map to same group
    return out.groupby(level=0, axis=1).sum()


def _plot_heatmap(mat: pd.DataFrame, out_png: Path, title: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    """
    Plot regions on y-axis, time on x-axis. No seaborn.
    """
    img = mat.T  # rows=regions, cols=time
    fig = plt.figure(figsize=(14, max(5, 0.25 * len(img.index))))
    ax = fig.add_subplot(111)

    data = img.to_numpy(dtype=float)
    im = ax.imshow(data, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Region")

    # x ticks: show up to ~12 labels
    ncols = data.shape[1]
    if ncols > 1:
        step = max(1, ncols // 12)
        xt = np.arange(0, ncols, step)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(img.columns[i])[:10] for i in xt], rotation=45, ha="right")

    ax.set_yticks(np.arange(len(img.index)))
    ax.set_yticklabels(img.index)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Violation share (of offshore groups)")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _time_aggregate_mean(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule.upper() in ["H", "HOURLY", "NONE", "RAW"]:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()
        df.index = pd.to_datetime(df.index)
    return df.resample(rule).mean()


# ---------------------------
# Core logic (offshore-only, group-aware)
# ---------------------------

def compute_offshore_grouped(
    n: pypsa.Network,
    offshore_like: str,
    group_regex: str,
) -> dict[str, object]:
    """
    Extract offshore-only generator dispatch and limits, aggregated to group ids.
    Returns:
      - groups: Index of group ids (unique)
      - P_group: snapshots x groups (MW)
      - Pmax_group: snapshots x groups (MW) using p_nom_opt * p_max_pu
      - group_region: Series(group -> region)
      - group_carrier: Series(group -> carrier) (first carrier encountered for that group)
    """
    gens = _select_offshore_generators(n, offshore_like)
    if len(gens) == 0:
        raise ValueError(f"No generators matched offshore_like={offshore_like!r} in this network.")

    group = _group_id_from_name(gens, group_regex)

    # Dispatch
    P = n.generators_t.p.reindex(columns=gens)

    # Limits
    p_nom_opt = n.generators.loc[gens, "p_nom_opt"].astype(float)
    p_max_pu = n.generators_t.p_max_pu.reindex(columns=gens).astype(float)
    Pmax = p_max_pu.mul(p_nom_opt, axis=1)

    # Aggregate to groups (sum)
    P_group = _aggregate_ts_by_group(P, group)
    Pmax_group = _aggregate_ts_by_group(Pmax, group)

    # Metadata per group
    bus_region = _infer_bus_region(n)
    gen_region = n.generators.loc[gens, "bus"].map(bus_region).fillna("UNKNOWN").astype(str)
    gen_carrier = n.generators.loc[gens, "carrier"].astype(str)

    # For region/carrier, collapse from per-generator to per-group:
    # - region: take the mode (most common) within group
    # - carrier: take the mode (most common) within group
    meta = pd.DataFrame({"group": group, "region": gen_region.to_numpy(), "carrier": gen_carrier.to_numpy()})

    def _mode(s: pd.Series) -> str:
        vc = s.value_counts(dropna=True)
        return str(vc.index[0]) if len(vc) else "UNKNOWN"

    group_region = meta.groupby("group")["region"].apply(_mode)
    group_carrier = meta.groupby("group")["carrier"].apply(_mode)

    return {
        "groups": P_group.columns,
        "P_group": P_group,
        "Pmax_group": Pmax_group,
        "group_region": group_region,
        "group_carrier": group_carrier,
    }


def violations_ref_under_alt_offshore_grouped(
    ref: dict[str, object],
    alt: dict[str, object],
) -> dict[str, object]:
    """
    Compare REF grouped dispatch against ALT grouped Pmax.
    """
    P_ref = ref["P_group"]
    Pmax_alt = alt["Pmax_group"]

    # Align group columns (intersection)
    common = P_ref.columns.intersection(Pmax_alt.columns)
    if len(common) == 0:
        raise ValueError(
            "No common offshore groups between REF and ALT after grouping.\n"
            "Adjust --group-regex so both runs map to a shared base group id."
        )

    P_ref = P_ref[common]
    Pmax_alt = Pmax_alt[common]

    viol = P_ref > (Pmax_alt + 1e-6)

    viol_rate_overall = float(viol.to_numpy().mean())
    viol_hours_any = float(viol.any(axis=1).mean())

    group_region_alt = alt["group_region"].reindex(common).fillna("UNKNOWN").astype(str)

    # share of violating groups per region per snapshot
    viol_by_region_time = {}
    for region, cols in group_region_alt.groupby(group_region_alt).groups.items():
        viol_by_region_time[region] = viol[cols].mean(axis=1)
    viol_by_region_time = pd.DataFrame(viol_by_region_time).sort_index(axis=1)

    viol_by_region_summary = pd.DataFrame({
        "mean_violation_share": viol_by_region_time.mean(axis=0),
        "max_violation_share": viol_by_region_time.max(axis=0),
    }).sort_values("mean_violation_share", ascending=False)

    return {
        "common_groups": common,
        "viol_bool": viol,
        "viol_rate_overall": viol_rate_overall,
        "viol_hours_any": viol_hours_any,
        "viol_by_region_time": viol_by_region_time,
        "viol_by_region_summary": viol_by_region_summary,
    }


def redispatch_offshore_grouped(
    ref: dict[str, object],
    alt: dict[str, object],
    dt_hours: pd.Series,
) -> dict[str, object]:
    """
    Grouped redispatch for offshore:
      dP = P_alt_group - P_ref_group
    """
    P_ref = ref["P_group"]
    P_alt = alt["P_group"]

    common = P_ref.columns.intersection(P_alt.columns)
    if len(common) == 0:
        raise ValueError(
            "No common offshore groups between REF and ALT after grouping.\n"
            "Adjust --group-regex so both runs map to a shared base group id."
        )

    dP = (P_alt[common] - P_ref[common])

    # energies (MWh)
    abs_MWh_by_group = dP.abs().mul(dt_hours, axis=0).sum(axis=0)
    net_MWh_by_group = dP.mul(dt_hours, axis=0).sum(axis=0)

    # aggregate by region / carrier using ALT metadata (post-change system)
    group_region = alt["group_region"].reindex(common).fillna("UNKNOWN").astype(str)
    group_carrier = alt["group_carrier"].reindex(common).fillna("UNKNOWN").astype(str)

    abs_by_region = abs_MWh_by_group.groupby(group_region).sum().sort_values(ascending=False)
    net_by_region = net_MWh_by_group.groupby(group_region).sum().sort_values(ascending=False)

    abs_by_carrier = abs_MWh_by_group.groupby(group_carrier).sum().sort_values(ascending=False)
    net_by_carrier = net_MWh_by_group.groupby(group_carrier).sum().sort_values(ascending=False)

    return {
        "dP": dP,
        "abs_redispatch_MWh_by_region": abs_by_region,
        "net_redispatch_MWh_by_region": net_by_region,
        "abs_redispatch_MWh_by_carrier": abs_by_carrier,
        "net_redispatch_MWh_by_carrier": net_by_carrier,
    }


# ---------------------------
# CLI
# ---------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Offshore-only, group-aware ex-post dispatch comparison for PyPSA networks.")
    p.add_argument("--ref", required=True, type=str, help="Path to reference solved network (NetCDF).")
    p.add_argument("--alt", required=True, type=str, help="Path to alternative solved network (NetCDF).")
    p.add_argument("--out", required=True, type=str, help="Output directory.")

    p.add_argument(
        "--offshore-like",
        default="offwind",
        type=str,
        help="Substring to select offshore generators (carrier or name). Default: offwind",
    )
    p.add_argument(
        "--group-regex",
        default=r" offwind-\w+$",
        type=str,
        help=(
            "Regex to STRIP from generator names to form a common group id across runs.\n"
            "Default: r\" offwind-\\\\w+$\" (strips ' offwind-XYZ' suffix)."
        ),
    )
    p.add_argument(
        "--time-agg",
        default="D",
        type=str,
        help="Temporal aggregation for violation heatmap (H/D/W/M/7D/...). Default: D",
    )
    p.add_argument(
        "--top-regions",
        default=25,
        type=int,
        help="Max number of regions to show on heatmap (highest mean violation share). Default: 25",
    )

    args = p.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_ref = pypsa.Network(args.ref)
    n_alt = pypsa.Network(args.alt)
    _assert_snapshots_match(n_ref, n_alt)

    # Extract grouped offshore data
    ref = compute_offshore_grouped(n_ref, offshore_like=args.offshore_like, group_regex=args.group_regex)
    alt = compute_offshore_grouped(n_alt, offshore_like=args.offshore_like, group_regex=args.group_regex)

    # Violations: REF dispatch under ALT limits
    viol = violations_ref_under_alt_offshore_grouped(ref, alt)

    # Write violation summaries
    summary = pd.DataFrame([{
        "ref_file": Path(args.ref).name,
        "alt_file": Path(args.alt).name,
        "offshore_like": args.offshore_like,
        "group_regex": args.group_regex,
        "n_common_groups": int(len(viol["common_groups"])),
        "viol_rate_overall": viol["viol_rate_overall"],
        "viol_hours_any": viol["viol_hours_any"],
    }])
    summary.to_csv(out_dir / "violation_summary_offshore_grouped.csv", index=False)
    viol["viol_by_region_summary"].to_csv(out_dir / "violations_by_region_summary_offshore_grouped.csv")

    # Heatmap: time x region (top regions)
    vbr = viol["viol_by_region_time"].copy()
    top_regions = viol["viol_by_region_summary"].head(args.top_regions).index
    vbr = vbr.reindex(columns=top_regions)

    vbr_agg = _time_aggregate_mean(vbr, args.time_agg)
    _plot_heatmap(
        vbr_agg,
        out_dir / f"heatmap_violation_share_by_region_offshore_grouped_{args.time_agg}.png",
        title=f"OFFSHORE grouped: REF dispatch violations under ALT availability\n(time agg: {args.time_agg})",
        vmin=0.0,
        vmax=1.0,
    )

    # Redispatch
    dt = _get_dt_hours(n_alt)
    red = redispatch_offshore_grouped(ref, alt, dt_hours=dt)

    red["abs_redispatch_MWh_by_region"].to_csv(out_dir / "abs_redispatch_MWh_by_region_offshore_grouped.csv")
    red["net_redispatch_MWh_by_region"].to_csv(out_dir / "net_redispatch_MWh_by_region_offshore_grouped.csv")
    red["abs_redispatch_MWh_by_carrier"].to_csv(out_dir / "abs_redispatch_MWh_by_carrier_offshore_grouped.csv")
    red["net_redispatch_MWh_by_carrier"].to_csv(out_dir / "net_redispatch_MWh_by_carrier_offshore_grouped.csv")

    # Quick plot: top absolute redispatch regions
    top_abs_regions = red["abs_redispatch_MWh_by_region"].head(20)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.bar(top_abs_regions.index.astype(str), top_abs_regions.values)
    ax.set_title("OFFSHORE grouped: Top 20 regions by absolute redispatch energy (MWh)")
    ax.set_ylabel("MWh")
    ax.set_xlabel("Region")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "top20_regions_abs_redispatch_MWh_offshore_grouped.png", dpi=200)
    plt.close(fig)

    print("Done.")
    print(f"Outputs: {out_dir.resolve()}")
    print(f"Common offshore groups: {len(viol['common_groups'])}")
    print(f"Overall violation rate (share of time×groups violating): {viol['viol_rate_overall']:.4f}")
    print(f"Share of hours with any violation: {viol['viol_hours_any']:.4f}")
    print("If common offshore groups is 0, tweak --group-regex to match your naming.")


if __name__ == "__main__":
    main()
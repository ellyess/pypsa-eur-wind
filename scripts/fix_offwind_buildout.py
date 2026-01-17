# scripts/fix_offwind_buildout.py
from __future__ import annotations

from pathlib import Path
import pypsa
import re
import pandas as pd


def _get_current_run(snakemake) -> str:
    return str(snakemake.wildcards.run)


def _ref_run_from_current(current_run: str) -> str:
    # 'glaum-s10000-biasFalse' -> 'base-s10000-biasFalse'
    parts = current_run.split("-", 1)
    return "base" if len(parts) == 1 else "base-" + parts[1]


def _get_clusters(snakemake) -> int:
    return int(snakemake.wildcards.clusters)


def _get_year(snakemake) -> int:
    # Prefer planning_horizons wildcard (you have it: planning_horizons=2030)
    if hasattr(snakemake.wildcards, "planning_horizons") and snakemake.wildcards.planning_horizons:
        return int(snakemake.wildcards.planning_horizons)

    # Fallback to costs year
    try:
        return int(snakemake.params.costs["year"])
    except Exception:
        return int(snakemake.config["costs"]["year"])


def _reference_postnetwork_file(snakemake, ref_run: str, clusters: int, year: int) -> Path:
    root = snakemake.config.get("fixed_offwind", {}).get("results_root")
    if not root:
        raise ValueError("Set fixed_offwind.results_root in config to point at results/...")

    return Path(root) / ref_run / "postnetworks" / f"base_s_{clusters}_lvopt___{year}.nc"


def fix_offwind_buildout(n: pypsa.Network, snapshots, snakemake):
    carrier = snakemake.config.get("fixed_offwind", {}).get("carrier", "offwind-combined")

    current_run = _get_current_run(snakemake)
    ref_run = _ref_run_from_current(current_run)

    clusters = _get_clusters(snakemake)
    year = _get_year(snakemake)

    ref_file = _reference_postnetwork_file(snakemake, ref_run, clusters, year)

    if not ref_file.is_file():
        raise FileNotFoundError(
            f"Reference network not found:\n  {ref_file}\n"
            f"Current run: {current_run}\nReference run: {ref_run}\n"
            "Make sure the base run has been solved and written to postnetworks."
        )

    ref = pypsa.Network(ref_file)

    mask = n.generators.carrier == carrier
    if not mask.any():
        return

    ref_mask = ref.generators.carrier == carrier
    if not ref_mask.any():
        raise ValueError(f"Reference has no generators with carrier={carrier!r}: {ref_file}")

    # Identify offshore gens in current + reference
    cur = n.generators.loc[mask].copy()
    refg = ref.generators.loc[ref_mask].copy()

    # Build "parent key" by stripping wake-split suffix " w<number>"
    # Example: "BE0 0_00000 offwind-combined w1" -> "BE0 0_00000 offwind-combined"
    def parent_key(index: pd.Index) -> pd.Index:
        return index.to_series().str.replace(r"\s+w\d+$", "", regex=True).rename("parent").values

    cur_parent = pd.Index(parent_key(cur.index), name="parent")
    ref_parent = pd.Index(parent_key(refg.index), name="parent")

    # Reference capacities by parent
    # If ref is unsplit, ref_parent == ref index; if ref is split, we aggregate to parent sum.
    ref_pnom_by_parent = refg["p_nom"].groupby(ref_parent).sum()

    # Allocate reference capacity to current generators:
    # - If current is unsplit: group size 1, gets full ref_p_nom.
    # - If current is split: distribute within each parent group by p_nom_max share (fallback equal).
    # weights by p_nom_max share within each parent
    cur_pnom_max = cur["p_nom_max"].copy()
    den = cur_pnom_max.groupby(cur_parent).transform("sum")

    weights = cur_pnom_max.div(den)

    # fallback to equal weights where den == 0 or NaN (or p_nom_max missing)
    # build equal-weight Series aligned to cur.index
    group_sizes = pd.Series(cur_parent, index=cur.index).groupby(cur_parent).transform("size")
    equal = 1.0 / group_sizes

    weights = weights.where(den > 0, equal).fillna(equal)

    # Map reference p_nom to each current row via parent
    mapped_ref = pd.Series(cur_parent, index=cur.index).map(ref_pnom_by_parent)

    missing_parents = mapped_ref[mapped_ref.isna()].index
    if len(missing_parents) > 0:
        ex = ", ".join(list(missing_parents[:5]))
        raise ValueError(
            "Some offshore generator parents not found in reference. "
            "This means clustering/regions differ between current and base reference. "
            f"First few missing current generators: {ex}"
        )

    new_pnom = (mapped_ref.values * weights.values)

    n.generators.loc[cur.index, "p_nom"] = new_pnom
    n.generators.loc[cur.index, "p_nom_extendable"] = False
    n.generators.loc[cur.index, "p_nom_min"] = new_pnom
    n.generators.loc[cur.index, "p_nom_max"] = new_pnom
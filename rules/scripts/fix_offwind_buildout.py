# scripts/fix_offwind_buildout.py
from __future__ import annotations

from pathlib import Path
import pypsa


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

    idx = n.generators.index[mask].intersection(ref.generators.index[ref_mask])

    missing = n.generators.index[mask].difference(ref.generators.index[ref_mask])
    if len(missing) > 0:
        ex = ", ".join(list(missing[:5]))
        raise ValueError(
            "Generator indices do not match reference (different clustering/busmap/regions?). "
            f"First missing in reference: {ex}"
        )

    # Copy + hard lock
    n.generators.loc[idx, "p_nom"] = ref.generators.loc[idx, "p_nom"]
    n.generators.loc[idx, "p_nom_extendable"] = False
    n.generators.loc[idx, "p_nom_min"] = n.generators.loc[idx, "p_nom"]
    n.generators.loc[idx, "p_nom_max"] = n.generators.loc[idx, "p_nom"]
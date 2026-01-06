# scripts/wake_helpers.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd

from scipy.spatial import Voronoi
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.prepared import prep
from sklearn.cluster import KMeans


# -----------------------------------------------------------------------------
# Types
# -----------------------------------------------------------------------------

PathLike = Union[str, Path]
Geometry = Union[Polygon, MultiPolygon]
ArrayLike2D = Union[np.ndarray, Sequence[Sequence[float]]]
CRSLike = Union[int, str, dict]


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def get_offshore_mods(config: dict) -> dict:
    """Return the `offshore_mods` sub-dictionary with a safe default."""
    return config.get("offshore_mods", {})


def _wind_threshold_key(technology: str) -> Optional[str]:
    """Return threshold key for wind technologies, otherwise None."""
    if technology.startswith("onwind"):
        return "onshore_threshold"
    if technology.startswith("offwind"):
        return "offshore_threshold"
    return None


def get_threshold(mods: dict, technology: str) -> int:
    """Return the configured area threshold (km²) for wind technologies."""
    key = _wind_threshold_key(technology)
    if key is None:
        raise ValueError(
            "get_threshold() is only defined for wind technologies. "
            f"Got technology={technology!r}."
        )
    val = mods.get(key)
    if val is None:
        raise ValueError(f"Missing config offshore_mods.{key}")
    return int(val)


# -----------------------------------------------------------------------------
# Paths & caching
# -----------------------------------------------------------------------------

def get_wake_dir(mods: dict) -> Path:
    """Return the wake cache directory (`wake_extra/<shared_files>`) and ensure it exists."""
    shared = str(mods.get("shared_files", "default"))
    d = Path("wake_extra") / shared
    d.mkdir(parents=True, exist_ok=True)
    return d



def regions_file(wake_dir: Path, technology: str, threshold: int) -> Optional[Path]:
    """Return the split-regions file path for wind technologies."""
    if technology.startswith("offwind"):
        return wake_dir / f"regions_offshore_s{threshold}.geojson"
    if technology.startswith("onwind"):
        return wake_dir / f"regions_onshore_s{threshold}.geojson"
    return None


def availability_cache_path(wake_dir: Path, clusters, technology: str, threshold: int) -> Path:
    """Return cache path for availability matrices."""
    return wake_dir / f"availability_matrix_{clusters}_{technology}_{threshold}.nc"


def profile_cache_path(
    wake_dir: Path,
    clusters,
    technology: str,
    threshold: int,
    bias: Optional[str] = None,
) -> Path:
    """Return cache path for renewable profiles."""
    suffix = f"_bias{bias}" if bias is not None else ""
    return wake_dir / f"profile_{clusters}_{technology}_{threshold}{suffix}.nc"


# -----------------------------------------------------------------------------
# Region loading
# -----------------------------------------------------------------------------

def load_regions(
    technology: str,
    threshold: int,
    wake_dir: Path,
    fallback_path: PathLike,
) -> gpd.GeoDataFrame:
    """Load split regions for wind technologies; otherwise load a fallback file."""
    p = regions_file(wake_dir, technology, threshold)
    if p is None:
        return gpd.read_file(fallback_path)

    if not p.is_file():
        raise FileNotFoundError(
            f"Expected split regions file not found: {p}. "
            "Make sure the splitting rule has run."
        )

    return gpd.read_file(p)


# -----------------------------------------------------------------------------
# Wake modelling utilities
# -----------------------------------------------------------------------------

def _offwind_region_mapping(n) -> pd.Series:
    """Map offwind generator names -> region name (without carrier suffix)."""
    gen_idx = n.generators.filter(like="offwind", axis=0).index
    return gen_idx.to_series().str.replace(r" offwind-\w+", "", regex=True)


def _split_profile_by_capacity(
    n,
    df: pd.DataFrame,
    num_splits: int,
    label_prefix: str = " w",
) -> Tuple[List[pd.Series], List[pd.Series], List[str], List[str]]:
    """Split each generator into up to `num_splits` segments with modified p_max_pu."""
    generators_to_add: List[pd.Series] = []
    pmax_to_add: List[pd.Series] = []
    labels: List[str] = []
    to_drop: List[str] = []

    for gen_name in df.index:
        base = df.loc[gen_name]
        base_pmax = n.generators_t.p_max_pu.loc[:, gen_name]

        to_drop.append(gen_name)

        remaining_p_nom_max = float(base.p_nom_max)
        remaining_p_nom = float(base.p_nom)
        remaining_p_nom_min = float(base.p_nom_min)

        for i in range(1, num_splits + 1):
            if remaining_p_nom_max <= 0:
                break

            seg_max = float(base.get(f"max_capacity_{i}", 0.0))
            if not np.isfinite(seg_max) or seg_max <= 0:
                seg_p_nom_max = remaining_p_nom_max
            else:
                seg_p_nom_max = min(seg_max, remaining_p_nom_max)

            seg = base.copy()
            seg["p_nom_max"] = seg_p_nom_max
            remaining_p_nom_max -= seg_p_nom_max

            seg_p_nom = min(seg_p_nom_max, remaining_p_nom) if remaining_p_nom > 0 else 0.0
            seg_p_nom_min = min(seg_p_nom_max, remaining_p_nom_min) if remaining_p_nom_min > 0 else 0.0

            seg["p_nom"] = seg_p_nom
            seg["p_nom_min"] = seg_p_nom_min
            remaining_p_nom -= seg_p_nom
            remaining_p_nom_min -= seg_p_nom_min

            wake_factor = float(base.get(f"factor_wake_{i}", 0.0))
            seg_pmax = base_pmax * (1.0 - wake_factor)

            labels.append(f"{gen_name}{label_prefix}{i}")
            generators_to_add.append(seg)
            pmax_to_add.append(seg_pmax)

    return generators_to_add, pmax_to_add, labels, to_drop


@dataclass(frozen=True)
class WakeSplitSpec:
    """Specification of segment capacities and wake factors."""
    factors: List[float]          # length = n_segments
    max_caps: List[float]         # length = n_segments (last can be np.inf)


def _new_more_spec() -> Tuple[WakeSplitSpec, List[float]]:
    """Return (spec, breakpoints) for the 'new_more' wake method."""
    def y(x: float) -> float:
        alpha = 7.3
        beta = 0.05
        gamma = -0.7
        delta = -14.6
        return alpha * np.exp(-x / beta) + gamma * x + delta

    def piecewise(x0: float, x1: float) -> float:
        return (y(x1) * x1 - y(x0) * x0) / (x1 - x0)

    x = [0, 0.025, 0.05, 0.25, 1, 2.5, 4]
    factors = [-(piecewise(x[i], x[i + 1])) / 100.0 for i in range(len(x) - 1)]
    return WakeSplitSpec(factors=factors, max_caps=[]), x


def _glaum_spec() -> WakeSplitSpec:
    """Return segment definition for the 'glaum' wake method (after global derate)."""
    f2 = 0.1279732
    f3_extra = 0.13902848
    f3 = 1.0 - ((1.0 - f2) * (1.0 - f3_extra))
    return WakeSplitSpec(
        factors=[0.0, f2, f3],
        max_caps=[2e3, 10e3, np.inf],
    )


def _assign_segment_count_from_cumcaps(p_nom_max: np.ndarray, cumcaps: np.ndarray) -> np.ndarray:
    """Return number of segments needed for each generator based on cumulative caps."""
    return 1 + (cumcaps <= p_nom_max[:, None]).sum(axis=1)


def add_wake_generators(n, snakemake, method: str) -> None:
    """Apply wake effects to offshore wind by splitting generators into segments."""
    if method not in {"new_more", "glaum"}:
        raise ValueError(f"Unknown wake method: {method!r}")

    mapping = _offwind_region_mapping(n)
    if mapping.empty:
        return

    wake_generators = n.generators.loc[mapping.index].copy()

    mods = get_offshore_mods(snakemake.config)
    wdir = get_wake_dir(mods)

    if method == "new_more":
        tech_for_threshold = "offwind"  # ✅ canonical offshore category
        threshold = get_threshold(mods, tech_for_threshold)

        offshore_reg = load_regions(
            technology=tech_for_threshold,
            threshold=threshold,
            wake_dir=wdir,
            fallback_path=snakemake.input.regions_offshore
            if "regions_offshore" in snakemake.input
            else snakemake.input.regions,
        )[["name", "area"]].set_index("name")

        wake_generators = wake_generators.assign(region=mapping.values)
        wake_generators = wake_generators.join(offshore_reg, on="region")

        if wake_generators["area"].isna().any():
            missing = wake_generators.loc[wake_generators["area"].isna(), "region"].unique()
            raise ValueError(
                "Missing offshore region areas for some generators. "
                f"First missing regions: {missing[:10]!r}"
            )

        spec, x = _new_more_spec()
        factors = spec.factors

        dx = np.diff(np.asarray(x, dtype=float))  # length 6
        area = wake_generators["area"].to_numpy(dtype=float)
        max_caps = np.column_stack([area * dx[i] for i in range(len(dx))])
        max_caps[:, -1] = np.inf

        for i, f in enumerate(factors, start=1):
            wake_generators[f"factor_wake_{i}"] = f
        for i in range(1, 6):
            wake_generators[f"max_capacity_{i}"] = max_caps[:, i - 1]
        wake_generators["max_capacity_6"] = np.inf

        cumcaps = np.cumsum(max_caps[:, :5], axis=1)
        pmax = wake_generators["p_nom_max"].to_numpy(dtype=float)
        seg_count = _assign_segment_count_from_cumcaps(pmax, cumcaps)

        split_generators: Dict[int, pd.DataFrame] = {
            k: wake_generators.loc[seg_count == k] for k in range(1, 7)
        }

    else:  # glaum
        n.generators_t.p_max_pu.loc[:, mapping.index] *= 0.906

        big = wake_generators[wake_generators.p_nom_max > 2e3].copy()
        if big.empty:
            return

        spec = _glaum_spec()
        for i, f in enumerate(spec.factors, start=1):
            big[f"factor_wake_{i}"] = f
        for i, cap in enumerate(spec.max_caps, start=1):
            big[f"max_capacity_{i}"] = cap

        pmax = big["p_nom_max"].to_numpy(dtype=float)
        seg_count = np.where(pmax <= (spec.max_caps[0] + spec.max_caps[1]), 2, 3)
        split_generators = {2: big.loc[seg_count == 2], 3: big.loc[seg_count == 3]}

    if not any(len(df) for df in split_generators.values()):
        return

    gens_to_add: List[pd.Series] = []
    pmax_to_add: List[pd.Series] = []
    labels: List[str] = []
    to_drop: List[str] = []

    for num_splits, df in split_generators.items():
        if df.empty:
            continue
        g_add, t_add, lab, drop = _split_profile_by_capacity(n, df, num_splits=num_splits)
        gens_to_add.extend(g_add)
        pmax_to_add.extend(t_add)
        labels.extend(lab)
        to_drop.extend(drop)

    if not labels:
        return

    n.generators.drop(index=to_drop, inplace=True)
    n.generators_t.p_max_pu.drop(columns=to_drop, inplace=True)

    add_df = pd.concat(gens_to_add, axis=1, keys=labels).T.infer_objects()
    add_t = pd.concat(pmax_to_add, axis=1, keys=labels)

    n.generators = pd.concat([n.generators, add_df], axis=0)
    n.generators_t.p_max_pu = pd.concat([n.generators_t.p_max_pu, add_t], axis=1)
    n.generators_t.p_max_pu.columns.names = ["Generator"]


def drop_non_dominant_offwind_generators(n) -> None:
    """Drop non-dominant offshore wind generators per region.

    Keeps exactly one offshore wind generator per region: the one with the largest
    `p_nom_max`. Ties are broken by larger `p_nom`, then by name.
    """
    mapping = _offwind_region_mapping(n)
    if mapping.empty:
        return

    gens = n.generators.loc[mapping.index].copy()
    gens["region"] = mapping.values

    gens["p_nom_max"] = pd.to_numeric(gens["p_nom_max"], errors="coerce").fillna(0.0)
    gens["p_nom"] = pd.to_numeric(gens.get("p_nom", 0.0), errors="coerce").fillna(0.0)

    gens = gens.assign(_name=gens.index.astype(str))
    gens_sorted = gens.sort_values(
        by=["region", "p_nom_max", "p_nom", "_name"],
        ascending=[True, True, True, True],
    )
    keep_idx = gens_sorted.groupby("region", sort=False).tail(1).index
    drop_idx = mapping.index.difference(keep_idx)

    if drop_idx.empty:
        return

    n.generators.drop(index=drop_idx, inplace=True)

    if hasattr(n, "generators_t") and hasattr(n.generators_t, "p_max_pu"):
        cols_to_drop = [c for c in drop_idx if c in n.generators_t.p_max_pu.columns]
        if cols_to_drop:
            n.generators_t.p_max_pu.drop(columns=cols_to_drop, inplace=True)


# -----------------------------------------------------------------------------
# Region splitting (Voronoi meshing)
# -----------------------------------------------------------------------------

def _as_2d_points(points: ArrayLike2D) -> np.ndarray:
    """Validate and coerce points to an (N, 2) float ndarray."""
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("`points` must have shape (N, 2).")
    if arr.shape[0] == 0:
        raise ValueError("`points` must contain at least one point.")
    return arr


def cluster_points(points: ArrayLike2D, n_clusters: int, random_state: int = 0) -> np.ndarray:
    """Cluster 2D points using k-means and return cluster centers."""
    pts = _as_2d_points(points)
    if n_clusters < 1:
        raise ValueError("`n_clusters` must be >= 1.")
    if n_clusters > len(pts):
        raise ValueError("`n_clusters` cannot exceed number of points.")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(pts)
    return np.asarray(kmeans.cluster_centers_, dtype=float)


def fill_shape_with_points(
    shape: Geometry,
    min_points: int,
    initial_num: int = 50,
    grow_step: int = 10,
    shrink_frac: float = 0.01,
    max_iter: int = 60,
) -> np.ndarray:
    """Generate interior points by grid-sampling a polygon until enough points exist."""
    if min_points < 1:
        raise ValueError("`min_points` must be >= 1.")
    if initial_num < 2:
        raise ValueError("`initial_num` must be >= 2.")

    prepared = prep(shape)
    x_min, y_min, x_max, y_max = shape.bounds

    collected: List[Tuple[float, float]] = []
    num = int(initial_num)

    for _ in range(max_iter):
        xs = np.linspace(x_min, x_max, num=num)
        ys = np.linspace(y_min, y_max, num=num)
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        coords = np.column_stack([xx.ravel(), yy.ravel()])

        for x, y in coords:
            if prepared.contains(Point(float(x), float(y))):
                collected.append((float(x), float(y)))

        if len(collected) >= min_points:
            uniq = np.unique(np.asarray(collected, dtype=float), axis=0)
            if len(uniq) >= min_points:
                return uniq

        num += int(grow_step)
        dx = (x_max - x_min) * shrink_frac
        dy = (y_max - y_min) * shrink_frac
        x_min += dx
        x_max -= dx
        y_min += dy
        y_max -= dy

        if x_max <= x_min or y_max <= y_min:
            break

    raise RuntimeError(
        f"Could not generate at least {min_points} interior points within max_iter={max_iter}."
    )


def voronoi_partition(points: ArrayLike2D, outline: Geometry) -> List[Polygon]:
    """Compute a Voronoi partition of `points` clipped to `outline`."""
    pts = _as_2d_points(points)

    if len(pts) == 1:
        if isinstance(outline, Polygon):
            return [outline]
        return list(outline.geoms)

    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    xspan = xmax - xmin
    yspan = ymax - ymin

    framing = np.array(
        [
            [xmin - 3.0 * xspan, ymin - 3.0 * yspan],
            [xmin - 3.0 * xspan, ymax + 3.0 * yspan],
            [xmax + 3.0 * xspan, ymin - 3.0 * yspan],
            [xmax + 3.0 * xspan, ymax + 3.0 * yspan],
        ],
        dtype=float,
    )
    vor = Voronoi(np.vstack([pts, framing]))

    cells: List[Polygon] = []
    for i in range(len(pts)):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]

        if not region or -1 in region:
            poly = outline
        else:
            poly = Polygon(vor.vertices[region])

        if not poly.is_valid:
            poly = poly.buffer(0)

        clipped = poly.intersection(outline)
        if clipped.is_empty:
            continue
        if isinstance(clipped, Polygon):
            cells.append(clipped)
        elif isinstance(clipped, MultiPolygon):
            cells.extend([g for g in clipped.geoms if isinstance(g, Polygon)])

    return cells


def mesh_region(
    geometry: Geometry,
    area_km2: float,
    threshold_km2: float,
    random_state: int = 0,
    min_points_factor: int = 5,
) -> List[Polygon]:
    """Split a region into Voronoi cells if above an area threshold."""
    if area_km2 <= threshold_km2:
        if isinstance(geometry, Polygon):
            return [geometry]
        return list(geometry.geoms)

    n_parts = int(np.ceil(area_km2 / threshold_km2))
    inner_pts = fill_shape_with_points(geometry, min_points=max(n_parts * min_points_factor, n_parts))
    centers = cluster_points(inner_pts, n_clusters=n_parts, random_state=random_state)
    return voronoi_partition(centers, geometry)


def split_regions(
    regions: gpd.GeoDataFrame,
    threshold_km2: float,
    bus_main_col: str = "bus_main",
    out_crs: CRSLike = 4326,
    area_crs: CRSLike = "EPSG:6933",
    random_state: int = 0,
) -> gpd.GeoDataFrame:
    """Split all regions so each part is at most `threshold_km2` in area."""
    if threshold_km2 <= 0:
        raise ValueError("`threshold_km2` must be positive.")
    if regions.geometry is None:
        raise ValueError("`regions` must have a geometry column.")
    if bus_main_col not in regions.columns:
        raise ValueError(f"`regions` must contain '{bus_main_col}' column.")

    reg = regions.copy().to_crs(out_crs)
    reg["_area_km2"] = reg.to_crs(area_crs).area / 1e6

    parts: List[gpd.GeoDataFrame] = []
    for _, row in reg.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue

        bus_main = row[bus_main_col]
        area_km2 = float(row["_area_km2"])

        sub_geoms = mesh_region(geom, area_km2, threshold_km2, random_state=random_state)
        parts.append(
            gpd.GeoDataFrame({bus_main_col: [bus_main] * len(sub_geoms), "geometry": sub_geoms}, crs=out_crs)
        )

    if not parts:
        out = gpd.GeoDataFrame(
            {bus_main_col: pd.Series(dtype=object), "geometry": gpd.GeoSeries([], crs=out_crs)},
            crs=out_crs,
        )
    else:
        out = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=out_crs)

    out["region"] = out.groupby(bus_main_col).cumcount().astype(str).str.zfill(5)
    out["name"] = out[bus_main_col].astype(str) + "_" + out["region"]
    out["country"] = out[bus_main_col].astype(str).str[:2]
    out["area"] = out.to_crs(area_crs).area / 1e6

    return out[["name", bus_main_col, "country", "geometry", "area"]].rename(columns={bus_main_col: "bus_main"})

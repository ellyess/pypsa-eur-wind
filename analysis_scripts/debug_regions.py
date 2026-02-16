#!/usr/bin/env python3
"""Debug script to check region matching in capacity density calculation."""

from pathlib import Path
import pandas as pd
import geopandas as gpd

# Check one of the empty CSVs
csv_path = Path("plots/wakes/capacity_density_by_region_s1000.csv")
df = pd.read_csv(csv_path)
print(f"CSV shape: {df.shape}")
print(f"CSV columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head(10))
print(f"\nAny non-null values? {df.notna().sum().sum()}")

# Check the region shapefile
region_file = Path("wake_extra/northsea/regions_offshore_s1000.geojson")
if region_file.exists():
    regions = gpd.read_file(region_file)
    print(f"\n\nRegion file columns: {regions.columns.tolist()}")
    print(f"Number of regions: {len(regions)}")
    print(f"\nFirst few region names:")
    for col in ['name', 'region', 'bus']:
        if col in regions.columns:
            print(f"  {col}: {regions[col].head().tolist()}")
else:
    print(f"\n\nRegion file not found: {region_file}")

# Try to load a network and check generator naming
import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    import pypsa
    net_path = Path("results/thesis-wake-2030-10-northsea-dominant-6h/base-s1000-biasFalse/postnetworks/base_s_10_elec_lvopt_2030.nc")
    if net_path.exists():
        print(f"\n\nLoading network: {net_path}")
        n = pypsa.Network(str(net_path))
        
        # Check offshore generators
        gens = n.generators
        offshore = gens[gens["carrier"].str.startswith("offwind")]
        print(f"Number of offshore generators: {len(offshore)}")
        print(f"\nFirst 5 generator indices:")
        for idx in offshore.index[:5]:
            print(f"  {idx}")
            # Try to parse region
            parts = str(idx).split()
            region = " ".join(parts[:2]) if len(parts) >= 2 else idx
            print(f"    -> parsed region: {region}")
    else:
        print(f"\n\nNetwork file not found: {net_path}")
except Exception as e:
    print(f"\n\nError loading network: {e}")

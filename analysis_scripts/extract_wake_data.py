#!/usr/bin/env python3
"""
extract_wake_data.py

Extract and aggregate data from PyPSA network results for wake analysis plotting.
Reads .nc files and/or csvs directories, then exports aggregated data to formats
expected by compare_wake_runs_new.py.

Usage:
    python extract_wake_data.py --results-dir results/thesis-wake-2030-10-northsea-dominant-6h \
                                 --out-dir data/wake_extracted \
                                 --scenarios base standard glaum new_more \
                                 --split 100000

This will:
1. Scan for scenario subdirectories matching pattern: {scenario}-s{split}-biasFalse/
2. Load networks from postnetworks/*.nc or read from csvs/
3. Extract wake losses, capacity densities, system metrics, etc.
4. Export to CSV/GeoJSON for plotting
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

try:
    import pypsa
except ImportError:
    pypsa = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None

# Suppress warnings
warnings.filterwarnings('ignore')

from network_utils import find_scenario_dirs


def load_network(scenario_dir: Path) -> pypsa.Network | None:
    """Load PyPSA network from postnetworks/*.nc file."""
    if pypsa is None:
        print("Warning: pypsa not available, will try to use CSVs only")
        return None
    
    postnet_dir = scenario_dir / "postnetworks"
    if not postnet_dir.exists():
        postnet_dir = scenario_dir / "networks"
    if not postnet_dir.exists():
        return None

    nc_files = list(postnet_dir.glob("*.nc"))
    if not nc_files:
        return None
    
    nc_file = nc_files[0]  # Take first .nc file
    print(f"  Loading network from {nc_file.name}")
    
    try:
        n = pypsa.Network(str(nc_file))
        return n
    except Exception as e:
        print(f"  Error loading network: {e}")
        return None


def read_pypsa_csv(filepath: Path, skiprows: int = 4) -> pd.DataFrame:
    """Read PyPSA-formatted CSV with metadata rows at top."""
    try:
        df = pd.read_csv(filepath, skiprows=skiprows)
        # First column might have no name after skip, use it as index
        if df.columns[0] == 'Unnamed: 0' or not df.columns[0]:
            df = pd.read_csv(filepath, skiprows=skiprows, index_col=0)
        return df
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")
        return pd.DataFrame()


def extract_wake_losses(scenario_dirs: dict[tuple[str, int], Path]) -> pd.DataFrame:
    """
    Extract wake loss multipliers for each scenario and split.
    Wake loss is calculated as the relative reduction in CF compared to the base scenario.
    Base scenario has wake_loss = 0 (it's the reference without wake model).
    Wake scenarios: wake_loss = 1 - (CF_wake / CF_base)
    
    Returns DataFrame with columns: scenario, split, wake_loss
    """
    if pypsa is None:
        print("Error: pypsa not available, cannot calculate from networks")
        return pd.DataFrame(columns=['scenario', 'split', 'wake_loss'])
    
    # First pass: collect all capacity factors from networks
    cf_data = []
    
    for (scenario, split), sdir in scenario_dirs.items():
        n = load_network(sdir)
        if n is None:
            continue
        
        # Get offshore wind generators
        offshore = n.generators[n.generators.carrier.str.contains('offwind', case=False, na=False)]
        
        if offshore.empty:
            continue
        
        # Calculate average capacity factor for each generator from time series
        for gen_name, gen in offshore.iterrows():
            try:
                # Get time-varying capacity factor (p_max_pu)
                if gen_name in n.generators_t.p_max_pu.columns:
                    cf = n.generators_t.p_max_pu[gen_name].mean()
                else:
                    # Fallback to static value
                    cf = gen.get('p_max_pu', 1.0)
                
                if pd.notna(cf) and 0 <= cf <= 1:
                    cf_data.append({
                        'scenario': scenario,
                        'split': split,
                        'bus': gen['bus'],
                        'technology': gen['carrier'],
                        'capacity_factor': cf
                    })
            except Exception as e:
                print(f"Warning: Error processing {gen_name}: {e}")
                continue
    
    if not cf_data:
        print("Warning: No capacity factor data found")
        return pd.DataFrame(columns=['scenario', 'split', 'wake_loss'])
    
    cf_df = pd.DataFrame(cf_data)
    
    # Second pass: calculate wake losses relative to base scenario
    wake_loss_rows = []
    
    for split in cf_df['split'].unique():
        split_data = cf_df[cf_df['split'] == split]
        
        # Get base scenario CFs for this split
        base_data = split_data[split_data['scenario'] == 'base']
        
        if base_data.empty:
            print(f"Warning: No base scenario found for split={split}, cannot calculate wake losses")
            continue
        
        # Group by bus to get average CF per bus for base
        base_cf_by_bus = base_data.groupby('bus')['capacity_factor'].mean()
        
        for scenario in split_data['scenario'].unique():
            scenario_data = split_data[split_data['scenario'] == scenario]
            
            if scenario == 'base':
                # Base scenario: wake_loss = 0 (it's the reference)
                for cf in scenario_data['capacity_factor']:
                    wake_loss_rows.append({
                        'scenario': scenario,
                        'split': split,
                        'wake_loss': 0.0
                    })
            else:
                # Wake scenarios: calculate relative to base
                for _, row in scenario_data.iterrows():
                    bus = row['bus']
                    cf_wake = row['capacity_factor']
                    
                    # Get corresponding base CF for this bus
                    if bus in base_cf_by_bus.index:
                        cf_base = base_cf_by_bus[bus]
                        
                        # Calculate wake loss as relative reduction
                        # wake_loss = 1 - (CF_wake / CF_base)
                        # If CF_wake < CF_base: positive wake loss
                        # If CF_wake >= CF_base: no wake loss (can happen due to optimization)
                        if cf_base > 0:
                            wake_loss = max(0.0, 1.0 - (cf_wake / cf_base))
                        else:
                            wake_loss = 0.0
                        
                        wake_loss_rows.append({
                            'scenario': scenario,
                            'split': split,
                            'wake_loss': wake_loss
                        })
    
    if not wake_loss_rows:
        print("Warning: No wake loss data calculated")
        return pd.DataFrame(columns=['scenario', 'split', 'wake_loss'])
    
    return pd.DataFrame(wake_loss_rows)


def extract_cf_metrics(scenario_dirs: dict[tuple[str, int], Path]) -> pd.DataFrame:
    """
    Extract capacity factor metrics for offshore wind: available CF, dispatch CF, and curtailment CF.
    
    Available CF: mean of p_max_pu time series (renewable availability after wake losses)
    Dispatch CF: mean of actual generation p / p_nom_opt
    Curtailment CF: Available CF - Dispatch CF
    
    Calculated at regional level from network time series data.
    
    Returns DataFrame with columns: scenario, split, region, tech, available_cf, dispatch_cf, curtailment_cf
    """
    if pypsa is None:
        print("Error: pypsa not available, cannot calculate from networks")
        return pd.DataFrame(columns=['scenario', 'split', 'region', 'tech', 'available_cf', 'dispatch_cf', 'curtailment_cf'])
    
    rows = []
    
    for (scenario, split), sdir in scenario_dirs.items():
        n = load_network(sdir)
        if n is None:
            continue
        
        # Get offshore wind generators
        offshore = n.generators[n.generators.carrier.str.contains('offwind', case=False, na=False)]
        
        if offshore.empty:
            continue
        
        # Calculate CFs for each generator from time series
        for gen_name, gen in offshore.iterrows():
            try:
                # Available CF: mean of p_max_pu time series
                if gen_name in n.generators_t.p_max_pu.columns:
                    available_cf = n.generators_t.p_max_pu[gen_name].mean()
                else:
                    available_cf = gen.get('p_max_pu', 1.0)
                
                # Dispatch CF: mean of actual generation / capacity
                capacity = gen['p_nom_opt']
                if capacity > 0 and gen_name in n.generators_t.p.columns:
                    actual_generation = n.generators_t.p[gen_name]
                    dispatch_cf = actual_generation.mean() / capacity
                else:
                    dispatch_cf = 0.0
                
                # Curtailment CF
                curtailment_cf = max(0, available_cf - dispatch_cf)
                
                rows.append({
                    'scenario': scenario,
                    'split': split,
                    'region': gen['bus'],
                    'tech': gen['carrier'],
                    'available_cf': available_cf,
                    'dispatch_cf': dispatch_cf,
                    'curtailment_cf': curtailment_cf
                })
            except Exception as e:
                print(f"Warning: Error processing {gen_name}: {e}")
                continue
    
    if not rows:
        print("Warning: No CF metrics data found")
        return pd.DataFrame(columns=['scenario', 'split', 'region', 'tech', 'available_cf', 'dispatch_cf', 'curtailment_cf'])
    
    return pd.DataFrame(rows)


def extract_wake_vs_density(scenario_dirs: dict[tuple[str, int], Path]) -> pd.DataFrame:
    """
    Extract wake losses vs installed capacity density.
    
    Returns DataFrame with columns: scenario, split, density_mw_per_km2, wake_loss
    """
    rows = []
    
    for (scenario, split), sdir in scenario_dirs.items():
        n = load_network(sdir)
        
        if n is not None:
            offshore = n.generators[n.generators.carrier.str.contains('offwind', case=False)]
            
            for idx, gen in offshore.iterrows():
                # Get capacity
                p_nom = gen.get('p_nom_opt', gen.get('p_nom', 0))
                
                # Get area (you may need to load this from regions)
                # Placeholder: assume area info is in bus or separate regions file
                area_km2 = 100  # Replace with actual area lookup
                
                density = p_nom / area_km2 if area_km2 > 0 else 0
                wake_loss = 1.0 - gen.get('p_nom_max_pu', 1.0)
                
                rows.append({
                    'scenario': scenario,
                    'split': split,
                    'density_mw_per_km2': density,
                    'wake_loss': wake_loss
                })
    
    if not rows:
        print("Warning: No wake vs density data found")
        return pd.DataFrame(columns=['scenario', 'split', 'density_mw_per_km2', 'wake_loss'])
    
    return pd.DataFrame(rows)


def extract_system_metrics(scenario_dirs: dict[tuple[str, int], Path]) -> pd.DataFrame:
    """
    Extract system-level metrics: cost, curtailment, emissions, etc.
    Calculated directly from network object.
    
    Returns DataFrame with columns: scenario, split, total_cost_eur, curtailment_twh, offshore_capacity_gw, ...
    """
    if pypsa is None:
        print("Error: pypsa not available, cannot calculate from networks")
        return pd.DataFrame()
    
    rows = []
    
    for (scenario, split), sdir in scenario_dirs.items():
        n = load_network(sdir)
        if n is None:
            continue
        
        metrics = {'scenario': scenario, 'split': split}
        
        try:
            # System costs
            if hasattr(n, 'objective'):
                metrics['total_cost_eur'] = float(n.objective)
            
            # Offshore wind capacity
            offshore = n.generators[n.generators.carrier.str.contains('offwind', case=False, na=False)]
            metrics['offshore_capacity_gw'] = offshore['p_nom_opt'].sum() / 1e3
            
            # Transmission capacity (AC lines + DC links)
            metrics['line_volume_ac'] = n.lines['s_nom_opt'].sum() / 1e3
            metrics['line_volume_dc'] = n.links['p_nom_opt'].sum() / 1e3
            metrics['transmission_capacity_gw'] = (n.lines['s_nom_opt'].sum() + n.links['p_nom_opt'].sum()) / 1e3
            
            # Calculate curtailment from time series
            curtailment_mwh = 0
            for gen_name in offshore.index:
                if gen_name in n.generators_t.p_max_pu.columns and gen_name in n.generators_t.p.columns:
                    p_max_pu = n.generators_t.p_max_pu[gen_name]
                    p_actual = n.generators_t.p[gen_name]
                    p_nom = offshore.loc[gen_name, 'p_nom_opt']
                    
                    # Curtailment = available - actual
                    curtailed = (p_max_pu * p_nom - p_actual).clip(lower=0).sum()
                    curtailment_mwh += curtailed
            
            metrics['curtailment_mwh'] = float(curtailment_mwh)
            metrics['curtailment_twh'] = float(curtailment_mwh) / 1e6
            
            # CO2 shadow prices
            if hasattr(n, 'global_constraints') and 'co2_limit' in n.global_constraints.index:
                if hasattr(n, 'global_constraints_t') and hasattr(n.global_constraints_t, 'mu'):
                    if 'co2_limit' in n.global_constraints_t.mu.columns:
                        metrics['co2_shadow'] = float(n.global_constraints_t.mu['co2_limit'].mean())
            
            # Total costs breakdown if available
            if hasattr(n, 'statistics'):
                try:
                    stats = n.statistics()
                    if 'Capital Expenditure' in stats.index:
                        metrics['capex'] = float(stats.loc['Capital Expenditure'].sum())
                    if 'Operational Expenditure' in stats.index:
                        metrics['opex'] = float(stats.loc['Operational Expenditure'].sum())
                except Exception as e:
                    print(f"  Warning: Could not extract statistics: {e}")
        
        except Exception as e:
            print(f"  Warning: Error extracting metrics for {scenario} split={split}: {e}")
        
        rows.append(metrics)
    
    return pd.DataFrame(rows)


def extract_resolution_metrics(results_base: Path, scenario: str, splits: list[int]) -> pd.DataFrame:
    """
    Extract metrics across different spatial resolutions (split sizes).
    Calculated directly from network object.
    
    Returns DataFrame with columns: scenario, split_km2, offshore_capacity_gw, total_cost_beur, n_offshore_buses
    """
    if pypsa is None:
        print("Error: pypsa not available, cannot calculate from networks")
        return pd.DataFrame()
    
    rows = []
    
    for split in splits:
        pattern = f"{scenario}-s{split}-biasFalse"
        sdir = results_base / pattern
        
        if not sdir.exists():
            continue
        
        metrics = {'scenario': scenario, 'split_km2': split}
        
        n = load_network(sdir)
        if n is not None:
            try:
                # Offshore capacity
                offshore = n.generators[n.generators.carrier.str.contains('offwind', case=False, na=False)]
                metrics['offshore_capacity_gw'] = offshore['p_nom_opt'].sum() / 1e3
                metrics['n_offshore_buses'] = len(offshore['bus'].unique())
                
                # System cost
                if hasattr(n, 'objective'):
                    metrics['total_cost_beur'] = float(n.objective) / 1e9
            except Exception as e:
                print(f"  Warning: Error extracting resolution metrics for split {split}: {e}")
        
        rows.append(metrics)
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract wake analysis data from PyPSA results")
    parser.add_argument("--results-dir", type=Path, required=True,
                       help="Base results directory (e.g., results/thesis-wake-2030-10-northsea-dominant-6h)")
    parser.add_argument("--out-dir", type=Path, required=True,
                       help="Output directory for extracted data")
    parser.add_argument("--scenarios", nargs="+", default=["base", "standard", "glaum", "new_more"],
                       help="Scenarios to extract")
    parser.add_argument("--splits", nargs="+", type=int, default=[100000],
                       help="Spatial split sizes to extract (default: [100000])")
    parser.add_argument("--resolution-analysis", action="store_true",
                       help="Also extract resolution metrics across all split sizes")
    
    args = parser.parse_args()
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExtracting wake data from: {args.results_dir}")
    print(f"Output directory: {args.out_dir}")
    print(f"Scenarios: {args.scenarios}")
    print(f"Split sizes: {args.splits}\n")
    
    # Find scenario directories
    scenario_dirs = find_scenario_dirs(args.results_dir, args.scenarios, args.splits)
    
    if not scenario_dirs:
        print("Error: No scenario directories found!")
        return 1
    
    # Extract wake losses
    print("\nExtracting wake losses...")
    wake_losses = extract_wake_losses(scenario_dirs)
    if not wake_losses.empty:
        out_file = args.out_dir / "wake_losses.csv"
        wake_losses.to_csv(out_file, index=False)
        print(f"  Saved to {out_file}")
        print(f"  Shape: {wake_losses.shape}")
    
    # Extract wake vs density
    print("\nExtracting wake vs density...")
    wake_density = extract_wake_vs_density(scenario_dirs)
    if not wake_density.empty:
        out_file = args.out_dir / "wake_density.csv"
        wake_density.to_csv(out_file, index=False)
        print(f"  Saved to {out_file}")
        print(f"  Shape: {wake_density.shape}")
    
    # Extract CF metrics (available, dispatch, curtailment)
    print("\nExtracting CF metrics (available, dispatch, curtailment)...")
    cf_metrics = extract_cf_metrics(scenario_dirs)
    if not cf_metrics.empty:
        out_file = args.out_dir / "cf_metrics.csv"
        cf_metrics.to_csv(out_file, index=False)
        print(f"  Saved to {out_file}")
        print(f"  Shape: {cf_metrics.shape}")
    
    # Extract system metrics
    print("\nExtracting system metrics...")
    system_metrics = extract_system_metrics(scenario_dirs)
    if not system_metrics.empty:
        out_file = args.out_dir / "system_metrics.csv"
        system_metrics.to_csv(out_file, index=False)
        print(f"  Saved to {out_file}")
        print(f"  Columns: {list(system_metrics.columns)}")
    
    # Resolution analysis
    if args.resolution_analysis:
        print("\nExtracting resolution metrics...")
        resolution_dfs = []
        for scenario in args.scenarios:
            df = extract_resolution_metrics(args.results_dir, scenario, args.splits)
            if not df.empty:
                resolution_dfs.append(df)
        
        if resolution_dfs:
            resolution_metrics = pd.concat(resolution_dfs, ignore_index=True)
            out_file = args.out_dir / "resolution_metrics.csv"
            resolution_metrics.to_csv(out_file, index=False)
            print(f"  Saved to {out_file}")
            print(f"  Shape: {resolution_metrics.shape}")
    
    print("\n✓ Data extraction complete!")
    print(f"\nTo generate plots, run:")
    print(f"python compare_wake_runs_new.py all \\")
    print(f"    --wake-losses {args.out_dir}/wake_losses.csv \\")
    print(f"    --wake-density {args.out_dir}/wake_density.csv \\")
    print(f"    --system {args.out_dir}/system_metrics.csv \\")
    if args.resolution_analysis:
        print(f"    --resolution {args.out_dir}/resolution_metrics.csv \\")
    print(f"    --scenarios {' '.join(args.scenarios)} \\")
    print(f"    --out-dir plots/wake/")
    
    # Show split info
    if len(args.splits) > 1:
        print(f"\nNote: Data includes {len(args.splits)} split sizes.")
        print(f"      Filter by split column in CSV if needed for specific plots.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


# EXAMPLE USAGE:
# python extract_wake_data.py \
#     --results-dir ../results/thesis-wake-2030-10-northsea-dominant-6h \
#     --out-dir data/wake_extracted \
#     --scenarios base standard glaum new_more \
#     --splits 1000 10000 100000 \
#     --resolution-analysis
# Analysis Scripts

This folder contains scripts for analyzing and plotting PyPSA-Eur results.

## Wake Analysis Pipeline

### 1. Extract Data from PyPSA Networks

Use `extract_wake_data.py` to extract metrics from PyPSA `.nc` files and CSVs:

```bash
python extract_wake_data.py \
    --results-dir ../results/thesis-wake-2030-10-northsea-dominant-6h \
    --out-dir ../data/wake_extracted \
    --scenarios base standard glaum new_more \
    --splits 100000
```

**Extract data from multiple split sizes:**

```bash
python extract_wake_data.py \
    --results-dir ../results/thesis-wake-2030-10-northsea-dominant-6h \
    --out-dir ../data/wake_extracted \
    --scenarios base standard glaum new_more \
    --splits 1000 5000 10000 50000 100000
```

**Options:**
- `--results-dir`: Base directory containing scenario subdirectories
- `--out-dir`: Output directory for extracted CSV files
- `--scenarios`: List of scenarios to extract
- `--splits`: Spatial split sizes to extract (default: [100000]). Can specify multiple.
- `--resolution-analysis`: Also extract resolution metrics across all splits

**What it extracts:**
- `wake_losses.csv`: Wake loss multipliers per scenario and split
- `system_metrics.csv`: System-level costs, curtailment, capacities per scenario and split
- `wake_density.csv`: Wake losses vs capacity density (if available)
- `resolution_metrics.csv`: Metrics across spatial resolutions (with `--resolution-analysis`)

**Note:** When extracting multiple splits, the output CSVs include a `split` column. You can filter by this column if you want to plot data for a specific split size only.

### 2. Generate Plots

Use `compare_wake_runs_new.py` to create thesis-ready plots:

#### Generate all plots at once:

```bash
python compare_wake_runs_new.py all \
    --wake-losses ../data/wake_extracted/wake_losses.csv \
    --system ../data/wake_extracted/system_metrics.csv \
    --scenarios base standard glaum new_more \
    --out-dir ../plots/wake_analysis/
```

#### Generate individual plots:

```bash
# Wake loss distribution (PDF)
python compare_wake_runs_new.py dist \
    --in ../data/wake_extracted/wake_losses.csv \
    --out ../plots/wake_loss_pdf.pdf

# Wake loss CDF
python compare_wake_runs_new.py cdf \
    --in ../data/wake_extracted/wake_losses.csv \
    --out ../plots/wake_loss_cdf.pdf

# Wake loss boxplot
python compare_wake_runs_new.py box \
    --in ../data/wake_extracted/wake_losses.csv \
    --out ../plots/wake_loss_box.pdf

# System metric bar chart
python compare_wake_runs_new.py system_bars \
    --in ../data/wake_extracted/system_metrics.csv \
    --y curtailment_twh \
    --out ../plots/curtailment.pdf
```

**Available commands:**
- `dist`: Wake loss PDF (histogram)
- `cdf`: Wake loss CDF
- `box`: Wake loss boxplot
- `loss_vs_density`: Wake loss vs capacity density scatter plot
- `delta_cf_map`: ΔCF map for a scenario (requires GeoJSON)
- `cap_map`: Capacity density map (requires GeoJSON)
- `cap_delta_map`: Δ capacity density vs baseline (requires GeoJSON)
- `resolution_lines`: Metric vs spatial resolution
- `system_bars`: System metric bar chart
- `all`: Generate all available plots

## Complete Workflow Example

**Single split size:**

```bash
# 1. Extract data from PyPSA results
python extract_wake_data.py \
    --results-dir ../results/thesis-wake-2030-10-northsea-dominant-6h \
    --out-dir ../data/wake_extracted \
    --scenarios base standard glaum new_more \
    --splits 100000

# 2. Generate all plots
python compare_wake_runs_new.py all \
    --wake-losses ../data/wake_extracted/wake_losses.csv \
    --system ../data/wake_extracted/system_metrics.csv \
    --scenarios base standard glaum new_more \
    --out-dir ../plots/wake_analysis/
```

**Multiple split sizes (for resolution analysis):**

```bash
# 1. Extract data from multiple splits
python extract_wake_data.py \
    --results-dir ../results/thesis-wake-2030-10-northsea-dominant-6h \
    --out-dir ../data/wake_extracted_multi \
    --scenarios base standard glaum new_more \
    --splits 1000 10000 100000 \
    --resolution-analysis

# 2. Generate plots (includes resolution metrics)
python compare_wake_runs_new.py all \
    --wake-losses ../data/wake_extracted_multi/wake_losses.csv \
    --system ../data/wake_extracted_multi/system_metrics.csv \
    --resolution ../data/wake_extracted_multi/resolution_metrics.csv \
    --scenarios base standard glaum new_more \
    --out-dir ../plots/wake_analysis_multi/
```

## Styling

Plots use the shared styling from:
- `thesis_colors.py`: Color schemes and scenario labels
- `plotting_style.py`: Matplotlib rc params and formatting

To modify colors or labels, edit `thesis_colors.py`.

## Working with Multiple Splits

When you extract data from multiple splits, the CSVs include a `split` column. The plotting scripts will use all data by default (combining splits). If you want to plot only a specific split size:

```python
import pandas as pd

# Read and filter for a specific split
df = pd.read_csv('data/wake_extracted_multi/wake_losses.csv')
df_single = df[df['split'] == 100000]
df_single.to_csv('data/wake_losses_100k_only.csv', index=False)

# Then use the filtered file for plotting
```

Or use the `--resolution-analysis` flag which automatically creates resolution comparison plots across all splits.

## Requirements

- pandas
- numpy
- matplotlib
- geopandas (optional, for map plots)
- pypsa (optional, for reading `.nc` files directly)

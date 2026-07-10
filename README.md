# pypsa-eur-wind
**Extended wind resource modelling for PyPSA-Eur**

## Purpose of this fork

This repository is a research fork of the upstream
[pypsa-eur](https://github.com/PyPSA/pypsa-eur) model, developed to support
methodological work on **wind resource representation and uncertainty**
for a doctoral thesis.

The fork extends the standard PyPSA-Eur wind workflow with:

1. **Configurable spatial resolution of wind resources**
2. **A novel wake correction implementation** (including a tiered-density model)
3. **Bias correction of wind resources using the PyVWF framework**
4. **A comprehensive analysis and visualisation suite** for thesis-ready figures
5. **Programmatic scenario generation** across wake, bias, and resolution dimensions

The aim is to enable **controlled sensitivity analysis** of wind resource
assumptions and their impact on power system modelling results.

## Relationship to upstream pypsa-eur

This repository **tracks the structure and philosophy of pypsa-eur**, but
introduces additional functionality that is **not present upstream**.

Unless stated otherwise:
- All standard PyPSA-Eur workflows remain unchanged
- Existing configuration files remain compatible
- Extensions are optional and can be toggled via configuration

This fork should be treated as **research software**, not a drop-in replacement
for the official pypsa-eur repository.

## Summary of methodological extensions

### 1. Variable spatial resolution of wind resources

This fork introduces the ability to **modify the spatial resolution** at which
wind resources are represented, independently of the power system network
resolution.

This enables experiments such as:
- Aggregating wind resources to finer grids
- Preserving high-resolution wind fields while clustering generators
- Quantifying sensitivity to spatial smoothing of wind variability

This functionality is intended for **methodological comparison**, not
operational forecasting.

### 2. Wake correction implementation

A novel wake correction approach is implemented to account for **wind farm
interaction effects** that are not represented in the standard pypsa-eur
workflow.

Three wake models are supported:

| Model | Description |
|-------|-------------|
| `standard` | Flat derate factor (default 0.8855) applied uniformly |
| `glaum` | Capacity-tiered wake model with distinct factors per capacity tier |
| `new_more` | **Tiered-density wake model** (thesis contribution) using fitted exponential curve parameters and density breakpoints |

Key characteristics:
- Applied at the wind resource / capacity factor level
- Designed to scale to national or continental systems
- Explicitly configurable and reproducible
- Wake coefficients are **config-driven** and can be overridden per scenario

The implementation is intended for system-level studies, not detailed
micrositing.

### 3. Bias correction using PyVWF

This fork integrates **bias correction of wind resources** using the
**Python Virtual Wind Farm (PyVWF)** framework.

Bias correction is applied to reanalysis-based wind resources using
observed wind generation data, allowing:

- Correction of systematic reanalysis biases
- Comparison between corrected and uncorrected wind inputs
- Explicit separation of meteorological and system modelling uncertainty

Three bias modes are available: `True` (PyVWF correction), `False` (none),
and `Uniform` (uniform scaling factor).

PyVWF is developed and maintained separately:
https://github.com/ellyess/PyVWF

## Code structure

```
pypsa-eur-wind/
├── Snakefile                        # Main Snakemake workflow
├── rules/                           # Snakemake rule definitions
│   ├── build_electricity.smk
│   ├── build_sector.smk
│   ├── solve_electricity.smk
│   ├── postprocess.smk
│   └── ...
│
├── scripts/                         # Core pipeline scripts
│   ├── wake_helpers.py              # Wake modelling module (region splitting,
│   │                                #   wake factors, multipart geometry, specs)
│   ├── build_renewable_profiles.py  # Wind/solar capacity factor generation
│   ├── add_electricity.py           # Network assembly & generator integration
│   ├── make_summary.py              # Results aggregation
│   ├── fix_offwind_buildout.py      # Offshore capacity constraint locking
│   └── ...                          # Standard pypsa-eur scripts
│
├── analysis_scripts/                # Thesis-specific analysis & plotting
│   ├── compare_wake_runs.py         # Wake model comparison (35+ plot types)
│   ├── compare_sensitivity_runs_tier1.py  # North Sea sensitivity analysis
│   ├── compare_sensitivity_runs_tier2.py  # Europe confirmatory analysis
│   ├── compare_spatial_runs.py      # Resolution diagnostic plots
│   ├── compare_bias_runs.py         # Bias correction comparison
│   ├── extract_wake_data.py         # Data extraction from PyPSA networks
│   ├── fit_new_more_breakpoints.py  # Tiered-density wake model calibration
│   ├── plotting_style.py            # Shared thesis matplotlib styling
│   ├── thesis_colors.py             # Canonical colour schemes & labels
│   ├── network_utils.py             # Shared PyPSA query utilities
│   └── ...                          # Debug & diagnostic scripts
│
├── config/                          # Configuration & scenario generation
│   ├── generate_scenarios.py        # Programmatic scenario YAML generator
│   ├── config.northsea_*.yaml       # Regional config variants
│   ├── scenarios-wake.yaml          # Wake model scenarios
│   ├── scenarios-sensitivity.yaml   # Exhaustive sensitivity matrix
│   ├── scenarios-bias.yaml          # Bias correction variants
│   └── scenarios-spatial.yaml       # Spatial resolution variants
│
├── atlite-bc/                       # Modified Atlite with bias correction
│   └── atlite/
│       ├── wind.py                  # Modified wind speed conversion
│       ├── convert.py               # Bias correction integration
│       └── ...
│
├── wake_extra/                      # Wake model caches & calibration data
│   ├── northsea/                    # North Sea region data & profiles
│   ├── europe/                      # Europe-wide data
│   └── new_more_fit/                # Tiered-density breakpoint calibration
│
├── plots/                           # Generated figures
│   ├── spatial_diagnostics/
│   ├── sensitivity/
│   │   ├── tier1/
│   │   └── tier2/
│   └── wake_analysis/
│
└── data/                            # Input & extracted data
    └── wake_extracted/              # Extracted results for plotting
```

## Analysis and visualisation suite

The `analysis_scripts/` directory provides a comprehensive plotting and
analysis framework for thesis figures, added as part of the recent
refactoring work.

### Plotting scripts

- **`compare_wake_runs.py`** — 35+ plotting functions for wake analysis:
  PDFs, CDFs, boxplots of wake losses; capacity density scatter plots;
  choropleth maps (delta CF, capacity density); system cost and curtailment
  breakdowns; resolution interaction plots.

- **`compare_sensitivity_runs_tier1.py`** — North Sea sector-coupled
  sensitivity at five spatial resolutions: CF distributions, capacity and
  curtailment vs resolution, transmission expansion, heatmaps, tornado plots.

- **`compare_sensitivity_runs_tier2.py`** — Europe-wide confirmatory analysis
  replicating tier-1 metrics at two resolutions.

- **`compare_spatial_runs.py`** — Spatial resolution diagnostics: built
  capacity, CF/curtailment distributions, market value, line volume, and
  price dispersion vs maximum region area.

- **`compare_bias_runs.py`** — Bias correction comparison: CF distributions
  (PDF/CDF/boxplot), dispatch profiles, curtailment analysis, and system
  metrics.

### Utility modules

- **`network_utils.py`** — Shared PyPSA query functions: snapshot-weighted
  capacity factors, objective cost extraction, carrier energy, bus-country
  mapping, transmission expansion metrics.

- **`plotting_style.py`** — Print-ready matplotlib configuration: serif fonts,
  600 DPI, log-scale axis formatting, resolution markers, standard axis
  cleanup.

- **`thesis_colors.py`** — Canonical colour palette and label definitions
  ensuring visual consistency across all figures (baseline, bias, wake,
  bias+wake).

### Calibration

- **`fit_new_more_breakpoints.py`** — Dynamic programming routine to find
  optimal density breakpoints for the tiered-density wake model, producing
  the calibration CSV used by `wake_helpers.py`.

## Scenario generation

Scenario YAML files are generated programmatically from compact profile
definitions via `config/generate_scenarios.py`, replacing hand-written
YAML.

| Profile | Scenarios | Dimensions |
|---------|-----------|------------|
| `wake` | 20 | 4 wake models x 5 resolutions |
| `wake-combined` | 20 | Same with `offwind-combined` carrier |
| `sensitivity` | 20 | 2 wake models x 5 resolutions x 2 bias settings |
| `sensitivity-europe` | 8 | 2 wake models x 2 resolutions x 2 bias settings |
| `spatial` | 5 | Baseline x 5 resolutions |
| `bias` | 3 | 3 bias modes at fixed resolution |

Usage:

```bash
# List available profiles
python config/generate_scenarios.py --list

# Generate a specific profile
python config/generate_scenarios.py --profile wake --output config/scenarios-wake.yaml

# Generate all profiles
python config/generate_scenarios.py --profile all
```

Scenario naming convention: `{wake_model}-s{resolution_km2}-bias{True|False|Uniform}`

## Pipeline improvements

### Config-driven wake parameters

Wake model coefficients are defined as module-level defaults in
`scripts/wake_helpers.py` and can be overridden per scenario via config:

```yaml
spatial_mods:
  wake_coefficients:
    standard:
      derate_factor: 0.90
    new_more:
      alpha: 8.0
```

Any keys not specified fall back to the built-in defaults.

### Cache robustness

- Profile cache paths now include the correction factor
  (`_cf{value}`) to prevent stale cache hits when switching between bias
  correction modes.
- Region fallback emits a warning instead of silently falling back to unsplit
  regions.

### Bias correction cleanup

- Debug output removed from `atlite-bc/atlite/wind.py`.
- `bias_corr` parameter accepts `bool | str | Path`: `True` uses the default
  dataset path, a string or `Path` uses a custom path.

### Offshore buildout fixing

`scripts/fix_offwind_buildout.py` locks offshore wind capacity to a reference
run, enabling controlled comparisons where only the wind resource
representation changes.

## Thesis chapters supported

| Chapter | Topic | Key scripts/configs |
|---------|-------|---------------------|
| Ch 5 | Spatial resolution sensitivity | `compare_spatial_runs.py`, `scenarios-spatial.yaml` |
| Ch 6 | Wake model validation | `compare_wake_runs.py`, `scenarios-wake.yaml`, `fit_new_more_breakpoints.py` |
| Ch 7 | Bias correction impact | `compare_bias_runs.py`, `scenarios-bias.yaml` |
| Ch 8a | North Sea exhaustive sensitivity | `compare_sensitivity_runs_tier1.py`, `scenarios-sensitivity.yaml` |
| Ch 8b | Europe confirmatory analysis | `compare_sensitivity_runs_tier2.py`, `scenarios-sensitivity-europe.yaml` |

## Intended use

This repository is intended for:

- Doctoral and academic research
- Sensitivity analysis of wind resource assumptions
- Methodological comparison of wind modelling approaches
- Reproducible experiments for peer-reviewed publication

It is **not** intended as:
- A production-ready power system model
- A replacement for upstream pypsa-eur
- A general-purpose wind forecasting tool

## Reproducibility and configuration

All extensions introduced in this fork are:

- Explicitly configurable
- Disabled by default unless activated
- Designed to be reproducible across systems

Users are expected to document:
- Spatial resolution choices
- Wake correction configuration
- Bias correction training periods
- Data sources and versions

## Citation and academic use

If this repository is used in academic work, please cite:

- The original **pypsa-eur** repository
- Any relevant PyPSA publications
- The PyVWF framework, if bias correction is enabled

This repository is part of an ongoing PhD thesis and may evolve as the research
progresses.

## Acknowledgements

This work builds directly on the pypsa-eur model developed by the
PyPSA community.

Upstream contributions and design decisions are gratefully acknowledged.

## Disclaimer

This repository represents **research code under active development**.
Results obtained using this fork should be interpreted in the context of
methodological exploration rather than operational modelling.

## License

The code in PyPSA-Eur is released as free software under the
[MIT License](https://opensource.org/licenses/MIT), see [`doc/licenses.md`](doc/licenses.md).
However, different licenses and terms of use may apply to the various
input data, see [`doc/data_sources.md`](doc/data_sources.md).

# pypsa-eur-wind  
**Extended wind resource modelling for PyPSA-Eur**

## Purpose of this fork

This repository is a research fork of the upstream
[pypsa-eur](https://github.com/PyPSA/pypsa-eur) model, developed to support
methodological work on **wind resource representation and uncertainty**
for a doctoral thesis.

The fork extends the standard PyPSA-Eur wind workflow with:

1. **Configurable spatial resolution of wind resources**
2. **A novel wake correction implementation**
3. **Bias correction of wind resources using the PyVWF framework**

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

Key characteristics:
- Applied at the wind resource / capacity factor level
- Designed to scale to national or continental systems
- Explicitly configurable and reproducible

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

PyVWF is developed and maintained separately:
https://github.com/ellyess/PyVWF

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

## Code structure (high-level)

The fork preserves the upstream directory structure, with additional modules
and modifications primarily affecting:

- Wind resource preparation
- Capacity factor generation
- Pre-processing of reanalysis data

Detailed implementation notes are documented in code-level docstrings.

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



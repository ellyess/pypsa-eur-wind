# Thesis Analysis Chapters: Scenario Summary

## Overview Table

| Chapter | Analysis Focus | Region/Scope | Scenarios | Spatial Resolution | Model Variant | Key Features |
|---------|----------------|--------------|-----------|-------------------|---------------|--------------|
| **Chapter 5** | Spatial Resolution | North Sea | Baseline | 1k, 5k, 10k, 50k, 100k, 1M | Standard wake | Resolution sensitivity |
| **Chapter 6** | Wake Models | North Sea | Base, Uniform, Tiered-capacity, Tiered-density | 1k, 5k, 10k, 50k, 100k | No bias correction | Wake model comparison |
| **Chapter 7** | Bias Correction | North Sea | Baseline, Uniform, PyVWF | 100k | Standard wake | Bias correction methods |
| **Chapter 8a** | Sensitivity (Tier 1) | North Sea (s_10) | Base, Bias, Wake, Bias+Wake | 1k, 5k, 10k, 50k, 100k | Dominant offshore | Sector-coupled, exhaustive sensitivity |
| **Chapter 8b** | Sensitivity (Tier 2) | Europe-wide (s_30) | Base, Bias+Wake | Coarse (100k), Fine (10k) | Dominant offshore | Sector-coupled, confirmatory |

---

## Detailed Scenario Definitions

### Scenario Key Convention

Scenarios follow the logic defined in `scenario_key(bias, wake)`:

| Wake Status | Bias Correction | **Scenario Label** | Display Name |
|-------------|-----------------|-------------------|--------------|
| Off | False | `base` | **Baseline** |
| Off | True | `bias` | **PyVWF bias** |
| On (any model) | False | `wake` | **Tiered-density wake** |
| On (any model) | True | `bias+wake` | **Bias + wake** |

---

## Chapter-by-Chapter Details

### Chapter 5: Spatial Resolution Analysis
**Script:** `compare_spatial_runs.py`  
**Results Pattern:** `thesis-spatial-2030-10-northsea-standard-6h/*/networks/*.nc`

| Resolution (m²) | Label | Description |
|----------------|-------|-------------|
| 1,000,000 | Coarse | Continental-scale aggregation |
| 100,000 | Medium-coarse | National/regional |
| 50,000 | Medium | Sub-regional |
| 10,000 | Fine | High-resolution |
| 5,000 | Very fine | Ultra-high resolution |
| 1,000 | Ultra-fine | Maximum resolution tested |

**Configuration:**
- **Region:** North Sea focus
- **Year:** 2030
- **Temporal:** 6h snapshots
- **Offshore strategy:** Standard (uniform wake)
- **Network type:** Electric-only

---

### Chapter 6: Wake Model Comparison
**Script:** `compare_wake_runs.py`  
**Extraction:** `extract_wake_data.py`  
**Results Pattern:** `thesis-wake-2030-10-northsea-dominant-6h/`

| Wake Model | Alias | Display Name | Description |
|------------|-------|--------------|-------------|
| `base` | `off`, `no_wake` | **Baseline** | No wake losses (uncorrected physics) |
| `standard` | `wakeoff` | **Uniform** | Uniform wake adjustment |
| `glaum` | - | **Tiered-capacity** | Capacity-based wake tiers |
| `new_more` | `density`, `density_based` | **Tiered-density** | Density-based wake tiers (thesis model) |

**Resolutions tested:** 1k, 5k, 10k, 50k, 100k  
**Bias correction:** False (all scenarios)

**Outputs:**
- Wake loss distributions (PDF, CDF, boxplots)
- Capacity density vs wake loss scatter
- System metrics (cost, curtailment, capacity)
- Resolution sensitivity analysis

---

### Chapter 7: Bias Correction Comparison
**Script:** `compare_bias_runs.py`  
**Results Pattern:** `thesis-bias-2030-10-northsea-standard-6h/*/networks/*.nc`

| Bias Method | Config Key | Display Name | Description |
|-------------|-----------|--------------|-------------|
| Raw ERA5 | `biasFalse` | **Baseline** | No bias correction |
| Uniform corr. | `biasUniform` | **Uniform** | Constant offset correction |
| PyVWF | `biasTrue` | **PyVWF** | Physics-derived variable wind field correction |

**Resolution:** 100k (s_10)  
**Wake model:** Standard  
**Region:** North Sea  

**Metrics:**
- CF distributions
- Capacity allocation
- System costs
- Curtailment patterns

---

### Chapter 8a: Sensitivity Analysis - Tier 1
**Script:** `compare_sensitivity_runs_tier1.py`  
**Results Pattern:** `thesis-sensitivity-2030-10-northsea-dominant-6h/**/postnetworks/*.nc`

**Full factorial design:**

| Scenario | Wake Model | Bias Correction | Resolution Tested |
|----------|-----------|----------------|-------------------|
| `base` | Off (`base`) | False | 1k, 5k, 10k, 50k, 100k |
| `bias` | Off (`base`) | True | 1k, 5k, 10k, 50k, 100k |
| `wake` | Tiered-density (`new_more`) | False | 1k, 5k, 10k, 50k, 100k |
| `bias+wake` | Tiered-density (`new_more`) | True | 1k, 5k, 10k, 50k, 100k |

**Configuration:**
- **Region:** North Sea focus (clustered s_10)
- **Network:** Sector-coupled (electricity + H₂ + heat + industry)
- **Offshore strategy:** Dominant offshore
- **Year:** 2030
- **Temporal resolution:** 6h

**Metrics tracked:**
- Onwind/offshore capacity (GW)
- Transmission expansion (TW·km)
- System objective (cost)
- Offshore curtailment fraction
- CF distributions (ECDF for dispatch & curtailment)

---

### Chapter 8b: Sensitivity Analysis - Tier 2
**Script:** `compare_sensitivity_runs_tier2.py`  
**Results Pattern:** `thesis-sensitivity-2030-30-europe-dominant-6h/**/postnetworks/*.nc`

**Reduced confirmatory design:**

| Scenario | Wake Model | Bias Correction | Resolution Tested |
|----------|-----------|----------------|-------------------|
| `base` | Off (`base`) | False | 10k (fine), 100k (coarse) |
| `bias+wake` | Tiered-density (`new_more`) | True | 10k (fine), 100k (coarse) |

**Configuration:**
- **Region:** Europe-wide (s_30 clusters)
- **Network:** **Sector-coupled** (includes H₂ electrolysers, storage, etc.)
- **Offshore strategy:** Dominant offshore
- **Year:** 2030
- **Temporal resolution:** 6h

**Additional metrics (sector-coupling lens):**
- Electrolyser capacity (GW)
- H₂ production (TWh)
- Cross-sector interactions
- System flexibility requirements

---

## Wake Model Aliases

Since wake models appear under different names across configs, the following aliases are consolidated:

```python
WAKE_ALIASES = {
    "base": "off",          # No wake losses
    "standard": "off",      # Uniform wake (legacy)
    "no_wake": "off",
    "wakeoff": "off",
    "off": "off",
    "new_more": "density",  # Tiered-density (thesis model)
    "density": "density",
    "density_based": "density",
    "density-based": "density",
}
```

---

## Color Scheme (Thesis-wide Consistency)

All plots use the following colorblind-safe, print-friendly palette:

| Scenario | Color (Hex) | Description |
|----------|------------|-------------|
| `base` | `#4D4D4D` | Charcoal (anchor/baseline) |
| `standard` | `#D55E00` | Muted orange (Okabe–Ito vermillion) |
| `bias` | `#5DAE8B` | Muted green (PyVWF correction) |
| `wake` | `#2F4B7C` | Muted blue (wake physics) |
| `bias+wake` | `#8172B2` | Muted purple (combined) |

**Wake model-specific colors:**
- **Baseline** (`base`): Charcoal
- **Uniform** (`standard`): Vermillion
- **Tiered-capacity** (`glaum`): Blue
- **Tiered-density** (`new_more`): Bluish-green

---

## Folder Structure

```
results/
├── thesis-wake-2030-10-northsea-dominant-6h/
│   ├── base-s100000-biasFalse/
│   ├── standard-s100000-biasFalse/
│   ├── glaum-s100000-biasFalse/
│   ├── new_more-s100000-biasFalse/
│   └── ...
├── thesis-bias-2030-10-northsea-standard-6h/
│   ├── base-s100000-biasFalse/
│   ├── base-s100000-biasTrue/
│   ├── base-s100000-biasUniform/
│   └── ...
├── thesis-spatial-2030-10-northsea-standard-6h/
│   └── base-s{1000,5000,10000,50000,100000,1000000}-biasFalse/
├── thesis-sensitivity-2030-10-northsea-dominant-6h/
│   ├── base-s{1000,...,100000}-biasFalse/
│   ├── base-s{1000,...,100000}-biasTrue/
│   ├── new_more-s{1000,...,100000}-biasFalse/
│   ├── new_more-s{1000,...,100000}-biasTrue/
│   └── ...
└── thesis-sensitivity-2030-30-europe-dominant-6h/
    ├── base-s{10000,100000}-biasFalse/
    ├── new_more-s{10000,100000}-biasTrue/
    └── ...
```

---

## Running the Analysis

### Individual chapters:
```bash
python run_all.py --chapters wake          # Chapter 6
python run_all.py --chapters bias          # Chapter 7
python run_all.py --chapters spatial       # Chapter 5
python run_all.py --chapters sensitivity   # Chapter 8 (both tiers)
```

### Specific tier:
```bash
python run_all.py --chapters sensitivity_tier1  # Chapter 8a (high-res)
python run_all.py --chapters sensitivity_tier2  # Chapter 8b (sector-coupled)
```

### All chapters:
```bash
python run_all.py
```

---

## Key Differences: Tier 1 vs Tier 2 (Chapter 8)

| Aspect | Tier 1 | Tier 2 |
|--------|--------|--------|
| **Purpose** | Exhaustive sensitivity analysis | Confirmatory reduced analysis |
| **Region** | North Sea (s_10) | Europe-wide (s_30) |
| **Scenarios** | 4 scenarios × 5 resolutions | 2 scenarios × 2 resolutions |
| **Network** | Sector-coupled | Sector-coupled |
| **Resolution range** | 1k – 100k | 10k, 100k |
| **Computational cost** | Lower (10 clusters) | Higher (30 clusters, larger geography) |
| **Focus** | High-resolution sensitivity | Geographic scalability |

---

## Notes

1. **Resolution notation**: `s_10` = 10 clusters, `s_30` = 30 clusters. The threshold (e.g., 100000 m²) controls spatial aggregation within clusters.

2. **Bias correction**: `biasTrue` applies PyVWF (Physics-derived Variable Wind Field) correction; `biasFalse` uses raw ERA5; `biasUniform` applies a constant offset.

3. **Wake models**:
   - `base` = no wake losses (uncorrected baseline)
   - `standard` = uniform wake adjustment
   - `glaum` = tiered wake based on capacity thresholds
   - `new_more` = tiered wake based on density thresholds (thesis contribution)

4. **Offshore strategy**:
   - `standard` = uniform offshore treatment
   - `dominant` = prioritize offshore wind in optimization

5. **Temporal resolution**: All runs use 6-hour snapshots (representative weeks/days).

---

**Last updated:** 2026-02-18  
**Author:** Ellyess  
**Repository:** pypsa-eur-wind (improving-functions branch)

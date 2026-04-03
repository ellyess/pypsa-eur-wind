# Quick Reference: Thesis Scenarios

## Scenario Matrix

### Four Core Scenarios (Chapters 8a & 8b)

```
                    │  No Bias Correction  │  PyVWF Bias Correction
────────────────────┼──────────────────────┼────────────────────────
No Wake Model       │   BASE (Baseline)    │   BIAS (PyVWF bias)
                    │   Color: Charcoal    │   Color: Green
────────────────────┼──────────────────────┼────────────────────────
Tiered-Density Wake │   WAKE (Wake only)   │   BIAS+WAKE (Combined)
                    │   Color: Blue        │   Color: Purple
```

## Resolution Levels

| Resolution | Area (km²) | Usage | Clusters (s_X) |
|------------|-----------|-------|----------------|
| **1k** | 1,000 | Ultra-fine | Variable |
| **5k** | 5,000 | Very fine | Variable |
| **10k** | 10,000 | Fine | s_10 (North Sea), s_30 (Europe) |
| **50k** | 50,000 | Medium | Variable |
| **100k** | 100,000 | Medium-coarse | Variable |
| **1M** | 1,000,000 | Coarse (continental) | Variable |

## Color Palette (Hex Codes)

- **Baseline:** `#4D4D4D` (Charcoal) - Reference scenario
- **Standard/Uniform:** `#D55E00` (Orange) - Legacy/alternative methods
- **PyVWF Bias:** `#5DAE8B` (Green) - Bias correction
- **Wake:** `#2F4B7C` (Blue) - Wake physics
- **Bias+Wake:** `#8172B2` (Purple) - Combined effects

### Wake Model Colors

- **Baseline (no wake):** `#4D4D4D` (Charcoal)
- **Uniform wake:** `#D55E00` (Vermillion)
- **Tiered-capacity (Glaum):** `#0072B2` (Blue)
- **Tiered-density (new_more):** `#009E73` (Bluish-green)

## Chapter Quick Reference

| Chapter | Key Question | Main Finding Area |
|---------|--------------|-------------------|
| **5** | How does spatial resolution affect results? | Resolution sensitivity |
| **6** | Which wake model best represents physics? | Wake model validation |
| **7** | What's the impact of bias correction? | Data quality effects |
| **8a** | Combined effects at high resolution? | Exhaustive factorial sensitivity (North Sea) |
| **8b** | Do findings scale to larger regions? | Geographic scalability (Europe-wide) |

## Model Configurations

### Wake Models

1. **Base** (`base`, `off`, `no_wake`) - No wake modeling
2. **Uniform** (`standard`) - Constant wake adjustment
3. **Tiered-capacity** (`glaum`) - Wake tiers based on capacity thresholds
4. **Tiered-density** (`new_more`, `density`) - Wake tiers based on density thresholds ⭐ *Thesis contribution*

### Bias Correction Methods

1. **None** (`biasFalse`) - Raw ERA5 reanalysis data
2. **Uniform** (`biasUniform`) - Constant offset correction
3. **PyVWF** (`biasTrue`) - Physics-derived Variable Wind Field correction

### Offshore Strategies

- **Standard:** Uniform offshore treatment
- **Dominant:** Prioritize offshore wind in optimization

## File Naming Convention

Results folders follow: `{wakeprefix}-s{resolution}-bias{True|False}/`

Examples:
- `base-s100000-biasFalse` → Baseline, 100k resolution, no bias correction
- `new_more-s10000-biasTrue` → Tiered-density wake, 10k resolution, PyVWF bias correction

## Analysis Metrics Tracked

### Wind Generation
- Onshore/offshore capacity (GW)
- Energy generation (TWh)
- Curtailment fraction
- Capacity factors (dispatch & curtailment)

### System-wide
- Transmission expansion (TW·km)
- System objective/cost (EUR)
- Network topology

### Sector-Coupling (Ch 8b only)
- Electrolyser capacity (GW)
- H₂ production (TWh)
- Cross-sector flows

## Common Analysis Commands

```bash
# Run all chapters
python run_all.py

# Run specific chapter
python run_all.py --chapters wake
python run_all.py --chapters sensitivity_tier1

# List available chapters
python run_all.py --list

# Dry run (see what would execute)
python run_all.py --dry-run
```

## Key Acronyms

- **PyVWF:** Physics-derived Variable Wind Field
- **CF:** Capacity Factor
- **ECDF:** Empirical Cumulative Distribution Function
- **TW·km:** Terawatt-kilometers (transmission capacity × distance)
- **TWh:** Terawatt-hours (energy)
- **GW:** Gigawatts (power capacity)

---

**For full details, see:** [SCENARIO_TABLE.md](SCENARIO_TABLE.md)

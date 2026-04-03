# Community Engagement Drafts for PyPSA-Eur Upstream Contribution

## 1. GitHub Discussion for PyPSA/pypsa-eur

**Title:** Density-dependent wake modeling and variable spatial resolution for offshore wind

---

Hi all,

I'd like to propose contributing several methodological extensions for offshore wind
modeling from my PhD research. These address gaps identified in Issues #153 and #331.

### What I'd like to contribute

**1. Density-dependent wake effect modeling**

Replaces the current flat `correction_factor: 0.8855` with configurable wake models:

- **Flat derate** (current default, preserved as backward-compatible baseline)
- **Capacity-tiered** (after Glaum et al. — loss depends on total installed capacity)
- **Tiered-density** (novel — loss depends on capacity density in MW/km², making it
  resolution-consistent)

The tiered-density model addresses the fact that a flat derate is not physically
consistent across different spatial resolutions: coarsely aggregated regions undercount
wake interactions. In our experiments, this changes offshore capacity allocation by up
to 68% and system cost by +1.6–1.8%.

**2. Variable spatial resolution for wind resources**

Currently, wind resource resolution is tied to the network clustering level. This
extension allows decoupling them: offshore (and optionally onshore) regions can be split
into sub-regions via K-means/Voronoi partitioning, with a configurable area threshold
(e.g., max 10,000 km² per wind region).

This enables sensitivity analysis of spatial smoothing effects on capacity factors,
which we find changes system cost by up to +8.5%.

**3. Wind speed bias correction (separate PR to atlite)**

Integration of station-derived bias correction (scalar + offset) for ERA5 wind speeds
using the PyVWF framework (Benmoufok et al., DOI: 10.1016/j.energy.2024.133759). This
complements the GWA-based correction in atlite PR #405 by using observed generation
data. Impact: +8.2% system cost.

### Proposed PR structure

1. **PR to `PyPSA/atlite`**: Bias correction (scalar + offset) — I'd like to coordinate
   with PR #405
2. **PR to `PyPSA/pypsa-eur`**: Variable spatial resolution (region splitting)
3. **PR to `PyPSA/pypsa-eur`**: Wake effect modeling (depends on #2 for region areas)
4. **PR to `PyPSA/pypsa-eur`**: Documentation and config schema

### Questions for maintainers

1. **Config structure**: Should wake model configuration go under
   `electricity.wake_model`, or do you prefer a different location? Currently in my
   fork it lives under `spatial_mods`.
2. **Region splitting**: Should this integrate into `cluster_network.py` or be a
   separate Snakemake rule?
3. **Naming**: What naming conventions would you prefer for the wake models? I've been
   using research working names (`new_more`, `glaum`) but propose `tiered_density` and
   `capacity_tiered` for upstream.

The thesis is in preparation and builds on the published PyVWF paper. Happy to discuss
any of these in detail.

---

## 2. Comment for Atlite PR #405 / Issue #373

---

Hi,

I have an existing implementation of wind speed bias correction that takes a
complementary approach to this PR. Rather than using Global Wind Atlas multiplicative
correction, my approach applies a **scalar + offset** correction derived from
station-level observed wind generation data, using the PyVWF framework (Benmoufok et
al., DOI: 10.1016/j.energy.2024.133759).

The implementation:

- Adds a `bias_corr` parameter to `extrapolate_wind_speed()` accepting `False`, `True`,
  string keywords (`"idw"`, `"kriging"`), or a custom file path
- Loads a NetCDF file containing 2D `scalar` and `offset` fields
- Applies: `wind_speed_corrected = (wind_speed * scalar) + offset`
- Handles both x/y and lon/lat coordinate conventions

This is complementary to the GWA approach in PR #405 — GWA provides a gridded
physically-based correction, while station-derived corrections capture biases specific
to actual generation observations. Both are useful for different validation contexts.

I'd like to contribute this, either:

- As an extension to PR #405's interface (if a unified `bias_correction` parameter can
  support both multiplicative and scalar+offset modes)
- As a separate follow-up PR building on whatever interface PR #405 establishes

I'm happy to adapt to whatever architecture you converge on. What would be most helpful?

---

## 3. Note: Relationship Between Split Regions and Upstream Resource Classes (Bins)

Since PyPSA-Eur now supports multiple capacity factor bins per bus (the `resource_classes`
parameter), it's worth clarifying that **split regions and resource classes are orthogonal
and complementary**, not redundant:

- **Resource classes (bins)** partition grid cells within a fixed geographic region by
  **capacity factor quality** — the optimizer can prefer building on windier sites within
  a region. The region boundaries do not change, and bins have no concept of area.

- **Split regions** partition large geographic regions into **smaller spatial areas** via
  Voronoi cells. Each sub-region gets its own temporal wind profile (capturing spatial
  decorrelation) and its own **area in km²**.

The critical distinction for the tiered-density wake model: it needs region area to compute
capacity density (MW/km²). Without split regions, a 100,000 km² North Sea cluster would
allow ~83 GW per density tier, making the wake penalty physically meaningless. With split
regions at a 10,000 km² threshold, tiers are capped at a realistic ~8 GW.

| Aspect                        | Resource Classes (Bins)       | Split Regions                      |
|-------------------------------|-------------------------------|------------------------------------|
| Differentiates by             | CF quality (wind speed)       | Geographic location                |
| Changes region geometry       | No                            | Yes                                |
| Provides area for wake model  | No                            | Yes                                |
| Captures spatial decorrelation| No                            | Yes                                |
| Prevents CF averaging         | Partially (by quality)        | Yes (by geography)                 |

Ideally both would be used together: split regions first to create physically meaningful
sub-regions for wake modeling and temporal decorrelation, then bins within each sub-region
for CF quality differentiation.

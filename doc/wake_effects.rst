..
  SPDX-FileCopyrightText: 2024 The PyPSA-Eur Authors

  SPDX-License-Identifier: CC-BY-4.0

.. _wake_effects:

######################
Wake Effect Modelling
######################

Offshore wind farms experience wake losses: downstream turbines receive reduced
wind speeds due to energy extraction by upstream turbines. At the power system
level, these losses reduce the effective capacity factor of offshore wind and
affect optimal capacity allocation, curtailment, and system costs.

PyPSA-Eur supports three wake modelling approaches, configured via the
``electricity.wake_model`` section in the configuration file.

Wake Models
===========

Flat Derate (``flat``)
-----------------------

The default approach. A uniform correction factor is applied to the capacity
factor time series (``p_max_pu``) of all offshore wind generators.

.. math::

   p^\text{max}_\text{pu,corrected}(t) = \text{derate\_factor} \times p^\text{max}_\text{pu}(t)

The default value of 0.8855 corresponds to an 11.45% flat wake loss, as adopted
in PyPSA-Eur since `PR #278 <https://github.com/PyPSA/pypsa-eur/pull/278>`_.
This approach is simple but does not account for differences in wind farm
density across regions.

Capacity-Tiered (``capacity_tiered``)
--------------------------------------

After Glaum et al., this model applies a global derate to all offshore
generators, then applies additional marginal wake losses for generators
exceeding absolute capacity thresholds:

- **Tier 1** (< 2 GW): No additional loss beyond global derate
- **Tier 2** (2–10 GW): Additional ~12.8% marginal loss
- **Tier 3** (> 10 GW): Additional ~26.1% marginal loss

Each generator is split into sub-generators per tier, each with a modified
``p_max_pu`` time series reflecting cumulative wake losses.

This model captures the observation that wake losses increase with wind farm
size, but uses absolute capacity thresholds that are not consistent across
spatial resolutions: a 5 GW generator in a small region has a different
physical density than a 5 GW generator in a large region.

Tiered-Density (``tiered_density``)
-------------------------------------

The tiered-density model is a novel approach where marginal wake losses depend
on **capacity density** (MW/km²) rather than absolute capacity. This makes the
model physically consistent across different spatial resolutions.

The total wake loss :math:`T(x)` at density :math:`x` MW/km² is given by a
fitted exponential curve:

.. math::

   T(x) = \alpha \cdot e^{-x/\beta} + \gamma \cdot x + \delta

where :math:`\alpha`, :math:`\beta`, :math:`\gamma`, :math:`\delta` are fitted
parameters. The density axis is discretised into tiers at configurable
breakpoints, and marginal losses are computed for each tier.

Generators are split into sub-generators, each assigned a density tier with a
corresponding wake loss factor. The capacity per tier is determined by the
region area multiplied by the density increment for that tier.

This model requires region area information (from the ``regions_offshore``
input) and is designed to work with the variable spatial resolution feature
(:ref:`region_splitting`).

**Default parameters** (fitted from wind farm interaction data):

- :math:`\alpha = 7.3`
- :math:`\beta = 0.05`
- :math:`\gamma = -0.7`
- :math:`\delta = -14.6`
- Breakpoints: ``[0, 0.037, 0.827, 1.511, 2.293, 3.172, 4]`` MW/km²

Configuration
=============

.. code-block:: yaml

   electricity:
     wake_model:
       method: "flat"  # "flat", "tiered_density", or "capacity_tiered"
       flat:
         derate_factor: 0.8855
       tiered_density:
         alpha: 7.3
         beta: 0.05
         gamma: -0.7
         delta: -14.6
         breakpoints: [0, 0.037, 0.827, 1.511, 2.293, 3.172, 4]
       capacity_tiered:
         global_derate: 0.906
         f2: 0.1279732
         f3_extra: 0.13902848
         max_caps: [2000, 10000]

Setting ``method: "flat"`` with ``derate_factor: 1.0`` disables wake effects
entirely.

Setting ``method: "flat"`` with ``derate_factor: 0.8855`` (default) reproduces
the current upstream PyPSA-Eur behaviour.

Implementation
==============

The wake model is applied in ``scripts/add_electricity.py`` after renewable
generator attachment and capacity estimation. The implementation is in
``scripts/wake_effects.py``.

For the ``capacity_tiered`` and ``tiered_density`` models, offshore wind
generators are:

1. Identified by carrier name matching ``offwind-*``
2. Grouped by region
3. Split into sub-generators, each representing a capacity tier
4. Each sub-generator's ``p_max_pu`` is modified by the cumulative wake loss
   factor for its tier

This splitting preserves the linear optimisation structure: the solver can
choose to build less capacity in high-wake-loss tiers.

References
==========

- Glaum, P. et al. (2023). Capacity-dependent wake losses for offshore wind.
- Benmoufok, E. et al. (in preparation). Tiered-density wake model for
  resolution-consistent power system analysis.
- PyPSA-Eur `Issue #153 <https://github.com/PyPSA/pypsa-eur/issues/153>`_
  ("Consider wake losses").

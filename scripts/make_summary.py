# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Create summary CSV files for all scenario runs including costs, capacities,
capacity factors, curtailment, energy balances, prices and other metrics.
"""

import logging
import sys

import numpy as np
import pandas as pd
import pypsa

from _helpers import configure_logging, get_snapshots, set_scenario_config
from prepare_sector_network import prepare_costs

idx = pd.IndexSlice
logger = logging.getLogger(__name__)

# which "nom" attribute to use per component type
opt_name = {"Store": "e", "Line": "s", "Transformer": "s"}


def assign_carriers(n):
    # Some networks may not have 'carrier' for lines.
    if "carrier" not in n.lines:
        n.lines["carrier"] = "AC"


def assign_locations(n):
    """
    Parse 'location' from component names like 'DE0 0 onwind' or 'DE0 0_00001 offwind-ac'.
    Keeps empty location for items without a space after the 4th char.
    """
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        if c.df.empty:
            continue
        if "location" not in c.df.columns:
            c.df["location"] = ""
        ifind = pd.Series(c.df.index.str.find(" ", start=4), index=c.df.index)
        for i in ifind.unique():
            names = ifind.index[ifind == i]
            c.df.loc[names, "location"] = "" if i == -1 else names.str[:i]


def _safe_cols(df: pd.DataFrame, cols) -> pd.Index:
    """Return cols ∩ df.columns as Index."""
    if df is None or getattr(df, "empty", True):
        return pd.Index([])
    return pd.Index(cols).intersection(df.columns)


def _safe_index(s: pd.Series, idx_like) -> pd.Index:
    """Return idx_like ∩ s.index as Index."""
    if s is None or getattr(s, "empty", True):
        return pd.Index([])
    return pd.Index(idx_like).intersection(s.index)


def _get_nom_col(c) -> str:
    return opt_name.get(c.name, "p") + "_nom_opt"


def calculate_nodal_cfs(n, label, nodal_cfs):
    # Beware: includes some extraneous 'locations' (e.g. EU-wide commodities) due to naming scheme.
    for c in n.iterate_components(
        (n.branch_components ^ {"Line", "Transformer"})
        | (n.controllable_one_port_components ^ {"Load", "StorageUnit"})
    ):
        if c.df.empty:
            continue

        nom_col = _get_nom_col(c)
        if nom_col not in c.df.columns:
            continue

        capacities_c = c.df.groupby(["location", "carrier"])[nom_col].sum()

        # mean dispatch
        if c.name == "Link":
            if getattr(c.pnl, "p0", None) is None or c.pnl.p0.empty:
                continue
            p = c.pnl.p0.abs().mean()
        elif c.name == "Generator":
            if getattr(c.pnl, "p", None) is None or c.pnl.p.empty:
                continue
            p = c.pnl.p.abs().mean()
        elif c.name == "Store":
            if getattr(c.pnl, "e", None) is None or c.pnl.e.empty:
                continue
            p = c.pnl.e.abs().mean()
        else:
            continue

        c.df["p"] = p.reindex(c.df.index)
        p_c = c.df.groupby(["location", "carrier"])["p"].sum()

        # avoid division by 0
        cf_c = (p_c / capacities_c.replace(0.0, np.nan)).fillna(0.0)

        index = pd.MultiIndex.from_tuples([(c.list_name,) + t for t in cf_c.index.to_list()])
        nodal_cfs = nodal_cfs.reindex(index.union(nodal_cfs.index))
        nodal_cfs.loc[index, label] = cf_c.values

    return nodal_cfs


def calculate_cfs(n, label, cfs):
    for c in n.iterate_components(n.branch_components | (n.controllable_one_port_components ^ {"Load", "StorageUnit"})):
        if c.df.empty:
            continue

        nom_col = _get_nom_col(c)
        if nom_col not in c.df.columns:
            continue

        capacities_c = c.df[nom_col].groupby(c.df.carrier).sum()

        if c.name in ["Link", "Line", "Transformer"]:
            p_df = getattr(c.pnl, "p0", None)
        elif c.name == "Store":
            p_df = getattr(c.pnl, "e", None)
        else:
            p_df = getattr(c.pnl, "p", None)

        if p_df is None or p_df.empty:
            continue

        p = p_df.abs().mean()
        p_c = p.groupby(c.df.carrier).sum()

        cf_c = (p_c / capacities_c.replace(0.0, np.nan)).fillna(0.0)
        cf_c = pd.concat([cf_c], keys=[c.list_name])

        cfs = cfs.reindex(cf_c.index.union(cfs.index))
        cfs.loc[cf_c.index, label] = cf_c

    return cfs


def calculate_nodal_costs(n, label, nodal_costs):
    for c in n.iterate_components(n.branch_components | (n.controllable_one_port_components ^ {"Load"})):
        if c.df.empty:
            continue

        nom_col = _get_nom_col(c)
        if nom_col not in c.df.columns:
            continue

        # capital costs
        cap = c.df.capital_cost * c.df[nom_col]
        capital_costs = cap.groupby([c.df.location, c.df.carrier]).sum()

        index = pd.MultiIndex.from_tuples([(c.list_name, "capital") + t for t in capital_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))
        nodal_costs.loc[index, label] = capital_costs.values

        # marginal costs (weighted energy)
        if c.name == "Link":
            p_df = getattr(c.pnl, "p0", None)
            if p_df is None or p_df.empty:
                continue
            p = p_df.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_df = getattr(c.pnl, "p", None)
            if p_df is None or p_df.empty:
                continue
            p_all = p_df.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p_df = getattr(c.pnl, "p", None)
            if p_df is None or p_df.empty:
                continue
            p = p_df.multiply(n.snapshot_weightings.generators, axis=0).sum()

        # correct sequestration cost
        if c.name == "Store" and "marginal_cost" in c.df.columns:
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)]
            c.df.loc[items, "marginal_cost"] = -20.0

        if "marginal_cost" not in c.df.columns:
            continue

        c.df["marginal_costs"] = p.reindex(c.df.index).fillna(0.0) * c.df.marginal_cost
        marginal_costs = c.df.groupby(["location", "carrier"])["marginal_costs"].sum()

        index = pd.MultiIndex.from_tuples([(c.list_name, "marginal") + t for t in marginal_costs.index.to_list()])
        nodal_costs = nodal_costs.reindex(index.union(nodal_costs.index))
        nodal_costs.loc[index, label] = marginal_costs.values

    return nodal_costs


def calculate_costs(n, label, costs):
    for c in n.iterate_components(n.branch_components | (n.controllable_one_port_components ^ {"Load"})):
        if c.df.empty:
            continue

        nom_col = _get_nom_col(c)
        if nom_col not in c.df.columns:
            continue

        # capital
        capital_costs = c.df.capital_cost * c.df[nom_col]
        capital_costs_grouped = capital_costs.groupby(c.df.carrier).sum()
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=["capital"])
        capital_costs_grouped = pd.concat([capital_costs_grouped], keys=[c.list_name])
        costs = costs.reindex(capital_costs_grouped.index.union(costs.index))
        costs.loc[capital_costs_grouped.index, label] = capital_costs_grouped

        # marginal
        if c.name == "Link":
            p_df = getattr(c.pnl, "p0", None)
            if p_df is None or p_df.empty:
                continue
            p = p_df.multiply(n.snapshot_weightings.generators, axis=0).sum()
        elif c.name == "Line":
            continue
        elif c.name == "StorageUnit":
            p_df = getattr(c.pnl, "p", None)
            if p_df is None or p_df.empty:
                continue
            p_all = p_df.multiply(n.snapshot_weightings.generators, axis=0)
            p_all[p_all < 0.0] = 0.0
            p = p_all.sum()
        else:
            p_df = getattr(c.pnl, "p", None)
            if p_df is None or p_df.empty:
                continue
            p = p_df.multiply(n.snapshot_weightings.generators, axis=0).sum()

        if c.name == "Store" and "marginal_cost" in c.df.columns:
            items = c.df.index[(c.df.carrier == "co2 stored") & (c.df.marginal_cost <= -100.0)]
            c.df.loc[items, "marginal_cost"] = -20.0

        if "marginal_cost" not in c.df.columns:
            continue

        marginal_costs = p.reindex(c.df.index).fillna(0.0) * c.df.marginal_cost
        marginal_costs_grouped = marginal_costs.groupby(c.df.carrier).sum()
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=["marginal"])
        marginal_costs_grouped = pd.concat([marginal_costs_grouped], keys=[c.list_name])
        costs = costs.reindex(marginal_costs_grouped.index.union(costs.index))
        costs.loc[marginal_costs_grouped.index, label] = marginal_costs_grouped

    return costs


def calculate_cumulative_cost():
    planning_horizons = snakemake.params.scenario["planning_horizons"]

    cumulative_cost = pd.DataFrame(
        index=df["costs"].sum().index,
        columns=pd.Series(data=np.arange(0, 0.1, 0.01), name="social discount rate"),
    )

    for r in cumulative_cost.columns:
        cumulative_cost[r] = [
            df["costs"].sum()[index] / ((1 + r) ** (index[-1] - planning_horizons[0]))
            for index in cumulative_cost.index
        ]

    for r in cumulative_cost.columns:
        for cluster in cumulative_cost.index.get_level_values(level=0).unique():
            for ll in cumulative_cost.index.get_level_values(level=1).unique():
                for sector_opts in cumulative_cost.index.get_level_values(level=2).unique():
                    cumulative_cost.loc[(cluster, ll, sector_opts, "cumulative cost"), r] = np.trapz(
                        cumulative_cost.loc[idx[cluster, ll, sector_opts, planning_horizons], r].values,
                        x=planning_horizons,
                    )

    return cumulative_cost


def calculate_nodal_capacities(n, label, nodal_capacities):
    for c in n.iterate_components(n.branch_components | (n.controllable_one_port_components ^ {"Load"})):
        if c.df.empty:
            continue

        nom_col = _get_nom_col(c)
        if nom_col not in c.df.columns:
            continue

        nodal_caps = c.df.groupby(["location", "carrier"])[nom_col].sum()
        index = pd.MultiIndex.from_tuples([(c.list_name,) + t for t in nodal_caps.index.to_list()])
        nodal_capacities = nodal_capacities.reindex(index.union(nodal_capacities.index))
        nodal_capacities.loc[index, label] = nodal_caps.values

    return nodal_capacities


def calculate_capacities(n, label, capacities):
    for c in n.iterate_components(n.branch_components | (n.controllable_one_port_components ^ {"Load"})):
        if c.df.empty:
            continue

        nom_col = _get_nom_col(c)
        if nom_col not in c.df.columns:
            continue

        caps = c.df[nom_col].groupby(c.df.carrier).sum()
        caps = pd.concat([caps], keys=[c.list_name])

        capacities = capacities.reindex(caps.index.union(capacities.index))
        capacities.loc[caps.index, label] = caps

    return capacities


def calculate_curtailment(n, label, curtailment):
    if n.generators.empty:
        return curtailment
    if getattr(n, "generators_t", None) is None:
        return curtailment
    if getattr(n.generators_t, "p_max_pu", None) is None or n.generators_t.p_max_pu.empty:
        return curtailment
    if getattr(n.generators_t, "p", None) is None or n.generators_t.p.empty:
        return curtailment

    avail = (
        n.generators_t.p_max_pu.multiply(n.generators.p_nom_opt, axis=1)
        .sum()
        .groupby(n.generators.carrier)
        .sum()
    )
    used = n.generators_t.p.sum().groupby(n.generators.carrier).sum()

    curtailment[label] = (((avail - used) / avail.replace(0.0, np.nan)) * 100).round(3)

    return curtailment


def calculate_energy(n, label, energy):
    for c in n.iterate_components(n.one_port_components | n.branch_components):
        if c.df.empty:
            continue

        if c.name in n.one_port_components:
            p_df = getattr(c.pnl, "p", None)
            if p_df is None or p_df.empty:
                continue

            cols = p_df.columns.intersection(c.df.index)
            if cols.empty:
                continue

            c_energies = (
                p_df[cols]
                .multiply(n.snapshot_weightings.generators, axis=0)
                .sum()
                .multiply(c.df.loc[cols, "sign"])
                .groupby(c.df.loc[cols, "carrier"])
                .sum()
            )
        else:
            c_energies = pd.Series(0.0, index=pd.Index(c.df.carrier.unique(), name=None))

            for port in [col[3:] for col in c.df.columns if col.startswith("bus")]:
                pport = getattr(c.pnl, "p" + port, None)
                if pport is None or pport.empty:
                    continue

                totals = pport.multiply(n.snapshot_weightings.generators, axis=0).sum()

                # remove values where bus is missing (bug in nomopyomo)
                no_bus = c.df.index[c.df["bus" + port] == ""]
                valid = _safe_index(totals, no_bus)

                # IMPORTANT: only touch indices that actually exist (prevents KeyError)
                if len(valid) > 0:
                    # fall back to link p0 sum if available; else set 0
                    if c.name == "Link" and getattr(n, "links_t", None) is not None and getattr(n.links_t, "p0", None) is not None:
                        totals.loc[valid] = float(n.links_t.p0.loc[:, valid].sum().sum())
                    else:
                        totals.loc[valid] = 0.0

                c_energies = c_energies - totals.groupby(c.df.carrier).sum()

        c_energies = pd.concat([c_energies], keys=[c.list_name])

        energy = energy.reindex(c_energies.index.union(energy.index))
        energy.loc[c_energies.index, label] = c_energies

    return energy


def calculate_supply(n, label, supply):
    """Max dispatch at buses aggregated by bus carrier."""
    bus_carriers = n.buses.carrier.unique()

    for bc in bus_carriers:
        bus_map = (n.buses.carrier == bc).copy()
        if "" in bus_map.index:
            bus_map.at[""] = False

        # one-port components
        for c in n.iterate_components(n.one_port_components):
            if c.df.empty:
                continue
            p_df = getattr(c.pnl, "p", None)
            if p_df is None or p_df.empty:
                continue

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]
            items = _safe_cols(p_df, items)
            if items.empty:
                continue

            s = (
                p_df[items]
                .max()
                .multiply(c.df.loc[items, "sign"])
                .groupby(c.df.loc[items, "carrier"])
                .sum()
            )
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[bc])

            supply = supply.reindex(s.index.union(supply.index))
            supply.loc[s.index, label] = s

        # branch components
        for c in n.iterate_components(n.branch_components):
            if c.df.empty:
                continue

            for end in [col[3:] for col in c.df.columns if col.startswith("bus")]:
                p_df = getattr(c.pnl, "p" + end, None)
                if p_df is None or p_df.empty:
                    continue

                items = c.df.index[c.df["bus" + end].map(bus_map).fillna(False)]
                items = _safe_cols(p_df, items)
                if items.empty:
                    continue

                s = (-1) ** (1 - int(end)) * (
                    (-1) ** int(end) * p_df[items]
                ).max().groupby(c.df.loc[items, "carrier"]).sum()
                s.index = s.index + end

                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[bc])

                supply = supply.reindex(s.index.union(supply.index))
                supply.loc[s.index, label] = s

    return supply


def calculate_supply_energy(n, label, supply_energy):
    """Total energy supply/consumption at buses aggregated by bus carrier."""
    bus_carriers = n.buses.carrier.unique()

    for bc in bus_carriers:
        bus_map = (n.buses.carrier == bc).copy()
        if "" in bus_map.index:
            bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):
            if c.df.empty:
                continue
            p_df = getattr(c.pnl, "p", None)
            if p_df is None or p_df.empty:
                continue

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]
            items = _safe_cols(p_df, items)
            if items.empty:
                continue

            s = (
                p_df[items]
                .multiply(n.snapshot_weightings.generators, axis=0)
                .sum()
                .multiply(c.df.loc[items, "sign"])
                .groupby(c.df.loc[items, "carrier"])
                .sum()
            )
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[bc])

            supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))
            supply_energy.loc[s.index, label] = s

        for c in n.iterate_components(n.branch_components):
            if c.df.empty:
                continue

            for end in [col[3:] for col in c.df.columns if col.startswith("bus")]:
                p_df = getattr(c.pnl, "p" + end, None)
                if p_df is None or p_df.empty:
                    continue

                items = c.df.index[c.df[f"bus{str(end)}"].map(bus_map).fillna(False)]
                items = _safe_cols(p_df, items)
                if items.empty:
                    continue

                s = (-1.0) * p_df[items].multiply(n.snapshot_weightings.generators, axis=0).sum()
                s = s.groupby(c.df.loc[items, "carrier"]).sum()
                s.index = s.index + end

                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[bc])

                supply_energy = supply_energy.reindex(s.index.union(supply_energy.index))
                supply_energy.loc[s.index, label] = s

    return supply_energy


def calculate_nodal_supply_energy(n, label, nodal_supply_energy):
    """Total energy supply/consumption at buses aggregated by bus carrier and node."""
    bus_carriers = n.buses.carrier.unique()

    for bc in bus_carriers:
        bus_map = (n.buses.carrier == bc).copy()
        if "" in bus_map.index:
            bus_map.at[""] = False

        for c in n.iterate_components(n.one_port_components):
            if c.df.empty:
                continue
            p_df = getattr(c.pnl, "p", None)
            if p_df is None or p_df.empty:
                continue

            items = c.df.index[c.df.bus.map(bus_map).fillna(False)]
            items = _safe_cols(p_df, items)
            if items.empty:
                continue

            s = (
                pd.concat(
                    [
                        p_df[items]
                        .multiply(n.snapshot_weightings.generators, axis=0)
                        .sum()
                        .multiply(c.df.loc[items, "sign"]),
                        c.df.loc[items][["bus", "carrier"]],
                    ],
                    axis=1,
                )
                .groupby(by=["bus", "carrier"])
                .sum()
                .iloc[:, 0]
            )
            s = pd.concat([s], keys=[c.list_name])
            s = pd.concat([s], keys=[bc])

            nodal_supply_energy = nodal_supply_energy.reindex(s.index.union(nodal_supply_energy.index))
            nodal_supply_energy.loc[s.index, label] = s

        for c in n.iterate_components(n.branch_components):
            if c.df.empty:
                continue

            for end in [col[3:] for col in c.df.columns if col.startswith("bus")]:
                p_df = getattr(c.pnl, "p" + end, None)
                if p_df is None or p_df.empty:
                    continue

                items = c.df.index[c.df["bus" + str(end)].map(bus_map).fillna(False)]
                items = _safe_cols(p_df, items)
                if items.empty:
                    continue

                s = (
                    pd.concat(
                        [
                            (-1.0) * p_df[items].multiply(n.snapshot_weightings.generators, axis=0).sum(),
                            c.df.loc[items][["bus0", "carrier"]],
                        ],
                        axis=1,
                    )
                    .groupby(by=["bus0", "carrier"])
                    .sum()
                    .iloc[:, 0]
                )

                s.index = s.index.map(lambda x: (x[0], x[1] + end))
                s = pd.concat([s], keys=[c.list_name])
                s = pd.concat([s], keys=[bc])

                nodal_supply_energy = nodal_supply_energy.reindex(s.index.union(nodal_supply_energy.index))
                nodal_supply_energy.loc[s.index, label] = s

    return nodal_supply_energy


def calculate_metrics(n, label, metrics):
    metrics_list = [
        "line_volume",
        "line_volume_limit",
        "line_volume_AC",
        "line_volume_DC",
        "line_volume_shadow",
        "co2_shadow",
    ]

    metrics = metrics.reindex(pd.Index(metrics_list).union(metrics.index))

    if not n.links.empty and "length" in n.links and "p_nom_opt" in n.links:
        metrics.at["line_volume_DC", label] = (n.links.length * n.links.p_nom_opt)[n.links.carrier == "DC"].sum()
    else:
        metrics.at["line_volume_DC", label] = 0.0

    if not n.lines.empty and "length" in n.lines and "s_nom_opt" in n.lines:
        metrics.at["line_volume_AC", label] = (n.lines.length * n.lines.s_nom_opt).sum()
    else:
        metrics.at["line_volume_AC", label] = 0.0

    metrics.at["line_volume", label] = metrics.loc[["line_volume_AC", "line_volume_DC"], label].sum()

    if "lv_limit" in n.global_constraints.index:
        metrics.at["line_volume_limit", label] = n.global_constraints.at["lv_limit", "constant"]
        metrics.at["line_volume_shadow", label] = n.global_constraints.at["lv_limit", "mu"]

    if "CO2Limit" in n.global_constraints.index:
        metrics.at["co2_shadow", label] = n.global_constraints.at["CO2Limit", "mu"]

    if "co2_sequestration_limit" in n.global_constraints.index:
        metrics.at["co2_storage_shadow", label] = n.global_constraints.at["co2_sequestration_limit", "mu"]

    return metrics


def calculate_prices(n, label, prices):
    prices = prices.reindex(prices.index.union(n.buses.carrier.unique()))
    if getattr(n, "buses_t", None) is None or getattr(n.buses_t, "marginal_price", None) is None or n.buses_t.marginal_price.empty:
        return prices
    prices[label] = n.buses_t.marginal_price.mean().groupby(n.buses.carrier).mean()
    return prices


def calculate_weighted_prices(n, label, weighted_prices):
    weighted_prices = weighted_prices.reindex(
        pd.Index(["electricity", "heat", "space heat", "urban heat", "space urban heat", "gas", "H2"])
    )

    if getattr(n, "buses_t", None) is None or getattr(n.buses_t, "marginal_price", None) is None or n.buses_t.marginal_price.empty:
        return weighted_prices

    link_loads = {
        "electricity": ["heat pump", "resistive heater", "battery charger", "H2 Electrolysis"],
        "heat": ["water tanks charger"],
        "urban heat": ["water tanks charger"],
        "space heat": [],
        "space urban heat": [],
        "gas": ["OCGT", "gas boiler", "CHP electric", "CHP heat"],
        "H2": ["Sabatier", "H2 Fuel Cell"],
    }

    for carrier, value in link_loads.items():
        if carrier == "electricity":
            suffix = ""
        elif carrier.startswith("space"):
            suffix = carrier[5:]
        else:
            suffix = " " + carrier

        buses = n.buses.index[n.buses.index.str[2:] == suffix]
        if buses.empty:
            continue

        if carrier in ["H2", "gas"]:
            load = pd.DataFrame(index=n.snapshots, columns=buses, data=0.0)
        else:
            load = n.loads_t.p_set[buses.intersection(n.loads.index)].copy()

        for tech in value:
            names = n.links.index[n.links.index.to_series().str.endswith(tech)]
            if not names.empty:
                load += n.links_t.p0[names].T.groupby(n.links.loc[names, "bus0"]).sum().T

        denom = load.sum().sum()
        if denom == 0.0:
            weighted_prices.loc[carrier, label] = np.nan
        else:
            weighted_prices.loc[carrier, label] = (load * n.buses_t.marginal_price[buses]).sum().sum() / denom

    return weighted_prices


def calculate_market_values(n, label, market_values):
    carrier = "AC"
    buses = n.buses.index[n.buses.carrier == carrier]

    if getattr(n, "buses_t", None) is None or getattr(n.buses_t, "marginal_price", None) is None or n.buses_t.marginal_price.empty:
        return market_values

    # generators
    if not n.generators.empty and getattr(n, "generators_t", None) is not None and getattr(n.generators_t, "p", None) is not None and not n.generators_t.p.empty:
        generators = n.generators.index[n.buses.loc[n.generators.bus, "carrier"] == carrier]
        techs = n.generators.loc[generators, "carrier"].value_counts().index
        market_values = market_values.reindex(market_values.index.union(techs))

        for tech in techs:
            gens = generators[n.generators.loc[generators, "carrier"] == tech]
            if gens.empty:
                continue
            cols = n.generators_t.p.columns.intersection(gens)
            if cols.empty:
                market_values.at[tech, label] = np.nan
                continue

            dispatch = (
                n.generators_t.p[cols]
                .T.groupby(n.generators.loc[cols, "bus"])
                .sum()
                .T.reindex(columns=buses, fill_value=0.0)
            )
            revenue = dispatch * n.buses_t.marginal_price[buses]
            total_dispatch = dispatch.sum().sum()
            market_values.at[tech, label] = revenue.sum().sum() / total_dispatch if total_dispatch else np.nan

    # links (port 0/1)
    if not n.links.empty and getattr(n, "links_t", None) is not None:
        for i in ["0", "1"]:
            p_df = getattr(n.links_t, "p" + i, None)
            if p_df is None or p_df.empty:
                continue

            all_links = n.links.index[n.buses.loc[n.links["bus" + i], "carrier"] == carrier]
            techs = n.links.loc[all_links, "carrier"].value_counts().index
            market_values = market_values.reindex(market_values.index.union(techs))

            for tech in techs:
                links = all_links[n.links.loc[all_links, "carrier"] == tech]
                cols = p_df.columns.intersection(links)
                if cols.empty:
                    market_values.at[tech, label] = np.nan
                    continue

                dispatch = (
                    p_df[cols]
                    .T.groupby(n.links.loc[cols, "bus" + i])
                    .sum()
                    .T.reindex(columns=buses, fill_value=0.0)
                )
                revenue = dispatch * n.buses_t.marginal_price[buses]
                total_dispatch = dispatch.sum().sum()
                market_values.at[tech, label] = revenue.sum().sum() / total_dispatch if total_dispatch else np.nan

    return market_values


def calculate_price_statistics(n, label, price_statistics):
    price_statistics = price_statistics.reindex(
        price_statistics.index.union(pd.Index(["zero_hours", "mean", "standard_deviation"]))
    )

    if getattr(n, "buses_t", None) is None or getattr(n.buses_t, "marginal_price", None) is None or n.buses_t.marginal_price.empty:
        return price_statistics

    buses = n.buses.index[n.buses.carrier == "AC"]
    if buses.empty:
        return price_statistics

    threshold = 0.1
    df0 = pd.DataFrame(data=0.0, columns=buses, index=n.snapshots)
    df0[n.buses_t.marginal_price[buses] < threshold] = 1.0

    price_statistics.at["zero_hours", label] = df0.sum().sum() / (df0.shape[0] * df0.shape[1])
    price_statistics.at["mean", label] = n.buses_t.marginal_price[buses].unstack().mean()
    price_statistics.at["standard_deviation", label] = n.buses_t.marginal_price[buses].unstack().std()

    return price_statistics


def make_summaries(networks_dict):
    outputs = [
        "nodal_costs",
        "nodal_capacities",
        "nodal_cfs",
        "cfs",
        "costs",
        "capacities",
        "curtailment",
        "energy",
        "supply",
        "supply_energy",
        "nodal_supply_energy",
        "prices",
        "weighted_prices",
        "price_statistics",
        "market_values",
        "metrics",
    ]

    columns = pd.MultiIndex.from_tuples(networks_dict.keys(), names=["cluster", "ll", "opt", "planning_horizon"])
    df = {output: pd.DataFrame(columns=columns, dtype=float) for output in outputs}

    for label, filename in networks_dict.items():
        logger.info("Make summary for scenario %s, using %s", label, filename)

        n = pypsa.Network(filename)
        assign_carriers(n)
        assign_locations(n)

        for output in outputs:
            fn = globals().get("calculate_" + output)
            if fn is None:
                logger.warning("Missing function calculate_%s; skipping output '%s'", output, output)
                continue
            try:
                df[output] = fn(n, label, df[output])
            except Exception:
                logger.exception("Failed calculating '%s' for label=%s", output, label)
                raise

    return df


def to_csv(df):
    for key in df:
        df[key].to_csv(snakemake.output[key])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake
        snakemake = mock_snakemake("make_summary")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    networks_dict = {
        (cluster, ll, opt + sector_opt, planning_horizon): (
            "results/"
            + snakemake.params.RDIR
            + f"/postnetworks/base_s_{cluster}_l{ll}_{opt}_{sector_opt}_{planning_horizon}.nc"
        )
        for cluster in snakemake.params.scenario["clusters"]
        for opt in snakemake.params.scenario["opts"]
        for sector_opt in snakemake.params.scenario["sector_opts"]
        for ll in snakemake.params.scenario["ll"]
        for planning_horizon in snakemake.params.scenario["planning_horizons"]
    }

    time = get_snapshots(snakemake.params.snapshots, snakemake.params.drop_leap_day)
    Nyears = len(time) / 8760

    # keep for side-effects / consistency with upstream workflow
    _costs_db = prepare_costs(
        snakemake.input.costs,
        snakemake.params.costs,
        Nyears,
    )

    df = make_summaries(networks_dict)

    df["metrics"].loc["total costs"] = df["costs"].sum()

    to_csv(df)

    if snakemake.params.foresight == "myopic":
        cumulative_cost = calculate_cumulative_cost()
        cumulative_cost.to_csv("results/" + snakemake.params.RDIR + "csvs/cumulative_cost.csv")
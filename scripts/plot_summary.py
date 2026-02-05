# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2020-2024 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT
"""
Creates plots from summary CSV files.
"""

import logging

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

from _helpers import configure_logging, set_scenario_config
from prepare_sector_network import co2_emissions_year

logger = logging.getLogger(__name__)
plt.style.use("ggplot")


# consolidate and rename
def rename_techs(label: str) -> str:
    prefix_to_remove = [
        "residential ",
        "services ",
        "urban ",
        "rural ",
        "central ",
        "decentral ",
    ]

    rename_if_contains = [
        "CHP",
        "gas boiler",
        "biogas",
        "solar thermal",
        "air heat pump",
        "ground heat pump",
        "resistive heater",
        "Fischer-Tropsch",
    ]

    rename_if_contains_dict = {
        "water tanks": "hot water storage",
        "retrofitting": "building retrofitting",
        "battery": "battery storage",
        "H2 for industry": "H2 for industry",
        "land transport fuel cell": "land transport fuel cell",
        "land transport oil": "land transport oil",
        "oil shipping": "shipping oil",
    }

    rename = {
        "solar": "solar PV",
        "Sabatier": "methanation",
        "offwind": "offshore wind",
        "offwind-ac": "offshore wind (AC)",
        "offwind-dc": "offshore wind (DC)",
        "offwind-float": "offshore wind (Float)",
        "onwind": "onshore wind",
        "ror": "hydroelectricity",
        "hydro": "hydroelectricity",
        "PHS": "hydroelectricity",
        "NH3": "ammonia",
        "co2 Store": "DAC",
        "co2 stored": "CO2 sequestration",
        "AC": "transmission lines",
        "DC": "transmission lines",
        "B2B": "transmission lines",
    }

    for ptr in prefix_to_remove:
        if label.startswith(ptr):
            label = label[len(ptr) :]

    for rif in rename_if_contains:
        if rif in label:
            label = rif

    for old, new in rename_if_contains_dict.items():
        if old in label:
            label = new

    for old, new in rename.items():
        if old == label:
            label = new

    return label


preferred_order = pd.Index(
    [
        "transmission lines",
        "hydroelectricity",
        "hydro reservoir",
        "run of river",
        "pumped hydro storage",
        "solid biomass",
        "biogas",
        "onshore wind",
        "offshore wind",
        "offshore wind (AC)",
        "offshore wind (DC)",
        "solar PV",
        "solar thermal",
        "solar rooftop",
        "solar",
        "building retrofitting",
        "ground heat pump",
        "air heat pump",
        "heat pump",
        "resistive heater",
        "power-to-heat",
        "gas-to-power/heat",
        "CHP",
        "OCGT",
        "gas boiler",
        "gas",
        "natural gas",
        "methanation",
        "ammonia",
        "hydrogen storage",
        "power-to-gas",
        "power-to-liquid",
        "battery storage",
        "hot water storage",
        "CO2 sequestration",
    ]
)


def _read_summary_csv(path: str, *, index_levels: int, n_header: int) -> pd.DataFrame:
    """
    Robust reader for PyPSA-Eur summary CSVs which are multi-index rows and multi-index columns.
    """
    return pd.read_csv(path, index_col=list(range(index_levels)), header=list(range(n_header)))


def _safe_colors_for(index: pd.Index) -> list:
    """
    Return a color list for the given tech index, without KeyError if tech_colors misses entries.
    """
    tc = snakemake.params.plotting.get("tech_colors", {})
    fallback = snakemake.params.plotting.get("fallback_color", "#808080")
    return [tc.get(i, fallback) for i in index]


def plot_costs():
    cost_df = _read_summary_csv(snakemake.input.costs, index_levels=3, n_header=n_header)

    # original costs.csv is (component, cost_type, carrier) on rows -> group by carrier-level
    df = cost_df.groupby(cost_df.index.get_level_values(2)).sum()

    # convert to billions
    df = df / 1e9

    df = df.groupby(df.index.map(rename_techs)).sum()

    thr = snakemake.params.plotting["costs_threshold"]
    to_drop = df.index[df.max(axis=1) < thr]

    logger.info(
        "Dropping technology with costs below %s EUR billion per year",
        thr,
    )
    logger.debug(df.loc[to_drop])

    df = df.drop(to_drop)

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_axis_off()
        fig.savefig(snakemake.output.costs, bbox_inches="tight")
        plt.close(fig)
        return

    logger.info("Total system cost of %s EUR billion per year", round(df.sum().iloc[0]))

    new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))

    fig, ax = plt.subplots(figsize=(12, 8))
    df.loc[new_index].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=_safe_colors_for(new_index),
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()

    ax.set_ylim([0, snakemake.params.plotting["costs_max"]])
    ax.set_ylabel("System Cost [EUR billion per year]")
    ax.set_xlabel("")
    ax.grid(axis="x")
    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)

    fig.savefig(snakemake.output.costs, bbox_inches="tight")
    plt.close(fig)


def plot_energy():
    energy_df = _read_summary_csv(snakemake.input.energy, index_levels=2, n_header=n_header)

    # original energy.csv is (component, carrier) on rows -> group by carrier-level
    df = energy_df.groupby(energy_df.index.get_level_values(1)).sum()

    # convert MWh to TWh
    df = df / 1e6

    df = df.groupby(df.index.map(rename_techs)).sum()

    thr = snakemake.params.plotting["energy_threshold"]
    to_drop = df.index[df.abs().max(axis=1) < thr]

    logger.info(
        "Dropping all technology with energy consumption or production below %s TWh/a",
        thr,
    )
    logger.debug(df.loc[to_drop])

    df = df.drop(to_drop)

    if df.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_axis_off()
        fig.savefig(snakemake.output.energy, bbox_inches="tight")
        plt.close(fig)
        return

    logger.info("Total energy of %s TWh/a", round(df.sum().iloc[0]))

    new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))

    fig, ax = plt.subplots(figsize=(12, 8))
    df.loc[new_index].T.plot(
        kind="bar",
        ax=ax,
        stacked=True,
        color=_safe_colors_for(new_index),
    )

    handles, labels = ax.get_legend_handles_labels()
    handles.reverse()
    labels.reverse()

    ax.set_ylim([snakemake.params.plotting["energy_min"], snakemake.params.plotting["energy_max"]])
    ax.set_ylabel("Energy [TWh/a]")
    ax.set_xlabel("")
    ax.grid(axis="x")
    ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)

    fig.savefig(snakemake.output.energy, bbox_inches="tight")
    plt.close(fig)


def plot_balances():
    co2_carriers = ["co2", "co2 stored", "process emissions"]

    balances_df = _read_summary_csv(snakemake.input.balances, index_levels=3, n_header=n_header)

    balances = {i.replace(" ", "_"): [i] for i in balances_df.index.levels[0]}
    balances["energy"] = [i for i in balances_df.index.levels[0] if i not in co2_carriers]

    for k, v in balances.items():
        df = balances_df.loc[v]
        df = df.groupby(df.index.get_level_values(2)).sum()

        # convert MWh to TWh (or MtCO2/a later)
        df = df / 1e6

        # remove trailing link ports (e.g. "H2 pipeline0")
        df.index = [
            (
                i[:-1]
                if (
                    (i not in ["co2", "NH3", "H2"])
                    and len(i) > 0
                    and (i[-1:] in ["0", "1", "2", "3", "4"])
                )
                else i
            )
            for i in df.index
        ]

        df = df.groupby(pd.Index(df.index).map(rename_techs)).sum()

        thr = snakemake.params.plotting["energy_threshold"] / 10
        to_drop = df.index[df.abs().max(axis=1) < thr]

        units = "MtCO2/a" if v[0] in co2_carriers else "TWh/a"
        logger.debug("Dropping technology energy balance smaller than %s %s", thr, units)
        logger.debug(df.loc[to_drop])

        df = df.drop(to_drop)

        if df.empty:
            continue

        new_index = preferred_order.intersection(df.index).append(df.index.difference(preferred_order))
        new_columns = df.columns.sort_values()

        fig, ax = plt.subplots(figsize=(12, 8))
        df.loc[new_index, new_columns].T.plot(
            kind="bar",
            ax=ax,
            stacked=True,
            color=_safe_colors_for(new_index),
        )

        handles, labels = ax.get_legend_handles_labels()
        handles.reverse()
        labels.reverse()

        ax.set_ylabel("CO2 [MtCO2/a]" if v[0] in co2_carriers else "Energy [TWh/a]")
        ax.set_xlabel("")
        ax.grid(axis="x")
        ax.legend(handles, labels, ncol=1, loc="upper left", bbox_to_anchor=[1, 1], frameon=False)

        fig.savefig(snakemake.output.balances[:-10] + k + ".svg", bbox_inches="tight")
        plt.close(fig)


def historical_emissions(countries, options):
    """
    Read historical emissions to add them to the carbon budget plot.
    """
    df = pd.read_csv(snakemake.input.co2, encoding="latin-1", low_memory=False)
    df.loc[df["Year"] == "1985-1987", "Year"] = 1986
    df["Year"] = df["Year"].astype(int)
    df = df.set_index(["Year", "Sector_name", "Country_code", "Pollutant_name"]).sort_index()

    e = pd.Series(dtype=object)
    e["electricity"] = "1.A.1.a - Public Electricity and Heat Production"
    e["residential non-elec"] = "1.A.4.b - Residential"
    e["services non-elec"] = "1.A.4.a - Commercial/Institutional"
    e["rail non-elec"] = "1.A.3.c - Railways"
    e["road non-elec"] = "1.A.3.b - Road Transportation"
    e["domestic navigation"] = "1.A.3.d - Domestic Navigation"
    e["international navigation"] = "1.D.1.b - International Navigation"
    e["domestic aviation"] = "1.A.3.a - Domestic Aviation"
    e["international aviation"] = "1.D.1.a - International Aviation"
    e["total energy"] = "1 - Energy"
    e["industrial processes"] = "2 - Industrial Processes and Product Use"
    e["agriculture"] = "3 - Agriculture"
    e["LULUCF"] = "4 - Land Use, Land-Use Change and Forestry"
    e["waste management"] = "5 - Waste management"
    e["other"] = "6 - Other Sector"
    e["indirect"] = "ind_CO2 - Indirect CO2"
    e["other LULUCF"] = "4.H - Other LULUCF"

    pol = ["CO2"]

    # normalize UK code handling
    countries = list(countries)
    if "GB" in countries:
        countries.remove("GB")
        countries.append("UK")

    year = df.index.levels[0][df.index.levels[0] >= 1990]

    missing = pd.Index(countries).difference(df.index.levels[2])
    if not missing.empty:
        logger.warning(
            "The following countries are missing and not considered when plotting historic CO2 emissions: %s",
            missing,
        )
        countries = pd.Index(df.index.levels[2]).intersection(countries)

    co2_totals = (
        df.loc[idx[year, e.values, countries, pol], "emissions"]
        .unstack("Year")
        .rename(index=pd.Series(e.index, e.values))
    )

    # GtCO2
    co2_totals = (1 / 1e6) * co2_totals.groupby(level=0, axis=0).sum()

    # derive industrial non-elec as residual
    co2_totals.loc["industrial non-elec"] = (
        co2_totals.loc["total energy"]
        - co2_totals.loc[
            [
                "electricity",
                "services non-elec",
                "residential non-elec",
                "road non-elec",
                "rail non-elec",
                "domestic aviation",
                "international aviation",
                "domestic navigation",
                "international navigation",
            ]
        ].sum()
    )

    emissions = co2_totals.loc["electricity"].copy()

    # options is a dict like {"transport": bool, "heating": bool, "industry": bool}
    if options.get("transport", False):
        emissions += co2_totals.loc[[i + " non-elec" for i in ["rail", "road"]]].sum()
    if options.get("heating", False):
        emissions += co2_totals.loc[[i + " non-elec" for i in ["residential", "services"]]].sum()
    if options.get("industry", False):
        emissions += co2_totals.loc[
            [
                "industrial non-elec",
                "industrial processes",
                "domestic aviation",
                "international aviation",
                "domestic navigation",
                "international navigation",
            ]
        ].sum()

    return emissions


def plot_carbon_budget_distribution(input_eurostat, options):
    """
    Plot historical carbon emissions in the EU and decarbonization path.
    """
    # NOTE: seaborn is optional; don't hard-fail if it's missing in your env
    try:
        import seaborn as sns

        sns.set()
        sns.set_style("ticks")
    except Exception:
        logger.warning("seaborn not available; continuing without seaborn styling")

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20

    emissions_scope = snakemake.params.emissions_scope
    input_co2 = snakemake.input.co2

    countries = snakemake.params.countries

    e_1990 = co2_emissions_year(
        countries,
        input_eurostat,
        options,
        emissions_scope,
        input_co2,
        year=1990,
    )

    emissions = historical_emissions(countries, options)

    # add other years
    for y, v in [(2019, 3.414362), (2020, 3.092434), (2021, 3.290418), (2022, 3.213025)]:
        emissions.loc[y] = v

    if snakemake.config["foresight"] == "myopic":
        path_cb = "results/" + snakemake.params.RDIR + "/csvs/"
        co2_cap = pd.read_csv(path_cb + "carbon_budget_distribution.csv", index_col=0)[["cb"]]
        co2_cap *= e_1990
    else:
        supply_energy = pd.read_csv(
            snakemake.input.balances, index_col=[0, 1, 2], header=list(range(n_header))
        )
        # balances.csv: (bus_carrier, component, carrier) -> take "co2" block
        co2_cap = supply_energy.loc["co2"].droplevel(0).drop("co2").sum().unstack().T / 1e9
        co2_cap.rename(index=lambda x: int(x), inplace=True)

    plt.figure(figsize=(10, 7))
    gs1 = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs1[0, 0])
    ax1.set_ylabel("CO$_2$ emissions \n [Gt per year]", fontsize=22)
    ax1.set_xlim([1990, snakemake.params.planning_horizons[-1] + 1])

    ax1.plot(emissions, color="black", linewidth=3, label=None)

    # committed/indicative targets
    ax1.plot([2020], [0.8 * emissions[1990]], marker="*", markersize=12, color="black")
    ax1.plot([2030], [0.45 * emissions[1990]], marker="*", markersize=12, color="black")
    ax1.plot([2030], [0.6 * emissions[1990]], marker="*", markersize=12, color="black")

    ax1.plot(
        [2050, 2050],
        [x * emissions[1990] for x in [0.2, 0.05]],
        color="gray",
        linewidth=2,
        marker="_",
        alpha=0.5,
    )
    ax1.plot([2050], [0.0 * emissions[1990]], marker="*", markersize=12, color="black", label="EU committed target")

    for col in co2_cap.columns:
        ax1.plot(co2_cap[col], linewidth=3, label=str(col))

    ax1.legend(fancybox=True, fontsize=18, loc=(0.01, 0.01), facecolor="white", frameon=True)
    plt.grid(axis="y")

    path = snakemake.output.balances.split("balances")[0] + "carbon_budget.svg"
    plt.savefig(path, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake("plot_summary")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # IMPORTANT: must match your summary CSV writing
    n_header = 4

    plot_costs()
    plot_energy()
    plot_balances()

    co2_budget = snakemake.params["co2_budget"]
    if (isinstance(co2_budget, str) and co2_budget.startswith("cb")) or snakemake.params["foresight"] == "perfect":
        options = snakemake.params.sector
        plot_carbon_budget_distribution(snakemake.input.eurostat, options)
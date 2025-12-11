import numpy as np
import random
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pypsa
import shapely
import warnings
import xarray as xr
import itertools
import geopandas as gpd

from geopandas import GeoDataFrame
from pypsa.plot import add_legend_patches
from shapely.errors import ShapelyDeprecationWarning
from mpl_toolkits.axes_grid1 import make_axes_locatable

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
cm = 1/2.54
bg_colour = '#f0f0f0'
custom_params = {'xtick.bottom': True, 'axes.edgecolor': 'black', 'axes.spines.right': False, 'axes.spines.top': False, 'mathtext.default': 'regular'}
sns.set_theme(style='ticks', rc=custom_params)
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["mathtext.default"] = "it"
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['legend.title_fontsize']=9
plt.rc('xtick', labelsize=9) 
plt.rc('ytick', labelsize=9) 

################################################################################
################################# PLOTTING #####################################
################################################################################

# PLOTTING FUNCTIONS
def plot_stacked(var,results, filter, colours, name):
    fig, ax = plt.subplots(figsize=(5, 7))
    
    if var ==  "Optimal Capacity":
        save = "p_opt"
        unit = " [GW]"
    elif var ==  "Energy Balance":
        save = "e_opt"
        unit = " [TWh]"
    elif var == "Capacity Factor":
        save = "cf_opt"
        unit = ""
        
    label = f"{var}{unit}"
    results[var].filter(regex=filter, axis=1).plot(kind='barh', stacked=True,legend=False, color=colours.values, ylabel="",xlabel=label,ax=ax)
    
    labels_y = list(results['region_max'].map('{:,.0f}'.format)) # add comma to number
    ax.set_yticklabels(labels_y)
    
    splits = results["region_max"].unique()
    models = results["wake_model"].unique()
    
    class_lines_y = np.arange(-0.5,len(models)*len(splits),len(splits))
    # print(class_lines_y)
    class_labels_y = np.arange(np.floor(len(splits)/2),len(models)*len(splits),len(splits))
    
    # labels classes:
    sec = ax.secondary_yaxis(location=0)
    sec.set_yticks(class_labels_y, labels=nice_names_for_plotting_label(models))
    sec.tick_params('y', length=60, width=0)
    sec.set_ylabel(r'Wake Model | Spatial Resolution ($A_{region}^{max}$) [km$\mathrm{^{2}}$]')
    
    # lines between the classes:
    sec2 = ax.secondary_yaxis(location=0)
    sec2.set_yticks(class_lines_y, labels=[])
    sec2.tick_params('y', length=120, width=1.5)
    
    x_max = results[var].filter(regex=filter, axis=1).mean(axis=0).sum()
    ax.hlines(y=class_lines_y[:-2]+len(splits), xmin=0, xmax=x_max, linewidth=1.5, color='r',linestyles='--')
    handles, labels = plt.gca().get_legend_handles_labels()
    
    if 'singlewind' in name:
        labels = ['Offshore Wind (Combined)']

    plt.figlegend(handles,[word.title() for word in labels],loc = 'upper center', bbox_to_anchor=(0.5, 0), ncol=3, title='Carrier', frameon=False)
    plt.savefig('plots/'+name+'_'+save+'_stacked.png', bbox_inches='tight')

def plot_stacked_multi(clusters,models,splits,var,filter, prefix_list):
    i=0
    fig, ax = plt.subplots(1,3,figsize=(16.4*cm, 18*cm),dpi=600,sharey=True,sharex=True)
    for prefix in prefix_list:
        results, colours = results_dataframe(clusters,models,splits,prefix)
        
        if "combined" in prefix:
            legend_labels = ['Combined']
        else:
            legend_labels = ['AC', 'DC','Floating']

        if var ==  "Optimal Capacity":
            save = "p_opt"
            unit = " [GW]"
        elif var ==  "Energy Balance":
            save = "e_opt"
            unit = " [TWh]"
        elif var == "Capacity Factor":
            save = "cf_opt"
            unit = ""
            
        label = f"{var}{unit}"
        results[var].filter(regex=filter, axis=1).plot(
            kind='barh', stacked=True,legend=True, color=colours.values, 
            ylabel="",xlabel=label,
            ax=ax[i]
            )
        ax[i].legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.06), ncol=1, title='Carrier', labels=legend_labels, frameon=False)
        

        labels_y = list(results['region_max'].map('{:,.0f}'.format)) # add comma to number
        ax[i].set_yticklabels(labels_y)

        splits_u = results["region_max"].unique()
        models_u = results["wake_model"].unique()

        class_lines_y = np.arange(-0.5,len(splits_u)*len(splits_u),len(splits_u))
        class_labels_y = np.arange(np.floor(len(splits_u)/2),len(models_u)*len(splits_u),len(splits_u))+2.5

        if i == 0:
            # labels classes:
            sec = ax[i].secondary_yaxis(location=0)
            sec.set_yticks(class_labels_y, labels=nice_names_for_plotting_label(models_u))
            sec.tick_params('y', length=0, width=0,labelrotation=90,pad=50)
            sec.set_ylabel(r'Wake Model | Spatial Resolution ($A_{region}^{max}$) [km$\mathrm{^{2}}$]')
        
            # lines between the classes:
            sec2 = ax[i].secondary_yaxis(location=0)
            sec2.set_yticks(class_lines_y, labels=[])
            sec2.tick_params('y', length=50, width=1.5)
        
        x_max = results[var].filter(regex=filter, axis=1).mean(axis=0).sum()
        
    ax[i].hlines(y=class_lines_y[:3]+len(splits), xmin=0, xmax=x_max, linewidth=1.5*cm, color='r',linestyles='--')
    i+=1
    
    fig.savefig('plots/'+'2030-10-northsea_all'+'_'+save+'_stacked.png', bbox_inches='tight')
    fig.savefig(
        'plots/'+'2030-10-northsea_standard'+'_'+save+'_stacked.png',
        # we need a bounding box in inches
        bbox_inches=mtransforms.Bbox(
            # This is in "figure fraction" for the bottom half
            # input in [[xmin, ymin], [xmax, ymax]]
            [[0, 0], [0.43, 1]]
        ).transformed(
            (fig.transFigure - fig.dpi_scale_trans)
        ),
    )
    fig.savefig(
        'plots/'+'2030-10-northsea_dominant'+'_'+save+'_stacked.png',
        bbox_inches=mtransforms.Bbox([[0.415, 0], [0.71, 1]]).transformed(
            fig.transFigure - fig.dpi_scale_trans
        ),
    )
    fig.savefig(
        'plots/'+'2030-10-northsea_combined'+'_'+save+'_stacked.png',
        bbox_inches=mtransforms.Bbox([[0.71, 0], [1, 1]]).transformed(
            fig.transFigure - fig.dpi_scale_trans
        ),
    )

def plot_capacities_split(results,filter,name,var):
    carriers = results[var].filter(regex=filter, axis=1).columns.values
    fig, axes = plt.subplots(1, len(carriers), figsize=(4.3*len(carriers), 4))
    for carrier, i in zip(carriers, range(len(carriers))):
    
        if i == range(len(carriers))[-1]:
            sns.lineplot(
                x='region_max',
                y= results[var][carrier],
                hue ="wake_model",
                style="wake_model",
                hue_order = ['base', 'standard', 'ellyess', 'glaum'],
                style_order= ['base', 'standard', 'ellyess', 'glaum'],
                data = results,
                ax = axes[i],
                legend = True
            )
        
        else:
            sns.lineplot(
                x='region_max',
                y= results[var][carrier],
                hue ="wake_model",
                style="wake_model",
                hue_order = ['base', 'standard', 'ellyess', 'glaum'],
                style_order= ['base', 'standard', 'ellyess', 'glaum'],
                data = results,
                ax = axes[i],
                legend = False
            )
        
        axes[i].set_xlabel(r'Maximum Region Size (km$\mathrm{^{2}}$)')
        axes[i].set_ylabel('Optimal Capacity (GW)')
        axes[i].set_title(carrier)
        axes[i].set_xlim(axes[i].get_xlim()[::-1])

    axes[range(len(carriers))[-1]].get_legend().remove()
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Base', 'Standard',  'New', 'Glaum et al.']
    plt.figlegend(handles, labels, loc = 'center right', bbox_to_anchor=(1.1, 0.5), ncol=1, title='Wake Model', frameon=False)
    plt.tight_layout()
    # plt.savefig('plots/'+name+'_capacity_split.png', bbox_inches='tight')
    
def plot_generation_series(clusters, models,splits, prefix):
    """Plot generation output series 

    Args:
        clusters (_type_): _description_
        models (_type_): _description_
        splits (_type_): _description_
        prefix (_type_): _description_
    """
    scenarios = scenario_list(models,splits)
    for scenario in scenarios:
        n = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")
        color = n.carriers.set_index("nice_name").color.where(
        lambda s: s != "", "lightgrey"
        )
        color = color[n.statistics.supply(comps=["Generator"], aggregate_time=False).droplevel(0).div(1e3).T.columns.get_level_values("carrier")]
        
        fig, ax = plt.subplots(figsize=(13, 4))
        
        n.statistics.supply(comps=["Generator"], aggregate_time=False).droplevel(0).div(1e3).T.plot.area(
            title="Generation in GW",
            ax=ax,
            legend=False,
            linewidth=0,
            color=color
        )
        ax.legend(bbox_to_anchor=(1, 0), loc="lower left", title=None, ncol=1)
        
def plot_regional_carrier_percentage(prefix,scenario_list):
    fig, ax = plt.subplots(1,3,figsize=(16.4*cm, 5*cm),dpi=600,sharex=True, sharey=True,layout='constrained')
    i = 0
    for scenario in scenario_list:
        generators = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_10_lvopt___2030.nc").generators.filter(regex="offwind", axis=0).copy()
        generators["region"] = generators.index.to_series().str.split(' ').str[:2].str.join(sep=" ")
        regions = gpd.read_file("wake_extra/northsea/regions_offshore_"+scenario.split('-')[1]+".geojson").set_index('name')
        generators = generators.merge(regions, right_index=True, left_on="region")
        # generators = generators[generators['p_nom_max']>=100]
        
        total = generators.groupby('region')['p_nom_opt'].transform('sum')
        generators['% of capacity'] = generators['p_nom_opt'].div(total)
        generators['% of capacity'] = generators['% of capacity'] * 100
        generators = generators.sort_values('% of capacity').drop_duplicates(['region'], keep='last')
        geo_df = df_to_geodf(generators, geom_col="geometry", crs="4326")
        
        carriers = {
            'offwind-ac': 'Reds',
            'offwind-dc': 'Blues',
            'offwind-float': 'Greens'
        }

        j=0
        for carrier, color in carriers.items():
            if carrier == 'offwind-ac':
                carrier_nice = 'AC'
            elif carrier == 'offwind-dc':
                carrier_nice = 'DC'
            elif carrier == 'offwind-float':
                carrier_nice = 'Floating'

            if j == 0 and i == 2:
                legend_value = True
            elif j == 1 and i == 2:
                legend_value = True
            elif j == 2 and i == 2:
                legend_value = True
            else:
                legend_value = False
                
            geo_df[geo_df['carrier'] == carrier].plot(
                column='% of capacity', 
                cmap=color,
                legend=legend_value, 
                ax=ax[i],
                vmin=0,vmax=100,
                legend_kwds={
                    'label': str(carrier_nice)+' % of ' + r"$P^{opt}_{nom}$",
                    "orientation": "vertical",
                    # 'fraction': 0.05,
                    'pad': 0.1,
                    'shrink': 1,
                    },
                linewidth=0.1,
                )
            j+=1
        
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()
        i += 1
    plt.savefig('plots/'+prefix+'_region_dominant_perc.png', bbox_inches='tight')

def plot_region_optimal_capacity(carrier,clusters,scenarios,prefix,vmin=None,vmax=None):
    fig, ax = plt.subplots(1,4,figsize=(16.4*cm, 4*cm),dpi=600,sharex=True, sharey=True,layout='constrained')
            
    for scenario, i in zip(scenarios, range(len(scenarios))):
        generators = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc").generators.filter(regex=carrier, axis=0).copy()
        generators["region"] = generators.index.to_series().str.split(' ').str[:2].str.join(sep=" ")
        # regions = gpd.read_file("wake_extra/"+prefix+"/regions_offshore_"+scenario.split('-')[1]+".geojson")
        regions = gpd.read_file("wake_extra/"+prefix.split('-')[2]+"/regions_offshore_"+scenario.split('-')[1]+".geojson")
        regions["region"] = regions["name"]
        
        df = generators.groupby("region").agg({"p_nom_opt": np.sum,})
        df = df.merge(regions, on="region")
        geo_df = df_to_geodf(df, geom_col="geometry", crs="4326")
        
        if i == range(len(scenarios))[-1]:
            legend_value = True
        else:
            legend_value = False
            
        geo_df.plot(
            column="p_nom_opt",
            # edgecolor="black",
            ax=ax[i],
            cmap='viridis',
            vmin = vmin, 
            vmax = vmax,
            legend=legend_value,
            linewidth=0.1,
            legend_kwds={
                'label': "Capacity ($P^{opt}_{nom}$) [MW]",
                "orientation": "vertical",
                # 'fraction': 0.1,
                'pad': 0.1,
                'shrink': 1,
            })
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()
        
    plt.savefig('plots/'+prefix+'_region_optimal_capacity.png', bbox_inches='tight')
    
def plot_region_optimal_density(carrier,clusters,scenarios,prefix,vmin=None,vmax=None):
    fig, ax = plt.subplots(1,4,figsize=(16.4*cm, 4.5*cm),dpi=600,sharex=True, sharey=True,layout='constrained')
            
    for scenario, i in zip(scenarios, range(len(scenarios))):
        generators = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc").generators.filter(regex=carrier, axis=0).copy()
        generators["region"] = generators.index.to_series().str.split(' ').str[:2].str.join(sep=" ")
        # regions = gpd.read_file("wake_extra/"+prefix+"/regions_offshore_"+scenario.split('-')[1]+".geojson")
        regions = gpd.read_file("wake_extra/"+prefix.split('-')[2]+"/regions_offshore_"+scenario.split('-')[1]+".geojson")
        regions["region"] = regions["name"]

        df = generators.groupby("region").agg({"p_nom_opt": np.sum})
        df = df.merge(regions, on="region")
        geo_df = df_to_geodf(df, geom_col="geometry", crs="4326")
        geo_df["icd"] = geo_df["p_nom_opt"]/geo_df["area"]
        
        if i == range(len(scenarios))[-1]:
            legend_value = True
        else:
            legend_value = False
            
        geo_df.plot(
            column="icd",
            # edgecolor="black",
            ax=ax[i],
            cmap='viridis',
            vmin = vmin, 
            vmax = vmax,
            legend=legend_value,
            linewidth=0.1,
            legend_kwds={
                'label': r"$\rho_{A_{region}}^{opt}$ [MW/km$\mathrm{^{2}}$]",
                "orientation": "vertical",
                # 'fraction': 0.1,
                'pad': 0.2,
                'shrink': 1,
            })
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()
        
    plt.savefig('plots/'+prefix+'_region_optimal_density.png', bbox_inches='tight')

def plot_distribution(wake_order, fixed, prefix):
    if fixed == "none":
        dist_df = result_dist
    elif fixed == "build-out":
        dist_df = build_dist
    elif fixed == "CF":
        dist_df = cf_dist
        
    g = sns.catplot(
        data = dist_df,y='value',x='region_max',orient='x',
        hue="wake_model",hue_order=wake_order,col="variable",
        kind="violin",split=False,cut=0,inner='quarter',
        height=6*cm,aspect=1.5,native_scale=False,
        sharey=False,sharex=True,
        legend=True,legend_out=True
    )
    i = 0
    for ax in g.axes.flat:
        ax.set_xlabel(r'Spatial Resolution ($A_{region}^{max}$) [km$\mathrm{^{2}}$]')
        if i == 0:
            ax.set_ylabel(r'Capacity Factor ($\mathrm{CF}$)')
        else:
            ax.set_ylabel(r'Curtailment ($\mathrm{F}_{curtail}$)')
        i+=1
    g._legend.set_title('Wake Model')
    g.set_titles("")
    plt.savefig('plots/'+prefix+'_ts_dist_'+fixed+'.png', bbox_inches='tight',dpi=600)
    
def plot_region_capacity_density(carrier,clusters,scenarios,prefix,vmin=None,vmax=None):
    fig, ax = plt.subplots(1,4,figsize=(16.4*cm, 5*cm),dpi=600,sharex=True, sharey=True,layout='tight')
            
    for scenario, i in zip(scenarios, range(len(scenarios))):
        generators = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc").generators.filter(regex=carrier, axis=0).copy()
        generators["region"] = generators.index.to_series().str.split(' ').str[:2].str.join(sep=" ")
        regions = gpd.read_file("wake_extra/"+prefix.split('-')[2]+"/regions_offshore_"+scenario.split('-')[1]+".geojson")
        regions["region"] = regions["name"]
        
        df = generators.groupby("region").agg({"p_nom_max": np.sum,})
        df = df.merge(regions, on="region")
        
        df["capacity_density"] = df["p_nom_max"]/df["area"]
        geo_df = df_to_geodf(df, geom_col="geometry", crs="4326")
        
        if i == range(len(scenarios))[-1]:
            legend_value = True
        else:
            legend_value = False
            
        geo_df.plot(
            column='capacity_density',
            ax=ax[i],
            cmap='Purples',
            # edgecolor="black",
            vmin=0,
            vmax=4,
            legend=legend_value,
            linewidth=0.1,
            legend_kwds={
                'label': r"$\rho_{A_{region}}^{opt}$ [MW/km$\mathrm{^{2}}$]",
                "orientation": "vertical",
                # 'fraction': 0.2,
                'pad': 0.1,
                'shrink': 1,
            })
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()
        
    plt.savefig('plots/'+prefix+'_region_capacity_density.png', bbox_inches='tight')

def plot_temporal_comp_data(results,filter,name,var):
    fig, axes = plt.subplots(figsize=(14*cm, 6*cm),dpi=600,layout='constrained')
    
    sns.lineplot(
        x='region_max',
        y= results[var],
        hue ="wake_model",
        style="temporal",
        hue_order = models,
        data = results,
        ax = axes,
        legend = True
    )

    if var ==  "Optimal Capacity":
        save = "p_opt"
        unit = " [GW]"
        symbol = r"($P^{opt}_{nom}$)"
    elif var ==  "Energy Balance":
        save = "e_opt"
        unit = " [TWh]"
    elif var == "Capacity Factor":
        save = "cf_opt"
        unit = ""

    axes.set_xlabel(r'Spatial Resolution ($A_{region}^{max}$) [km$\mathrm{^{2}}$]')
    axes.get_xaxis().set_major_formatter(
    mticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    axes.set_ylabel(f"{var} {symbol} {unit}")
    axes.set_xlim(axes.get_xlim()[::-1])
    
    axes.get_legend().remove()
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Wake Model',r"$\mathrm{WC}: \alpha = 1.0$", r"$\mathrm{WC}: \alpha = 0.8855$",  r'$\mathrm{WC}:\mathrm{T}(\rho_{A_{region}})$', r'$t_{res}$','6 h', '12 h','24 h']
    plt.figlegend(handles, labels, loc = 'center right', bbox_to_anchor=(1.3, 0.5), ncol=1, title=' ', frameon=False)
    plt.savefig('plots/'+name+'_opt_cap_temporal.png', bbox_inches='tight')


################################################################################
################################ PROCESSING ####################################
################################################################################
# taken online to allow me to convert pandas into gdf for plotting
def df_to_geodf(df, geom_col="geom", crs=None, wkt=True):
    """
    Transforms a pandas DataFrame into a GeoDataFrame.
    The column 'geom_col' must be a geometry column in WKB representation.
    To be used to convert df based on pd.read_sql to gdf.
    Parameters
    ----------
    df : DataFrame
        pandas DataFrame with geometry column in WKB representation.
    geom_col : string, default 'geom'
        column name to convert to shapely geometries
    crs : pyproj.CRS, optional
        CRS to use for the returned GeoDataFrame. The value can be anything accepted
        by :meth:`pyproj.CRS.from_user_input() <pyproj.crs.CRS.from_user_input>`,
        such as an authority string (eg "EPSG:4326") or a WKT string.
        If not set, tries to determine CRS from the SRID associated with the
        first geometry in the database, and assigns that to all geometries.
    Returns
    -------
    GeoDataFrame
    """

    if geom_col not in df:
        raise ValueError("Query missing geometry column '{}'".format(geom_col))

    geoms = df[geom_col].dropna()

    if not geoms.empty:
        if wkt == True:
            load_geom = shapely.wkt.loads
        else:
            load_geom_bytes = shapely.wkb.loads
            """Load from Python 3 binary."""

            def load_geom_buffer(x):
                """Load from Python 2 binary."""
                return shapely.wkb.loads(str(x))

            def load_geom_text(x):
                """Load from binary encoded as text."""
                return shapely.wkb.loads(str(x), hex=True)

            if isinstance(geoms.iat[0], bytes):
                load_geom = load_geom_bytes
            else:
                load_geom = load_geom_text

        # df[geom_col] = geoms = geoms.apply(load_geom)
        if crs is None:
            srid = shapely.geos.lgeos.GEOSGetSRID(geoms.iat[0]._geom)
            # if no defined SRID in geodatabase, returns SRID of 0
            if srid != 0:
                crs = "epsg:{}".format(srid)

    return GeoDataFrame(df, crs=crs, geometry=geom_col)

def scenario_list(models, splits):
    template = """
    {config_value}-s{config_value2}:
        offshore_mods:
            wake_model: {config_value}
            region_area_threshold: {config_value2}
    """
    # Define all possible combinations of config values.
    # This must define all config values that are used in the template.
    config_values = dict(
        config_value=models, 
        config_value2=splits,
        config_value3=[1]
        )

    combinations = [
        dict(zip(config_values.keys(), values))
        for values in itertools.product(*config_values.values())
    ]

    scenarios=[]
    # Print the Titles
    for i, config in enumerate(combinations):
        scenarios.append("{config_value}-s{config_value2}".format(scenario_number=i, **config))
    return scenarios

def results_dataframe(clusters,models,splits,prefix):
    scenarios = scenario_list(models,splits)
    total_df =[]
    for scenario in scenarios:
        n = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")
        
    scenarios = scenario_list(models,splits)
    total_df =[]
    for scenario in scenarios:
        n = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")
        
        df = n.statistics().filter(regex="Generator", axis=0)[["Energy Balance","Optimal Capacity", "Capacity Factor"]]
        df["Energy Balance"] = df["Energy Balance"] / 1e6
        df["Optimal Capacity"] = df["Optimal Capacity"] / 1e3
        df['scenario'] = scenario
        df['scenario_nice'] = nice_names_for_plotting(scenario)
        total_df.append(df)
            
    results, colours = clean_results(n, total_df, scenarios)
    return results, colours

def clean_results(n, df, scenarios):
    results = pd.concat(df, axis=0, ignore_index=False).reset_index(level=0,drop=True).reset_index(names="carrier")#.reset_index(name="carrier")
    results = results.pivot(index=['scenario','scenario_nice'],columns=['carrier'])

    mapping = {scenario: i for i, scenario in enumerate(scenarios)}
    key = results.index.get_level_values(0).map(mapping)
    color = n.carriers.set_index("nice_name").color.where(lambda s: s != "", "lightgrey")
    color = color[results.columns.get_level_values("carrier")]

    # slicing and reversing so the top value is is (a)
    results = results.iloc[key.argsort()[::-1][:len(scenarios)]].reset_index(level='scenario')
    results['wake_model'] = results['scenario'].str.split('-',expand=True)[0]
    results['region_max'] = results['scenario'].str.split('-',expand=True)[1].str[1:].astype(float)
    return results, color

def nice_names_for_plotting(scenario):
    if scenario.split('-')[0] == "base":
        nice_name = r"(a) "+r"$\mathrm{WC}: \alpha = 1.0$"
    elif scenario.split('-')[0] == "standard":
        nice_name = r"(b) "+r"$\mathrm{WC}: \alpha = 0.8855$"
    elif scenario.split('-')[0] == "new_more":
        nice_name = r"(d) "+r'$\mathrm{WC}:\mathrm{T}(\rho_{A_{region}})$'
    elif scenario.split('-')[0] == "glaum":
        nice_name = r"(c) "+r'$\mathrm{WC}: \mathrm{T}({P^{max}_{nom}})$'

    area = int(scenario.split('-')[1][1:])
    nice_name += "\n" + r"$A_{region}^{max}:$ " + f"{area:,}" + r" km$\mathrm{^{2}}$"
    return nice_name

def nice_names_for_plotting_label(models):
    labels = []
    if "new_more" in models:
        labels.append(r'$\mathrm{WC}:\mathrm{T}(\rho_{A_{region}})$')
    if "glaum" in models:
        labels.append(r'$\mathrm{WC}: \mathrm{T}({P^{max}_{nom}})$')
    if "standard" in models:
        labels.append(r"$\mathrm{WC}: \alpha = 0.8855$")
    if "base" in models:
        labels.append(r"$\mathrm{WC}: \alpha = 1.0$")
    return labels

def nice_names_dist(scenario):
    if scenario.split('-')[0] == "base":
        nice_name = r"$\mathrm{WC}: \alpha = 1.0$"
    elif scenario.split('-')[0] == "standard":
        nice_name = r"$\mathrm{WC}: \alpha = 0.8855$"
    elif scenario.split('-')[0] == "new_more":
        nice_name = r'$\mathrm{WC}:\mathrm{T}(\rho_{A_{region}})$'
    elif scenario.split('-')[0] == "glaum":
        nice_name = r'$\mathrm{WC}: \mathrm{T}({P^{max}_{nom}})$'
    else:
        nice_name = r"$\mathrm{WC}: \alpha = 1.0$ (Uniform)"
    return nice_name

def prepare_dist_series(fixed, clusters, models,splits, prefix):
    scenarios = scenario_list(models,splits)
    total_df = []
    for scenario in scenarios:      
        # load the region info
        regions = gpd.read_file("wake_extra/"+prefix.split('-')[2]+"/regions_offshore_"+scenario.split('-')[1]+".geojson")
        regions["region"] = regions["name"]
    
        # load network and process build-out data
        if scenario.split('-')[0] == "uniform" or fixed == "build-out":
            n_build = pypsa.Network("results/"+prefix+"/base-"+scenario.split('-')[1]+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")
        else:
            n_build = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")

        # prep build out data
        ds_buildout = n_build.generators.filter(regex='offwind', axis=0).copy()
        ds_buildout['gen'] = ds_buildout.index.to_series().str.split(' ').str[:3].str.join(sep=" ")
        ds_buildout = ds_buildout.groupby("gen").agg({"p_nom_opt": np.sum,"p_nom_max":np.sum})
        ds_buildout["region"] = ds_buildout.index.to_series().str.split(' ').str[:2].str.join(sep=" ")
        ds_buildout = ds_buildout.reset_index().merge(regions[['area','region']], on="region")

        # making build-out uniform
        if (scenario.split('-')[0] == "uniform") or fixed == "build-out":
            ds_buildout['p_nom_opt'] = 4*ds_buildout["area"]

        # if (scenario.split('-')[0] == "uniform"):
        #   ds_buildout['p_nom_opt'] = 4*ds_buildout["area"]

        # load network for the capacity factor time series
        if scenario.split('-')[0] == "uniform" or fixed == 'CF':
            n_wake = pypsa.Network("results/"+prefix+"/base-"+scenario.split('-')[1]+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")
        else:
            n_wake = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")

        # preparing cf
        # ds_cf = n_wake.generators_t.p.filter(like="offwind", axis=1).divide(n_wake.generators.filter(regex='offwind', axis=0).p_nom_opt).T
        ds_cf = n_wake.generators_t.p_max_pu.filter(like="offwind", axis=1).multiply(n_wake.generators.filter(regex='offwind', axis=0).p_nom_opt).T
        ds_cf['gen'] = ds_cf.index.to_series().str.split(' ').str[:3].str.join(sep=" ")
        ds_cf = ds_cf.groupby("gen").mean()
        
        # preparing curtailment
        ds_avail = n_wake.generators_t.p_max_pu.filter(like="offwind", axis=1).multiply(n_wake.generators.filter(regex='offwind', axis=0).p_nom_opt).T
        ds_avail['gen'] = ds_avail.index.to_series().str.split(' ').str[:3].str.join(sep=" ")
        ds_avail = ds_avail.groupby("gen").sum()
        ds_used = n_wake.generators_t.p.filter(like="offwind", axis=1).T
        ds_used['gen'] = ds_used.index.to_series().str.split(' ').str[:3].str.join(sep=" ")
        ds_used = ds_used.groupby("gen").sum()
        ds_curtail = (ds_avail - ds_used) / ds_avail
        
        # combining datasets
        df1 = ds_curtail.melt(
        ignore_index=False,var_name='date', value_name="curtail"
        ).reset_index()
        df2 = ds_cf.melt(
        ignore_index=False,var_name='date', value_name="cf"
        ).reset_index()
        
        ds = pd.merge(df1, df2, on=['gen','date']).set_index('gen')
        ds['region'] = ds.index.to_series().str.split(' ').str[:2].str.join(sep=" ")
        ds = ds.reset_index()
        ds['region_max'] = int(scenario.split('-')[1][1:])
        ds['wake_model'] = nice_names_dist(scenario)
        ds['date_int'] = pd.to_numeric(pd.to_datetime(ds['date']))
        
        # calculating results
        if fixed in ("CF",'build-out'):
            ds = ds.merge(ds_buildout[["gen","p_nom_opt"]], on='gen')
            ds['supply'] = ds['cf']*ds['p_nom_opt']
            ds['curtailed'] = ds['curtail']*ds['p_nom_opt']
            ds = ds.groupby(["region_max","date_int","wake_model"]).agg({"curtailed": np.sum,"supply": np.sum,"p_nom_opt":np.sum})
            ds['cf'] = ds['supply'] / ds['p_nom_opt']
            ds['curtail'] = ds['curtailed'] / ds['p_nom_opt']
        
        else:
            def weighted_avg(df, values, weights):
                return (df[values] * df[weights]).sum() / df[weights].sum()
            ds = ds.merge(regions[['area','region']], on="region")
            ds['w_cf'] = ds['cf']
            wavg = lambda x: weighted_avg(ds.loc[x.index], 'cf', 'area')
            ds = ds.groupby(["region_max","date_int","wake_model"]).agg({"curtail": np.mean,"w_cf": wavg, "cf": np.mean})
        
        ds = ds.reset_index()
        total_df.append(ds)
    
    results = pd.concat(total_df, axis=0, ignore_index=True)
    results = results[['region_max','date_int','wake_model','cf','curtail']].melt(id_vars=['region_max','date_int','wake_model'],value_name="value",var_name='variable')
    return results

def prepare_runtime_data(models,splits,temporals,prefix):
    scenarios = scenario_list(models,splits)
    total_df = []
    for temp in temporals:
        for scenario in scenarios:      
            log = open("results/"+prefix+"-"+str(temp)+"h/"+scenario+"/benchmarks/solve_sector_network/base_s_10_lvopt___2030", "r")
            df = pd.read_csv(io.StringIO('\n'.join(log)), sep='\t')[["s","mean_load"]]
            # print(scenario.split('-')[1][1:])
            df['model'] = nice_names_for_plotting_label(scenario.split('-')[0])
            
            df['spatial'] = scenario.split('-')[1][1:]
            # df['spatial'] = int(scenario.split('-')[1][1:])
            df['temporal'] = temp
            total_df.append(df)

    results = pd.concat(total_df, axis=0, ignore_index=True)
    return results

def temporal_comp_data(clusters,models,splits,temporals,prefix):
    scenarios = scenario_list(models,splits)
    total_df =[]
    for temp in temporals:
        for scenario in scenarios:
            n = pypsa.Network("results/"+prefix+"-"+str(temp)+"h/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")
                
            df = n.statistics().filter(regex="Generator", axis=0)[["Energy Balance","Optimal Capacity", "Capacity Factor"]].filter(regex="Offshore", axis=0).groupby(level=0, axis=0).sum()
            df["Energy Balance"] = df["Energy Balance"] / 1e6
            df["Optimal Capacity"] = df["Optimal Capacity"] / 1e3
            df['scenario'] = scenario
            # df['scenario_nice'] = nice_names_for_plotting(scenario)
            df['temporal'] = temp
            total_df.append(df)
            
    results = pd.concat(total_df, axis=0, ignore_index=True)
    results['wake_model'] = results['scenario'].str.split('-',expand=True)[0]
    results['region_max'] = results['scenario'].str.split('-',expand=True)[1].str[1:].astype(float)
    return results

def wake_losses(area):
    """
    calculating the wake losses as a function of installed capacity density
    for a given area and different wake models

    Args:
        area (int): area of a region in km2

    Returns:
        pd.dataframe: wake losses dataframe
    """
    ######################################################################
    # capacity density tiers
    def y(x):
        alpha = 7.3
        beta = 0.05
        gamma = -0.7
        delta = -14.6
        return alpha*np.exp(-x/beta) + gamma*x + delta

    def piecewise(x0,x1):
        return (y(x1)*x1 - y(x0)*x0)/(x1-x0)

    def capacity_density_tiers(row):
        x0,x1,x2,x3,x4,x5,x6 = 0.0000,0.0250,0.0500,0.2500,1.0000,2.5000,4.0000
        wf_1,wf_2,wf_3,wf_4,wf_5,wf_6 = 0,0,0,0,0,0
        if x0 < row["ICD"] <= x1:
            wf_1 = (1-(piecewise(x0,x1)/100)) * area * (row["ICD"])
            print(wf_1)
        elif x1 < row["ICD"] <= x2:
            wf_1 = (1-(piecewise(x0,x1)/100)) * area * (x1-x0)
            wf_2 = (1-(piecewise(x1,x2)/100)) * area * (row["ICD"]-(x1))
        elif x2 < row["ICD"] <= x3:
            wf_1 = (1-(piecewise(x0,x1)/100)) * area * (x1-x0)
            wf_2 = (1-(piecewise(x1,x2)/100)) * area * (x2-x1)
            wf_3 = (1-(piecewise(x2,x3)/100)) * area * (row["ICD"]-(x2))
        elif x3 < row["ICD"] <= x4:
            wf_1 = (1-(piecewise(x0,x1)/100)) * area * (x1-x0)
            wf_2 = (1-(piecewise(x1,x2)/100)) * area * (x2-x1)
            wf_3 = (1-(piecewise(x2,x3)/100)) * area * (x3-x2)
            wf_4 = (1-(piecewise(x3,x4)/100)) * area * (row["ICD"]-(x3))
        elif x4 < row["ICD"] <= x5:
            wf_1 = (1-(piecewise(x0,x1)/100)) * area * (x1-x0)
            wf_2 = (1-(piecewise(x1,x2)/100)) * area * (x2-x1)
            wf_3 = (1-(piecewise(x2,x3)/100)) * area * (x3-x2)
            wf_4 = (1-(piecewise(x3,x4)/100)) * area * (x4-x3)
            wf_5 = (1-(piecewise(x4,x5)/100)) * area * (row["ICD"]-(x4))
        elif x5 < row["ICD"]:
            wf_1 = (1-(piecewise(x0,x1)/100)) * area * (x1-x0)
            wf_2 = (1-(piecewise(x1,x2)/100)) * area * (x2-x1)
            wf_3 = (1-(piecewise(x2,x3)/100)) * area * (x3-x2)
            wf_4 = (1-(piecewise(x3,x4)/100)) * area * (x4-x3)
            wf_5 = (1-(piecewise(x4,x5)/100)) * area * (x5-x4)
            wf_6 = (1-(piecewise(x5,x6)/100)) * area * (row["ICD"]-(x5))
        return (1-(wf_1 + wf_2 + wf_3 + wf_4 + wf_5 + wf_6 )/ row["p_nom"]) * 100

######################################################################
# capacity tiers from glaum et al.
    def capacity_tiers(row):
        if 0 < row["p_nom"] <= 2e3:
            glaum_1 = 0.906 * row["p_nom"]
            glaum_2 = 0
            glaum_3 = 0
        elif 2e3 < row["p_nom"] <= 12e3:
            glaum_1 = 0.906 * 2000
            glaum_2 = (row["p_nom"]-2000) * 0.906 * (1-0.1279732) 
            glaum_3 = 0
        elif row["p_nom"] > 12e3:
            glaum_1 = 0.906 * 2000
            glaum_2 = (10000) * 0.906 * (1-0.1279732) 
            glaum_3 = (row["p_nom"]-12000) * 0.906* (1-(1 - ((1-0.1279732)*(1-0.1390210410))))
        else:
            return (1-0.906) * 100
        return - (1 - (glaum_1 + glaum_2 + glaum_3)/row["p_nom"]) * 100

    ######################################################################
    # calculating wake losses for different installed capacity densities
    ICD = np.linspace(0,4,100) # installed capacity density in MW/km2
    data = {"ICD":ICD,"p_nom":ICD * area} # installed capacity in MW
    df = pd.DataFrame(data=data)

    df[r"$\mathrm{WC}: \alpha = 0.8855$"] = -((1-0.8855) * 100) # standard wake model
    df[r'$\mathrm{WC}: \mathrm{T}({P^{max}_{nom}})$'] = df.apply(capacity_tiers, axis=1) # glaum et al. wake model
    df[r'$\mathrm{WC}:\mathrm{T}(\rho_{A_{region}})$'] = df.apply(capacity_density_tiers, axis=1) # new wake model

    df = df.melt(id_vars = ["ICD","p_nom"],var_name="model",value_name="wake_loss")
    df["wake_loss"] = abs(df["wake_loss"])
    return df
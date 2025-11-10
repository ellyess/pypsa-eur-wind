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
# plt.rc("figure", figsize=(10, 8))

bg_colour = '#f0f0f0'
custom_params = {'xtick.bottom': True, 'axes.edgecolor': 'black', 'axes.spines.right': False, 'axes.spines.top': False, 'mathtext.default': 'regular'}
sns.set_theme(style='ticks', rc=custom_params)

################################################################################
############################### PROCESSING ##################################
################################################################################

##### taken online to allow me to convert pandas into gdf for plotting
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
        nice_name = "(a) None"
    elif scenario.split('-')[0] == "standard":
        nice_name = "(b) Single Tier"
    elif scenario.split('-')[0] == "new_more":
        nice_name = "(d) Variable Tiers"
    elif scenario.split('-')[0] == "glaum":
        nice_name = "(f) Fixed Tiers"

    area = int(scenario.split('-')[1][1:])
    nice_name += " | " + f"{area:,}" + r" km$\mathrm{^{2}}$"
    return nice_name
  
  
################################################################################
################################# PLOTTING #####################################
################################################################################
def nice_names_for_plotting_label(models):
    labels = []
    if "glaum" in models:
        labels.append("Fixed Tiers |")
    if "new_more" in models:
        labels.append("Variable Tiers |")
    if "ellyess" in models:
        labels.append("New |")
    if "standard" in models:
        labels.append("Single Tier |")
    if "base" in models:
        labels.append("None |")
    return labels

def plot_stacked(var,results, filter, colours, name):
  fig, ax = plt.subplots(figsize=(5, 7))
  
  if var ==  "Optimal Capacity":
    save = "p_opt"
    unit = ", GW"
  elif var ==  "Energy Balance":
    save = "e_opt"
    unit = ", TWh"
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
  print(class_lines_y)
  class_labels_y = np.arange(np.floor(len(splits)/2),len(models)*len(splits),len(splits))
  
  # labels classes:
  sec = ax.secondary_yaxis(location=0)
  sec.set_yticks(class_labels_y, labels=nice_names_for_plotting_label(models))
  sec.tick_params('y', length=60, width=0)
  sec.set_ylabel(r'Wake Model | Max Region Area (km$\mathrm{^{2}}$)')
  
  # lines between the classes:
  sec2 = ax.secondary_yaxis(location=0)
  sec2.set_yticks(class_lines_y, labels=[])
  sec2.tick_params('y', length=120, width=1.5)
  
  x_max = results[var].filter(regex=filter, axis=1).mean(axis=0).sum()
      
  ax.hlines(y=class_lines_y[:-2]+len(splits), xmin=0, xmax=x_max, linewidth=1.5, color='r',linestyles='--')
  
  handles, labels = plt.gca().get_legend_handles_labels()
  
  if 'singlewind' in name:
      labels = ['Offshore Wind (All-Inclusive)']

  plt.figlegend(handles,[word.title() for word in labels],loc = 'upper center', bbox_to_anchor=(0.5, 0), ncol=3, title='Carrier', frameon=False)
  # plt.xticks(rotation=90)
  plt.tight_layout()
  plt.savefig('plots/'+name+'_'+save+'_stacked.png', bbox_inches='tight')


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
    labels = ['None', 'Single Tier',  'Variable Tiers', 'Fixed Tiers']
    plt.figlegend(handles, labels, loc = 'center right', bbox_to_anchor=(1.1, 0.5), ncol=1, title='Wake Model', frameon=False)
    plt.tight_layout()
    # plt.savefig('plots/'+name+'_capacity_split.png', bbox_inches='tight')
    
def plot_generation_series(clusters, models,splits, prefix):
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
    fig, ax = plt.subplots(1,3,figsize=(12,4),sharex=True, sharey=True,layout='tight')
    i = 0
    for scenario in scenario_list:
        generators = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_10_lvopt___2030.nc").generators.filter(regex="offwind", axis=0).copy()
        generators["region"] = generators.index.to_series().str.split(' ').str[:2].str.join(sep=" ")
        regions = gpd.read_file("wake_extra/"+prefix.split('-')[2]+"/regions_offshore_"+scenario.split('-')[1]+".geojson").set_index('name')
        generators = generators.merge(regions, right_index=True, left_on="region")
        # generators = generators[generators['p_nom_max']>=100]
        
        total = generators.groupby('region')['p_nom_opt'].transform('sum')
        generators['% of capacity'] = generators['p_nom_opt'].div(total)
        generators = generators.sort_values('% of capacity').drop_duplicates(['region'], keep='last')
        geo_df = df_to_geodf(generators, geom_col="geometry", crs="4326")
        
        carriers = {
            'offwind-ac': 'Reds',
            'offwind-dc': 'Blues',
            'offwind-float': 'Greens'
        }

        j=0
        for carrier, color in carriers.items():
            if j == 0 and i == 0:
                legend_value = True
            elif j == 1 and i == 1:
                legend_value = True
            elif j == 2 and i == 2:
                legend_value = True
            else:
                legend_value = False
                
            if carrier == 'offwind-ac':
                nice_name = 'Offshore Wind (AC)'
            elif carrier == 'offwind-dc':
                nice_name = 'Offshore Wind (DC)'
            elif carrier == 'offwind-float':
                nice_name = 'Offshore Wind (Floating)'


            geo_df[geo_df['carrier'] == carrier].plot(
                column='% of capacity', 
                cmap=color,
                legend=legend_value, 
                ax=ax[i],
                vmin=0,vmax=1,
                legend_kwds={
                    'label': '% of '+str(nice_name),
                    "orientation": "horizontal",
                    # 'fraction': 0.05,
                    'pad': -0.01,
                    'shrink': 0.5,
                    },
                linewidth=0.1,
                )
            j+=1
        
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()
        i += 1
        plt.savefig('plots/'+prefix+'_region_dominant.png', bbox_inches='tight')

def plot_region_optimal_capacity(carrier,clusters,scenarios,prefix,vmin=None,vmax=None):
    fig, ax = plt.subplots(1,4,figsize=(16,4),sharex=True, sharey=True,layout='tight')
            
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
                'label': "Optimal Capacity",
                "orientation": "vertical",
                'fraction': 0.1,
                'pad': 0.1,
                'shrink': 0.8,
            })
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()
        
    plt.savefig('plots/'+prefix+'_region_optimal_capacity.png', bbox_inches='tight')
    
def plot_region_optimal_density(carrier,clusters,scenarios,prefix,vmin=None,vmax=None):
    fig, ax = plt.subplots(1,4,figsize=(16,4),sharex=True, sharey=True,layout='tight')
            
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
                'label': "Optimal Capacity Density",
                "orientation": "vertical",
                'fraction': 0.1,
                'pad': 0.1,
                'shrink': 0.8,
            })
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()
        
    plt.savefig('plots/'+prefix+'_region_optimal_density.png', bbox_inches='tight')
    
def plot_region_capacity_density(carrier,clusters,scenarios,prefix,vmin=None,vmax=None):
    fig, ax = plt.subplots(1,4,figsize=(16,4),sharex=True, sharey=True,layout='tight')
            
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
                'label': "Max Capacity Density",
                "orientation": "vertical",
                'fraction': 0.1,
                'pad': 0.1,
                'shrink': 0.8,
            })
        ax[i].set_title(nice_names_for_plotting(scenario)[3:])
        ax[i].set_axis_off()
        
    plt.savefig('plots/'+prefix+'_region_capacity_density.png', bbox_inches='tight')
    
    
def prepare_gen_series(clusters, models,splits, prefix,filter):
  scenarios = scenario_list(models,splits)
  total_df = []
  for scenario in scenarios:
      
    n = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")
    
    ds = n.statistics.supply(comps=["Generator"], aggregate_time=False).droplevel(0).div(1e3).filter(regex=filter, axis=0)

    ds['wake_model'] = scenario.split('-')[0]
    ds['region_max'] = int(scenario.split('-')[1][1:])
    
    ds = ds.reset_index("carrier").set_index(['carrier','wake_model','region_max'])
    # print(ds)
    total_df.append(ds)
    
  results = pd.concat(total_df, axis=0, ignore_index=False)
  results = results.melt(
    ignore_index=False,
    var_name='date', value_name='p_gen'
    )
  results['date_int'] = pd.to_numeric(pd.to_datetime(results['date']))
  results = results.reset_index()

  return results

def prepare_cf_series(clusters, models,splits, prefix,filter):
  scenarios = scenario_list(models,splits)
  total_df = []
  for scenario in scenarios:
      
    n = pypsa.Network("results/"+prefix+"/"+scenario+"/postnetworks/base_s_"+str(clusters)+"_lvopt___2030.nc")
    
    ds = n.statistics.capacity_factor(comps=["Generator"], aggregate_time=False).droplevel(0).filter(regex=filter, axis=0)

    ds['wake_model'] = scenario.split('-')[0]
    ds['region_max'] = int(scenario.split('-')[1][1:])
    
    ds = ds.reset_index("carrier").set_index(['carrier','wake_model','region_max'])
    # print(ds)
    total_df.append(ds)
    
  results = pd.concat(total_df, axis=0, ignore_index=False)
  results = results.melt(
    ignore_index=False,
    var_name='date', value_name='cf'
    )
  results['date_int'] = pd.to_numeric(pd.to_datetime(results['date']))
  results = results.reset_index()

  return results
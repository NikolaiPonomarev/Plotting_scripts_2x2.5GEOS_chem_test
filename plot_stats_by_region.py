import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
import matplotlib.patches as mpatches

# ----------------------
# Load dataset
# ----------------------
ds = xr.open_dataset('/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/daily_binned_XCO2.nc')

lat = ds['lat'].values
lon = ds['lon'].values
time = ds['time'].values
n_lat, n_lon = len(lat), len(lon)

# ----------------------
# Create proper land mask
# ----------------------
land_mask = np.zeros((n_lat, n_lon), dtype=bool)
land_shp = shpreader.natural_earth(resolution='110m', category='physical', name='land')
land_geoms = list(shpreader.Reader(land_shp).geometries())

for i in range(n_lat):
    for j in range(n_lon):
        pt = Point(lon[j], lat[i])
        land_mask[i, j] = any(pt.within(geom) for geom in land_geoms)

ocean_mask = ~land_mask

# ----------------------
# Latitude bands
# ----------------------
def lat_band_mask(lat_min, lat_max):
    mask = ((lat >= lat_min) & (lat <= lat_max))[:, None]
    mask = np.repeat(mask, n_lon, axis=1)
    return mask

tropics_mask = lat_band_mask(-23.5, 23.5)
northern_mid_mask = lat_band_mask(23.5, 60)
southern_mid_mask = lat_band_mask(-60, -23.5)
polar_mask = ((lat > 60)[:, None] | (lat < -60)[:, None])
polar_mask = np.repeat(polar_mask, n_lon, axis=1)
tropics_land_mask = tropics_mask & land_mask  # combine tropics + land

regions = {
    'Land': land_mask,
    'Ocean': ocean_mask,
    'Tropics': tropics_mask,
    'Tropics Land': tropics_land_mask,
    'Northern Mid-Latitudes': northern_mid_mask,
    'Southern Mid-Latitudes': southern_mid_mask,
    'Polar': polar_mask
}

# ---------------------- 
# Functions to compute stats
# ----------------------

def annual_stats(ds_obs, ds_model, ds_corrected, mask):
    """Compute annual bias, rmse, crmse (from daily values), mean, and std for boxplots of obs, model, corrected."""
    obs_masked = ds_obs.where(mask)
    model_masked = ds_model.where(mask)
    corrected_masked = ds_corrected.where(mask)
    
    years = np.unique(ds_obs['time.year'].values)
    bias_list, rmse_list, crmse_list = [], [], []
    bias_list_corr, rmse_list_corr, crmse_list_corr = [], [], []
    mean_obs_list, mean_model_list, mean_corr_list = [], [], []
    std_obs_list, std_model_list, std_corr_list = [], [], []

    for y in years:
        obs_year = obs_masked.sel(time=ds_obs['time.year']==y)
        model_year = model_masked.sel(time=ds_model['time.year']==y)
        corr_year = corrected_masked.sel(time=ds_corrected['time.year']==y)
        
        obs_flat = obs_year.values.flatten()
        model_flat = model_year.values.flatten()
        corr_flat = corr_year.values.flatten()
        
        valid = ~np.isnan(obs_flat) & ~np.isnan(model_flat)
        if np.sum(valid) < 2:
            bias_list.append(np.nan)
            rmse_list.append(np.nan)
            crmse_list.append(np.nan)
            mean_obs_list.append([])
            mean_model_list.append([])
            mean_corr_list.append([])
            std_obs_list.append(np.nan)
            std_model_list.append(np.nan)
            std_corr_list.append(np.nan)
            continue
        
        diff = model_flat[valid] - obs_flat[valid]
        bias_list.append(np.mean(diff))
        rmse_list.append(np.sqrt(np.mean(diff**2)))
        crmse_list.append(np.sqrt(np.mean((diff - np.mean(diff))**2)))

        diffc = model_flat[valid] - corr_flat[valid]
        bias_list_corr.append(np.mean(diffc))
        rmse_list_corr.append(np.sqrt(np.mean(diffc**2)))
        crmse_list_corr.append(np.sqrt(np.mean((diffc - np.mean(diffc))**2)))
        
        # store full values for boxplots
        mean_obs_list.append(obs_flat[valid])
        mean_model_list.append(model_flat[valid])
        mean_corr_list.append(corr_flat[valid])
        std_obs_list.append(np.std(obs_flat[valid]))
        std_model_list.append(np.std(model_flat[valid]))
        std_corr_list.append(np.std(corr_flat[valid]))

    return (np.array(bias_list), np.array(rmse_list), np.array(crmse_list),
            np.array(bias_list_corr), np.array(rmse_list_corr), np.array(crmse_list_corr),
            mean_obs_list, mean_model_list, mean_corr_list,
            np.array(std_obs_list), np.array(std_model_list), np.array(std_corr_list))

def monthly_correlation(ds_obs, ds_model, mask):
    """Compute monthly correlation within region and average for each year"""
    obs_masked = ds_obs.where(mask)
    model_masked = ds_model.where(mask)
    
    corr_list = []
    for year, group in obs_masked.groupby('time.year'):
        # group is a DataArray with daily values in this year
        monthly_corrs = []
        for month, mgroup in group.groupby('time.month'):
            obs_month = mgroup
            model_month = model_masked.sel(time=mgroup.time)
            
            obs_flat = obs_month.values.flatten()
            model_flat = model_month.values.flatten()
            valid = ~np.isnan(obs_flat) & ~np.isnan(model_flat)
            if np.sum(valid) > 1:
                c = np.corrcoef(obs_flat[valid], model_flat[valid])[0,1]
                monthly_corrs.append(c)
        if len(monthly_corrs) > 0:
            corr_list.append(np.mean(monthly_corrs))
        else:
            corr_list.append(np.nan)
    return np.array(corr_list)


# ----------------------
# Compute statistics for all regions
# ----------------------
all_stats = {}
for name, mask in regions.items():
    # compute annual metrics + spatial means for boxplots
    (bias, rmse, crmse,
     bias_corr, rmse_corr, crmse_corr,
     mean_obs, mean_model, mean_corr,
     std_obs, std_model, std_corr) = annual_stats(
        ds['obs'], ds['model'], ds['corrected'], mask
    )
    
    # compute annual correlation for both model vs obs and model vs corrected obs
    corr = monthly_correlation(ds['obs'], ds['model'], mask)
    corr_corr = monthly_correlation(ds['corrected'], ds['model'], mask)
    
    all_stats[name] = dict(
        # model vs obs
        bias=bias,
        rmse=rmse,
        crmse=crmse,
        corr=corr,
        mean_obs=mean_obs,
        mean_model=mean_model,
        std_obs=std_obs,
        std_model=std_model,
        # model vs corrected obs
        bias_corr=bias_corr,
        rmse_corr=rmse_corr,
        crmse_corr=crmse_corr,
        corr_corr=corr_corr,
        mean_corr=mean_corr,
        std_corr=std_corr
    )


years = np.unique(ds['time.year'].values)
n_years = len(years)
n_regions = len(all_stats)

# ----------------------
# Plotting
# ----------------------
def plot_annual_boxplots(all_stats, years, var_name, compare_name,
                         title_vars=('Var1', 'Var2'),
                         figure_title='Annual XCO₂ Comparison',
                         save_path=None,
                         regions_colors=None,
                         dpi=300):
    n_years = len(years)
    n_regions = len(all_stats)
    width = 0.13
    positions_base = np.arange(n_years)

    if regions_colors is None:
        regions_colors = ['lightblue','orange','green','red','purple','cyan','gold']

    # create legend handles
    handles = [mpatches.Patch(facecolor=c, label=name, alpha=0.6)
               for c, name in zip(regions_colors, all_stats.keys())]

    # -----------------------------
    # Create figure with 2 subplots
    # -----------------------------
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), dpi=dpi, sharex=True)
    var_list = [var_name, compare_name]

    for ax, var, title_var in zip(axes, var_list, title_vars):
        for k, (region_name, stats) in enumerate(all_stats.items()):
            data_per_year = [np.ravel(x) for x in stats[var]]
            pos = positions_base + (k - n_regions/2)*width + width/2
            ax.boxplot(data_per_year, positions=pos, widths=width*0.9,
                       patch_artist=True,
                       boxprops=dict(facecolor=regions_colors[k], alpha=0.6),
                       medianprops=dict(color='black'),
                       labels=['' for _ in range(n_years)])
        ax.set_title(title_var)
        ax.set_ylabel('XCO₂ [ppm]')
        ax.grid(True)

    # x-axis labels
    axes[-1].set_xticks(positions_base)
    axes[-1].set_xticklabels(years, rotation=45)
    axes[-1].set_xlabel('Year')

    # legend
    axes[0].legend(handles=handles, title='Region')
    fig.suptitle(figure_title, fontsize=16, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi)

# Obs vs Model
plot_annual_boxplots(all_stats, years,
                     var_name='mean_obs', compare_name='mean_model',
                     title_vars=('Obs XCO₂', 'Model XCO₂'),
                     figure_title='Obs vs Model Annual XCO₂',
                     save_path='annual_obs_model.png')

# Corrected vs Model
plot_annual_boxplots(all_stats, years,
                     var_name='mean_corr', compare_name='mean_model',
                     title_vars=('Corrected Obs XCO₂', 'Model XCO₂'),
                     figure_title='Corrected Obs vs Model Annual XCO₂',
                     save_path='annual_corr_model.png')



def plot_stats_bar(all_stats, years, metrics_keys, titles, colors, filename, comparison_label):

    n_years = len(years)
    n_regions = len(all_stats)
    width = 0.13  # bar width

    fig, axes = plt.subplots(len(metrics_keys), 1, figsize=(12,16), dpi=300, sharex=True)

    for ax, metric, title in zip(axes, metrics_keys, titles):
        for k, (name, stats) in enumerate(all_stats.items()):
            data = stats[metric]  # 1D array, one value per year
            pos = np.arange(n_years) + (k - n_regions/2)*width + width/2
            ax.bar(pos, data, width=width*0.9, color=colors[k], alpha=0.6, label=name if metric==metrics_keys[0] else "")
        ax.set_title(f"{title} ({comparison_label})")
        ax.grid(True)

    # x-axis
    axes[-1].set_xticks(np.arange(n_years))
    axes[-1].set_xticklabels(years, rotation=45)
    axes[-1].set_xlabel('Year')

    # legend on first subplot
    axes[0].legend(title='Region')

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


# ----------------------
# Call function for model vs obs
# ----------------------
metrics_keys_obs = ['bias','rmse','crmse','corr']
titles = ['Bias [ppm]','RMSE [ppm]','cRMSE [ppm]','Correlation']
colors = ['lightblue','orange','green','red','purple','cyan','gold']

plot_stats_bar(
    all_stats=all_stats,
    years=years,
    metrics_keys=metrics_keys_obs,
    titles=titles,
    colors=colors,
    filename='/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/annual_stats_model_vs_obs.png',
    comparison_label='Model vs Obs'
)

# ----------------------
# Call function for model vs corrected obs
# ----------------------
metrics_keys_corr = ['bias_corr','rmse_corr','crmse_corr','corr_corr']

plot_stats_bar(
    all_stats=all_stats,
    years=years,
    metrics_keys=metrics_keys_corr,
    titles=titles,
    colors=colors,
    filename='/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/annual_stats_model_vs_corrected.png',
    comparison_label='Model vs Corrected Obs'
)




def taylor_diagram(all_stats, years, colors, filename, comparison='Model vs Obs', top_right_only=True):
    """
    Taylor diagram: normalized standard deviation (r-axis) and correlation (theta-axis).
    std normalized by reference (obs) so ref=1. Angle = arccos(correlation).
    """
    # Collect statistics
    std_refs = []
    stds = []
    corrs = []
    for name, stats in all_stats.items():
        if 'corrected' in comparison.lower():
            std_ref = stats['std_corr']
            std_model = stats['std_model']
            corr_vals = stats['corr_corr']
        else:
            std_ref = stats['std_obs']
            std_model = stats['std_model']
            corr_vals = stats['corr']
        std_refs.append(std_ref)
        stds.append(std_model)
        corrs.append(corr_vals)

    # Normalize stds
    ref_mean_std = np.mean([np.mean(s) for s in std_refs])
    stds_norm = [np.array(s)/ref_mean_std for s in stds]

    # Polar plot
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_direction(-1)
    ax.set_theta_zero_location('N')

    if top_right_only:
        ax.set_thetamin(0)
        ax.set_thetamax(90)

    # Reference circle
    theta = np.linspace(0, np.pi/2 if top_right_only else np.pi, 100)
    ax.plot(theta, np.ones_like(theta), color='k', linestyle='--', label='Reference')

    # Plot points
    for k, name in enumerate(all_stats.keys()):
        for i in range(len(years)):
            angle = np.arccos(np.clip(corrs[k][i], -1, 1))
            r = stds_norm[k][i]
            ax.plot(angle, r, 'o', color=colors[k], label=name if i==0 else "")

    # Correlation ticks on theta
    theta_ticks = np.radians(np.linspace(0, 90, 7)) # 0° → 90°
    corr_labels = [f"{np.cos(t):.2f}" for t in theta_ticks][::-1] 
    print(theta_ticks, corr_labels)
    ax.set_xticks(theta_ticks)
    ax.set_xticklabels(corr_labels)
    ax.text(
        np.pi/5,                # angle along the circle (adjust if needed)
        ax.get_rmax()*1.02,     # slightly closer to the circle center to move down
        "Correlation", 
        ha='center', 
        va='center', 
        fontsize=12,
        rotation=-30            # clockwise rotation in degrees
    )
    ax.set_xlabel('Standard deviation')
    ax.set_ylim(0, 1.5)
    ax.set_rlabel_position(135)
    ax.set_title(f'Taylor Diagram ({comparison})', fontsize=14)
    ax.grid(True)

    # Legend inside
    handles = [mpatches.Patch(color=c, label=name) for c, name in zip(colors, all_stats.keys())]
    ax.legend(handles=handles, title='Region', loc='upper right', bbox_to_anchor=(1, 1))

    # Adjust spacing
    plt.subplots_adjust(top=0.90, bottom=0.05, left=0.05, right=0.95)
    plt.savefig(filename, dpi=300)
    plt.close(fig)


colors = ['lightblue','orange','green','red','purple','cyan','gold']

# Taylor diagram for Model vs Obs
taylor_diagram(
    all_stats=all_stats,
    years=years,
    colors=colors,
    filename='/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/taylor_model_vs_obs.png',
    comparison='Model vs Obs'
)

# Taylor diagram for Model vs Corrected Obs
taylor_diagram(
    all_stats=all_stats,
    years=years,
    colors=colors,
    filename='/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/taylor_model_vs_corrected.png',
    comparison='Model vs Corrected Obs'
)
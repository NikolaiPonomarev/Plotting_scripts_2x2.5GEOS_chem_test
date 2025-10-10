import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

outdir = '/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/'

file = '/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/oco2_obs_model/oco2_v11.2021D003.nc'
ds = xr.open_dataset(file)

def plot_obs_vs_model(ds, obs_var='obs', model_var='model_xco2', lon_var='lon', lat_var='lat', outdir='./', prefix=''):
    lon = ds[lon_var].values
    lat = ds[lat_var].values

    # Convert time to string (first observation)
    time_dt = str(pd.to_datetime(ds.time.values[0], unit='s')).split(' ')[0]

    obs = ds[obs_var].values
    model = ds[model_var].values
    diff = model - obs

    # Compute mean values (nanmean)
    mean_obs = np.nanmean(obs)
    mean_model = np.nanmean(model)
    mean_diff = np.nanmean(diff)
    std_diff = np.nanstd(diff)
    # Color limits
    vmin = min(np.nanmin(obs), np.nanmin(model))
    vmax = max(np.nanmax(obs), np.nanmax(model))
    diff_abs = np.max(np.abs(diff))

    # Create figure with Cartopy projections
    fig, axs = plt.subplots(3, 1, figsize=(9,7),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            constrained_layout=True)

    for i, ax in enumerate(axs):
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_extent([-180, 180, -80, 80], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = gl.right_labels = False
        if i < 2:
            gl.bottom_labels = False

    # Observations
    sc0 = axs[0].scatter(lon, lat, c=obs, vmin=vmin, vmax=vmax, cmap='viridis', s=20, transform=ccrs.PlateCarree())
    axs[0].set_title(f'Observations ({obs_var}), Mean: {mean_obs:.1f}')
    plt.colorbar(sc0, ax=axs[0], orientation='vertical', label='XCO2, ppm', shrink=0.5, pad=0.01)

    # Model
    sc1 = axs[1].scatter(lon, lat, c=model, vmin=vmin, vmax=vmax, cmap='viridis', s=20, transform=ccrs.PlateCarree())
    axs[1].set_title(f'Model ({model_var}), Mean: {mean_model:.1f}')
    plt.colorbar(sc1, ax=axs[1], orientation='vertical', label='XCO2, ppm', shrink=0.5, pad=0.01)

    # Difference
    sc2 = axs[2].scatter(lon, lat, c=diff, vmin=-diff_abs, vmax=diff_abs, cmap='bwr', s=20, transform=ccrs.PlateCarree())
    axs[2].set_title(f'Model - Observations, Mean: {mean_diff:.1f}, Std: {std_diff:.1f}')
    plt.colorbar(sc2, ax=axs[2], orientation='vertical', label='XCO2 Difference, ppm', shrink=0.5, pad=0.01)

    # Save figure
    filename = os.path.join(outdir, f"{time_dt}_{prefix}_XCO2_comparison.png")
    print('Plotting', filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)



plot_obs_vs_model(ds, outdir=outdir)
plot_obs_vs_model(ds, obs_var='corrected_obs', outdir=outdir, prefix='corrected_obs')
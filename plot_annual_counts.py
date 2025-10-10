import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ----------------------
# PARAMETERS
# ----------------------
nc_file = '/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/daily_binned_XCO2.nc'  # change to your file
outdir = '/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/'
os.makedirs(outdir, exist_ok=True)

# ----------------------
# LOAD DATA
# ----------------------
ds_daily = xr.open_dataset(nc_file)

# ----------------------
# FUNCTION TO PLOT ANNUAL COUNTS
# ----------------------
def plot_annual_counts(ds_daily, year, outdir):
    # Select days in the year
    time_sel = ds_daily.time.dt.year == year

    # Sum counts over all days in the year
    annual_counts = ds_daily['counts'].sel(time=time_sel).sum(dim='time')

    lon = ds_daily.lon.values
    lat = ds_daily.lat.values

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           constrained_layout=True)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-180, 180, -90, 90])
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False

    im = ax.pcolormesh(lon, lat, annual_counts.values, cmap='Reds', transform=ccrs.PlateCarree())
    ax.set_title(f'{year} Annual Observation Counts, Mean {np.nanmean(annual_counts):.2f}')
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01, label='Number of observations')

    filename = os.path.join(outdir, f'{year}_annual_counts.png')
    print('Plotting', filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ----------------------
# LOOP OVER YEARS
# ----------------------
years = np.unique(ds_daily.time.dt.year)
for year in years:
    plot_annual_counts(ds_daily, year, outdir)


def plot_annual_bias_count_corr(ds_daily, year, outdir, typeobs='corrected'):
    # Select days in the year
    time_sel = ds_daily.time.dt.year == year

    # Extract arrays
    obs = ds_daily[typeobs].sel(time=time_sel).values  # shape: (time, lat, lon)
    model = ds_daily['model'].sel(time=time_sel).values
    diff = np.abs(model - obs)
    n_lat, n_lon = obs.shape[1], obs.shape[2]
    corr_map = np.full((n_lat, n_lon), np.nan)
    annual_counts = ds_daily['counts'].sel(time=time_sel).values
    # Compute per-grid-cell correlation
    for i in range(n_lat):
        for j in range(n_lon):
            mask = ~np.isnan(diff[:, i, j]) & ~np.isnan(annual_counts[:, i, j])
            if np.sum(mask) > 1:
                corr_map[i, j] = np.corrcoef(diff[mask, i, j], annual_counts[mask, i, j])[0,1]

    # Plot
    lon = ds_daily.lon.values
    lat = ds_daily.lat.values

    fig, ax = plt.subplots(1, 1, figsize=(12, 6),
                           subplot_kw={'projection': ccrs.PlateCarree()},
                           constrained_layout=True)

    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.set_extent([-180, 180, -90, 90])
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = gl.right_labels = False

    im = ax.pcolormesh(lon, lat, corr_map, cmap='coolwarm', vmin=-1, vmax=1, transform=ccrs.PlateCarree())
    ax.set_title(f'{year} Annual correlation abs. bias vs obs. counts, Mean {np.nanmean(corr_map):.2f}')
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.01, label='r')

    filename = os.path.join(outdir, f'{typeobs}_{year}_annual_corr.png')
    print('Plotting', filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

for year in years:
    plot_annual_bias_count_corr(ds_daily, year, outdir)
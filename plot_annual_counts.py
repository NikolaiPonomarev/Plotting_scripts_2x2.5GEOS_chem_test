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

    im = ax.pcolormesh(lon, lat, annual_counts.values, cmap='viridis', transform=ccrs.PlateCarree())
    ax.set_title(f'{year} Annual Observation Counts')
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
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


def plot_hovmoller(ds_daily, outdir, axis='lat'):
    counts = ds_daily['counts']
    time = ds_daily.time.values
    
    if axis == 'lat':
        # Sum over longitude
        hov = counts.sum(dim='lon')
        y = ds_daily.lat.values
        ylabel = 'Latitude'
        fname = 'hovmoller_lat_time.png'
    elif axis == 'lon':
        # Sum over latitude
        hov = counts.sum(dim='lat')
        y = ds_daily.lon.values
        ylabel = 'Longitude'
        fname = 'hovmoller_lon_time.png'
    else:
        raise ValueError("axis must be 'lat' or 'lon'")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.pcolormesh(time, y, hov.T, cmap='Reds')  # transpose so y vs time
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.set_title(f'Hovmöller Plot of Counts (sum over {"lon" if axis=="lat" else "lat"})')
    plt.colorbar(im, ax=ax, label='Number of counts')
    plt.tight_layout()
    
    filename = os.path.join(outdir, fname)
    print('Plotting', filename)
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    

# Example usage:
plot_hovmoller(ds_daily, outdir, axis='lat')
plot_hovmoller(ds_daily, outdir, axis='lon')
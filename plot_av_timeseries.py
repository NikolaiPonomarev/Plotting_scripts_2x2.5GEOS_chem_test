import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from concurrent.futures import ProcessPoolExecutor, as_completed 
import matplotlib.dates as mdates

indir = '/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/oco2_obs_model/'
outdir = '/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/'
os.makedirs(outdir, exist_ok=True)

files = sorted(glob.glob(os.path.join(indir, 'oco2_v11.*.nc')))

# ----------------------
# FUNCTION TO PROCESS ONE FILE
# ----------------------
def process_file(filename):
    ds = xr.open_dataset(filename)
    
    # Time: average across nobs
    time_sec = np.nanmean(ds.time.values)
    date = pd.to_datetime(time_sec, unit='s')
    
    # Daily means across nobs
    mean_obs = float(ds.obs.mean().values)
    mean_model = float(ds.model_xco2.mean().values)
    mean_corr = float(ds.corrected_obs.mean().values)
    
    # Return also lon, lat, and values for annual scatter map
    return {
        'date': date,
        'mean_obs': mean_obs,
        'mean_model': mean_model,
        'mean_corr': mean_corr,
        'lon': ds.lon.values,
        'lat': ds.lat.values,
        'obs': ds.obs.values,
        'model': ds.model_xco2.values,
        'corr': ds.corrected_obs.values
    }

# ----------------------
# READ FILES IN PARALLEL
# ----------------------
results = []
with ProcessPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(process_file, f) for f in files]
    for fut in as_completed(futures):
        results.append(fut.result())


#Plot timeseries 

results.sort(key=lambda x: x['date'])
dates = [r['date'] for r in results]
mean_obs_ts = [r['mean_obs'] for r in results]
mean_model_ts = [r['mean_model'] for r in results]
mean_corr_ts = [r['mean_corr'] for r in results]


# Mask to ignore NaNs for bias/cRMSE
mask_obs = ~np.isnan(mean_obs_ts) & ~np.isnan(mean_model_ts)
mask_corr = ~np.isnan(mean_corr_ts) & ~np.isnan(mean_model_ts)

# Bias / cRMSE
bias_obs = mean_model_ts[mask_obs] - mean_obs_ts[mask_obs]
cRMSE_obs = np.sqrt(np.mean((bias_obs - np.mean(bias_obs))**2))
bias_mean_obs = np.mean(bias_obs)

bias_corr = mean_model_ts[mask_corr] - mean_corr_ts[mask_corr]
cRMSE_corr = np.sqrt(np.mean((bias_corr - np.mean(bias_corr))**2))
bias_mean_corr = np.mean(bias_corr)

# Daily correlations (full spatial fields)
daily_corr_obs = []
daily_corr_corr = []

for r in results:
    obs = r['obs'].flatten()
    model = r['model'].flatten()
    corr = r['corr'].flatten()
    
    mask = ~np.isnan(obs) & ~np.isnan(model)
    if np.sum(mask) > 1:
        daily_corr_obs.append(np.corrcoef(obs[mask], model[mask])[0,1])
    
    mask_corr = ~np.isnan(corr) & ~np.isnan(model)
    if np.sum(mask_corr) > 1:
        daily_corr_corr.append(np.corrcoef(corr[mask_corr], model[mask_corr])[0,1])

# Mean daily correlation
mean_corr_obs = np.nanmean(daily_corr_obs)
mean_corr_corr = np.nanmean(daily_corr_corr)

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

# Top subplot: daily means
axs[0].plot(dates, mean_obs_ts, label='Obs', color='tab:blue')
axs[0].plot(dates, mean_model_ts, label='Model', color='tab:orange')
axs[0].plot(dates, mean_corr_ts, label='Corrected Obs', color='tab:green')
axs[0].set_ylabel('XCO2 [ppm]')
axs[0].set_title(
    f'Daily mean XCO2 | '
    f'Model-Obs Bias={bias_mean_obs:.2f}, RMSE={cRMSE_obs:.2f}, r={mean_corr_obs:.2f} | '
    f'Model-Corr.Obs. Bias={bias_mean_corr:.2f}, RMSE={cRMSE_corr:.2f}, r={mean_corr_corr:.2f}'
)
axs[0].legend()
axs[0].xaxis.set_minor_locator(mdates.MonthLocator())
axs[0].grid(True)

# Bottom subplot: differences
axs[1].plot(dates, mean_model_ts - mean_obs_ts, label='Model - Obs', color='tab:red')
axs[1].plot(dates, mean_model_ts - mean_corr_ts, label='Model - Corrected Obs', color='tab:purple')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('ΔXCO2 [ppm]')
axs[1].legend()
axs[1].xaxis.set_minor_locator(mdates.MonthLocator())
axs[1].grid(True)

# Format x-axis nicely
axs[1].xaxis.set_major_locator(mdates.AutoDateLocator())
axs[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axs[1].xaxis.get_major_locator()))

plt.savefig(os.path.join(outdir, 'XCO2_timeseries_with_diff.png'), dpi=300)
plt.close(fig)

# plt.figure(figsize=(8,5))
# plt.plot(dates, mean_obs_ts, label='Obs')
# plt.plot(dates, mean_model_ts, label='Model')
# plt.plot(dates, mean_corr_ts, label='Corrected Obs')
# plt.xlabel('Date')
# plt.ylabel('XCO2 [ppm]')
# plt.title('Daily mean XCO2 timeseries')
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(os.path.join(outdir, 'XCO2_timeseries.png'), dpi=300)
# plt.close()
# Create figure



# ----------------------
# ANNUAL SCATTER MAP WITH TOLERANCE (VECTORIZED)
# ----------------------
# ----------------------
# LOAD MODEL GRID
# ----------------------
grid_file = "/exports/geos.ed.ac.uk/palmer_group/run_test_2x25/OutputDir/GEOSChem.SatDiagn.20210101_0000z.nc4"
ds_grid = xr.open_dataset(grid_file)
lon_grid = ds_grid.lon.values
lat_grid = ds_grid.lat.values
ds_grid.close()

# Compute grid edges
dlat = np.diff(lat_grid).mean()
dlon = np.diff(lon_grid).mean()
lat_edges = np.concatenate(([lat_grid[0] - dlat/2], (lat_grid[:-1] + lat_grid[1:]) / 2, [lat_grid[-1] + dlat/2]))
lon_edges = np.concatenate(([lon_grid[0] - dlon/2], (lon_grid[:-1] + lon_grid[1:]) / 2, [lon_grid[-1] + dlon/2]))

n_lat = len(lat_grid)
n_lon = len(lon_grid)

# ----------------------
# Daily arrays
# ----------------------
# Collect all unique dates
dates = sorted(pd.to_datetime(r["date"]).normalize() for r in results)
unique_dates = np.unique(dates)
n_days = len(unique_dates)

# Initialize daily arrays with zeros, not NaNs
obs_daily = np.zeros((n_days, n_lat, n_lon))
model_daily = np.zeros((n_days, n_lat, n_lon))
corr_daily = np.zeros((n_days, n_lat, n_lon))

# Count array
count_daily = np.zeros((n_days, n_lat, n_lon))

# Map from date to index
date_to_idx = {d: i for i, d in enumerate(unique_dates)}

# ----------------------
# Bin data for each day
# ----------------------
for r in results:
    day_idx = date_to_idx[pd.to_datetime(r["date"]).normalize()]
    lon_idx = np.digitize(r["lon"], lon_edges) - 1
    lat_idx = np.digitize(r["lat"], lat_edges) - 1

    valid = (lon_idx >= 0) & (lon_idx < n_lon) & (lat_idx >= 0) & (lat_idx < n_lat)

    # Add obs, model, corr to daily arrays using np.add.at
    np.add.at(obs_daily[day_idx], (lat_idx[valid], lon_idx[valid]), r["obs"][valid])
    np.add.at(model_daily[day_idx], (lat_idx[valid], lon_idx[valid]), r["model"][valid])
    np.add.at(corr_daily[day_idx], (lat_idx[valid], lon_idx[valid]), r["corr"][valid])
    np.add.at(count_daily[day_idx], (lat_idx[valid], lon_idx[valid]), 1)


# Divide by count where >0
mask = count_daily > 0
obs_daily[mask] /= count_daily[mask]
model_daily[mask] /= count_daily[mask]
corr_daily[mask] /= count_daily[mask]
# Set all grid points that never had any obs to NaN
obs_daily[count_daily == 0] = np.nan
model_daily[count_daily == 0] = np.nan
corr_daily[count_daily == 0] = np.nan

# ----------------------
# Save as NetCDF
# ----------------------
out_nc = os.path.join(outdir, "daily_binned_XCO2.nc")
ds_out = xr.Dataset(
    {
        "obs": (("time", "lat", "lon"), obs_daily),
        "model": (("time", "lat", "lon"), model_daily),
        "corrected": (("time", "lat", "lon"), corr_daily),
        "counts": (("time", "lat", "lon"), count_daily)
    },
    coords={
        "time": unique_dates,
        "lat": lat_grid,
        "lon": lon_grid
    }
)
ds_out.to_netcdf(out_nc)
print(f"Saved daily binned XCO2 to {out_nc}")


ds_daily = xr.open_dataset(out_nc)

# # ----------------------
# # PRECOMPUTE COLORBAR LIMITS
# # ----------------------
# all_obs = ds_daily.obs.values
# all_model = ds_daily.model.values

# # Mean maps
# vmin_map = np.nanpercentile([all_obs, all_model], 1)
# vmax_map = np.nanpercentile([all_obs, all_model], 95)

# # Difference maps using 1-99 percentile to avoid outliers
# diff_all = all_model - all_obs
# vmin_diff = np.nanpercentile(diff_all, 1)
# vmax_diff = np.nanpercentile(diff_all, 99)
# diff_abs_map = max(abs(vmin_diff), abs(vmax_diff))

# # Correlation and std maps
# std_abs_all = np.nanstd(diff_all)  # global std for scaling


# # Bias limits
# vmin_bias = vmin_diff
# vmax_bias = vmax_diff
# bias_abs = diff_abs_map

# # Std limits (avoid outliers)
# vmax_std = np.nanpercentile(np.nanstd(diff_all, axis=0).flatten(), 99)

# # Correlation limits
# vmin_corr, vmax_corr = -1, 1


def precompute_colorbar_limits(ds, obs_var='obs', model_var='model', perc_low=1, perc_high=99):

    obs = ds[obs_var].values
    model = ds[model_var].values

    # Mean maps (Obs / Model)
    vmin_map = np.nanpercentile([obs, model], perc_low)
    vmax_map = np.nanpercentile([obs, model], perc_high)

    # Difference maps (model - obs)
    diff_all = model - obs
    vmin_diff = np.nanpercentile(diff_all, perc_low)
    vmax_diff = np.nanpercentile(diff_all, perc_high)
    diff_abs_map = max(abs(vmin_diff), abs(vmax_diff))

    # Bias limits
    vmin_bias = vmin_diff
    vmax_bias = vmax_diff
    bias_abs = diff_abs_map

    # Std maps
    std_abs_all = np.nanstd(diff_all)  # global std
    vmax_std = np.nanpercentile(np.nanstd(diff_all, axis=0).flatten(), perc_high)

    # Correlation
    vmin_corr, vmax_corr = -1, 1

    return (vmin_map, vmax_map, diff_abs_map,
            vmin_bias, vmax_bias, bias_abs,
            vmax_std, vmin_corr, vmax_corr)

vmin_map, vmax_map, diff_abs_map, vmin_bias, vmax_bias, bias_abs, vmax_std, vmin_corr, vmax_corr = \
    precompute_colorbar_limits(ds_daily, obs_var='obs', model_var='model')

# vmin_map_c, vmax_map_c, diff_abs_map_c, vmin_bias_c, vmax_bias_c, bias_abs_c, vmax_std_c, vmin_corr_c, vmax_corr_c = \
#     precompute_colorbar_limits(ds_daily, obs_var='corrected', model_var='model')

# ----------------------
# PLOTTING FUNCTIONS
# ----------------------
def plot_yearly_maps(ds_daily, year, outdir, vmin=vmin_map, vmax=vmax_map, diff_abs=diff_abs_map, typeobs='obs'):
    time_sel = ds_daily.time.dt.year == year
    obs_map = ds_daily[typeobs].sel(time=time_sel).mean(dim="time").values
    model_map = ds_daily.model.sel(time=time_sel).mean(dim="time").values
    diff_map = model_map - obs_map

    lon = ds_daily.lon.values
    lat = ds_daily.lat.values

    fig, axs = plt.subplots(3, 1, figsize=(9, 7),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            constrained_layout=True)

    for ax in axs:
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_extent([-180, 180, -90, 90])
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = gl.right_labels = False

    # Obs
    im0 = axs[0].pcolormesh(lon, lat, obs_map, vmin=vmin, vmax=vmax,
                            cmap='viridis', transform=ccrs.PlateCarree())
    axs[0].set_title(f'{year} Annual mean Observations (mean={np.nanmean(obs_map):.1f})')
    plt.colorbar(im0, ax=axs[0], shrink=0.5, pad=0.01, label='XCO2 [ppm]')

    # Model
    im1 = axs[1].pcolormesh(lon, lat, model_map, vmin=vmin, vmax=vmax,
                            cmap='viridis', transform=ccrs.PlateCarree())
    axs[1].set_title(f'{year} Annual mean Model (mean={np.nanmean(model_map):.1f})')
    plt.colorbar(im1, ax=axs[1], shrink=0.5, pad=0.01, label='XCO2 [ppm]')

    # Difference with percentile scaling
    diff_clipped = np.clip(diff_map, -diff_abs, diff_abs)
    im2 = axs[2].pcolormesh(lon, lat, diff_clipped, vmin=-diff_abs, vmax=diff_abs,
                            cmap='bwr', transform=ccrs.PlateCarree())
    axs[2].set_title(f'{year} Model - Obs (mean={np.nanmean(diff_map):.1f}, std={np.nanstd(diff_map):.1f})')
    plt.colorbar(im2, ax=axs[2], shrink=0.5, pad=0.01, label='ΔXCO2 [ppm]')

    filename = os.path.join(outdir, f"{typeobs}{year}_annual_XCO2_comparison.png")
    print('Plotting', filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_yearly_stats(ds_daily, year, outdir, vmin_bias=bias_abs, vmax_bias=bias_abs,
                      vmax_std=vmax_std, vmin_corr=vmin_corr, vmax_corr=vmax_corr, typeobs='obs'):
    time_sel = ds_daily.time.dt.year == year
    obs = ds_daily[typeobs].sel(time=time_sel).values
    model = ds_daily.model.sel(time=time_sel).values

    n_lat, n_lon = obs.shape[1], obs.shape[2]
    bias_map = np.full((n_lat, n_lon), np.nan)
    std_map = np.full((n_lat, n_lon), np.nan)
    corr_map = np.full((n_lat, n_lon), np.nan)

    # Compute per-grid-cell stats
    for i in range(n_lat):
        for j in range(n_lon):
            mask = ~np.isnan(obs[:, i, j]) & ~np.isnan(model[:, i, j])
            if np.sum(mask) > 1:
                diff = model[mask, i, j] - obs[mask, i, j]
                bias_map[i, j] = np.mean(diff)
                std_map[i, j] = np.std(diff)
                corr_map[i, j] = np.corrcoef(obs[mask, i, j], model[mask, i, j])[0,1]

    # Compute mean/std for titles
    bias_mean, bias_std = np.nanmean(bias_map), np.nanstd(bias_map)
    std_mean, std_std   = np.nanmean(std_map), np.nanstd(std_map)
    corr_mean, corr_std = np.nanmean(corr_map), np.nanstd(corr_map)

    lon = ds_daily.lon.values
    lat = ds_daily.lat.values

    fig, axs = plt.subplots(3, 1, figsize=(9, 12),
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            constrained_layout=True)

    for ax in axs:
        ax.coastlines()
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.set_extent([-180, 180, -90, 90])

    # Bias
    im0 = axs[0].pcolormesh(lon, lat, bias_map, cmap='bwr', vmin=-vmax_bias, vmax=vmax_bias,
                            transform=ccrs.PlateCarree())
    axs[0].set_title(f'{year} Bias (mean={bias_mean:.2f}, std={bias_std:.2f})')
    plt.colorbar(im0, ax=axs[0], shrink=0.6, pad=0.01, label='ΔXCO2 [ppm]')

    # Std
    std_clipped = np.clip(std_map, 0, vmax_std)
    im1 = axs[1].pcolormesh(lon, lat, std_clipped, cmap='Reds', vmin=0, vmax=vmax_std,
                            transform=ccrs.PlateCarree())
    axs[1].set_title(f'{year} Std (mean={std_mean:.2f}, std={std_std:.2f})')
    plt.colorbar(im1, ax=axs[1], shrink=0.6, pad=0.01, label='XCO2 [ppm]')

    # Correlation
    im2 = axs[2].pcolormesh(lon, lat, corr_map, cmap='coolwarm', vmin=vmin_corr, vmax=vmax_corr,
                            transform=ccrs.PlateCarree())
    axs[2].set_title(f'{year} Corr (mean={corr_mean:.2f}, std={corr_std:.2f})')
    plt.colorbar(im2, ax=axs[2], shrink=0.6, pad=0.01, label='r')

    filename = os.path.join(outdir, f"{typeobs}{year}_XCO2_stats.png")
    print('Plotting', filename)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


# ----------------------
# Plot yearly stats
# ----------------------
years = np.unique(ds_daily.time.dt.year)
for year in years:
    # plot_yearly_maps(ds_daily, year, outdir)
    # plot_yearly_stats(ds_daily, year, outdir)
    # For corrected obs
    plot_yearly_maps(ds_daily, year, outdir,
                        typeobs='corrected')
    # For corrected obs
    plot_yearly_stats(ds_daily, year, outdir,
                    typeobs='corrected')
# # ----------------------
# # ACCUMULATE PER YEAR
# # ----------------------
# # Collect all unique years
# years = sorted(set(pd.to_datetime(r["date"]).year for r in results))

# for year in years:
#     sum_obs = np.zeros((n_lat, n_lon))
#     sum_model = np.zeros((n_lat, n_lon))
#     count = np.zeros((n_lat, n_lon))

#     for r in results:
#         r_year = pd.to_datetime(r["date"]).year
#         if r_year != year:
#             continue

#         lon_idx = np.digitize(r["lon"], lon_edges) - 1
#         lat_idx = np.digitize(r["lat"], lat_edges) - 1

#         valid = (
#             (lon_idx >= 0) & (lon_idx < n_lon) &
#             (lat_idx >= 0) & (lat_idx < n_lat)
#         )

#         np.add.at(sum_obs, (lat_idx[valid], lon_idx[valid]), r["obs"][valid])
#         np.add.at(sum_model, (lat_idx[valid], lon_idx[valid]), r["model"][valid])
#         np.add.at(count, (lat_idx[valid], lon_idx[valid]), 1)

#     mean_obs_map = np.where(count > 0, sum_obs / count, np.nan)
#     mean_model_map = np.where(count > 0, sum_model / count, np.nan)
#     diff_map = mean_model_map - mean_obs_map

#     # ----------------------
#     # PLOT MAPS FOR THIS YEAR
#     # ----------------------
#     fig, axs = plt.subplots(3, 1, figsize=(9, 7),
#                             subplot_kw={'projection': ccrs.PlateCarree()},
#                             constrained_layout=True)

#     vmin = np.nanmin([mean_obs_map, mean_model_map])
#     vmax = np.nanmax([mean_obs_map, mean_model_map])
#     diff_abs = np.nanmax(np.abs(diff_map))

#     for ax in axs:
#         ax.coastlines()
#         ax.add_feature(cfeature.BORDERS, linestyle=':')
#         ax.set_extent([-180, 180, -90, 90])
#         gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
#         gl.top_labels = gl.right_labels = False

#     # Obs
#     im0 = axs[0].pcolormesh(lon_grid, lat_grid, mean_obs_map,
#                             vmin=vmin, vmax=vmax, cmap='viridis',
#                             transform=ccrs.PlateCarree())
#     axs[0].set_title(f'{year} Annual mean Observations (mean={np.nanmean(mean_obs_map):.1f})')
#     plt.colorbar(im0, ax=axs[0], shrink=0.5, pad=0.01, label='XCO2 [ppm]')

#     # Model
#     im1 = axs[1].pcolormesh(lon_grid, lat_grid, mean_model_map,
#                             vmin=vmin, vmax=vmax, cmap='viridis',
#                             transform=ccrs.PlateCarree())
#     axs[1].set_title(f'{year} Annual mean Model (mean={np.nanmean(mean_model_map):.1f})')
#     plt.colorbar(im1, ax=axs[1], shrink=0.5, pad=0.01, label='XCO2 [ppm]')

#     # Difference
#     im2 = axs[2].pcolormesh(lon_grid, lat_grid, diff_map,
#                             vmin=-diff_abs, vmax=diff_abs, cmap='bwr',
#                             transform=ccrs.PlateCarree())
#     axs[2].set_title(f'{year} Model - Obs (mean={np.nanmean(diff_map):.1f}, std={np.nanstd(diff_map):.1f})')
#     plt.colorbar(im2, ax=axs[2], shrink=0.5, pad=0.01, label='ΔXCO2 [ppm]')

#     filename = os.path.join(outdir, f"{year}_annual_XCO2_comparison.png")
#     print('Plotting', filename)
#     plt.savefig(filename, dpi=300, bbox_inches='tight')
#     plt.close(fig)
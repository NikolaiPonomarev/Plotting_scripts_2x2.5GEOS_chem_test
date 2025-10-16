import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import os
from matplotlib.colors import SymLogNorm, LogNorm

def plot_all_emission_variables(emis_dir, year):
    # Output directory for plots
    outdir = "/exports/geos.ed.ac.uk/palmer_group/oco2_sample_v11r/plotting_scripts/emissions"
    os.makedirs(outdir, exist_ok=True)

    # Collect monthly emission files for the chosen year
    files = sorted([os.path.join(emis_dir, f) 
                    for f in os.listdir(emis_dir)
                    if f.startswith("HEMCO_diagnostics") and f.endswith(".nc") and str(year) in f])
    if not files:
        raise FileNotFoundError(f"No emission files for {year} found in {emis_dir}")

    print(f"Found {len(files)} monthly emission files for {year}")
    print("Loading with xarray...")

    # Open dataset using context manager to ensure proper closing
    with xr.open_mfdataset(files, combine="by_coords", chunks='auto') as ds:

        # Select variables that represent emissions
        emis_vars = [v for v in ds.data_vars if v.startswith("Emis")]
        print(f"Emission variables found: {', '.join(emis_vars)}")

        # Compute total emissions once for percentage calculation
        total_emis_all = None
        if 'EmisCO2_Total' in ds:
            total_emis_all = ds['EmisCO2_Total']
            if 'lev' in total_emis_all.dims:
                total_emis_all = total_emis_all.sum(dim='lev', skipna=True)
            # SUM over time + lat + lon for total yearly emissions
            total_emis_all = float(total_emis_all.mean(dim='time').sum(dim=('lat','lon')).compute())

        for varname in emis_vars:
            print(f"Processing {varname}...")
            emis = ds[varname]

            # Handle fill values if present
            fill = emis.attrs.get("_FillValue", None)
            if fill is not None:
                emis = emis.where(emis != fill)

            # Sum over vertical levels if present
            if "lev" in emis.dims:
                emis = emis.sum(dim="lev", skipna=True)

            # Compute yearly mean
            emis_mean = emis.mean(dim="time")

            # Extract coordinates
            lon, lat = emis_mean.lon, emis_mean.lat

            # Choose color scale
            if np.any(emis_mean < 0) or np.all(emis_mean == 0):
                # SymLogNorm for negative or all-zero cases
                linthresh = 1e-14
                vmax = float(np.max(np.abs(emis_mean)).compute())
                norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax)
                cmap = "seismic"
                emis_plot = emis_mean
            else:
                # LogNorm for positive values (mask zeros)
                emis_plot = np.where(emis_mean > 0, emis_mean, np.nan)
                vmin = float(np.nanmin(emis_plot))
                vmax = float(np.nanmax(emis_plot))
                norm = LogNorm(vmin=vmin, vmax=vmax)
                cmap = "inferno"

            # Plot setup
            fig = plt.figure(figsize=(9, 5))
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(cfeature.BORDERS, linestyle=":")
            ax.set_global()

            im = ax.pcolormesh(lon, lat, emis_plot, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02, label="CO₂ emissions [kg m⁻² s⁻¹]")

            # Total emissions and percentage — SUM over time + lat + lon
            total_emis = float(emis_mean.sum(dim=('lat','lon')).compute())
            perc_str = ""
            if total_emis_all is not None:
                perc = 100 * total_emis / total_emis_all
                perc_str = f", {perc:.1f}% of total"

            long_name = emis.attrs.get("long_name", varname)
            ax.set_title(f"{long_name}\nYearly mean {year}, Total={total_emis:.2e} kg{perc_str}")

            # Save figure
            filename = os.path.join(outdir, f"{varname}_{year}.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {filename}")

# Main loop
emis_dir = "/exports/geos.ed.ac.uk/palmer_group/run_test_2x25/OutputDir/"
for year in range(2021, 2025):
    plot_all_emission_variables(emis_dir, year=year)
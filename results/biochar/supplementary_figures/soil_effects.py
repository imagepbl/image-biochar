from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter

from timer.cdr.cdr_util import land_mask


"""
Three-panel map of biochar-induced soil and crop effects.

Panels
------
a. Biochar-induced crop yield effects
b. Biochar-induced effects on soil N2O emissions
c. Biochar induced effects on soil CH4 emissions

Model variables
---------------
The module reads the gridded TIMER/PRISM outputs:

    yield_effects.nc
        variable: yield_effects

    n2o_effects.nc
        variable: n2o_effects

    ch4_effects.nc
        variable: ch4_effects

All three outputs are already expressed as percentage changes.
Yield effects are positive, while N2O and CH4 effects are non-positive
(negative values indicate emission reductions).

Map treatment
-------------
- All SSP2 cropland is shown in light grey.
- Only non-zero effect values are coloured.
- Yield effects use a green scale.
- N2O effects use a brown scale, reversed so that larger negative effects
  are darker.
- CH4 effects use a blue scale, reversed so that larger negative effects
  are darker.
- The plot titles intentionally omit the year.
"""


# =============================================================================
# Configuration
# =============================================================================

CDR_OUTPUT_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output"
)
SCENARIO_DIR = CDR_OUTPUT_DIR / "biochar_c400"

YIELD_FILE = SCENARIO_DIR / "yield_effects.nc"
YIELD_VAR = "yield_effects"

N2O_FILE = SCENARIO_DIR / "n2o_effects.nc"
N2O_VAR = "n2o_effects"

CH4_FILE = SCENARIO_DIR / "ch4_effects.nc"
CH4_VAR = "ch4_effects"

SSP2_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\biochar\SSP2_M_new"
)
GLANDCOVER_FILE = SSP2_DIR / "GLANDCOVER.NC"
GAREA_FILE = SSP2_DIR / "GAREA.NC"

OUTPUT_DIR = CDR_OUTPUT_DIR / "figures"
OUTPUT_STEM = "biochar_effects_yield_n2o_ch4_3panel"

# Files may be static or may optionally contain a time dimension.
BACKGROUND_YEAR = 2100
DATA_YEAR = 2100

# In the detailed representation, yield effects are stored for both
# technologies but only technology index 0 carries the gridded response.
YIELD_TECH_INDEX = 0

RESOLUTION = "5m"
CROPLAND_INDEX = 0
MIN_CROPLAND_HA = 1.0

SAVE_DPI = 600
SHOW_FIGURE = True

FIGSIZE = (15.8, 5.4)

BACKGROUND_COLOR = "0.88"
BACKGROUND_POINT_SIZE = 0.28
DATA_POINT_SIZE = 0.42
COASTLINE_WIDTH = 0.35
BORDER_WIDTH = 0.20

YIELD_CMAP = "Greens"
N2O_CMAP = "YlOrBr_r"
CH4_CMAP = "PuBu_r"

COLORBAR_SHRINK = 0.82
COLORBAR_ASPECT = 28
COLORBAR_FRACTION = 0.050
COLORBAR_PAD = 0.055

# Robust scaling settings.
POSITIVE_VMAX_PERCENTILE = 98.0
NEGATIVE_VMIN_PERCENTILE = 2.0

YIELD_FIXED_VMAX = None
N2O_FIXED_VMIN = None
CH4_FIXED_VMIN = None


# =============================================================================
# Generic helpers
# =============================================================================

def _find_dim(da, candidates, required=True):
    lower = {d.lower(): d for d in da.dims}

    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any dimension in {candidates} for {da.name}. "
            f"Available dimensions: {da.dims}"
        )

    return None


def _year_values(coord):
    values = coord.values

    if np.issubdtype(values.dtype, np.datetime64):
        return coord.dt.year.values.astype(int)

    return np.asarray(values).astype(int)


def _select_year(da, year):
    time_dim = _find_dim(da, ["time", "year"], required=False)

    if time_dim is None:
        return da

    years = _year_values(da[time_dim])
    matches = np.where(years == int(year))[0]

    if matches.size == 0:
        raise ValueError(
            f"Year {year} not available in {da.name}. "
            f"Available years: {np.unique(years).tolist()}"
        )

    return da.isel({time_dim: int(matches[0])})


def _panel_title(letter, title):
    return rf"$\bf{{{letter}.}}$ {title}"


def _formatter():
    def _format(x, pos):
        ax = abs(x)

        if ax >= 1000:
            return f"{x:,.0f}"
        if ax >= 10:
            return f"{x:.0f}"
        if ax >= 1:
            return f"{x:.1f}"
        if ax >= 0.1:
            return f"{x:.1f}"
        if ax == 0:
            return "0"

        return f"{x:.2f}"

    return FuncFormatter(_format)


def _nice_number(value):
    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        return 1.0

    exponent = np.floor(np.log10(value))
    scale = 10.0 ** exponent
    fraction = value / scale

    candidates = np.array(
        [1.0, 2.0, 2.5, 5.0, 10.0],
        dtype=np.float64,
    )

    index = int(np.argmin(np.abs(candidates - fraction)))
    return float(candidates[index] * scale)


def _round_up(value):
    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        return 1.0

    exponent = np.floor(np.log10(value))
    scale = 10.0 ** exponent
    fraction = value / scale

    for candidate in (1.0, 2.0, 2.5, 5.0, 10.0):
        if fraction <= candidate + 1e-12:
            return float(candidate * scale)

    return float(10.0 * scale)


def _round_down_negative(value):
    """
    Round a negative number away from zero to a clean value.
    """
    value = float(value)

    if not np.isfinite(value) or value >= 0.0:
        return -1.0

    return -_round_up(abs(value))


def _positive_vmax(values, fixed_vmax=None):
    if fixed_vmax is not None:
        fixed_vmax = float(fixed_vmax)

        if fixed_vmax <= 0.0:
            raise ValueError("Fixed positive vmax must be > 0.")

        return fixed_vmax

    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0.0)

    if not np.any(valid):
        return 1.0

    vmax = float(
        np.nanpercentile(
            values[valid],
            POSITIVE_VMAX_PERCENTILE,
        )
    )

    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = float(np.nanmax(values[valid]))

    return _round_up(vmax)


def _negative_vmin(values, fixed_vmin=None):
    if fixed_vmin is not None:
        fixed_vmin = float(fixed_vmin)

        if fixed_vmin >= 0.0:
            raise ValueError("Fixed negative vmin must be < 0.")

        return fixed_vmin

    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values) & (values < 0.0)

    if not np.any(valid):
        return -1.0

    vmin = float(
        np.nanpercentile(
            values[valid],
            NEGATIVE_VMIN_PERCENTILE,
        )
    )

    if not np.isfinite(vmin) or vmin >= 0.0:
        vmin = float(np.nanmin(values[valid]))

    return _round_down_negative(vmin)


def _ticks_positive(vmax, target_intervals=4):
    vmax = float(vmax)

    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.array([0.0, 1.0])

    raw_step = vmax / float(target_intervals)
    step = _nice_number(raw_step)

    ticks = np.arange(
        0.0,
        vmax + 0.5 * step,
        step,
        dtype=np.float64,
    )
    ticks = ticks[ticks <= vmax * (1.0 + 1e-10)]

    if ticks.size == 0 or not np.isclose(ticks[-1], vmax):
        ticks = np.append(ticks, vmax)

    return np.unique(ticks)


def _ticks_negative(vmin, target_intervals=4):
    """
    Rounded ticks for a negative-to-zero colour scale.
    """
    vmin = float(vmin)

    if not np.isfinite(vmin) or vmin >= 0.0:
        return np.array([-1.0, 0.0])

    span = abs(vmin)
    raw_step = span / float(target_intervals)
    step = _nice_number(raw_step)

    ticks = np.arange(
        vmin,
        0.0 + 0.5 * step,
        step,
        dtype=np.float64,
    )

    if ticks.size == 0 or not np.isclose(ticks[0], vmin):
        ticks = np.insert(ticks, 0, vmin)

    if not np.isclose(ticks[-1], 0.0):
        ticks = np.append(ticks, 0.0)

    ticks = np.unique(np.round(ticks, 10))
    ticks = ticks[(ticks >= vmin - 1e-10) & (ticks <= 0.0 + 1e-10)]

    return ticks


def _check_required_files():
    files = {
        "Yield effects": YIELD_FILE,
        "N2O effects": N2O_FILE,
        "CH4 effects": CH4_FILE,
        "SSP2 GLANDCOVER": GLANDCOVER_FILE,
        "SSP2 GAREA": GAREA_FILE,
    }

    missing = [
        f"{label}: {path}"
        for label, path in files.items()
        if not Path(path).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required input file(s) not found:\n"
            + "\n".join(f"  - {item}" for item in missing)
        )


# =============================================================================
# SSP2 cropland and LandGrid coordinates
# =============================================================================

class CroplandData:
    def __init__(
        self,
        glandcover_file=GLANDCOVER_FILE,
        garea_file=GAREA_FILE,
        cropland_index=CROPLAND_INDEX,
        resolution=RESOLUTION,
    ):
        self.glandcover_file = Path(glandcover_file)
        self.garea_file = Path(garea_file)
        self.cropland_index = int(cropland_index)
        self.resolution = resolution

        self.ds_lc = xr.open_dataset(
            self.glandcover_file,
            engine="netcdf4",
        )
        self.ds_area = xr.open_dataset(
            self.garea_file,
            engine="netcdf4",
        )

        self.lc = self.ds_lc["GLANDCOVER"]
        self.garea = self.ds_area["GAREA"]

        self.lat_dim = _find_dim(
            self.lc,
            ["lat", "latitude", "y"],
        )
        self.lon_dim = _find_dim(
            self.lc,
            ["lon", "longitude", "x"],
        )
        self.lc_dim = _find_dim(
            self.lc,
            ["landcover", "land_cover", "nglndcov", "cover", "class"],
        )

        self.mask_land = land_mask(self.resolution)
        self.lats, self.lons = self._land_coordinates()

    def close(self):
        self.ds_lc.close()
        self.ds_area.close()

    def _land_coordinates(self):
        lat = np.asarray(
            self.ds_lc[self.lat_dim].values,
            dtype=np.float64,
        )
        lon = np.asarray(
            self.ds_lc[self.lon_dim].values,
            dtype=np.float64,
        )

        if np.nanmax(lon) > 180.0:
            lon = ((lon + 180.0) % 360.0) - 180.0

        lon2d, lat2d = np.meshgrid(lon, lat)

        if self.mask_land.size != lon2d.size:
            raise ValueError(
                "land_mask size does not match SSP2 lat-lon grid: "
                f"{self.mask_land.size} vs {lon2d.size}"
            )

        return (
            lat2d.reshape(-1)[self.mask_land],
            lon2d.reshape(-1)[self.mask_land],
        )

    def _global_or_land_to_landgrid(
        self,
        da,
        lat_dim=None,
        lon_dim=None,
    ):
        if lat_dim is None:
            lat_dim = _find_dim(
                da,
                ["lat", "latitude", "y"],
                required=False,
            )

        if lon_dim is None:
            lon_dim = _find_dim(
                da,
                ["lon", "longitude", "x"],
                required=False,
            )

        if lat_dim is not None and lon_dim is not None:
            da = da.transpose(lat_dim, lon_dim)
            values = np.asarray(
                da.values,
                dtype=np.float64,
            ).reshape(-1)

            if values.size != self.mask_land.size:
                raise ValueError(
                    f"{da.name}: global-grid length {values.size} does not "
                    f"match land-mask length {self.mask_land.size}."
                )

            return values[self.mask_land]

        values = np.asarray(
            da.values,
            dtype=np.float64,
        ).squeeze()

        if values.ndim != 1:
            raise ValueError(
                f"{da.name}: expected 1D LandGrid or 2D lat-lon data, "
                f"got shape {values.shape}."
            )

        if values.size != self.lats.size:
            raise ValueError(
                f"{da.name}: LandGrid length {values.size} does not match "
                f"expected {self.lats.size}."
            )

        return values

    def cropland_area_ha(self, year):
        lc_year = _select_year(
            self.lc,
            year,
        )

        crop_fraction = lc_year.isel(
            {
                self.lc_dim: self.cropland_index
            }
        )

        crop_fraction_land = self._global_or_land_to_landgrid(
            crop_fraction,
            self.lat_dim,
            self.lon_dim,
        )

        area_year = _select_year(
            self.garea,
            year,
        ).squeeze(drop=True)

        area_land = self._global_or_land_to_landgrid(
            area_year
        )

        crop_fraction_land = np.clip(
            crop_fraction_land,
            0.0,
            None,
        )

        # GAREA is km2; convert to ha.
        return crop_fraction_land * area_land * 100.0


# =============================================================================
# Output loading
# =============================================================================

def _load_landgrid_variable(
    nc_file,
    var_name,
    expected_landgrid_length,
    year=None,
    tech_index=None,
):
    nc_file = Path(nc_file)

    with xr.open_dataset(
        nc_file,
        engine="netcdf4",
    ) as ds:
        if var_name not in ds.data_vars:
            raise KeyError(
                f"{var_name} not found in {nc_file}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[var_name]

        if year is not None:
            da = _select_year(
                da,
                year,
            )

        tech_dim = _find_dim(
            da,
            ["tech", "technology"],
            required=False,
        )

        if tech_index is not None:
            if tech_dim is None:
                raise ValueError(
                    f"{var_name} has no technology dimension."
                )

            da = da.isel({tech_dim: int(tech_index)})

        da = da.squeeze(drop=True)

        values = np.asarray(
            da.values,
            dtype=np.float64,
        ).squeeze()

        units = da.attrs.get(
            "units",
            ds[var_name].attrs.get("units", ""),
        )
        description = da.attrs.get(
            "description",
            ds[var_name].attrs.get("description", ""),
        )

    if values.ndim != 1:
        raise ValueError(
            f"{nc_file.name}:{var_name} did not resolve to a 1D LandGrid "
            f"array. Shape={values.shape}"
        )

    if values.size != expected_landgrid_length:
        raise ValueError(
            f"{nc_file.name}:{var_name} has {values.size} grid cells, "
            f"expected {expected_landgrid_length}."
        )

    return {
        "values": values,
        "units": units,
        "description": description,
        "file": nc_file,
    }


# =============================================================================
# Diagnostics
# =============================================================================

def _summarise_field(label, values, cropland_area_ha):
    values = np.asarray(values, dtype=np.float64)
    cropland_area_ha = np.asarray(cropland_area_ha, dtype=np.float64)

    crop = (
        np.isfinite(cropland_area_ha)
        & (cropland_area_ha > MIN_CROPLAND_HA)
    )
    nonzero = crop & np.isfinite(values) & (values != 0.0)

    if np.any(nonzero):
        x = values[nonzero]
        quantiles = np.nanpercentile(x, [5, 25, 50, 75, 95])
        mean = float(np.nanmean(x))
        minimum = float(np.nanmin(x))
        maximum = float(np.nanmax(x))
    else:
        quantiles = np.full(5, np.nan)
        mean = np.nan
        minimum = np.nan
        maximum = np.nan

    return {
        "Metric": label,
        "Non-zero cells": int(np.count_nonzero(nonzero)),
        "Cropland cells": int(np.count_nonzero(crop)),
        "Mean non-zero value": mean,
        "P05": float(quantiles[0]),
        "P25": float(quantiles[1]),
        "Median": float(quantiles[2]),
        "P75": float(quantiles[3]),
        "P95": float(quantiles[4]),
        "Minimum": minimum,
        "Maximum": maximum,
    }


def print_quantitative_summary(
    yield_values,
    n2o_values,
    ch4_values,
    cropland_area_ha,
    yield_vmax,
    n2o_vmin,
    ch4_vmin,
):
    rows = [
        _summarise_field("Yield effects", yield_values, cropland_area_ha),
        _summarise_field("N2O effects", n2o_values, cropland_area_ha),
        _summarise_field("CH4 effects", ch4_values, cropland_area_ha),
    ]

    df = pd.DataFrame(rows).set_index("Metric")

    print("\n" + "=" * 112)
    print("BIOCHAR-INDUCED YIELD AND SOIL-GHG EFFECTS")
    print("=" * 112)
    print(df.round(3).to_string())

    print("\nColour-scale diagnostics")
    print(
        f"  Yield vmax = {yield_vmax:g} % "
        f"({'fixed' if YIELD_FIXED_VMAX is not None else f'P{POSITIVE_VMAX_PERCENTILE:g}, rounded up'})"
    )
    print(
        f"  N2O vmin  = {n2o_vmin:g} % "
        f"({'fixed' if N2O_FIXED_VMIN is not None else f'P{NEGATIVE_VMIN_PERCENTILE:g}, rounded away from zero'})"
    )
    print(
        f"  CH4 vmin  = {ch4_vmin:g} % "
        f"({'fixed' if CH4_FIXED_VMIN is not None else f'P{NEGATIVE_VMIN_PERCENTILE:g}, rounded away from zero'})"
    )

    return df


# =============================================================================
# Plotting
# =============================================================================

def _setup_map(ax):
    ax.set_global()

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white",
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAND,
        facecolor="white",
        zorder=0,
    )
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=COASTLINE_WIDTH,
        zorder=5,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=BORDER_WIDTH,
        alpha=0.55,
        zorder=5,
    )


def _plot_positive_panel(
    fig,
    ax,
    lons,
    lats,
    cropland_area_ha,
    values,
    title,
    cmap_name,
    vmax,
):
    _setup_map(ax)

    crop_mask = (
        np.isfinite(cropland_area_ha)
        & (cropland_area_ha > MIN_CROPLAND_HA)
    )

    ax.scatter(
        lons[crop_mask],
        lats[crop_mask],
        s=BACKGROUND_POINT_SIZE,
        color=BACKGROUND_COLOR,
        transform=ccrs.PlateCarree(),
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    colour_mask = (
        crop_mask
        & np.isfinite(values)
        & (values > 0.0)
    )

    norm = mpl.colors.Normalize(
        vmin=0.0,
        vmax=vmax,
    )
    cmap = plt.get_cmap(cmap_name)

    actual_max = (
        float(np.nanmax(values[colour_mask]))
        if np.any(colour_mask)
        else vmax
    )
    extend = (
        "max"
        if actual_max > vmax * (1.0 + 1e-9)
        else "neither"
    )

    sc = ax.scatter(
        lons[colour_mask],
        lats[colour_mask],
        c=values[colour_mask],
        s=DATA_POINT_SIZE,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    ax.set_title(
        title,
        loc="left",
        fontsize=11.2,
        fontweight="normal",
        pad=5,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        ticks=_ticks_positive(vmax),
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
        aspect=COLORBAR_ASPECT,
        shrink=COLORBAR_SHRINK,
        extend=extend,
        extendfrac=0.045,
    )
    cbar.set_label("%", fontsize=9.2, labelpad=4)
    cbar.ax.tick_params(labelsize=8.3, length=2.5)
    cbar.ax.xaxis.set_major_formatter(_formatter())
    cbar.outline.set_linewidth(0.5)

    return cbar


def _plot_negative_panel(
    fig,
    ax,
    lons,
    lats,
    cropland_area_ha,
    values,
    title,
    cmap_name,
    vmin,
):
    _setup_map(ax)

    crop_mask = (
        np.isfinite(cropland_area_ha)
        & (cropland_area_ha > MIN_CROPLAND_HA)
    )

    ax.scatter(
        lons[crop_mask],
        lats[crop_mask],
        s=BACKGROUND_POINT_SIZE,
        color=BACKGROUND_COLOR,
        transform=ccrs.PlateCarree(),
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    colour_mask = (
        crop_mask
        & np.isfinite(values)
        & (values < 0.0)
    )

    norm = mpl.colors.Normalize(
        vmin=vmin,
        vmax=0.0,
    )
    cmap = plt.get_cmap(cmap_name)

    actual_min = (
        float(np.nanmin(values[colour_mask]))
        if np.any(colour_mask)
        else vmin
    )
    extend = (
        "min"
        if actual_min < vmin * (1.0 + 1e-9)
        else "neither"
    )

    sc = ax.scatter(
        lons[colour_mask],
        lats[colour_mask],
        c=values[colour_mask],
        s=DATA_POINT_SIZE,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree(),
        linewidths=0,
        rasterized=True,
        zorder=2,
    )

    ax.set_title(
        title,
        loc="left",
        fontsize=11.2,
        fontweight="normal",
        pad=5,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        ticks=_ticks_negative(vmin),
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
        aspect=COLORBAR_ASPECT,
        shrink=COLORBAR_SHRINK,
        extend=extend,
        extendfrac=0.045,
    )
    cbar.set_label("%", fontsize=9.2, labelpad=4)
    cbar.ax.tick_params(labelsize=8.3, length=2.5)
    cbar.ax.xaxis.set_major_formatter(_formatter())
    cbar.outline.set_linewidth(0.5)

    return cbar


# =============================================================================
# Main
# =============================================================================

def plot_biochar_effects_figure(save=True, show=True):
    _check_required_files()

    cropland = CroplandData()

    try:
        crop_background = cropland.cropland_area_ha(BACKGROUND_YEAR)

        yield_info = _load_landgrid_variable(
            YIELD_FILE,
            YIELD_VAR,
            year=DATA_YEAR,
            tech_index=YIELD_TECH_INDEX,
            expected_landgrid_length=cropland.lats.size,
        )
        n2o_info = _load_landgrid_variable(
            N2O_FILE,
            N2O_VAR,
            year=DATA_YEAR,
            expected_landgrid_length=cropland.lats.size,
        )
        ch4_info = _load_landgrid_variable(
            CH4_FILE,
            CH4_VAR,
            year=DATA_YEAR,
            expected_landgrid_length=cropland.lats.size,
        )

        yield_map = np.where(
            crop_background > MIN_CROPLAND_HA,
            yield_info["values"],
            np.nan,
        )
        n2o_map = np.where(
            crop_background > MIN_CROPLAND_HA,
            n2o_info["values"],
            np.nan,
        )
        ch4_map = np.where(
            crop_background > MIN_CROPLAND_HA,
            ch4_info["values"],
            np.nan,
        )

        yield_vmax = _positive_vmax(yield_map, YIELD_FIXED_VMAX)
        n2o_vmin = _negative_vmin(n2o_map, N2O_FIXED_VMIN)
        ch4_vmin = _negative_vmin(ch4_map, CH4_FIXED_VMIN)

        summary = print_quantitative_summary(
            yield_values=yield_map,
            n2o_values=n2o_map,
            ch4_values=ch4_map,
            cropland_area_ha=crop_background,
            yield_vmax=yield_vmax,
            n2o_vmin=n2o_vmin,
            ch4_vmin=ch4_vmin,
        )

        print("\nInput metadata")
        print(
            f"  {YIELD_VAR}: file={yield_info['file']}, "
            f"units='{yield_info['units']}', "
            f"description='{yield_info['description']}'"
        )
        print(
            f"  {N2O_VAR}: file={n2o_info['file']}, "
            f"units='{n2o_info['units']}', "
            f"description='{n2o_info['description']}'"
        )
        print(
            f"  {CH4_VAR}: file={ch4_info['file']}, "
            f"units='{ch4_info['units']}', "
            f"description='{ch4_info['description']}'"
        )

        projection = ccrs.Robinson()

        fig, axes = plt.subplots(
            1,
            3,
            figsize=FIGSIZE,
            subplot_kw={"projection": projection},
        )

        _plot_positive_panel(
            fig=fig,
            ax=axes[0],
            lons=cropland.lons,
            lats=cropland.lats,
            cropland_area_ha=crop_background,
            values=yield_map,
            title=_panel_title("a", "Biochar-induced crop yield effects"),
            cmap_name=YIELD_CMAP,
            vmax=yield_vmax,
        )

        _plot_negative_panel(
            fig=fig,
            ax=axes[1],
            lons=cropland.lons,
            lats=cropland.lats,
            cropland_area_ha=crop_background,
            values=n2o_map,
            title=_panel_title("b", "Biochar-induced effects on soil N$_2$O emissions"),
            cmap_name=N2O_CMAP,
            vmin=n2o_vmin,
        )

        _plot_negative_panel(
            fig=fig,
            ax=axes[2],
            lons=cropland.lons,
            lats=cropland.lats,
            cropland_area_ha=crop_background,
            values=ch4_map,
            title=_panel_title("c", "Biochar induced effects on soil CH$_4$ emissions"),
            cmap_name=CH4_CMAP,
            vmin=ch4_vmin,
        )

        fig.subplots_adjust(
            left=0.02,
            right=0.99,
            top=0.93,
            bottom=0.13,
            wspace=0.08,
        )

        outputs = {
            "figure": fig,
            "axes": axes,
            "summary": summary,
            "yield_values": yield_map,
            "n2o_values": n2o_map,
            "ch4_values": ch4_map,
            "yield_vmax": yield_vmax,
            "n2o_vmin": n2o_vmin,
            "ch4_vmin": ch4_vmin,
        }

        if save:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

            png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
            pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
            summary_csv = OUTPUT_DIR / f"{OUTPUT_STEM}_summary.csv"

            fig.savefig(
                png_path,
                dpi=SAVE_DPI,
                bbox_inches="tight",
            )
            fig.savefig(
                pdf_path,
                bbox_inches="tight",
            )
            summary.to_csv(summary_csv)

            outputs.update(
                {
                    "png_path": png_path,
                    "pdf_path": pdf_path,
                    "summary_csv": summary_csv,
                }
            )

            print("\nSaved:")
            print(f"  {png_path}")
            print(f"  {pdf_path}")
            print(f"  {summary_csv}")

        if show:
            plt.show()

        return outputs

    finally:
        cropland.close()


if __name__ == "__main__":
    plot_biochar_effects_figure(
        save=True,
        show=SHOW_FIGURE,
    )
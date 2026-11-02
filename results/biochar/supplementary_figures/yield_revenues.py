from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle

from timer.cdr.cdr_util import land_mask


"""
Four-panel map of potential crop-yield revenues from biochar application.

Panels
------
a. Potential crop-yield revenue, 2030
b. Potential crop-yield revenue, 2050
c. Potential crop-yield revenue, 2075
d. Potential crop-yield revenue, 2100

Model variable
--------------
The module reads the gridded TIMER/PRISM output:

    yield_revenues_ha.nc
        variable: yield_revenues_ha
        units:    $/ha

The variable represents the potential crop-yield revenue per hectare of
cropland over the biochar application/effect window.

Map treatment
-------------
- SSP2 cropland is shown in light grey for each plotted year.
- Cropland with zero potential crop-yield revenue is shown in grey.
- Positive potential crop-yield revenues are shown with a continuous colour scale.
- The grey zero class is also shown explicitly at the left of each colourbar.
- Each year has its own continuous positive-value scale so that spatial variation
  within that year remains visible.
- By default, vmax is based on the 98th percentile of positive cropland cells in
  each year and rounded upward to a publication-friendly value. Values above
  that threshold use a colourbar extension.
- Fixed year-specific vmax values can be supplied below if preferred.
- The scenario annotation is positioned above Antarctica to avoid overlap
  with the Antarctic coastline/border.
"""


# =============================================================================
# Configuration
# =============================================================================

CDR_OUTPUT_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output"
)

# Detailed biochar scenario.
SCENARIO_DIR = CDR_OUTPUT_DIR / "biochar_c400"

YIELD_REVENUE_FILE = SCENARIO_DIR / "yield_revenues_ha.nc"
YIELD_REVENUE_VAR = "yield_revenues_ha"

# SSP2 cropland background.
SSP2_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\biochar\SSP2_M_new"
)
GLANDCOVER_FILE = SSP2_DIR / "GLANDCOVER.NC"
GAREA_FILE = SSP2_DIR / "GAREA.NC"

OUTPUT_DIR = CDR_OUTPUT_DIR / "figures"
OUTPUT_STEM = "biochar_potential_crop_yield_revenue_4panel_v3"

YEARS = (2030, 2050, 2075, 2100)
PANEL_LETTERS = ("a", "b", "c", "d")

RESOLUTION = "5m"

CROPLAND_INDEX = 0
MIN_CROPLAND_HA = 1.0

SAVE_DPI = 600
SHOW_FIGURE = True

FIGSIZE = (12.8, 8.0)

# Map styling.
BACKGROUND_COLOR = "0.84"
BACKGROUND_POINT_SIZE = 0.28
DATA_POINT_SIZE = 0.42
COASTLINE_WIDTH = 0.35
BORDER_WIDTH = 0.20

YIELD_CMAP = "YlGn"

# One colourbar per panel.
COLORBAR_SHRINK = 0.76
COLORBAR_ASPECT = 28
COLORBAR_FRACTION = 0.046
COLORBAR_PAD = 0.035

# Robust colour scaling is calculated independently for every year.
ROBUST_VMAX_PERCENTILE = 98.0

# Optional fixed vmax values for final publication.
# Leave as None to use the year-specific robust percentile.
FIXED_VMAX_BY_YEAR = {
    2030: None,
    2050: None,
    2075: None,
    2100: None,
}

VMIN = 0.0

# Scenario label shown inside each map.
MAP_ANNOTATION = r"SSP2 carbon price 400 \$ tCO$_2^{-1}$"

# Move the annotation upward relative to the original figure so that it no
# longer overlaps the Antarctic coastline.
MAP_ANNOTATION_X = 0.64
MAP_ANNOTATION_Y = 0.185

CBAR_LABEL = r"Potential crop-yield revenue (\$2010 ha$^{-1}$)"


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
    time_dim = _find_dim(
        da,
        ["time", "year"],
        required=False,
    )

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


def _panel_title(letter, year):
    return rf"$\bf{{{letter}.}}$ Potential crop-yield revenue, {year}"


def _formatter():
    """
    Format colourbar labels using rounded values.
    """
    def _format(x, pos):
        ax = abs(x)

        if ax >= 1000:
            return f"{x:,.0f}"
        if ax >= 1:
            return f"{x:.0f}"
        if ax >= 0.1:
            return f"{x:.1f}"
        if ax == 0:
            return "0"

        return f"{x:.2f}"

    return FuncFormatter(_format)


def _nice_number(value):
    """
    Return a publication-friendly number close to ``value``.

    The preferred mantissas are 1, 2, 2.5, 5, and 10.
    """
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


def _round_vmax_up(value):
    """
    Round a colour-scale maximum upward to a clean value.
    """
    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        return 1.0

    exponent = np.floor(np.log10(value))
    scale = 10.0 ** exponent
    fraction = value / scale

    candidates = (1.0, 2.0, 2.5, 5.0, 10.0)

    for candidate in candidates:
        if fraction <= candidate + 1e-12:
            return float(candidate * scale)

    return float(10.0 * scale)


def _nice_ticks(vmax, target_intervals=4):
    """
    Generate rounded positive-value colourbar ticks.

    Zero is deliberately excluded because it is represented separately by the
    grey zero-value key attached to the left of the continuous colourbar.
    """
    vmax = float(vmax)

    if not np.isfinite(vmax) or vmax <= 0.0:
        return np.array([1.0])

    raw_step = vmax / float(target_intervals)
    step = _nice_number(raw_step)

    ticks = np.arange(
        step,
        vmax + 0.5 * step,
        step,
        dtype=np.float64,
    )
    ticks = ticks[ticks <= vmax * (1.0 + 1e-10)]

    if ticks.size == 0:
        ticks = np.array([vmax], dtype=np.float64)
    elif not np.isclose(ticks[-1], vmax):
        ticks = np.append(ticks, vmax)

    return np.unique(ticks)


def _robust_vmax(values, fixed_vmax=None):
    if fixed_vmax is not None:
        fixed_vmax = float(fixed_vmax)

        if fixed_vmax <= 0.0:
            raise ValueError("Fixed vmax must be positive.")

        return fixed_vmax

    values = np.asarray(values, dtype=np.float64)

    valid = (
        np.isfinite(values)
        & (values > 0.0)
    )

    if not np.any(valid):
        return 1.0

    raw_vmax = float(
        np.nanpercentile(
            values[valid],
            ROBUST_VMAX_PERCENTILE,
        )
    )

    if not np.isfinite(raw_vmax) or raw_vmax <= 0.0:
        raw_vmax = float(np.nanmax(values[valid]))

    return _round_vmax_up(raw_vmax)


def _add_zero_grey_key(cbar):
    """
    Add the grey zero-value class immediately to the left of the continuous
    positive-value colourbar.

    Exact zero-valued cropland is therefore represented consistently in the
    map and legend, following the convention used in the other spatial
    biochar figures.
    """
    box_width = 0.075

    rect = Rectangle(
        (-box_width, 0.0),
        box_width,
        1.0,
        transform=cbar.ax.transAxes,
        facecolor=BACKGROUND_COLOR,
        edgecolor="0.35",
        linewidth=0.6,
        clip_on=False,
        zorder=10,
    )
    cbar.ax.add_patch(rect)

    cbar.ax.text(
        -0.5 * box_width,
        -0.42,
        "0",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.0,
        clip_on=False,
    )


def _check_required_files():
    files = {
        "Potential crop-yield revenue": YIELD_REVENUE_FILE,
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

        if "GLANDCOVER" not in self.ds_lc:
            raise KeyError(
                f"GLANDCOVER not found in {self.glandcover_file}"
            )

        if "GAREA" not in self.ds_area:
            raise KeyError(
                f"GAREA not found in {self.garea_file}"
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
            [
                "landcover",
                "land_cover",
                "nglndcov",
                "cover",
                "class",
            ],
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
# Potential crop-yield revenue loading
# =============================================================================

def _load_yield_revenue_per_ha(
    nc_file,
    var_name,
    year,
    expected_landgrid_length,
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

        da = _select_year(
            ds[var_name],
            year,
        ).squeeze(drop=True)

        values = np.asarray(
            da.values,
            dtype=np.float64,
        ).squeeze()

        units = da.attrs.get(
            "units",
            ds[var_name].attrs.get(
                "units",
                "",
            ),
        )

        description = da.attrs.get(
            "description",
            ds[var_name].attrs.get(
                "description",
                "",
            ),
        )

    if values.ndim != 1:
        raise ValueError(
            f"{nc_file.name}:{var_name} did not resolve to a 1D LandGrid "
            f"array after selecting {year}. Shape={values.shape}"
        )

    if values.size != expected_landgrid_length:
        raise ValueError(
            f"{nc_file.name}:{var_name} has {values.size} grid cells, "
            f"expected {expected_landgrid_length}."
        )

    # Yield revenues should be non-negative in the current detailed model.
    # Preserve NaN and zero, but clean tiny numerical negatives.
    tiny_negative = (
        np.isfinite(values)
        & (values < 0.0)
        & (values > -1e-10)
    )
    values[tiny_negative] = 0.0

    if np.any(np.isfinite(values) & (values < 0.0)):
        negative_min = float(
            np.nanmin(
                values[
                    np.isfinite(values)
                    & (values < 0.0)
                ]
            )
        )

        raise ValueError(
            f"{var_name} contains materially negative crop-yield revenues; "
            f"minimum={negative_min:g}. Check model output."
        )

    return {
        "values": values,
        "units": units,
        "description": description,
        "file": nc_file,
    }


# =============================================================================
# Quantitative diagnostics
# =============================================================================

def _summarise_field(
    year,
    values,
    cropland_area_ha,
    vmax,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    cropland_area_ha = np.asarray(
        cropland_area_ha,
        dtype=np.float64,
    )

    crop = (
        np.isfinite(cropland_area_ha)
        & (cropland_area_ha > MIN_CROPLAND_HA)
    )

    positive = (
        crop
        & np.isfinite(values)
        & (values > 0.0)
    )

    if np.any(positive):
        x = values[positive]

        quantiles = np.nanpercentile(
            x,
            [5, 25, 50, 75, 95, 98, 99, 99.5],
        )

        mean = float(np.nanmean(x))
        maximum = float(np.nanmax(x))
    else:
        quantiles = np.full(8, np.nan)
        mean = np.nan
        maximum = np.nan

    return {
        "Year": int(year),
        "Positive cells": int(np.count_nonzero(positive)),
        "Cropland cells": int(np.count_nonzero(crop)),
        "Share of cropland cells with positive revenue (%)": (
            100.0
            * np.count_nonzero(positive)
            / np.count_nonzero(crop)
            if np.count_nonzero(crop) > 0
            else np.nan
        ),
        "Mean positive value": mean,
        "P05": float(quantiles[0]),
        "P25": float(quantiles[1]),
        "Median": float(quantiles[2]),
        "P75": float(quantiles[3]),
        "P95": float(quantiles[4]),
        "P98": float(quantiles[5]),
        "P99": float(quantiles[6]),
        "P99.5": float(quantiles[7]),
        "Maximum": maximum,
        "Colourbar vmax": float(vmax),
    }


def print_quantitative_summary(
    values_by_year,
    cropland_by_year,
    vmax_by_year,
):
    rows = [
        _summarise_field(
            year=year,
            values=values_by_year[year],
            cropland_area_ha=cropland_by_year[year],
            vmax=vmax_by_year[year],
        )
        for year in YEARS
    ]

    df = pd.DataFrame(rows).set_index("Year")

    print("\n" + "=" * 112)
    print("POTENTIAL CROP-YIELD REVENUE PER HECTARE")
    print("=" * 112)
    print(df.round(3).to_string())

    print("\nYear-specific colour scales")
    for year in YEARS:
        fixed = FIXED_VMAX_BY_YEAR.get(year)
        source = (
            "fixed"
            if fixed is not None
            else f"P{ROBUST_VMAX_PERCENTILE:g}, rounded up"
        )

        print(
            f"  {year}: vmax = {vmax_by_year[year]:,.3f} "
            f"$/ha ({source})"
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


def _plot_panel(
    fig,
    ax,
    lons,
    lats,
    cropland_area_ha,
    values,
    title,
    vmax,
):
    _setup_map(ax)

    crop_mask = (
        np.isfinite(cropland_area_ha)
        & (cropland_area_ha > MIN_CROPLAND_HA)
    )

    # Draw all cropland first. Cells with exactly zero crop-yield revenue remain
    # visible as this grey background layer.
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

    # Only positive crop-yield revenues are over-plotted with the continuous
    # colour scale. Zero-valued cropland therefore remains grey.
    colour_mask = (
        crop_mask
        & np.isfinite(values)
        & (values > 0.0)
    )

    norm = mpl.colors.Normalize(
        vmin=VMIN,
        vmax=vmax,
    )
    cmap = plt.get_cmap(YIELD_CMAP)

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
        fontsize=11.5,
        fontweight="normal",
        pad=5,
    )

    # Position the scenario annotation above the Antarctic coastline.
    ax.text(
        MAP_ANNOTATION_X,
        MAP_ANNOTATION_Y,
        MAP_ANNOTATION,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.2,
        zorder=6,
    )

    # Independent continuous positive-value colour scale for this year.
    # Exact zero is shown separately in grey at the left of the bar.
    cbar = fig.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        ticks=_nice_ticks(vmax),
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
        aspect=COLORBAR_ASPECT,
        shrink=COLORBAR_SHRINK,
        extend=extend,
        extendfrac=0.045,
    )

    cbar.set_label(
        CBAR_LABEL,
        fontsize=9.0,
        labelpad=4,
    )
    cbar.ax.tick_params(
        labelsize=8.0,
        length=2.5,
    )
    cbar.ax.xaxis.set_major_formatter(
        _formatter()
    )
    cbar.outline.set_linewidth(0.5)

    _add_zero_grey_key(cbar)

    return cbar


# =============================================================================
# Main
# =============================================================================

def plot_potential_crop_yield_revenue_maps(
    save=True,
    show=True,
):
    _check_required_files()

    cropland = CroplandData()

    try:
        values_by_year = {}
        cropland_by_year = {}
        vmax_by_year = {}
        metadata_by_year = {}

        for year in YEARS:
            crop_area = cropland.cropland_area_ha(
                year
            )

            info = _load_yield_revenue_per_ha(
                YIELD_REVENUE_FILE,
                YIELD_REVENUE_VAR,
                year,
                expected_landgrid_length=cropland.lats.size,
            )

            values = info["values"]

            # Keep only SSP2 cropland in the map domain.
            values_map = np.where(
                crop_area > MIN_CROPLAND_HA,
                values,
                np.nan,
            )

            vmax = _robust_vmax(
                values_map,
                FIXED_VMAX_BY_YEAR.get(year),
            )

            cropland_by_year[year] = crop_area
            values_by_year[year] = values_map
            vmax_by_year[year] = vmax
            metadata_by_year[year] = info

        summary = print_quantitative_summary(
            values_by_year=values_by_year,
            cropland_by_year=cropland_by_year,
            vmax_by_year=vmax_by_year,
        )

        first_info = metadata_by_year[YEARS[0]]

        print("\nInput metadata")
        print(
            f"  {YIELD_REVENUE_VAR}: "
            f"file={first_info['file']}, "
            f"units='{first_info['units']}', "
            f"description='{first_info['description']}'"
        )

        projection = ccrs.Robinson()

        fig, axes = plt.subplots(
            2,
            2,
            figsize=FIGSIZE,
            subplot_kw={
                "projection": projection
            },
        )

        axes = np.asarray(axes).ravel()

        for ax, letter, year in zip(
            axes,
            PANEL_LETTERS,
            YEARS,
        ):
            _plot_panel(
                fig=fig,
                ax=ax,
                lons=cropland.lons,
                lats=cropland.lats,
                cropland_area_ha=cropland_by_year[year],
                values=values_by_year[year],
                title=_panel_title(
                    letter,
                    year,
                ),
                vmax=vmax_by_year[year],
            )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.955,
            bottom=0.075,
            wspace=0.06,
            hspace=0.24,
        )

        outputs = {
            "figure": fig,
            "axes": axes,
            "summary": summary,
            "values_by_year": values_by_year,
            "cropland_by_year": cropland_by_year,
            "vmax_by_year": vmax_by_year,
        }

        if save:
            OUTPUT_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            png_path = OUTPUT_DIR / (
                f"{OUTPUT_STEM}.png"
            )
            pdf_path = OUTPUT_DIR / (
                f"{OUTPUT_STEM}.pdf"
            )
            summary_csv = OUTPUT_DIR / (
                f"{OUTPUT_STEM}_summary.csv"
            )

            fig.savefig(
                png_path,
                dpi=SAVE_DPI,
                bbox_inches="tight",
            )
            fig.savefig(
                pdf_path,
                bbox_inches="tight",
            )
            summary.to_csv(
                summary_csv
            )

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
    plot_potential_crop_yield_revenue_maps(
        save=True,
        show=SHOW_FIGURE,
    )
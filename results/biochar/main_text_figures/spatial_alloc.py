import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Rectangle
from pathlib import Path
from timer.cdr.cdr_util import land_mask

BASE_OUTPUT = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output")
MITIGATION_DIR = BASE_OUTPUT / "biochar_c400"
AGRICULTURE_DIR = BASE_OUTPUT / "biochar_c400_LT"
SSP2_DIR = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\biochar\SSP2_M_new")
GLANDCOVER_FILE = SSP2_DIR / "GLANDCOVER.NC"
GAREA_FILE = SSP2_DIR / "GAREA.NC"
OUTPUT_DIR = BASE_OUTPUT / "figures"
OUTPUT_STEM = "biochar_spatial_potential_realised_4panel_v9"
YEAR = 2100
START_YEAR = 2030
END_YEAR = 2100
APPLICATION_RATE = 20.0
HT_TECH_INDEX = 1
CROPLAND_INDEX = 0
MIN_CROPLAND_HA = 1.0
RESOLUTION = "5m"
FIGSIZE = (12.0, 8.2)
BACKGROUND_COLOR = "0.84"
BACKGROUND_POINT_SIZE = 0.28
DATA_POINT_SIZE = 0.42
COASTLINE_WIDTH = 0.35
BORDER_WIDTH = 0.20
MITIGATION_CMAP = "YlOrBr"
AGRICULTURE_CMAP = "YlGn"
COLORBAR_SHRINK = 0.72
COLORBAR_ASPECT = 28
COLORBAR_FRACTION = 0.040
COLORBAR_PAD = 0.040
SAVE_DPI = 600
MAP_ANNOTATION = r"SSP2 carbon price 400 \$ tCO$_2^{-1}$"
MAP_ANNOTATION_X = 0.52
MAP_ANNOTATION_Y = 0.115
PANEL_A_VMAX = 54.0
PANEL_B_VMAX = 10000.0
PANEL_C_VMAX = 700.0
PANEL_D_VMAX = 79599.0

def _find_dim(da, candidates, required=True):
    lower = {d.lower(): d for d in da.dims}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]

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

    return da.isel({time_dim: int(matches[0])})

def _available_years(da):
    time_dim = _find_dim(da, ["time", "year"], required=False)
    if time_dim is None:
        return np.array([], dtype=int)

    return _year_values(da[time_dim])

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

        self.ds_lc = xr.open_dataset(self.glandcover_file, engine="netcdf4")
        self.ds_area = xr.open_dataset(self.garea_file, engine="netcdf4")

        self.lc = self.ds_lc["GLANDCOVER"]
        self.garea = self.ds_area["GAREA"]

        self.lat_dim = _find_dim(self.lc, ["lat", "latitude", "y"])
        self.lon_dim = _find_dim(self.lc, ["lon", "longitude", "x"])
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
        lat = np.asarray(self.ds_lc[self.lat_dim].values, dtype=np.float64)
        lon = np.asarray(self.ds_lc[self.lon_dim].values, dtype=np.float64)

        if np.nanmax(lon) > 180.0:
            lon = ((lon + 180.0) % 360.0) - 180.0

        lon2d, lat2d = np.meshgrid(lon, lat)

        return (
            lat2d.reshape(-1)[self.mask_land],
            lon2d.reshape(-1)[self.mask_land],
        )

    def _global_or_land_to_landgrid(self, da, lat_dim=None, lon_dim=None):
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
            values = np.asarray(da.values, dtype=np.float64).reshape(-1)
            return values[self.mask_land]

        values = np.asarray(da.values, dtype=np.float64).squeeze()

        return values

    def cropland_area_ha(self, year):
        lc_year = _select_year(self.lc, year)
        crop_fraction = lc_year.isel({self.lc_dim: self.cropland_index})

        crop_fraction_land = self._global_or_land_to_landgrid(
            crop_fraction,
            self.lat_dim,
            self.lon_dim,
        )

        area_year = _select_year(self.garea, year).squeeze(drop=True)
        area_land = self._global_or_land_to_landgrid(area_year)
        crop_fraction_land = np.clip(crop_fraction_land, 0.0, None)

        return crop_fraction_land * area_land * 100.0

def _load_landgrid_variable(
    nc_file,
    var_name,
    year=None,
    tech_index=None,
):
    nc_file = Path(nc_file)
    with xr.open_dataset(nc_file, engine="netcdf4") as ds:
        da = ds[var_name]
        if year is not None:
            da = _select_year(da, year)
        tech_dim = _find_dim(
            da,
            ["tech", "technology"],
            required=False,
        )
        if tech_index is not None:
            da = da.isel({tech_dim: int(tech_index)})

        da = da.squeeze(drop=True)
        values = np.asarray(da.values, dtype=np.float64).squeeze()

    return values

def potential_field_cdr_2100(cropland):
    retained_c = _load_landgrid_variable(
        MITIGATION_DIR / "biochar_retained_carbon.nc",
        "biochar_retained_carbon",
        year=YEAR,
        tech_index=HT_TECH_INDEX,
    )

    crop_area = cropland.cropland_area_ha(YEAR)
    potential = APPLICATION_RATE * retained_c * (44.0 / 12.0)
    return np.where(crop_area > MIN_CROPLAND_HA, potential, np.nan)

def potential_yield_revenue_2100(cropland):
    revenue = _load_landgrid_variable(
        AGRICULTURE_DIR / "yield_revenues_ha.nc",
        "yield_revenues_ha",
        year=YEAR,
    )
    crop_area = cropland.cropland_area_ha(YEAR)
    return np.where(crop_area > MIN_CROPLAND_HA, revenue, np.nan)

def cumulative_realised_field_cdr(cropland):
    nc_file = MITIGATION_DIR / "BiocharFieldCDRGrid.nc"
    with xr.open_dataset(nc_file, engine="netcdf4") as ds:
        da = ds["BiocharFieldCDRGrid"]
        years = _available_years(da)
        years = years[(years >= START_YEAR) & (years <= END_YEAR)]

        total_per_ha = None

        for year in years:
            field_cdr = np.asarray(
                _select_year(da, int(year)).values,
                dtype=np.float64,
            ).squeeze()

            crop_area = cropland.cropland_area_ha(int(year))

            per_ha = np.divide(
                field_cdr,
                crop_area,
                out=np.zeros_like(field_cdr, dtype=np.float64),
                where=crop_area > MIN_CROPLAND_HA,
            )

            if total_per_ha is None:
                total_per_ha = per_ha
            else:
                total_per_ha += per_ha

    crop_2100 = cropland.cropland_area_ha(YEAR)
    return np.where(crop_2100 > MIN_CROPLAND_HA, total_per_ha, np.nan)

def cumulative_realised_yield_revenue(cropland):
    revenue_file = AGRICULTURE_DIR / "yield_revenues_ha.nc"
    alloc_file = AGRICULTURE_DIR / "BiocharAlloc.nc"
    with (
        xr.open_dataset(revenue_file, engine="netcdf4") as ds_rev,
        xr.open_dataset(alloc_file, engine="netcdf4") as ds_alloc,
    ):
        revenue = ds_rev["yield_revenues_ha"]
        allocation = ds_alloc["BiocharAlloc"]
        years = np.intersect1d(
            _available_years(revenue),
            _available_years(allocation),
        )
        years = years[(years >= START_YEAR) & (years <= END_YEAR)]
        total = None

        for year in years:
            rev = np.asarray(
                _select_year(revenue, int(year)).values,
                dtype=np.float64,
            ).squeeze()

            alloc = np.asarray(
                _select_year(allocation, int(year)).values,
                dtype=np.float64,
            ).squeeze()

            realised = rev * (alloc > 0.0)

            if total is None:
                total = realised
            else:
                total += realised

    crop_2100 = cropland.cropland_area_ha(YEAR)
    return np.where(crop_2100 > MIN_CROPLAND_HA, total, np.nan)

def _panel_title(letter, text):
    return rf"$\bf{{{letter}.}}$ {text}"

def _setup_map(ax):
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="white", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=COASTLINE_WIDTH, zorder=5)
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=BORDER_WIDTH,
        alpha=0.55,
        zorder=5,
    )

def _formatter():
    def _fmt(x, pos):
        ax = abs(x)
        if ax >= 1000:
            return f"{x:,.0f}"
        if ax >= 100:
            return f"{x:.0f}"
        if ax >= 10:
            return f"{x:.0f}"
        if ax >= 1:
            return f"{x:.1f}"
        return f"{x:.2f}"
    return FuncFormatter(_fmt)

def _make_ticks(vmax, zero_to_grey=False):
    if np.isclose(vmax, PANEL_A_VMAX):
        return np.array([36.0, 42.0, 48.0, 54.0])

    if np.isclose(vmax, PANEL_B_VMAX):
        ticks = np.array([0.0, 2500.0, 5000.0, 10000.0])
    elif np.isclose(vmax, PANEL_C_VMAX):
        ticks = np.array([0.0, 250.0, 500.0, 700.0])
    elif np.isclose(vmax, PANEL_D_VMAX):
        ticks = np.array([0.0, 25000.0, 50000.0, 75000.0])
    else:
        return None

    if zero_to_grey:
        ticks = ticks[ticks > 0.0]

    return ticks

def _add_zero_grey_key(cbar):
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
        "0.0",
        transform=cbar.ax.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
        clip_on=False,
    )

def _plot_panel(
    fig,
    ax,
    lons,
    lats,
    crop_background_ha,
    values,
    title,
    cbar_label,
    cmap_name,
    vmin,
    vmax,
    zero_to_grey=False,
):
    _setup_map(ax)

    crop_mask = (
        np.isfinite(crop_background_ha)
        & (crop_background_ha > MIN_CROPLAND_HA)
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

    finite_mask = crop_mask & np.isfinite(values)

    if zero_to_grey:
        colour_mask = finite_mask & (values > 0.0)
    else:
        colour_mask = finite_mask

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)
    actual_max = float(np.nanmax(values[colour_mask])) if np.any(colour_mask) else vmax
    extend = "max" if actual_max > vmax * (1.0 + 1e-9) else "neither"

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

    ax.text(
        MAP_ANNOTATION_X,
        MAP_ANNOTATION_Y,
        MAP_ANNOTATION,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color="#222222",
        zorder=6,
    )

    cbar = fig.colorbar(
        sc,
        ax=ax,
        orientation="horizontal",
        ticks=_make_ticks(vmax, zero_to_grey=zero_to_grey),
        fraction=COLORBAR_FRACTION,
        pad=COLORBAR_PAD,
        aspect=COLORBAR_ASPECT,
        shrink=COLORBAR_SHRINK,
        extend=extend,
        extendfrac=0.045,
    )
    cbar.set_label(cbar_label, fontsize=9.5, labelpad=4)
    cbar.ax.tick_params(labelsize=8.5, length=2.5)
    cbar.ax.xaxis.set_major_formatter(_formatter())
    cbar.outline.set_linewidth(0.5)

    if zero_to_grey:
        _add_zero_grey_key(cbar)

    return cbar

def plot_four_panel_biochar_maps(save=True, show=True):
    cropland = CroplandData()

    try:
        crop_2100 = cropland.cropland_area_ha(YEAR)
        panel_a = potential_field_cdr_2100(cropland)
        panel_b = potential_yield_revenue_2100(cropland)
        panel_c = cumulative_realised_field_cdr(cropland)
        panel_d = cumulative_realised_yield_revenue(cropland)
        panel_a_vmin = float(np.nanmin(panel_a[np.isfinite(panel_a)]))
        projection = ccrs.Robinson()

        fig, axes = plt.subplots(
            2,
            2,
            figsize=FIGSIZE,
            subplot_kw={"projection": projection},
        )

        _plot_panel(
            fig=fig,
            ax=axes[0, 0],
            lons=cropland.lons,
            lats=cropland.lats,
            crop_background_ha=crop_2100,
            values=panel_a,
            title=_panel_title("a", "Biochar for mitigation, 2100"),
            cbar_label=(
                r"Potential field C sequestration (tCO$_2$ ha$^{-1}$)"
            ),
            cmap_name=MITIGATION_CMAP,
            vmin=panel_a_vmin,
            vmax=PANEL_A_VMAX,
            zero_to_grey=False,
        )

        _plot_panel(
            fig=fig,
            ax=axes[0, 1],
            lons=cropland.lons,
            lats=cropland.lats,
            crop_background_ha=crop_2100,
            values=panel_b,
            title=_panel_title("b", "Biochar for agriculture, 2100"),
            cbar_label=(
                r"Potential crop-yield revenue (\$2010 ha$^{-1}$)"
            ),
            cmap_name=AGRICULTURE_CMAP,
            vmin=0.0,
            vmax=PANEL_B_VMAX,
            zero_to_grey=True,
        )

        _plot_panel(
            fig=fig,
            ax=axes[1, 0],
            lons=cropland.lons,
            lats=cropland.lats,
            crop_background_ha=crop_2100,
            values=panel_c,
            title=_panel_title("c", "Biochar for mitigation, 2030–2100"),
            cbar_label=(
                r"Realised cumulative field C sequestration (tCO$_2$ ha$^{-1}$)"
            ),
            cmap_name=MITIGATION_CMAP,
            vmin=0.0,
            vmax=PANEL_C_VMAX,
            zero_to_grey=True,
        )

        _plot_panel(
            fig=fig,
            ax=axes[1, 1],
            lons=cropland.lons,
            lats=cropland.lats,
            crop_background_ha=crop_2100,
            values=panel_d,
            title=_panel_title("d", "Biochar for agriculture, 2030–2100"),
            cbar_label=(
                r"Realised cumulative crop-yield revenue (\$2010 ha$^{-1}$)"
            ),
            cmap_name=AGRICULTURE_CMAP,
            vmin=0.0,
            vmax=PANEL_D_VMAX,
            zero_to_grey=True,
        )

        fig.subplots_adjust(
            left=0.025,
            right=0.985,
            top=0.975,
            bottom=0.055,
            wspace=0.06,
            hspace=0.16,
        )

        if save:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
            pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"

            fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
            fig.savefig(pdf_path, bbox_inches="tight")
        if show:
            plt.show()

        return fig, axes

    finally:
        cropland.close()

if __name__ == "__main__":
    plot_four_panel_biochar_maps(save=True, show=True)

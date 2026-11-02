import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from pathlib import Path
from timer.cdr.cdr_util import land_mask, region_to_grid


BASE_OUTPUT = Path(r"X:\user\orhalem\IMAGE_CDR\3_IMAGE_land\scen")
BIOCHAR_DIR = BASE_OUTPUT / "SSP2_M_Biochar_altijd_yield" / "netcdf"
BASELINE_DIR = BASE_OUTPUT / "SSP2_M" / "netcdf"
BIOCHAR_GLANDCOVER = BIOCHAR_DIR / "GLANDCOVER.NC"
BIOCHAR_GAREA = BIOCHAR_DIR / "GAREA.NC"
BIOCHAR_GNLCT = BIOCHAR_DIR / "GNLCT.NC"
BASELINE_GLANDCOVER = BASELINE_DIR / "GLANDCOVER.NC"
BASELINE_GAREA = BASELINE_DIR / "GAREA.NC"
OUTPUT_DIR = Path(r"X:\user\orhalem\TIMER_CDR\figures")
OUTPUT_STEM = "biochar_land_use_4panel_2030_2100_v12"
START_YEAR = 2030
END_YEAR = 2100
RESOLUTION = "5m"
SAVE_DPI = 600
SHOW_FIGURE = True
FIGSIZE = (15.4, 11.1)
MAP_GAIN_AVOIDED = "#1FA67A"
MAP_GAIN_REVERSION = "#377EB8"
MAP_BG_CROPLAND = "#C8C8C8"     
MAP_LAND = "#FFFFFF"
MAP_OCEAN = "#FFFFFF"
MAP_GAIN_THRESHOLD_HA = 1.0
MAP_GAIN_MARKER_SIZE = 0.65
MAP_ANNOTATION = r"SSP2 carbon price 400 \$ tCO$_2^{-1}$"
PANEL_D_REGIONS = [
    "North America",
    "Latin America",
    "Sub-Saharan Africa",
    "South Asia",
    "East Asia",
    "Southeast Asia",
]
WF_POS = "#1A9850"
WF_NEG = "#D73027"
WF_ENDPOINT = "#5A5A5A"
WF_CONNECTOR = "0.45"
GNLCT_CLASSES = {
    1: "Agriculture",
    2: "Extensive grassland",
    3: "Carbon plantation",
    4: "Regrowth forest abandoned",
    5: "Regrowth forest timber",
    6: "Biofuel",
    7: "Ice",
    8: "Tundra",
    9: "Wooded tundra",
    10: "Boreal forest",
    11: "Cool conifer",
    12: "Temperate mixed forest",
    13: "Temperate deciduous forest",
    14: "Warm mixed",
    15: "Grassland/steppe",
    16: "Hot desert",
    17: "Scrubland",
    18: "Savanna",
    19: "Tropical woodland",
    20: "Tropical forest",
}
GNLCT_GROUPS = {
    "Regrowth forest": (4, 5),
    "Cold forest & tundra": (8, 9, 10, 11),
    "Temperate forest": (12, 13, 14),
    "Grassland & savanna": (2, 15, 18),
    "Scrubland & desert": (16, 17),
    "Tropical forest & woodland": (19, 20),
}
GNLCT_COLORS = {
    "Regrowth forest": "#4DAF4A",
    "Cold forest & tundra": "#80B1D3",
    "Temperate forest": "#B3DE69",
    "Grassland & savanna": "#999999",
    "Scrubland & desert": "#E6AB02",
    "Tropical forest & woodland": "#1B9E77",
}
IMAGE_REGIONS = [
    "CAN", "USA", "MEX", "RCAM", "BRA", "RSAM", "NAF", "WAF", "EAF", "SAF",
    "WEU", "CEU", "TUR", "UKR", "STAN", "RUS", "ME", "INDIA", "KOR", "CHN",
    "SEAS", "INDO", "JAP", "OCE", "RSAS", "RSAF",
]
R10 = {
    "North America": ["CAN", "USA"],
    "Latin America": ["MEX", "RCAM", "BRA", "RSAM"],
    "Europe": ["WEU", "CEU", "TUR"],
    "Eurasia": ["UKR", "STAN", "RUS"],
    "Middle East & North Africa": ["NAF", "ME"],
    "Sub-Saharan Africa": ["WAF", "EAF", "SAF", "RSAF"],
    "South Asia": ["INDIA", "RSAS"],
    "East Asia": ["CHN", "KOR", "JAP"],
    "Southeast Asia": ["SEAS", "INDO"],
    "Oceania": ["OCE"],
}
R10_DISPLAY = {
    "North America": "North America",
    "Latin America": "Latin America",
    "Europe": "Europe",
    "Eurasia": "Eurasia",
    "Middle East & North Africa": "MENA",
    "Sub-Saharan Africa": "SSA",
    "South Asia": "South Asia",
    "East Asia": "East Asia",
    "Southeast Asia": "Southeast Asia",
    "Oceania": "Oceania",
}
LANDCOVER_INDICES = {
    0: "Cropland",
    1: "Pasture",
    2: "Forest",
    3: "Other natural land",
    4: "Built-up area",
}
MECHANISM_OUTPUT_STEM = (
    "biochar_land_sparing_mechanisms_2030_2100_v4"
)

def set_style():
    plt.rcParams.update(
        {
            "font.size": 10.6,
            "axes.labelsize": 11.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

def panel_title(letter, text):
    return rf"$\bf{{{letter}.}}$ {text}"

def find_dim(da, candidates, required=True):
    lower = {d.lower(): d for d in da.dims}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None

def year_values(coord):
    vals = coord.values
    if np.issubdtype(vals.dtype, np.datetime64):
        return pd.to_datetime(vals).year.to_numpy(dtype=int)
    return np.asarray(vals).astype(int)

def select_year(da, year):
    time_dim = find_dim(da, ["time", "year"], required=False)
    if time_dim is None:
        return da
    yrs = year_values(da[time_dim])
    idx = np.where(yrs == int(year))[0]
    return da.isel({time_dim: int(idx[0])})

def standardize_region_cells(region_cells):
    region_cells = np.asarray(region_cells).astype(int)
    valid = region_cells[region_cells >= 0]

    if valid.min() >= 1 and valid.max() <= 26:
        region_cells = np.where(region_cells >= 0, region_cells - 1, -1)

    valid = region_cells[region_cells >= 0]
    return region_cells

def r26_to_r10(values_r26):
    values_r26 = np.asarray(values_r26, dtype=np.float64)
    out = np.zeros(values_r26.shape[:-1] + (len(R10),), dtype=np.float64)
    for r10_i, (_, regs) in enumerate(R10.items()):
        idx = [IMAGE_REGIONS.index(r) for r in regs]
        out[..., r10_i] = np.nansum(values_r26[..., idx], axis=-1)
    return out

def land_from_full_grid(values_2d):
    flat = np.asarray(values_2d).reshape(-1)
    mask = land_mask(RESOLUTION)
    return flat[mask]

def fmt_num(x):
    ax = abs(x)
    if ax >= 100:
        return f"{x:.0f}"
    if ax >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"

def aggregate_full_ha_to_r26(area_ha_full, region_cells):
    region_cells = standardize_region_cells(region_cells)
    land = land_from_full_grid(area_ha_full).astype(np.float64)
    out = np.zeros(len(IMAGE_REGIONS), dtype=np.float64)
    for reg in range(len(IMAGE_REGIONS)):
        mask_reg = region_cells == reg
        if np.any(mask_reg):
            out[reg] = np.nansum(
                land[mask_reg],
                dtype=np.float64
            )

    return pd.Series(out / 1e6, index=IMAGE_REGIONS, dtype=np.float64)

def aggregate_full_ha_to_r10(area_ha_full, region_cells):
    region_cells = standardize_region_cells(region_cells)
    land = land_from_full_grid(area_ha_full).astype(np.float64)

    r26 = np.zeros(len(IMAGE_REGIONS), dtype=np.float64)
    for reg in range(len(IMAGE_REGIONS)):
        mask_reg = region_cells == reg
        if np.any(mask_reg):
            r26[reg] = np.nansum(land[mask_reg], dtype=np.float64)

    r10_mha = r26_to_r10(r26) / 1e6
    return pd.Series(r10_mha, index=list(R10.keys()), dtype=np.float64)

def aggregate_land_ha_to_r10(area_ha_land, region_cells):
    region_cells = standardize_region_cells(region_cells)
    land = np.asarray(area_ha_land, dtype=np.float64)
    r26 = np.zeros(len(IMAGE_REGIONS), dtype=np.float64)
    for reg in range(len(IMAGE_REGIONS)):
        mask_reg = region_cells == reg
        if np.any(mask_reg):
            r26[reg] = np.nansum(land[mask_reg], dtype=np.float64)

    return pd.Series(r26_to_r10(r26) / 1e6, index=list(R10.keys()),dtype=np.float64)

def load_garea_full_for_year(garea_nc, year, var_name="GAREA"):
    with xr.open_dataset(garea_nc, engine="netcdf4") as ds:
        da = select_year(ds[var_name], year).squeeze(drop=True)
        lat_dim = find_dim(da, ["latitude", "lat", "y"])
        lon_dim = find_dim(da, ["longitude", "lon", "x"])
        da = da.transpose(lat_dim, lon_dim)
        area = np.asarray(da.values, dtype=np.float64)
        lat = np.asarray(ds[lat_dim].values, dtype=np.float64)
        lon = np.asarray(ds[lon_dim].values, dtype=np.float64)

    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180
        order = np.argsort(lon)
        lon = lon[order]
        area = area[:, order]

    return area, lat, lon

def load_glandcover_full_area_for_year(
    glandcover_nc,
    garea_nc,
    year,
    var_name="GLANDCOVER",
    garea_var="GAREA",
):
    with xr.open_dataset(glandcover_nc, engine="netcdf4") as ds:
        da = select_year(ds[var_name], year)

        lc_dim = find_dim(da, ["NGLNDCOV", "landcover", "cover", "class", "land_cover"])
        lat_dim = find_dim(da, ["latitude", "lat", "y"])
        lon_dim = find_dim(da, ["longitude", "lon", "x"])
        idx = list(LANDCOVER_INDICES.keys())
        labels = [LANDCOVER_INDICES[i] for i in idx]
        da = da.isel({lc_dim: idx})
        da = da.transpose(lat_dim, lon_dim, lc_dim)
        fractions = np.asarray(da.values, dtype=np.float64)
        lat = np.asarray(ds[lat_dim].values, dtype=np.float64)
        lon = np.asarray(ds[lon_dim].values, dtype=np.float64)

    garea, garea_lat, garea_lon = load_garea_full_for_year(garea_nc, year, var_name=garea_var)

    if np.nanmax(lon) > 180:
        lon = ((lon + 180) % 360) - 180
        order = np.argsort(lon)
        lon = lon[order]
        fractions = fractions[:, order, :]

    fractions = np.clip(np.where(np.isfinite(fractions), fractions, 0.0), 0.0, 1.0)
    garea = np.where(np.isfinite(garea) & (garea > 0.0), garea, 0.0)
    return fractions * garea[:, :, None], garea, lat, lon, labels

def aggregate_landcover_full_to_r10(area_km2, region_cells):
    region_cells = standardize_region_cells(region_cells)
    n_lc = area_km2.shape[-1]
    land_by_lc = np.column_stack([land_from_full_grid(area_km2[:, :, i]) for i in range(n_lc)])
    r26 = np.zeros((n_lc, len(IMAGE_REGIONS)), dtype=np.float64)
    for reg in range(len(IMAGE_REGIONS)):
        mask_reg = region_cells == reg
        if np.any(mask_reg):
            r26[:, reg] = np.nansum(land_by_lc[mask_reg, :], axis=0)

    return r26_to_r10(r26)

def compute_glandcover_scenario_net_change_r10(glandcover_nc, garea_nc, region_cells):
    start_area, _, _, _, labels = load_glandcover_full_area_for_year(glandcover_nc, garea_nc, START_YEAR)
    end_area, _, _, _, _ = load_glandcover_full_area_for_year(glandcover_nc, garea_nc, END_YEAR)
    start_r10 = aggregate_landcover_full_to_r10(start_area, region_cells)
    end_r10 = aggregate_landcover_full_to_r10(end_area, region_cells)

    return pd.DataFrame((end_r10 - start_r10).T, index=list(R10.keys()), columns=labels)


def compute_spatial_natural_land_effect():
    bio_start, _, lat, lon, labels = load_glandcover_full_area_for_year(
        BIOCHAR_GLANDCOVER, BIOCHAR_GAREA, START_YEAR
    )
    bio_end, bio_garea_2100, _, _, _ = load_glandcover_full_area_for_year(
        BIOCHAR_GLANDCOVER, BIOCHAR_GAREA, END_YEAR
    )
    base_start, _, _, _, _ = load_glandcover_full_area_for_year(
        BASELINE_GLANDCOVER, BASELINE_GAREA, START_YEAR
    )
    base_end, _, _, _, _ = load_glandcover_full_area_for_year(
        BASELINE_GLANDCOVER, BASELINE_GAREA, END_YEAR
    )

    cropland_i = labels.index("Cropland")
    forest_i = labels.index("Forest")
    natural_i = labels.index("Other natural land")

    bio_nat_change = (
        bio_end[:, :, forest_i] + bio_end[:, :, natural_i]
        - bio_start[:, :, forest_i] - bio_start[:, :, natural_i]
    )
    base_nat_change = (
        base_end[:, :, forest_i] + base_end[:, :, natural_i]
        - base_start[:, :, forest_i] - base_start[:, :, natural_i]
    )
    effect_ha = (bio_nat_change - base_nat_change) * 100.0

    cropland_frac_2100 = np.divide(
        bio_end[:, :, cropland_i],
        bio_garea_2100,
        out=np.zeros_like(bio_garea_2100, dtype=np.float64),
        where=bio_garea_2100 > 0.0,
    )
    cropland_bg_mask = cropland_frac_2100 >= 0.5

    return effect_ha, cropland_bg_mask, lat, lon

def _sum_r26_table_to_r10(r26_df, additive_columns):
    rows = {}
    for r10_name, r26_names in R10.items():
        rows[r10_name] = {col: float(np.nansum(r26_df.loc[r26_names, col].to_numpy(dtype=np.float64),dtype=np.float64)) for col in additive_columns}
    return pd.DataFrame.from_dict(rows, orient="index")

def _add_cropland_mechanism_shares(df):
    out = df.copy()

    identified = (
        out["Avoided cropland expansion (Mha)"]
        + out["Additional cropland reversion (Mha)"]
    )
    reduced = out["Reduced cropland relative to SSP2 (Mha)"]

    out["Avoided share of reduced cropland (%)"] = np.divide(
        100.0 * out["Avoided cropland expansion (Mha)"],
        reduced,
        out=np.full(len(out), np.nan, dtype=np.float64),
        where=reduced.to_numpy(dtype=np.float64) > 0.0,
    )
    out["Reversion share of reduced cropland (%)"] = np.divide(
        100.0 * out["Additional cropland reversion (Mha)"],
        reduced,
        out=np.full(len(out), np.nan, dtype=np.float64),
        where=reduced.to_numpy(dtype=np.float64) > 0.0,
    )
    out["Mechanism closure error (Mha)"] = (
        out["Avoided cropland expansion (Mha)"]
        + out["Additional cropland reversion (Mha)"]
        - out["Offsetting additional cropland use (Mha)"]
        - out["Signed cropland saving, SSP2 - biochar (Mha)"]
    )

    return out


def analyse_land_sparing_mechanisms(region_cells):
    """
    Diagnose *how biochar changes regional cropland demand* and separately
    diagnose *where the resulting land-cover changes appear*.

    Why R26?
    --------
    IMAGE balances agricultural demands and allocates agricultural land within
    its 26 land-use regions. A grid-cell requirement that the reduction in
    cropland and the increase in natural land occur in exactly the same 5'
    cell is therefore too restrictive and can create a large artificial
    "unresolved" category.

    Primary mechanism decomposition
    -------------------------------
    The avoided-expansion / reversion decomposition is therefore performed
    after aggregating cropland stocks to R26.

    Let C0 be SSP2 cropland in START_YEAR and define a common-start biochar
    endpoint as

        C_bio_adj = C0 + (C_bio_END - C_bio_START).

    This removes the tiny scenario mismatch already present at START_YEAR while
    preserving the simulated biochar-induced cropland change.

    For each R26 region:

        SSP2 expansion   = max(C_SSP2_END - C0, 0)
        Biochar expansion = max(C_bio_adj - C0, 0)

        avoided expansion =
            max(SSP2 expansion - Biochar expansion, 0)

        SSP2 reversion   = max(C0 - C_SSP2_END, 0)
        Biochar reversion = max(C0 - C_bio_adj, 0)

        additional reversion =
            max(Biochar reversion - SSP2 reversion, 0)

    These two quantities exactly decompose any *reduction* in regional
    cropland. If biochar instead produces more cropland in a region, that is
    reported separately as "offsetting additional cropland use".

    Land-cover consequence
    ----------------------
    Natural-land change is NOT forced into the same grid cells as the cropland
    response. Instead, for each R26 region the code reports the biochar-induced
    change in the change of cropland, pasture, forest, other natural land, and
    built-up land:

        effect_L =
            (L_bio_END - L_bio_START)
            - (L_SSP2_END - L_SSP2_START)

    The five land-cover effects should close to approximately zero. This
    regional mass balance is the appropriate diagnostic for asking whether
    reduced cropland demand is expressed as more natural land, more pasture,
    or another land-cover adjustment.

    Grid-cell frontier fields are retained only for the separate productivity
    analysis, where spatial selection of the SSP2 agricultural frontier is the
    actual research question.
    """
    region_cells = standardize_region_cells(region_cells)

    bio_start, _, lat, lon, labels = load_glandcover_full_area_for_year(
        BIOCHAR_GLANDCOVER,
        BIOCHAR_GAREA,
        START_YEAR,
    )
    bio_end, _, _, _, _ = load_glandcover_full_area_for_year(
        BIOCHAR_GLANDCOVER,
        BIOCHAR_GAREA,
        END_YEAR,
    )
    base_start, _, _, _, _ = load_glandcover_full_area_for_year(
        BASELINE_GLANDCOVER,
        BASELINE_GAREA,
        START_YEAR,
    )
    base_end, _, _, _, _ = load_glandcover_full_area_for_year(
        BASELINE_GLANDCOVER,
        BASELINE_GAREA,
        END_YEAR,
    )

    # ------------------------------------------------------------------
    # Full-grid fields in ha
    # ------------------------------------------------------------------
    class_idx = {
        label: labels.index(label)
        for label in LANDCOVER_INDICES.values()
    }

    def field_ha(area_cube, label):
        return (
            area_cube[:, :, class_idx[label]]
            * 100.0
        ).astype(np.float64, copy=False)

    base_start_ha = {
        label: field_ha(base_start, label)
        for label in LANDCOVER_INDICES.values()
    }
    base_end_ha = {
        label: field_ha(base_end, label)
        for label in LANDCOVER_INDICES.values()
    }
    bio_start_ha = {
        label: field_ha(bio_start, label)
        for label in LANDCOVER_INDICES.values()
    }
    bio_end_ha = {
        label: field_ha(bio_end, label)
        for label in LANDCOVER_INDICES.values()
    }

    c0 = base_start_ha["Cropland"]
    cb = base_end_ha["Cropland"]
    cs0 = bio_start_ha["Cropland"]
    cs = bio_end_ha["Cropland"]

    n0 = (
        base_start_ha["Forest"]
        + base_start_ha["Other natural land"]
    )
    nb = (
        base_end_ha["Forest"]
        + base_end_ha["Other natural land"]
    )
    ns0 = (
        bio_start_ha["Forest"]
        + bio_start_ha["Other natural land"]
    )
    ns = (
        bio_end_ha["Forest"]
        + bio_end_ha["Other natural land"]
    )

    # ------------------------------------------------------------------
    # Grid-cell fields retained for the productivity analysis.
    #
    # The biochar cropland endpoint is normalised to the SSP2 START_YEAR
    # stock so that the small initial mismatch does not create artificial
    # "avoided expansion".
    # ------------------------------------------------------------------
    biochar_cropland_end_adjusted = (
        c0 + (cs - cs0)
    )

    baseline_expansion_grid = np.maximum(
        cb - c0,
        0.0,
    )
    biochar_expansion_grid = np.maximum(
        biochar_cropland_end_adjusted - c0,
        0.0,
    )
    avoided_expansion_grid = np.maximum(
        baseline_expansion_grid - biochar_expansion_grid,
        0.0,
    )

    baseline_reversion_grid = np.maximum(
        c0 - cb,
        0.0,
    )
    biochar_reversion_grid = np.maximum(
        c0 - biochar_cropland_end_adjusted,
        0.0,
    )
    additional_reversion_grid = np.maximum(
        biochar_reversion_grid - baseline_reversion_grid,
        0.0,
    )

    baseline_natural_loss_grid = np.maximum(
        n0 - nb,
        0.0,
    )
    baseline_natural_to_crop_grid = np.minimum(
        baseline_expansion_grid,
        baseline_natural_loss_grid,
    )

    # Existing panel-C definition, retained for exact comparison with the
    # manuscript figure.
    natural_effect_grid = (
        (ns - ns0)
        - (nb - n0)
    )

    # ------------------------------------------------------------------
    # Aggregate stocks to R26 BEFORE the nonlinear decomposition.
    # ------------------------------------------------------------------
    stocks = pd.DataFrame(
        index=IMAGE_REGIONS,
        dtype=np.float64,
    )

    stocks["SSP2 cropland 2030 (Mha)"] = aggregate_full_ha_to_r26(
        c0,
        region_cells,
    )
    stocks["SSP2 cropland 2100 (Mha)"] = aggregate_full_ha_to_r26(
        cb,
        region_cells,
    )
    stocks["Biochar cropland 2030 (Mha)"] = aggregate_full_ha_to_r26(
        cs0,
        region_cells,
    )
    stocks["Biochar cropland 2100 (Mha)"] = aggregate_full_ha_to_r26(
        cs,
        region_cells,
    )

    stocks["Biochar cropland 2100, common-start adjusted (Mha)"] = (
        stocks["SSP2 cropland 2030 (Mha)"]
        + stocks["Biochar cropland 2100 (Mha)"]
        - stocks["Biochar cropland 2030 (Mha)"]
    )

    c0_r = stocks["SSP2 cropland 2030 (Mha)"].to_numpy(dtype=np.float64)
    cb_r = stocks["SSP2 cropland 2100 (Mha)"].to_numpy(dtype=np.float64)
    cs_adj_r = stocks[
        "Biochar cropland 2100, common-start adjusted (Mha)"
    ].to_numpy(dtype=np.float64)

    baseline_expansion_r = np.maximum(
        cb_r - c0_r,
        0.0,
    )
    biochar_expansion_r = np.maximum(
        cs_adj_r - c0_r,
        0.0,
    )
    avoided_r = np.maximum(
        baseline_expansion_r - biochar_expansion_r,
        0.0,
    )

    baseline_reversion_r = np.maximum(
        c0_r - cb_r,
        0.0,
    )
    biochar_reversion_r = np.maximum(
        c0_r - cs_adj_r,
        0.0,
    )
    reversion_r = np.maximum(
        biochar_reversion_r - baseline_reversion_r,
        0.0,
    )

    signed_saving_r = cb_r - cs_adj_r
    reduced_crop_r = np.maximum(
        signed_saving_r,
        0.0,
    )
    additional_crop_r = np.maximum(
        -signed_saving_r,
        0.0,
    )

    r26_mechanisms = stocks.copy()
    r26_mechanisms["SSP2 cropland expansion from 2030 (Mha)"] = (
        baseline_expansion_r
    )
    r26_mechanisms["Biochar cropland expansion from 2030 (Mha)"] = (
        biochar_expansion_r
    )
    r26_mechanisms["SSP2 cropland reversion from 2030 (Mha)"] = (
        baseline_reversion_r
    )
    r26_mechanisms["Biochar cropland reversion from 2030 (Mha)"] = (
        biochar_reversion_r
    )
    r26_mechanisms["Avoided cropland expansion (Mha)"] = avoided_r
    r26_mechanisms["Additional cropland reversion (Mha)"] = reversion_r
    r26_mechanisms["Offsetting additional cropland use (Mha)"] = (
        additional_crop_r
    )
    r26_mechanisms["Reduced cropland relative to SSP2 (Mha)"] = (
        reduced_crop_r
    )
    r26_mechanisms["Signed cropland saving, SSP2 - biochar (Mha)"] = (
        signed_saving_r
    )
    r26_mechanisms["Biochar-induced cropland change (Mha)"] = (
        -signed_saving_r
    )

    r26_mechanisms = _add_cropland_mechanism_shares(
        r26_mechanisms
    )

    # ------------------------------------------------------------------
    # R10 summary: sum the R26 mechanism results, do not decompose at R10.
    # ------------------------------------------------------------------
    mechanism_additive_cols = [
        "Avoided cropland expansion (Mha)",
        "Additional cropland reversion (Mha)",
        "Offsetting additional cropland use (Mha)",
        "Reduced cropland relative to SSP2 (Mha)",
        "Signed cropland saving, SSP2 - biochar (Mha)",
        "Biochar-induced cropland change (Mha)",
        "SSP2 cropland expansion from 2030 (Mha)",
        "Biochar cropland expansion from 2030 (Mha)",
        "SSP2 cropland reversion from 2030 (Mha)",
        "Biochar cropland reversion from 2030 (Mha)",
    ]
    r10_mechanisms = _sum_r26_table_to_r10(
        r26_mechanisms,
        mechanism_additive_cols,
    )
    r10_mechanisms = _add_cropland_mechanism_shares(
        r10_mechanisms
    )

    # ------------------------------------------------------------------
    # Land-cover balance at R26: scenario-change difference by cover class.
    # ------------------------------------------------------------------
    r26_balance = pd.DataFrame(
        index=IMAGE_REGIONS,
        dtype=np.float64,
    )

    for label in LANDCOVER_INDICES.values():
        effect_grid = (
            (bio_end_ha[label] - bio_start_ha[label])
            - (base_end_ha[label] - base_start_ha[label])
        )
        r26_balance[f"{label} effect (Mha)"] = (
            aggregate_full_ha_to_r26(
                effect_grid,
                region_cells,
            )
        )

    r26_balance["Natural land effect (Mha)"] = (
        r26_balance["Forest effect (Mha)"]
        + r26_balance["Other natural land effect (Mha)"]
    )
    r26_balance["Land-cover closure error (Mha)"] = (
        r26_balance["Cropland effect (Mha)"]
        + r26_balance["Pasture effect (Mha)"]
        + r26_balance["Forest effect (Mha)"]
        + r26_balance["Other natural land effect (Mha)"]
        + r26_balance["Built-up area effect (Mha)"]
    )
    r26_balance["Lower cropland magnitude (Mha)"] = np.maximum(
        -r26_balance["Cropland effect (Mha)"].to_numpy(dtype=np.float64),
        0.0,
    )
    r26_balance["Positive natural-land effect (Mha)"] = np.maximum(
        r26_balance["Natural land effect (Mha)"].to_numpy(dtype=np.float64),
        0.0,
    )

    balance_additive_cols = [
        "Cropland effect (Mha)",
        "Pasture effect (Mha)",
        "Forest effect (Mha)",
        "Other natural land effect (Mha)",
        "Built-up area effect (Mha)",
        "Natural land effect (Mha)",
        "Land-cover closure error (Mha)",
        "Lower cropland magnitude (Mha)",
        "Positive natural-land effect (Mha)",
    ]
    r10_balance = _sum_r26_table_to_r10(
        r26_balance,
        balance_additive_cols,
    )

    # ------------------------------------------------------------------
    # Global summaries.
    # ------------------------------------------------------------------
    avoided_global = float(
        np.nansum(
            r26_mechanisms["Avoided cropland expansion (Mha)"].to_numpy(
                dtype=np.float64
            ),
            dtype=np.float64,
        )
    )
    reversion_global = float(
        np.nansum(
            r26_mechanisms["Additional cropland reversion (Mha)"].to_numpy(
                dtype=np.float64
            ),
            dtype=np.float64,
        )
    )
    additional_global = float(
        np.nansum(
            r26_mechanisms["Offsetting additional cropland use (Mha)"].to_numpy(
                dtype=np.float64
            ),
            dtype=np.float64,
        )
    )
    reduced_global = float(
        np.nansum(
            r26_mechanisms["Reduced cropland relative to SSP2 (Mha)"].to_numpy(
                dtype=np.float64
            ),
            dtype=np.float64,
        )
    )
    signed_crop_saving_global = float(
        np.nansum(
            r26_mechanisms[
                "Signed cropland saving, SSP2 - biochar (Mha)"
            ].to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
    )

    global_land_effects = {
        col: float(
            np.nansum(
                r26_balance[col].to_numpy(dtype=np.float64),
                dtype=np.float64,
            )
        )
        for col in balance_additive_cols
    }

    positive_nat_cell = float(
        np.nansum(
            np.maximum(
                natural_effect_grid,
                0.0,
            ),
            dtype=np.float64,
        )
        / 1e6
    )
    negative_nat_cell = float(
        np.nansum(
            np.minimum(
                natural_effect_grid,
                0.0,
            ),
            dtype=np.float64,
        )
        / 1e6
    )
    net_nat_cell = float(
        np.nansum(
            natural_effect_grid,
            dtype=np.float64,
        )
        / 1e6
    )

    start_crop_difference = cs0 - c0
    start_natural_difference = ns0 - n0

    global_summary = pd.Series(
        {
            "Avoided cropland expansion (Mha)": avoided_global,
            "Additional cropland reversion (Mha)": reversion_global,
            "Offsetting additional cropland use (Mha)": additional_global,
            "Reduced cropland relative to SSP2 (Mha)": reduced_global,
            "Signed cropland saving, SSP2 - biochar (Mha)":
                signed_crop_saving_global,
            "Avoided share of reduced cropland (%)":
                (
                    100.0 * avoided_global / reduced_global
                    if reduced_global > 0.0
                    else np.nan
                ),
            "Reversion share of reduced cropland (%)":
                (
                    100.0 * reversion_global / reduced_global
                    if reduced_global > 0.0
                    else np.nan
                ),
            "Net natural-land effect (Mha)": net_nat_cell,
            "Positive natural-land gains at grid-cell level (Mha)":
                positive_nat_cell,
            "Offsetting natural-land losses at grid-cell level (Mha)":
                negative_nat_cell,
            "Global cropland effect from land-cover balance (Mha)":
                global_land_effects["Cropland effect (Mha)"],
            "Global pasture effect (Mha)":
                global_land_effects["Pasture effect (Mha)"],
            "Global forest effect (Mha)":
                global_land_effects["Forest effect (Mha)"],
            "Global other-natural-land effect (Mha)":
                global_land_effects[
                    "Other natural land effect (Mha)"
                ],
            "Global built-up effect (Mha)":
                global_land_effects["Built-up area effect (Mha)"],
            "Global land-cover closure error (Mha)":
                global_land_effects[
                    "Land-cover closure error (Mha)"
                ],
            "Natural-land aggregation consistency error (Mha)":
                (
                    global_land_effects["Natural land effect (Mha)"]
                    - net_nat_cell
                ),
            "Natural-land gain per Mha net cropland saving":
                (
                    net_nat_cell / signed_crop_saving_global
                    if signed_crop_saving_global > 0.0
                    else np.nan
                ),
            "2030 signed cropland difference, biochar - SSP2 (Mha)":
                float(
                    np.nansum(
                        start_crop_difference,
                        dtype=np.float64,
                    )
                    / 1e6
                ),
            "2030 absolute cropland mismatch (Mha)":
                float(
                    np.nansum(
                        np.abs(start_crop_difference),
                        dtype=np.float64,
                    )
                    / 1e6
                ),
            "2030 signed natural-land difference, biochar - SSP2 (Mha)":
                float(
                    np.nansum(
                        start_natural_difference,
                        dtype=np.float64,
                    )
                    / 1e6
                ),
            "2030 absolute natural-land mismatch (Mha)":
                float(
                    np.nansum(
                        np.abs(start_natural_difference),
                        dtype=np.float64,
                    )
                    / 1e6
                ),
            "Maximum R26 mechanism closure error (Mha)":
                float(
                    np.nanmax(
                        np.abs(
                            r26_mechanisms[
                                "Mechanism closure error (Mha)"
                            ].to_numpy(dtype=np.float64)
                        )
                    )
                ),
            "Maximum absolute R26 land-cover closure error (Mha)":
                float(
                    np.nanmax(
                        np.abs(
                            r26_balance[
                                "Land-cover closure error (Mha)"
                            ].to_numpy(dtype=np.float64)
                        )
                    )
                ),
        },
        dtype=np.float64,
    )

    print("\n" + "=" * 122)
    print("R26 CROPLAND MECHANISM DECOMPOSITION")
    print("=" * 122)
    print(
        "Primary interpretation scale: IMAGE R26, matching regional agricultural "
        "demand/land allocation. The natural-land response is analysed separately "
        "through the regional land-cover balance."
    )
    print(
        f"Reduced cropland relative to SSP2: "
        f"{global_summary['Reduced cropland relative to SSP2 (Mha)']:.2f} Mha"
    )
    print(
        f"  Avoided future cropland expansion: "
        f"{global_summary['Avoided cropland expansion (Mha)']:.2f} Mha "
        f"({global_summary['Avoided share of reduced cropland (%)']:.1f}%)"
    )
    print(
        f"  Additional cropland reversion/contraction: "
        f"{global_summary['Additional cropland reversion (Mha)']:.2f} Mha "
        f"({global_summary['Reversion share of reduced cropland (%)']:.1f}%)"
    )
    print(
        f"Offsetting additional cropland use in regions where biochar has more "
        f"cropland than SSP2: "
        f"{global_summary['Offsetting additional cropland use (Mha)']:.2f} Mha"
    )
    print(
        f"Signed global cropland saving: "
        f"{global_summary['Signed cropland saving, SSP2 - biochar (Mha)']:.2f} Mha"
    )
    print(
        f"Net natural-land effect: "
        f"{global_summary['Net natural-land effect (Mha)']:.2f} Mha "
        f"(grid-cell positive gains "
        f"{global_summary['Positive natural-land gains at grid-cell level (Mha)']:.2f} Mha; "
        f"offsetting losses "
        f"{global_summary['Offsetting natural-land losses at grid-cell level (Mha)']:.2f} Mha)"
    )
    print(
        f"Global land-cover balance: cropland "
        f"{global_summary['Global cropland effect from land-cover balance (Mha)']:+.2f}, "
        f"pasture {global_summary['Global pasture effect (Mha)']:+.2f}, "
        f"natural land {global_summary['Net natural-land effect (Mha)']:+.2f}, "
        f"built-up {global_summary['Global built-up effect (Mha)']:+.2f} Mha."
    )
    print(
        f"2030 SSP2/Biochar absolute mismatch: cropland "
        f"{global_summary['2030 absolute cropland mismatch (Mha)']:.4f} Mha; "
        f"natural land "
        f"{global_summary['2030 absolute natural-land mismatch (Mha)']:.4f} Mha."
    )
    print(
        f"Closure checks: max R26 mechanism error "
        f"{global_summary['Maximum R26 mechanism closure error (Mha)']:.3e} Mha; "
        f"max R26 land-cover balance error "
        f"{global_summary['Maximum absolute R26 land-cover closure error (Mha)']:.3e} Mha; "
        f"natural-land aggregation consistency error "
        f"{global_summary['Natural-land aggregation consistency error (Mha)']:.3e} Mha."
    )

    print("\nR10 cropland mechanism summary [Mha; shares in %]")
    print(
        r10_mechanisms[
            [
                "Avoided cropland expansion (Mha)",
                "Additional cropland reversion (Mha)",
                "Offsetting additional cropland use (Mha)",
                "Reduced cropland relative to SSP2 (Mha)",
                "Signed cropland saving, SSP2 - biochar (Mha)",
                "Avoided share of reduced cropland (%)",
                "Reversion share of reduced cropland (%)",
            ]
        ]
        .round(3)
        .to_string()
    )

    print("\nR10 land-cover balance [biochar-induced change relative to SSP2, Mha]")
    print(
        r10_balance[
            [
                "Cropland effect (Mha)",
                "Pasture effect (Mha)",
                "Natural land effect (Mha)",
                "Built-up area effect (Mha)",
                "Land-cover closure error (Mha)",
            ]
        ]
        .round(3)
        .to_string()
    )

    return {
        "global_summary": global_summary,
        "r26_mechanism_summary": r26_mechanisms,
        "regional_summary": r10_mechanisms,
        "r26_land_balance": r26_balance,
        "r10_land_balance": r10_balance,
        "lat": lat,
        "lon": lon,
        "natural_effect_ha": natural_effect_grid,
        "baseline_expansion_ha": baseline_expansion_grid,
        "biochar_expansion_ha": biochar_expansion_grid,
        "avoided_expansion_ha": avoided_expansion_grid,
        "baseline_natural_to_crop_ha": baseline_natural_to_crop_grid,
        "additional_reversion_ha": additional_reversion_grid,
    }


def build_panel_c_mechanism_classification(
    mechanism_results,
    region_cells,
):
    """
    Classify each positive natural-land-gain cell by the dominant cropland
    mechanism of its IMAGE R26 allocation region.

    Codes
    -----
    0 = not mapped in Panel C (no positive natural-land gain or no identified
        regional cropland-saving mechanism)
    1 = avoided cropland expansion dominates the R26 cropland saving
    2 = additional cropland contraction/reversion dominates

    The classification deliberately does not require the natural-land gain and
    cropland reduction to occur in the same grid cell. IMAGE satisfies
    agricultural demand at R26, so the attribution scale matches the land
    allocation scale.
    """
    region_cells = standardize_region_cells(region_cells)
    r26 = mechanism_results["r26_mechanism_summary"]

    code_by_r26 = np.zeros(len(IMAGE_REGIONS), dtype=np.int8)
    share_avoided_by_r26 = np.full(
        len(IMAGE_REGIONS),
        np.nan,
        dtype=np.float64,
    )

    for i, region_name in enumerate(IMAGE_REGIONS):
        avoided = float(
            r26.loc[
                region_name,
                "Avoided cropland expansion (Mha)",
            ]
        )
        contraction = float(
            r26.loc[
                region_name,
                "Additional cropland reversion (Mha)",
            ]
        )
        identified = avoided + contraction

        if identified > 0.0:
            share_avoided_by_r26[i] = avoided / identified
            code_by_r26[i] = 1 if avoided >= contraction else 2

    natural_effect = np.asarray(
        mechanism_results["natural_effect_ha"],
        dtype=np.float64,
    )
    land_mask_flat = land_mask(RESOLUTION)

    if land_mask_flat.sum() != region_cells.size:
        raise ValueError(
            "land_mask and region_to_grid land-vector sizes differ."
        )

    land_codes = np.zeros(region_cells.size, dtype=np.int8)
    valid_region = (
        (region_cells >= 0)
        & (region_cells < len(IMAGE_REGIONS))
    )
    land_codes[valid_region] = code_by_r26[
        region_cells[valid_region]
    ]

    full_codes = np.zeros(land_mask_flat.size, dtype=np.int8)
    full_codes[land_mask_flat] = land_codes
    full_codes = full_codes.reshape(natural_effect.shape)

    positive_gain = (
        np.isfinite(natural_effect)
        & (natural_effect > MAP_GAIN_THRESHOLD_HA)
    )
    full_codes = np.where(
        positive_gain,
        full_codes,
        0,
    ).astype(np.int8)

    rows = []
    for code, label in [
        (1, "Avoided cropland expansion"),
        (2, "Cropland contraction/reversion"),
    ]:
        mask = full_codes == code
        area_mha = float(
            np.nansum(
                np.where(mask, natural_effect, 0.0),
                dtype=np.float64,
            )
            / 1e6
        )
        rows.append(
            {
                "Panel-C mechanism class": label,
                "Positive natural land gain in class (Mha)": area_mha,
            }
        )

    summary = pd.DataFrame(rows)

    omitted_positive_gain_mha = float(
        np.nansum(
            np.where(
                positive_gain & (full_codes == 0),
                natural_effect,
                0.0,
            ),
            dtype=np.float64,
        )
        / 1e6
    )

    print(
        "\nPanel C regional mechanism attribution of positive natural land gains"
    )
    print("-" * 100)
    print(summary.round(3).to_string(index=False))
    if omitted_positive_gain_mha > 0.0:
        print(
            f"  Diagnostic only: {omitted_positive_gain_mha:.3f} Mha of "
            "positive natural land gain occurs in R26 regions without either "
            "cropland-saving mechanism and is not assigned a Panel-C colour."
        )
    print(
        "Interpretation: map colours identify the dominant cropland-saving "
        "mechanism of the R26 region containing each positive natural land "
        "gain cell; they do not imply same-cell causality."
    )

    return {
        "classification_grid": full_codes,
        "code_by_r26": pd.Series(
            code_by_r26,
            index=IMAGE_REGIONS,
            name="Panel-C dominant mechanism code",
        ),
        "avoided_share_by_r26": pd.Series(
            share_avoided_by_r26,
            index=IMAGE_REGIONS,
            name="Avoided share of identified cropland mechanism",
        ),
        "map_summary": summary,
        "omitted_positive_gain_mha": omitted_positive_gain_mha,
    }


# =============================================================================
# Waterfall
# =============================================================================

def build_waterfall_inputs(baseline_net_df, biochar_net_df, classes):
    baseline = baseline_net_df.loc[list(R10.keys()), classes].sum(axis=1) / 10000.0
    biochar = biochar_net_df.loc[list(R10.keys()), classes].sum(axis=1) / 10000.0
    effect = biochar - baseline
    return {
        "baseline_total": float(baseline.sum()),
        "biochar_total": float(biochar.sum()),
        "regional_effects": effect,
    }


def plot_waterfall(ax, wf, ylabel):
    regional = wf["regional_effects"].copy()

    labels = ["SSP2"] + [R10_DISPLAY[r] for r in regional.index] + ["SSP2-Biochar"]
    x = np.arange(len(labels))

    bottoms = [0.0]
    heights = [wf["baseline_total"]]
    colors = [WF_ENDPOINT]

    running = wf["baseline_total"]
    connector_levels = [running]

    for effect in regional.values:
        if effect >= 0:
            bottoms.append(running)
            heights.append(effect)
            colors.append(WF_POS)
        else:
            bottoms.append(running + effect)
            heights.append(-effect)
            colors.append(WF_NEG)
        running += effect
        connector_levels.append(running)

    bottoms.append(0.0)
    heights.append(wf["biochar_total"])
    colors.append(WF_ENDPOINT)

    ax.bar(
        x, heights, bottom=bottoms, color=colors, edgecolor="black",
        linewidth=0.45, width=0.72, zorder=3
    )
    ax.axhline(0.0, color="black", linewidth=0.9, zorder=2)

    for i in range(len(labels) - 2):
        ax.plot(
            [x[i] + 0.36, x[i + 1] - 0.36],
            [connector_levels[i], connector_levels[i]],
            color=WF_CONNECTOR, linewidth=0.8, zorder=4,
        )
    ax.plot(
        [x[-2] + 0.36, x[-1] - 0.36],
        [running, running],
        color=WF_CONNECTOR, linewidth=0.8, zorder=4,
    )

    y_min = min(0.0, wf["baseline_total"], wf["biochar_total"], float(np.nanmin(bottoms)))
    y_max = max(
        0.0,
        wf["baseline_total"],
        wf["biochar_total"],
        float(np.nanmax(np.asarray(bottoms) + np.asarray(heights))),
    )
    y_range = max(y_max - y_min, 1.0)
    pad = 0.025 * y_range

    baseline_value = wf["baseline_total"]
    baseline_sign = 1.0 if baseline_value >= 0.0 else -1.0
    ax.text(
        x[0],
        baseline_value + baseline_sign * pad,
        fmt_num(baseline_value),
        ha="center",
        va="bottom" if baseline_value >= 0.0 else "top",
        fontsize=8.2,
        fontweight="bold",
    )

    current = wf["baseline_total"]
    for i, effect in enumerate(regional.values, start=1):
        if effect >= 0:
            y = current + effect + pad
            va = "bottom"
            text = f"+{fmt_num(effect)}"
        else:
            y = current + effect - pad
            va = "top"
            text = fmt_num(effect)
        ax.text(x[i], y, text, ha="center", va=va, fontsize=7.4)
        current += effect

    biochar_value = wf["biochar_total"]
    biochar_sign = 1.0 if biochar_value >= 0.0 else -1.0
    ax.text(
        x[-1],
        biochar_value + biochar_sign * pad,
        fmt_num(biochar_value),
        ha="center",
        va="bottom" if biochar_value >= 0.0 else "top",
        fontsize=8.2,
        fontweight="bold",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=48, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.30, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(y_min - 0.12 * y_range, y_max + 0.17 * y_range)


# =============================================================================
# GNLCT / panel d
# =============================================================================

def load_gnlct_land_for_year(gnlct_nc, year, var_name="GNLCT"):
    with xr.open_dataset(gnlct_nc, engine="netcdf4") as ds:
        da = select_year(ds[var_name], year).squeeze(drop=True)
        lat_dim = find_dim(da, ["latitude", "lat", "y"])
        lon_dim = find_dim(da, ["longitude", "lon", "x"])
        da = da.transpose(lat_dim, lon_dim)
        full = np.asarray(da.values, dtype=np.float64)

    if full.ndim != 2:
        raise ValueError("GNLCT must resolve to a 2-D lat-lon grid.")

    flat = full.reshape(-1)
    mask = land_mask(RESOLUTION)
    if mask.size != flat.size:
        raise ValueError("GNLCT grid does not match land_mask.")
    return flat[mask].astype(int)


def build_panel_d_composition(effect_ha_full, region_cells):
    region_cells = standardize_region_cells(region_cells)
    gain_ha_land = land_from_full_grid(effect_ha_full).astype(np.float64)
    gain_ha_land = np.where(np.isfinite(gain_ha_land) & (gain_ha_land > 0.0), gain_ha_land, 0.0)

    gnlct_2100_land = load_gnlct_land_for_year(BIOCHAR_GNLCT, END_YEAR)

    if gain_ha_land.size != region_cells.size or gnlct_2100_land.size != region_cells.size:
        raise ValueError("Panel-d inputs do not match region_to_grid size.")

    grouped = pd.DataFrame(0.0, index=PANEL_D_REGIONS, columns=list(GNLCT_GROUPS.keys()))

    for region in PANEL_D_REGIONS:
        reg_ids = [IMAGE_REGIONS.index(r) for r in R10[region]]
        reg_mask = np.isin(region_cells, reg_ids) & (gain_ha_land > 0.0)

        for group, ids in GNLCT_GROUPS.items():
            grouped.loc[region, group] = np.nansum(
                gain_ha_land[reg_mask & np.isin(gnlct_2100_land, ids)]
            )

    totals = grouped.sum(axis=1)
    composition = grouped.div(totals.replace(0.0, np.nan), axis=0) * 100.0
    composition = composition.fillna(0.0)
    return composition


def plot_panel_d(ax, composition_pct):
    y = np.arange(len(composition_pct))
    left = np.zeros(len(composition_pct), dtype=np.float64)

    for group in composition_pct.columns:
        values = composition_pct[group].to_numpy(dtype=np.float64)
        ax.barh(
            y, values, left=left, height=0.70,
            color=GNLCT_COLORS[group], label=group
        )
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels([R10_DISPLAY.get(r, r) for r in composition_pct.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Relative composition of natural land gains (%)")

    ax.grid(axis="x", linestyle="--", linewidth=0.7, alpha=0.30)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=9.4,
        columnspacing=1.0,
        handletextpad=0.5,
    )


# =============================================================================
# Panel c map
# =============================================================================

def plot_panel_c_map(
    ax,
    effect_ha,
    cropland_bg_mask,
    lat,
    lon,
    mechanism_classification,
):
    """
    Plot positive biochar-induced natural land gains, coloured by the dominant
    avoided-expansion versus contraction mechanism of the containing R26 region.

    The classification is regional attribution and does not imply same-cell
    causality between reduced cropland and increased natural land.
    """
    ax.set_global()
    ax.add_feature(
        cfeature.OCEAN,
        facecolor=MAP_OCEAN,
        zorder=0,
    )
    ax.add_feature(
        cfeature.LAND,
        facecolor=MAP_LAND,
        zorder=0,
    )
    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.35,
        zorder=4,
    )
    ax.add_feature(
        cfeature.BORDERS,
        linewidth=0.20,
        alpha=0.50,
        zorder=4,
    )

    lon2d, lat2d = np.meshgrid(lon, lat)

    bg = np.ma.masked_where(
        ~cropland_bg_mask,
        np.ones_like(effect_ha, dtype=np.float64),
    )
    ax.pcolormesh(
        lon,
        lat,
        bg,
        cmap=ListedColormap([MAP_BG_CROPLAND]),
        shading="auto",
        transform=ccrs.PlateCarree(),
        rasterized=True,
        zorder=1,
    )

    classes = mechanism_classification["classification_grid"]
    if classes.shape != effect_ha.shape:
        raise ValueError(
            "Panel-C mechanism classification does not match effect grid."
        )

    for code, color in [
        (1, MAP_GAIN_AVOIDED),
        (2, MAP_GAIN_REVERSION),
    ]:
        mask = (
            np.isfinite(effect_ha)
            & (effect_ha > MAP_GAIN_THRESHOLD_HA)
            & (classes == code)
        )
        ax.scatter(
            lon2d[mask],
            lat2d[mask],
            s=MAP_GAIN_MARKER_SIZE,
            c=color,
            marker="s",
            linewidths=0,
            transform=ccrs.PlateCarree(),
            rasterized=True,
            zorder=3,
        )

    handles = [
        Patch(
            facecolor=MAP_BG_CROPLAND,
            edgecolor="none",
            label="Remaining cropland",
        ),
        Patch(
            facecolor=MAP_GAIN_AVOIDED,
            edgecolor="none",
            label="Natural land gained by avoided expansion",
        ),
        Patch(
            facecolor=MAP_GAIN_REVERSION,
            edgecolor="none",
            label="Natural land gained by contraction",
        ),
    ]

    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=1,
        frameon=False,
        fontsize=9.5,
        handlelength=1.2,
        handletextpad=0.5,
        borderpad=0.2,
        labelspacing=0.40,
    )

    ax.text(
        0.60,
        0.16,
        MAP_ANNOTATION,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.0,
    )


# =============================================================================
# Manuscript-oriented quantitative output
# =============================================================================

def print_figure_results_summary(
    wf_cropland,
    wf_natural,
    mechanism_results,
    panel_c_mechanism,
    composition_pct,
):
    """
    Print the headline quantities most directly useful for the Results section
    accompanying the four-panel land-use figure.
    """
    g = mechanism_results["global_summary"]
    r10_mech = mechanism_results["regional_summary"]
    r10_balance = mechanism_results["r10_land_balance"]

    print("\n" + "=" * 122)
    print("FOUR-PANEL LAND-USE FIGURE: RESULTS SUMMARY")
    print("=" * 122)

    crop_effect = (
        wf_cropland["biochar_total"]
        - wf_cropland["baseline_total"]
    )
    natural_effect = (
        wf_natural["biochar_total"]
        - wf_natural["baseline_total"]
    )

    print("\nPanels a-b: global land-use change, 2030-2100")
    print(
        f"  Cropland net change: SSP2={wf_cropland['baseline_total']:+.2f} Mha; "
        f"SSP2-Biochar={wf_cropland['biochar_total']:+.2f} Mha; "
        f"biochar-induced difference={crop_effect:+.2f} Mha."
    )
    print(
        f"  Natural land net change: SSP2={wf_natural['baseline_total']:+.2f} Mha; "
        f"SSP2-Biochar={wf_natural['biochar_total']:+.2f} Mha; "
        f"biochar-induced difference={natural_effect:+.2f} Mha."
    )
    print(
        f"  Global land-cover balance of the biochar effect: "
        f"cropland {g['Global cropland effect from land-cover balance (Mha)']:+.2f}, "
        f"pasture {g['Global pasture effect (Mha)']:+.2f}, "
        f"natural land {g['Net natural-land effect (Mha)']:+.2f}, "
        f"built-up {g['Global built-up effect (Mha)']:+.2f} Mha."
    )

    natural_sorted = r10_balance[
        "Natural land effect (Mha)"
    ].sort_values(ascending=False)
    crop_sorted = r10_balance[
        "Cropland effect (Mha)"
    ].sort_values()

    print("\nLargest regional biochar-induced natural land gains")
    for region, value in natural_sorted.head(5).items():
        print(
            f"  {R10_DISPLAY.get(region, region):<20s} {value:+.2f} Mha"
        )

    print("\nLargest regional cropland reductions")
    for region, value in crop_sorted.head(5).items():
        print(
            f"  {R10_DISPLAY.get(region, region):<20s} {value:+.2f} Mha"
        )

    print("\nPanel c: mechanism of reduced cropland")
    print(
        f"  Reduced cropland relative to SSP2: "
        f"{g['Reduced cropland relative to SSP2 (Mha)']:.2f} Mha."
    )
    print(
        f"  Avoided future cropland expansion: "
        f"{g['Avoided cropland expansion (Mha)']:.2f} Mha "
        f"({g['Avoided share of reduced cropland (%)']:.1f}%)."
    )
    print(
        f"  Additional cropland contraction/reversion: "
        f"{g['Additional cropland reversion (Mha)']:.2f} Mha "
        f"({g['Reversion share of reduced cropland (%)']:.1f}%)."
    )
    print(
        f"  Offsetting additional cropland use in other R26 regions: "
        f"{g['Offsetting additional cropland use (Mha)']:.2f} Mha."
    )

    expansion_dom = r10_mech[
        r10_mech["Avoided share of reduced cropland (%)"] > 50.0
    ]["Reduced cropland relative to SSP2 (Mha)"].sort_values(
        ascending=False
    )
    contraction_dom = r10_mech[
        r10_mech["Reversion share of reduced cropland (%)"] > 50.0
    ]["Reduced cropland relative to SSP2 (Mha)"].sort_values(
        ascending=False
    )

    if len(expansion_dom):
        print(
            "  Avoided-expansion-dominated R10 regions: "
            + ", ".join(
                f"{R10_DISPLAY.get(r, r)} ({v:.2f} Mha)"
                for r, v in expansion_dom.items()
            )
        )
    if len(contraction_dom):
        print(
            "  Contraction-dominated R10 regions: "
            + ", ".join(
                f"{R10_DISPLAY.get(r, r)} ({v:.2f} Mha)"
                for r, v in contraction_dom.items()
            )
        )

    mapped = panel_c_mechanism["map_summary"].set_index(
        "Panel-C mechanism class"
    )
    for label in [
        "Avoided cropland expansion",
        "Cropland contraction/reversion",
    ]:
        value = float(
            mapped.loc[
                label,
                "Positive natural land gain in class (Mha)",
            ]
        )
        print(
            f"  Positive natural land occurring in R26 regions dominated by "
            f"{label.lower()}: {value:.2f} Mha."
        )

    omitted = float(
        panel_c_mechanism.get(
            "omitted_positive_gain_mha",
            0.0,
        )
    )
    if omitted > 0.0:
        print(
            f"  Diagnostic only: {omitted:.2f} Mha of positive natural land "
            "gain occurs in R26 regions with neither cropland-saving mechanism "
            "and is therefore not assigned a Panel-C colour."
        )

    print("\nPanel d: relative composition of positive natural land gains (%)")
    print(composition_pct.round(1).to_string())

    print("\nDominant natural land class by region")
    for region, row in composition_pct.iterrows():
        dominant = row.idxmax()
        share = float(row.max())
        print(
            f"  {R10_DISPLAY.get(region, region):<20s} "
            f"{dominant}: {share:.1f}%"
        )

    print("\nAccounting checks")
    print(
        f"  Net natural land effect: "
        f"{g['Net natural-land effect (Mha)']:.2f} Mha; "
        f"positive gains="
        f"{g['Positive natural-land gains at grid-cell level (Mha)']:.2f} Mha; "
        f"offsetting losses="
        f"{g['Offsetting natural-land losses at grid-cell level (Mha)']:.2f} Mha."
    )
    print(
        f"  Maximum R26 mechanism closure error: "
        f"{g['Maximum R26 mechanism closure error (Mha)']:.3e} Mha."
    )
    print(
        f"  Maximum R26 land-cover closure error: "
        f"{g['Maximum absolute R26 land-cover closure error (Mha)']:.3e} Mha."
    )


# =============================================================================
# Main
# =============================================================================

def plot_biochar_land_use_figure(save=True, show=SHOW_FIGURE):
    set_style()

    region_cells, _ = region_to_grid(RESOLUTION)
    region_cells = standardize_region_cells(region_cells)

    baseline_net_df = compute_glandcover_scenario_net_change_r10(
        BASELINE_GLANDCOVER, BASELINE_GAREA, region_cells
    )
    biochar_net_df = compute_glandcover_scenario_net_change_r10(
        BIOCHAR_GLANDCOVER, BIOCHAR_GAREA, region_cells
    )

    wf_cropland = build_waterfall_inputs(baseline_net_df, biochar_net_df, ["Cropland"])
    wf_natural = build_waterfall_inputs(baseline_net_df, biochar_net_df, ["Forest", "Other natural land"])

    effect_ha, cropland_bg_mask, lat, lon = compute_spatial_natural_land_effect()
    composition_pct = build_panel_d_composition(effect_ha, region_cells)

    mechanism_results = analyse_land_sparing_mechanisms(
        region_cells
    )
    panel_c_mechanism = build_panel_c_mechanism_classification(
        mechanism_results,
        region_cells,
    )

    fig = plt.figure(figsize=FIGSIZE)
    outer = fig.add_gridspec(2, 1, height_ratios=(1.0, 1.10), hspace=0.60)
    gs_top = outer[0].subgridspec(1, 2, wspace=0.20, width_ratios=(1.0, 1.0))
    gs_bottom = outer[1].subgridspec(1, 2, wspace=0.18, width_ratios=(1.14, 0.86))

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_bottom[0, 0], projection=ccrs.Robinson())
    ax_d = fig.add_subplot(gs_bottom[0, 1])

    plot_waterfall(ax_a, wf_cropland, "Net cropland change (Mha)")
    ax_a.set_title(
        panel_title("a", "Biochar-induced net change in cropland, 2030-2100"),
        loc="left", fontweight="normal", pad=18,
    )

    plot_waterfall(ax_b, wf_natural, "Net forest + natural land change (Mha)")
    ax_b.set_title(
        panel_title("b", "Biochar-induced net change in natural land, 2030-2100"),
        loc="left", fontweight="normal", pad=18,
    )

    plot_panel_c_map(
        ax_c,
        effect_ha,
        cropland_bg_mask,
        lat,
        lon,
        panel_c_mechanism,
    )
    ax_c.set_title(
        panel_title(
            "c",
            "Biochar-induced natural land gains, 2030-2100",
        ),
        loc="left", fontweight="normal", pad=12,
    )

    plot_panel_d(ax_d, composition_pct)
    ax_d.set_title(
        panel_title("d", "Composition of natural land gains, 2030-2100"),
        loc="left", fontweight="normal", pad=12,
    )

    # Shared legend for panels a and b, placed below the top-row x-axis labels.
    legend_handles = [
        Patch(facecolor=WF_ENDPOINT, edgecolor="black", linewidth=0.45, label="Scenario net change"),
        Patch(facecolor=WF_POS, edgecolor="none", label="Biochar increases area"),
        Patch(facecolor=WF_NEG, edgecolor="none", label="Biochar decreases area"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.545),
        ncol=3,
        frameon=False,
        fontsize=10.2,
        handlelength=1.4,
        columnspacing=1.6,
    )

    fig.subplots_adjust(left=0.055, right=0.985, top=0.93, bottom=0.085)

    outputs = {
        "fig": fig,
        "axes": {"a": ax_a, "b": ax_b, "c": ax_c, "d": ax_d},
        "baseline_net_df_km2": baseline_net_df,
        "biochar_net_df_km2": biochar_net_df,
        "wf_cropland": wf_cropland,
        "wf_natural": wf_natural,
        "panel_c_effect_ha": effect_ha,
        "panel_d_composition_pct": composition_pct,
        "panel_c_positive_gain_sum_mha": float(np.nansum(np.where(effect_ha > 0.0, effect_ha, 0.0)) / 1e6),
        "panel_c_net_gain_sum_mha": float(np.nansum(effect_ha) / 1e6),
        "mechanism_results": mechanism_results,
        "panel_c_mechanism": panel_c_mechanism,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
        pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
        composition_csv = OUTPUT_DIR / f"{OUTPUT_STEM}_panel_d_composition_pct.csv"

        fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        composition_pct.to_csv(composition_csv)

        mechanism_global_csv = OUTPUT_DIR / (
            f"{MECHANISM_OUTPUT_STEM}_global_summary.csv"
        )
        mechanism_regional_csv = OUTPUT_DIR / (
            f"{MECHANISM_OUTPUT_STEM}_r10_summary.csv"
        )
        mechanism_r26_csv = OUTPUT_DIR / (
            f"{MECHANISM_OUTPUT_STEM}_r26_summary.csv"
        )
        mechanism_r26_balance_csv = OUTPUT_DIR / (
            f"{MECHANISM_OUTPUT_STEM}_r26_land_balance.csv"
        )
        mechanism_r10_balance_csv = OUTPUT_DIR / (
            f"{MECHANISM_OUTPUT_STEM}_r10_land_balance.csv"
        )
        panel_c_mechanism_csv = OUTPUT_DIR / (
            f"{OUTPUT_STEM}_panel_c_mechanism_summary.csv"
        )

        mechanism_results["global_summary"].to_frame(
            name="Value"
        ).to_csv(mechanism_global_csv)
        mechanism_results["regional_summary"].to_csv(
            mechanism_regional_csv
        )
        mechanism_results["r26_mechanism_summary"].to_csv(
            mechanism_r26_csv
        )
        mechanism_results["r26_land_balance"].to_csv(
            mechanism_r26_balance_csv
        )
        mechanism_results["r10_land_balance"].to_csv(
            mechanism_r10_balance_csv
        )
        panel_c_mechanism["map_summary"].to_csv(
            panel_c_mechanism_csv,
            index=False,
        )


        outputs.update(
            {
                "png_path": png_path,
                "pdf_path": pdf_path,
                "composition_csv": composition_csv,
                "mechanism_global_csv": mechanism_global_csv,
                "mechanism_regional_csv": mechanism_regional_csv,
                "mechanism_r26_csv": mechanism_r26_csv,
                "mechanism_r26_balance_csv": mechanism_r26_balance_csv,
                "mechanism_r10_balance_csv": mechanism_r10_balance_csv,
                "panel_c_mechanism_csv": panel_c_mechanism_csv,
            }
        )

        print("Saved:")
        print(png_path)
        print(pdf_path)
        print(composition_csv)
        print(mechanism_global_csv)
        print(mechanism_regional_csv)
        print(mechanism_r26_csv)
        print(mechanism_r26_balance_csv)
        print(mechanism_r10_balance_csv)
        print(panel_c_mechanism_csv)
    print_figure_results_summary(
        wf_cropland=wf_cropland,
        wf_natural=wf_natural,
        mechanism_results=mechanism_results,
        panel_c_mechanism=panel_c_mechanism,
        composition_pct=composition_pct,
    )

    print(
        "\nPanel C check:\n"
        f"  Net sum of mapped natural land effect = {outputs['panel_c_net_gain_sum_mha']:.2f} Mha\n"
        f"  Sum of positive mapped natural land gains only = {outputs['panel_c_positive_gain_sum_mha']:.2f} Mha\n"
        "  Positive-gain cells are coloured by the dominant avoided-expansion vs contraction mechanism\n"
        "  of their IMAGE R26 allocation region; the colour is regional attribution, not same-cell causality."
    )

    if show:
        plt.show()

    return outputs


if __name__ == "__main__":
    plot_biochar_land_use_figure(save=True, show=True)

import prism
import pym 
import os
import io
import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from matplotlib.colors import BoundaryNorm, ListedColormap
from timer.globals.constants import EPS

""" Define dimensions"""
Grid = prism.Dimension("cell", list(range(2160*4320)))
GridLat = prism.Dimension("cell_x", list(range(2160)))
GridLon = prism.Dimension("cell-Y", list(range(4320)))
Grid30mX = prism.Dimension("cell_x", list(range(360)))
Grid30mY = prism.Dimension("cell_y", list(range(720)))
Grid30m = prism.Dimension("cell", list(range(360*720)))
LandGrid = prism.Dimension("grid_timer", list(range(2215829)))
LandGrid30m = prism.Dimension("grid_timer", list(range(66663)))
Region26 = prism.Dimension('region26', list(range(26)))
Region27 = prism.Dimension('region', list(range(27)))
ImportRegion = prism.Dimension('import_reg', list(range(27)))
Region28 = prism.Dimension('region', list(range(28)))
LandCover = prism.Dimension("land_cover", list(range(5)))
LandCoverPlus = prism.Dimension("land_cover_det", list(range(7)))
Biomes = prism.Dimension('biome', list(range(20)))
GrassLand = prism.Dimension("grassland", list(range(2)))
Crop12 = prism.Dimension('crop12', list(range(12)))
Crop22 = prism.Dimension('crop22', list(range(22)))
Crop15 = prism.Dimension('crop15', list(range(15)))
Crop16 = prism.Dimension('crop16', list(range(16)))
Crop17 = prism.Dimension('crop17', list(range(17)))
Crop19 = prism.Dimension('crop19', list(range(19)))
Crop26 = prism.Dimension('crop26', list(range(26)))
Crop37 = prism.Dimension('crop38', list(range(37)))
Crop38 = prism.Dimension('crop38', list(range(38)))
BioenergyCrop3 = prism.Dimension('magnet_bfcrop', list(range(3)))
BioenergyCrop5 = prism.Dimension('bio_crop', list(range(5)))
Residue = prism.Dimension("res", list(range(17)))
ForestManagement = prism.Dimension("forest_manag", list(range(3)))
ForestManagementPlus = prism.Dimension("forest_manag_plus", list(range(4)))
CarbonPool = prism.Dimension("c_pool", list(range(9)))
CarbonPool4 = prism.Dimension("carb", list(range(4)))
CarbonPoolReduced = prism.Dimension("c_pool", list(range(2)))
PopDim = prism.Dimension("pop_dim", list(range(7)))
ResCalc = prism.Dimension('res_calc', list(range(2)))
NFCAREAT = prism.Dimension('nfcareat', list(range(3)))
NBC2 = prism.Dimension('nbc', list(range(20)))
NBC = prism.Dimension('nbc', list(range(10)))
DiscCat = prism.Dimension('cat', list(range(14)))
Region = prism.Dimension('region', list(range(27)))
LandRentBin = prism.Dimension('bin', list(range(401)))
BioTechnologyPlus = prism.Dimension('bio_tech', list(range(20)))
BioTechnology = prism.Dimension('bio_tech', list(range(11)))
BLFTechnology = prism.Dimension('bio_tech', list(range(18)))
BLFTechnologyMain = prism.Dimension('bio_tech', list(range(9)))
BSFTechnology = prism.Dimension('bio_tech', list(range(2)))
BSFTechnologyPlus = prism.Dimension('bio_tech', list(range(3)))
BioType = prism.Dimension('bio_type', list(range(3)))
NLandType = prism.Dimension('land_type', list(range(2)))
Sector = prism.Dimension('sector', list(range(5)))
SectorBLF = prism.Dimension('sector', list(range(6)))
SectorBSF = prism.Dimension('sector', list(range(6)))
SectorBio = prism.Dimension('sector', list(range(8)))
Carrier = prism.Dimension('carrier', list(range(8)))
EmisSector = prism.Dimension('sector', list(range(13)))
Fuel = prism.Dimension('fuel', list(range(6)))
H2_Fuel = prism.Dimension('fuel', list(range(5)))
NonEn_Fuel = prism.Dimension('fuel', list(range(7)))
BioCostCat = prism.Dimension('bio_cost', list(range(6)))
ERWGrade = prism.Dimension('erw_grade', list(range(100)))
ERWDim = prism.Dimension('erw_dim', list(range(2)))
NGST = prism.Dimension('ngst', list(range(3)))
NFPT = prism.Dimension('nfpt', list(range(6)))
NAT = prism.Dimension('nat', list(range(6)))
NFHTT = prism.Dimension('nfhtt', list(range(5)))
NWCT = prism.Dimension('nwct', list(range(4)))
NTRAD = prism.Dimension('ntrad', list(range(4)))
NWC = prism.Dimension('nwd', list(range(3)))
WoodProduct = prism.Dimension('wood_prod', list(range(2)))
RCC = prism.Dimension('bin', list(range(9)))
RCCPlus = prism.Dimension('bin_plus', list(range(10)))
PyrolysisTechnology = prism.Dimension('tech', list(range(2)))
BiocharFeedstock = prism.Dimension('feed', list(range(16)))
YieldHorizon = prism.Dimension('yield_year', list(range(5)))
TransportMode = prism.Dimension('mode', list(range(7)))
EmisCat = prism.Dimension('cat', list(range(8)))
Fertiliser = prism.Dimension('fertiliser', list(range(5)))
SoilDepth = prism.Dimension('soil_depth', list(range(6)))
TimePeriods = prism.Dimension('period', list(range(27)))
HarvestGroup3 = prism.Dimension("harvest_group3", list(range(3)))

""" Specify data directories """
global_data_dir = Path(r"X:\user\orhalem\image_biochar\timer\data\data\global")
land_data_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/sa/landuse')
land_emis_data_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/data/landemis')
land_output_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/sa/landuse/output')
affor_data_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/sa/afforestation')
biocrops_data_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/sa/biomass')
res_data_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/data/residues')
erw_data_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/data/erw')
erw_sa_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/sa/erw')
biochar_data_dir = Path(r'X:/user/orhalem/image_biochar/timer/data/data/biochar')
emis_results_data_dir = Path(r'X:\user\scherrenbm\ModelDevelopment\IMAGE_Development\IMAGE_Biochar\Biochar_Code\SSP2\output\emissions')

""" Define flags """
@prism.global_flag
class FlagUseAg(prism.Flag):
    default = True
    options = [False, True]

@prism.global_flag
class FlagUseYear1(prism.Flag):
    default = True
    options = [False, True]

@prism.global_flag
class FlagEmisType1(prism.Flag):
    default = True
    options = [False, True]

@prism.global_flag
class FlagEmisType2(prism.Flag):
    default = True
    options = [False, True]


""" Define parameters """
t_scen: prism.Scalar = prism.Q_(2025, 'year')

""" Define helper functions """
def region_to_grid(resolution: str): # resolution: 5m or 30m
    mat_data = sio.loadmat(affor_data_dir / 'run_data.mat')
    GREG = mat_data[f'GREG{resolution}']
    GREG = GREG.reshape(-1)
    mask_land = ~np.isnan(GREG) 
    region_cells = GREG[mask_land].astype(np.int8) - 1
    region_mask = (region_cells >= 0) & (region_cells < 26)   
    region_cells[~region_mask] = -1
   
    return region_cells.astype(np.int16), region_mask

def land_mask(resolution: str, order_F=False):
    mat_data = sio.loadmat(affor_data_dir / 'run_data.mat')
    GREG = mat_data[f'GREG{resolution}']

    if not order_F:
        GREG = GREG.reshape(-1)
        return ~np.isnan(GREG) 
    else: 
        GREG = GREG.reshape(-1, order="F")
        return ~np.isnan(GREG) 

def time_step(t):
    return int((t.magnitude - 1970) / 5)

def time_step_scen(t):
    return int((t.magnitude - 2025) / 5)

def safe_division(numerator, denominator, mask):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((denominator > EPS) & mask, numerator / denominator, 0.0)
    
def create_alt5min_file():
    IMAGE_grid = xr.open_dataset(biocrops_data_dir / "IMAGE_grid_5min.nc", engine="netcdf4")
    etopo = xr.open_dataset(biocrops_data_dir / "ETOPO_2022_v1_60s_N90W180_surface.nc", engine="netcdf4")
    z = etopo['z']

    def mode_func(x, axis=None):
        x = np.asarray(x)
        mode_result = stats.mode(x, axis=axis, nan_policy="omit", keepdims=False)
        return mode_result.mode

    z_5min = z.coarsen(lat=5, lon=5, boundary="trim").reduce(mode_func)

    IMAGE_grid_trimmed = IMAGE_grid.isel(g=slice(0, -17221))

    alt5min = z_5min.interp(
        lat=xr.DataArray(IMAGE_grid_trimmed["lat"].values, dims="g"),
        lon=xr.DataArray(IMAGE_grid_trimmed["lon"].values, dims="g"),
    )
    alt5min = prism.array(alt5min, dims=[LandGrid])
    pym.write_mym(alt5min.values, filename="alt5min.out", variable_name="Altitude")

def read_excel_column_F(filepath: str, sheet: str = 'landprice_selected',
    nrows: int = 338, skiprows: int = 1) -> np.ndarray:
    
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Excel file not found: {filepath}")
    try:
        df = pd.read_excel(filepath, sheet_name=sheet,
            usecols='F', header=None,
            skiprows=skiprows, nrows=nrows, engine='openpyxl')

    except Exception as e:

        raise RuntimeError(f"Failed to read column F from {filepath} (sheet={sheet}). "
                f"Original error: {e}") from e

    arr = df.iloc[:, 0].to_numpy(dtype=float).squeeze()
    if arr.size != nrows:
        raise ValueError(
            f"Read {arr.size} values from {filepath} but expected {nrows}. ")

    return arr

def correct_affor_cell_order(vals_1d):
    mat_data = sio.loadmat(affor_data_dir / "run_data.mat")
    GREG5m = mat_data["GREG5m"].astype(np.float32, copy=False)

    GREG5m_full_F = GREG5m.reshape(-1, order="F")
    mask_land_5m = ~np.isnan(GREG5m_full_F)

    GREG5m_full_noF = GREG5m.reshape(-1)
    mask_land_5m_noF = ~np.isnan(GREG5m_full_noF)

    full_grid = np.full(2160 * 4320, np.nan, dtype=np.float32)
    full_grid[mask_land_5m] = vals_1d
    full_grid_2d = full_grid.reshape(2160, 4320, order="F")
    reshaped = full_grid_2d.reshape(-1)

    return reshaped[mask_land_5m_noF]

def convert_to_affor_cell_order(vals_1d):
    mat_data = sio.loadmat(affor_data_dir / "run_data.mat")
    GREG5m = mat_data["GREG5m"].astype(np.float32, copy=False)

    GREG5m_full_F = GREG5m.reshape(-1, order="F")
    mask_land_5m_F = ~np.isnan(GREG5m_full_F)

    GREG5m_full_noF = GREG5m.reshape(-1)
    mask_land_5m_noF = ~np.isnan(GREG5m_full_noF)

    full_grid = np.full(2160 * 4320, np.nan, dtype=np.float32)
    full_grid[mask_land_5m_noF] = vals_1d

    full_grid_2d = full_grid.reshape(2160, 4320)
    reshaped = full_grid_2d.reshape(-1, order="F")

    return reshaped[mask_land_5m_F]

# def plot_gridded_variable(
#     prism_var,
#     res,
#     years,
#     title_prefix,
#     cmap,
#     affor_var=False,
#     vmin=None,
#     vmax=None,
# ):
    
#     if res == "5min":
#         grid_file = r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\sa\biomass\IMAGE_grid_5min.nc"
#         grid_ds = xr.open_dataset(grid_file, engine="netcdf4")
#         lats = grid_ds["lat"].isel(g=slice(0, -17221)).values
#         lons = grid_ds["lon"].isel(g=slice(0, -17221)).values
#     elif res == "30min":
#         grid_file = r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\sa\biomass\IMAGE_grid_coordinates.nc"
#         grid_ds = xr.open_dataset(grid_file)
#         lats = grid_ds["lat"].isel(grid=slice(0, -233)).values
#         lons = grid_ds["lon"].isel(grid=slice(0, -233)).values
#     else:
#         raise ValueError("res must be '5min' or '30min'")

#     if affor_var:
#         if res != "5min":
#             raise ValueError("affor_var=True only supported for res='5min'")

#         value_list = []
#         year_labels = []

#         if years is None:
#             vals = correct_affor_cell_order(prism_var.values)
#             value_list.append(vals)
#             year_labels.append(None)
#         else:
#             for year in years:
#                 t = prism.Q_(year, "year")
#                 vals = correct_affor_cell_order(prism_var.get(t).values)
#                 value_list.append(vals)
#                 year_labels.append(year)

#     else:
#         value_list = []
#         year_labels = []

#         if years is None:
#             value_list.append(prism_var.values)
#             year_labels.append(None)
#         else:
#             for year in years:
#                 t = prism.Q_(year, "year")
#                 vals = prism_var.get(t).values
#                 value_list.append(vals)
#                 year_labels.append(year)

#     all_vals = np.concatenate([np.ravel(v) for v in value_list])
#     finite_mask = np.isfinite(all_vals)

#     if vmin is None:
#         vmin = np.nanpercentile(all_vals[finite_mask], 2)
#     if vmax is None:
#         vmax = np.nanpercentile(all_vals[finite_mask], 98)

#     def plot_single(vals, title_suffix=None):
#         mask = np.isfinite(vals)
#         fig, ax = plt.subplots(figsize=(10, 5))
#         sc = ax.scatter(
#             lons[mask],
#             lats[mask],
#             c=vals[mask],
#             s=1,
#             cmap=cmap,
#             vmin=vmin,
#             vmax=vmax,
#         )
#         cbar = plt.colorbar(sc, ax=ax)
#         cbar.set_label(f"{title_prefix} [units]")

#         title = title_prefix if title_suffix is None else f"{title_prefix} – {title_suffix}"
#         ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
#         fig.tight_layout()
#         plt.show()

#     for label, vals in zip(year_labels, value_list):
#         plot_single(vals, title_suffix=label)


def plot_gridded_variable(
    prism_var,
    res,
    years,
    title_prefix,
    cmap,
    affor_var=False,
    vmin=None,
    vmax=None,
    categorical=False,
    category_labels=None,
    exclude_negative=True,
    point_size=1,
):

    
    if res == "5min":
        grid_file = r"X:\user\orhalem\image_biochar\timer\data\sa\biomass\IMAGE_grid_5min.nc"
        grid_ds = xr.open_dataset(grid_file, engine="netcdf4")
        lats = grid_ds["lat"].isel(g=slice(0, -17221)).values
        lons = grid_ds["lon"].isel(g=slice(0, -17221)).values
    elif res == "30min":
        grid_file = r"X:\user\orhalem\image_biochar\timer\data\sa\biomass\IMAGE_grid_coordinates.nc"
        grid_ds = xr.open_dataset(grid_file)
        lats = grid_ds["lat"].isel(grid=slice(0, -233)).values
        lons = grid_ds["lon"].isel(grid=slice(0, -233)).values
    else:
        raise ValueError("res must be '5min' or '30min'")

    if affor_var:
        if res != "5min":
            raise ValueError("affor_var=True only supported for res='5min'")

        value_list = []
        year_labels = []

        if years is None:
            vals = correct_affor_cell_order(prism_var.values)
            value_list.append(vals)
            year_labels.append(None)
        else:
            for year in years:
                t = prism.Q_(year, "year")
                vals = correct_affor_cell_order(prism_var.get(t).values)
                value_list.append(vals)
                year_labels.append(year)

    else:
        value_list = []
        year_labels = []

        if years is None:
            value_list.append(prism_var.values)
            year_labels.append(None)
        else:
            for year in years:
                t = prism.Q_(year, "year")
                vals = prism_var.get(t).values
                value_list.append(vals)
                year_labels.append(year)

    def plot_single(vals, title_suffix=None):
        fig, ax = plt.subplots(figsize=(10, 5))

        if categorical:
            mask = np.isfinite(vals)
            if exclude_negative:
                mask &= (vals >= 0)

            valid_vals = vals[mask]
            if valid_vals.size == 0:
                ax.set(
                    title=title_prefix if title_suffix is None else f"{title_prefix} – {title_suffix}",
                    xlabel="Longitude",
                    ylabel="Latitude",
                )
                fig.tight_layout()
                plt.show()
                return

            classes = np.unique(valid_vals.astype(int))
            n_classes = len(classes)

            if isinstance(cmap, str):
                base_cmap = plt.get_cmap(cmap, n_classes)
                colors = [base_cmap(i) for i in range(n_classes)]
                cmap_used = ListedColormap(colors)
            elif isinstance(cmap, (list, tuple, np.ndarray)):
                if len(cmap) < n_classes:
                    raise ValueError(
                        f"Not enough colors provided for categorical plot: "
                        f"{len(cmap)} colors for {n_classes} classes."
                    )
                cmap_used = ListedColormap(list(cmap)[:n_classes])
            else:
                cmap_used = ListedColormap(cmap(np.linspace(0, 1, n_classes)))

            class_to_index = {cls: i for i, cls in enumerate(classes)}
            color_indices = np.array([class_to_index[int(v)] for v in valid_vals], dtype=int)

            bounds = np.arange(-0.5, n_classes + 0.5, 1)
            norm = BoundaryNorm(bounds, cmap_used.N)

            sc = ax.scatter(
                lons[mask],
                lats[mask],
                c=color_indices,
                s=point_size,
                cmap=cmap_used,
                norm=norm,
            )

            cbar = plt.colorbar(sc, ax=ax, ticks=np.arange(n_classes))
            if category_labels is not None:
                tick_labels = [category_labels.get(int(cls), str(int(cls))) for cls in classes]
            else:
                tick_labels = [str(int(cls)) for cls in classes]
            cbar.ax.set_yticklabels(tick_labels)
            cbar.set_label(title_prefix)

        else:
            mask = np.isfinite(vals)
            valid_vals = vals[mask]

            if valid_vals.size == 0:
                ax.set(
                    title=title_prefix if title_suffix is None else f"{title_prefix} – {title_suffix}",
                    xlabel="Longitude",
                    ylabel="Latitude",
                )
                fig.tight_layout()
                plt.show()
                return

            vmin_use = vmin
            vmax_use = vmax

            if vmin_use is None:
                vmin_use = np.nanpercentile(valid_vals, 2)
            if vmax_use is None:
                vmax_use = np.nanpercentile(valid_vals, 98)

            sc = ax.scatter(
                lons[mask],
                lats[mask],
                c=vals[mask],
                s=point_size,
                cmap=cmap,
                vmin=vmin_use,
                vmax=vmax_use,
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(f"{title_prefix} [units]")

        title = title_prefix if title_suffix is None else f"{title_prefix} – {title_suffix}"
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
        fig.tight_layout()
        plt.show()

    for label, vals in zip(year_labels, value_list):
        plot_single(vals, title_suffix=label)

def plot_regional_maps(
        prism_timevariable,
        years,
        cmap,
        legend_title,
        title_prefix
    ):
        import geopandas as gpd
        import pandas as pd
        import matplotlib.pyplot as plt

        shp_path = r'X:\user\orhalem\image_biochar\timer\data\sa\biochar\ne_110m_admin_0_countries.zip'
        world = gpd.read_file(shp_path)
        csv_file = r"X:\user\orhalem\image_biochar\timer\data\sa\biochar\country_to_regions.csv"  
        country_df = pd.read_csv(csv_file, sep=",")

        country_df = country_df[['ISO alpha-3', 'IMAGE region']].rename(
            columns={'ISO alpha-3': 'iso_a3', 'IMAGE region': 'timer_region'}
        )

        iso_corrections = {
            'US1': 'USA',
            'CH1': 'CHN',
            'GB1': 'GBR',
            'NL1': 'NLD',
            'FR1': 'FRA',
            'IS1': 'ISL',
            'DN1': 'DNK',
            'KA1': 'KAZ',
            'CU1': 'CUB',
            'NZ1': 'NZL',
            'AU1': 'AUS',
            'FI1': 'FIN',
            'CYN': 'CYP',
            'SAH': 'ESH',   
            'SOL': 'SOM',   
            'KOS': 'XKX',   
            'SDS': 'SSD',   
        }

        world['iso_a3'] = world['SOV_A3'].replace(iso_corrections)
        world['iso_a3'] = world['iso_a3'].where(world['iso_a3'].isin(country_df['iso_a3']), world['iso_a3'])
        world = world.merge(country_df, on='iso_a3', how='left')

        years_to_plot = years
        deployment_by_region = {}
        for year in years_to_plot:
            t = prism.Q_(year, 'year')
            region_values = prism_timevariable.get(t).values
            timer_region_names = Region.coords[:-1] 
            deployment_by_region[year] = dict(zip(timer_region_names, region_values))

        for year in years_to_plot:
            world['deployment'] = world['timer_region'].map(deployment_by_region[year])
            
            fig, ax = plt.subplots(1, 1, figsize=(12,6))
            world.plot(column='deployment', ax=ax, legend=True,
                    cmap=cmap, missing_kwds={'color': 'lightgrey'},
                    legend_kwds={'label': f"{legend_title} in {year}"})
            
            ax.set_title(f"{title_prefix} - {year}")
            ax.axis('off')
            plt.show()

def calc_cost_curve(
        cell_in_region, 
        nr_of_regions, 
        cost, 
        pot, 
        nr_of_classes
    ):

    NC = len(cell_in_region)  
    
    costcurve = [[0.0] * nr_of_classes for _ in range(nr_of_regions)]
    cellindex = [[-1] * nr_of_classes for _ in range(nr_of_regions)]
    
    for r in range(nr_of_regions):
        regcost = []
        for i in range(NC):
            if cell_in_region[i] == r:
                if np.isfinite(cost[i]) and np.isfinite(pot[i]):  
                    regcost.append((cost[i], i)) 

        if not regcost:
            continue
            
        regcost.sort(key=lambda x: x[0], reverse=True)
        
        regpot = [pot[idx] for _, idx in regcost]
        regpotcum = np.cumsum(regpot)
        
        step = regpotcum[-1] / nr_of_classes
        
        k = 1
        costcurve[r][0] = regcost[0][0]
        cellindex[r][0] = regcost[0][1]
        
        for i in range(len(regcost)):
            while k < nr_of_classes and (k * step) <= regpotcum[i]:
                costcurve[r][k] = regcost[i][0]
                cellindex[r][k] = regcost[i][1]
                k += 1
            if k >= nr_of_classes:
                break
    
    return np.asarray(costcurve, dtype=np.float32)

def coarsen_soil_data():
    
    import rasterio
    import pym
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin

    src_path = biochar_data_dir / "SBIO1_Annual_Mean_Temperature_5_15cm.tif"

    dst_res = 5 / 60  
    dst_width, dst_height = 4320, 2160
    dst_crs = "EPSG:4326"

    dst_transform = from_origin(-180.0, 90.0, dst_res, dst_res)

    with rasterio.open(src_path) as src:
        src_nodata = src.nodata
        src_arr = src.read(1).astype(np.float32)

        dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    
    soil_temp_flat = dst.reshape(-1)
    mask_land_5m = land_mask("5m")
    soil_temp_grid = soil_temp_flat[mask_land_5m]

    pym.write_mym(np.asarray(soil_temp_grid), file="soil_temp.dat", variable_name="soil_temp")

def reproject_to_5min(path, dst_height=2160, dst_width=4320):
    
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    res = 5.0 / 60.0  # degrees
    dst_transform = from_origin(-180.0, 90.0, res, res)
    dst_crs = CRS.from_epsg(4326)

    with rasterio.open(path) as src:

        src_arr = src.read(1).astype(np.float32)

        src_nodata = src.nodata
        if src_nodata is not None:
            src_arr[src_arr == src_nodata] = np.nan

        dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.average,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    return dst

def calculate_weighted_soil_mean(_05_layer, _515_layer, _1530_layer):
    
    # Compute thickness-weighted mean for 0–20 cm
    w0, w1, w2 = 5.0/20.0, 10.0/20.0, 5.0/20.0

    # Weighted mean with NaN-safe normalization
    weights_sum = (
        w0 * np.isfinite(_05_layer).astype(np.float32) +
        w1 * np.isfinite(_515_layer).astype(np.float32) +
        w2 * np.isfinite(_1530_layer).astype(np.float32)
    )

    soil_var_0_20 = (
        w0 * np.nan_to_num(_05_layer, nan=0.0) +
        w1 * np.nan_to_num(_515_layer, nan=0.0) +
        w2 * np.nan_to_num(_1530_layer, nan=0.0)
    )
    
    weights_mask = weights_sum > 0.0
    soil_var_0_20[weights_mask] = soil_var_0_20[weights_mask] / weights_sum[weights_mask]
    soil_var_0_20[~weights_mask] = np.nan
    
    mask_land_5m = land_mask("5m")
    soil_var_0_20_grid = soil_var_0_20.reshape(-1)
    soil_var_0_20_grid = soil_var_0_20_grid[mask_land_5m]
    
    return prism.array(soil_var_0_20_grid, dims=[LandGrid])

def reproject_n2o_emis_data():

    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    from rasterio.transform import from_origin, rowcol

    n2o_emis = pd.read_csv(
        biochar_data_dir / "S1_Cropland_N2O_1960_2020.out",
        sep=r"\s+"
    )
    n2o_emis_2020 = n2o_emis[n2o_emis["Year"] == 2020]
    
    IMAGE_grid = xr.open_dataset(biocrops_data_dir / "IMAGE_grid_5min.nc", engine="netcdf4")

    src = n2o_emis_2020[["Lon", "Lat", "N2O"]].copy()
    src = src.dropna(subset=["Lon", "Lat", "N2O"])

    lon_vals = np.sort(src["Lon"].unique())
    lat_vals = np.sort(src["Lat"].unique())

    dlon = np.unique(np.round(np.diff(lon_vals), 6))
    dlat = np.unique(np.round(np.diff(lat_vals), 6))
    print("Unique source dlon:", dlon[:10])
    print("Unique source dlat:", dlat[:10])

    src_grid = (
        src.pivot(index="Lat", columns="Lon", values="N2O")
           .reindex(index=np.sort(lat_vals)[::-1], columns=np.sort(lon_vals))
    )

    src_arr = src_grid.to_numpy(dtype=np.float32)

    src_res_x = float(np.median(np.diff(lon_vals)))
    src_res_y = float(np.median(np.diff(lat_vals)))

    src_transform = from_origin(
        west=lon_vals.min() - src_res_x / 2,
        north=lat_vals.max() + src_res_y / 2,
        xsize=src_res_x,
        ysize=src_res_y
    )

    src_crs = CRS.from_epsg(4326)

    dst_res = 5.0 / 60.0   # 5 arc-minutes
    dst_width = 4320
    dst_height = 2160
    dst_transform = from_origin(-180.0, 90.0, dst_res, dst_res)
    dst_crs = CRS.from_epsg(4326)

    dst_5min = np.full((dst_height, dst_width), np.nan, dtype=np.float32)

    resampling_map = {
        "average": Resampling.average,
        "nearest": Resampling.nearest,
    }

    reproject(
        source=src_arr,
        destination=dst_5min,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=resampling_map["average"],
    )

    target_lon = IMAGE_grid["lon"].values
    target_lat = IMAGE_grid["lat"].values
    target_g = IMAGE_grid["g"].values

    rows, cols = rowcol(dst_transform, target_lon, target_lat)

    rows = np.asarray(rows)
    cols = np.asarray(cols)

    valid = (
        (rows >= 0) & (rows < dst_height) &
        (cols >= 0) & (cols < dst_width)
    )

    n2o_vals = np.full(target_g.shape, np.nan, dtype=np.float32)
    n2o_vals[valid] = dst_5min[rows[valid], cols[valid]]

    out_ds = xr.Dataset(
        data_vars={
            "N2O": ("g", n2o_vals)
        },
        coords={
            "g": ("g", target_g),
            "lon": ("g", target_lon),
            "lat": ("g", target_lat),
        }
    )

    n2o_emis_2020 = out_ds.isel(g=slice(0, -17221))
    
    pym.write_mym(n2o_emis_2020["N2O"].to_numpy(), filename="n2o_emis_2020.dat", variable_name="n2o_emis_2020")

def calculate_soil_soc():
    # Read in and format SOC data
    soc_0_5_path   = biochar_data_dir / "soc_0-5cm_mean_5000.tif"
    soc_5_15_path  = biochar_data_dir / "soc_5-15cm_mean_5000.tif"
    soc_15_30_path = biochar_data_dir / "soc_15-30cm_mean_5000.tif"

    soc_0_5   = reproject_to_5min(soc_0_5_path)
    soc_5_15  = reproject_to_5min(soc_5_15_path)
    soc_15_30 = reproject_to_5min(soc_15_30_path)

    # Compute depth-weighted mean for 0–20 cm
    soil_organic_carbon = calculate_weighted_soil_mean(soc_0_5, soc_5_15, soc_15_30).values
    soil_organic_carbon *= 0.1 # from dg/kg to g/kg

    # Prevent extrapolation beyond training range
    soc_low = 3.03
    soc_high = 34.83
    soil_organic_carbon = np.clip(soil_organic_carbon, soc_low, soc_high)

    pym.write_mym(np.asarray(soil_organic_carbon), filename="soil_soc.dat", variable_name="soil_soc")

def calculate_soil_n():
    # Read in and format soil nitrogen data
    soiln_0_5_path   = biochar_data_dir / "nitrogen_0-5cm_mean_5000.tif"
    soiln_5_15_path  = biochar_data_dir / "nitrogen_5-15cm_mean_5000.tif"
    soiln_15_30_path = biochar_data_dir / "nitrogen_15-30cm_mean_5000.tif"

    soiln_0_5   = reproject_to_5min(soiln_0_5_path)
    soiln_5_15  = reproject_to_5min(soiln_5_15_path)
    soiln_15_30 = reproject_to_5min(soiln_15_30_path)

    # Compute depth-weighted mean for 0–20 cm
    soil_nitrogen = calculate_weighted_soil_mean(soiln_0_5, soiln_5_15, soiln_15_30).values
    soil_nitrogen *= 0.01 # from cg/kg to g/kg

    # Prevent extrapolation beyond training range
    soiln_low = 0.39
    soiln_high = 2.70
    soil_nitrogen = np.clip(soil_nitrogen, soiln_low, soiln_high)

    pym.write_mym(np.asarray(soil_nitrogen), filename="soil_n.dat", variable_name="soil_n")

def calculate_soil_ph():
    # Read in and format soil pH data
    soilph_0_5_path   = biochar_data_dir / "phh2o_0-5cm_mean_5000.tif"
    soilph_5_15_path  = biochar_data_dir / "phh2o_5-15cm_mean_5000.tif"
    soilph_15_30_path = biochar_data_dir / "phh2o_15-30cm_mean_5000.tif"

    soilph_0_5   = reproject_to_5min(soilph_0_5_path)
    soilph_5_15  = reproject_to_5min(soilph_5_15_path)
    soilph_15_30 = reproject_to_5min(soilph_15_30_path)

    # Compute depth-weighted mean for 0–20 cm
    soil_ph = calculate_weighted_soil_mean(soilph_0_5, soilph_5_15, soilph_15_30).values
    soil_ph *= 0.1 # correct *10 scaling

    pym.write_mym(np.asarray(soil_ph), filename="soil_ph.dat", variable_name="soil_ph")

def estimate_n_fertiliser_use():
    # Read in and format N fertiliser rate data
    fiber_n_path   = biochar_data_dir / "Fiber_crops_N_2019.tiff"
    fruits_n_path   = biochar_data_dir / "Fruits_N_2019.tiff"
    maize_n_path   = biochar_data_dir / "Maize_N_2019.tiff"
    cereals_n_path   = biochar_data_dir / "Other_Cereals_N_2019.tiff"
    crops_n_path   = biochar_data_dir / "Other_Crops_N_2019.tiff"
    oilseeds_n_path   = biochar_data_dir / "Other_Oilseeds_N_2019.tiff"
    palmoil_n_path   = biochar_data_dir / "Palm_Oil_fruit_N_2019.tiff"
    rice_n_path   = biochar_data_dir / "Rice_N_2019.tiff"
    roots_tubers_n_path   = biochar_data_dir / "Roots_and_tubers_N_2019.tiff"
    soybean_n_path   = biochar_data_dir / "Soybean_N_2019.tiff"
    sugarcrops_n_path   = biochar_data_dir / "Sugar_crops_N_2019.tiff"
    vegetables_n_path   = biochar_data_dir / "Vegetables_N_2019.tiff"
    wheat_n_path   = biochar_data_dir / "Wheat_N_2019.tiff"

    fiber_n = reproject_to_5min(fiber_n_path)
    fruits_n = reproject_to_5min(fruits_n_path)
    maize_n = reproject_to_5min(maize_n_path)
    cereals_n = reproject_to_5min(cereals_n_path)
    crops_n = reproject_to_5min(crops_n_path)
    oilseeds_n = reproject_to_5min(oilseeds_n_path)
    palmoil_n = reproject_to_5min(palmoil_n_path)
    rice_n = reproject_to_5min(rice_n_path)
    roots_n = reproject_to_5min(roots_tubers_n_path)
    soybean_n = reproject_to_5min(soybean_n_path)
    sugarcrops_n = reproject_to_5min(sugarcrops_n_path)
    vegetables_n = reproject_to_5min(vegetables_n_path)
    wheat_n = reproject_to_5min(wheat_n_path)

    n_fertiliser = (
        fiber_n + fruits_n + maize_n + cereals_n +
        crops_n + oilseeds_n + palmoil_n + rice_n +
        roots_n + soybean_n + sugarcrops_n +
        vegetables_n + wheat_n
    )

    n_fert_flat = n_fertiliser.reshape(-1)

    mask_land_5m = land_mask("5m")
    n_fertiliser = n_fert_flat[mask_land_5m]

    pym.write_mym(np.asarray(n_fertiliser), filename="n_fertiliser.dat", variable_name="n_fertiliser")

def calculate_soil_clay():
    # Read in and format soil clay data
    soilclay_0_5_path   = biochar_data_dir / "clay_0-5cm_mean_5000.tif"
    soilclay_5_15_path  = biochar_data_dir / "clay_5-15cm_mean_5000.tif"
    soilclay_15_30_path = biochar_data_dir / "clay_15-30cm_mean_5000.tif"

    soilclay_0_5   = reproject_to_5min(soilclay_0_5_path)
    soilclay_5_15  = reproject_to_5min(soilclay_5_15_path)
    soilclay_15_30 = reproject_to_5min(soilclay_15_30_path)

    # Compute depth-weighted mean for 0–20 cm
    soil_clay = calculate_weighted_soil_mean(soilclay_0_5, soilclay_5_15, soilclay_15_30).values
    soil_clay /= 10 # from permille to percent
    
    pym.write_mym(np.asarray(soil_clay), filename="soil_clay.dat", variable_name="soil_clay")

def export_variable_to_netcdf(
    prism_var,
    filename,
    var_name,
    dims,
    coords,
    times=None,
    units=None,
    description=None,
    output_dir=None,
):
    
    output_dir = Path(
        "." if output_dir is None else output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if times is None:
        # Non-time variable
        values = prism_var.values
        da_dims = dims
        da_coords = coords

    else:
        # Time variable
        values = np.stack(
            [prism_var[prism.Q_(t, "year")].values for t in times],
            axis=0
        )
        da_dims = ("time", *dims)
        da_coords = {
            "time": np.asarray(times, dtype=np.int32),
            **coords,
        }

    da = xr.DataArray(
        values,
        dims=da_dims,
        coords=da_coords,
        attrs={
            "units": units or "",
            "description": description or "",
        },
    )

    ds = xr.Dataset({var_name: da})
    ds.to_netcdf(output_dir / filename)


def _find_dim(da, candidates, required=True):
    for cand in candidates:
        if cand in da.dims:
            return cand

    if required:
        raise ValueError(
            f"Could not find dimension among {candidates}. "
            f"Available dims: {da.dims}"
        )

    return None

def load_grppc_30min_as_5min_landgrid(
    grppc_nc,
    year,
    Grid,
    LandGrid,
    Crop38,
    var_name="GRPPC_30min",
    crop_dim_candidates=("NGFBFC",),
    lat_dim_candidates=("latitude", "lat",),
    lon_dim_candidates=("longitude", "lon",),
    time_dim_candidates=("time", "Time", "year", "Year"),
    time_step_func=time_step,
):

    grppc_nc = Path(grppc_nc)

    with xr.open_dataset(
        grppc_nc,
        engine="netcdf4",
        chunks={"time": 1},
    ) as ds:
        if var_name not in ds.data_vars:
            raise KeyError(
                f"{var_name} not found in {grppc_nc}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[var_name]

        time_dim = _find_dim(
            da,
            time_dim_candidates,
            required=False,
        )

        crop_dim = _find_dim(
            da,
            crop_dim_candidates,
            required=True,
        )

        lat_dim = _find_dim(
            da,
            lat_dim_candidates,
            required=True,
        )

        lon_dim = _find_dim(
            da,
            lon_dim_candidates,
            required=True,
        )

        # --------------------------------------------------------
        # Select year
        # --------------------------------------------------------
        if time_dim is not None:
            if time_step_func is not None:
                da_y = da.isel({time_dim: int(time_step_func(year))})
            else:
                time_vals = np.asarray(da[time_dim].values)

                # Try direct coordinate selection first.
                if int(year) in time_vals.astype(int):
                    da_y = da.sel({time_dim: int(year)})
                else:
                    raise ValueError(
                        f"Year {year} not found in {time_dim}. "
                        f"Available values: {time_vals}"
                    )
        else:
            da_y = da

        # Standardize to latitude x longitude x crop.
        da_y = da_y.transpose(lat_dim, lon_dim, crop_dim)

        arr_30 = np.asarray(da_y.values, dtype=np.float32)

    if arr_30.ndim != 3:
        raise ValueError(
            f"Expected GRPPC after year selection to be 3D "
            f"(lat, lon, crop). Got shape {arr_30.shape}."
        )

    nlat_30, nlon_30, ncrop = arr_30.shape

    if ncrop != len(Crop38.coords):
        raise ValueError(
            f"Expected {len(Crop38.coords)} crop classes, got {ncrop}."
        )

    if (nlat_30, nlon_30) != (360, 720):
        raise ValueError(
            f"Expected 30-min grid shape 360 x 720. "
            f"Got {nlat_30} x {nlon_30}."
        )

    # ------------------------------------------------------------
    # 30-min -> 5-min nearest-neighbour replication
    # ------------------------------------------------------------
    arr_5m = np.repeat(
        np.repeat(arr_30, 6, axis=0),
        6,
        axis=1,
    )

    nlat_5m, nlon_5m, _ = arr_5m.shape

    if (nlat_5m, nlon_5m) != (2160, 4320):
        raise ValueError(
            f"Expected 5-min grid shape 2160 x 4320 after upscaling. "
            f"Got {nlat_5m} x {nlon_5m}."
        )

    full_grid_size = len(Grid.coords)

    if full_grid_size != nlat_5m * nlon_5m:
        raise ValueError(
            f"Grid.coords length does not match 5-min full grid. "
            f"len(Grid.coords)={full_grid_size}, "
            f"5-min grid size={nlat_5m * nlon_5m}."
        )

    mask_land_5m = land_mask('5m')
    mask_land_5m = np.asarray(mask_land_5m, dtype=bool)

    # Flatten full 5-min grid to full Grid x Crop38.
    arr_5m_flat = arr_5m.reshape(full_grid_size, ncrop)

    # Select IMAGE land cells
    arr_land = arr_5m_flat[mask_land_5m, :]

    if arr_land.shape != (len(LandGrid.coords), len(Crop38.coords)):
        raise ValueError(
            f"Expected land-grid GRPPC shape "
            f"{(len(LandGrid.coords), len(Crop38.coords))}. "
            f"Got {arr_land.shape}."
        )

    return arr_land

COUNTRY_NAMES = np.array([
    "Canada",
    "Saint Pierre and Miquelon",
    "United States",
    "United States Minor Outlying Islands",
    "Mexico",
    "Anguilla",
    "Antigua and Barbuda",
    "Aruba",
    "Bahamas",
    "Barbados",
    "Belize",
    "Bermuda",
    "Cayman Islands",
    "Costa Rica",
    "Cuba",
    "Dominica",
    "Dominican Republic",
    "El Salvador",
    "Grenada",
    "Guadeloupe",
    "Guatemala",
    "Haiti",
    "Honduras",
    "Jamaica",
    "Martinique",
    "Montserrat",
    "Netherlands Antilles",
    "Nicaragua",
    "Panama",
    "Puerto Rico",
    "Saint Kitts and Nevis",
    "Saint Lucia",
    "Saint Vincent and the Grenadines",
    "Trinidad and Tobago",
    "Turks and Caicos Islands",
    "Virgin Islands British",
    "Virgin Islands U.S.",
    "Brazil",
    "Argentina",
    "Bolivia",
    "Bouvet Island",
    "Chile",
    "Colombia",
    "Ecuador",
    "Falklands Islands (Malvinas)",
    "French Guiana",
    "Guyana",
    "Paraguay",
    "Peru",
    "South Georgia and the South Sandwich Islands",
    "Suriname",
    "Uruguay",
    "Venezuela",
    "Algeria",
    "Egypt",
    "Libyan Arab Jamahiriya",
    "Morocco",
    "Tunisia",
    "Western Sahara",
    "Benin",
    "Burkina Faso",
    "Cameroon",
    "Cape Verde",
    "Central African Republic",
    "Chad",
    "Congo",
    "Congo (DRC)",
    "Côte d'Ivoire",
    "Equatorial Guinea",
    "Gabon",
    "Gambia",
    "Ghana",
    "Guinea",
    "Guinea-Bissau",
    "Liberia",
    "Mali",
    "Mauritania",
    "Niger",
    "Nigeria",
    "Saint Helena",
    "Sao Tome and Principe",
    "Senegal",
    "Sierra Leone",
    "Togo",
    "Burundi",
    "Comoros",
    "Djibouti",
    "Eritrea",
    "Ethiopia",
    "Kenya",
    "Madagascar",
    "Mauritius",
    "Mayotte",
    "Réunion",
    "Rwanda",
    "Seychelles",
    "Somalia",
    "Sudan",
    "Uganda",
    "Angola",
    "Botswana",
    "Lesotho",
    "Malawi",
    "Mozambique",
    "Namibia",
    "South Africa",
    "Swaziland",
    "Tanzania",
    "Zambia",
    "Zimbabwe",
    "Andorra",
    "Austria",
    "Belgium",
    "Denmark",
    "Faroe Islands",
    "Finland",
    "France",
    "Germany",
    "Gibraltar",
    "Greece",
    "Vatican City State",
    "Iceland",
    "Ireland",
    "Italy",
    "Liechtenstein",
    "Luxembourg",
    "Malta",
    "Monaco",
    "Netherlands",
    "Norway",
    "Portugal",
    "San Marino",
    "Spain",
    "Svalbard and Jan Mayen",
    "Sweden",
    "Switzerland",
    "United Kingdom",
    "Albania",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Hungary",
    "Macedonia",
    "Poland",
    "Romania",
    "Slovakia",
    "Slovenia",
    "Yugoslavia",
    "Estonia",
    "Latvia",
    "Lithuania",
    "Turkey",
    "Belarus",
    "Moldova",
    "Ukraine",
    "Kazakstan",
    "Kyrgyzstan",
    "Tajikistan",
    "Turkmenistan",
    "Uzbekistan",
    "Armenia",
    "Azerbaijan",
    "Georgia",
    "Russia",
    "Bahrain",
    "Iran",
    "Iraq",
    "Israel",
    "Jordan",
    "Kuwait",
    "Lebanon",
    "Oman",
    "Qatar",
    "Saudi Arabia",
    "Syria",
    "United Arab Emirates",
    "Yemen",
    "Afghanistan",
    "Bangladesh",
    "Bhutan",
    "British Indian Ocean Territory",
    "India",
    "Maldives",
    "Nepal",
    "Pakistan",
    "Sri Lanka",
    "North Korea",
    "South Korea",
    "China",
    "Hong Kong",
    "Macau",
    "Mongolia",
    "Taiwan",
    "Brunei Darussalam",
    "Cambodia",
    "East Timor",
    "Lao People's Democratic Republic",
    "Malaysia",
    "Myanmar",
    "Philippines",
    "Singapore",
    "Thailand",
    "Vietnam",
    "Indonesia",
    "Papua New Guinea",
    "Japan",
    "American Samoa",
    "Australia",
    "Christmas Island",
    "Cocos (Keeling) Islands",
    "Cook Islands",
    "Fiji",
    "French Polynesia",
    "French Southern Territories",
    "Guam",
    "Heard Island and McDonald Islands",
    "Kiribati",
    "Marshall Islands",
    "Micronesia Federated States of",
    "Nauru",
    "New Caledonia",
    "New Zealand",
    "Niue",
    "Norfolk Island",
    "Northern Mariana Islands",
    "Palau",
    "Pitcairn",
    "Samoa",
    "Solomon Islands",
    "Tokelau",
    "Tonga",
    "Tuvalu",
    "Vanuatu",
    "Wallis and Futuna",
], dtype=object)

def country_name_from_index(country_index, missing_value=""):

    if country_index is None:
        return missing_value

    if not np.isfinite(country_index):
        return missing_value

    country_index = int(country_index)

    if 1 <= country_index <= len(COUNTRY_NAMES):
        return COUNTRY_NAMES[country_index - 1]

    return missing_value

def country_indices_to_names(country_index_array, missing_value=""):

    arr = np.asarray(country_index_array)

    out = np.full(
        arr.shape,
        missing_value,
        dtype=object,
    )

    valid = (
        np.isfinite(arr)
        & (arr >= 1)
        & (arr <= len(COUNTRY_NAMES))
    )

    out[valid] = COUNTRY_NAMES[arr[valid].astype(int) - 1]

    return out


def _fao_5min_coordinates(header, shape):
    """Return FAO ASCII-grid cell-centre lat/lon coordinates."""
    nrows, ncols = shape
    cellsize = float(header["cellsize"])

    if "xllcorner" in header:
        lon = (
            float(header["xllcorner"])
            + (np.arange(ncols, dtype=np.float64) + 0.5) * cellsize
        )
    elif "xllcenter" in header:
        lon = (
            float(header["xllcenter"])
            + np.arange(ncols, dtype=np.float64) * cellsize
        )
    else:
        lon = (
            -180.0
            + (np.arange(ncols, dtype=np.float64) + 0.5) * cellsize
        )

    if "yllcorner" in header:
        lat_south_to_north = (
            float(header["yllcorner"])
            + (np.arange(nrows, dtype=np.float64) + 0.5) * cellsize
        )
    elif "yllcenter" in header:
        lat_south_to_north = (
            float(header["yllcenter"])
            + np.arange(nrows, dtype=np.float64) * cellsize
        )
    else:
        lat_south_to_north = (
            -90.0
            + (np.arange(nrows, dtype=np.float64) + 0.5) * cellsize
        )

    # ESRI ASCII rows are stored north -> south.
    lat = lat_south_to_north[::-1]

    return lat, lon


def _read_gmted_slope15_netcdf(path):
    """
    Read the previously generated GMTED2010 ds75 5' supplement.

    Expected variables:
        Slope15Frac
        lat
        lon

    Optional:
        ValidFineFrac
    """
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError(
            "Reading the GMTED NetCDF requires netCDF4.\n"
            "Install with:\n"
            "    pip install netCDF4"
        ) from exc

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"GMTED slope NetCDF does not exist:\n{path}"
        )

    with Dataset(path, "r") as ds:
        required = {"Slope15Frac", "lat", "lon"}
        missing = required - set(ds.variables)

        if missing:
            raise KeyError(
                f"GMTED NetCDF is missing variables: {sorted(missing)}"
            )

        lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)

        var = ds.variables["Slope15Frac"]
        dims = tuple(var.dimensions)
        raw = var[:]

        if np.ma.isMaskedArray(raw):
            slope = np.ma.filled(raw, np.nan).astype(
                np.float32,
                copy=False,
            )
        else:
            slope = np.asarray(raw, dtype=np.float32)

        if dims == ("lat", "lon"):
            pass
        elif dims == ("lon", "lat"):
            slope = slope.T
        else:
            raise ValueError(
                "Unexpected Slope15Frac dimensions in GMTED NetCDF:\n"
                f"  {dims}\n"
                "Expected ('lat', 'lon') or ('lon', 'lat')."
            )

        valid_fine_frac = None

        if "ValidFineFrac" in ds.variables:
            v = ds.variables["ValidFineFrac"]
            vdims = tuple(v.dimensions)
            vraw = v[:]

            if np.ma.isMaskedArray(vraw):
                valid_fine_frac = np.ma.filled(
                    vraw,
                    np.nan,
                ).astype(
                    np.float32,
                    copy=False,
                )
            else:
                valid_fine_frac = np.asarray(
                    vraw,
                    dtype=np.float32,
                )

            if vdims == ("lat", "lon"):
                pass
            elif vdims == ("lon", "lat"):
                valid_fine_frac = valid_fine_frac.T
            else:
                raise ValueError(
                    "Unexpected ValidFineFrac dimensions in GMTED NetCDF:\n"
                    f"  {vdims}"
                )

    # Normalize orientation to north -> south, west -> east.
    if lat.size > 1 and lat[1] > lat[0]:
        lat = lat[::-1]
        slope = slope[::-1, :]
        if valid_fine_frac is not None:
            valid_fine_frac = valid_fine_frac[::-1, :]

    if lon.size > 1 and lon[1] < lon[0]:
        lon = lon[::-1]
        slope = slope[:, ::-1]
        if valid_fine_frac is not None:
            valid_fine_frac = valid_fine_frac[:, ::-1]

    return slope, lat, lon, valid_fine_frac


def _slope_validation_stats(fao, gmted):
    """Calculate direct 5' FAO-vs-GMTED comparison statistics."""
    fao = np.asarray(fao, dtype=np.float64)
    gmted = np.asarray(gmted, dtype=np.float64)

    if fao.size != gmted.size:
        raise ValueError("Validation arrays have different sizes.")

    if fao.size == 0:
        return None

    diff = gmted - fao

    if (
        fao.size >= 2
        and np.std(fao) > 0.0
        and np.std(gmted) > 0.0
    ):
        correlation = float(np.corrcoef(fao, gmted)[0, 1])
    else:
        correlation = np.nan

    levels = [5, 25, 50, 75, 95]

    return {
        "n": int(fao.size),
        "mean_fao": float(np.mean(fao)),
        "mean_gmted": float(np.mean(gmted)),
        "bias_gmted_minus_fao": float(np.mean(diff)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff ** 2))),
        "correlation": correlation,
        "fao_percentiles": np.percentile(fao, levels),
        "gmted_percentiles": np.percentile(gmted, levels),
        "percentile_levels": levels,
    }


def _print_slope_validation(title, stats):
    print("\n" + title)

    if stats is None:
        print("  No cells available.")
        return

    print(f"  N cells                  : {stats['n']:,}")
    print(f"  Mean FAO                 : {stats['mean_fao']:.6f}")
    print(f"  Mean GMTED               : {stats['mean_gmted']:.6f}")
    print(
        "  Mean bias (GMTED - FAO) : "
        f"{stats['bias_gmted_minus_fao']:+.6f}"
    )
    print(f"  MAE                      : {stats['mae']:.6f}")
    print(f"  RMSE                     : {stats['rmse']:.6f}")
    print(f"  Pearson correlation      : {stats['correlation']:.6f}")
    print(f"  Percentile levels        : {stats['percentile_levels']}")
    print(f"  FAO percentiles          : {stats['fao_percentiles']}")
    print(f"  GMTED percentiles        : {stats['gmted_percentiles']}")


def _export_landgrid_netcdf(
    values,
    coords,
    output_path,
    var_name="SlopeGT15Frac",
    description=None,
):
    """Export the final 1-D IMAGE LandGrid slope array."""
    try:
        from netCDF4 import Dataset
    except ImportError as exc:
        raise ImportError(
            "Exporting the final NetCDF requires netCDF4.\n"
            "Install with:\n"
            "    pip install netCDF4"
        ) from exc

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    coords = np.asarray(coords).reshape(-1)

    if coords.size != values.size:
        raise ValueError(
            "image_land_coords and final slope array have different sizes:\n"
            f"  coords: {coords.size:,}\n"
            f"  slope : {values.size:,}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Dataset(output_path, "w", format="NETCDF4") as ds:
        ds.createDimension("grid", values.size)

        if np.issubdtype(coords.dtype, np.integer):
            coord_var = ds.createVariable("grid", "i8", ("grid",))
            coord_var[:] = coords.astype(np.int64, copy=False)
        elif np.issubdtype(coords.dtype, np.floating):
            coord_var = ds.createVariable("grid", "f8", ("grid",))
            coord_var[:] = coords.astype(np.float64, copy=False)
        else:
            coord_var = ds.createVariable("grid", str, ("grid",))
            coord_var[:] = coords.astype(str)

        coord_var.long_name = "IMAGE/TIMER LandGrid coordinate"

        slope_var = ds.createVariable(
            var_name,
            "f4",
            ("grid",),
            zlib=True,
            complevel=4,
            fill_value=np.float32(-9999.0),
        )

        output_values = values.copy()
        unresolved = ~np.isfinite(output_values)
        output_values[unresolved] = np.float32(-9999.0)
        slope_var[:] = output_values

        slope_var.units = "-"
        slope_var.long_name = (
            "fraction of land with terrain slope "
            "greater than or equal to 15 percent"
        )
        slope_var.source = (
            "FAO HWSD terrain slope classes C6+C7+C8 where available; "
            "GMTED2010 ds75-derived 5 arc-minute slope fraction only "
            "where FAO is missing."
        )

        if description is not None:
            slope_var.description = str(description)

        ds.title = (
            "IMAGE/TIMER SlopeGT15Frac using FAO plus GMTED2010 gap filling"
        )
        ds.history = (
            "FAO retained as primary source. GMTED2010 used only for "
            "missing FAO cells within the configured fill latitude band."
        )

    return output_path


def load_fao_slope_gt15(
    rar_dir,
    gmted_nc_path=None,
    image_land_mask=None,
    image_land_coords=None,
    output_nc_path=None,
    output_var_name="SlopeGT15Frac",
    save_path=None,
    count_tolerance=0.5,
    raise_on_incomplete=False,
    validation_lat_min=55.0,
    validation_lat_max=60.0,
    validation_high_coverage_frac=0.90,
    gmted_fill_lat_min=55.0,
    gmted_fill_lat_max=84.0,
    gmted_min_valid_fine_frac=0.0,
    return_diagnostics=False,
    verbose=True,
):

    try:
        import rarfile

    except ImportError as exc:
        raise ImportError(
            "This function requires the Python package 'rarfile'.\n"
            "Install with:\n"
            "    pip install rarfile\n\n"
            "You may additionally need 7-Zip, unrar, or bsdtar."
        ) from exc


    rar_dir = Path(rar_dir)

    if not rar_dir.exists():
        raise FileNotFoundError(
            f"Directory does not exist:\n{rar_dir}"
        )

    def _read_ascii_from_rar(rar_path):

        if not rar_path.exists():
            raise FileNotFoundError(
                f"Missing required FAO archive:\n{rar_path}"
            )

        try:
            rf = rarfile.RarFile(rar_path)

        except Exception as exc:
            raise RuntimeError(
                f"Could not open RAR archive:\n{rar_path}"
            ) from exc


        with rf:

            members = [
                member
                for member in rf.infolist()
                if not member.isdir()
            ]

            if not members:
                raise RuntimeError(
                    f"No files in archive:\n{rar_path}"
                )


            candidates = [
                member
                for member in members
                if member.filename.lower().endswith(
                    (".asc", ".txt")
                )
            ]

            if candidates:
                member = max(
                    candidates,
                    key=lambda x: x.file_size,
                )
            else:
                member = max(
                    members,
                    key=lambda x: x.file_size,
                )


            if verbose:
                print(
                    f"  Archive member: {member.filename}"
                )


            try:
                binary_stream = rf.open(member)

            except rarfile.RarCannotExec as exc:
                raise RuntimeError(
                    "rarfile cannot find a RAR extraction backend.\n"
                    "Install 7-Zip, unrar or bsdtar and ensure that "
                    "it is available on PATH."
                ) from exc


            with binary_stream:

                text_stream = io.TextIOWrapper(
                    binary_stream,
                    encoding="ascii",
                    errors="strict",
                    newline=None,
                )

                header = {}

                for _ in range(6):

                    line = text_stream.readline()

                    if not line:
                        raise RuntimeError(
                            f"Incomplete header in:\n{rar_path}"
                        )

                    parts = line.strip().split()

                    if len(parts) < 2:
                        raise RuntimeError(
                            f"Invalid header line:\n{line}"
                        )

                    header[
                        parts[0].lower()
                    ] = float(parts[1])


                required = {
                    "ncols",
                    "nrows",
                    "cellsize",
                    "nodata_value",
                }

                missing = (
                    required
                    - set(header)
                )

                if missing:
                    raise RuntimeError(
                        f"Missing header fields {missing} "
                        f"in:\n{rar_path}"
                    )


                nrows = int(
                    header["nrows"]
                )

                ncols = int(
                    header["ncols"]
                )


                data = np.loadtxt(
                    text_stream,
                    dtype=np.float32,
                )


        expected_size = (
            nrows * ncols
        )

        if data.size != expected_size:
            raise RuntimeError(
                f"Unexpected raster size in {rar_path.name}:\n"
                f"  expected = {expected_size:,}\n"
                f"  found    = {data.size:,}"
            )


        data = data.reshape(
            nrows,
            ncols,
        )

        return data, header

    if verbose:

        print(
            "\n"
            + "=" * 72
        )

        print(
            "READING FAO GloLand_5min"
        )

        print(
            "=" * 72
        )


    land_path = (
        rar_dir
        / "GloLand_5min.rar"
    )


    glo_land_raw, land_header = (
        _read_ascii_from_rar(
            land_path
        )
    )


    land_nodata = (
        land_header["nodata_value"]
    )


    valid_glo_land = (
        np.isfinite(glo_land_raw)
        & ~np.isclose(
            glo_land_raw,
            land_nodata,
        )
    )

    fao_land = (
        valid_glo_land
        & (glo_land_raw > 0.0)
    )


    shape = glo_land_raw.shape

    glo_land_count = np.zeros(
        shape,
        dtype=np.float64,
    )

    glo_land_count[
        fao_land
    ] = glo_land_raw[
        fao_land
    ]


    if verbose:

        land_values = glo_land_count[
            fao_land
        ]

        print(
            f"\nGrid shape             : {shape}"
        )

        print(
            f"Total global cells     : {fao_land.size:,}"
        )

        print(
            f"FAO cells with land    : {np.sum(fao_land):,}"
        )

        print(
            f"Cells without FAO land : {np.sum(~fao_land):,}"
        )

        print(
            f"Cell size              : "
            f"{land_header['cellsize']} degrees"
        )

        print(
            "\nGloLand fine-pixel count:"
        )

        print(
            "  [min, p1, p5, p25, p50, "
            "p75, p95, p99, max] ="
        )

        print(
            np.percentile(
                land_values,
                [
                    0,
                    1,
                    5,
                    25,
                    50,
                    75,
                    95,
                    99,
                    100,
                ],
            )
        )

    class_sum = np.zeros(
        shape,
        dtype=np.float64,
    )

    slope_gt15_count = np.zeros(
        shape,
        dtype=np.float64,
    )

    explicit_class_count = np.zeros(
        shape,
        dtype=np.uint8,
    )

    nodata_class_count = np.zeros(
        shape,
        dtype=np.uint8,
    )

    nodata_counts_by_class = {}

    class_stats = {}

    for slope_class in range(1, 9):

        rar_path = (
            rar_dir
            / f"GloSlopesCl{slope_class}_5min.rar"
        )


        if verbose:
            print(
                f"\nReading slope class C{slope_class}: "
                f"{rar_path.name}"
            )


        raw, header = (
            _read_ascii_from_rar(
                rar_path
            )
        )

        if raw.shape != shape:
            raise RuntimeError(
                f"Grid shape mismatch for C{slope_class}:\n"
                f"  GloLand: {shape}\n"
                f"  C{slope_class}: {raw.shape}"
            )


        for key in (
            "nrows",
            "ncols",
            "cellsize",
        ):

            if not np.isclose(
                header[key],
                land_header[key],
            ):
                raise RuntimeError(
                    f"Header mismatch for C{slope_class}: {key}"
                )


        for key in (
            "xllcorner",
            "xllcenter",
            "yllcorner",
            "yllcenter",
        ):

            if (
                key in header
                and key in land_header
                and not np.isclose(
                    header[key],
                    land_header[key],
                )
            ):
                raise RuntimeError(
                    f"Coordinate mismatch for "
                    f"C{slope_class}: {key}"
                )

        nodata = (
            header["nodata_value"]
        )


        is_nodata = (
            ~np.isfinite(raw)
            | np.isclose(
                raw,
                nodata,
            )
        )


        valid_on_land = (
            fao_land
            & ~is_nodata
        )


        nodata_on_land = (
            fao_land
            & is_nodata
        )


        n_nodata_on_land = int(
            np.sum(
                nodata_on_land
            )
        )


        nodata_counts_by_class[
            slope_class
        ] = n_nodata_on_land


        if verbose:

            print(
                f"  NODATA value: {nodata}"
            )

            print(
                f"  NODATA on FAO-land cells: "
                f"{n_nodata_on_land:,}"
            )

        impossible_negative = (
            valid_on_land
            & (raw < 0.0)
        )


        impossible_global_max = (
            valid_on_land
            & (raw > 10000.0)
        )


        impossible_land_count = (
            valid_on_land
            & (
                raw
                >
                (
                    glo_land_count
                    + count_tolerance
                )
            )
        )


        if np.any(impossible_negative):

            raise ValueError(
                f"C{slope_class} contains "
                f"{np.sum(impossible_negative):,} "
                f"negative non-NODATA values."
            )


        if np.any(impossible_global_max):

            raise ValueError(
                f"C{slope_class} contains "
                f"{np.sum(impossible_global_max):,} "
                f"counts > 10000."
            )


        if np.any(impossible_land_count):

            bad = raw[
                impossible_land_count
            ]

            raise ValueError(
                f"C{slope_class} contains "
                f"{bad.size:,} values larger than "
                f"GloLand for the same cell.\n"
                f"Example values: {bad[:20]}"
            )

        clean_count = np.where(
            valid_on_land,
            raw,
            0.0,
        ).astype(
            np.float64,
            copy=False,
        )


        class_sum += clean_count


        if slope_class >= 6:

            slope_gt15_count += (
                clean_count
            )


        explicit_class_count[
            valid_on_land
        ] += 1


        nodata_class_count[
            nodata_on_land
        ] += 1


        values = raw[
            valid_on_land
        ]


        if values.size:

            class_stats[
                slope_class
            ] = {
                "min": float(
                    np.min(values)
                ),
                "mean": float(
                    np.mean(values)
                ),
                "max": float(
                    np.max(values)
                ),
            }

        else:

            class_stats[
                slope_class
            ] = {
                "min": np.nan,
                "mean": np.nan,
                "max": np.nan,
            }


        del raw
        del clean_count

    count_error = np.abs(
        class_sum
        - glo_land_count
    )


    complete_distribution = (
        fao_land
        & (
            count_error
            <= count_tolerance
        )
    )

    complete_all_explicit = (
        complete_distribution
        & (
            explicit_class_count == 8
        )
    )


    complete_with_class_nodata = (
        complete_distribution
        & (
            explicit_class_count < 8
        )
    )


    incomplete_distribution = (
        fao_land
        & ~complete_distribution
    )


    no_explicit_classes = (
        fao_land
        & (
            explicit_class_count == 0
        )
    )


    n_land = int(
        np.sum(fao_land)
    )

    n_complete = int(
        np.sum(complete_distribution)
    )

    n_complete_all_explicit = int(
        np.sum(complete_all_explicit)
    )

    n_complete_with_nodata = int(
        np.sum(complete_with_class_nodata)
    )

    n_incomplete = int(
        np.sum(incomplete_distribution)
    )

    n_no_explicit = int(
        np.sum(no_explicit_classes)
    )


    if verbose:

        print(
            "\n"
            + "=" * 72
        )

        print(
            "FAO SLOPE-COUNT VALIDATION"
        )

        print(
            "=" * 72
        )


        print(
            f"\nFAO land cells:"
            f" {n_land:,}"
        )

        print(
            f"Complete distributions:"
            f" {n_complete:,}"
            f" ({100.0 * n_complete / n_land:.6f}%)"
        )

        print(
            f"  all 8 classes explicitly reported:"
            f" {n_complete_all_explicit:,}"
        )

        print(
            f"  complete despite class NODATA:"
            f" {n_complete_with_nodata:,}"
        )

        print(
            f"Incomplete distributions:"
            f" {n_incomplete:,}"
        )

        print(
            f"No explicit slope classes:"
            f" {n_no_explicit:,}"
        )


        print(
            "\nNODATA occurrences on FAO-land cells:"
        )

        for slope_class in range(1, 9):

            print(
                f"  C{slope_class}: "
                f"{nodata_counts_by_class[slope_class]:,}"
            )


        print(
            "\nPer-class count diagnostics:"
        )

        for slope_class in range(1, 9):

            stats = (
                class_stats[
                    slope_class
                ]
            )

            print(
                f"  C{slope_class}: "
                f"min={stats['min']:.1f}, "
                f"mean={stats['mean']:.2f}, "
                f"max={stats['max']:.1f}"
            )

        land_class_sum = (
            class_sum[
                fao_land
            ]
        )

        land_glo_count = (
            glo_land_count[
                fao_land
            ]
        )

        land_error = (
            land_class_sum
            - land_glo_count
        )


        print(
            "\nSum(C1...C8) diagnostics:"
        )

        print(
            "  [min, p1, p5, p25, p50, "
            "p75, p95, p99, max] ="
        )

        print(
            np.percentile(
                land_class_sum,
                [
                    0,
                    1,
                    5,
                    25,
                    50,
                    75,
                    95,
                    99,
                    100,
                ],
            )
        )


        print(
            "\nGloLand diagnostics:"
        )

        print(
            "  [min, p1, p5, p25, p50, "
            "p75, p95, p99, max] ="
        )

        print(
            np.percentile(
                land_glo_count,
                [
                    0,
                    1,
                    5,
                    25,
                    50,
                    75,
                    95,
                    99,
                    100,
                ],
            )
        )


        print(
            "\nSum(C1...C8) - GloLand:"
        )

        print(
            "  [min, p1, p5, p50, "
            "p95, p99, max] ="
        )

        print(
            np.percentile(
                land_error,
                [
                    0,
                    1,
                    5,
                    50,
                    95,
                    99,
                    100,
                ],
            )
        )


        print(
            f"\nMax absolute count error:"
            f" {np.max(np.abs(land_error)):.6f}"
        )


        if n_incomplete > 0:

            incomplete_error = (
                count_error[
                    incomplete_distribution
                ]
            )

            print(
                "\nIncomplete-cell |sum - GloLand|:"
            )

            print(
                "  [min, p1, p5, p50, "
                "p95, p99, max] ="
            )

            print(
                np.percentile(
                    incomplete_error,
                    [
                        0,
                        1,
                        5,
                        50,
                        95,
                        99,
                        100,
                    ],
                )
            )


    if (
        raise_on_incomplete
        and
        n_incomplete > 0
    ):

        raise ValueError(
            f"{n_incomplete:,} FAO land cells have "
            f"sum(C1...C8) != GloLand within "
            f"{count_tolerance} fine pixels."
        )

    slope_gt15 = np.full(
        shape,
        np.nan,
        dtype=np.float32,
    )


    slope_gt15[
        complete_distribution
    ] = np.divide(
        slope_gt15_count[
            complete_distribution
        ],
        glo_land_count[
            complete_distribution
        ],
    ).astype(
        np.float32,
        copy=False,
    )


    slope_gt15[
        complete_distribution
    ] = np.clip(
        slope_gt15[
            complete_distribution
        ],
        0.0,
        1.0,
    )


    # Keep an untouched FAO copy for validation and source accounting.
    fao_slope_gt15 = slope_gt15.copy()

    fao_lat, fao_lon = _fao_5min_coordinates(
        land_header,
        shape,
    )

    gmted_validation = None
    gmted_validation_high_coverage = None
    n_gmted_filled = 0
    n_gmted_available_in_fill_domain = 0

    if gmted_nc_path is not None:

        (
            gmted_slope_gt15,
            gmted_lat,
            gmted_lon,
            gmted_valid_fine_frac,
        ) = _read_gmted_slope15_netcdf(
            gmted_nc_path
        )

        if gmted_slope_gt15.shape != shape:
            raise ValueError(
                "FAO and GMTED 5' grids have different shapes:\n"
                f"  FAO   : {shape}\n"
                f"  GMTED : {gmted_slope_gt15.shape}"
            )

        if gmted_lat.size != shape[0]:
            raise ValueError(
                "GMTED latitude coordinate length does not match "
                "the FAO grid."
            )

        if gmted_lon.size != shape[1]:
            raise ValueError(
                "GMTED longitude coordinate length does not match "
                "the FAO grid."
            )

        # The GMTED preprocessing step already mapped native 40x40
        # ds75 blocks onto the nominal global 5' cell-centre grid.
        if not np.allclose(
            gmted_lat,
            fao_lat,
            atol=1e-6,
            rtol=0.0,
        ):
            max_lat_diff = float(
                np.max(np.abs(gmted_lat - fao_lat))
            )
            raise ValueError(
                "GMTED and FAO latitude coordinates do not align.\n"
                f"Maximum absolute difference: "
                f"{max_lat_diff:.12f} degrees"
            )

        if not np.allclose(
            gmted_lon,
            fao_lon,
            atol=1e-6,
            rtol=0.0,
        ):
            max_lon_diff = float(
                np.max(np.abs(gmted_lon - fao_lon))
            )
            raise ValueError(
                "GMTED and FAO longitude coordinates do not align.\n"
                f"Maximum absolute difference: "
                f"{max_lon_diff:.12f} degrees"
            )

        # ============================================================
        # FAO-GMTED OVERLAP VALIDATION
        # ============================================================

        validation_rows = (
            (fao_lat >= validation_lat_min)
            & (fao_lat < validation_lat_max)
        )

        validation_domain = (
            validation_rows[:, None]
            & np.isfinite(fao_slope_gt15)
            & np.isfinite(gmted_slope_gt15)
        )

        if not np.any(validation_domain):
            raise ValueError(
                "No finite FAO-GMTED overlap cells were found in "
                f"{validation_lat_min}-{validation_lat_max}N."
            )

        gmted_validation = _slope_validation_stats(
            fao_slope_gt15[validation_domain],
            gmted_slope_gt15[validation_domain],
        )

        if gmted_valid_fine_frac is not None:

            high_coverage_domain = (
                validation_domain
                & np.isfinite(gmted_valid_fine_frac)
                & (
                    gmted_valid_fine_frac
                    >= validation_high_coverage_frac
                )
            )

            if np.any(high_coverage_domain):
                gmted_validation_high_coverage = (
                    _slope_validation_stats(
                        fao_slope_gt15[high_coverage_domain],
                        gmted_slope_gt15[high_coverage_domain],
                    )
                )

        if verbose:
            print("\n" + "=" * 72)
            print("FAO - GMTED2010 ds75 OVERLAP VALIDATION")
            print("=" * 72)

            print(
                f"\nValidation latitude band: "
                f"{validation_lat_min} <= lat < "
                f"{validation_lat_max}"
            )

            _print_slope_validation(
                "All finite overlapping 5' cells:",
                gmted_validation,
            )

            if gmted_validation_high_coverage is not None:
                _print_slope_validation(
                    (
                        "High GMTED fine-pixel coverage "
                        f"(ValidFineFrac >= "
                        f"{validation_high_coverage_frac:.2f}):"
                    ),
                    gmted_validation_high_coverage,
                )

            print(
                "\nNOTE: validation is diagnostic only. "
                "No bias correction or calibration is applied."
            )

        # ============================================================
        # FILL ONLY FAO-MISSING CELLS
        # ============================================================

        fill_rows = (
            (fao_lat >= gmted_fill_lat_min)
            & (fao_lat < gmted_fill_lat_max)
        )

        gmted_usable = np.isfinite(gmted_slope_gt15)

        if (
            gmted_valid_fine_frac is not None
            and gmted_min_valid_fine_frac > 0.0
        ):
            gmted_usable &= (
                np.isfinite(gmted_valid_fine_frac)
                & (
                    gmted_valid_fine_frac
                    >= gmted_min_valid_fine_frac
                )
            )

        gmted_available_in_fill_domain = (
            fill_rows[:, None]
            & gmted_usable
        )

        n_gmted_available_in_fill_domain = int(
            np.sum(gmted_available_in_fill_domain)
        )

        fill_from_gmted = (
            ~np.isfinite(fao_slope_gt15)
            & gmted_available_in_fill_domain
        )

        n_gmted_filled = int(
            np.sum(fill_from_gmted)
        )

        slope_gt15[fill_from_gmted] = (
            gmted_slope_gt15[fill_from_gmted]
        )

        if verbose:
            print("\n" + "=" * 72)
            print("FAO + GMTED MERGE")
            print("=" * 72)

            print(
                f"\nGMTED fill latitude band: "
                f"{gmted_fill_lat_min} <= lat < "
                f"{gmted_fill_lat_max}"
            )

            print(
                f"Finite GMTED cells in fill domain: "
                f"{n_gmted_available_in_fill_domain:,}"
            )

            print(
                f"FAO-missing cells filled from GMTED: "
                f"{n_gmted_filled:,}"
            )

            print(
                "Finite FAO cells overwritten by GMTED: 0"
            )

    # Final distribution now includes both FAO and GMTED-filled cells.
    valid_slope = np.isfinite(slope_gt15)


    slope_values = (
        slope_gt15[
            valid_slope
        ]
    )


    if verbose and slope_values.size:

        print(
            "\n"
            + "=" * 72
        )

        print(
            "FINAL >=15% SLOPE FRACTION"
        )

        print(
            "=" * 72
        )


        print(
            "\nSlopeGT15Frac "
            "[min, p1, p5, p25, p50, "
            "p75, p95, p99, max] ="
        )

        print(
            np.percentile(
                slope_values,
                [
                    0,
                    1,
                    5,
                    25,
                    50,
                    75,
                    95,
                    99,
                    100,
                ],
            )
        )


        print(
            f"\nMean >=15% fraction:"
            f" {np.mean(slope_values):.6f}"
        )

        print(
            f"Mean <15% fraction:"
            f" {np.mean(1.0 - slope_values):.6f}"
        )


        print(
            f"\nCells with no >=15% land:"
            f" {np.sum(np.isclose(slope_values, 0.0)):,}"
        )

        print(
            f"Cells entirely >=15%:"
            f" {np.sum(np.isclose(slope_values, 1.0)):,}"
        )


    if image_land_mask is not None:

        mask = np.asarray(
            image_land_mask,
            dtype=bool,
        ).reshape(-1)


        slope_flat = (
            slope_gt15
            .reshape(
                -1,
                order="C",
            )
        )


        if mask.size != slope_flat.size:

            raise ValueError(
                "IMAGE/TIMER mask and FAO grid contain different "
                "numbers of cells:\n\n"
                f"  FAO   : {slope_flat.size:,}\n"
                f"  IMAGE : {mask.size:,}\n\n"
                "Check dimensions/orientation before applying "
                "the mask."
            )


        result = (
            slope_flat[
                mask
            ]
            .astype(
                np.float32,
                copy=False,
            )
        )


        n_result_nan = int(
            np.sum(
                ~np.isfinite(result)
            )
        )


        if verbose:

            print(
                "\n"
                + "=" * 72
            )

            print(
                "IMAGE/TIMER LANDGRID CONVERSION"
            )

            print(
                "=" * 72
            )

            print(
                f"\nFull global cells:"
                f" {slope_flat.size:,}"
            )

            print(
                f"IMAGE LandGrid cells:"
                f" {result.size:,}"
            )

            print(
                f"Finite values:"
                f" {np.sum(np.isfinite(result)):,}"
            )

            print(
                f"NaNs:"
                f" {n_result_nan:,}"
            )


            if n_result_nan > 0:

                print(
                    "\nWARNING:"
                    "\nSome IMAGE land cells do not correspond "
                    "to a validated FAO slope cell."
                    "\nThis may reflect differences between the "
                    "IMAGE and FAO land masks and should be checked "
                    "before filling the missing values."
                )


    else:

        result = (
            slope_gt15.astype(
                np.float32,
                copy=False,
            )
        )


    n_remaining_missing_global = int(
        np.sum(
            ~np.isfinite(slope_gt15)
        )
    )

    n_remaining_missing_result = int(
        np.sum(
            ~np.isfinite(result)
        )
    )

    if verbose:
        print("\n" + "=" * 72)
        print("FINAL MERGED SLOPE COVERAGE")
        print("=" * 72)

        print(
            f"\nGlobal 5' cells still NaN: "
            f"{n_remaining_missing_global:,}"
        )

        print(
            f"Returned/export grid cells still NaN: "
            f"{n_remaining_missing_result:,}"
        )

        if (
            image_land_mask is not None
            and n_remaining_missing_result > 0
        ):
            mask_flat = np.asarray(
                image_land_mask,
                dtype=bool,
            ).reshape(-1)

            missing_flat = (
                ~np.isfinite(slope_gt15)
            ).reshape(
                -1,
                order="C",
            )

            remaining_land = missing_flat[
                mask_flat
            ]

            lat_full = np.repeat(
                fao_lat,
                shape[1],
            )

            lat_land = lat_full[
                mask_flat
            ]

            missing_lat = lat_land[
                remaining_land
            ]

            if missing_lat.size:
                print(
                    "Remaining IMAGE-cell NaN latitude range: "
                    f"{np.min(missing_lat):.4f} to "
                    f"{np.max(missing_lat):.4f}"
                )

                print(
                    "Remaining IMAGE-cell NaNs >=84N: "
                    f"{np.sum(missing_lat >= 84.0):,}"
                )

                print(
                    "Remaining IMAGE-cell NaNs <55N: "
                    f"{np.sum(missing_lat < 55.0):,}"
                )


    if output_nc_path is not None:

        if image_land_mask is None:
            raise ValueError(
                "output_nc_path exports the final IMAGE LandGrid array, "
                "so image_land_mask must be supplied."
            )

        if image_land_coords is None:
            raise ValueError(
                "image_land_coords must be supplied when "
                "output_nc_path is used. "
                "Pass np.array(LandGrid.coords)."
            )

        output_path = _export_landgrid_netcdf(
            values=result,
            coords=image_land_coords,
            output_path=output_nc_path,
            var_name=output_var_name,
            description=(
                "Primary FAO HWSD terrain slope fraction. Missing FAO "
                "cells within the GMTED supplement domain are filled "
                "from GMTED2010 ds75-derived slope fractions. "
                f"FAO-GMTED validation band: "
                f"{validation_lat_min}-{validation_lat_max}N."
            ),
        )

        if verbose:
            print(
                f"\nExported merged NetCDF:\n"
                f"  {output_path}"
            )


    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            save_path,
            result,
        )


        if verbose:

            print(
                f"\nSaved slope array to:\n"
                f"  {save_path}"
            )

    diagnostics = {

        "grid_shape":
            shape,

        "n_global_cells":
            int(fao_land.size),

        "n_fao_land_cells":
            n_land,

        "n_complete":
            n_complete,

        "n_complete_all_explicit":
            n_complete_all_explicit,

        "n_complete_with_class_nodata":
            n_complete_with_nodata,

        "n_incomplete":
            n_incomplete,

        "n_no_explicit_classes":
            n_no_explicit,

        "nodata_counts_by_class":
            nodata_counts_by_class,

        "count_tolerance":
            count_tolerance,

        "gmted_nc_path":
            (
                None
                if gmted_nc_path is None
                else str(gmted_nc_path)
            ),

        "gmted_validation":
            gmted_validation,

        "gmted_validation_high_coverage":
            gmted_validation_high_coverage,

        "n_gmted_available_in_fill_domain":
            n_gmted_available_in_fill_domain,

        "n_gmted_filled":
            n_gmted_filled,

        "n_remaining_missing_global":
            n_remaining_missing_global,

        "n_remaining_missing_result":
            n_remaining_missing_result,
    }


    if slope_values.size:

        diagnostics[
            "mean_slope_gt15"
        ] = float(
            np.mean(
                slope_values
            )
        )

        diagnostics[
            "mean_slope_lt15"
        ] = float(
            np.mean(
                1.0
                - slope_values
            )
        )

        diagnostics[
            "slope_gt15_percentiles"
        ] = np.percentile(
            slope_values,
            [
                0,
                1,
                5,
                25,
                50,
                75,
                95,
                99,
                100,
            ],
        )


    if return_diagnostics:

        return (
            result,
            diagnostics,
        )


    return result


# biomass_dir = Path(
#     r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\data\biomass"
# )

# mask_land = land_mask("5m")

# SlopeGT15Frac, slope_diag = load_fao_slope_gt15(
#     rar_dir=(
#         biomass_dir
#         / "Slope_input_files"
#     ),
#     gmted_nc_path=(
#         biomass_dir
#         / "Slope15Frac_GMTED75_5min.nc"
#     ),
#     image_land_mask=mask_land,
#     image_land_coords=np.array(
#         LandGrid.coords
#     ),
#     output_nc_path=(
#         biomass_dir
#         / "SlopeGT15Frac_complete.nc"
#     ),
#     output_var_name="SlopeGT15Frac",
#     count_tolerance=0.5,
#     raise_on_incomplete=False,
#     validation_lat_min=55.0,
#     validation_lat_max=60.0,
#     validation_high_coverage_frac=0.90,
#     gmted_fill_lat_min=55.0,
#     gmted_fill_lat_max=84.0,
#     gmted_min_valid_fine_frac=0.0,
#     return_diagnostics=True,
#     verbose=True,
# )

# print(
#     "Final LandGrid NaNs:",
#     np.sum(
#         ~np.isfinite(
#             SlopeGT15Frac
#         )
#     ),
# )

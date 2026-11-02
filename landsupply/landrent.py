import prism
import warnings
import numpy as np
import pandas as pd
import scipy.io as sio
from pathlib import Path 
from pint.errors import UnitStrippedWarning
from typing import List
from scipy.io import savemat
from timer.cdr.cdr_util import (
    region_to_grid,
    read_excel_column_F,
    load_grppc_30min_as_5min_landgrid,
    export_variable_to_netcdf,
    land_data_dir,
    Crop22, Crop38, 
    GrassLand,
    Region26, Region, LandGrid, Grid,
    HarvestGroup3, Crop17,
    LandRentBin, PyrolysisTechnology
)

warnings.simplefilter("ignore", UnitStrippedWarning)
warnings.filterwarnings("ignore", message="The specified chunks separate the stored chunks", category=UserWarning)

def build_landrent_lut(
    magnet_scens: List[str],
    land_supply_file: str,
    affor_scen_dir,
    output_file=None,
):
    affor_scen_dir = Path(affor_scen_dir)

    if output_file is None:
        output_file = affor_scen_dir / "affor_LUT_input_data.mat"
    else:
        output_file = Path(output_file)

    nscens = len(magnet_scens)

    landrenttmp = np.zeros((13, len(Region26.coords), nscens), dtype=np.float32)

    for sc_idx, scen in enumerate(magnet_scens):
        filepath = affor_scen_dir / f"{scen}.xlsx"

        if not filepath.exists():
            raise FileNotFoundError(f"MAGNET land-price file not found: {filepath}")

        colF = read_excel_column_F(
            filepath,
            sheet="landprice_selected",
            nrows=338,
            skiprows=1,
        )

        reshaped = np.reshape(
            colF,
            (13, len(Region26.coords)),
            order="F",
        )

        landrenttmp[:, :, sc_idx] = reshaped

    landrenttmp = 1e6 * landrenttmp

    landrent = np.zeros((27, len(Region26.coords), nscens), dtype=np.float32)

    matlab_target_1based = list(range(11, 18)) + list(range(19, 28, 2))
    target_zero_based = [i - 1 for i in matlab_target_1based]

    landrent[target_zero_based, :, :] = landrenttmp[1:13, :, :]
    landrent[0, :, :] = landrent[10, :, :]

    x_known = np.array([1] + list(range(11, 18)) + list(range(19, 28, 2)))
    xi_full = np.arange(1, 28)

    for reg in range(len(Region26.coords)):
        for sc in range(nscens):
            y_known = landrent[x_known - 1, reg, sc]
            landrent[:, reg, sc] = np.interp(xi_full, x_known, y_known)

    aff_land_reds = np.zeros((27, 5), dtype=np.float32)

    land_supply_file_path = land_data_dir / land_supply_file

    df_reds = pd.read_excel(
        land_supply_file_path,
        sheet_name="land_supply_reductions_v2024",
        usecols="B:F",
        header=None,
        skiprows=1,
        nrows=20,
        engine="openpyxl",
    )

    arr_reds = df_reds.to_numpy(dtype=np.float32)
    aff_land_reds[7:27, :] = arr_reds

    aff_land_reds_lut = np.empty((27, 401), dtype=np.float32)
    landrent_lut = np.empty((27, len(Region26.coords), 401), dtype=np.float32)

    vars1 = np.arange(1, 402, 100)
    xi_lut = np.arange(1, 402)

    for yr in range(27):
        aff_land_reds_lut[yr, :] = np.interp(
            xi_lut,
            vars1,
            aff_land_reds[yr, :],
        )

        for reg in range(len(Region26.coords)):
            landrent_lut[yr, reg, :] = np.interp(
                xi_lut,
                vars1,
                landrent[yr, reg, :],
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)

    savemat(
        output_file,
        {
            "landrent_lut": landrent_lut,
            "aff_land_reds_lut": aff_land_reds_lut,
        },
        do_compression=True,
    )

    return output_file


@prism.interface
class LandRent(prism.Model):
    
    def configure_paths(self, landrent_scen_dir):
        directory = Path(landrent_scen_dir) 
        self._landrent_scen_dir = directory

    # Export variables 
    glandrent_scaler: prism.Array[LandGrid, np.float32] = prism.export() 
    LandRent: prism.Array[LandGrid, np.float32] = prism.export()
    LandRentBiochar: prism.Array[LandGrid, np.float32] = prism.export()

    # Local variables 
    MF_2010: prism.Array[Crop38, Region26, np.float32] = prism.private()
    GRPPC_2015: prism.Array[LandGrid, Crop38, np.float32] = prism.private()
    frharvtot_2015: prism.Array[Crop38, Region26, np.float32] = prism.private()
    cropprice: prism.Array[Crop38, np.float32] = prism.private()
    glandrent: prism.Array[LandGrid, np.float32] = prism.private()
    region_cells:  prism.Array[LandGrid, np.int16] = prism.private()

    def compute_initial_values(self, _ : prism.Timeline):

        landrent_scen_dir = getattr(self, "_landrent_scen_dir", None)
        
        # Import variables 
        var_timeline = prism.Timeline(start=prism.Q_(1970, 'year'), end=prism.Q_(2100, 'year'), stepsize=prism.Q_(1, 'year'))

        MF_tmp = prism.TimeVariable[Crop22, Region26, np.float32](
            file=landrent_scen_dir / 'MF.scn',
            timeline=var_timeline
        ) 
        GI_tmp = prism.TimeVariable[GrassLand, Region26, np.float32](
            file=landrent_scen_dir / 'GRAZINTENS.out',
            timeline=var_timeline
        )
        FRHARV = prism.TimeVariable[Region, np.float32](
            file=landrent_scen_dir / "FRHARV.SCN",
            timeline=var_timeline
        )
        FRHARVMAPSPAM = prism.Array[HarvestGroup3, Crop17, Region, np.float32](
            file=landrent_scen_dir / "FRHARVMAPSPAM.SCN"
        )
        
        t_MF = prism.Q_(2010, 'year')
        t_FRHARV = prism.Q_(2015, "year")

        self.region_cells.values, _ = region_to_grid("5m")

        self.MF_2010.values[0, :] = GI_tmp[t_MF].values[0, :]
        self.MF_2010.values[1:17, :] = MF_tmp[t_MF].values[1:17, :]
        self.MF_2010.values[22:38, :] = MF_tmp[t_MF].values[1:17, :]

        self.GRPPC_2015.values = load_grppc_30min_as_5min_landgrid(
            grppc_nc=landrent_scen_dir / "GRPPC_30MIN.NC",
            year=t_FRHARV,
            Grid=Grid,
            LandGrid=LandGrid,
            Crop38=Crop38,
            var_name="GRPPC_30MIN",
        )

        frharv = np.asarray(FRHARV[t_FRHARV].values[:26], dtype=np.float32)
        frharvmapspam = np.asarray(FRHARVMAPSPAM.values[:, :, :26], dtype=np.float32)

        self.frharvtot_2015.values[1:17, :] = (
            frharvmapspam[0, 1:17, :]
            * frharv[None, :]
        )

        self.frharvtot_2015.values[22:38, :] = (
            frharvmapspam[1, 1:17, :]
            * frharv[None, :]
        )

        cropprice_raw = pd.read_excel(
            land_data_dir / 'Commodity_data.xlsx', 
            sheet_name='data', 
            usecols='G', 
            skiprows=21, 
            nrows=17, 
            header=None
        )
        cropprice_in = cropprice_raw[6].to_numpy()
        
        self.cropprice.values = np.zeros(len(Crop38.coords), dtype=np.float32)
        self.cropprice.values[:17] = cropprice_in
        self.cropprice.values[22:38] = cropprice_in[1:17]
        
    def compute_values(
        self,
        time: prism.Time,
        GLCT: prism.Array[LandGrid, np.int8],
        GFORMAN: prism.Array[LandGrid, np.int8],
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GAREA5m:  prism.Array[LandGrid, np.float32],
        YieldEffectsBiochar: prism.Array[LandGrid, PyrolysisTechnology, np.float32],
        LandRent_LUT: prism.Array[Region26, LandRentBin, np.float32],
        AfforLandRedsLUT: prism.Array[LandRentBin, np.float32]
    ):
        
        self._calculate_gridded_landrent(
            time=time,
            GLCT=GLCT,
            GFORMAN=GFORMAN,
            GFRAC=GFRAC,
            GAREA5m=GAREA5m
        )

        self._calculate_final_landrent(
            time=time,
            GLCT=GLCT,
            LandRent_LUT=LandRent_LUT,
            AfforLandRedsLUT=AfforLandRedsLUT,
        )

        self._calculate_final_landrent_biochar(
            time=time,
            GLCT=GLCT,
            LandRent_LUT=LandRent_LUT,
            AfforLandRedsLUT=AfforLandRedsLUT,
            YieldEffectsBiochar=YieldEffectsBiochar
        )

        # self._write_results_to_files(
        #     time=time
        # )

    def _calculate_gridded_landrent(
        self,
        time: prism.Time,
        GLCT: prism.Array[LandGrid, np.int8],
        GFORMAN: prism.Array[LandGrid, np.int8],
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GAREA5m:  prism.Array[LandGrid, np.float32],
    ):
        t = time.t

        region_cells = self.region_cells.values

        print(f"Starting land rent calculations for year {t.magnitude}")

        glandrent = np.zeros(len(LandGrid.coords), dtype=np.float32)
        glandrent_scaler = np.zeros(len(LandGrid.coords), dtype=np.float32)

        valid_region_mask = (region_cells >= 0) & (region_cells < len(Region26.coords))

        grass_like_mask = (
            valid_region_mask
            & (GFORMAN.values == 4)
            & np.isfinite(self.GRPPC_2015.values[:, 0])
        )

        if np.any(grass_like_mask):
            grass_mf = np.take(
                self.MF_2010.values[0, :],
                region_cells,
                mode="clip",
            )
            grass_mf[region_cells < 0] = 0.0

            grass_yld = self.GRPPC_2015.values[:, 0]
            grass_price = float(self.cropprice.values[0])

            glandrent[grass_like_mask] += (
                grass_yld[grass_like_mask]
                * grass_mf[grass_like_mask]
                * grass_price
            )

        for c in Crop38:
            frac = GFRAC.values[:, c]

            if np.all(frac == 0):
                continue

            reg = region_cells

            mf = np.take(
                self.MF_2010.values[c, :],
                reg,
                mode="clip",
            )
            mf[reg < 0] = 0.0

            price = float(self.cropprice.values[c])
            yld = self.GRPPC_2015.values[:, c]

            if c == 0:
                mask = (
                    valid_region_mask
                    & (frac > 0.0)
                    & (GLCT.values == 1)
                )

                glandrent[mask] += (
                    frac[mask]
                    * yld[mask]
                    * mf[mask]
                    * price
                )

            elif (1 <= c <= 11) or (22 <= c <= 32):
                mask = (
                    valid_region_mask
                    & (frac > 0.0)
                )

                frh = np.take(
                    self.frharvtot_2015.values[c, :],
                    reg,
                    mode="clip",
                )
                frh[reg < 0] = 0.0

                glandrent[mask] += (
                    frac[mask]
                    * yld[mask]
                    * mf[mask]
                    * frh[mask]
                    * price
                )

        for reg in Region26:
            reg_int = int(reg)

            mask = (
                valid_region_mask
                & (region_cells == reg_int)
                & np.isfinite(glandrent)
                & (glandrent > 0.0)
                & np.isfinite(GAREA5m.values)
                & (GAREA5m.values > 0.0)
            )

            if np.any(mask):
                mean_val = np.average(
                    glandrent[mask],
                    weights=GAREA5m.values[mask],
                )

                if np.isfinite(mean_val) and mean_val > 0.0:
                    glandrent_scaler[mask] = glandrent[mask] / mean_val
            else:
                print(
                    f"Warning: no positive land-rent cells found for region {reg_int} "
                    f"in year {t.magnitude}."
                )

        self.glandrent.values = glandrent
        self.glandrent_scaler.values = glandrent_scaler

 
    def _calculate_final_landrent(
            self,
            time: prism.Time,
            GLCT: prism.Array[LandGrid, np.int8],
            LandRent_LUT: prism.Array[Region26, LandRentBin, np.float32],
            AfforLandRedsLUT: prism.Array[LandRentBin, np.float32]
    ):
        
        t = time.t
        region_cells = self.region_cells.values

        for reg in Region26:
            region_mask = (region_cells == reg)

            cdr_share = 0.0

            idx = int(np.argmin(np.abs(AfforLandRedsLUT.values - cdr_share))) 

            self.LandRent.values[region_mask] = np.where(
                (GLCT.values[region_mask] == 1) | (GLCT.values[region_mask] == 2),
                self.glandrent_scaler.values[region_mask] * LandRent_LUT.sel(bin=idx, region26=reg).values,
                self.glandrent_scaler.values[region_mask] * LandRent_LUT.sel(bin=0, region26=reg).values
            )
        

    def _calculate_final_landrent_biochar(
            self,
            time: prism.Time,
            GLCT: prism.Array[LandGrid, np.int8],
            LandRent_LUT: prism.Array[Region26, LandRentBin, np.float32],
            AfforLandRedsLUT: prism.Array[LandRentBin, np.float32],
            YieldEffectsBiochar: prism.Array[LandGrid, PyrolysisTechnology, np.float32],
    ):
        t = time.t
        region_cells = self.region_cells.values

        for reg in Region26:
            region_mask = (region_cells == reg)

            cdr_share = 0.0

            idx = int(np.argmin(np.abs(AfforLandRedsLUT.values - cdr_share))) 

            yield_effects = YieldEffectsBiochar[:,0]

            self.LandRentBiochar.values[region_mask] = np.where(
                (GLCT.values[region_mask] == 1) | (GLCT.values[region_mask] == 2),
                self.glandrent_scaler.values[region_mask] * LandRent_LUT.sel(bin=idx, region26=reg).values * (1.0 + yield_effects.values[region_mask]/100.0), 
                self.glandrent_scaler.values[region_mask] * LandRent_LUT.sel(bin=0, region26=reg).values * (1.0 + yield_effects.values[region_mask]/100.0)
            )


    def _write_results_to_files(
                self,
                time: prism.Time
        ):
            t, dt, t_start, t_end = time.t, time.dt, time.start, time.end

            times = np.arange(
                t_start.magnitude,
                t_end.magnitude + dt.magnitude,
                dt.magnitude,
                dtype=int
            )

            if t == t_end:
                export_variable_to_netcdf(
                    prism_var=self.glandrent,
                    filename="glandrent.nc",
                    var_name="glandrent",
                    dims=("grid",),
                    coords={
                        "grid": np.array(LandGrid.coords),
                    },
                    times=None,
                    units="-",
                    description="",
                )

                export_variable_to_netcdf(
                    prism_var=self.glandrent_scaler,
                    filename="glandrent_scaler.nc",
                    var_name="glandrent_scaler",
                    dims=("grid",),
                    coords={
                        "grid": np.array(LandGrid.coords),
                    },
                    times=None,
                    units="-",
                    description="",
                )

if __name__ == "__main__":
    print("Running land rent standalone model")

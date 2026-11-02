import prism
import time
import argparse 
import numpy as np
import xarray as xr
import scipy.io as sio
from pathlib import Path
from timer.cdr.landsupply.landrent import LandRent
from timer.cdr.biomass.biomass import Biomass
from timer.cdr.biochar.biochar import Biochar
from timer.cdr.biochar.config import configure_biochar_model
from timer.cdr.scenario_config import (
    load_biochar_config,
    resolve_runs,
    write_resolved_run_config
)
from timer.cdr.cdr_util import (
    time_step, 
    time_step_scen,
    land_mask,
    biocrops_data_dir, res_data_dir, affor_data_dir, global_data_dir,
    Grid, LandGrid,
    Region26, Region27, Region28, Region,
    LandCover, LandCoverPlus, Fertiliser, SoilDepth,
    Crop22, Crop38, BioenergyCrop3, BioenergyCrop5,
    ForestManagementPlus, CarbonPool, LandRentBin,
    Sector, Carrier, TransportMode, RCCPlus, Residue,
)

@prism.interface
class CO2Removal(prism.Model):

    def init_submodels(self, timeline: prism.Timeline):
        self.landrent = LandRent(timeline) 
        self.biomass = Biomass(timeline)
        self.biochar = Biochar(timeline)

    def compute_values(
        self,
        time: prism.Time,
        GNLCT: prism.Array[LandGrid, np.int8],
        GLCT: prism.Array[LandGrid, np.int8],
        GLCT_landrent: prism.Array[LandGrid, np.int8],
        GFORMAN: prism.Array[LandGrid, np.int8],
        GFORMAN_landrent: prism.Array[LandGrid, np.int8],
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GFRAC_landrent: prism.Array[LandGrid, Crop38, np.float32],
        GRPPC: prism.Array[LandGrid, Crop38, np.float32],
        GLANDCOVERFrac: prism.Array[LandGrid, LandCover, np.float32],
        GLANDCOVERFracDetail: prism.Array[LandGrid, LandCoverPlus, np.float32],
        GBIOMASS: prism.Array[LandGrid, CarbonPool, np.float32],
        GTIMBER: prism.Array[LandGrid, ForestManagementPlus, np.float32],
        GAREA5m: prism.Array[LandGrid, np.float32],
        GPOPD: prism.Array[LandGrid, np.float32],
        GCH4: prism.Array[LandGrid, np.float32],
        GN2O: prism.Array[LandGrid, np.float32],
        GFERTILISER: prism.Array[LandGrid, Fertiliser, np.float32],
        SOILTEMP: prism.Array[LandGrid, SoilDepth, np.float32],
        MF_Agr: prism.Array[Crop22, Region26, np.float32],
        FracHarv_in: prism.Array[Region26, np.float32],
        BFProdShare: prism.Array[BioenergyCrop3, Region27, np.float32],
        LandRent_LUT: prism.Array[Region26, LandRentBin, np.float32],
        AfforLandRedsLUT: prism.Array[LandRentBin, np.float32],
        DiscountRate: prism.Array[Region, np.float32],
        CCTransport: prism.Array[Region27, np.float32],
        FeedstockEnergyUse: prism.Array[Region27,BioenergyCrop5, np.float32],
        Interest: prism.Array[Region27, np.float32],
        BioRiskFactor: prism.Array[Region27, np.float32],
        ctax: prism.Array[Region28, np.float32],
        AAPRSE: prism.TimeVariable[Region27, Sector, Carrier, np.float32],
        MF: prism.Array[Region27,BioenergyCrop5, np.float32],
        GDPpc: prism.Array[Region28, np.float32],
        GDPpc_tscen: prism.Array[Region28, np.float32],
        BioLabourCost_in: prism.Array[Region27, np.float32], 
        CO2STORCOST:  prism.Array[Region27, np.float32],
        PriceSecFuel: prism.Array[Region27, Sector, Carrier, np.float32],
        CCpertkm: prism.Array[Region26, TransportMode, np.float32],
        ResCostCurveDM: prism.TimeVariable[Region26, Residue, RCCPlus, np.float32],
        ResPotCurveDM: prism.TimeVariable[Region26, Residue, RCCPlus, np.float32],
        # ExoResidues:  prism.Array[Region27, np.float32],
    ):
        
        # Ensure LandRentBiochar sees the CURRENT timestep's age-averaged
        # response before NPV is constructed (and not last timestep's map).
        if self.biochar.agronomic_response_mode == "aging_within_window":
            self.biochar._estimate_soil_effects(
                time=time,
                GAREA5m=GAREA5m,
                GLANDCOVERFrac=GLANDCOVERFrac,
                GFRAC=GFRAC,
                GFERTILISER=GFERTILISER,
                GCH4=GCH4,
                GN2O=GN2O,
            )

        self.landrent.compute_values(
            time=time,
            GFRAC=GFRAC_landrent,
            GLCT=GLCT_landrent,
            GFORMAN=GFORMAN_landrent,
            GAREA5m=GAREA5m,
            YieldEffectsBiochar=self.biochar.yield_effects,
            LandRent_LUT=LandRent_LUT,
            AfforLandRedsLUT=AfforLandRedsLUT
        )
        
        # self.biomass.compute_values(
        #     time=time,
        #     GFRAC=GFRAC,
        #     GRPPC=GRPPC,
        #     # GLANDCOVERFrac=GLANDCOVERFrac,
        #     # GLANDCOVERFracDetail=GLANDCOVERFracDetail,
        #     GNLCT=GNLCT,
        #     GBIOMASS=GBIOMASS,
        #     GTIMBER=GTIMBER,
        #     GAREA5m=GAREA5m,
        #     GPOPD=GPOPD,
        #     BioLabourCost_in=BioLabourCost_in, 
        #     # LandRent=self.landrent.LandRent,
        #     MF_Agr=MF_Agr,
        #     FracHarv_in=FracHarv_in,
        #     BFProdShare=BFProdShare, 
        #     # CCTransport=CCTransport,
        #     # FeedstockEnergyUse=FeedstockEnergyUse,
        #     # Interest=Interest,
        #     # BioRiskFactor=BioRiskFactor,
        #     # ctax=ctax,
        #     MF=MF,
        #     # GDPpc=GDPpc,
        #     # GDPpc_tscen=GDPpc_tscen,
        #     # ExoResidues=ExoResidues
        # )

        self.biochar.compute_values(
            time=time,
            GFRAC=GFRAC,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GAREA5m=GAREA5m,
            GCH4=GCH4,
            GN2O=GN2O,
            GFERTILISER=GFERTILISER,
            SOILTEMP=SOILTEMP,
            GPOPD=GPOPD,
            LandRent=self.landrent.LandRent,
            LandRentBiochar=self.landrent.LandRentBiochar,
            ResCostCurveDM=ResCostCurveDM,
            ResPotCurveDM=ResPotCurveDM,
            ctax=ctax,
            energy_price=PriceSecFuel,
            AAPRSE=AAPRSE,
            regional_discount_rates=DiscountRate,
            CO2STORCOST=CO2STORCOST,
            CCpertkm=CCpertkm,
        )
        

def run_scenario(run_config, timeline_config):
    name = run_config["name"]

    ctax_dir = run_config["carbon_tax_directory"]
    biochar_land_dir = run_config["biochar_land_directory"]
    landrent_dir = run_config["landrent_directory"]
    residue_dir = run_config["residue_directory"]
    output_dir = run_config["output_directory"]

    settings = run_config.get("settings", {})

    print()
    print("=" * 70)
    print(f"Running scenario: {name}")
    print(f"Carbon tax:      {run_config['carbon_tax_name']}")
    print(f"Land scenario:   {run_config['land_name']}")
    print(f"Output:          {output_dir}")
    print(f"Settings:        {settings}")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_config_file = write_resolved_run_config(
        run_config=run_config,
        timeline_config=timeline_config,
    )

    simulation_timeline = prism.Timeline(
        start=prism.Q_(timeline_config["start"], "year"),
        end=prism.Q_(timeline_config["end"], "year"),
        stepsize=prism.Q_(timeline_config["step"], "year"),
    )

    mask_land_5m = land_mask("5m")

    CCTransport = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "CCTransport.out"
    ) 

    FeedstockEnergyUse = prism.TimeVariable[Region27,BioenergyCrop5, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "FeedStockEnergyUse.dat"
    ) 
    Interest = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "Interest.out"
    ) 
    BioRiskFactor = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "biorisk.scn"
    ) 
    ctax = prism.TimeVariable[Region28, np.float32](
        timeline=simulation_timeline,
        file=ctax_dir / "ctax.out"
    )
    MF_Agr = prism.TimeVariable[Crop22, Region26, np.float32](
        timeline=simulation_timeline,
        file=biochar_land_dir / "MF.scn"
    )
    FracHarv_in = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline,
        file=biochar_land_dir / "FRHARV.SCN"
    )
    BFProdShare = prism.TimeVariable[BioenergyCrop3, Region27, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BFPRODSHARE.out"
    )
    MF = prism.TimeVariable[Region27,BioenergyCrop5, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / 'MF.out'
    )
    GDPpc = prism.TimeVariable[Region28, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "GDPpc.out"
    )
    BioLabourCost_in = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline, 
        file=res_data_dir / "BioLabourCost.out"
    )
    AAPRSE = prism.TimeVariable[Region27,Sector, Carrier, np.float32](
        timeline=simulation_timeline, 
        file=ctax_dir / "AAPRSE_exo.dat"
    )
    CO2STORCOST = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline, 
        file=ctax_dir / "CO2STORCOST.out"
    )
    PriceSecFuel = prism.TimeVariable[Region27, Sector, Carrier, np.float32](
        timeline=simulation_timeline, 
        file=ctax_dir / "PriceSecFuel.out"
    )
    CCpertkm = prism.TimeVariable[Region26, TransportMode, np.float32](
        timeline=simulation_timeline, 
        file=ctax_dir / "CCpertkm.out"
    )

    # ExoResidues = prism.TimeVariable[Region, np.float32](
    #     timeline=simulation_timeline, 
    #     file=residue_dir / "ExoResidues.dat"
    # )

    ds_ResCostCurveDM = xr.open_dataset(
        residue_dir / "ResCostCurveDM.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["ResCostCurveDM"]
    ds_ResPotCurveDM = xr.open_dataset(
        residue_dir / "ResPotCurveDM.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["ResPotCurveDM"]

    ds_gfrac = xr.open_dataset(
        biochar_land_dir / "GFRAC.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GFRAC"]
    ds_gfrac_landrent = xr.open_dataset(
        landrent_dir / "GFRAC.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GFRAC"]
    ds_grppc = xr.open_dataset(
        biochar_land_dir / "GRPPC.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GRPPC"]
    ds_glandcover = xr.open_dataset(
        biochar_land_dir / "GLANDCOVER.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GLANDCOVER"]
    ds_glandcoverdet = xr.open_dataset(
        biochar_land_dir / "GLANDCOVERDETAIL.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GLANDCOVERDETAIL"]
    ds_gnlct = xr.open_dataset(
        biochar_land_dir / "GNLCT.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )[ "GNLCT"]
    ds_glct = xr.open_dataset(
        biochar_land_dir / "GLCT.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GLCT"]
    ds_glct_landrent = xr.open_dataset(
        landrent_dir / "GLCT.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GLCT"]
    ds_gforman = xr.open_dataset(
        biochar_land_dir / "GFORMAN.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )[ "GFORMAN"]
    ds_gforman_landrent = xr.open_dataset(
        landrent_dir / "GFORMAN.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )[ "GFORMAN"]
    ds_gbiomass = xr.open_dataset(
        biochar_land_dir / "GBIOMASS.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GBIOMASS"]
    ds_gtimber = xr.open_dataset(
        biochar_land_dir / "GTIMBERHARVEST.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GTIMBERHARVEST"]
    ds_garea = xr.open_dataset(
        global_data_dir / "GAREA.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GAREA"]
    ds_gpopd = xr.open_dataset(
        biochar_land_dir / "GPOPD.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GPOPD"]
    ds_n2o = xr.open_dataset(
        biochar_land_dir / "GEN2OFE.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GEN2OFE"]
    ds_ch4 = xr.open_dataset(
        biochar_land_dir / "GECH4RI.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GECH4RI"]
    ds_gfertiliser = xr.open_dataset(
        biochar_land_dir / "GFERTILIZER.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GFERTILIZER"]
    ds_soiltemp = xr.open_dataset(
        biochar_land_dir / "soil_temp.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["soiltemp"]

    mat_data_land = sio.loadmat(landrent_dir / 'affor_LUT_input_data.mat')
    landrent_lut = mat_data_land["landrent_lut"].astype(np.float32, copy=False)
    aff_land_reds_lut = mat_data_land["aff_land_reds_lut"].astype(np.float32, copy=False)
    
    regional_discrates = sio.loadmat(landrent_dir / 'regional_scenario_discountrates.mat')
    regional_discrates = regional_discrates['DR_reg_gdppcmer'].astype(np.float32, copy=False)

    def input_fun(t):
        sub = ds_ResPotCurveDM.isel(time=time_step_scen(t))
        sub_np = np.asarray(sub, dtype=np.float32)   
        sub_np = sub_np.reshape(len(Region26.coords), len(Residue.coords), len(RCCPlus.coords)) 
        ResPotCurveDM = prism.array(sub_np, dims=[Region26, Residue, RCCPlus])
        del sub, sub_np

        sub = ds_ResCostCurveDM.isel(time=time_step_scen(t))
        sub_np = np.asarray(sub, dtype=np.float32)   
        sub_np = sub_np.reshape(len(Region26.coords), len(Residue.coords), len(RCCPlus.coords)) 
        ResCostCurveDM = prism.array(sub_np, dims=[Region26, Residue, RCCPlus])
        del sub, sub_np

        sub = ds_gfrac.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)     
        GFRAC_flat = sub_np.reshape(len(Grid.coords), len(Crop38.coords)) 
        GFRAC_flat = GFRAC_flat[mask_land_5m, :]
        GFRAC = prism.array(GFRAC_flat, dims=[LandGrid, Crop38])
        del sub, sub_np, GFRAC_flat

        sub = ds_gfrac_landrent.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)     
        GFRAC_flat = sub_np.reshape(len(Grid.coords), len(Crop38.coords)) 
        GFRAC_flat = GFRAC_flat[mask_land_5m, :]
        GFRAC_landrent = prism.array(GFRAC_flat, dims=[LandGrid, Crop38])
        del sub, sub_np, GFRAC_flat

        sub = ds_grppc.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                             
        GRPPC_flat = sub_np.reshape(len(Grid.coords), len(Crop38.coords))
        GRPPC_flat = GRPPC_flat[mask_land_5m, :]
        GRPPC = prism.array(GRPPC_flat, dims=[LandGrid, Crop38])
        del sub, sub_np, GRPPC_flat

        sub = ds_glandcover.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)           
        GLANDCOVERFrac_flat = sub_np.reshape(len(Grid.coords), len(LandCover.coords)) 
        GLANDCOVERFrac_flat = GLANDCOVERFrac_flat[mask_land_5m, :] 
        GLANDCOVERFrac = prism.array(GLANDCOVERFrac_flat, dims=[LandGrid, LandCover])
        del sub, sub_np, GLANDCOVERFrac_flat
        
        sub = ds_glandcoverdet.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                               
        GLANDCOVERFracDetail_flat = sub_np.reshape(len(Grid.coords), len(LandCoverPlus.coords)) 
        GLANDCOVERFracDetail_flat = GLANDCOVERFracDetail_flat[mask_land_5m, :]
        GLANDCOVERFracDetail = prism.array(GLANDCOVERFracDetail_flat, dims=[LandGrid, LandCoverPlus])
        del sub, sub_np, GLANDCOVERFracDetail_flat

        sub = ds_gnlct.isel(time=time_step(t))
        sub_np = np.asarray(sub)                                 
        GNLCT_flat = sub_np.reshape(len(Grid.coords)) 
        GNLCT_flat = GNLCT_flat[mask_land_5m]
        GNLCT = prism.array(GNLCT_flat, dims=[LandGrid])
        del sub, sub_np, GNLCT_flat

        sub = ds_glct.isel(time=time_step(t))
        sub_np = np.asarray(sub)                                  
        GLCT_flat = sub_np.reshape(len(Grid.coords)) 
        GLCT_flat = GLCT_flat[mask_land_5m]
        GLCT = prism.array(GLCT_flat, dims=[LandGrid])
        del sub, sub_np, GLCT_flat

        sub = ds_glct_landrent.isel(time=time_step(t))
        sub_np = np.asarray(sub)                                  
        GLCT_flat = sub_np.reshape(len(Grid.coords)) 
        GLCT_flat = GLCT_flat[mask_land_5m]
        GLCT_landrent = prism.array(GLCT_flat, dims=[LandGrid])
        del sub, sub_np, GLCT_flat

        sub = ds_gforman.isel(time=time_step(t), NFORMID=0)
        sub_np = np.asarray(sub)                                
        GFORMAN_flat = sub_np.reshape(len(Grid.coords)) 
        GFORMAN_flat = GFORMAN_flat[mask_land_5m]
        GFORMAN = prism.array(GFORMAN_flat, dims=[LandGrid])
        del sub, sub_np, GFORMAN_flat

        sub = ds_gforman_landrent.isel(time=time_step(t), NFORMID=0)
        sub_np = np.asarray(sub)                                
        GFORMAN_flat = sub_np.reshape(len(Grid.coords)) 
        GFORMAN_flat = GFORMAN_flat[mask_land_5m]
        GFORMAN_landrent = prism.array(GFORMAN_flat, dims=[LandGrid])
        del sub, sub_np, GFORMAN_flat

        sub = ds_gbiomass.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                              
        GBIOMASS_flat = sub_np.reshape(len(Grid.coords), len(CarbonPool.coords)) 
        GBIOMASS_flat = GBIOMASS_flat[mask_land_5m]
        GBIOMASS = prism.array(GBIOMASS_flat, dims=[LandGrid, CarbonPool])
        del sub, sub_np, GBIOMASS_flat

        sub = ds_gtimber.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                              
        GTIMBERHARVEST_flat = sub_np.reshape(len(Grid.coords), len(ForestManagementPlus.coords)) 
        GTIMBERHARVEST_flat = GTIMBERHARVEST_flat[mask_land_5m]
        GTIMBER = prism.array(GTIMBERHARVEST_flat, dims=[LandGrid, ForestManagementPlus])
        del sub, sub_np, GTIMBERHARVEST_flat

        sub = ds_garea.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                         
        GAREA5m_flat = sub_np.reshape(len(Grid.coords))
        GAREA5m_flat = GAREA5m_flat[mask_land_5m]
        GAREA5m = prism.array(GAREA5m_flat, dims=[LandGrid])
        del sub, sub_np, GAREA5m_flat

        sub = ds_gpopd.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                          
        GPOPD_flat = sub_np.reshape(len(Grid.coords))
        GPOPD_flat = GPOPD_flat[mask_land_5m]
        GPOPD = prism.array(GPOPD_flat, dims=[LandGrid])
        del sub, sub_np, GPOPD_flat

        sub = ds_ch4.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                          
        GCH4_flat = sub_np.reshape(len(Grid.coords))
        GCH4_flat = GCH4_flat[mask_land_5m]
        GCH4 = prism.array(GCH4_flat, dims=[LandGrid])
        del sub, sub_np, GCH4_flat

        sub = ds_n2o.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                          
        GN2O_flat = sub_np.reshape(len(Grid.coords))
        GN2O_flat = GN2O_flat[mask_land_5m]
        GN2O = prism.array(GN2O_flat, dims=[LandGrid])
        del sub, sub_np, GN2O_flat

        sub = ds_gfertiliser.isel(time=time_step(t))
        sub_np = np.asarray(sub, dtype=np.float32)                          
        GFERTILISER_flat = sub_np.reshape(len(Grid.coords), len(Fertiliser.coords))
        GFERTILISER_flat = GFERTILISER_flat[mask_land_5m]
        GFERTILISER = prism.array(GFERTILISER_flat, dims=[LandGrid, Fertiliser])
        del sub, sub_np, GFERTILISER_flat

        sub = ds_soiltemp.isel(time=time_step(t))
        full_lat = np.arange(89.9583333333, -89.9583333333 - 1e-6, -5/60, dtype=np.float32)
        full_lon = np.arange(-179.9583333333, 179.9583333333 + 1e-6, 5/60, dtype=np.float32)

        soil_full = sub.interp(
            latitude=full_lat,
            longitude=full_lon,
            method="linear"
        ).astype(np.float32)

        soil_full = soil_full.transpose("latitude", "longitude", "layer")
        soil_full = soil_full.values
        SOILTEMP_flat = soil_full.reshape(
            len(Grid.coords),
            len(SoilDepth.coords)
        )
        SOILTEMP_flat = SOILTEMP_flat[mask_land_5m]
        SOILTEMP = prism.array(
            SOILTEMP_flat,
            dims=[LandGrid, SoilDepth]
        )
        del sub, soil_full, SOILTEMP_flat

        LandRent_LUT = prism.array(landrent_lut[time_step(t),:,:], dims=[Region26, LandRentBin])
        AfforLandRedsLUT = aff_land_reds_lut[time_step(t),:]
        AfforLandRedsLUT = prism.array(aff_land_reds_lut[time_step(t),:], dims=[LandRentBin])

        DiscountRate = prism.array(np.round(regional_discrates[time_step(t),:] / 100.0, 2), dims=[Region])

        cc_raw = CCpertkm.get(t)
        cc_values = np.asarray(cc_raw.values, dtype=np.float32).copy()
        cc_values[:, 3] = np.maximum(cc_values[:, 3], 0.0)
        CCpertkm_biochar = prism.array(cc_values, dims=[Region26, TransportMode])

        return dict(
            GFRAC=GFRAC,
            GFRAC_landrent=GFRAC_landrent,
            GRPPC=GRPPC,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GLANDCOVERFracDetail=GLANDCOVERFracDetail,
            GNLCT=GNLCT,
            GLCT=GLCT,
            GLCT_landrent=GLCT_landrent,
            GFORMAN=GFORMAN,
            GFORMAN_landrent=GFORMAN_landrent,
            GPOPD=GPOPD,
            GCH4=GCH4,
            GN2O=GN2O,
            GFERTILISER=GFERTILISER,
            SOILTEMP=SOILTEMP,
            GAREA5m=GAREA5m,
            MF_Agr=MF_Agr.get(t),
            FracHarv_in=FracHarv_in.get(t),
            BFProdShare=BFProdShare.get(t),
            GBIOMASS=GBIOMASS,
            GTIMBER=GTIMBER,
            LandRent_LUT=LandRent_LUT,
            AfforLandRedsLUT=AfforLandRedsLUT,
            DiscountRate=DiscountRate,
            CCTransport=CCTransport.get(t),
            FeedstockEnergyUse=FeedstockEnergyUse.get(t),
            Interest=Interest.get(t),
            BioRiskFactor=BioRiskFactor.get(t),
            ctax=ctax.get(t),
            MF=MF.get(t),
            GDPpc=GDPpc.get(t),
            GDPpc_tscen=GDPpc.get(prism.Q_(2025, "year")),
            BioLabourCost_in=BioLabourCost_in.get(t),
            AAPRSE=AAPRSE,
            CO2STORCOST=CO2STORCOST.get(t),
            PriceSecFuel=PriceSecFuel.get(t),
            CCpertkm=CCpertkm_biochar,
            ResCostCurveDM=ResCostCurveDM,
            ResPotCurveDM=ResPotCurveDM,
            # ExoResidues=ExoResidues.get(t)
        )
    
    start_wall = time.perf_counter()

    cdr = CO2Removal(simulation_timeline)
    
    cdr.landrent.configure_paths(
        landrent_scen_dir=landrent_dir
    )

    configure_biochar_model(
        model=cdr.biochar,
        run_config=run_config,
    )
    cdr.simulate(
        simulation_timeline, 
        inputs=input_fun
    )

    end_wall = time.perf_counter()

    print("Finished running carbon dioxide removal model")
    print(f"Total running time: {((end_wall - start_wall)/60):.2f} min")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=Path,
        help="Path to biochar scenario YAML file",
    )

    args = parser.parse_args()
    config = load_biochar_config(args.config)
    runs = resolve_runs(config)

    print(f"Found {len(runs)} biochar scenario(s)")

    batch_start = time.perf_counter()

    for i, run_config in enumerate(runs, start=1):
        print()
        print(
            f"[{i}/{len(runs)}] "
            f"{run_config['name']}"
        )
        start = time.perf_counter()
        run_scenario(
            run_config=run_config,
            timeline_config=config["timeline"],
        )
        elapsed = (time.perf_counter() - start) / 60
        print(
            f"Scenario {run_config['name']} "
            f"finished in {elapsed:.2f} min"
        )

    total_elapsed = (time.perf_counter() - batch_start) / 60

    print()
    print("=" * 70)
    print(
        f"Finished {len(runs)} scenario(s) "
        f"in {total_elapsed:.2f} min"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()

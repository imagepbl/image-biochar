import prism
import numpy as np
import xarray as xr
import scipy.io as sio
from scipy import stats
from timer.cdr.biomass.biocrops import BioCrop
from timer.cdr.biomass.residues import Residues
from timer.cdr.cdr_util import (
    time_step,
    land_mask, 
    biocrops_data_dir, land_data_dir,
    res_data_dir, 
    land_output_dir,
    Grid, LandGrid,
    Region26, Region27, Region28, Region, ImportRegion,
    LandCover, LandCoverPlus, NLandType,
    Crop22, Crop38, BioenergyCrop3, BioenergyCrop5,
    ForestManagementPlus,
    CarbonPool, Fuel, Sector, Carrier, H2_Fuel, NonEn_Fuel,
    BioTechnology, BLFTechnology, BSFTechnology, BSFTechnologyPlus,
    LandRentBin, NBC2, BioCostCat, EmisSector, SectorBSF, SectorBLF,
)


@prism.interface
class Biomass(prism.Model):
    
    def init_submodels(self, timeline: prism.Timeline):
        self.biocrops = BioCrop(timeline)
        self.residues = Residues(timeline)

    def compute_values(
        self,
        time: prism.Time,
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GRPPC: prism.Array[LandGrid, Crop38, np.float32],
        # GLANDCOVERFrac:  prism.Array[LandGrid, LandCover, np.float32],
        # GLANDCOVERFracDetail: prism.Array[LandGrid, LandCoverPlus, np.float32],
        GNLCT: prism.Array[LandGrid, np.int8],
        GBIOMASS: prism.Array[LandGrid, CarbonPool, np.float32],
        GTIMBER: prism.Array[LandGrid, ForestManagementPlus, np.float32],
        GAREA5m: prism.Array[LandGrid, np.float32],
        GPOPD: prism.Array[LandGrid, np.float32],
        BioLabourCost_in: prism.Array[Region27, np.float32], 
        # LandRent: prism.Array[LandGrid, np.float32],
        MF_Agr: prism.Array[Crop22, Region26, np.float32],
        FracHarv_in: prism.Array[Region26, np.float32],
        BFProdShare: prism.Array[BioenergyCrop3, Region, np.float32], 
        # CCTransport: prism.Array[Region27, np.float32],
        # FeedstockEnergyUse: prism.Array[Region27,BioenergyCrop5, np.float32],
        # Interest: prism.Array[Region27, np.float32],
        # BioRiskFactor: prism.Array[Region27, np.float32],
        # ctax: prism.Array[Region27, np.float32],
        MF: prism.Array[Region27,BioenergyCrop5, np.float32],
        # GDPpc: prism.Array[Region28, np.float32],
        # GDPpc_tscen: prism.Array[Region28, np.float32],
        # ExoResidues: prism.Array[Region27, np.float32],
    ):
           
        self.biocrops.compute_values(
            time,
            GAREA5m=GAREA5m,
            GFRAC=GFRAC,
            GRPPC=GRPPC,
            # GLANDCOVERFrac=GLANDCOVERFrac,
            # GLANDCOVERFracDetail=GLANDCOVERFracDetail,
            # GNLCT=GNLCT,
            MF_Agr=MF_Agr,
            FracHarv_in=FracHarv_in,
            BFProdShare=BFProdShare, 
            # CTransport=CCTransport,
            # FeedstockEnergyUse=FeedstockEnergyUse,
            # Interest=Interest,
            # BioRiskFactor=BioRiskFactor,
            # ctax=ctax,
            MF=MF,
            # GDPpc=GDPpc,
            # GDPpc_tscen=GDPpc_tscen,
        )

        self.residues.compute_values(
            time,
            BioLabourCost_in=BioLabourCost_in,
            GBIOMASS=GBIOMASS,
            GTIMBER=GTIMBER,
            GNLCT=GNLCT,
            GAREA5m=GAREA5m,
            GPOPD=GPOPD,
            CropYield_Cor=self.biocrops.CropYield_Cor,
            CropProd=self.biocrops.CropProd,
            CropFrac=self.biocrops.CropFrac,
            # ExoResidues=ExoResidues
        )

if __name__ == "__main__":
    print("Running biomass standalone model")

    simulation_timeline = prism.Timeline(start=prism.Q_(2020, 'year'), end=prism.Q_(2050, 'year'), stepsize=prism.Q_(5, 'year'))

    mask_land_5m = land_mask('5m')

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
    ctax = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "CarbonTax.out"
    )
    BLFConversionEmission = prism.TimeVariable[Region27, BLFTechnology, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BLFConversionEmission.out"
    )
    BSFConversionEmission = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BSFConversionEmission.out"
    )
    BLFDemSpec = prism.TimeVariable[Region27, SectorBLF, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BLFDemSpec.out"
    )
    BSFDemSpec = prism.TimeVariable[Region27, SectorBSF, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BSFDemSpec.out"
    )
    BLFNetTradeMatrix = prism.TimeVariable[Region27, ImportRegion, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BLFNetTradeMatrix.out"
    )
    BSFNetTradeMatrix = prism.TimeVariable[Region27, ImportRegion, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BSFNetTradeMatrix.out"
    )
    BLFPriceNT = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BLFPriceMargNT.out"
    )
    BSFPriceNT = prism.TimeVariable[Region27, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BSFPriceMargNT.out"
    )
    SecondaryBLFEfficiency = prism.TimeVariable[Region27, BioTechnology, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "SecBiEff.dat"
    )
    BLFProd = prism.TimeVariable[Region27, BLFTechnology, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BLFProd.out"
    )
    BSFProd = prism.TimeVariable[Region27, BSFTechnologyPlus, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BSFProd.out"
    )
    CarbonCapturedSpec2 = prism.TimeVariable[Region28, EmisSector, Fuel, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "CarbonCapturedSpec2.out"
    )
    BLF_CostPerCrop_spec1 = prism.TimeVariable[Region27, BLFTechnology, NLandType, BioCostCat, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BLF_CostPerCrop_spec1.out"
    )
    BSF_CostPerCrop_spec1 = prism.TimeVariable[Region27, BSFTechnology, NLandType, BioCostCat, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BSF_CostPerCrop_spec1.out"
    )
    BLFCostTransp = prism.TimeVariable[Region27, ImportRegion, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BLFCostTransp.out"
    )
    BSFCostTransp = prism.TimeVariable[Region27, ImportRegion, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "BSFCostTransp.out"
    )
    MF_Agr = prism.TimeVariable[Crop22, Region26, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "MF.scn"
    )
    FracHarv_in = prism.TimeVariable[Region26, np.float32](
        timeline=simulation_timeline,
        file=biocrops_data_dir / "FRHARV.out"
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

    ds_gfrac = xr.open_dataset(
        biocrops_data_dir / "GFRAC.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GFRAC"]
    ds_grppc = xr.open_dataset(
        biocrops_data_dir / "GRPPC.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GRPPC"]
    ds_glandcover = xr.open_dataset(
        biocrops_data_dir / "GLANDCOVER.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GLANDCOVER"]
    ds_glandcoverdet = xr.open_dataset(
        biocrops_data_dir / "GLANDCOVERDETAIL.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GLANDCOVERDETAIL"]
    ds_gnlct = xr.open_dataset(
        biocrops_data_dir / "GNLCT.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )[ "GNLCT"]
    ds_gbiomass = xr.open_dataset(
        res_data_dir / "GBIOMASS.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GBIOMASS"]
    ds_gtimber = xr.open_dataset(
        res_data_dir / "GTIMBERHARVEST.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GTIMBERHARVEST"]
    ds_garea = xr.open_dataset(
        res_data_dir / "GAREA.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GAREA"]
    ds_gpopd = xr.open_dataset(
        res_data_dir / "GPOPD.NC", 
        engine="netcdf4", 
        chunks={"time": 1}
    )["GPOPD"]

    def input_fun(t):

        sub = ds_gfrac.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.float32)                              
        GFRAC_flat = sub_np.reshape(len(Grid.coords), len(Crop38.coords)) 
        GFRAC_flat = GFRAC_flat[mask_land_5m, :]
        GFRAC = prism.array(GFRAC_flat, dims=[LandGrid, Crop38])

        sub = ds_grppc.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.float32)                              
        GRPPC_flat = sub_np.reshape(len(Grid.coords), len(Crop38.coords))
        GRPPC_flat = GRPPC_flat[mask_land_5m, :]
        GRPPC = prism.array(GRPPC_flat, dims=[LandGrid, Crop38])

        sub = ds_glandcover.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.float32)                   
        GLANDCOVERFrac_flat = sub_np.reshape(len(Grid.coords), len(LandCover.coords)) 
        GLANDCOVERFrac_flat = GLANDCOVERFrac_flat[mask_land_5m, :] 
        GLANDCOVERFrac = prism.array(GLANDCOVERFrac_flat, dims=[LandGrid, LandCover])
        
        sub = ds_glandcoverdet.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.float32)                                   
        GLANDCOVERFracDetail_flat = sub_np.reshape(len(Grid.coords), len(LandCoverPlus.coords)) 
        GLANDCOVERFracDetail_flat = GLANDCOVERFracDetail_flat[mask_land_5m, :]
        GLANDCOVERFracDetail = prism.array(GLANDCOVERFracDetail_flat, dims=[LandGrid, LandCoverPlus])

        sub = ds_gnlct.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.int8)                                  
        GNLCT_flat = sub_np.reshape(len(Grid.coords)) 
        GNLCT_flat = GNLCT_flat[mask_land_5m]
        GNLCT = prism.array(GNLCT_flat, dims=[LandGrid])

        sub = ds_gbiomass.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.float32)                              
        GBIOMASS_flat = sub_np.reshape(len(Grid.coords), len(CarbonPool.coords))
        GBIOMASS_flat = GBIOMASS_flat[mask_land_5m, :]
        GBIOMASS = prism.array(GBIOMASS_flat, dims=[LandGrid, CarbonPool])

        sub = ds_gtimber.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.float32)                              
        GTIMBER_flat = sub_np.reshape(len(Grid.coords), len(ForestManagementPlus.coords))
        GTIMBER_flat = GTIMBER_flat[mask_land_5m, :]
        GTIMBER = prism.array(GTIMBER_flat, dims=[LandGrid, ForestManagementPlus])

        sub = ds_garea.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.float32)                              
        GAREA5m_flat = sub_np.reshape(len(Grid.coords))
        GAREA5m_flat = GAREA5m_flat[mask_land_5m]
        GAREA5m = prism.array(GAREA5m_flat, dims=[LandGrid])

        sub = ds_gpopd.isel(time=time_step(t))
        sub_np = np.asarray(sub).astype(np.float32)                              
        GPOPD_flat = sub_np.reshape(len(Grid.coords))
        GPOPD_flat = GPOPD_flat[mask_land_5m]
        GPOPD = prism.array(GPOPD_flat, dims=[LandGrid])

        return dict(
                GFRAC=GFRAC,
                GRPPC=GRPPC,
                GLANDCOVERFrac=GLANDCOVERFrac,
                GLANDCOVERFracDetail=GLANDCOVERFracDetail,
                GNLCT=GNLCT,
                MF_Agr=MF_Agr.get(t),
                FracHarv_in=FracHarv_in.get(t),
                BFProdShare=BFProdShare.get(t),
                CCTransport=CCTransport.get(t),
                FeedstockEnergyUse=FeedstockEnergyUse.get(t),
                Interest=Interest.get(t),
                BioRiskFactor=BioRiskFactor.get(t),
                ctax=ctax.get(t),
                BLFConversionEmission=BLFConversionEmission.get(t),
                BSFConversionEmission=BSFConversionEmission.get(t),
                BLFDemSpec=BLFDemSpec.get(t),
                BSFDemSpec=BSFDemSpec.get(t),
                BLFNetTradeMatrix=BLFNetTradeMatrix.get(t),
                BSFNetTradeMatrix=BSFNetTradeMatrix.get(t),
                BLFPriceNT=BLFPriceNT.get(t),
                BSFPriceNT=BSFPriceNT.get(t),
                SecondaryBLFEfficiency=SecondaryBLFEfficiency.get(t),
                BLFProd=BLFProd.get(t),
                BSFProd=BSFProd.get(t),
                CarbonCapturedSpec2=CarbonCapturedSpec2.get(t),
                BLF_CostPerCrop_spec1=BLF_CostPerCrop_spec1.get(t),
                BSF_CostPerCrop_spec1=BSF_CostPerCrop_spec1.get(t),
                BLFCostTransp=BLFCostTransp.get(t),
                BSFCostTransp=BSFCostTransp.get(t),
                MF=MF.get(t),
                GDPpc=GDPpc.get(t),
                GDPpc_tscen=GDPpc.get(prism.Q_(2025, "year")),
                BioLabourCost_in=BioLabourCost_in.get(t),
                GBIOMASS=GBIOMASS,
                GTIMBER=GTIMBER,
                GAREA5m=GAREA5m,
                GPOPD=GPOPD
            )
    
    biomass = Biomass(simulation_timeline)
    biomass.simulate(simulation_timeline, inputs=input_fun)

    import matplotlib.pyplot as plt

    supply = biomass.residues.ResPotCurveDM.values
    cost = biomass.residues.ResCostCurveDM.values

    n_residues = supply.shape[1]

    for res in range(n_residues):
        supply_res = supply[:, res, :]  
        cost_res = cost[:, res, :]       

        supply_flat = supply_res.flatten()
        cost_flat = cost_res.flatten()

        mask = (supply_flat > 0) & (cost_flat <= 10)
        supply_flat = supply_flat[mask]
        cost_flat = cost_flat[mask]

        if len(supply_flat) == 0:
            continue

        order = np.argsort(cost_flat)
        supply_sorted = supply_flat[order]
        cost_sorted = cost_flat[order]

        cum_supply = np.cumsum(supply_sorted)

        # Plot
        plt.figure(figsize=(6, 4))
        plt.step(cum_supply, cost_sorted, where="post")
        plt.xlabel("Cumulative biomass residue supply")
        plt.ylabel("Cost ($/GJ)")
        plt.title(f"Residue {res + 1}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # supply = biomass.residues.ResPotCurve.values       
    # cost = biomass.residues.ResCostCurve.values     

    # n_regions = supply.shape[0]

    # for r in range(n_regions):

    #     supply_r = supply[r, :]
    #     cost_r = cost[r, :]

    #     # keep only positive supply and cost ≤ 10
    #     mask = (supply_r > 0) & (cost_r <= 10)

    #     supply_filtered = supply_r[mask]
    #     cost_filtered = cost_r[mask]

    #     if len(supply_filtered) == 0:
    #         continue

    #     # sort by cost
    #     order = np.argsort(cost_filtered)
    #     supply_sorted = supply_filtered[order]
    #     cost_sorted = cost_filtered[order]

    #     # cumulative supply
    #     cum_supply = np.cumsum(supply_sorted)

    #     plt.figure(figsize=(6,4))
    #     plt.step(cum_supply, cost_sorted, where="post")

    #     plt.xlabel("Cumulative biomass residue supply")
    #     plt.ylabel("Cost ($/GJ)")
    #     plt.title(f"Region {r+1}")
    #     plt.grid(True)

    #     plt.show()

    # # Extract arrays
    # supply = biomass.residues.ResPotCurve.values       
    # cost = biomass.residues.ResCostCurve.values         

    # # Flatten to global list of supply-cost pairs
    # supply_flat = supply.flatten()
    # cost_flat = cost.flatten()

    # # Remove zero supply entries
    # mask = (supply_flat > 0) & (cost_flat <= 10)
    # supply_flat = supply_flat[mask]
    # cost_flat = cost_flat[mask]

    # # Sort by cost
    # order = np.argsort(cost_flat)
    # cost_sorted = cost_flat[order]
    # supply_sorted = supply_flat[order]

    # # Compute cumulative supply
    # cum_supply = np.cumsum(supply_sorted)

    # # Plot cost-supply curve
    # plt.figure(figsize=(7,5))
    # plt.step(cum_supply, cost_sorted, where="post")
    # plt.xlabel("Cumulative biomass residue supply")
    # plt.ylabel("Cost")
    # plt.title("Global Biomass Residue Cost-Supply Curve")
    # plt.grid(True)

    # plt.show()


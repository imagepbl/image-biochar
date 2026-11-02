import prism
import warnings
import numpy as np
import xarray as xr
from pint.errors import UnitStrippedWarning
from timer.globals.constants import Region, EPS
from timer.cdr.cdr_util import (
    region_to_grid, 
    time_step, 
    safe_division,
    land_mask,
    calc_cost_curve,
    plot_gridded_variable,
    export_variable_to_netcdf,
    res_data_dir,
    Grid, LandGrid, LandGrid30m, Crop16,
    Region26, Region27,Biomes,
    Crop15, Crop26, Residue,
    ForestManagement, ForestManagementPlus,
    CarbonPool, PopDim, ResCalc, Sector,
    NGST, NFPT, NAT, NFHTT, NWCT, NTRAD, NWC,
    RCC, RCCPlus,
    WoodProduct
)

warnings.simplefilter("ignore", UnitStrippedWarning)

@prism.interface
class Residues(prism.Model):
    EroConstAgr: prism.Scalar = 350
    SOCConst: prism.Scalar = 0.0
    FellperWood: prism.Scalar = 1.3
    kgWoodperkgC: prism.Scalar = 2.08
    ForHHV: prism.Scalar = 18
    CropHaulingCost: prism.Scalar = 0.07
    AgrHarvestCostCrop_in: prism.Scalar = 4788.77
    TranspCost_in: prism.Scalar = 0.0119
    StorCost: prism.Scalar = 1.149
    ForChipCost_in: prism.Scalar = 1.5
    ForForwCost_in: prism.Scalar = 0.86
    ForAddiCost_in: prism.Scalar = 0.5
    ResidueProcCost: prism.Scalar = 2.0

    ResiduesTotal: prism.Array[Region26, np.float32] = prism.export()
    ResiduesSupply: prism.Array[Region26, Residue, np.float32] = prism.export()

    ExoResiduesPotCurve: prism.TimeVariable[Region26, RCCPlus, np.float32] = prism.export()
    TimerResiduesPotCurve: prism.TimeVariable[Region26, RCCPlus, np.float32] = prism.export()

    PopDens: prism.Array[PopDim] = prism.private(file=res_data_dir / 'pop_density.dat')
    TranspDist: prism.Array[PopDim] = prism.private(file=res_data_dir / 'transp_distance.dat')
    CropRPR: prism.Array[Crop26] = prism.private(file=res_data_dir / 'crop_RPR.dat')
    RPRFac: prism.Array[Region, ResCalc, Crop26] = prism.private(file=res_data_dir / 'RPRFac.dat')
    MoistureContent: prism.Array[Crop15, np.float32] = prism.private(file=res_data_dir / 'MoistureContent.dat')
    CropHHV: prism.Array[Crop15, np.float32] = prism.private(file=res_data_dir / 'crop_HHV.dat')
    CropProcessRatio: prism.Array[Crop26, np.float32] = prism.private(file=res_data_dir / 'crop_process_ratio.dat')
    ForResPotPerVeg: prism.Array[Biomes, ForestManagement, np.float32] = prism.private(file=res_data_dir / 'ForResPotperVeg.dat')
    ForEcoFrac: prism.Array[Biomes, ForestManagement, np.float32] = prism.private(file=res_data_dir / 'ForEcoFrac.dat')
    ForProcessRatio: prism.Array[WoodProduct, np.float32] = prism.private(file=res_data_dir / 'ForProcessRatio.dat')
    CropFertCost: prism.Array[Crop15] = prism.private(file=res_data_dir / 'crop_fert_cost.dat')

    SE_Trad: prism.TimeVariable[Region27, Sector, NTRAD, np.float32] = prism.private(file=res_data_dir / 'SE_Trad.out')
    FeedDemand_in: prism.TimeVariable[NGST, NFPT, NAT, Region27, np.float32] = prism.private(file=res_data_dir / 'TFeed.out')
    WoodHarvest_in: prism.TimeVariable[NFHTT, NWCT, Region27, np.float32] = prism.private(file=res_data_dir / 'BiomHarvWood.out')

    SOC: prism.Array[LandGrid, np.float32] = prism.private()
    AgrRPR: prism.Array[LandGrid, Crop26, np.float32] = prism.private()
    AgrTheoPot: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    AgrTheoPotpkm: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    AgrProcPotEn: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    AgrEcoPotEnpkm: prism.Array[LandGrid, Crop15, np.float32] = prism.private()

    AgrEcoPot: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    AgrEcoPotEn: prism.Array[LandGrid, Crop15, np.float32] = prism.private()

    GTIMBER: prism.Array[LandGrid, ForestManagement, np.float32] = prism.private()
    WoodProdSpec: prism.Array[LandGrid, ForestManagement, np.float32] = prism.private()
    ForResPotPerVeg_full: prism.Array[LandGrid, ForestManagement, np.float32] = prism.private()
    ForTheoPotSpec: prism.Array[LandGrid, ForestManagement, np.float32] = prism.private()
    ForEcoFrac_full: prism.Array[LandGrid, ForestManagement, np.float32] = prism.private()
    ForEcoPotSpec: prism.Array[LandGrid, ForestManagement, np.float32] = prism.private()
    ForEcoPotTotEn: prism.Array[LandGrid, np.float32] = prism.private()
    WoodDemand: prism.Array[Region27, WoodProduct, np.float32] = prism.private()
    ForProcPot: prism.Array[Region26, np.float32] = prism.private()

    BioLabourCost: prism.Array[LandGrid, np.float32] = prism.private()
    TransportDistance: prism.Array[LandGrid, np.float32] = prism.private()
    TranspCost: prism.Array[LandGrid, np.float32, np.float32] = prism.private()
    AgrHarvestCostCrop: prism.Array[LandGrid, Crop15] = prism.private()
    AgrHarvestCost: prism.Array[LandGrid, np.float32] = prism.private()
    AgrOperCostCrop: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    AgrOperCost: prism.Array[LandGrid, np.float32] = prism.private()
    AgrCostCropTot: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    AgrCostTot: prism.Array[LandGrid, np.float32] = prism.private()

    ForChipCost: prism.Array[LandGrid, np.float32] = prism.private()
    ForForwCost: prism.Array[LandGrid, np.float32] = prism.private()
    ForAddiCost: prism.Array[LandGrid, np.float32] = prism.private()
    ForCostTot: prism.Array[LandGrid, np.float32] = prism.private()

    TradFuelUse: prism.Array[Region26, np.float32] = prism.private()
    FeedDemand: prism.Array[Region26, np.float32] = prism.private()
    FeedDemandCrop: prism.Array[Region26, Crop15, np.float32] = prism.private()
    AgrEcoPotR: prism.Array[Region26, Crop15, np.float32] = prism.private()
    AgrProcPotREn: prism.Array[Region26, Crop15, np.float32] = prism.private()
    AgrTechPotREn: prism.Array[Region26, Crop15, np.float32] = prism.private()
    ForEcoPotTotREn: prism.Array[Region26, np.float32] = prism.private()

    ResidueCost: prism.Array[LandGrid, np.float32] = prism.private()
    ResCostCurvePot: prism.Array[LandGrid, np.float32] = prism.private()
    ResCostCurvePotR: prism.Array[Region26, np.float32] = prism.private()
    ResCostCurve1: prism.Array[Region26, RCC, np.float32] = prism.private()
    ResCostCurve2: prism.Array[Region26, RCC, np.float32] = prism.private()
    ResPotCurve1: prism.Array[Region26, RCC, np.float32] = prism.private()
    ResPotCurve2: prism.Array[Region26, RCC, np.float32] = prism.private()
    ResEconUse: prism.Array[Region26, np.float32] = prism.private()
    ResTechCat1: prism.Array[Region26, np.float32] = prism.private()
    ResTechCat: prism.Array[Region26, np.int8] = prism.private()
    ResCostCurve: prism.Array[Region26, RCCPlus, np.float32] = prism.private()
    ResPotCurve: prism.Array[Region26, RCCPlus, np.float32] = prism.private()

    AgrEcoPotDry: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    AgrProcPotDry: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    ForEcoPotTotDry: prism.Array[LandGrid, np.float32] = prism.private()
    ForProcPotDryR: prism.Array[Region26, np.float32] = prism.private()

    AgrEcoPotDryR: prism.Array[Region26, Crop15, np.float32] = prism.private()
    AgrProcPotDryR: prism.Array[Region26, Crop15, np.float32] = prism.private()
    AgrTechPotDryR: prism.Array[Region26, Crop15, np.float32] = prism.private()
    ForEcoPotTotDryR: prism.Array[Region26, np.float32] = prism.private()
    TradFuelUseDry: prism.Array[Region26, np.float32] = prism.private()
    
    ResiduesSupplyDM: prism.Array[Region26, Residue, np.float32] = prism.private()
    ResiduePotDM: prism.Array[LandGrid, Residue, np.float32] = prism.private()

    ResidueCostDM: prism.Array[LandGrid, Residue, np.float32] = prism.private()
    ResCostCurvePotDM: prism.Array[LandGrid, Residue, np.float32] = prism.private()
    ResCostCurvePotRDM: prism.Array[Region26, Residue, np.float32] = prism.private()
    ResCostCurve1DM: prism.Array[Region26, Residue, RCC, np.float32] = prism.private()
    ResCostCurve2DM: prism.Array[Region26, Residue, RCC, np.float32] = prism.private()
    ResPotCurve1DM: prism.Array[Region26, Residue, RCC, np.float32] = prism.private()
    ResPotCurve2DM: prism.Array[Region26, Residue, RCC, np.float32] = prism.private()
    ResEconUseDM: prism.Array[Region26, Residue, np.float32] = prism.private()
    ResTechCat1DM: prism.Array[Region26, Residue, np.float32] = prism.private()
    ResTechCatDM: prism.Array[Region26, Residue, np.int8] = prism.private()
    ResCostCurveDM: prism.TimeVariable[Region26, Residue, RCCPlus, np.float32] = prism.private()
    ResPotCurveDM: prism.TimeVariable[Region26, Residue, RCCPlus, np.float32] = prism.private()

    def compute_values(
            self,
            time: prism.Time,
            BioLabourCost_in: prism.Array[Region27, np.float32],
            GBIOMASS: prism.Array[LandGrid, CarbonPool, np.float32],
            GTIMBER: prism.Array[LandGrid, ForestManagementPlus, np.float32],
            GNLCT: prism.Array[LandGrid, np.int8],
            GAREA5m: prism.Array[LandGrid, np.float32],
            GPOPD: prism.Array[LandGrid, np.float32],
            CropYield_Cor: prism.Array[LandGrid, Crop26, np.float32],
            CropProd: prism.Array[LandGrid, Crop26, np.float32],
            CropFrac: prism.Array[LandGrid, Crop15, np.float32],
            # ExoResidues: prism.Array[Region27, RCCPlus, np.float32],
    ):
        self._calculate_agr_supply(
            time=time,
            GBIOMASS=GBIOMASS,
            GAREA5m=GAREA5m,
            CropYield_Cor=CropYield_Cor,
            CropProd=CropProd,
            CropFrac=CropFrac
        )

        self._calculate_for_supply(
            time=time,
            GTIMBER=GTIMBER,
            GNLCT=GNLCT
        )

        self._calculate_res_cost(
            time=time,
            BioLabourCost_in=BioLabourCost_in,
            CropFrac=CropFrac,
            GPOPD=GPOPD
        )

        self._calculate_regional_supply(
            time=time
        )

        self._calculate_regional_curves(
            time=time
        )

        self._calculate_regional_mass_supply(
            time=time
        )

        self._calculate_regional_type_curves(
            time=time
        )

        # self._calculate_timer_exogenous_residue_curve(
        #     time=time,
        #     ExoResidues=ExoResidues
        # )

        self._write_results_to_files(
            time=time
        )

    def _calculate_agr_supply(
        self,
        time: prism.Time,
        GBIOMASS: prism.Array[LandGrid, CarbonPool, np.float32],
        GAREA5m: prism.Array[LandGrid, np.float32],
        CropYield_Cor: prism.Array[LandGrid, Crop26, np.float32],
        CropProd: prism.Array[LandGrid, Crop26, np.float32],
        CropFrac: prism.Array[LandGrid, Crop15, np.float32]
    ):
        t = time.t
        print(f"Start residues calculations for year {t.magnitude}")

        region_cells, region_mask = region_to_grid('5m')

        # vals = np.zeros((len(LandGrid.coords), len(ResCalc.coords), len(Crop26.coords)))
        # vals[region_mask, :, :] = self.RPRFac.values[region_cells[region_mask], :, :]
        # self.RPRFac_full.values = vals

        RPR_0 = self.RPRFac.values[region_cells[region_mask], 0, :].astype(np.float32, copy=False)
        RPR_1 = self.RPRFac.values[region_cells[region_mask], 1, :].astype(np.float32, copy=False)
        yield_vals = np.maximum(CropYield_Cor.values[region_mask, :], EPS).astype(np.float32, copy=False)

        self.AgrRPR.values[region_mask, :] = np.where(
            (CropYield_Cor.values[region_mask, :] > EPS),
            np.maximum(
                self.CropRPR.values[None, :], 
                RPR_0 * np.log(yield_vals) + RPR_1
            ).astype(np.float32, copy=False),
            0.0
        ).astype(np.float32, copy=False)
        
        agr_theo_pot1 = np.zeros((len(LandGrid.coords), len(Crop26.coords)), dtype=np.float32)
        valid_crop26 = np.r_[0:11, 12:26]
        agr_theo_pot1[:, valid_crop26] = (
            CropProd.values[:, valid_crop26].astype(np.float32, copy=False) *
            self.AgrRPR.values[:, valid_crop26].astype(np.float32, copy=False)
        )

        self.AgrTheoPot.values = np.zeros((len(LandGrid.coords), len(Crop15.coords)), dtype=np.float32)
        left = np.arange(15, dtype=np.int16)
        right = left + 15
        use_pair = left < 11

        self.AgrTheoPot.values[:, use_pair] = agr_theo_pot1[:, left[use_pair]] + agr_theo_pot1[:, right[use_pair]]
        self.AgrTheoPot.values[:, ~use_pair] = agr_theo_pot1[:, left[~use_pair]]

        # for crop in range(11):
        #     self.AgrTheoPot.loc[dict(crop15=crop)] = self.AgrTheoPot1.sel(crop26=crop) + self.AgrTheoPot1.sel(crop26=crop+15)
        # for crop in range(11,15):
        #     self.AgrTheoPot.loc[dict(crop15=crop)] = self.AgrTheoPot1.sel(crop26=crop)

        cropfrac_area = (
            CropFrac.values * GAREA5m.values[:, None]
        ).astype(np.float32, copy=False)
        self.AgrTheoPotpkm.values = np.divide(
            self.AgrTheoPot.values,
            cropfrac_area,
            out=np.zeros_like(self.AgrTheoPot.values, dtype=np.float32),
            where=cropfrac_area > EPS
        )

        agr_proc_pot1 = (
            self.CropProcessRatio.values[None, :] * CropProd.values
        ).astype(np.float32, copy=False)

        agr_proc_pot = np.zeros((len(LandGrid.coords), len(Crop15.coords)), dtype=np.float32)
        left = np.arange(15, dtype=np.int16)
        right = left + 15
        use_pair = left < 11

        agr_proc_pot[:, use_pair] = agr_proc_pot1[:, left[use_pair]] + agr_proc_pot1[:, right[use_pair]]
        agr_proc_pot[:, ~use_pair] = agr_proc_pot1[:, left[~use_pair]]

        self.AgrProcPotDry.values = agr_proc_pot * (1 - self.MoistureContent.values[None, :])
        self.AgrProcPotEn.values = agr_proc_pot * (1 - self.MoistureContent.values[None, :]) * self.CropHHV.values[None, :]

        self.SOC.values = (GBIOMASS.values[:,5] + GBIOMASS.values[:,6])/1000.0

        self.AgrEcoPot.values = np.maximum(
            0.0, 
            self.AgrTheoPotpkm - self.EroConstAgr
        ) * CropFrac * GAREA5m

        self.AgrEcoPotDry = self.AgrEcoPot * (1.0 - self.MoistureContent)

        self.AgrEcoPotEn = self.AgrEcoPotDry * self.CropHHV
    
        self.AgrEcoPotEnpkm.values = np.where(
            (CropFrac * GAREA5m) > EPS,
            self.AgrEcoPotEn / (CropFrac * GAREA5m),
            0.0
        ).astype(np.float32, copy=False)

    def _calculate_for_supply(
            self,
            time: prism.Time,
            GTIMBER: prism.Array[LandGrid, ForestManagementPlus, np.float32],
            GNLCT: prism.Array[LandGrid, np.int8]
    ):
        t, dt, t_start = time.t, time.dt, time.start

        # for m in ForestManagement:
        #     self.GTIMBER[t].loc[dict(forest_manag=m)] = GTIMBER.values[:,m]

        for m in ForestManagement:
            self.GTIMBER.values[:, m] = GTIMBER.values[:,m]

        wood_prod = np.zeros((len(LandGrid.coords), len(ForestManagement.coords)), dtype=np.float32)
        # for k in range(1,11):
        #     tt = max(t - k * dt, t_start)
        #     wood_prod += self.GTIMBER[tt] 

        self.WoodProdSpec.values = (
            # wood_prod / 10 * self.FellperWood * self.kgWoodperkgC * 1/(1 - 0.2)
            self.GTIMBER.values / 10 * self.FellperWood * self.kgWoodperkgC * 1/(1 - 0.2)
        )

        biome_idx = GNLCT.astype(np.int8).values - 1

        self.ForResPotPerVeg_full.values = np.where(
            self.WoodProdSpec > EPS,
            self.ForResPotPerVeg.values[biome_idx, :],
            0.0
        )

        self.ForTheoPotSpec.values = np.where(
            (biome_idx[:, None] >= 0) & 
            (self.WoodProdSpec > EPS) & 
            ((1 - self.ForResPotPerVeg_full) > EPS),
            self.WoodProdSpec * self.ForResPotPerVeg_full / (1 - self.ForResPotPerVeg_full),
            0.0
        )

        self.ForEcoFrac_full.values = np.where(
            self.WoodProdSpec > EPS,
            self.ForEcoFrac.values[biome_idx, :],
            0.0
        )
        
        self.ForEcoPotSpec.values = np.where(
            self.WoodProdSpec > EPS,
            self.ForTheoPotSpec * self.ForEcoFrac_full,
            0.0
        )

        self.ForEcoPotTotDry = self.ForEcoPotSpec.sum(ForestManagement.label) * (1 - 0.2)
        self.ForEcoPotTotEn = self.ForEcoPotTotDry * self.ForHHV

        wood_demand_reg = np.zeros((len(Region27.coords), 2), dtype=np.float32)
        for k in range(1, 11):
            tt = max(t - k * dt, t_start)
            wood_demand_reg[:, 0] += self.WoodHarvest_in[tt].sel(nfhtt=4, nwct=0).values
            wood_demand_reg[:, 1] += self.WoodHarvest_in[tt].sel(nfhtt=4, nwct=1).values

        self.WoodDemand.values = (wood_demand_reg / 10) * self.kgWoodperkgC * self.ForHHV

        self.ForProcPot = (self.WoodDemand.sel(region=slice(0,25)) * self.ForProcessRatio).sum(WoodProduct.label)
        self.ForProcPotDryR = self.ForProcPot / self.ForHHV

    def _calculate_res_cost(
        self,
        time: prism.Time,
        BioLabourCost_in: prism.Array[Region27, np.float32], 
        CropFrac: prism.Array[LandGrid, Crop15, np.float32],
        GPOPD: prism.Array[LandGrid, np.float32]
    ):
        t = time.t
        region_cells, region_mask = region_to_grid("5m")

        labour_cost = np.zeros((len(LandGrid.coords)), dtype=np.float32)
        labour_cost[region_mask] = BioLabourCost_in.values[region_cells[region_mask]]
        self.BioLabourCost.values = labour_cost

        self.TransportDistance.values = np.where(
            GPOPD.values > 200,
            20,
            np.interp(GPOPD.values, self.PopDens, self.TranspDist)
        )

        has_residue = (
            (self.AgrEcoPotEn.values.sum(axis=1) > EPS) |
            (self.ForEcoPotTotEn.values > EPS)
        )
        self.TranspCost.values = np.where(
            has_residue,
            self.TransportDistance.values * self.TranspCost_in,
            0.0
        ).astype(np.float32, copy=False)

        self.AgrHarvestCostCrop.values = np.where(
            (self.AgrEcoPotEnpkm > EPS),
            (self.AgrHarvestCostCrop_in * (self.BioLabourCost / BioLabourCost_in.sel(region=1)) / self.AgrEcoPotEnpkm),
            0.0
        )
    
        num = (self.AgrHarvestCostCrop.values * CropFrac.values).sum(axis=1)
        den = CropFrac.values.sum(axis=1)
        self.AgrHarvestCost.values = safe_division(num, den, den > EPS)

        self.AgrOperCostCrop = self.CropHaulingCost + self.CropFertCost

        num = ((self.CropHaulingCost + self.CropFertCost.values) * CropFrac.values).sum(axis=1)
        den = CropFrac.values.sum(axis=1)
        self.AgrOperCost.values = safe_division(num, den, den > EPS)

        self.AgrCostCropTot.values = np.where(
            self.AgrEcoPot > EPS,
            self.StorCost + self.AgrHarvestCostCrop + self.AgrOperCostCrop + self.TranspCost,
            0.0
        )
        self.AgrCostTot.values = np.where(
            self.AgrEcoPot.sum(Crop15.label) > EPS,
            self.StorCost + self.AgrHarvestCost + self.AgrOperCost + self.TranspCost,
            0.0
        )

        num = self.ForChipCost_in * self.BioLabourCost
        den = BioLabourCost_in.sel(region=10)
        self.ForChipCost.values = np.maximum(0.5, safe_division(num, den, (den > EPS)))

        num = self.ForForwCost_in * self.BioLabourCost
        den = BioLabourCost_in.sel(region=10)
        self.ForForwCost.values = np.maximum(0.3, safe_division(num, den, (den > EPS)))

        num = self.ForAddiCost_in * self.BioLabourCost
        den = BioLabourCost_in.sel(region=10)
        self.ForAddiCost.values = np.maximum(0.3, safe_division(num, den, (den > EPS)))

        self.ForCostTot.values = np.where(
            self.ForEcoPotTotEn > EPS,
            self.ForChipCost + self.TranspCost + self.ForForwCost + self.ForAddiCost,
            0.0
        )
    
    def _calculate_regional_supply(
            self,
            time: prism.Time,
    ):  
        t, dt, t_start = time.t, time.dt, time.start
        region_cells, region_mask = region_to_grid("5m")

        agr_eco_pot = self.AgrEcoPot.values[region_mask, :]
        agr_eco_pot_reg = np.zeros((len(Region26.coords), (len(Crop15.coords))), dtype=np.float32)
        np.add.at(agr_eco_pot_reg, region_cells[region_mask], agr_eco_pot)
        self.AgrEcoPotR.values = agr_eco_pot_reg

        agr_eco_share = np.where(
            self.AgrEcoPotR > EPS,
            self.AgrEcoPotR / self.AgrEcoPotR.sum(Crop15.label),
            0.0
        )
        self.FeedDemandCrop.values = (
            self.FeedDemand_in[t].sel(nfpt=2, ngst=2, nat=5, region=slice(0, 25)).values[:, None] 
            * 1000 
            * agr_eco_share
        )

        fuel_harvest_an = np.zeros(len(Region26.coords), dtype=np.float32)
        for k in range(1, 11):
            tt = max(t - k * dt, t_start)
            fuel_harvest_an += self.WoodHarvest_in[tt].sel(nfhtt=4, nwct=2, region=slice(0,25)).values
        
        fuel_wood_harvest = (fuel_harvest_an / 10) * self.kgWoodperkgC * self.ForHHV

        self.TradFuelUse.values = np.maximum(
            0.0, 
            self.SE_Trad[t].sel(sector=2, ntrad=2, region=slice(0,25)) - fuel_wood_harvest
        )

        self.AgrTechPotREn.values = np.maximum(
            0.0,
            (self.AgrEcoPotR * (1 - self.MoistureContent)) - self.FeedDemandCrop
        ) * self.CropHHV

        agr_proc_pot = self.AgrProcPotEn.values[region_mask, :]
        agr_proc_pot_reg = np.zeros((len(Region26.coords), (len(Crop15.coords))), dtype=np.float32)
        np.add.at(agr_proc_pot_reg, region_cells[region_mask], agr_proc_pot)
        self.AgrProcPotREn.values = agr_proc_pot_reg
 
        self.ForEcoPotTotREn.values = np.bincount(
            region_cells[region_mask], 
            weights=self.ForEcoPotTotEn.values[region_mask], 
            minlength=26
        )

        self.ResiduesSupply.values[:, :15] = self.AgrTechPotREn.values + self.AgrProcPotREn.values
        self.ResiduesSupply.values[:, 15] = self.ForEcoPotTotREn.values + self.ForProcPot.values

        self.ResiduesTotal = self.ResiduesSupply.sum(Residue.label) - self.TradFuelUse
    
    def _calculate_regional_curves(
            self,
            time: prism.Time
    ):
        t = time.t
        region_cells, region_mask = region_to_grid("5m")

        self.ResidueCost.values = np.where(
            (self.AgrEcoPotEn.sum(Crop15.label).values > EPS) | (self.ForEcoPotTotEn.values > EPS),
            np.maximum(self.AgrCostTot.values, self.ForCostTot.values),
            0.0
        )

        self.ResCostCurvePot.values = self.ForEcoPotTotEn.values + self.AgrEcoPotEn.sum(Crop15.label).values

        self.ResCostCurvePotR.values = np.bincount(
            region_cells[region_mask], 
            weights=self.ResCostCurvePot.values[region_mask], 
            minlength=26
        )

        self.ResCostCurve1.values = calc_cost_curve(
            region_cells,
            len(Region26.coords),
            self.ResidueCost.values,
            self.ResCostCurvePot.values,
            9
        )

        self.ResCostCurve2.values = self.ResCostCurve1.values[:, ::-1]
        
        self.ResPotCurve1.values = np.repeat(
            self.ResCostCurvePotR.values[:, None] / len(RCC.coords),
            repeats=len(RCC.coords),
            axis=1
        )

        res_eco_potential = (
            self.AgrTechPotREn.sum(Crop15.label).values + self.ForEcoPotTotREn.values 
            - self.TradFuelUse.values
        )

        self.ResEconUse.values = np.maximum(0.0, self.ResCostCurvePotR.values - res_eco_potential)

        self.ResTechCat1.values = np.where(
            self.ResPotCurve1.values[:, 0] > 0.0,
            self.ResEconUse.values / self.ResPotCurve1.values[:, 0] + 1.0,
            1.0
        )

        self.ResTechCat.values = np.clip(
            np.floor(self.ResTechCat1), 
            1, 
            len(RCC.coords)
        ).astype(np.int16)

        self.ResPotCurve2.values = np.zeros_like(self.ResPotCurve1.values, dtype=np.float32)

        pc_list = np.arange(1, len(RCCPlus.coords))
        mask_full = pc_list[None, :] > self.ResTechCat.values[:, None]
        self.ResPotCurve2.values[mask_full] = self.ResPotCurve1.values[mask_full]

        frac_remaining = 1.0 - (self.ResTechCat1.values - self.ResTechCat.values)
        frac_remaining = np.clip(frac_remaining, 0.0, 1.0)

        mask_partial = pc_list[None, :] == self.ResTechCat.values[:, None]
        self.ResPotCurve2.values[mask_partial] = (
            self.ResPotCurve1.values * frac_remaining[:, None]
        )[mask_partial]

        ProcCat1 = (self.ResidueProcCost > self.ResCostCurve2.values).astype(np.int8)

        ProcCat2 = ProcCat1.sum(axis=1)[:, None]

        self.ResCostCurve.values = np.zeros((len(Region26.coords), len(RCCPlus.coords)), dtype=np.float32)

        pc0 = np.arange(len(RCCPlus.coords))[None, :]
        left_mask  = (pc0 < ProcCat2)[:, :len(RCC.coords)]                                   
        right_mask = pc0 > ProcCat2                                   
        eq_mask    = pc0 == ProcCat2

        self.ResCostCurve.values[:, :len(RCC.coords)][left_mask] = self.ResCostCurve2.values[left_mask]

        for bin in RCCPlus:
            self.ResCostCurve.values[:, bin] = np.where(
                right_mask[:, bin] & (bin > 0),
                self.ResCostCurve2.values[:, bin - 1],
                self.ResCostCurve.values[:, bin]
            )

        self.ResCostCurve.values[eq_mask] = self.ResidueProcCost

        self.ResPotCurve.values = np.zeros((len(Region26.coords), len(RCCPlus.coords)), dtype=np.float32)

        self.ResPotCurve.values[:, :len(RCC.coords)][left_mask] = self.ResPotCurve2.values[left_mask]

        proc_pot = self.AgrProcPotREn.sum(Crop15.label).values + self.ForProcPot.values

        self.ResPotCurve.values = np.where(
            eq_mask, 
            proc_pot[:, None], 
            self.ResPotCurve.values
        )
    
        for bin in RCCPlus: 
            self.ResPotCurve.values[:, bin] = np.where(
                right_mask[:, bin] & (bin > 0),
                self.ResPotCurve2.values[:, bin - 1],
                self.ResPotCurve.values[:, bin]
            )
        
        # for reg in Region26:
        #     for bin in RCCPlus: 
        #         print(f"ResCostCurveDM region {reg} feedstock cost bin {bin}: {np.nanmean(self.ResCostCurve.values[reg, bin])}")

    def _calculate_regional_mass_supply(
            self,
            time: prism.Time,
    ):

        t, dt, t_start = time.t, time.dt, time.start
        region_cells, region_mask = region_to_grid("5m")

        agr_eco_dm_reg = np.zeros((len(Region26.coords), (len(Crop15.coords))), dtype=np.float32)
        np.add.at(agr_eco_dm_reg, region_cells[region_mask], self.AgrEcoPotDry.values[region_mask, :])
        self.AgrEcoPotDryR.values = agr_eco_dm_reg

        agr_proc_dm_reg = np.zeros((len(Region26.coords), (len(Crop15.coords))), dtype=np.float32)
        np.add.at(agr_proc_dm_reg, region_cells[region_mask], self.AgrProcPotDry.values[region_mask, :])
        self.AgrProcPotDryR.values = agr_proc_dm_reg

        self.ForEcoPotTotDryR.values = np.bincount(
            region_cells[region_mask],
            weights= self.ForEcoPotTotDry.values[region_mask],
            minlength=len(Region26.coords)
        )

        agr_eco_share = np.where(
            self.AgrEcoPotDryR.values > EPS,
            self.AgrEcoPotDryR.values / self.AgrEcoPotDryR.values.sum(axis=1, keepdims=True),
            0.0
        )

        self.FeedDemandCrop.values = (
            self.FeedDemand_in[t]
                .sel(nfpt=2, ngst=2, nat=5, region=slice(0, 25))
                .values[:, None]
            * 1000.0
            * agr_eco_share
        )

        fuel_harvest_an = np.zeros(len(Region26.coords), dtype=np.float32)
        for k in range(1, 11):
            tt = max(t - k * dt, t_start)
            fuel_harvest_an += self.WoodHarvest_in[tt].sel(
                nfhtt=4, nwct=2, region=slice(0, 25)
            ).values

        fuel_wood_harvest_en = (fuel_harvest_an / 10.0) * self.kgWoodperkgC * self.ForHHV

        self.TradFuelUse.values = np.maximum(
            0.0,
            self.SE_Trad[t].sel(sector=2, ntrad=2, region=slice(0, 25)) - fuel_wood_harvest_en
        )

        self.TradFuelUseDry.values = self.TradFuelUse.values / self.ForHHV

        self.AgrTechPotDryR.values = np.maximum(
            0.0,
            self.AgrEcoPotDryR.values - self.FeedDemandCrop.values
        )

        self.ResiduesSupplyDM.values[:, :15] = (
            self.AgrTechPotDryR.values + self.AgrProcPotDryR.values
        )

        self.ResiduesSupplyDM.values[:, 15] = np.maximum(
            0.0,
            self.ForEcoPotTotDryR.values + self.ForProcPotDryR.values - self.TradFuelUseDry.values
        )

        # Estimate grid-level residue supply ignoring traditional and livestock uses
        self.ResiduePotDM.values[:, :15] = self.AgrEcoPotDry.values + self.AgrProcPotDry.values
        self.ResiduePotDM.values[:, 15] = self.ForEcoPotTotDry.values

    def _calculate_regional_type_curves(
            self,
            time: prism.Time
    ):
        t = time.t
        region_cells, region_mask = region_to_grid("5m")

        self.ResidueCostDM.values[:, :15] = np.where(
            self.ResiduePotDM.values[:, :15] > EPS,
            self.AgrCostCropTot.values * self.CropHHV.values[None, :],
            0.0
        ).astype(np.float32)

        self.ResidueCostDM.values[:, 15] = np.where(
            self.ResiduePotDM.values[:, 15] > EPS,
            self.ForCostTot.values * self.ForHHV,
            0.0
        ).astype(np.float32)

        self.ResCostCurvePotDM.values[:, :15] = self.AgrEcoPotDry.values.astype(np.float32)

        self.ResCostCurvePotDM.values[:, 15] = self.ForEcoPotTotDry.values

        for r in Residue:
            self.ResCostCurvePotRDM.values[:, r] = np.bincount(
                region_cells[region_mask],
                weights=self.ResCostCurvePotDM.values[region_mask, r],
                minlength=len(Region26.coords)
            ).astype(np.float32)

        for r in Residue:
            self.ResCostCurve1DM.values[:, r, :] = calc_cost_curve(
                region_cells,
                len(Region26.coords),
                self.ResidueCostDM.values[:, r],
                self.ResCostCurvePotDM.values[:, r],
                len(RCC.coords)
            )

        self.ResCostCurve2DM.values = self.ResCostCurve1DM.values[:, :, ::-1]

        self.ResPotCurve1DM.values = np.repeat(
            self.ResCostCurvePotRDM.values[:, :, None] / len(RCC.coords),
            repeats=len(RCC.coords),
            axis=2
        ).astype(np.float32)

        self.ResEconUseDM.values[:, :15] = np.minimum(
            self.ResCostCurvePotRDM.values[:, :15],
            np.maximum(
                0.0,
                self.AgrEcoPotDryR.values - self.AgrTechPotDryR.values
            )
        ).astype(np.float32)

        self.ResEconUseDM.values[:, 15] = np.minimum(
            self.ResCostCurvePotRDM.values[:, 15],
            self.TradFuelUseDry.values
        ).astype(np.float32)

        self.ResTechCat1DM.values = np.where(
            self.ResPotCurve1DM.values[:, :, 0] > 0.0,
            self.ResEconUseDM.values / self.ResPotCurve1DM.values[:, :, 0] + 1.0,
            1.0
        )


        self.ResTechCatDM.values = np.clip(
            np.floor(self.ResTechCat1DM.values),
            1,
            len(RCC.coords)
        ).astype(np.int16)

        pc_list = np.arange(1, len(RCCPlus.coords))

        for r in Residue:
            mask_full = pc_list[None, :] > self.ResTechCatDM.values[:, r][:, None]
            self.ResPotCurve2DM.values[:, r, :][mask_full] = self.ResPotCurve1DM.values[:, r, :][mask_full]

            frac_remaining = 1.0 - (
                self.ResTechCat1DM.values[:, r] - self.ResTechCatDM.values[:, r]
            )
            frac_remaining = np.clip(frac_remaining, 0.0, 1.0)

            mask_partial = pc_list[None, :] == self.ResTechCatDM.values[:, r][:, None]
            self.ResPotCurve2DM.values[:, r, :][mask_partial] = (
                self.ResPotCurve1DM.values[:, r, :] * frac_remaining[:, None]
            )[mask_partial]

        proc_pot_dm = np.zeros((len(Region26.coords), len(Residue.coords)), dtype=np.float32)
        proc_pot_dm[:, :15] = self.AgrProcPotDryR.values
        proc_pot_dm[:, 15] = self.ForProcPotDryR.values

        proc_cost_dm = np.zeros(len(Residue.coords), dtype=np.float32)
        proc_cost_dm[:15] = self.ResidueProcCost * self.CropHHV
        proc_cost_dm[15] = self.ResidueProcCost * self.ForHHV

        pc0 = np.arange(len(RCCPlus.coords))[None, :]

        for r in Residue:
            proc_cat1 = (proc_cost_dm[r] > self.ResCostCurve2DM.values[:, r, :]).astype(np.int8)
            proc_cat2 = proc_cat1.sum(axis=1)[:, None]

            left_mask = (pc0 < proc_cat2)[:, :len(RCC.coords)]
            right_mask = pc0 > proc_cat2
            eq_mask = pc0 == proc_cat2

            self.ResCostCurveDM[t].values[:, r, :len(RCC.coords)][left_mask] = self.ResCostCurve2DM.values[:, r, :][left_mask]

            for b in RCCPlus:
                if b > 0:
                    self.ResCostCurveDM[t].values[:, r, b] = np.where(
                        right_mask[:, b],
                        self.ResCostCurve2DM.values[:, r, b - 1],
                        self.ResCostCurveDM[t].values[:, r, b]
                    )

            self.ResCostCurveDM[t].values[:, r, :][eq_mask] = proc_cost_dm[r]

            self.ResPotCurveDM[t].values[:, r, :len(RCC.coords)][left_mask] = self.ResPotCurve2DM.values[:, r, :][left_mask]

            self.ResPotCurveDM[t].values[:, r, :] = np.where(
                eq_mask, 
                proc_pot_dm[:, r][:, None], 
                self.ResPotCurveDM[t].values[:, r, :] 
            )

            for b in RCCPlus:
                if b > 0:
                    self.ResPotCurveDM[t].values[:, r, b] = np.where(
                        right_mask[:, b],
                        self.ResPotCurve2DM.values[:, r, b - 1],
                        self.ResPotCurveDM[t].values[:, r, b]
                    )

    def _calculate_timer_exogenous_residue_curve(
        self,
        time: prism.Time,
        ExoResidues: prism.TimeVariable[Region27, np.float32],
    ):

        t = time.t

        shared_potential_gj = np.asarray(
            self.ResPotCurve.values[:len(Region26.coords), :len(RCCPlus.coords)],
            dtype=np.float64,
        )

        shared_potential_gj = np.nan_to_num(
            shared_potential_gj,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        shared_potential_gj = np.maximum(
            shared_potential_gj,
            0.0,
        )

        res_cost_curve = np.asarray(
            self.ResCostCurve.values[:len(Region26.coords), :len(RCCPlus.coords)],
            dtype=np.float64,
        )

        exo_residues_gj = np.asarray(
            ExoResidues.values[:len(Region26.coords)],
            dtype=np.float64,
        )

        exo_residues_gj = np.nan_to_num(
            exo_residues_gj,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        exo_residues_gj = np.maximum(
            exo_residues_gj,
            0.0,
        )

        cat_less_10 = np.isfinite(res_cost_curve) & (res_cost_curve < 10.0)

        num_cat_less_10 = np.sum(
            cat_less_10,
            axis=1,
            dtype=np.int16,
        )

        exo_res_pc = np.divide(
            exo_residues_gj,
            num_cat_less_10,
            out=np.zeros(
                len(Region26.coords),
                dtype=np.float64,
            ),
            where=num_cat_less_10 > 0,
        )

        bin_index = np.arange(
            len(RCCPlus.coords),
            dtype=np.int16,
        )[None, :]

        receives_exogenous = (
            bin_index
            < num_cat_less_10[:, None]
        )

        exo_curve_gj = np.where(
            receives_exogenous,
            exo_res_pc[:, None],
            0.0,
        )

        timer_total_potential_gj = (
            shared_potential_gj
            + exo_curve_gj
        )

        self.ExoResiduesPotCurve[t] = (
            exo_curve_gj.astype(np.float32)
        )

        self.TimerResiduesPotCurve[t] = (
            timer_total_potential_gj.astype(np.float32)
        )


        expected_low_cost_mask = receives_exogenous

        if np.any(
            cat_less_10 != expected_low_cost_mask
        ):
            n_bad = int(
                np.sum(
                    np.any(
                        cat_less_10
                        != expected_low_cost_mask,
                        axis=1,
                    )
                )
            )

            warnings.warn(
                f"{int(t.magnitude)}: {n_bad} regions have "
                "non-contiguous <10 $/GJ residue bins. "
                "The exogenous residue allocation follows the "
                "original TIMER PC <= NumCatLess10 rule."
            )

        allocated_exo_gj = np.sum(
            exo_curve_gj,
            axis=1,
        )

        no_low_cost_bin = (
            num_cat_less_10 == 0
        )

        if np.any(
            no_low_cost_bin
            & (exo_residues_gj > EPS)
        ):
            bad_regions = np.flatnonzero(
                no_low_cost_bin
                & (exo_residues_gj > EPS)
            )

            warnings.warn(
                f"{int(t.magnitude)}: ExoResidues > 0 but no "
                f"<10 $/GJ residue bin in regions "
                f"{bad_regions.tolist()}. "
                "Following TIMER, their exogenous supply is not "
                "added to the residue curve."
            )

        # Numerical audit
        valid_alloc = ~no_low_cost_bin

        if np.any(valid_alloc):
            diff = (
                allocated_exo_gj[valid_alloc]
                - exo_residues_gj[valid_alloc]
            )

            max_abs_diff = float(
                np.max(np.abs(diff))
            )

            tolerance = max(
                1.0,
                1.0e-6
                * float(
                    np.max(
                        exo_residues_gj[valid_alloc]
                    )
                ),
            )

            if max_abs_diff > tolerance:
                warnings.warn(
                    f"{int(t.magnitude)}: exogenous residue "
                    f"curve fails regional conservation check; "
                    f"max difference = {max_abs_diff:.6g} GJ/yr."
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
                    prism_var=self.ResPotCurveDM,
                    filename="ResPotCurveDM.nc",
                    var_name="ResPotCurveDM",
                    dims=("region", "res", "bin"),
                    coords={
                        "region": np.array(Region26.coords),
                        "res": np.array(Residue.coords),
                        "bin": np.array(RCCPlus.coords),
                    },
                    times=times,
                    units="",
                    description="",
                )
                export_variable_to_netcdf(
                    prism_var=self.ResCostCurveDM,
                    filename="ResCostCurveDM.nc",
                    var_name="ResCostCurveDM",
                    dims=("region", "res", "bin"),
                    coords={
                        "region": np.array(Region26.coords),
                        "res": np.array(Residue.coords),
                        "bin": np.array(RCCPlus.coords),
                    },
                    times=times,
                    units="",
                    description="",
                )
                # export_variable_to_netcdf(
                #     prism_var=self.ExoResiduesPotCurve,
                #     filename="ExoResiduesPotCurve.nc",
                #     var_name="ExoResiduesPotCurve",
                #     dims=("region","bin"),
                #     coords={
                #         "region": np.array(Region26.coords),
                #         "bin": np.array(RCCPlus.coords),
                #     },
                #     times=times,
                #     units="",
                #     description="",
                # )
                # export_variable_to_netcdf(
                #     prism_var=self.TimerResiduesPotCurve,
                #     filename="TimerResiduesPotCurve.nc",
                #     var_name="TimerResiduesPotCurve",
                #     dims=("region","bin"),
                #     coords={
                #         "region": np.array(Region26.coords),
                #         "bin": np.array(RCCPlus.coords),
                #     },
                #     times=times,
                #     units="",
                #     description="",
                # )

if __name__ == "__main__":
    print("Running standalone residues model")

    simulation_timeline = prism.Timeline(start=prism.Q_(2040, 'year'), end=prism.Q_(2050, 'year'), stepsize=prism.Q_(5, 'year'))

    def input_fun(t):
        return dict() 

    residues = Residues(simulation_timeline)
    residues.simulate(simulation_timeline, inputs=input_fun)

    
import warnings
from pathlib import Path
import numpy as np
import prism
import xarray as xr
from pint.errors import UnitStrippedWarning
from timer.globals.constants import EPS
from timer.cdr.cdr_util import (
    region_to_grid,
    safe_division,
    export_variable_to_netcdf,
    biocrops_data_dir,
    LandGrid,
    Region26, Region27, Region28,
    LandCover, LandCoverPlus, Biomes,
    Crop15, Crop17, Crop22, Crop26, Crop38,
    BioenergyCrop3, BioenergyCrop5,
    NFCAREAT, DiscCat,
)


warnings.simplefilter("ignore", UnitStrippedWarning)
warnings.filterwarnings(
    "ignore",
    message="The specified chunks separate the stored chunks",
    category=UserWarning,
)
warnings.filterwarnings(
    "error",
    message="invalid value encountered in cast",
    category=RuntimeWarning,
)


@prism.interface
class BioCrop(prism.Model):
    """Spatial bioenergy-crop potential and provisional NPV module."""

    # Economic / implementation parameters
    Deflator: prism.Scalar = 1.12
    BioFuelEconomicLifetime: prism.Scalar = 30
    BioLandNPVHorizon: prism.Scalar = 30
    ImplFracCrop: prism.Scalar = 0.75
    ImplFracPasture: prism.Scalar = 0.25

    CropYield_Cor: prism.Array[LandGrid, Crop26, np.float32] = prism.export()
    CropFrac: prism.Array[LandGrid, Crop15, np.float32] = prism.export()
    CropProd: prism.Array[LandGrid, Crop26, np.float32] = prism.export()
    BioNPVByLand: prism.TimeVariable[LandGrid, BioenergyCrop5, LandCover, np.float32] = prism.export()
    BioNPV: prism.TimeVariable[LandGrid, BioenergyCrop5, np.float32] = prism.export()

    MCBiomass: prism.Array[BioenergyCrop5, np.float32] = prism.private(
        file=biocrops_data_dir / "biomass_moisture_content.dat"
    )
    HHVBiomass: prism.Array[BioenergyCrop5, np.float32] = prism.private(
        file=biocrops_data_dir / "biomass_HHV.dat"
    )
    FracHarv2: prism.Array[NFCAREAT, Crop17, Region27, np.float32] = prism.private(
        file=biocrops_data_dir / "FRHARVMAPSPAM.scn"
    )
    ImplFrac: prism.Array[Biomes, np.float32] = prism.private(
        file=biocrops_data_dir / "impl_frac.dat"
    )
    DiscRateWACC: prism.Array[Region27, DiscCat, np.float32] = prism.private(
        file=biocrops_data_dir / "bioenergy_wacc.scn"
    )
    LabCD_ini: prism.Array[Region27, BioenergyCrop5, np.float32] = prism.private(
        file=biocrops_data_dir / "LabCD_ini.out"
    )
    CapCD_ini_in: prism.Array[Region27, BioenergyCrop5, np.float32] = prism.private(
        file=biocrops_data_dir / "Cap_ini.dat"
    )
    FeedstockCostTransp_in: prism.Array[BioenergyCrop5, np.float32] = prism.private(
        file=biocrops_data_dir / "FeedstockTransportCosts.dat"
    )
    PartialDerivativeWACC: prism.Array[Region27, np.float32] = prism.private(
        file=biocrops_data_dir / "bioenergy_wacc_psi.scn"
    )
    CobbDouglasCoeff: prism.Array[BioenergyCrop5, np.float32] = prism.private(
        file=biocrops_data_dir / "CDcoeff.dat"
    )
    FeedstockN2OEmission: prism.Array[Region27, BioenergyCrop5, np.float32] = prism.private(
        file=biocrops_data_dir / "N2OemissFeedstock.dat"
    )
    alt5min: prism.Array[LandGrid, np.float32] = prism.private(
        file=biocrops_data_dir / "alt5min.out"
    )

    region_cells: prism.Array[LandGrid, np.int16] = prism.private()
    region_mask: prism.Array[LandGrid] = prism.private()
    CropFrac_Cor: prism.Array[LandGrid, Crop26, np.float32] = prism.private()
    MF_AgrCor: prism.Array[LandGrid, Crop15, np.float32] = prism.private()
    FracHarv1: prism.Array[LandGrid, np.float32] = prism.private()
    FracHarv2_Map: prism.Array[LandGrid, Crop38, np.float32] = prism.private()
    FracHarv2_MapCor: prism.Array[LandGrid, Crop26, np.float32] = prism.private()
    BFCropProd: prism.Array[LandGrid, Crop26, np.float32] = prism.private()

    Slope15Frac: prism.Array[LandGrid, np.float32] = prism.private()
    CropLandFrac: prism.Array[LandGrid, np.float32] = prism.private()
    PastureFrac: prism.Array[LandGrid, np.float32] = prism.private()
    ForestFrac: prism.Array[LandGrid, np.float32] = prism.private()
    OtherLandFrac: prism.Array[LandGrid, np.float32] = prism.private()
    UrbanFrac: prism.Array[LandGrid, np.float32] = prism.private()
    BioReserve: prism.Array[LandGrid, np.float32] = prism.private()
    BFCellFrac: prism.Array[LandGrid, np.float32] = prism.private()
    FoodCropLandFrac: prism.Array[LandGrid, np.float32] = prism.private()
    BFLandUse: prism.Array[LandGrid, Crop26, np.float32] = prism.private()
    BiomassYield: prism.TimeVariable[LandGrid, BioenergyCrop5, np.float32] = prism.private()
    BiomassProductivity: prism.Array[LandGrid, BioenergyCrop5, np.float32] = prism.private()
    BiomassPotential: prism.Array[LandGrid, BioenergyCrop5, np.float32] = prism.private()
    BioPotFrac: prism.Array[LandGrid, BioenergyCrop5, np.float32] = prism.private()
    BioPotArea: prism.Array[LandGrid, BioenergyCrop5, np.float32] = prism.private()
    ReferenceLandFrac: prism.Array[LandGrid, LandCover, np.float32] = prism.private()
    BioPotFracByLand: prism.Array[LandGrid, BioenergyCrop5, LandCover, np.float32] = prism.private()
    BioPotAreaByLand: prism.Array[LandGrid, BioenergyCrop5, LandCover, np.float32] = prism.private()

    FeedstockCostTransp: prism.Array[BioenergyCrop5, np.float32] = prism.private()
    CapCD_ini: prism.Array[Region26, BioenergyCrop5, np.float32] = prism.private()
    BioLabourCost: prism.Array[Region26, np.float32] = prism.private()
    FeedstockEmission: prism.Array[Region26, BioenergyCrop5, np.float32] = prism.private()
    DiscRateBioProdWeighted: prism.Array[Region26, np.float32] = prism.private()
    RegionalCreditRisk: prism.Array[Region26, np.float32] = prism.private()
    DiscRateBio: prism.Array[Region26, np.float32] = prism.private()
    BioAnnuity: prism.Array[Region26, np.float32] = prism.private()
    Bio_Pl_Pk: prism.Array[Region26, np.float32] = prism.private()
    Bio_Pl_PKCDcoeff: prism.Array[Region26, BioenergyCrop5, np.float32] = prism.private()
    CapitalRequired: prism.Array[Region26, BioenergyCrop5, np.float32] = prism.private()
    LaborRequired: prism.Array[Region26, BioenergyCrop5, np.float32] = prism.private()
    CapitalCostBiom: prism.Array[Region26, BioenergyCrop5, np.float32] = prism.private()
    LaborCostBiom: prism.Array[Region26, BioenergyCrop5, np.float32] = prism.private()
    EmissionCostBiom: prism.Array[Region26, BioenergyCrop5, np.float32] = prism.private()
    BioFeedstockValue: prism.TimeVariable[LandGrid, BioenergyCrop5, np.float32] = prism.private()
    BioAnnualRevenue: prism.TimeVariable[LandGrid, BioenergyCrop5, np.float32] = prism.private()
    BioAnnualNonLandCost: prism.TimeVariable[LandGrid, BioenergyCrop5, np.float32] = prism.private()

    def compute_initial_values(self, _: prism.Timeline):
        self.region_cells.values, self.region_mask.values = region_to_grid("5m")

        slope_path = Path(
            r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\data\biomass\SlopeGT15Frac_complete.NC"
        )
        with xr.open_dataset(slope_path, engine="netcdf4") as ds:
            slope = np.asarray(ds["SlopeGT15Frac"].values, dtype=np.float32).reshape(-1)

        if slope.size != len(LandGrid.coords):
            raise ValueError(
                f"{slope_path} contains {slope.size} slope values; "
                f"expected {len(LandGrid.coords)} LandGrid cells."
            )
        self.Slope15Frac.values = slope

    def compute_values(
        self,
        time: prism.Time,
        GAREA5m: prism.Array[LandGrid, np.float32],
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GRPPC: prism.Array[LandGrid, Crop38, np.float32],
        # GLANDCOVERFrac_REF: prism.Array[LandGrid, LandCover, np.float32],
        #  GLANDCOVERFracDetail_REF: prism.Array[LandGrid, LandCoverPlus, np.float32],
        # GNLCT_REF: prism.Array[LandGrid, np.int16],
        MF_Agr: prism.Array[Crop22, Region26, np.float32],
        FracHarv_in: prism.Array[Region26, np.float32],
        BFProdShare: prism.Array[BioenergyCrop3, Region27, np.float32],
        #CCTransport: prism.Array[Region27, np.float32],
        #FeedstockEnergyUse: prism.Array[Region27, BioenergyCrop5, np.float32],
        #Interest: prism.Array[Region27, np.float32],
        #BioRiskFactor: prism.Array[Region27, np.float32],
        #ctax: prism.Array[Region28, np.float32],
        MF: prism.Array[Region27, BioenergyCrop5, np.float32],
        #GDPpc: prism.Array[Region28, np.float32],
        #GDPpc_tscen: prism.Array[Region28, np.float32],
        #CropLandRent: prism.Array[LandGrid, np.float32],
        #PastureLandRent: prism.Array[LandGrid, np.float32],
        #BioConversionNetback: prism.Array[Region27, BioenergyCrop5, np.float32],
    ):
        self._calculate_crop_fractions(time=time, GFRAC=GFRAC)
        self._calculate_management_factors(
            time=time, GFRAC=GFRAC, MF_Agr=MF_Agr, MF=MF
        )
        self._calculate_crop_yields(time=time, GFRAC=GFRAC, GRPPC=GRPPC)
        self._calculate_crop_harvest(GFRAC=GFRAC)
        self._calculate_biomass_prod(
            time=time,
            GAREA5m=GAREA5m,
            FracHarv_in=FracHarv_in,
            BFProdShare=BFProdShare,
        )
        # self._calculate_available_land(
        #     time=time,
        #     GAREA5m=GAREA5m,
        #     GLANDCOVERFrac_REF=GLANDCOVERFrac_REF,
        #     GLANDCOVERFracDetail_REF=GLANDCOVERFracDetail_REF,
        #     GNLCT_REF=GNLCT_REF,
        #     GRPPC=GRPPC,
        # )
        # self._calculate_biomass_costs(
        #     time=time,
        #     CCTransport=CCTransport,
        #     FeedstockEnergyUse=FeedstockEnergyUse,
        #     Interest=Interest,
        #     BioRiskFactor=BioRiskFactor,
        #     ctax=ctax,
        #     GDPpc=GDPpc,
        #     GDPpc_tscen=GDPpc_tscen,
        #     CropLandRent=CropLandRent,
        #     PastureLandRent=PastureLandRent,
        #     BioConversionNetback=BioConversionNetback,
        # )
        self._write_results_to_files(time=time)

    @staticmethod
    def crop_mapping():
        mapping = {
            0: {
                "name": "Temperate cereals (rainfed)",
                "frac_rule": {"indices": [1, 5]},
                "mf_rule": {"numerator_terms": [(1, [1, 22]), (5, [5, 27])]},
                "yield_rule": {"grp_idx": [1, 5], "frac_idx": [1, 5], "mf_idx": 0},
                "harvest_rule": [(1, 1), (5, 5)],
            },
            1: { 
                "name": "Rice (rainfed)",
                "frac_rule": {"indices": [2]},
                "mf_rule": {"direct_idx": 2},
                "yield_rule": {"grp_idx": [2], "frac_idx": [2], "mf_idx": 1},
                "harvest_rule": [(2, 2)],
            },
            2: {  
                "name": "Maize (rainfed)",
                "frac_rule": {"indices": [3]},
                "mf_rule": {"direct_idx": 3},
                "yield_rule": {"grp_idx": [3], "frac_idx": [3], "mf_idx": 2},
                "harvest_rule": [(3, 3)],
                "bf_rule": {"bf_idx": 1},
            },
            3: { 
                "name": "Tropical cereals (rainfed)",
                "frac_rule": {"indices": [4]},
                "mf_rule": {"direct_idx": 4},
                "yield_rule": {"grp_idx": [4], "frac_idx": [4], "mf_idx": 3},
                "harvest_rule": [(4, 4)],
            },
            4: { 
                "name": "Pulses (rainfed)",
                "frac_rule": {"indices": [6]},
                "mf_rule": {"direct_idx": 6},
                "yield_rule": {"grp_idx": [6], "frac_idx": [6], "mf_idx": 4},
                "harvest_rule": [(6, 6)],
            },
            5: {
                "name": "Roots & tubers (rainfed)",
                "frac_rule": {"indices": [10, 11]},
                "mf_rule": {"numerator_terms": [(10, [10, 31]), (11, [11, 32])]},
                "yield_rule": {"grp_idx": [10, 11], "frac_idx": [10, 11], "mf_idx": 5},
                "harvest_rule": [(10, 10), (11, 11)],
            },
            6: { 
                "name": "Oil crops (rainfed)",
                "frac_rule": {"indices": [7, 8, 9, 13]},
                "mf_rule": {"numerator_terms": [(7, [7, 28]), (8, [8, 29]), (9, [9, 30]), (13, [13, 34])]},
                "yield_rule": {"grp_idx": [7, 8, 9, 13], "frac_idx": [7, 8, 9, 13], "mf_idx": 6},
                "harvest_rule": [(7, 7), (8, 8), (9, 9), (13, 13)],
                "bf_rule": {"bf_idx": 0},
            },
            7: { 
                "name": "Fibre (rainfed)",
                "frac_rule": {"indices": [16]},
                "mf_rule": {"direct_idx": 16},
                "yield_rule": {"grp_idx": [16], "frac_idx": [16], "mf_idx": 7},
                "harvest_rule": [(16, 16)],
            },
            8: {
                "name": "Fruit (rainfed)",
                "frac_rule": {"indices": [14]},
                "mf_rule": {"direct_idx": 14},
                "yield_rule": {"grp_idx": [14], "frac_idx": [8], "mf_idx": 8},
                "harvest_rule": [(14, None)], 
            },
            9: { 
                "name": "Vegetables (rainfed)",
                "frac_rule": {"indices": [14]},
                "mf_rule": {"direct_idx": 14},
                "yield_rule": {"grp_idx": [14], "frac_idx": [9], "mf_idx": 9},
                "harvest_rule": [(14, None)],
            },
            10: {  
                "name": "Sugar crops (rainfed)",
                "frac_rule": {"indices": [12]},
                "mf_rule": {"direct_idx": 12},
                "yield_rule": {"grp_idx": [12], "frac_idx": [12], "mf_idx": 10},
                "harvest_rule": [(12, 12)],
                "bf_rule": {"bf_idx": 2},  
            },
            11: { 
                "name": "Biofuel sugarcane",
                "frac_rule": {"indices": [19]},
                "mf_rule": {"mf_idx": 2},
                "yield_rule": {"grp_idx": [19], "frac_idx": [19], "mf_idx": 11},
                "harvest_rule": [(19, 19)],
            },
            12: { 
                "name": "Biofuel maize",
                "frac_rule": {"indices": [17]},
                "mf_rule": {"mf_idx": 1},
                "yield_rule": {"grp_idx": [17], "frac_idx": [17], "mf_idx": 12},
                "harvest_rule": [(17, 17)],
            },
            13: {  
                "name": "Biofuel woody",
                "frac_rule": {"indices": [20]},
                "mf_rule": {"mf_idx": 0},
                "yield_rule": {"grp_idx": [20], "frac_idx": [20], "mf_idx": 13},
                "harvest_rule": [(20, 20)],
            },
            14: {  
                "name": "Biofuel non-woody",
                "frac_rule": {"indices": [21]},
                "mf_rule": {"mf_idx": 4},
                "yield_rule": {"grp_idx": [21], "frac_idx": [21], "mf_idx": 14},
                "harvest_rule": [(21, 21)],
            },
            15: { 
                "name": "Temperate cereals (irrigated)",
                "frac_rule": {"indices": [22, 26]},
                "yield_rule": {"grp_idx": [22, 26], "frac_idx": [22, 26], "mf_idx": 0},
                "harvest_rule": [(22, 22), (26, 26)],
            },
            16: { 
                "name": "Rice (irrigated)",
                "frac_rule": {"indices": [23]},
                "yield_rule": {"grp_idx": [23], "frac_idx": [23], "mf_idx": 1},
                "harvest_rule": [(23, 23)],
            },
            17: {  
                "name": "Maize (irrigated)",
                "frac_rule": {"indices": [24]},
                "yield_rule": {"grp_idx": [24], "frac_idx": [24], "mf_idx": 2},
                "harvest_rule": [(24, 24)],
                "bf_rule": {"bf_idx": 1},
            },
            18: {  
                "name": "Tropical cereals (irrigated)",
                "frac_rule": {"indices": [25]},
                "yield_rule": {"grp_idx": [25], "frac_idx": [25], "mf_idx": 3},
                "harvest_rule": [(25, 25)],
            },
            19: { 
                "name": "Pulses (irrigated)",
                "frac_rule": {"indices": [27]},
                "yield_rule": {"grp_idx": [27], "frac_idx": [27], "mf_idx": 4},
                "harvest_rule": [(27, 27)],
            },
            20: { 
                "name": "Roots & tubers (irrigated)",
                "frac_rule": {"indices": [31, 32]},
                "yield_rule": {"grp_idx": [31, 32], "frac_idx": [31, 32], "mf_idx": 5},
                "harvest_rule": [(31, 31), (32, 32)],
            },
            21: { 
                "name": "Oil crops (irrigated)",
                "frac_rule": {"indices": [28, 29, 30, 34]},
                "yield_rule": {"grp_idx": [28, 29, 30, 34], "frac_idx": [28, 29, 30, 34], "mf_idx": 6},
                "harvest_rule": [(28, 28), (29, 29), (30, 30), (34, 34)],
                "bf_rule": {"bf_idx": 0},
            },
            22: { 
                "name": "Fibre (irrigated)",
                "frac_rule": {"indices": [37]},
                "yield_rule": {"grp_idx": [37], "frac_idx": [37], "mf_idx": 7},
                "harvest_rule": [(37, 37)],
            },
            23: {  
                "name": "Fruit (irrigated)",
                "frac_rule": {"indices": [35]},
                "yield_rule": {"grp_idx": [35], "frac_idx": [23], "mf_idx": 8},
                "harvest_rule": [(35, None)],
            },
            24: { 
                "name": "Vegetables (irrigated)",
                "frac_rule": {"indices": [35]},
                "yield_rule": {"grp_idx": [35], "frac_idx": [24], "mf_idx": 9},
                "harvest_rule": [(35, None)],
            },
            25: { 
                "name": "Sugar cane (irrigated)",
                "frac_rule": {"indices": [33]},
                "yield_rule": {"grp_idx": [33], "frac_idx": [33], "mf_idx": 10},
                "harvest_rule": [(33, 33)],
                "bf_rule": {"bf_idx": 2},
            },
        }
        return mapping
        
        
    def _calculate_crop_fractions(
            self, 
            time: prism.Time,
            GFRAC: prism.Array[LandGrid, Crop38, np.float32]
    ):
        t = time.t
        crop_mapping = self.crop_mapping()
        print(f"Start biocrops calculations for year {t.magnitude}")

        crop_data = GFRAC.values
        for crop26, rule in crop_mapping.items():
            indices = rule["frac_rule"]["indices"]
            summed = crop_data[:, indices].sum(axis=1)
            if crop26 in {8, 9, 23, 24}:
                summed /= 2
            self.CropFrac_Cor.values[:, crop26] = summed

        for i in range(11): 
            self.CropFrac.values[:, i] = self.CropFrac_Cor.values[:, i] + self.CropFrac_Cor.values[:, i+15]
        for i in range(11, 15): 
            self.CropFrac.values[:, i] = self.CropFrac_Cor.values[:, i]


    def _calculate_management_factors(
        self, 
        time: prism.Time,
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        MF_Agr: prism.Array[Crop22, Region26, np.float32],
        MF: prism.Array[Region27,BioenergyCrop5, np.float32]
    ):
        
        t = time.t
        crop_mapping = self.crop_mapping()
        regions = self.region_cells.values
        is_region = self.region_mask.values

        MF_Agr_np = MF_Agr.values
        MF_np = MF.values
    
        for crop15, rule in crop_mapping.items():
            mf_rule = rule.get("mf_rule")
            if not mf_rule:
                continue

            if "numerator_terms" in mf_rule:
                num = sum(
                    MF_Agr_np[mf_idx, regions] * GFRAC.values[:, crop_idx].sum(axis=1)
                    for mf_idx, crop_idx in mf_rule["numerator_terms"]
                )
                den = self.CropFrac.values[:, crop15]
                self.MF_AgrCor.values[:, crop15] = safe_division(num, den, (den > EPS) & is_region)
            elif "direct_idx" in mf_rule:
                self.MF_AgrCor.values[:, crop15] = np.where(
                    is_region, MF_Agr_np[mf_rule["direct_idx"], regions], 0.0
                )
            elif "mf_idx" in mf_rule:
                if t.magnitude == 1970:
                    self.MF_AgrCor.values[:, crop15] = 0.0
                else:
                    self.MF_AgrCor.values[:, crop15] = np.where(
                        is_region, MF_np[regions, mf_rule["mf_idx"]], 0.0
                    )
            else:
                self.MF_AgrCor.values[:, crop15] = 0.0


    def _calculate_crop_yields(
        self,
        time: prism.Time,
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GRPPC: prism.Array[LandGrid, Crop38, np.float32]
    ):
        mapping = self.crop_mapping()

        for crop26, rules in mapping.items():
            yrule = rules.get("yield_rule", {})
            grp_idx = yrule.get("grp_idx", [])
            frac_idx = yrule.get("frac_idx", [])
            mf_idx   = yrule.get("mf_idx")

            den = self.CropFrac_Cor.values[:, crop26]

            num = sum(
                GRPPC.values[:, g] *
                (self.CropFrac_Cor.values[:, f] if crop26 in {8, 9, 23, 24}
                else GFRAC.values[:, f])
                for g, f in zip(grp_idx, frac_idx)
            )

            self.CropYield_Cor.values[:, crop26] = (
                safe_division(num, den, den > EPS)
                * self.MF_AgrCor.values[:, mf_idx]
            ).astype(np.float32, copy=False)
        

    def _calculate_crop_harvest(
        self,
        GFRAC: prism.Array[LandGrid, Crop38, np.float32]
    ):
        crop_mapping = self.crop_mapping()
        regions = self.region_cells.values

        self.FracHarv2_Map.values[:, 0] = 0.0
        for crop in range(16):
            self.FracHarv2_Map.values[:, crop+1] = self.FracHarv2.values[0, crop, regions]
        for crop in range(17, 22):
            self.FracHarv2_Map.values[:, crop] = 0.0
        for crop in range(22, 38):
            self.FracHarv2_Map.values[:, crop] = self.FracHarv2.values[1, crop - 22, regions]

        frac_in = GFRAC.values
        frac_cor = self.CropFrac_Cor.values
        frac_harv2 = self.FracHarv2_Map.values

        for crop26, rule in crop_mapping.items():
            harvest_rule = rule.get("harvest_rule")
            if not harvest_rule:
                continue

            den = frac_cor[:, crop26]
            num = 0.0
            for (frac_idx, crop_in_idx) in harvest_rule:
                if crop_in_idx is None:
                    num += frac_harv2[:, frac_idx] * frac_cor[:, crop26]
                else:
                    num += frac_harv2[:, frac_idx] * frac_in[:, crop_in_idx]

            self.FracHarv2_MapCor.values[:, crop26] = safe_division(num, den, den > EPS)


    def _calculate_biomass_prod(
            self,
            time: prism.Time,
            GAREA5m: prism.Array[LandGrid, np.float32],
            FracHarv_in: prism.Array[Region26, np.float32],
            BFProdShare: prism.Array[BioenergyCrop3, Region27, np.float32]
    ):
        crop_mapping = self.crop_mapping()
        regions = self.region_cells.values
        is_region = self.region_mask.values

        vals = np.zeros((len(LandGrid.coords)), dtype=np.float32)
        vals[is_region] = FracHarv_in.values[regions[is_region]]
        self.FracHarv1.values = vals

        for crop in Crop26:
            self.CropProd.values[:, crop] = (
                self.CropYield_Cor.values[:, crop]
                * self.CropFrac_Cor.values[:, crop]
                * self.FracHarv1.values
                * self.FracHarv2_MapCor.values[:, crop]
                * GAREA5m.values
            ).astype(np.float32, copy=False)

        self.BFCropProd.values = np.zeros((len(LandGrid.coords),len(Crop26.coords)), dtype=np.float32)
        for crop26, rule in crop_mapping.items():
            bf_rule = rule.get("bf_rule")
            if not bf_rule:
                continue

            self.BFCropProd.values[:, crop26] = (
                self.CropProd.values[:, crop26]
                * BFProdShare.values[bf_rule['bf_idx'], regions]
            ).astype(np.float32, copy=False)

    def _calculate_available_land(
            self,
            time: prism.Time,
            GAREA5m: prism.Array[LandGrid, np.float32],
            GLANDCOVERFrac_REF: prism.Array[LandGrid, LandCover, np.float32],
            GLANDCOVERFracDetail_REF: prism.Array[LandGrid, LandCoverPlus, np.float32],
            GNLCT_REF: prism.Array[LandGrid, np.int16],
            GRPPC: prism.Array[LandGrid, Crop38, np.float32],
    ):
        """
        Calculate crop-specific bioenergy land potential while retaining the
        reference land-use category from SSP2_M.

        Reference LandCover ordering:
            0 = cropland
            1 = pasture
            2 = forest
            3 = other land
            4 = urban

        BioPotFracByLand[g, c, l] therefore identifies how much of grid cell g
        can be used by bioenergy crop c from reference land class l.

        BioPotFrac[g, c] remains the total potential fraction summed over
        reference land classes for compatibility with the existing module.
        """

        t, dt, t_start = time.t, time.dt, time.start
        t_last = max(t - dt, t_start)

        n_grid = len(LandGrid.coords)
        n_biocrop = len(BioenergyCrop5.coords)
        n_landcover = len(LandCover.coords)

        # ------------------------------------------------------------------
        # 1. Reference land-cover fractions from SSP2_M
        # ------------------------------------------------------------------

        valid_area = (
            np.isfinite(GAREA5m.values)
            & (GAREA5m.values > EPS)
        )

        ref_lc_frac = np.divide(
            GLANDCOVERFrac_REF.values,
            GAREA5m.values[:, None],
            out=np.zeros_like(
                GLANDCOVERFrac_REF.values,
                dtype=np.float32,
            ),
            where=valid_area[:, None],
        )

        ref_lc_detail_frac = np.divide(
            GLANDCOVERFracDetail_REF.values,
            GAREA5m.values[:, None],
            out=np.zeros_like(
                GLANDCOVERFracDetail_REF.values,
                dtype=np.float32,
            ),
            where=valid_area[:, None],
        )

        crop_base = np.clip(ref_lc_frac[:, 0], 0.0, 1.0)
        pasture_base = np.clip(ref_lc_frac[:, 1], 0.0, 1.0)
        forest_base = np.clip(ref_lc_frac[:, 2], 0.0, 1.0)
        other_base = np.clip(ref_lc_frac[:, 3], 0.0, 1.0)

        # ------------------------------------------------------------------
        # 2. Urban land is already a separate GLANDCOVER category.
        #    Do not multiply the other categories by (1 - urban) again.
        # ------------------------------------------------------------------

        urban_raw = ref_lc_frac[:, 4]

        bad_urban = (
            valid_area
            & (
                ~np.isfinite(urban_raw)
                | (urban_raw < -1e-4)
                | (urban_raw > 1.0 + 1e-4)
            )
        )

        if np.any(bad_urban):
            print(
                f"Warning for {t.magnitude}: "
                f"{np.sum(bad_urban):,} cells have invalid "
                "reference UrbanFrac values. Clipping to [0, 1]."
            )

        urban = np.clip(
            np.nan_to_num(
                urban_raw,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        self.UrbanFrac.values = urban

        crop_ref = crop_base
        pasture_ref = pasture_base
        forest_ref = forest_base
        other_ref = other_base

        self.CropLandFrac.values = crop_ref
        self.PastureFrac.values = pasture_ref
        self.ForestFrac.values = forest_ref
        self.OtherLandFrac.values = other_ref

        # ------------------------------------------------------------------
        # 3. Exogenous protected areas
        # ------------------------------------------------------------------

        self.BioReserve.values = np.clip(
            np.nan_to_num(
                ref_lc_detail_frac[:, 5],
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        reserve_mask = self.BioReserve.values > 0.5

        # ------------------------------------------------------------------
        # 4. Existing dedicated bioenergy land
        #
        # This preserves the current implementation. If the novel allocator
        # should reallocate ALL bioenergy land from scratch, remove this block
        # and simply set FoodCropLandFrac = crop_ref.
        # ------------------------------------------------------------------

        valid_crop_yield = (
            np.isfinite(self.CropYield_Cor.values)
            & np.isfinite(self.BFCropProd.values)
            & (self.CropYield_Cor.values > EPS)
        )

        self.BFLandUse.values = np.divide(
            self.BFCropProd.values,
            self.CropYield_Cor.values,
            out=np.zeros_like(
                self.BFCropProd.values,
                dtype=np.float32,
            ),
            where=valid_crop_yield,
        )

        bf_cell_frac_raw = np.divide(
            self.BFLandUse.sum(Crop26.label).values,
            GAREA5m.values,
            out=np.zeros_like(
                GAREA5m.values,
                dtype=np.float32,
            ),
            where=valid_area,
        )

        bf_cell_frac_raw = np.clip(
            np.nan_to_num(
                bf_cell_frac_raw,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            0.0,
            1.0,
        )

        bf_exceeds_crop = bf_cell_frac_raw > crop_ref + EPS

        if np.any(bf_exceeds_crop):
            print(
                f"Warning for {t.magnitude}: "
                f"{np.sum(bf_exceeds_crop):,} cells have existing "
                "bioenergy area larger than reference non-urban cropland. "
                "Capping existing bioenergy by reference cropland."
            )

        self.BFCellFrac.values = np.minimum(
            bf_cell_frac_raw,
            crop_ref,
        ).astype(np.float32, copy=False)

        self.FoodCropLandFrac.values = np.maximum(
            0.0,
            crop_ref - self.BFCellFrac.values,
        ).astype(np.float32, copy=False)

        # Store the economically relevant reference source fractions.
        self.ReferenceLandFrac.values[:, :] = 0.0

        self.ReferenceLandFrac.values[:, 0] = (
            self.FoodCropLandFrac.values
        )
        self.ReferenceLandFrac.values[:, 1] = pasture_ref
        self.ReferenceLandFrac.values[:, 2] = forest_ref
        self.ReferenceLandFrac.values[:, 3] = other_ref

        # Urban remains zero by construction.

        # ------------------------------------------------------------------
        # 5. Biome feasibility and implementation fractions
        # ------------------------------------------------------------------

        valid_biome = (
            np.isfinite(GNLCT_REF.values)
            & (GNLCT_REF.values >= 1)
            & (GNLCT_REF.values <= len(Biomes.coords))
        )

        excluded_biome = np.isin(
            GNLCT_REF.values,
            np.asarray(
                [7, 8, 9, 16],  # ice, tundra, wooded tundra, hot desert
                dtype=GNLCT_REF.values.dtype,
            ),
        )

        biome_impl = np.zeros(
            n_grid,
            dtype=np.float32,
        )

        biome_impl[valid_biome] = self.ImplFrac.values[
            GNLCT_REF.values[valid_biome].astype(np.int64) - 1
        ]

        # Retain source land explicitly.
        source_land_impl = np.zeros(
            (n_grid, n_landcover),
            dtype=np.float32,
        )

        source_land_impl[:, 0] = (
            self.FoodCropLandFrac.values
            * self.ImplFracCrop
        )

        source_land_impl[:, 1] = (
            pasture_ref
            * self.ImplFracPasture
        )

        source_land_impl[:, 2] = (
            forest_ref
            * biome_impl
        )

        source_land_impl[:, 3] = (
            other_ref
            * biome_impl
        )

        source_land_impl = np.clip(
            source_land_impl,
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        # ------------------------------------------------------------------
        # 6. Slope constraint
        # ------------------------------------------------------------------

        bad_slope = (
            np.isfinite(self.Slope15Frac.values)
            & (
                (self.Slope15Frac.values < -1e-6)
                | (self.Slope15Frac.values > 1.0 + 1e-6)
            )
        )

        if np.any(bad_slope):
            print(
                f"Warning for {t.magnitude}: "
                f"{np.sum(bad_slope):,} Slope15Frac values "
                "lie outside 0-1. Clipping to [0, 1]."
            )

        missing_slope = (
            valid_area
            & ~np.isfinite(self.Slope15Frac.values)
        )

        if np.any(missing_slope):
            print(
                f"Warning for {t.magnitude}: "
                f"{np.sum(missing_slope):,} IMAGE land-grid cells "
                "have missing Slope15Frac. "
                "No slope reduction is applied to those cells."
            )

        slope15 = np.where(
            np.isfinite(self.Slope15Frac.values),
            np.clip(
                self.Slope15Frac.values,
                0.0,
                1.0,
            ),
            0.0,
        ).astype(np.float32, copy=False)

        source_land_available = (
            source_land_impl
            * (1.0 - slope15[:, None])
        )

        source_land_available = np.clip(
            source_land_available,
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        # ------------------------------------------------------------------
        # 7. Cell-level physical eligibility
        # ------------------------------------------------------------------

        altitude_ok = (
            np.isfinite(self.alt5min.values)
            & (self.alt5min.values <= 3000.0)
        )

        cell_eligible = (
            valid_area
            & ~reserve_mask
            & altitude_ok
            & valid_biome
            & ~excluded_biome
        )

        # ------------------------------------------------------------------
        # 8. Crop-specific dry-mass yields
        # ------------------------------------------------------------------

        mapping = {
            0: 20,  # woody
            1: 17,  # maize
            2: 19,  # sugar crop
            3: 18,  # oil crops
            4: 21,  # non-woody
        }

        for bio_crop, grp_idx in mapping.items():

            grp_yield = GRPPC.values[:, grp_idx]

            self.BiomassYield[t].values[:, bio_crop] = np.where(
                np.isfinite(grp_yield),
                (
                    grp_yield
                    * (1.0 - self.MCBiomass.values[bio_crop])
                ),
                0.0,
            ).astype(np.float32, copy=False)

        # ------------------------------------------------------------------
        # 9. Existing temporal smoothing for woody yield
        # ------------------------------------------------------------------

        biomass_yield_effective = (
            self.BiomassYield[t].values.copy()
        )

        wood_yield_sum = np.zeros(
            n_grid,
            dtype=np.float32,
        )

        for k in range(1, 5):

            tt = max(t - k * dt, t_start)

            wood_yield_sum += np.asarray(
                self.BiomassYield[tt].values[:, 0],
                dtype=np.float32,
            )

        wood_yield_prev_avg = wood_yield_sum / 4.0

        wood_spike = (
            np.isfinite(
                self.BiomassYield[t].values[:, 0]
            )
            & np.isfinite(
                self.BiomassYield[t_last].values[:, 0]
            )
            & (
                self.BiomassYield[t].values[:, 0]
                > 2.0
                * self.BiomassYield[t_last].values[:, 0]
            )
        )

        biomass_yield_effective[:, 0] = np.where(
            wood_spike,
            wood_yield_prev_avg,
            self.BiomassYield[t].values[:, 0],
        ).astype(np.float32, copy=False)

        # ------------------------------------------------------------------
        # 10. Minimum-yield suitability
        # ------------------------------------------------------------------

        min_yield_thresholds = np.full(
            n_biocrop,
            np.inf,
            dtype=np.float32,
        )

        for bio_crop in BioenergyCrop5:

            values = biomass_yield_effective[:, bio_crop]

            valid_for_global_max = (
                valid_area
                & np.isfinite(values)
                & (values > 0.0)
            )

            if np.any(valid_for_global_max):

                min_yield_thresholds[bio_crop] = (
                    0.10
                    * np.max(values[valid_for_global_max])
                )

        yield_ok = (
            np.isfinite(biomass_yield_effective)
            & (
                biomass_yield_effective
                >= min_yield_thresholds[None, :]
            )
        )

        # ------------------------------------------------------------------
        # 11. Crop × reference-land potential
        # ------------------------------------------------------------------

        pot_frac_by_land = np.where(
            (
                cell_eligible[:, None, None]
                & yield_ok[:, :, None]
            ),
            source_land_available[:, None, :],
            0.0,
        ).astype(np.float32, copy=False)

        self.BioPotFracByLand.values = pot_frac_by_land

        self.BioPotFrac.values = np.clip(
            np.sum(
                pot_frac_by_land,
                axis=2,
            ),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        self.BioPotAreaByLand.values = (
            self.BioPotFracByLand.values
            * GAREA5m.values[:, None, None]
        ).astype(np.float32, copy=False)

        self.BioPotArea.values = (
            self.BioPotFrac.values
            * GAREA5m.values[:, None]
        ).astype(np.float32, copy=False)

        # ------------------------------------------------------------------
        # 12. Primary biomass productivity, GJ / ha / yr
        # ------------------------------------------------------------------

        km2ha = 1.0 / 100.0

        raw_productivity = (
            biomass_yield_effective
            * self.HHVBiomass.values[None, :]
            * km2ha
        )

        self.BiomassProductivity.values = np.where(
            self.BioPotFrac.values > EPS,
            raw_productivity,
            0.0,
        ).astype(np.float32)

        # ------------------------------------------------------------------
        # 13. Total primary biomass potential, GJ / yr
        # ------------------------------------------------------------------

        self.BiomassPotential.values = (
            self.BiomassProductivity.values
            * (
                self.BioPotArea.values
                / km2ha
            )
        ).astype(np.float32, copy=False)

    def _calculate_biomass_costs(
        self,
        time: prism.Time,
        CCTransport: prism.Array[Region27, np.float32],
        FeedstockEnergyUse: prism.Array[Region27, BioenergyCrop5, np.float32],
        Interest: prism.Array[Region27, np.float32],
        BioRiskFactor: prism.Array[Region27, np.float32],
        ctax: prism.Array[Region28, np.float32],
        GDPpc: prism.Array[Region28, np.float32],
        GDPpc_tscen: prism.Array[Region28, np.float32],
        CropLandRent: prism.Array[LandGrid, np.float32],
        PastureLandRent: prism.Array[LandGrid, np.float32],
        BioConversionNetback: prism.Array[Region27, BioenergyCrop5, np.float32],
    ):
        """Calculate annual economics and provisional NPV by crop/source land.

        ``BioConversionNetback`` is the TIMER-derived temporary value of one
        GJ of primary biomass after downstream conversion costs, in $/GJ-primary.
        It replaces the earlier in-module reconstruction from BLF/BSF marginal
        prices, conversion efficiencies and non-feed conversion costs.

        LUC carbon, forest opportunity cost and one-off land-conversion costs
        are intentionally not included yet.
        """
        t = time.t
        t_wacc_switch = prism.Q_(2030, "year")
        regions = self.region_cells.values
        is_region = self.region_mask.values

        n_grid = len(LandGrid.coords)
        n_reg = len(Region26.coords)
        n_biocrop = len(BioenergyCrop5.coords)
        n_landcover = len(LandCover.coords)

        gdp = np.maximum(GDPpc.values[:n_reg], EPS).astype(np.float32)
        gdp0 = np.maximum(GDPpc_tscen.values[:n_reg], EPS).astype(np.float32)
        interest = np.maximum(Interest.values[:n_reg], EPS).astype(np.float32)
        tax = ctax.values[:n_reg].astype(np.float32)
        risk = BioRiskFactor.values[:n_reg].astype(np.float32)

        def _print_array_check(name, x):
            x = np.asarray(x, dtype=np.float64)

            finite = np.isfinite(x)

            print(
                f"  {name:28s}: "
                f"shape={x.shape}, "
                f"finite={finite.sum():,}/{x.size:,}",
                end=""
            )

            if np.any(finite):
                print(
                    f", min={x[finite].min():.6g}, "
                    f"mean={x[finite].mean():.6g}, "
                    f"max={x[finite].max():.6g}"
                )
            else:
                print(", ALL NON-FINITE")

        _print_array_check(
            "BioConversionNetback",
            BioConversionNetback.values[:n_reg, :]
        )

        _print_array_check(
            "Interest",
            Interest.values[:n_reg]
        )

        _print_array_check(
            "BioRiskFactor",
            BioRiskFactor.values[:n_reg]
        )

        _print_array_check(
            "GDPpc",
            GDPpc.values[:n_reg]
        )

        _print_array_check(
            "CCTransport",
            CCTransport.values[:n_reg]
        )

        _print_array_check(
            "FeedstockEnergyUse",
            FeedstockEnergyUse.values[:n_reg, :]
        )

        _print_array_check(
            "CropLandRent",
            CropLandRent.values
        )

        _print_array_check(
            "PastureLandRent",
            PastureLandRent.values
        )

        # Primary-biomass cultivation requirements from TIMER.
        self.CapCD_ini.values = self.CapCD_ini_in.values[:n_reg, :] * self.Deflator
        self.FeedstockCostTransp.values = (
            1.3 * self.FeedstockCostTransp_in.values * self.Deflator
        ).astype(np.float32)

        self.BioLabourCost.values = np.where(
            gdp > EPS, (4.2028 * gdp ** -0.1787) * gdp, 0.0
        ).astype(np.float32)
        self.Bio_Pl_Pk.values = np.where(
            interest > EPS, self.BioLabourCost.values / interest, 0.0
        ).astype(np.float32)

        alpha = self.CobbDouglasCoeff.values[None, :]
        self.Bio_Pl_PKCDcoeff.values = (
            self.Bio_Pl_Pk.values[:, None] * alpha / (1.0 - alpha)
        ).astype(np.float32)

        cap_lab = self.CapCD_ini.values / np.maximum(
            self.LabCD_ini.values[:n_reg, :], EPS
        )
        pkcd = np.maximum(self.Bio_Pl_PKCDcoeff.values, EPS)
        self.LaborRequired.values = (
            (cap_lab ** alpha)
            * (pkcd ** (-alpha))
            * self.LabCD_ini.values[:n_reg, :]
        ).astype(np.float32)
        self.CapitalRequired.values = (
            self.Bio_Pl_PKCDcoeff.values * self.LaborRequired.values
        ).astype(np.float32)

        # Recurring feedstock-production emissions in kg C/GJ-primary.
        self.FeedstockEmission.values = (
            self.FeedstockN2OEmission.values[:n_reg, :] * 12.0 / 44.0
            + CCTransport.values[:n_reg, None]
            * FeedstockEnergyUse.values[:n_reg, :]
        ).astype(np.float32)

        # Regional discount rate / WACC.
        self.DiscRateBioProdWeighted.values = 0.5 * (
            self.DiscRateWACC.values[:n_reg, 0]
            + self.DiscRateWACC.values[:n_reg, 12]
        )
        delta_log_gdp = np.minimum(11.0, np.log(gdp)) - np.minimum(
            11.0, np.log(gdp0)
        )
        self.RegionalCreditRisk.values = -0.0059 * delta_log_gdp
        wacc = (
            self.DiscRateBioProdWeighted.values
            + self.PartialDerivativeWACC.values[:n_reg]
            * self.RegionalCreditRisk.values
        )
        wacc = np.clip(wacc, 0.01, 0.25)
        self.DiscRateBio.values = (
            interest if t < t_wacc_switch else wacc.astype(np.float32)
        )
        dr = self.DiscRateBio.values

        # Annualized capital service cost.
        L_capital = float(self.BioFuelEconomicLifetime)
        denom = 1.0 - (1.0 + dr) ** (-L_capital)
        self.BioAnnuity.values = np.where(
            dr > 0.001,
            dr / np.where(np.abs(denom) > EPS, denom, np.inf),
            1.0 / L_capital,
        ).astype(np.float32)

        for crop in BioenergyCrop5:
            self.CapitalCostBiom.values[:, crop] = (
                1.3
                * risk
                * self.BioAnnuity.values
                * self.CapitalRequired.values[:, crop]
            ).astype(np.float32)
            self.LaborCostBiom.values[:, crop] = (
                1.3
                * risk
                * self.BioLabourCost.values
                * self.LaborRequired.values[:, crop]
            ).astype(np.float32)
            self.EmissionCostBiom.values[:, crop] = np.where(
                tax > EPS,
                self.FeedstockEmission.values[:, crop] * tax / 1000.0,
                0.0,
            ).astype(np.float32)

        # TIMER temporary primary-biomass value ($/GJ-primary).
        raw_netback_reg = np.asarray(
            BioConversionNetback.values[:n_reg, :], dtype=np.float32
        )
        raw_netback_grid = np.full(
            (n_grid, n_biocrop), np.nan, dtype=np.float32
        )
        raw_netback_grid[is_region, :] = raw_netback_reg[regions[is_region], :]
        self.BioFeedstockValue[t] = raw_netback_grid

        # Negative netback is informative diagnostically but cannot generate
        # positive willingness to pay for primary biomass in the allocator.
        allocation_netback_grid = np.maximum(raw_netback_grid, 0.0)
        prod = np.asarray(self.BiomassProductivity.values, dtype=np.float32)
        self.BioAnnualRevenue[t] = (
            prod * allocation_netback_grid
        ).astype(np.float32)

        annual_nonland_cost = np.full(
            (n_grid, n_biocrop), np.nan, dtype=np.float32
        )
        for crop in BioenergyCrop5:
            valid_prod = (
                is_region
                & np.isfinite(prod[:, crop])
                & (prod[:, crop] > EPS)
            )
            r = regions[valid_prod]
            fixed_cost_per_ha = (
                self.CapitalCostBiom.values[r, crop]
                + self.LaborCostBiom.values[r, crop]
            )
            variable_cost_per_gj = (
                self.EmissionCostBiom.values[r, crop]
                + self.FeedstockCostTransp.values[crop]
            )
            annual_nonland_cost[valid_prod, crop] = (
                fixed_cost_per_ha
                + prod[valid_prod, crop] * variable_cost_per_gj
            ).astype(np.float32)
        self.BioAnnualNonLandCost[t] = annual_nonland_cost

        # Opportunity cost by reference land category. Forest and other land
        # are zero in this first version; forestry opportunity cost enters later.
        land_oc = np.zeros((n_grid, n_landcover), dtype=np.float32)
        land_oc[:, 0] = np.asarray(CropLandRent.values, dtype=np.float32)
        land_oc[:, 1] = np.asarray(PastureLandRent.values, dtype=np.float32)
        if n_landcover > 4:
            land_oc[:, 4:] = np.nan

        # Constant-real-flow 30-year NPV for the initial implementation.
        H = float(self.BioLandNPVHorizon)
        pv_factor_reg = np.where(
            dr > 0.001,
            (1.0 - (1.0 + dr) ** (-H)) / dr,
            H,
        ).astype(np.float32)
        pv_factor_grid = np.full(n_grid, np.nan, dtype=np.float32)
        pv_factor_grid[is_region] = pv_factor_reg[regions[is_region]]

        annual_margin = (
            self.BioAnnualRevenue[t]
            - self.BioAnnualNonLandCost[t]
        )
        available = np.asarray(self.BioPotFracByLand.values, dtype=np.float32)
        npv_by_land = np.full(
            (n_grid, n_biocrop, n_landcover), np.nan, dtype=np.float32
        )

        for land_idx in range(n_landcover):
            source_available = available[:, :, land_idx] > EPS
            annual_net = annual_margin - land_oc[:, land_idx][:, None]
            valid = (
                source_available
                & np.isfinite(annual_net)
                & np.isfinite(pv_factor_grid[:, None])
            )
            npv_by_land[:, :, land_idx] = np.where(
                valid,
                pv_factor_grid[:, None] * annual_net,
                np.nan,
            ).astype(np.float32)

        # Future insertion point:
        # npv_by_land -= BioLUCCarbonNPV[t]
        # npv_by_land -= BioLandConversionCost.values
        self.BioNPVByLand[t] = npv_by_land

        valid_weight = (available > EPS) & np.isfinite(npv_by_land)
        weights = np.where(valid_weight, available, 0.0)
        weight_sum = np.sum(weights, axis=2)
        weighted_npv = np.sum(
            np.where(valid_weight, available * npv_by_land, 0.0), axis=2
        )
        self.BioNPV[t] = np.divide(
            weighted_npv,
            weight_sum,
            out=np.full((n_grid, n_biocrop), np.nan, dtype=np.float32),
            where=weight_sum > EPS,
        ).astype(np.float32)

    def _write_results_to_files(self, time: prism.Time):
        t, dt, t_start, t_end = time.t, time.dt, time.start, time.end
        if t != t_end:
            return

        times = np.arange(
            t_start.magnitude,
            t_end.magnitude + dt.magnitude,
            dt.magnitude,
            dtype=int,
        )
        grid = np.array(LandGrid.coords)
        biocrop = np.array(BioenergyCrop5.coords)
        landcover = np.array(LandCover.coords)

        exports = [
            (self.BioPotFrac, "BioPotFrac.nc", "BioPotFrac", ("grid", "biocrop"), None, "-", "Eligible fraction of each grid cell by bioenergy crop."),
            (self.BioPotFracByLand, "BioPotFracByLand.nc", "BioPotFracByLand", ("grid", "biocrop", "landcover"), None, "-", "Eligible fraction by bioenergy crop and SSP2_M reference land category."),
            (self.BiomassYield, "BiomassYield.nc", "BiomassYield", ("grid", "biocrop"), times, "kgDM/km2/yr", "Dry-matter bioenergy-crop yield."),
            (self.BiomassProductivity, "BiomassProductivity.nc", "BiomassProductivity", ("grid", "biocrop"), None, "GJ/ha/yr", "Primary biomass energy productivity."),
            (self.BiomassPotential, "BiomassPotential.nc", "BiomassPotential", ("grid", "biocrop"), None, "GJ/yr", "Total primary biomass potential on eligible area."),
            (self.BioFeedstockValue, "BioFeedstockValue.nc", "BioFeedstockValue", ("grid", "biocrop"), times, "$/GJ-primary", "Raw TIMER BioConversionNetback mapped to grid cells."),
            (self.BioAnnualRevenue, "BioAnnualRevenue.nc", "BioAnnualRevenue", ("grid", "biocrop"), times, "$/ha/yr", "Annual biomass revenue before primary production costs."),
            (self.BioAnnualNonLandCost, "BioAnnualNonLandCost.nc", "BioAnnualNonLandCost", ("grid", "biocrop"), times, "$/ha/yr", "Annual non-land primary biomass production cost."),
            (self.BioNPV, "BioNPV.nc", "BioNPV", ("grid", "biocrop"), times, "$/ha", "Area-weighted provisional bioenergy NPV across eligible reference land categories."),
            (self.BioNPVByLand, "BioNPVByLand.nc", "BioNPVByLand", ("grid", "biocrop", "landcover"), times, "$/ha", "Provisional bioenergy NPV by crop and SSP2_M reference land category."),
        ]

        for var, filename, var_name, dims, var_times, units, description in exports:
            coords = {"grid": grid, "biocrop": biocrop}
            if "landcover" in dims:
                coords["landcover"] = landcover
            export_variable_to_netcdf(
                prism_var=var,
                filename=filename,
                var_name=var_name,
                dims=dims,
                coords=coords,
                times=var_times,
                units=units,
                description=description,
            )

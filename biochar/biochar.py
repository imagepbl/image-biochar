import prism
import warnings
import numpy as np
from pathlib import Path
from pint.errors import UnitStrippedWarning

from timer.cdr.cdr_util import (
    region_to_grid,
    biochar_data_dir,
    res_data_dir,
    LandGrid,
    Region27,
    Region28,
    Region26,
    LandCover,
    Sector,
    Carrier,
    Residue,
    RCC,
    RCCPlus,
    PopDim,
    Fertiliser,
    SoilDepth,
    Crop38,
    PyrolysisTechnology,
    BiocharFeedstock,
    TransportMode,
)
from . import physics
from . import economics
from . import allocation
from . import output
from . import diagnostics

warnings.simplefilter("ignore", UnitStrippedWarning)

@prism.interface
class Biochar(prism.Model):
    
    output_dir = Path(".")
    carbon_tax_directory = None
    residue_directory = None
    residue_rationing_reference_directory = None
    proportional_residue_rationing = False
    disabled_technologies = ()
    simplify_biochar = False
    simplify_biochar_LT = False
    simplified_technology_index = 1
    simplified_baseline_year = 2025
    carbon_persistence_model = "azzi_power"
    print_diagnostics = True
    print_residue_rationing_diagnostics = False
    calculate_timer_residue_multipliers = False

    # Parameters
    deflator_eur_2023: prism.Scalar = 0.721
    deflator_eur_2020: prism.Scalar = 0.7576
    deflator_usd_2010: prism.Scalar = 0.813
    inflator_usd_2025: prism.Scalar = 1.67
    conversion_eur_usd_2005: prism.Scalar = 1.244
    
    application_rate: prism.Scalar = 20.0 # ton biochar/ha
    application_interval: prism.Scalar = 5 # years
    max_cumulative_application_t_ha = None
    agronomic_response_mode = "fixed_window"
    agronomic_full_effect_years = 2
    agronomic_year5_retention = 0.5
    application_cost: prism.Scalar = 68.0 # $2010/ha (Williams et al. 2010)
    MaxBiocharRate: prism.Scalar = 0.085 # fractional annual cap on regional first-time treated-area growth
    MaxBiocharCapacityGrowth: prism.Scalar = 10.0 # -/yr; muted 
    biochar_initial_capacity_fraction: prism.Scalar = 0.01 # -

    plant_lifetime: prism.Scalar = 25 # yr
    construction_time: prism.Scalar = 5 # yr
    plant_cf: prism.Scalar = 0.85 # capacity factor (-)
    plant_discount_rate: prism.Scalar = 0.08 # -/yr
    plant_capture_rate: prism.Scalar = 0.90 # - (Beiron et al. 2022)
    plant_ccs_capex: prism.Scalar = 15520 # kEUR2020 (Beiron et al. 2022)
    plant_ccs_opex: prism.Scalar = 0.06 # fraction of capex/yr (Beiron et al. 2022)
    TranspCost_in: prism.Scalar = 0.22 # $/ton/km (Deng et al. 2024)

    # Export variables
    yield_effects: prism.Array[LandGrid, PyrolysisTechnology, np.float32] = prism.export()
    TimerSharedResidueMultiplier: prism.TimeVariable[Region26, RCCPlus, np.float32] = prism.export()
    TimerResidualResidueMultiplier: prism.TimeVariable[Region26, RCCPlus, np.float32] = prism.export()

    # Import variables
    lignin_content: prism.Array[BiocharFeedstock, np.float32] = prism.private(file=biochar_data_dir / "feedstock_lignin_content.dat") 
    ash_content: prism.Array[BiocharFeedstock, np.float32] = prism.private(file=biochar_data_dir / "feedstock_ash_content.dat") 
    hc_slope: prism.Array[BiocharFeedstock, np.float32] = prism.private(file=biochar_data_dir / "feedstock_HC_ratio_slope.dat") 
    hc_intercept: prism.Array[BiocharFeedstock, np.float32] = prism.private(file=biochar_data_dir / "feedstock_HC_ratio_intercept.dat") 

    pyrolysis_temp: prism.Array[PyrolysisTechnology, np.float32] = prism.private(file=biochar_data_dir / "pyrolysis_temperature.dat") 
    pyrolysis_capex: prism.Array[PyrolysisTechnology, np.float32] = prism.private(file=biochar_data_dir / "pyrolysis_capex.dat") 
    pyrolysis_opex: prism.Array[PyrolysisTechnology, np.float32] = prism.private(file=biochar_data_dir / "pyrolysis_opex.dat") 
    pyrolysis_prod_rate: prism.Array[PyrolysisTechnology, np.float32] = prism.private(file=biochar_data_dir / "pyrolysis_prod_rate.dat") 
    pyrolysis_bio_oil_yield: prism.Array[PyrolysisTechnology, np.float32] = prism.private(file=biochar_data_dir / "pyrolysis_bio_oil_yield.dat") 
    pyrolysis_syngas_carbon_flow: prism.Array[PyrolysisTechnology, np.float32] = prism.private(file=biochar_data_dir / "pyrolysis_syngas_carbon_flow.dat") 

    soil_soc : prism.Array[LandGrid, np.float32] = prism.private(file=biochar_data_dir / "soil_soc.dat")
    soil_n: prism.Array[LandGrid, np.float32] = prism.private(file=biochar_data_dir / "soil_n.dat")
    soil_ph: prism.Array[LandGrid, np.float32] = prism.private(file=biochar_data_dir / "soil_ph.dat")
    soil_clay: prism.Array[LandGrid, np.float32] = prism.private(file=biochar_data_dir / "soil_clay.dat")

    PopDens: prism.Array[PopDim] = prism.private(file=res_data_dir / 'pop_density.dat')
    TranspDist: prism.Array[PopDim] = prism.private(file=res_data_dir / 'transp_distance.dat')

    # Local variables 
    region_cells: prism.Array[LandGrid, np.int16] = prism.private()
    region_mask: prism.Array[LandGrid] = prism.private()

    decay_rate_ref: prism.Array[PyrolysisTechnology, BiocharFeedstock,  np.float32] = prism.private()
    decay_rate: prism.Array[LandGrid, PyrolysisTechnology, BiocharFeedstock,  np.float32] = prism.private()
    n2o_effects: prism.Array[LandGrid, np.float32] = prism.private()
    ch4_effects: prism.Array[LandGrid, np.float32] = prism.private()
    biochar_retained_carbon: prism.TimeVariable[LandGrid, PyrolysisTechnology, np.float32] = prism.private()

    biochar_CDR_potential: prism.Array[LandGrid, BiocharFeedstock, np.float32] = prism.private()
    n2o_emis_mitigated: prism.Array[LandGrid, np.float32] = prism.private()
    ch4_emis_mitigated: prism.Array[LandGrid, np.float32] = prism.private()
    carbon_revenues: prism.Array[LandGrid, BiocharFeedstock, np.float32] = prism.private()
    yield_revenues: prism.Array[LandGrid, PyrolysisTechnology, np.float32] = prism.private()
    biochar_npv_regional: prism.Array[LandGrid, np.float32] = prism.private()
    BiocharCarbonFraction: prism.Array[PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()
    BiocharFeedstockInput: prism.Array[PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()

    BiocharArea: prism.TimeVariable[Region26, np.float32] = prism.private()
    BiocharAreaTech: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharCDR: prism.TimeVariable[Region26, np.float32] = prism.private()
    BiocharMitigationGridCum: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    BiocharMitigationGrid: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    BiocharAlloc: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    BiocharFieldCDRGrid: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    BiocharAlloc_cum: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    BiocharLastApplYear: prism.TimeVariable[LandGrid, np.int16] = prism.private()
    BiocharCumulativeAppliedPerHa: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    BiocharApplicationCount: prism.TimeVariable[LandGrid, np.int16] = prism.private()
    BiocharRealisedYieldEffect: prism.TimeVariable[LandGrid, np.float32] = prism.private()

    biochar_land_rank: prism.Array[LandGrid, np.int32] = prism.private()
    yield_revenues_ha: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    n2o_emis_mitigated_ha: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    ch4_emis_mitigated_ha: prism.TimeVariable[LandGrid, np.float32] = prism.private()
    biochar_new_capacity: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    biochar_capacity: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    retired_capacity: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    biochar_cumulative_capacity: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    biochar_capacity_investment: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharUnmetProfitableDemand: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharMitigationCost: prism.TimeVariable[LandGrid, PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()
    BiocharCostYieldRevenue: prism.TimeVariable[LandGrid, PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()
    BiocharCostFeedstock: prism.TimeVariable[Region26, PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()
    BiocharCostProduction: prism.TimeVariable[Region26, PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()
    BiocharCostBioenergyRevenue: prism.TimeVariable[Region26, PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()
    BiocharCostBiocharTransport: prism.TimeVariable[Region26, PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()
    BiocharCostApplication: prism.TimeVariable[Region26, PyrolysisTechnology, BiocharFeedstock, np.float32] = prism.private()
    BiocharCH4Mitig: prism.TimeVariable[Region26, np.float32] = prism.private()
    BiocharN2OMitig: prism.TimeVariable[Region26, np.float32] = prism.private()
    BiocharFossilMitig: prism.TimeVariable[Region26, np.float32] = prism.private()
    BiocharSyngasCCS: prism.TimeVariable[Region26, np.float32] = prism.private()
    BiocharFieldCDR: prism.TimeVariable[Region26, np.float32] = prism.private()

    BiocharExpectedNPVHa: prism.TimeVariable[LandGrid, PyrolysisTechnology, np.float32] = prism.private()
    expected_best_tech: prism.TimeVariable[LandGrid, np.int8] = prism.private()
    BiocharResidueDMUsed: prism.TimeVariable[Region26, PyrolysisTechnology, BiocharFeedstock, RCC, np.float32] = prism.private()

    BiocharRealisedMitigation: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedBiocharProduction: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedCostFeedstock: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedCostProduction: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedCostApplication: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedCostTransport: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedCostTransportCarbon: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedRevenueYield: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedRevenueBioenergy: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()
    BiocharRealisedNetCost: prism.TimeVariable[Region26, PyrolysisTechnology, np.float32] = prism.private()

    def _export(self, **kwargs):
        output.export_variable(self, **kwargs)

    def _simplification_enabled(self):
        return bool(
            getattr(self, "simplify_biochar", False)
            or getattr(self, "simplify_biochar_LT", False)
        )

    def _simplified_yield_revenue_enabled(self):
        return bool(getattr(self, "simplify_biochar_LT", False))

    def _technology_enabled_mask(self):
        ntech = len(PyrolysisTechnology.coords)
        enabled = np.ones(ntech, dtype=bool)

        for tech in self.disabled_technologies:
            if tech < 0 or tech >= ntech:
                raise ValueError(
                    f"Invalid technology index {tech}; available technologies are {list(PyrolysisTechnology.coords)}"
                )
            enabled[tech] = False

        return enabled

    def compute_carbon_content(self):
        physics.compute_carbon_content(self)

    def compute_initial_values(self, timeline: prism.Timeline):
        self.simulation_timeline = timeline
        self.region_cells.values, self.region_mask.values = region_to_grid('5m')

        physics.configure_simplified_technology(self)
        self.compute_carbon_content()

    def compute_values(
            self,
            time: prism.Time,
            GFRAC: prism.Array[LandGrid, Crop38, np.float32],
            GLANDCOVERFrac: prism.Array[LandGrid, LandCover, np.float32],
            GAREA5m: prism.Array[LandGrid, np.float32],
            GCH4: prism.Array[LandGrid, np.float32],
            GN2O: prism.Array[LandGrid, np.float32],
            GFERTILISER: prism.Array[LandGrid, Fertiliser, np.float32],
            SOILTEMP: prism.Array[LandGrid, SoilDepth, np.float32],
            GPOPD: prism.Array[LandGrid, np.float32],
            LandRent: prism.Array[LandGrid, np.float32],
            LandRentBiochar: prism.Array[LandGrid, np.float32],
            ctax: prism.Array[Region28, np.float32],
            energy_price: prism.Array[Region27, Sector, Carrier, np.float32],
            AAPRSE: prism.TimeVariable[Region27, Sector, Carrier, np.float32],
            regional_discount_rates: prism.Array[Region27, np.float32],
            ResCostCurveDM: prism.Array[Region26, Residue, RCCPlus, np.float32],
            ResPotCurveDM: prism.Array[Region26, Residue, RCCPlus, np.float32],
            CO2STORCOST: prism.Array[Region27, np.float32],
            CCpertkm: prism.Array[Region26, TransportMode, np.float32],
    ):

        if self._simplification_enabled() and not getattr(self, "_simplified_technology_configured", False): 
            physics.configure_simplified_technology(self)
            self.compute_carbon_content()

        self.compute_decay_rate(
            time=time, 
            SOILTEMP=SOILTEMP
        )

        self._estimate_soil_effects(
            time=time,
            GAREA5m=GAREA5m,
            GFRAC=GFRAC,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GFERTILISER=GFERTILISER,
            GCH4=GCH4,
            GN2O=GN2O,
        )

        self._determine_land_suitability(
            time=time,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GFRAC=GFRAC,
            GAREA5m=GAREA5m,
            GCH4=GCH4,
            GN2O=GN2O,
            GPOPD=GPOPD,
            ctax=ctax,
            energy_price=energy_price,
            AAPRSE=AAPRSE,
            regional_discount_rates=regional_discount_rates,
            LandRent=LandRent,
            LandRentBiochar=LandRentBiochar,
            ResCostCurveDM=ResCostCurveDM,
            ResPotCurveDM=ResPotCurveDM,
            CO2STORCOST=CO2STORCOST,
        )

        self._simulate_production_stock(
            time=time
        )

        self._allocate_biochar_application(
            time=time,
            GAREA5m=GAREA5m,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GPOPD=GPOPD,
            AAPRSE=AAPRSE,
            energy_price=energy_price,
            regional_discount_rates=regional_discount_rates,
            ResPotCurveDM=ResPotCurveDM,
            ResCostCurveDM=ResCostCurveDM,
            ctax=ctax,
            CO2STORCOST=CO2STORCOST,
            CCpertkm=CCpertkm,
        )

        self._compute_capacity_investments(
            time=time
        )

        self._calculate_mitigation_costs(
            time=time,
            GAREA5m=GAREA5m,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GPOPD=GPOPD,
            AAPRSE=AAPRSE,
            energy_price=energy_price,
            regional_discount_rates=regional_discount_rates,
            ResCostCurveDM=ResCostCurveDM,
            CO2STORCOST=CO2STORCOST,
            CCpertkm=CCpertkm
        )

        self._write_results_to_files(
            time=time
        )

    def compute_decay_rate(
            self,
            time: prism.Time,
            SOILTEMP: prism.Array[LandGrid, SoilDepth, np.float32]
    ):
        diagnostics.print_timestep_start(
            model=self,
            time=time,
        )
        physics.compute_decay_rate(
            self,
            time=time,
            SOILTEMP=SOILTEMP,
        )

    def _estimate_soil_effects(
            self,
            time: prism.Time,
            GAREA5m: prism.Array[LandGrid, np.float32],
            GLANDCOVERFrac: prism.Array[LandGrid, LandCover, np.float32],
            GFRAC: prism.Array[LandGrid, Crop38, np.float32],
            GFERTILISER: prism.Array[LandGrid, Fertiliser, np.float32],
            GCH4: prism.Array[LandGrid, np.float32],
            GN2O: prism.Array[LandGrid, np.float32],
    ):
        physics.estimate_soil_effects(
            self,
            time=time,
            GAREA5m=GAREA5m,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GFRAC=GFRAC,
            GFERTILISER=GFERTILISER,
            GCH4=GCH4,
            GN2O=GN2O,
        )

    def _determine_land_suitability(
            self,
            time: prism.Time,
            GLANDCOVERFrac: prism.Array[LandGrid, LandCover, np.float32],
            GFRAC: prism.Array[LandGrid, Crop38, np.float32],
            GAREA5m: prism.Array[LandGrid, np.float32],
            GCH4: prism.Array[LandGrid, np.float32],
            GN2O: prism.Array[LandGrid, np.float32],
            GPOPD: prism.Array[LandGrid, np.float32],
            ctax: prism.Array[Region28, np.float32],
            energy_price: prism.Array[Region27, Sector, Carrier, np.float32],
            AAPRSE: prism.TimeVariable[Region27, Sector, Carrier, np.float32],
            regional_discount_rates: prism.Array[Region27, np.float32],
            LandRent: prism.Array[LandGrid, np.float32],
            LandRentBiochar: prism.Array[LandGrid, np.float32],
            ResCostCurveDM: prism.Array[Region26, Residue, RCCPlus, np.float32],
            ResPotCurveDM: prism.Array[Region26, Residue, RCCPlus, np.float32],
            CO2STORCOST: prism.Array[Region27, np.float32],
    ):
        economics.determine_land_suitability(
            self,
            time=time,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GFRAC=GFRAC,
            GAREA5m=GAREA5m,
            GCH4=GCH4,
            GN2O=GN2O,
            GPOPD=GPOPD,
            ctax=ctax,
            energy_price=energy_price,
            AAPRSE=AAPRSE,
            regional_discount_rates=regional_discount_rates,
            LandRent=LandRent,
            LandRentBiochar=LandRentBiochar,
            ResCostCurveDM=ResCostCurveDM,
            ResPotCurveDM=ResPotCurveDM,
            CO2STORCOST=CO2STORCOST,
        )

    def _simulate_production_stock(self, time: prism.Time):
        allocation.simulate_production_stock(self, time=time)

    def _allocate_biochar_application(
            self,
            time: prism.Time,
            GAREA5m: prism.Array[LandGrid, np.float32],
            GLANDCOVERFrac: prism.Array[LandGrid, LandCover, np.float32],
            GPOPD: prism.Array[LandGrid, np.float32],
            AAPRSE: prism.TimeVariable[Region27, Sector, Carrier, np.float32],
            energy_price: prism.Array[Region27, Sector, Carrier, np.float32],
            regional_discount_rates: prism.Array[Region27, np.float32],
            ResCostCurveDM: prism.Array[Region26, Residue, RCCPlus, np.float32],
            ResPotCurveDM: prism.Array[Region26, Residue, RCCPlus, np.float32],
            ctax: prism.Array[Region28, np.float32],
            CO2STORCOST: prism.Array[Region27, np.float32],
            CCpertkm: prism.Array[Region26, TransportMode, np.float32],
    ):
        allocation.allocate_biochar_application(
            self,
            time=time,
            GAREA5m=GAREA5m,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GPOPD=GPOPD,
            AAPRSE=AAPRSE,
            energy_price=energy_price,
            regional_discount_rates=regional_discount_rates,
            ResCostCurveDM=ResCostCurveDM,
            ResPotCurveDM=ResPotCurveDM,
            ctax=ctax,
            CO2STORCOST=CO2STORCOST,
            CCpertkm=CCpertkm,
        )

    def _compute_capacity_investments(self, time: prism.Time):
        allocation.compute_capacity_investments(self, time=time)

    def _calculate_mitigation_costs(
            self,
            time: prism.Time,
            GAREA5m: prism.Array[LandGrid, np.float32],
            GLANDCOVERFrac: prism.Array[LandGrid, LandCover, np.float32],
            GPOPD: prism.Array[LandGrid, np.float32],
            AAPRSE: prism.TimeVariable[Region27, Sector, Carrier, np.float32],
            energy_price: prism.Array[Region27, Sector, Carrier, np.float32],
            regional_discount_rates: prism.Array[Region27, np.float32],
            ResCostCurveDM: prism.Array[Region26, Residue, RCCPlus, np.float32],
            CO2STORCOST: prism.Array[Region27, np.float32],
            CCpertkm: prism.Array[Region26, TransportMode, np.float32],
    ):
        economics.calculate_mitigation_costs(
            self,
            time=time,
            GAREA5m=GAREA5m,
            GLANDCOVERFrac=GLANDCOVERFrac,
            GPOPD=GPOPD,
            AAPRSE=AAPRSE,
            energy_price=energy_price,
            regional_discount_rates=regional_discount_rates,
            ResCostCurveDM=ResCostCurveDM,
            CO2STORCOST=CO2STORCOST,
            CCpertkm=CCpertkm,
        )

    def _write_results_to_files(self, time: prism.Time):
        output.write_results_to_files(self, time=time)

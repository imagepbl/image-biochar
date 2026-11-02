import prism
import numpy as np
from timer.cdr.cdr_util import (
    export_variable_to_netcdf,
    LandGrid,
    Region26,
    RCC,
    PyrolysisTechnology,
    BiocharFeedstock,
)

def export_variable(model, **kwargs):
    export_variable_to_netcdf(
        output_dir=model.output_dir,
        **kwargs,
    )

def _export_realised_economics(self, times):
    coords = {
        "region": np.array(Region26.coords),
        "tech": np.array(PyrolysisTechnology.coords),
    }
    specs = (
        ("BiocharRealisedMitigation", "tCO2-eq", "Net mitigation from realised deployments"),
        ("BiocharRealisedBiocharProduction", "tonne biochar", "Biochar applied in realised deployments"),
        ("BiocharRealisedCostFeedstock", "$2025", "Realised feedstock expenditure"),
        ("BiocharRealisedCostProduction", "$2025", "Realised production expenditure, including CCS and storage when enabled"),
        ("BiocharRealisedCostApplication", "$2025", "Realised biochar application expenditure"),
        ("BiocharRealisedCostTransport", "$2025", "Realised biochar transport expenditure"),
        ("BiocharRealisedCostTransportCarbon", "$2025", "Carbon-price penalty on realised transport emissions"),
        ("BiocharRealisedRevenueYield", "$2025", "Realised crop-yield revenue"),
        ("BiocharRealisedRevenueBioenergy", "$2025", "Realised bio-oil energy revenue"),
        ("BiocharRealisedNetCost", "$2025", "Realised resource cost net of yield and bioenergy revenues"),
    )

    for name, units, description in specs:
        export_variable(
            self,
            prism_var=getattr(self, name),
            filename=f"{name}.nc",
            var_name=name,
            dims=("region", "tech"),
            coords=coords,
            times=times,
            units=units,
            description=description,
        )

def write_results_to_files(
        self,
        time: prism.Time
):
    t, dt, t_start, t_end = time.t, time.dt, time.start, time.end

    if t == t_end:
        for variable, units, description in (
            ("BiocharCumulativeAppliedPerHa", "tonne biochar/ha", "Cumulative applied biochar per hectare"),
            ("BiocharApplicationCount", "application events", "Number of successful applications"),
            ("BiocharRealisedYieldEffect", "percent", "Yield effect associated with agronomic application history; diagnostic, not automatically coupled to IMAGE"),
        ):
            export_variable(
                self,
                prism_var=getattr(self, variable),
                filename=f"{variable}.nc",
                var_name=variable,
                dims=("grid",),
                coords={"grid": np.array(LandGrid.coords)},
                times=np.arange(t_start.magnitude, t_end.magnitude + dt.magnitude, dt.magnitude, dtype=int),
                units=units,
                description=description,
            )

    times = np.arange(
        t_start.magnitude,
        t_end.magnitude + dt.magnitude,
        dt.magnitude,
        dtype=int
    )

    if t == t_end:
        export_variable(
            self,
            prism_var=self.n2o_emis_mitigated,
            filename="n2o_emis_mitigated.nc",
            var_name="n2o_emis_mitigated",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=None,
            units="tCO2-eq",
            description="",
        )
        export_variable(
            self,
            prism_var=self.ch4_emis_mitigated,
            filename="ch4_emis_mitigated.nc",
            var_name="ch4_emis_mitigated",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=None,
            units="tCO2-eq",
            description="",
        )
        export_variable(
            self,
            prism_var=self.n2o_effects,
            filename="n2o_effects.nc",
            var_name="n2o_effects",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=None,
            units="fraction",
            description="Biochar effects on soil N2O emissions",
        )
        export_variable(
            self,
            prism_var=self.ch4_effects,
            filename="ch4_effects.nc",
            var_name="ch4_effects",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=None,
            units="fraction",
            description="Biochar effects on soil CH4 emissions",
        )
        export_variable(
            self,
            prism_var=self.yield_effects,
            filename="yield_effects.nc",
            var_name="yield_effects",
            dims=("grid", "tech"),
            coords={
                "grid": np.array(LandGrid.coords),
                "tech": np.array(PyrolysisTechnology.coords),

            },
            times=None,
            units="fraction",
            description="Biochar effects on crop yields",
        )
        export_variable(
            self,
            prism_var=self.BiocharAreaTech,
            filename="BiocharAreaTech.nc",
            var_name="BiocharAreaTech",
            dims=("region", "tech"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords)
            },
            times=times,
            units="ha",
            description="Cropland area treated with biochar",
        )
        export_variable(
            self,
            prism_var=self.BiocharCDR,
            filename="BiocharCDR.nc",
            var_name="BiocharCDR",
            dims=("region",),
            coords={
                "region": np.array(Region26.coords),
            },
            times=times,
            units="tCO2",
            description="Carbon sequestered through biochar application",
        )
        export_variable(
            self,
            prism_var=self.BiocharAlloc,
            filename="BiocharAlloc.nc",
            var_name="BiocharAlloc",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="-",
            description="Biochar deployment",
        )
        export_variable(
            self,
            prism_var=self.BiocharFieldCDRGrid,
            filename="BiocharFieldCDRGrid.nc",
            var_name="BiocharFieldCDRGrid",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="tCO2",
            description="Biochar gridded field CDR",
        )
        export_variable(
            self,
            prism_var=self.BiocharAlloc_cum,
            filename="BiocharAlloc_cum.nc",
            var_name="BiocharAlloc_cum",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="-",
            description="Cumulative biochar deployment",
        )
        export_variable(
            self,
            prism_var=self.biochar_capacity,
            filename="biochar_capacity.nc",
            var_name="biochar_capacity",
            dims=("region", "tech"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords),
            },
            times=times,
            units="tonne biochar/year",
            description="Regional biochar production capacity",
        )
        export_variable(
            self,
            prism_var=self.biochar_capacity_investment,
            filename="biochar_capacity_investment.nc",
            var_name="biochar_capacity_investment",
            dims=("region", "tech"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords),
            },
            times=times,
            units="tonne biochar/year",
            description="Regional biochar production capacity investments",
        )
        export_variable(
            self,
            prism_var=self.BiocharMitigationGrid,
            filename="BiocharMitigationGrid.nc",
            var_name="BiocharMitigationGrid",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="tCO2-eq",
            description="Gridded biochar GHG mitigation",
        )
        export_variable(
            self,
            prism_var=self.BiocharMitigationGridCum,
            filename="BiocharMitigationGridCum.nc",
            var_name="BiocharMitigationGridCum",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="tCO2-eq",
            description="Cumulative gridded biochar GHG mitigation",
        )
        export_variable(
            self,
            prism_var=self.yield_revenues_ha,
            filename="yield_revenues_ha.nc",
            var_name="yield_revenues_ha",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="$/ha",
            description="Yield revenues per hectare",
        )
        export_variable(
            self,
            prism_var=self.n2o_emis_mitigated_ha,
            filename="n2o_emis_mitigated_ha.nc",
            var_name="n2o_emis_mitigated_ha",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="tCO2-eq/ha",
            description="N2O emissions mitigated per hectare",
        )
        export_variable(
            self,
            prism_var=self.ch4_emis_mitigated_ha,
            filename="ch4_emis_mitigated_ha.nc",
            var_name="ch4_emis_mitigated_ha",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="tCO2-eq/ha",
            description="CH4 emissions mitigated per hectare",
        )
        export_variable(
            self,
            prism_var=self.biochar_retained_carbon,
            filename="biochar_retained_carbon.nc",
            var_name="biochar_retained_carbon",
            dims=("grid", "tech"),
            coords={
                "grid": np.array(LandGrid.coords),
                "tech": np.array(PyrolysisTechnology.coords),
            },
            times=times,
            units="-",
            description="Fraction of recalcitrant carbon in biochar ",
        )
        export_variable(
            self,
            prism_var=self.BiocharMitigationCost,
            filename="BiocharMitigationCost.nc",
            var_name="BiocharMitigationCost",
            dims=("grid", "tech", "feed"),
            coords={
                "grid": np.array(LandGrid.coords),
                "tech": np.array(PyrolysisTechnology.coords),
                "feed": np.array(BiocharFeedstock.coords),
            },
            times=times,
            units="$2025/tCO2",
            description="Net mitigation cost of biochar application ",
        )    
        export_variable(
            self,
            prism_var=self.BiocharCostFeedstock,
            filename="BiocharCostFeedstock.nc",
            var_name="BiocharCostFeedstock",
            dims=("region", "tech", "feed"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords),
                "feed": np.array(BiocharFeedstock.coords),
            },
            times=times,
            units="$2025/tCO2",
            description="Feedstock cost of biochar production ",
        )    
        export_variable(
            self,
            prism_var=self.BiocharCostProduction,
            filename="BiocharCostProduction.nc",
            var_name="BiocharCostProduction",
            dims=("region", "tech", "feed"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords),
                "feed": np.array(BiocharFeedstock.coords),
            },
            times=times,
            units="$2025/tCO2",
            description="Production cost of biochar ",
        )   
        export_variable(
            self,
            prism_var=self.BiocharCostBioenergyRevenue,
            filename="BiocharCostBioenergyRevenue.nc",
            var_name="BiocharCostBioenergyRevenue",
            dims=("region", "tech", "feed"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords),
                "feed": np.array(BiocharFeedstock.coords),
            },
            times=times,
            units="$2025/tCO2",
            description="Bioenergy revenues of biochar production",
        )   
        export_variable(
            self,
            prism_var=self.BiocharCostYieldRevenue,
            filename="BiocharCostYieldRevenue.nc",
            var_name="BiocharCostYieldRevenue",
            dims=("grid", "tech", "feed"),
            coords={
                "grid": np.array(LandGrid.coords),
                "tech": np.array(PyrolysisTechnology.coords),
                "feed": np.array(BiocharFeedstock.coords),
            },
            times=times,
            units="$2025/tCO2",
            description="Crop yield revenues of biochar production",
        )
        export_variable(
            self,
            prism_var=self.BiocharCostBiocharTransport,
            filename="BiocharCostBiocharTransport.nc",
            var_name="BiocharCostBiocharTransport",
            dims=("region", "tech", "feed"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords),
                "feed": np.array(BiocharFeedstock.coords),
            },
            times=times,
            units="$2025/tCO2",
            description="Cost of biochar transport",
        )    
        export_variable(
            self,
            prism_var=self.BiocharCostApplication,
            filename="BiocharCostApplication.nc",
            var_name="BiocharCostApplication",
            dims=("region", "tech", "feed"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords),
                "feed": np.array(BiocharFeedstock.coords),
            },
            times=times,
            units="$2025/tCO2",
            description="Cost of biochar application to croplands",
        )      
        export_variable(
            self,
            prism_var=self.BiocharCH4Mitig,
            filename="BiocharCH4Mitig.nc",
            var_name="BiocharCH4Mitig",
            dims=("region",),
            coords={
                "region": np.array(Region26.coords),
            },
            times=times,
            units="tCO2-eq",
            description="Regional CH4 mitigated through biochar application",
        )
        export_variable(
            self,
            prism_var=self.BiocharN2OMitig,
            filename="BiocharN2OMitig.nc",
            var_name="BiocharN2OMitig",
            dims=("region",),
            coords={
                "region": np.array(Region26.coords),
            },
            times=times,
            units="tCO2-eq",
            description="Regional N2O mitigated through biochar application",
        )
        export_variable(
            self,
            prism_var=self.BiocharFossilMitig,
            filename="BiocharFossilMitig.nc",
            var_name="BiocharFossilMitig",
            dims=("region",),
            coords={
                "region": np.array(Region26.coords),
            },
            times=times,
            units="tCO2",
            description="Regional fossil CO2 mitigated through biochar application",
        )
        export_variable(
            self,
            prism_var=self.BiocharSyngasCCS,
            filename="BiocharSyngasCCS.nc",
            var_name="BiocharSyngasCCS",
            dims=("region",),
            coords={
                "region": np.array(Region26.coords),
            },
            times=times,
            units="tCO2",
            description="Regional syngas CCS from biochar production",
        )
        export_variable(
            self,
            prism_var=self.BiocharFieldCDR,
            filename="BiocharFieldCDR.nc",
            var_name="BiocharFieldCDR",
            dims=("region",),
            coords={
                "region": np.array(Region26.coords),
            },
            times=times,
            units="tCO2",
            description="Regional carbon sequestered through biochar application",
        )
        export_variable(
            self,
            prism_var=self.expected_best_tech,
            filename="expected_best_tech.nc",
            var_name="expected_best_tech",
            dims=("grid",),
            coords={
                "grid": np.array(LandGrid.coords),
            },
            times=times,
            units="-",
            description="",
        ) 
        export_variable(
            self,
            prism_var=self.BiocharExpectedNPVHa,
            filename="BiocharExpectedNPVHa.nc",
            var_name="BiocharExpectedNPVHa",
            dims=("grid", "tech"),
            coords={
                "grid": np.array(LandGrid.coords),
                "tech": np.array(PyrolysisTechnology.coords),
            },
            times=times,
            units="$/ha",
            description="",
        ) 
        export_variable(
            self,
            prism_var=self.BiocharResidueDMUsed,
            filename="BiocharResidueDMUsed.nc",
            var_name="BiocharResidueDMUsed",
            dims=("region", "tech", "feed", "bin"),
            coords={
                "region": np.array(Region26.coords),
                "tech": np.array(PyrolysisTechnology.coords),
                "feed": np.array(BiocharFeedstock.coords),
                "bin": np.array(RCC.coords),
            },
            times=times,
            units="",
            description="",
        ) 

        _export_realised_economics(self, times)


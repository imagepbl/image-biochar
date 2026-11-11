import prism
import numpy as np
from timer.globals.constants import EPS
from timer.cdr.cdr_util import (
    LandGrid,
    Region26,
    Region27,
    Region28,
    LandCover,
    Sector,
    Carrier,
    Residue,
    RCCPlus,
    Crop38,
    PyrolysisTechnology,
    BiocharFeedstock,
    YieldHorizon,
    TransportMode,
)
from timer.cdr.biochar.feedstock import (
    get_cost_bins,
    prepare_residue_supply,
    prepare_residue_costs,
    get_ligneous_feedstock_mask,
    get_simplified_feedstock_properties,
    get_cached_simplified_feedstock_properties,
)

def determine_land_suitability(
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

    t = time.t
    regions = self.region_cells.values

    cropland_area = (
        GLANDCOVERFrac.values[:, 0] 
        * GAREA5m.values * 100.0
    ).astype(np.float32, copy=False) # ha

    # Determine biochar demand in ton biochar per grid cell
    biochar_demand = (
        cropland_area * self.application_rate
    ).astype(np.float32, copy=False)

    cropland_mask = (cropland_area > 1.0) & np.isfinite(biochar_demand)

    # Extract regional feedstock supply and feedstock mix
    cost_bins, _ = get_cost_bins()

    residue_state = prepare_residue_supply(
        model=self,
        time=time,
        ResPotCurveDM=ResPotCurveDM,
        copy=False,
        collect_diagnostics=False,
    )
    residue_supply_dm = residue_state.supply_dm

    # Extract the residue cost curves
    residue_cost_dm = prepare_residue_costs(ResCostCurveDM)

    # Effective feedstock properties
    if self._simplification_enabled():
        simplified_feedstock = get_simplified_feedstock_properties(
            model=self,
            time=time,
            residue_supply_dm=residue_supply_dm,
            cropland_area=cropland_area,
        )
        feedstock_input = simplified_feedstock.feedstock_input
    else:
        feedstock_input = np.broadcast_to(
            self.BiocharFeedstockInput.values[None, :, :],
            (
                len(Region26.coords),
                len(PyrolysisTechnology.coords),
                len(BiocharFeedstock.coords),
            ),
        )

    # Convert available feedstock supply from ton dry matter to ton biochar 
    regional_biochar_supply_by_residue = np.divide(
        np.sum(residue_supply_dm, axis=2)[:, None, :], # Sum across cost bins 
        feedstock_input, # ton feedstock DM per ton biochar
        out=np.zeros(
            (
                len(Region26.coords), 
                len(PyrolysisTechnology.coords), 
                len(BiocharFeedstock.coords)
            ), dtype=np.float32),
        where=feedstock_input > EPS,
    )

    # Sum across feedstock types
    regional_total_biochar_supply = np.sum(
        regional_biochar_supply_by_residue,
        axis=2,
        dtype=np.float32,
    )

    # Calculate shares of feedstock types in terms of biochar output
    residue_weights = np.divide(
        regional_biochar_supply_by_residue,
        regional_total_biochar_supply[:, :, None],
        out=np.zeros_like(regional_biochar_supply_by_residue, dtype=np.float32),
        where=regional_total_biochar_supply[:, :, None] > EPS,
    )

    # Supply-weighted feedstock costs by region and technology
    biochar_supply_by_bin = np.divide(
        residue_supply_dm[:, None, :, :],
        feedstock_input[:, :, :, None],
        out=np.zeros(
            (
                len(Region26.coords), 
                len(PyrolysisTechnology.coords), 
                len(BiocharFeedstock.coords),
                cost_bins
            ), dtype=np.float32),
        where=feedstock_input[:, :, :, None] > EPS,
    )

    feedstock_cost_per_t_biochar_by_bin = (
        residue_cost_dm[:, None, :, :]
        * feedstock_input[:, :, :, None]
    ).astype(np.float32)

    valid_cost_weight = (
        np.isfinite(feedstock_cost_per_t_biochar_by_bin)
        & np.isfinite(biochar_supply_by_bin)
        & (biochar_supply_by_bin > EPS)
    )

    feedstock_cost_numerator = np.sum(
        np.where(
            valid_cost_weight,
            feedstock_cost_per_t_biochar_by_bin * biochar_supply_by_bin,
            0.0,
        ),
        axis=(2, 3), # Sum across bins and feedstock types 
        dtype=np.float32,
    )

    feedstock_cost_denominator = np.sum(
        np.where(valid_cost_weight, biochar_supply_by_bin, 0.0),
        axis=(2, 3), # Sum across bins and feedstock types 
        dtype=np.float32,
    )

    # Regionally weighted feedstock costs in $ per ton biochar 
    expected_feedstock_cost_per_t = np.divide(
        feedstock_cost_numerator,
        feedstock_cost_denominator,
        out=np.zeros((len(Region26.coords), len(PyrolysisTechnology.coords)), dtype=np.float32),
        where=feedstock_cost_denominator > EPS,
    ).astype(np.float32)

    # Calculate regionally weighted retained carbon coefficients.
    biochar_retained_c_avg = np.zeros(
        (len(LandGrid.coords), len(PyrolysisTechnology.coords)),
        dtype=np.float32,
    )

    if self._simplification_enabled():
        retained_c_coeff_reg = (
            simplified_feedstock.carbon_fraction
            * np.exp(
                -simplified_feedstock.decay_rate * 100.0
            )
        ).astype(np.float32)

        for reg in Region26:
            reg_mask = (regions == reg)
            biochar_retained_c_avg[reg_mask, :] = np.sum(
                retained_c_coeff_reg[reg, :, :]
                * residue_weights[reg, :, :],
                axis=1,
                dtype=np.float32,
            )[None, :]

    else:
        retained_c_coeff = (
            self.BiocharCarbonFraction.values[None, :, :].astype(
                np.float32,
                copy=False,
            )
            * np.exp(
                -self.decay_rate.values.astype(
                    np.float32,
                    copy=False,
                )
                * 100.0
            )
        ).astype(np.float32)  # tC retained per ton biochar

        for reg in Region26:
            reg_mask = (regions == reg)
            w_reg = residue_weights[reg][None, :, :]

            biochar_retained_c_avg[reg_mask, :] = np.sum(
                retained_c_coeff[reg_mask, :, :] * w_reg,
                axis=2,
                dtype=np.float32,
            )

    self.biochar_retained_carbon[t] = biochar_retained_c_avg

    # Approximate transport distance per grid cell
    transport_distance = np.interp(
        GPOPD.values, 
        self.PopDens, 
        self.TranspDist
    )

    # Calculate potential soil N2O and CH4 mitigation
    crop_frac = np.nansum(
        GFRAC.values[:, 1:], # Exclude grassland 
        axis=1,
        dtype=np.float32,
    )

    paddy_frac = np.nansum(
        GFRAC.values[:, [2, 23]],
        axis=1,
        dtype=np.float32,
    )

    crop_frac = np.maximum(crop_frac, 0.0)
    paddy_frac = np.maximum(paddy_frac, 0.0)

    paddy_share = np.divide(
        paddy_frac,
        crop_frac,
        out=np.zeros_like(crop_frac, dtype=np.float32),
        where=crop_frac > EPS,
    )
    paddy_share = np.clip(paddy_share, 0.0, 1.0)
    upland_share = 1.0 - paddy_share

    paddy_area = cropland_area * paddy_share
    upland_area = cropland_area * upland_share

    # Specify N2O parameters 
    GWP_N2O_100 = 310.0
    N_to_N2O = 44.0 / 28.0

    n2o_tCO2eq = (
        GN2O.values
        * GAREA5m.values # km2
        * N_to_N2O
        * GWP_N2O_100
        / 1000.0  # kg to tonne
    ).astype(np.float32)

    # Apply only to upland cropland (exclude paddy cropland)
    n2o_tCO2eq_upland = n2o_tCO2eq * upland_share

    # Calculate total potential mitigation per grid cell 
    n2o_emis_mitig = (
        n2o_tCO2eq_upland
        * (-1.0 * self.n2o_effects.values / 100.0) # Derive absolute value, convert % to fraction
        * len(YieldHorizon.coords) # Sum across each year of the application window
    ).astype(np.float32) 

    self.n2o_emis_mitigated.values = n2o_emis_mitig

    # Calculate potential mitigation per unit of cropland area (ha)
    self.n2o_emis_mitigated_ha[t] = np.divide(
        n2o_emis_mitig,
        upland_area,
        out=np.zeros_like(cropland_area, dtype=np.float32),
        where=upland_area > 1.0,
    ).astype(np.float32)

    # Specify CH4 parameters
    GWP_CH4_100 = 30.0

    ch4_tCO2eq = (
        GCH4.values # kgCH4/km2-grid cell area/yr 
        * GAREA5m.values # km2
        * GWP_CH4_100
        / 1000.0 # kg to tonne 
    ).astype(np.float32)

    # Calculate total potential mitigation per grid cell 
    ch4_emis_mitig = (
        ch4_tCO2eq
        * (-1.0 * self.ch4_effects.values / 100.0) # Derive absolute value, convert % to fraction
        * len(YieldHorizon.coords) # Sum across each year of the application window
    ).astype(np.float32)

    self.ch4_emis_mitigated.values = ch4_emis_mitig

    # Calculate potential mitigation per unit of cropland area (ha)
    self.ch4_emis_mitigated_ha[t] = np.divide(
        ch4_emis_mitig,
        paddy_area,
        out=np.zeros_like(cropland_area, dtype=np.float32),
        where=paddy_area > EPS,
    ).astype(np.float32)

    # Calculate potential crop yield revenues
    annual_yield_revenues = np.zeros_like(LandRentBiochar.values, dtype=np.float32)
    land_rent = LandRent.values.astype(np.float32, copy=False) 
    land_rent_biochar = LandRentBiochar.values.astype(np.float32, copy=False)

    annual_yield_revenues = (
        (land_rent_biochar - land_rent) # $/km2/yr
        * (cropland_area / 100.0)  # ha to km2
    )

    n_years = len(YieldHorizon.coords)
    mode = getattr(self, "agronomic_response_mode", "fixed_window")
    if mode == "aging_within_window":
        if (n_years != int(self.application_interval) or n_years != int(time.dt.magnitude)):
            raise ValueError(
                "aging_within_window requires YieldHorizon, application_interval, and the simulation timestep to cover the same number of years"
            )
    elif mode != "fixed_window":
        raise ValueError(
            f"Unknown agronomic_response_mode: {mode!r}"
        )

    yield_revenues = annual_yield_revenues * np.float32(n_years)

    self.yield_revenues.values[:, :] = 0.0

    if getattr(self, "simplify_biochar", False):
        yield_revenues[...] = 0.0
    else:
        yield_tech = (
            int(self.simplified_technology_index)
            if getattr(self, "simplify_biochar_LT", False)
            else 0
        )
        self.yield_revenues.values[:, yield_tech] = np.where(
            cropland_area > 1.0,
            yield_revenues,
            0.0,
        ).astype(np.float32)

    # Calculate potential revenues per unit of cropland area (ha)
    self.yield_revenues_ha[t] = np.divide(
        yield_revenues,
        cropland_area,
        out=np.zeros_like(cropland_area, dtype=np.float32),
        where=cropland_area > 1.0,
    ).astype(np.float32)

    captured_c_per_t_biochar = (
        self.pyrolysis_syngas_carbon_flow.values # kgC per hour
        * self.plant_capture_rate # fraction
        / self.pyrolysis_prod_rate.values # ton biochar produced per hour
        / 1000.0 # kgC to tC
    ).astype(np.float32)

    production_cost_per_t_regtech = np.zeros((len(Region26.coords), len(PyrolysisTechnology.coords)), dtype=np.float32)
    bioenergy_value_per_t_regtech = np.zeros((len(Region26.coords), len(PyrolysisTechnology.coords)), dtype=np.float32)
    bioenergy_c_mitig_regtech = np.zeros((len(Region26.coords), len(PyrolysisTechnology.coords)), dtype=np.float32)

    fuel_c_density = np.array([25.5, 19.3, 15.3], dtype=np.float32)  # coal, oil, NG kgC/GJ; data from TIMER 

    for reg in Region26:
        r_reg = float(regional_discount_rates.values[reg])
        L = float(self.plant_lifetime)

        # Calculate regional capital recovery factor (crf); used to annualise upfront investment costs
        crf_reg = (
            (r_reg * (1.0 + r_reg) ** L)
            / ((1.0 + r_reg) ** L - 1.0)
        )

        production_cost_per_t = (
            (
                crf_reg * self.pyrolysis_capex.values
                / (self.pyrolysis_prod_rate.values # ton biochar per hour
                   * 8760.0 # hours per year
                   * self.plant_cf # Correct for plant capacity factor
                )
                + self.pyrolysis_opex.values # in $ per ton biochar produced
            )
            * self.deflator_eur_2023
            * self.conversion_eur_usd_2005
        ).astype(np.float32)

        captured_co2_flow_kg_s = (
            self.pyrolysis_syngas_carbon_flow.values # kgC per hour
            * self.plant_capture_rate # fraction
            * (44.0 / 12.0) # C to CO2
            / 3600.0 # hour to second
        ).astype(np.float32)

        total_capex_ccs = (
            self.plant_ccs_capex * 1000.0 # kEUR to EUR
            * self.deflator_eur_2020
            * self.conversion_eur_usd_2005
            * (captured_co2_flow_kg_s ** 0.6339)
        ).astype(np.float32)

        capex_ccs_per_t = (
            crf_reg
            * total_capex_ccs
            / (
                self.pyrolysis_prod_rate.values
                * 8760.0
                * self.plant_cf
            )
        ).astype(np.float32)

        opex_ccs_per_t = (
            total_capex_ccs
            * self.plant_ccs_opex
            / (
                self.pyrolysis_prod_rate.values
                * 8760.0
                * self.plant_cf
            )
        ).astype(np.float32)

        storage_cost_per_t = (
            CO2STORCOST.values[reg]
            * captured_c_per_t_biochar
        ).astype(np.float32)

        production_cost_per_t_regtech[reg, :] = (
            production_cost_per_t
            + capex_ccs_per_t
            + opex_ccs_per_t
            + storage_cost_per_t
        ).astype(np.float32)

        bioenergy_value_per_t_regtech[reg, :] = (
            energy_price.values[reg, 0, 4] # $ per GJ; value for modern bioenergy in industry; excluding carbon tax
            * self.pyrolysis_bio_oil_yield.values # GJ per ton biochar
        ).astype(np.float32) # $ per ton biochar 

        # Calculate regional mitigation value of the co-produced bio-oil. First calculate the industrial fossil fuel energy mix per region 
        fuel_use = AAPRSE[t].values[reg, 0, :3].astype(np.float32)
        fuel_total = np.sum(fuel_use, dtype=np.float32)
        if fuel_total > EPS:
            ind_fuel_mix_reg = fuel_use / fuel_total
        else:
            ind_fuel_mix_reg = np.zeros(3, dtype=np.float32)

        avg_fuel_c_density_reg = np.sum(
            ind_fuel_mix_reg * fuel_c_density, 
            dtype=np.float32
        ) # kg C per GJ

        bioenergy_c_mitig_regtech[reg, :] = (
            avg_fuel_c_density_reg # kg C per GJ
            * self.pyrolysis_bio_oil_yield.values # GJ per ton biochar
            / 1000.0 # kgC to tC
        ).astype(np.float32)

    # Expected net NPV per grid cell and technology
    carbon_tax = ctax.values.astype(np.float32, copy=False) # $ per tC

    expected_npv_cell = np.full((len(LandGrid.coords), len(PyrolysisTechnology.coords)), -np.inf, dtype=np.float32)
    expected_npv_ha = np.full((len(LandGrid.coords), len(PyrolysisTechnology.coords)), -np.inf, dtype=np.float32)

    # Total soil GHG mitigation in tC
    soil_nonco2_revenue_tC = (
        (ch4_emis_mitig + n2o_emis_mitig)
        * (12.0 / 44.0) # tCO2 to tC
    ).astype(np.float32)

    for reg in Region26:
        reg_cells = np.flatnonzero((regions == reg) & cropland_mask)

        ctax_reg = carbon_tax[reg]

        cell_demand_reg = biochar_demand[reg_cells]
        cell_area_reg = cropland_area[reg_cells]

        transport_distance_reg = transport_distance[reg_cells]

        for tech in PyrolysisTechnology:
            # Carbon revenues in $ per grid cell
            field_cdr_revenue = (
                cell_demand_reg # ton biochar
                * biochar_retained_c_avg[reg_cells, tech] # tC per ton biochar
                * ctax_reg # $ per tC
            ).astype(np.float32)

            soil_nonco2_revenue = (
                soil_nonco2_revenue_tC[reg_cells] # tC per grid cell
                * ctax_reg # $ per tC
            ).astype(np.float32)

            syngas_ccs_revenue = (
                captured_c_per_t_biochar[tech] # tC per ton biochar 
                * cell_demand_reg # ton biochar
                * ctax_reg # $ per tC
            ).astype(np.float32)

            fossil_displacement_revenue = (
                bioenergy_c_mitig_regtech[reg, tech] # tC per ton biochar 
                * cell_demand_reg # ton biochar
                * ctax_reg # $ per tC
            ).astype(np.float32)

            bioenergy_revenue = (
                bioenergy_value_per_t_regtech[reg, tech] # $ per ton biochar
                * cell_demand_reg # ton biochar
            ).astype(np.float32)

            # Crop yield revenues in $ per grid cell 
            yield_revenue = self.yield_revenues.values[reg_cells, tech].astype(np.float32, copy=False)

            # Cost terms in $ per grid cell
            feedstock_cost = (
                expected_feedstock_cost_per_t[reg, tech] # $ per ton biochar
                * cell_demand_reg # ton biochar
            ).astype(np.float32)

            production_cost = (
                production_cost_per_t_regtech[reg, tech] # $ per ton biochar
                * cell_demand_reg # ton biochar
            ).astype(np.float32)

            biochar_transport_cost = (
                transport_distance_reg
                * cell_demand_reg
                * self.TranspCost_in
            ).astype(np.float32)

            # Calculate net NPV by summing all costs and revenues
            npv_cell = (
                field_cdr_revenue
                + soil_nonco2_revenue
                + syngas_ccs_revenue
                + fossil_displacement_revenue
                + bioenergy_revenue
                + yield_revenue
                - feedstock_cost
                - production_cost
                - biochar_transport_cost
            ).astype(np.float32)

            expected_npv_cell[reg_cells, tech] = npv_cell

            expected_npv_ha[reg_cells, tech] = np.divide(
                npv_cell,
                cell_area_reg, # Cropland area in ha 
                out=np.full(reg_cells.size, -np.inf, dtype=np.float32),
                where=cell_area_reg > EPS,
            ).astype(np.float32)

    tech_enabled = self._technology_enabled_mask()
    expected_npv_cell[:, ~tech_enabled] = -np.inf
    expected_npv_ha[:, ~tech_enabled] = -np.inf

    # Save expected net NPV for diagnostics 
    self.BiocharExpectedNPVHa[t] = np.where(
        cropland_area[:, None] > EPS,
        expected_npv_ha,
        np.nan,
    ).astype(np.float32)

    # Extract the index of the best-performing technology option per grid cell 
    valid_any_tech = np.any(
        np.isfinite(expected_npv_ha),
        axis=1,
    )

    best_tech = np.full(
        len(LandGrid.coords),
        -1,
        dtype=np.int8,
    )

    best_tech[valid_any_tech] = np.argmax(
        expected_npv_ha[valid_any_tech, :],
        axis=1,
    ).astype(np.int8)

    self.expected_best_tech[t] = best_tech

    # Extract the net NPV of the best-performing technology option per grid cell 
    biochar_land_suitability_max = np.full(
        len(LandGrid.coords),
        -np.inf,
        dtype=np.float32,
    )

    biochar_land_suitability_max[valid_any_tech] = np.max(
        expected_npv_ha[valid_any_tech, :],
        axis=1,
    ).astype(np.float32)

    # Rank land cells within each region
    biochar_land_rank = np.full(len(LandGrid.coords), -1, dtype=np.int32)

    for reg in Region26:
        reg_cells = np.flatnonzero((regions == reg) & cropland_mask)

        finite_rank = np.isfinite(biochar_land_suitability_max[reg_cells])
        if not np.any(finite_rank):
            continue

        reg_cells = reg_cells[finite_rank]

        order = np.argsort(
            biochar_land_suitability_max[reg_cells],
            kind="mergesort",
        )[::-1] # Order from high to low

        ranked_cells = reg_cells[order]
        biochar_land_rank[ranked_cells] = np.arange(ranked_cells.size, dtype=np.int32)

    self.biochar_land_rank.values = biochar_land_rank


    from timer.cdr.biochar.diagnostics import print_land_rent_yield_diagnostics

    print_land_rent_yield_diagnostics(
        model=self,
        time=time,
        LandRent=LandRent,
        LandRentBiochar=LandRentBiochar,
        cropland_area=cropland_area,
    )



def calculate_mitigation_costs(
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
    t = time.t
    regions = self.region_cells.values

    cropland_area = (
        GLANDCOVERFrac.values[:, 0] 
        * GAREA5m.values * 100.0  # ha
    ) 

    has_cropland = (cropland_area > 1.0)

    biochar_demand = cropland_area * self.application_rate  # t biochar / grid cell

    net_cost_tco2 = np.full(
        (len(LandGrid.coords), 
         len(PyrolysisTechnology.coords), 
         len(BiocharFeedstock.coords)), 
         np.nan, dtype=np.float32
    )
    feedstock_cost_tco2 = np.full(
        (len(LandGrid.coords), 
         len(PyrolysisTechnology.coords), 
         len(BiocharFeedstock.coords)), 
         np.nan, dtype=np.float32
    )
    prod_cost_tco2 = np.full(
        (len(LandGrid.coords), 
         len(PyrolysisTechnology.coords), 
         len(BiocharFeedstock.coords)), 
         np.nan, dtype=np.float32
    )
    bioenergy_revenue_tco2 = np.full(
        (len(LandGrid.coords), 
         len(PyrolysisTechnology.coords), 
         len(BiocharFeedstock.coords)), 
         np.nan, dtype=np.float32
    )
    yield_revenue_tco2 = np.full(
        (len(LandGrid.coords), 
         len(PyrolysisTechnology.coords), 
         len(BiocharFeedstock.coords)), 
         np.nan, dtype=np.float32
    )
    transport_cost_tco2 = np.full(
        (len(LandGrid.coords), 
         len(PyrolysisTechnology.coords), 
         len(BiocharFeedstock.coords)), 
         np.nan, dtype=np.float32
    )
    application_cost_tco2 = np.full(
        (len(LandGrid.coords), 
         len(PyrolysisTechnology.coords), 
         len(BiocharFeedstock.coords)), 
         np.nan, dtype=np.float32
    )

    # Fossil displacement value of bio-oil coproduct
    fuel_c_density = np.array([25.5, 19.3, 15.3], dtype=np.float32)  # coal, oil, gas kgC/GJ

    # Classify feedstock types for yield-revenue eligibility
    ligneous_mask = get_ligneous_feedstock_mask(self)

    simplified_feedstock = None
    if self._simplification_enabled():
        simplified_feedstock = (
            get_cached_simplified_feedstock_properties(
                model=self,
                time=time,
            )
        )

    # Compute feedstock cost
    cost_bins = len(RCCPlus.coords) - 1
    cost_bins_slice = slice(0, cost_bins)

    residue_cost_dm_by_bin = ResCostCurveDM.values[:, :len(BiocharFeedstock.coords), cost_bins_slice]
    residue_cost_dm_by_bin = np.where(
        np.isfinite(residue_cost_dm_by_bin),
        residue_cost_dm_by_bin,
        np.inf
    )
    residue_dm_used_by_bin = self.BiocharResidueDMUsed[t].values

    residue_cost_numerator = np.sum(
        (residue_dm_used_by_bin * residue_cost_dm_by_bin[:, None, :, :]),
        axis=3,
        dtype=np.float32,
    )

    residue_cost_denominator = np.sum(
        residue_dm_used_by_bin,
        axis=3,
        dtype=np.float32,
    )

    residue_cost_dm = np.divide(
        residue_cost_numerator,
        residue_cost_denominator,
        out=np.full(
            (
                len(Region26.coords),
                len(PyrolysisTechnology.coords),
                len(BiocharFeedstock.coords),
            ),
            np.nan,
            dtype=np.float32,
        ),
        where=residue_cost_denominator > EPS,
    ).astype(np.float32)

    for reg in Region26:
        reg_cells = np.flatnonzero(has_cropland & (regions == reg))

        if reg_cells.size == 0:
            continue

        if simplified_feedstock is not None:
            feedstock_input_reg = simplified_feedstock.feedstock_input[reg]
            carbon_fraction_reg = simplified_feedstock.carbon_fraction[reg]
            decay_rate_reg = simplified_feedstock.decay_rate[reg]
        else:
            feedstock_input_reg = self.BiocharFeedstockInput.values
            carbon_fraction_reg = self.BiocharCarbonFraction.values
            decay_rate_reg = None

        # Regional financial parameters
        r = regional_discount_rates.values[reg]
        L = self.plant_lifetime

        crf_reg = (
            (r * (1.0 + r) ** L)
            / ((1.0 + r) ** L - 1.0)
        )

        # Base pyrolysis production cost ($/tonne biochar)
        production_cost_per_t = (
            (
                crf_reg * self.pyrolysis_capex.values
                / (self.pyrolysis_prod_rate.values * 8760.0 * self.plant_cf)
                + self.pyrolysis_opex.values
            )
            * self.deflator_eur_2023
            * self.conversion_eur_usd_2005
        ).astype(np.float32)

        captured_c_per_t_biochar = (
            self.pyrolysis_syngas_carbon_flow.values
            * self.plant_capture_rate
            / self.pyrolysis_prod_rate.values
            / 1000.0
        ).astype(np.float32)

        captured_co2_flow_kg_s = (
            self.pyrolysis_syngas_carbon_flow.values
            * self.plant_capture_rate
            * (44.0 / 12.0)
            / 3600.0
        ).astype(np.float32)

        total_capex_ccs = (
            self.plant_ccs_capex * 1000.0
            * self.deflator_eur_2020
            * self.conversion_eur_usd_2005
            * (captured_co2_flow_kg_s ** 0.6339)
        ).astype(np.float32)

        capex_ccs_per_t = (
            crf_reg
            * total_capex_ccs
            / (self.pyrolysis_prod_rate.values * 8760.0 * self.plant_cf)
        ).astype(np.float32)

        opex_ccs_per_t = (
            total_capex_ccs * self.plant_ccs_opex
            / (self.pyrolysis_prod_rate.values * 8760.0 * self.plant_cf)
        ).astype(np.float32)

        storage_cost_per_t = (
            CO2STORCOST.values[reg]
            * captured_c_per_t_biochar
        ).astype(np.float32)

        production_cost_per_t = (
            production_cost_per_t
            + capex_ccs_per_t
            + opex_ccs_per_t
            + storage_cost_per_t
        ).astype(np.float32)

        # Bio-oil energy revenue ($/tonne biochar)
        bioenergy_value_per_t = (
            energy_price.values[reg, 0, 4]
            * self.pyrolysis_bio_oil_yield.values
        ).astype(np.float32)

        # Bio-oil fossil displacement (tC/t biochar)
        fossil = AAPRSE[t].values[reg, 0, :3]
        fossil_total = np.sum(fossil)

        if fossil_total > EPS:
            ind_fuel_mix = fossil / fossil_total
            avg_fuel_c_density = np.sum(ind_fuel_mix * fuel_c_density)
        else:
            avg_fuel_c_density = 0.0

        bioenergy_mitig_tC_per_t = (
            avg_fuel_c_density
            * self.pyrolysis_bio_oil_yield.values
            / 1000.0
        ).astype(np.float32)

        for cell in reg_cells:
            demand = biochar_demand[cell]

            feedstock_cost = (
                residue_cost_dm[reg, :, :]
                * feedstock_input_reg
                * biochar_demand[cell]
            ).astype(np.float32)

            production_cost = (
                production_cost_per_t
                * biochar_demand[cell]
            ).astype(np.float32)

            application_cost = (
                self.application_cost
                * self.deflator_usd_2010
                * cropland_area[cell]
            ).astype(np.float32)

            transport_distance = np.interp(
                GPOPD.values[cell],
                self.PopDens,
                self.TranspDist
            ).astype(np.float32)

            biochar_transport_cost = (
                transport_distance
                * self.TranspCost_in
                * biochar_demand[cell]
            ).astype(np.float32)

            bioenergy_revenue = (
                bioenergy_value_per_t
                * biochar_demand[cell]
            ).astype(np.float32)

            yield_revenue = np.repeat(
                self.yield_revenues.values[cell, :][:, None],
                len(BiocharFeedstock.coords),
                axis=1
            ).astype(np.float32)

            if not self._simplification_enabled():
                yield_revenue[:, ligneous_mask] = 0.0

            net_cost = (
                feedstock_cost
                + production_cost[:, None]
                + application_cost
                + biochar_transport_cost
                - bioenergy_revenue[:, None]
                - yield_revenue
            ).astype(np.float32)

            # Mitigation denominator (tCO2-eq)
            if simplified_feedstock is not None:
                retained_c_fraction = (
                    carbon_fraction_reg * np.exp(-decay_rate_reg * 100.0)
                ).astype(np.float32)
            else:
                retained_c_fraction = (
                    carbon_fraction_reg
                    * np.exp(-self.decay_rate.values[cell, :, :] * 100.0)
                ).astype(np.float32)

            biochar_cdr = (
                demand
                * retained_c_fraction
                * (44.0 / 12.0)
            ).astype(np.float32)

            syngas_ccs_cdr = (
                demand
                * captured_c_per_t_biochar
                * (44.0 / 12.0)
            ).astype(np.float32)

            soil_ghg_mitigation = (
                self.ch4_emis_mitigated.values[cell]
                + self.n2o_emis_mitigated.values[cell]
            ).astype(np.float32)

            bioenergy_mitigation = (
                bioenergy_mitig_tC_per_t
                * demand
                * (44.0 / 12.0)
            ).astype(np.float32)

            biochar_transport_emis_tC = (
                CCpertkm.values[reg, 3]
                * transport_distance
                * demand
                / 1000.0
            ).astype(np.float32)

            feedstock_transport_emis_tC = (
                CCpertkm.values[reg, 3]
                * transport_distance
                * demand
                * feedstock_input_reg
                / 1000.0
            ).astype(np.float32)

            transport_emissions = (
                biochar_transport_emis_tC
                + feedstock_transport_emis_tC
            ) * (44.0 / 12.0)

            total_mitigation = (
                biochar_cdr
                + syngas_ccs_cdr[:, None]
                + soil_ghg_mitigation
                + bioenergy_mitigation[:, None]
                - transport_emissions
            ).astype(np.float32)

            valid = total_mitigation > EPS

            net_cost_tco2[cell, :, :] = np.where(valid, net_cost / total_mitigation, np.nan).astype(np.float32)
            feedstock_cost_tco2[cell, :, :] = np.where(valid, feedstock_cost / total_mitigation, np.nan).astype(np.float32)
            prod_cost_tco2[cell, :, :] = np.where(valid, production_cost[:, None] / total_mitigation, np.nan).astype(np.float32)
            transport_cost_tco2[cell, :, :] = np.where(valid, biochar_transport_cost / total_mitigation, np.nan).astype(np.float32)
            bioenergy_revenue_tco2[cell, :, :] = np.where(valid, bioenergy_revenue[:, None] / total_mitigation, np.nan).astype(np.float32)
            yield_revenue_tco2[cell, :, :] = np.where(valid, yield_revenue / total_mitigation, np.nan).astype(np.float32)
            application_cost_tco2[cell, :, :] = np.where(valid, application_cost / total_mitigation, np.nan).astype(np.float32)

    def grid_to_region_mean(grid_values):
        out = np.full(
            (
                len(Region26.coords),
                len(PyrolysisTechnology.coords),
                len(BiocharFeedstock.coords),
            ),
            np.nan,
            dtype=np.float32,
        )

        for reg in Region26:
            values = grid_values[regions == reg]
            if values.size:
                count = np.sum(~np.isnan(values), axis=0)
                np.divide(
                    np.nansum(values, axis=0),
                    count,
                    out=out[reg],
                    where=count > 0,
                )
        return out

    if self._simplification_enabled():
        tech_enabled = self._technology_enabled_mask()
        net_cost_tco2[:, ~tech_enabled, :] = np.nan
        feedstock_cost_tco2[:, ~tech_enabled, :] = np.nan
        prod_cost_tco2[:, ~tech_enabled, :] = np.nan
        bioenergy_revenue_tco2[:, ~tech_enabled, :] = np.nan
        yield_revenue_tco2[:, ~tech_enabled, :] = np.nan
        transport_cost_tco2[:, ~tech_enabled, :] = np.nan
        application_cost_tco2[:, ~tech_enabled, :] = np.nan

    self.BiocharMitigationCost[t] = net_cost_tco2 * self.inflator_usd_2025
    self.BiocharCostYieldRevenue[t] = yield_revenue_tco2 * self.inflator_usd_2025

    self.BiocharCostFeedstock[t] = grid_to_region_mean(feedstock_cost_tco2) * self.inflator_usd_2025
    self.BiocharCostProduction[t] = grid_to_region_mean(prod_cost_tco2) * self.inflator_usd_2025
    self.BiocharCostBioenergyRevenue[t] = grid_to_region_mean(bioenergy_revenue_tco2) * self.inflator_usd_2025
    self.BiocharCostBiocharTransport[t] = grid_to_region_mean(transport_cost_tco2) * self.inflator_usd_2025
    self.BiocharCostApplication[t] = grid_to_region_mean(application_cost_tco2) * self.inflator_usd_2025
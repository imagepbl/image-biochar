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
    PyrolysisTechnology,
    BiocharFeedstock,
    TransportMode,
)
from timer.cdr.biochar.diagnostics import (
    print_timestep_diagnostics,
    print_proportional_residue_diagnostics,
)
from timer.cdr.biochar.feedstock import (
    get_cost_bins,
    prepare_residue_supply,
    prepare_residue_costs,
    get_ligneous_feedstock_mask,
    get_simplified_feedstock_properties,
)
from timer.cdr.biochar.application_management import eligible_under_full_dose_cap
from timer.cdr.biochar.residue_rationing import (
    calculate_timer_residue_multipliers,
)

(_RC_FEEDSTOCK,
    _RC_PRODUCTION,
    _RC_APPLICATION,
    _RC_TRANSPORT,
    _RC_TRANSPORT_CARBON,
    _RC_YIELD_REVENUE,
    _RC_BIOENERGY_REVENUE,
) = range(7)

def _new_realised_economics():
    shape = (7, len(Region26.coords), len(PyrolysisTechnology.coords))
    return (
        np.zeros(shape, dtype=np.float64),
        np.zeros(shape[1:], dtype=np.float64),
        np.zeros(shape[1:], dtype=np.float64),
    )

def _store_realised_economics(self, t, costs, mitigation, biochar):
    scale = float(self.inflator_usd_2025)

    self.BiocharRealisedMitigation[t] = mitigation.astype(np.float32)
    self.BiocharRealisedBiocharProduction[t] = biochar.astype(np.float32)
    self.BiocharRealisedCostFeedstock[t] = (costs[_RC_FEEDSTOCK] * scale).astype(np.float32)
    self.BiocharRealisedCostProduction[t] = (costs[_RC_PRODUCTION] * scale).astype(np.float32)
    self.BiocharRealisedCostApplication[t] = (costs[_RC_APPLICATION] * scale).astype(np.float32)
    self.BiocharRealisedCostTransport[t] = (costs[_RC_TRANSPORT] * scale).astype(np.float32)
    self.BiocharRealisedCostTransportCarbon[t] = (costs[_RC_TRANSPORT_CARBON] * scale).astype(np.float32)
    self.BiocharRealisedRevenueYield[t] = (costs[_RC_YIELD_REVENUE] * scale).astype(np.float32)
    self.BiocharRealisedRevenueBioenergy[t] = (costs[_RC_BIOENERGY_REVENUE] * scale).astype(np.float32)

    net_cost = (
        costs[_RC_FEEDSTOCK]
        + costs[_RC_PRODUCTION]
        + costs[_RC_APPLICATION]
        + costs[_RC_TRANSPORT]
        - costs[_RC_YIELD_REVENUE]
        - costs[_RC_BIOENERGY_REVENUE]
    )
    self.BiocharRealisedNetCost[t] = (net_cost * scale).astype(np.float32)

def simulate_production_stock(
        self, 
        time: prism.Time
):
    t, dt = time.t, time.dt

    zero_capacity = prism.array(0.0, [Region26, PyrolysisTechnology])

    if self.construction_time % dt.magnitude > EPS:
        raise ValueError("Construction time should be a whole number of timesteps")

    t_construction = prism.Q_(self.construction_time, "year")

    previous_capacity = self.biochar_capacity.get(
        t - dt,
        zero_capacity
    )

    new_capacity = self.biochar_capacity_investment.get(
        t - t_construction,
        zero_capacity
    )

    retirement_fraction = dt.magnitude / self.plant_lifetime

    retired_capacity = (previous_capacity.values * retirement_fraction).astype(np.float32)

    self.biochar_new_capacity[t] = new_capacity
    self.retired_capacity[t] = retired_capacity

    self.biochar_capacity[t] = np.maximum(
        previous_capacity.values
        + new_capacity.values
        - retired_capacity,
        0.0
    ).astype(np.float32)

    self.biochar_cumulative_capacity[t] = (
        self.biochar_cumulative_capacity.get(t - dt, zero_capacity).values
        + new_capacity.values
    ).astype(np.float32)


def allocate_biochar_application(
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
    CO2STORCOST:  prism.Array[Region27, np.float32],
    CCpertkm: prism.Array[Region26, TransportMode, np.float32],
):
    t, dt, t_start = time.t, time.dt, time.start
    t_last = max(t - dt, t_start)
    t_scen = prism.Q_(2025, 'year')

    current_year = int(t.magnitude)
    regions = self.region_cells.values

    cropland_area = (
        GLANDCOVERFrac.values[:, 0]
        * GAREA5m.values * 100.0
    ).astype(np.float32) # km2 to ha
    cropland_mask = cropland_area > 1.0

    biochar_demand = (cropland_area * self.application_rate).astype(np.float32)

    land_rank = self.biochar_land_rank.values

    cost_bins, _ = get_cost_bins()

    residue_state = prepare_residue_supply(
        model=self,
        time=time,
        ResPotCurveDM=ResPotCurveDM,
        copy=True,
        collect_diagnostics=self.proportional_residue_rationing,
    )
    residue_supply_dm = residue_state.supply_dm

    residue_cost_dm = prepare_residue_costs(ResCostCurveDM)

    simplified_feedstock = None
    if self._simplification_enabled():
        simplified_feedstock = get_simplified_feedstock_properties(
            model=self,
            time=time,
            residue_supply_dm=residue_supply_dm,
            cropland_area=cropland_area,
        )

    # Current available production capacity
    remaining_capacity = self.biochar_capacity[t].values * dt.magnitude

    # Classify feedstock types
    ligneous_mask = get_ligneous_feedstock_mask(self)

    # Initialise states
    if t == t_start:
        biochar_alloc_cum = np.zeros(len(LandGrid.coords), dtype=np.int8)
        last_appl_year = np.full(len(LandGrid.coords), -1, dtype=np.int16)
        prev_biochar_area_reg = np.zeros(len(Region26.coords), dtype=np.float32)
        prev_biochar_cdr_reg = np.zeros(len(Region26.coords), dtype=np.float32)
        prev_mitigation_grid_cum = np.zeros(len(LandGrid.coords), dtype=np.float32)
        cumulative_applied = np.zeros(len(LandGrid.coords), dtype=np.float32)
        application_count = np.zeros(len(LandGrid.coords), dtype=np.int16)

    else:
        prev_biochar_area_reg = self.BiocharArea[t_last].values.copy()
        prev_biochar_cdr_reg = self.BiocharCDR[t_last].values.copy()
        biochar_alloc_cum = self.BiocharAlloc_cum[t_last].values.copy()
        last_appl_year = self.BiocharLastApplYear[t_last].values.copy()
        prev_mitigation_grid_cum = self.BiocharMitigationGridCum[t_last].values.copy()
        cumulative_applied = self.BiocharCumulativeAppliedPerHa[t_last].values.copy()
        application_count = self.BiocharApplicationCount[t_last].values.copy()

    eligible_reapply = (
        (last_appl_year < 0) |
        ((current_year - last_appl_year) >= self.application_interval)
    )
    cap = getattr(self, "max_cumulative_application_t_ha", None)
    eligible_reapply &= eligible_under_full_dose_cap(
        cumulative_applied, float(self.application_rate), cap
    )

    new_yield_application = np.zeros(len(LandGrid.coords), dtype=bool)

    biochar_alloc = np.zeros(len(LandGrid.coords), dtype=np.int8)
    tech_choice = np.full(len(LandGrid.coords), -1, dtype=np.int16)
    mitig_biochar_grid = np.zeros(len(LandGrid.coords), dtype=np.float32)
    field_cdr_grid = np.zeros(len(LandGrid.coords), dtype=np.float32)
    new_area_reg = np.zeros(len(Region26.coords), dtype=np.float32)
    biochar_area_tech = np.zeros((len(Region26.coords), len(PyrolysisTechnology.coords)), dtype=np.float32)
    new_cdr_reg = np.zeros(len(Region26.coords), dtype=np.float32)
    ch4_mitig_reg = np.zeros(len(Region26.coords), dtype=np.float32)
    n2o_mitig_reg = np.zeros(len(Region26.coords), dtype=np.float32)
    fossil_mitig_reg = np.zeros(len(Region26.coords), dtype=np.float32)
    syngas_ccs_reg = np.zeros(len(Region26.coords), dtype=np.float32)
    biochar_cdr_reg = np.zeros(len(Region26.coords), dtype=np.float32)
    unmet_profitable_demand = np.zeros((len(Region26.coords), len(PyrolysisTechnology.coords)), dtype=np.float32)
    residue_dm_used = np.zeros((len(Region26.coords), len(PyrolysisTechnology.coords), len(BiocharFeedstock.coords), cost_bins), dtype=np.float32)
    realised_costs, realised_mitigation, realised_biochar = _new_realised_economics()

    # Formulate regional annual growth caps
    max_biochar_delta = self.MaxBiocharRate * dt.magnitude

    regional_crop_area = np.zeros(len(Region26.coords), dtype=np.float32)
    for reg in Region26:
        reg_mask = (regions == reg)
        regional_crop_area[reg] = np.sum(cropland_area[reg_mask], dtype=np.float32)

    max_new_area_reg = np.zeros(len(Region26.coords), dtype=np.float32)
    for reg in Region26:
        if prev_biochar_area_reg[reg] > EPS:
            max_new_area_reg[reg] = prev_biochar_area_reg[reg] * max_biochar_delta
        else:
            max_new_area_reg[reg] = regional_crop_area[reg] * 0.01

    # Prevent deployment before scenario year
    if t < t_scen:
        self.BiocharAlloc[t] = biochar_alloc
        self.BiocharLastApplYear[t] = last_appl_year
        self.BiocharAlloc_cum[t] = biochar_alloc_cum
        self.BiocharCumulativeAppliedPerHa[t] = cumulative_applied
        self.BiocharApplicationCount[t] = application_count
        self.BiocharRealisedYieldEffect[t] = np.zeros(len(LandGrid.coords), dtype=np.float32)
        self.BiocharArea[t] = prev_biochar_area_reg + new_area_reg
        self.BiocharAreaTech[t] = biochar_area_tech
        self.BiocharCDR[t] = prev_biochar_cdr_reg + new_cdr_reg
        self.BiocharUnmetProfitableDemand[t] = unmet_profitable_demand
        self.BiocharMitigationGrid[t] = np.full(len(LandGrid.coords), np.nan, dtype=np.float32)
        self.BiocharMitigationGridCum[t] = prev_mitigation_grid_cum
        self.BiocharResidueDMUsed[t] = residue_dm_used
        _store_realised_economics(
            self, t, realised_costs, realised_mitigation, realised_biochar
        )
        return

    # Captured carbon from pyrolysis CCS
    captured_c_per_t_biochar = (
        self.pyrolysis_syngas_carbon_flow.values
        * self.plant_capture_rate
        / self.pyrolysis_prod_rate.values
    ) / 1000.0 # tC/t biochar

    carbon_tax = (ctax.values).astype(np.float32, copy=False)

    # Allocate biochar deployment by region
    for reg in Region26:
        reg_mask = (regions == reg)

        candidate_cells = np.flatnonzero(
            reg_mask
            & cropland_mask
            & eligible_reapply
        )

        if candidate_cells.size == 0:
            continue

        # Filter the ranked land cells
        valid_rank = land_rank[candidate_cells] >= 0

        candidate_cells = candidate_cells[valid_rank]
        order = np.argsort(land_rank[candidate_cells], kind="mergesort")
        candidate_cells = candidate_cells[order]

        # Initialise variables
        remaining_supply_dm = residue_supply_dm[reg, :, :].copy()
        remaining_capacity_reg = remaining_capacity[reg, :].copy()

        if simplified_feedstock is not None:
            feedstock_input_reg = simplified_feedstock.feedstock_input[reg]
            carbon_fraction_reg = simplified_feedstock.carbon_fraction[reg]
            decay_rate_reg = simplified_feedstock.decay_rate[reg]
        else:
            feedstock_input_reg = self.BiocharFeedstockInput.values
            carbon_fraction_reg = self.BiocharCarbonFraction.values
            decay_rate_reg = None

        area_used_reg = 0.0

        # Calculate production costs
        r = regional_discount_rates.values[reg]
        L = self.plant_lifetime
        crf_reg = (
            (r * (1.0 + r) ** L)
            / ((1.0 + r) ** L - 1.0)
        )

        production_cost_per_t_reg = (
            (crf_reg * self.pyrolysis_capex.values
                / (self.pyrolysis_prod_rate.values * 8760.0 * self.plant_cf)
                + self.pyrolysis_opex.values)
            * self.deflator_eur_2023
            * self.conversion_eur_usd_2005
        ).astype(np.float32)

        captured_co2_flow_kg_s = (
            self.pyrolysis_syngas_carbon_flow.values
            * self.plant_capture_rate
            * (44/12)
            / 3600
            ) # kgC/hr to kgCO2/s

        total_capex_ccs_reg = (
            self.plant_ccs_capex * 1000
            * self.deflator_eur_2020
            * self.conversion_eur_usd_2005
            * (captured_co2_flow_kg_s ** 0.6339)
        )

        capex_ccs_per_t_reg = (
            crf_reg
            * total_capex_ccs_reg
            / (self.pyrolysis_prod_rate.values * 8760.0 * self.plant_cf)
        )

        opex_ccs_per_t_reg = (
            total_capex_ccs_reg * self.plant_ccs_opex
            / (self.pyrolysis_prod_rate.values * 8760.0 * self.plant_cf)
        )

        co2_stor_cost_per_t_reg = (
            CO2STORCOST.values[reg] * captured_c_per_t_biochar
        )

        production_cost_per_t_reg += (
            capex_ccs_per_t_reg
            + opex_ccs_per_t_reg
            + co2_stor_cost_per_t_reg
        )

        # Calculate bioenergy revenues 
        bioenergy_value_per_t_reg = (
            energy_price.values[reg, 0, 4]
            * self.pyrolysis_bio_oil_yield.values
        ).astype(np.float32)

        # Calculate mitigation from bio-oil co-product
        fuel_use = AAPRSE[t].values[reg, 0, :3].astype(np.float32)
        fuel_total = np.sum(fuel_use, dtype=np.float32)
        if fuel_total > EPS:
            ind_fuel_mix_reg = fuel_use / fuel_total
        else:
            ind_fuel_mix_reg = np.zeros(3, dtype=np.float32)

        fuel_c_density = np.array([25.5, 19.3, 15.3]) # coal, oil, NG in kgC/GJ

        avg_fuel_c_density_reg = np.sum(
            ind_fuel_mix_reg * fuel_c_density
        )

        bioenergy_c_mitig_reg = (
            avg_fuel_c_density_reg
            * self.pyrolysis_bio_oil_yield.values
            / 1000.0 # kg to tonne
        ) # tC / t biochar

        bioenergy_c_mitig_revenue = (
            bioenergy_c_mitig_reg * carbon_tax[reg]
        )

        syngas_ccs_revenue = (
            captured_c_per_t_biochar * carbon_tax[reg]
        ).astype(np.float32)

        # Process cells in order of land suitability
        for cell in candidate_cells:
            if biochar_alloc[cell] == 1:
                continue

            cell_area = cropland_area[cell]
            cell_demand = biochar_demand[cell]

            if cell_demand <= EPS:
                continue

            is_new_land = (biochar_alloc_cum[cell] == 0)

            # Growth cap applies only to first-time allocations on new land
            if is_new_land and (area_used_reg + cell_area > max_new_area_reg[reg]):
                continue

            # Calculate application cost
            application_cost_cell = (
                self.application_cost
                * self.deflator_usd_2010
                * cell_area
            )

            # Calculate biochar transport cost
            transport_distance = np.interp(
                GPOPD.values[cell], 
                self.PopDens, 
                self.TranspDist
            )

            transport_cost = (
                transport_distance 
                * cell_demand 
                * self.TranspCost_in
            )

            transport_tkm = cell_demand * transport_distance

            biochar_transport_emis_C = (
                transport_tkm 
                * CCpertkm.values[reg, 3] 
                / 1000.0 # tC/grid cell
            )

            feedstock_transport_emis_C = (
                CCpertkm.values[reg, 3] 
                * transport_distance 
                * cell_demand * np.mean(feedstock_input_reg)
                / 1000.0
            )

            transport_emis_C = biochar_transport_emis_C + feedstock_transport_emis_C

            transport_c_cost_penalty = transport_emis_C * carbon_tax[reg]

            yield_revenue_option = np.repeat(
                self.yield_revenues.values[cell, :][:, None],
                len(BiocharFeedstock.coords),
                axis=1
            ).astype(np.float32)

            if not self._simplification_enabled():
                yield_revenue_option[:, ligneous_mask] = 0.0

            avoided_soil_emis_revenue = (
                self.ch4_emis_mitigated.values[cell]
                + self.n2o_emis_mitigated.values[cell]
            ) * (12.0/44.0) * carbon_tax[reg]

            # Calculate grid-cell-specific retained carbon value by technology/feedstock
            if simplified_feedstock is not None:
                retained_c_option = (
                    carbon_fraction_reg
                    * np.exp(-decay_rate_reg * 100.0)
                ).astype(np.float32)
            else:
                retained_c_option = (
                    carbon_fraction_reg
                    * np.exp(
                        -self.decay_rate.values[cell, :, :] * 100.0
                    )
                ).astype(np.float32)

            field_cdr_revenue_option = (
                cell_demand
                * retained_c_option
                * carbon_tax[reg]
            ).astype(np.float32)

            # Cell-specific base term for realised NPV, excluding residue/bin feedstock cost
            base_npv_cell = (
                field_cdr_revenue_option
                + yield_revenue_option
                + avoided_soil_emis_revenue 
                + (syngas_ccs_revenue * cell_demand)[:, None]
                + (bioenergy_c_mitig_revenue * cell_demand)[:, None]
                + (bioenergy_value_per_t_reg * cell_demand)[:, None]
                - (production_cost_per_t_reg * cell_demand)[:, None]
                - application_cost_cell
                - transport_cost
                - transport_c_cost_penalty
            ).astype(np.float32)

            tech_enabled = self._technology_enabled_mask()
            base_npv_cell[~tech_enabled, :] = -np.inf

            # Feasible residue/bin slices must still have enough remaining supply
            residue_dm_required = cell_demand * feedstock_input_reg

            feasible_supply = (
                remaining_supply_dm[None, :, :] >= residue_dm_required[:, :, None]
            )
            if not np.any(feasible_supply):
                continue

            # Define capacity constraint
            feasible_capacity = (remaining_capacity_reg >= cell_demand)

            feedstock_cost_option = (
                residue_cost_dm[reg, :, :][None, :, :]
                * residue_dm_required[:, :, None]
            ).astype(np.float32)

            option_npv_supply_only = (
                base_npv_cell[:, :, None]
                - feedstock_cost_option
            ).astype(np.float32)

            option_npv_supply_only[~feasible_supply] = -np.inf

            # Best profitable option ignoring capacity constraint 
            best_supply_flat = int(np.argmax(option_npv_supply_only))
            best_supply_tech, best_supply_res, best_supply_bin = np.unravel_index(
                best_supply_flat, option_npv_supply_only.shape
            )
            best_supply_npv = option_npv_supply_only[best_supply_tech, best_supply_res, best_supply_bin]

            # Apply capacity constraint
            feasible_biochar = feasible_supply & feasible_capacity[:, None, None]

            option_npv = option_npv_supply_only.copy()
            option_npv[~feasible_biochar] = -np.inf

            best_flat = int(np.argmax(option_npv))

            best_tech, best_res, best_bin = np.unravel_index(
                best_flat, option_npv.shape
            )
            best_npv = option_npv[best_tech, best_res, best_bin]

            # Allocate only if best realised NPV is positive
            if np.isfinite(best_npv) and best_npv > 0.0:
                biochar_alloc[cell] = 1
                last_appl_year[cell] = current_year
                cumulative_applied[cell] += float(self.application_rate)
                application_count[cell] += 1
                if (
                    self.agronomic_response_mode == "aging_within_window"
                    and best_tech == 0
                    and not ligneous_mask[best_res]
                ):
                    new_yield_application[cell] = True
                tech_choice[cell] = best_tech
                biochar_area_tech[reg, best_tech] += cell_area 

                selected_residue_dm_required = (
                    cell_demand * feedstock_input_reg[best_tech, best_res]
                )

                remaining_supply_dm[best_res, best_bin] -= selected_residue_dm_required

                residue_dm_used[reg, best_tech, best_res, best_bin] += selected_residue_dm_required

                remaining_capacity_reg[best_tech] -= cell_demand

                if is_new_land:
                    area_used_reg += cell_area
                    new_area_reg[reg] += cell_area

                # Realised carbon removal from actual chosen residue
                if simplified_feedstock is not None:
                    retained_c_selected = (
                        carbon_fraction_reg[best_tech, best_res]
                        * np.exp(
                            -decay_rate_reg[best_tech, best_res] * 100.0
                        )
                    )
                else:
                    retained_c_selected = (
                        carbon_fraction_reg[best_tech, best_res]
                        * np.exp(
                            -self.decay_rate.values[
                                cell,
                                best_tech,
                                best_res,
                            ]
                            * 100.0
                        )
                    )

                cell_cdr_biochar = (
                    cell_demand
                    * retained_c_selected
                    * (44.0 / 12.0)
                ).astype(np.float32)

                field_cdr_grid[cell] = cell_cdr_biochar

                syngas_ccs_cell = (
                    cell_demand
                    * captured_c_per_t_biochar[best_tech]
                    * (44.0 / 12.0)
                )

                cell_cdr = cell_cdr_biochar + syngas_ccs_cell
                new_cdr_reg[reg] += cell_cdr

                ch4_cell = self.ch4_emis_mitigated.values[cell]
                n2o_cell = self.n2o_emis_mitigated.values[cell]
                fossil_cell = (
                    bioenergy_c_mitig_reg[best_tech]
                    * (44.0 / 12.0)
                    * cell_demand
                )

                cell_mitig = (
                    ch4_cell
                    + n2o_cell
                    + fossil_cell
                    - (transport_emis_C * (44.0 / 12.0))
                ) # tCO2-eq

                mitig_biochar_grid[cell] = cell_cdr + cell_mitig

                ch4_mitig_reg[reg] += ch4_cell
                n2o_mitig_reg[reg] += n2o_cell
                fossil_mitig_reg[reg] += fossil_cell
                syngas_ccs_reg[reg] += syngas_ccs_cell
                biochar_cdr_reg[reg] += cell_cdr_biochar

                # Ex-post accounting for the actual selected option
                realised_costs[_RC_FEEDSTOCK, reg, best_tech] += float(
                    feedstock_cost_option[best_tech, best_res, best_bin]
                )
                realised_costs[_RC_PRODUCTION, reg, best_tech] += float(
                    production_cost_per_t_reg[best_tech] * cell_demand
                )
                realised_costs[_RC_APPLICATION, reg, best_tech] += float(
                    application_cost_cell
                )
                realised_costs[_RC_TRANSPORT, reg, best_tech] += float(
                    transport_cost
                )
                realised_costs[_RC_TRANSPORT_CARBON, reg, best_tech] += float(
                    transport_c_cost_penalty
                )
                realised_costs[_RC_YIELD_REVENUE, reg, best_tech] += float(
                    yield_revenue_option[best_tech, best_res]
                )
                realised_costs[_RC_BIOENERGY_REVENUE, reg, best_tech] += float(
                    bioenergy_value_per_t_reg[best_tech] * cell_demand
                )
                realised_mitigation[reg, best_tech] += float(
                    mitig_biochar_grid[cell]
                )
                realised_biochar[reg, best_tech] += float(cell_demand)

                continue

            if np.isfinite(best_supply_npv) and best_supply_npv > 0.0:
                if remaining_capacity_reg[best_supply_tech] < cell_demand:
                    unmet_profitable_demand[reg, best_supply_tech] += cell_demand
        
        remaining_capacity[reg,:] = remaining_capacity_reg

    self.BiocharAlloc_cum[t] = np.maximum(biochar_alloc_cum, biochar_alloc)
    self.BiocharAlloc[t] = biochar_alloc
    self.BiocharFieldCDRGrid[t] = field_cdr_grid
    self.BiocharLastApplYear[t] = last_appl_year
    self.BiocharCumulativeAppliedPerHa[t] = cumulative_applied
    self.BiocharApplicationCount[t] = application_count
    if self.agronomic_response_mode == "aging_within_window":
        self.BiocharRealisedYieldEffect[t] = np.where(
            new_yield_application,
            np.nan_to_num(self.yield_effects.values[:, 0], nan=0.0),
            0.0,
        ).astype(np.float32)
    else:
        self.BiocharRealisedYieldEffect[t] = np.zeros(len(LandGrid.coords), dtype=np.float32)

    self.BiocharCDR[t] = prev_biochar_cdr_reg + new_cdr_reg
    self.BiocharArea[t] = prev_biochar_area_reg + new_area_reg
    self.BiocharAreaTech[t] = biochar_area_tech
    self.BiocharUnmetProfitableDemand[t] = unmet_profitable_demand
    self.BiocharCH4Mitig[t] = ch4_mitig_reg
    self.BiocharN2OMitig[t] = n2o_mitig_reg
    self.BiocharFossilMitig[t] = fossil_mitig_reg
    self.BiocharSyngasCCS[t] = syngas_ccs_reg
    self.BiocharFieldCDR[t] = biochar_cdr_reg

    biochar_mitigation_grid_current = np.divide(
        mitig_biochar_grid,
        cropland_area,
        out=np.zeros(len(LandGrid.coords), dtype=np.float32),
        where=(cropland_area > 1.0) & (mitig_biochar_grid > EPS)
    ).astype(np.float32)

    self.BiocharMitigationGrid[t] = biochar_mitigation_grid_current

    self.BiocharMitigationGridCum[t] = (
        np.nan_to_num(prev_mitigation_grid_cum, nan=0.0)
        + biochar_mitigation_grid_current
    ).astype(np.float32)

    self.BiocharResidueDMUsed[t] = residue_dm_used
    _store_realised_economics(
        self, t, realised_costs, realised_mitigation, realised_biochar
    )

    if  self.proportional_residue_rationing: 
        timer_multiplier_rb, timer_residual_multiplier_rb = (
            calculate_timer_residue_multipliers(
                raw_residue_supply_dm=(
                    residue_state.raw_supply_dm
                ),
                timer_utilisation=(
                    residue_state.timer_utilisation
                ),
                timer_share=(
                    residue_state.timer_share
                ),
                actual_biochar_used_dm=(
                    residue_dm_used
                ),
                reference_biochar_residue_use=(
                    residue_state.reference_biochar_residue_use
                ),
            )
        )

        timer_multiplier_full = np.ones(
            (len(Region26.coords), len(RCCPlus.coords)),
            dtype=np.float32,
        )

        timer_residual_multiplier_full = np.ones(
            (len(Region26.coords), len(RCCPlus.coords)),
            dtype=np.float32,
        )

        timer_multiplier_full[:, :cost_bins] = timer_multiplier_rb
        timer_residual_multiplier_full[:, :cost_bins] = timer_residual_multiplier_rb

        self.TimerSharedResidueMultiplier[t] = timer_multiplier_full
        self.TimerResidualResidueMultiplier[t] = timer_residual_multiplier_full

    if self.proportional_residue_rationing: 
        print_proportional_residue_diagnostics(
            time=t,
            raw_residue_supply_dm=residue_state.raw_supply_dm,
            timer_utilisation=residue_state.timer_utilisation,
            biochar_share=residue_state.biochar_share,
            timer_share=residue_state.timer_share,
            actual_biochar_used_dm=residue_dm_used,
            reference_biochar_residue_use=residue_state.reference_biochar_residue_use,
            timer_waste_diagnostics=residue_state.timer_waste_diagnostics,
            top_n=10,
        )

    print_timestep_diagnostics(
        model=self,
        time=time,
        mitig_biochar_grid=mitig_biochar_grid,
        biochar_cdr_reg=biochar_cdr_reg,
        syngas_ccs_reg=syngas_ccs_reg,
        ch4_mitig_reg=ch4_mitig_reg,
        n2o_mitig_reg=n2o_mitig_reg,
        fossil_mitig_reg=fossil_mitig_reg,
        biochar_alloc=biochar_alloc,
        biochar_demand=biochar_demand,
        cropland_area=cropland_area,
        new_area_reg=new_area_reg,
        remaining_capacity=remaining_capacity,
        unmet_profitable_demand=unmet_profitable_demand,
        new_cdr_reg=new_cdr_reg,
    )


def compute_capacity_investments(
        self, 
        time: prism.Time
):
    t, dt = time.t, time.dt

    zero_capacity = prism.array(0.0, [Region26, PyrolysisTechnology])

    current_capacity = self.biochar_capacity.get(t, zero_capacity)

    unmet_demand = self.BiocharUnmetProfitableDemand.get(t, zero_capacity)
    annualised_unmet_demand = unmet_demand / dt.magnitude

    retired_capacity = self.retired_capacity.get(t, zero_capacity)

    max_growth_fraction = self.MaxBiocharCapacityGrowth * dt.magnitude

    max_growth_investment = prism.array(0.0, [Region26, PyrolysisTechnology])

    for reg in Region26:
        for tech in PyrolysisTechnology:
            cap = current_capacity.loc[reg, tech]

            if cap > prism.Q_(EPS, "dimensionless"):
                max_growth_investment.loc[reg, tech] = cap * max_growth_fraction
            else:
                # max_growth_investment.loc[reg, tech] = (
                #   annualised_unmet_demand.loc[reg, tech] * self.biochar_initial_capacity_fraction
                # )
                max_growth_investment.loc[reg, tech] = annualised_unmet_demand.loc[reg, tech]

    expansion_investment = np.minimum(
        annualised_unmet_demand.values,
        max_growth_investment.values
    )

    replacement_investment = retired_capacity.values

    self.biochar_capacity_investment[t] = (
        replacement_investment
        + expansion_investment
    ).astype(np.float32)
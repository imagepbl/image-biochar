import numpy as np
from timer.globals.constants import EPS

def _total(values):
    return float(
        np.nansum(values)
    )

def _ratio(numerator, denominator):
    if abs(denominator) <= EPS:
        return 0.0

    return numerator / denominator

def _time_year(time):
    value = getattr(
        time,
        "magnitude",
        time,
    )

    return int(value)

def _print_check(name, passed):
    print(
        f"  {'PASS' if passed else 'FAIL':4s}  {name}"
    )

def print_timestep_start(model, time):
    if not model.print_diagnostics:
        return

    print(
        f"Start biochar calculations for year {int(time.t.magnitude)}"
    )

def print_timestep_diagnostics(
        model,
        time,
        mitig_biochar_grid,
        biochar_cdr_reg,
        syngas_ccs_reg,
        ch4_mitig_reg,
        n2o_mitig_reg,
        fossil_mitig_reg,
        biochar_alloc,
        biochar_demand,
        cropland_area,
        new_area_reg,
        remaining_capacity,
        unmet_profitable_demand,
        new_cdr_reg,
):
    if not model.print_diagnostics:
        return

    year = int(time.t.magnitude)
    dt = float(time.dt.magnitude)

    net_mitigation_step = _total(
        mitig_biochar_grid
    )

    annual_net_mitigation = (
        net_mitigation_step
        / max(dt, EPS)
    )

    field_cdr_step = _total(
        biochar_cdr_reg
    )

    syngas_ccs_step = _total(
        syngas_ccs_reg
    )

    ch4_mitigation_step = _total(
        ch4_mitig_reg
    )

    n2o_mitigation_step = _total(
        n2o_mitig_reg
    )

    fossil_displacement_step = _total(
        fossil_mitig_reg
    )

    gross_mitigation_before_transport = (
        field_cdr_step
        + syngas_ccs_step
        + ch4_mitigation_step
        + n2o_mitigation_step
        + fossil_displacement_step
    )

    transport_emissions_step = (
        gross_mitigation_before_transport
        - net_mitigation_step
    )

    if not hasattr(
        model,
        "_diagnostic_mitigation",
    ):
        model._diagnostic_mitigation = {}

    model._diagnostic_mitigation[
        year
    ] = net_mitigation_step

    cumulative_net_mitigation = sum(
        value
        for diagnostic_year, value
        in model._diagnostic_mitigation.items()
        if diagnostic_year <= year
    )

    cumulative_cdr = _total(
        model.BiocharCDR[
            time.t
        ].values
    )

    deployed = (
        np.asarray(
            biochar_alloc
        )
        == 1
    )

    deployed_cells = int(
        np.count_nonzero(deployed)
    )

    biochar_applied_step = _total(
        np.asarray(
            biochar_demand
        )[
            deployed
        ]
    )

    treated_area_step = _total(
        np.asarray(
            cropland_area
        )[
            deployed
        ]
    )

    new_area_step = _total(
        new_area_reg
    )

    reapplication_area_step = max(
        0.0,
        treated_area_step
        - new_area_step,
    )

    cumulative_treated_land_base = _total(
        model.BiocharArea[
            time.t
        ].values
    )

    mitigation_per_t_biochar = _ratio(
        net_mitigation_step,
        biochar_applied_step,
    )

    residue_use_step = _total(
        model.BiocharResidueDMUsed[
            time.t
        ].values
    )

    residue_per_t_biochar = _ratio(
        residue_use_step,
        biochar_applied_step,
    )

    installed_capacity_rate = _total(
        model.biochar_capacity[
            time.t
        ].values
    )

    capacity_available_step = (
        installed_capacity_rate
        * dt
    )

    unused_capacity_step = _total(
        remaining_capacity
    )

    capacity_used_step = (
        capacity_available_step
        - unused_capacity_step
    )

    capacity_used_display = max(
        0.0,
        capacity_used_step,
    )

    unused_capacity_display = max(
        0.0,
        unused_capacity_step,
    )

    capacity_utilisation = _ratio(
        capacity_used_display,
        capacity_available_step,
    )

    unmet_profitable_demand_step = _total(
        unmet_profitable_demand
    )

    new_cdr_step = _total(
        new_cdr_reg
    )

    cdr_component_error = abs(
        new_cdr_step
        - field_cdr_step
        - syngas_ccs_step
    )

    capacity_drawdown_error = abs(
        capacity_used_step
        - biochar_applied_step
    )

    area_split_error = abs(
        treated_area_step
        - new_area_step
        - reapplication_area_step
    )

    finite_values = bool(
        np.all(
            np.isfinite(
                [
                    net_mitigation_step,
                    cumulative_net_mitigation,
                    cumulative_cdr,
                    biochar_applied_step,
                    treated_area_step,
                    residue_use_step,
                    capacity_available_step,
                    capacity_used_step,
                    unmet_profitable_demand_step,
                ]
            )
        )
    )

    checks = {
        "core diagnostic values are finite": finite_values,

        "new CDR = field biochar CDR + syngas CCS": (
            cdr_component_error
            <= 1e-6
            * max(
                abs(new_cdr_step),
                1.0,
            )
        ),

        "capacity drawdown matches biochar deployment": (
            capacity_drawdown_error
            <= 1e-6
            * max(
                abs(biochar_applied_step),
                1.0,
            )
        ),

        "deployment does not exceed available capacity": (
            capacity_used_step
            <= capacity_available_step
            + 1e-6
            * max(
                capacity_available_step,
                1.0,
            )
        ),

        "treated area = new area + reapplication area": (
            area_split_error
            <= 1e-6
            * max(
                treated_area_step,
                1.0,
            )
        ),

        "residue use is non-negative": (
            residue_use_step
            >= -EPS
        ),
    }

    status = (
        "PASS"
        if all(checks.values())
        else "CHECK"
    )

    print()
    print("=" * 78)
    print(
        f"BIOCHAR TIMESTEP SUMMARY | {year} | "
        f"{dt:g}-year step | {status}"
    )
    print("=" * 78)

    print()
    print(
        f"GHG mitigation (current {dt:g}-year model timestep)"
    )
    print(
        f"  Net GHG mitigation this step          "
        f"{net_mitigation_step / 1e9:10.4f} GtCO2-eq"
    )
    print(
        f"  Average annual net mitigation         "
        f"{annual_net_mitigation / 1e9:10.4f} GtCO2-eq/yr"
    )
    print(
        f"  Field biochar carbon removal          "
        f"{field_cdr_step / 1e9:10.4f} GtCO2"
    )
    print(
        f"  Syngas carbon captured and stored     "
        f"{syngas_ccs_step / 1e9:10.4f} GtCO2"
    )
    print(
        f"  Avoided CH4 emissions                 "
        f"{ch4_mitigation_step / 1e9:10.4f} GtCO2-eq"
    )
    print(
        f"  Avoided N2O emissions                 "
        f"{n2o_mitigation_step / 1e9:10.4f} GtCO2-eq"
    )
    print(
        f"  Fossil-fuel displacement              "
        f"{fossil_displacement_step / 1e9:10.4f} GtCO2-eq"
    )
    print(
        f"  Transport emissions deducted          "
        f"{transport_emissions_step / 1e9:10.4f} GtCO2-eq"
    )
    print(
        f"  Cumulative net mitigation             "
        f"{cumulative_net_mitigation / 1e9:10.4f} GtCO2-eq"
    )
    print(
        f"  Cumulative BiocharCDR model state     "
        f"{cumulative_cdr / 1e9:10.4f} GtCO2"
    )
    print(
        f"  Net mitigation per tonne biochar      "
        f"{mitigation_per_t_biochar:10.3f} tCO2-eq/t biochar"
    )

    print()
    print("Biochar deployment")
    print(
        f"  Biochar applied this step             "
        f"{biochar_applied_step / 1e6:10.3f} Mt biochar"
    )
    print(
        f"  Grid cells receiving biochar          "
        f"{deployed_cells:10d} cells"
    )
    print(
        f"  Cropland treated this step            "
        f"{treated_area_step / 1e6:10.3f} Mha"
    )
    print(
        f"  First-time treated cropland           "
        f"{new_area_step / 1e6:10.3f} Mha"
    )
    print(
        f"  Reapplication area                    "
        f"{reapplication_area_step / 1e6:10.3f} Mha"
    )
    print(
        f"  Cumulative treated land base          "
        f"{cumulative_treated_land_base / 1e6:10.3f} Mha"
    )
    print(
        f"  Residue feedstock consumed            "
        f"{residue_use_step / 1e6:10.3f} MtDM"
    )
    print(
        f"  Residue input per tonne biochar       "
        f"{residue_per_t_biochar:10.3f} tDM/t biochar"
    )

    print()
    print("Production capacity")
    print(
        f"  Installed production capacity         "
        f"{installed_capacity_rate / 1e6:10.3f} Mt biochar/yr"
    )
    print(
        f"  Capacity available over this step     "
        f"{capacity_available_step / 1e6:10.3f} Mt biochar"
    )
    print(
        f"  Capacity used over this step          "
        f"{capacity_used_display / 1e6:10.3f} Mt biochar"
    )
    print(
        f"  Unused capacity over this step        "
        f"{unused_capacity_display / 1e6:10.3f} Mt biochar"
    )
    print(
        f"  Capacity utilisation                  "
        f"{100.0 * capacity_utilisation:10.2f} %"
    )
    print(
        f"  Profitable demand blocked by capacity "
        f"{unmet_profitable_demand_step / 1e6:10.3f} Mt biochar"
    )

    print()
    print("Accounting checks")

    for name, passed in checks.items():
        _print_check(
            name=name,
            passed=passed,
        )

def print_proportional_residue_diagnostics(
        time,
        raw_residue_supply_dm,
        timer_utilisation,
        biochar_share,
        timer_share,
        actual_biochar_used_dm,
        reference_biochar_residue_use,
        timer_waste_diagnostics=None,
        top_n=10,
        enabled=True,
):

    if not enabled:
        return

    supply = np.nansum(
        np.asarray(
            raw_residue_supply_dm,
            dtype=np.float64,
        ),
        axis=1,
    )

    reference_used = np.nansum(
        np.asarray(
            reference_biochar_residue_use,
            dtype=np.float64,
        ),
        axis=1,
    )

    reference_used = np.nansum(
        reference_used,
        axis=1,
    )

    actual_used = np.asarray(
        actual_biochar_used_dm,
        dtype=np.float64,
    )

    if actual_used.ndim == 4:
        actual_used = np.nansum(
            actual_used,
            axis=1,
        )

    actual_used = np.nansum(
        actual_used,
        axis=1,
    )

    timer_utilisation = np.clip(
        np.asarray(
            timer_utilisation,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    biochar_share = np.clip(
        np.asarray(
            biochar_share,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    timer_share = np.clip(
        np.asarray(
            timer_share,
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )

    reference_biochar_utilisation = np.divide(
        reference_used,
        supply,
        out=np.zeros_like(
            supply,
            dtype=np.float64,
        ),
        where=supply > EPS,
    )

    reference_biochar_utilisation = np.clip(
        reference_biochar_utilisation,
        0.0,
        1.0,
    )

    total_claim = (
        timer_utilisation
        + reference_biochar_utilisation
    )

    oversubscribed = (
        total_claim > 1.0
    ) & (
        supply > EPS
    )

    scarcity = (
        np.maximum(
            total_claim - 1.0,
            0.0,
        )
        * supply
    )

    total_supply = _total(
        supply
    )

    timer_claim = _total(
        timer_utilisation
        * supply
    )

    reference_biochar_use = _total(
        reference_used
    )

    reference_biochar_claim = _total(
        reference_biochar_utilisation
        * supply
    )

    timer_allocated = _total(
        timer_share
        * supply
    )

    biochar_available_after_rationing = _total(
        biochar_share
        * supply
    )

    actual_biochar_use = _total(
        actual_used
    )

    unused_biochar_allocation = max(
        0.0,
        biochar_available_after_rationing
        - actual_biochar_use,
    )

    total_scarcity = _total(
        scarcity
    )

    oversubscribed_bins = int(
        np.count_nonzero(
            oversubscribed
        )
    )

    available_bins = int(
        np.count_nonzero(
            supply > EPS
        )
    )

    timer_allocated_share = _ratio(
        timer_allocated,
        total_supply,
    )

    biochar_available_share = _ratio(
        biochar_available_after_rationing,
        total_supply,
    )

    actual_biochar_use_share = _ratio(
        actual_biochar_use,
        total_supply,
    )

    year = _time_year(time)

    print()
    print("-" * 78)
    print(
        f"PROPORTIONAL RESIDUE RATIONING | {year}"
    )
    print("-" * 78)

    print()
    print("Shared residue resource")
    print(
        f"  Total Biochar-accessible supply       "
        f"{total_supply / 1e6:10.3f} MtDM"
    )
    print(
        f"  TIMER claim on shared residues        "
        f"{timer_claim / 1e6:10.3f} MtDM"
    )
    print(
        f"  Biochar reference-run residue use     "
        f"{reference_biochar_use / 1e6:10.3f} MtDM"
    )
    print(
        f"  Biochar claim used for rationing      "
        f"{reference_biochar_claim / 1e6:10.3f} MtDM"
    )
    print(
        f"  Total oversubscription                "
        f"{total_scarcity / 1e6:10.3f} MtDM"
    )
    print(
        f"  Oversubscribed region/bin combinations"
        f"{oversubscribed_bins:10d} / {available_bins:d}"
    )

    print()
    print("Allocation after proportional rationing")
    print(
        f"  TIMER allocation                      "
        f"{timer_allocated / 1e6:10.3f} MtDM "
        f"({100.0 * timer_allocated_share:6.2f}% of supply)"
    )
    print(
        f"  Biochar maximum available supply      "
        f"{biochar_available_after_rationing / 1e6:10.3f} MtDM "
        f"({100.0 * biochar_available_share:6.2f}% of supply)"
    )
    print(
        f"  Biochar actual realised use           "
        f"{actual_biochar_use / 1e6:10.3f} MtDM "
        f"({100.0 * actual_biochar_use_share:6.2f}% of supply)"
    )
    print(
        f"  Biochar allocation left unused        "
        f"{unused_biochar_allocation / 1e6:10.3f} MtDM"
    )

    if timer_waste_diagnostics is not None:
        timer_exogenous_use = _total(
            timer_waste_diagnostics[
                "timer_exo_use_gj"
            ]
        )

        timer_shared_use = _total(
            timer_waste_diagnostics[
                "timer_shared_use_gj"
            ]
        )

        print()
        print("TIMER residue accounting")
        print(
            f"  TIMER exogenous/waste residue use     "
            f"{timer_exogenous_use / 1e9:10.3f} EJ/yr"
        )
        print(
            f"  TIMER shared residue use              "
            f"{timer_shared_use / 1e9:10.3f} EJ/yr"
        )

    if (
        top_n is None
        or top_n <= 0
        or oversubscribed_bins == 0
    ):
        return

    print()
    print(
        f"Largest residue shortages (top {top_n})"
    )
    print(
        "  Region   Bin   Combined claim   Shortage [MtDM]"
    )

    order = np.argsort(
        scarcity.ravel()
    )[::-1]

    shown = 0

    for index in order:
        region, cost_bin = np.unravel_index(
            index,
            scarcity.shape,
        )

        if (
            scarcity[region, cost_bin] <= EPS
            or shown >= top_n
        ):
            break

        print(
            f"  {region + 1:6d} "
            f"{cost_bin + 1:5d} "
            f"{total_claim[region, cost_bin]:16.3f} "
            f"{scarcity[region, cost_bin] / 1e6:17.3f}"
        )

        shown += 1
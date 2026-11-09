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



def _as_values(value, dtype=np.float64):
    """Accept a PRISM array or a plain ndarray without modifying it."""
    return np.asarray(getattr(value, "values", value), dtype=dtype)


def _signed_statistics(values):
    """Report positive and negative contributions separately to avoid cancellation."""
    x = np.asarray(values, dtype=np.float64)
    good = np.isfinite(x)
    valid = x[good]
    if valid.size == 0:
        return dict(n_valid=0, n_positive=0, n_negative=0,
                    positive=0.0, negative=0.0, net=0.0)
    pos = valid[valid > 0.0]
    neg = valid[valid < 0.0]
    return dict(
        n_valid=int(valid.size),
        n_positive=int(pos.size), n_negative=int(neg.size),
        positive=float(pos.sum(dtype=np.float64)),
        negative=float(neg.sum(dtype=np.float64)),
        net=float(valid.sum(dtype=np.float64)),
    )


def print_land_rent_yield_diagnostics(
        model, time, LandRent, LandRentBiochar, cropland_area,
):
    """Print CURRENT-timestep pre-allocation agricultural revenue diagnostics.

    Call AFTER determine_land_suitability() has populated self.yield_revenues,
    self.yield_revenues_ha[t] and BiocharExpectedNPVHa[t], and BEFORE allocation.

    LandRent and LandRentBiochar are PRISM input arrays in economics.py, not
    attributes of the Biochar model, so this function needs an explicit call.
    It deliberately does not change any model state or overwrite revenues.
    """
    if not getattr(model, "print_diagnostics", False):
        return

    year = _time_year(time.t)
    a = _as_values(cropland_area)
    rent = _as_values(LandRent)
    rent_bc = _as_values(LandRentBiochar)
    yield_pct = _as_values(model.yield_effects)
    revenue = _as_values(model.yield_revenues)

    if not (a.ndim == rent.ndim == rent_bc.ndim == 1):
        print("  [RENT QA] Cannot diagnose land rent: expected 1-D grids.")
        return
    if not (a.shape == rent.shape == rent_bc.shape):
        print("  [RENT QA] Land rent and area have different grid shapes.")
        return
    if (yield_pct.ndim != 2 or revenue.ndim != 2
            or yield_pct.shape[0] != len(a)
            or revenue.shape[0] != len(a)):
        print("  [RENT QA] Technology-resolved yield fields have invalid shapes.")
        return

    cropland = np.isfinite(a) & (a > 1.0)
    finite_rent = cropland & np.isfinite(rent) & np.isfinite(rent_bc)
    good_yield = cropland & np.isfinite(yield_pct[:, 0])
    n_cropland = int(cropland.sum())
    n_finite_rent = int(finite_rent.sum())
    n_bad = n_cropland - n_finite_rent
    rent_positive = int(np.count_nonzero(finite_rent & (rent > 0.0)))
    rent_biochar_positive = int(np.count_nonzero(finite_rent & (rent_bc > 0.0)))
    delta = rent_bc - rent

    area_rent = a[finite_rent]
    if area_rent.size > 0 and float(area_rent.sum()) > 0:
        weight = float(area_rent.sum())
        baseline_mean = float(np.sum(rent[finite_rent] * area_rent) / weight)
        biochar_mean = float(np.sum(rent_bc[finite_rent] * area_rent) / weight)
        delta_mean = float(np.sum(delta[finite_rent] * area_rent) / weight)
    else:
        baseline_mean = biochar_mean = delta_mean = float("nan")

    # The uploaded land-rent module implements LandRentBiochar =
    # LandRent * (1 + yield_effect_pct/100). Failure of this check can reveal
    # a stale YieldEffectsBiochar array used by landrent.compute_values().
    rent_yield_mask = finite_rent & good_yield
    n_rent_mismatch = 0
    max_rent_mismatch = float("nan")
    if np.any(rent_yield_mask):
        expected = rent[rent_yield_mask] * (
            yield_pct[rent_yield_mask, 0] / 100.0
        )
        difference = np.abs(delta[rent_yield_mask] - expected)
        tolerance = np.maximum(
            1e-2,
            1e-4 * np.maximum(np.abs(rent[rent_yield_mask]),
                              np.abs(rent_bc[rent_yield_mask])),
        )
        n_rent_mismatch = int(np.count_nonzero(difference > tolerance))
        max_rent_mismatch = float(difference.max())

    n_nonfinite_yield = n_cropland - int(good_yield.sum())
    if np.any(good_yield):
        yield_values = yield_pct[good_yield, 0]
        weighted_yield = float(
            np.sum(yield_values * a[good_yield]) / np.sum(a[good_yield])
        )
        minimum_yield = float(np.min(yield_values))
        maximum_yield = float(np.max(yield_values))
        n_pos_yield = int((yield_values > 0).sum())
        n_neg_yield = int((yield_values < 0).sum())
    else:
        weighted_yield = minimum_yield = maximum_yield = float("nan")
        n_pos_yield = n_neg_yield = 0

    # Currency in this stage is native to land rent/economics, not USD2025.
    # Potential revenue reflects ALL eligible cropland, not selected cells.
    tech_revenue = revenue[cropland, :]
    n_years = int(round(float(model.application_interval)))
    inferred_native = np.where(
        finite_rent,
        delta * a / 100.0 * n_years,
        0.0,
    )
    revenue_ha = _as_values(model.yield_revenues_ha[time.t])

    print()
    print("-" * 78)
    print(f"LAND RENT AND POTENTIAL CROP-YIELD REVENUE | {year} | PRE-ALLOCATION")
    print("-" * 78)
    print(f"  Agronomic response mode              {model.agronomic_response_mode}")
    if getattr(model, "agronomic_response_mode", "fixed_window") == "aging_within_window":
        try:
            from timer.cdr.biochar.application_management import mean_within_window_aging_multiplier
            factor = mean_within_window_aging_multiplier(
                benefit_years=n_years,
                full_effect_years=int(model.agronomic_full_effect_years),
                final_year_retention=float(model.agronomic_year5_retention),
            )
            print(f"  Expected within-window aging factor  {factor:14.6f} (fraction)")
        except (ImportError, ValueError, AttributeError, TypeError) as exc:
            print(f"  Expected aging factor unavailable: {exc}")
    print(f"  Cropland cells >1 ha                 {n_cropland:13,d}")
    print(f"  Cells with finite baseline/BC rent  {n_finite_rent:13,d}")
    print(f"  Cells with nonfinite rent            {n_bad:13,d}")
    print(f"  Cells with positive baseline rent    {rent_positive:13,d}")
    print(f"  Cells with positive BC rent          {rent_biochar_positive:13,d}")
    print(f"  Mean baseline rent (area weighted)   {baseline_mean:14,.5g} native/km2/yr")
    print(f"  Mean biochar rent (area weighted)    {biochar_mean:14,.5g} native/km2/yr")
    print(f"  Mean land-rent increment             {delta_mean:14,.5g} native/km2/yr")
    if np.any(finite_rent):
        d = delta[finite_rent]
        print(f"  Positive/negative land-rent deltas   {int(np.sum(d > 0)):6,d} / {int(np.sum(d < 0)):6,d}")
        print(f"  Min/max baseline land rent           {np.min(rent[finite_rent]):12,.5g} / {np.max(rent[finite_rent]):12,.5g} native/km2/yr")
    print(f"  Cells with nonfinite yield effect    {n_nonfinite_yield:13,d}")
    print(f"  Positive/negative yield cells        {n_pos_yield:6,d} / {n_neg_yield:6,d}")
    print(f"  Mean LT yield response (area wtd)    {weighted_yield:14,.4f} %")
    print(f"  Min/max LT yield response            {minimum_yield:8,.3f} / {maximum_yield:8,.3f} %")
    print(f"  Land-rent/yield identity mismatches  {n_rent_mismatch:13,d}")
    print(f"  Max absolute land-rent mismatch      {max_rent_mismatch:14,.5g} native/km2/yr")
    print(f"  Total implied land-rent revenue       {np.nansum(inferred_native)/1e9:14,.5g} bn native/step")

    for tech in range(revenue.shape[1]):
        stats = _signed_statistics(tech_revenue[:, tech])
        print(
            f"  Potential yield revenue, tech {tech}    "
            f"+{stats['positive']/1e9:.5g} / "
            f"{stats['negative']/1e9:.5g} / "
            f"net {stats['net']/1e9:.5g} bn native/step"
        )
        print(
            f"  Revenue-bearing cells, tech {tech}     "
            f"positive={stats['n_positive']:,}, "
            f"negative={stats['n_negative']:,}"
        )

    if revenue_ha.ndim == 1 and len(revenue_ha) == len(a):
        good_ha = cropland & np.isfinite(revenue_ha)
        if good_ha.any():
            print(
                f"  Potential revenue/ha finite min/max  "
                f"{np.min(revenue_ha[good_ha]):.5g} / "
                f"{np.max(revenue_ha[good_ha]):.5g} native/ha/step"
            )

    try:
        npv = _as_values(model.BiocharExpectedNPVHa[time.t])
        if npv.ndim == 2 and npv.shape[0] == len(a):
            for tech in range(npv.shape[1]):
                good = cropland & np.isfinite(npv[:, tech])
                profitable = good & (npv[:, tech] > 0.0)
                max_npv = float(np.max(npv[good, tech])) if good.any() else np.nan
                print(
                    f"  Expected NPV tech {tech}: finite/positive "
                    f"{int(good.sum()):,} / {int(profitable.sum()):,}; "
                    f"max={max_npv:.6g} native/ha"
                )
            print("  NOTE: expected NPV is a ranking approximation, not realised option NPV.")
    except (KeyError, IndexError, AttributeError, ValueError) as exc:
        print(f"  Expected-NPV diagnostics unavailable: {exc}")

    if n_cropland and n_finite_rent == 0:
        print("  WARNING: all cropland has missing/nonfinite land rent!")
    if n_rent_mismatch:
        print("  WARNING: biochar rent differs from current yield-effect map.")
        print("           Check that soil effects were updated BEFORE land rent.")
    if n_cropland and n_nonfinite_yield:
        print("  WARNING: nonfinite yield effects on cropland can exclude cells from NPV ranking.")
    if not np.any(np.abs(tech_revenue) > 1e-6):
        if getattr(model, "simplify_biochar", False):
            print("  INFO: zero potential crop-yield revenues are intended in the main stylised run.")
        else:
            print("  WARNING: potential crop-yield revenues are zero everywhere on cropland.")




def _print_realised_yield_revenue_diagnostics(model, time, deployed_cells):
    """Report post-allocation economics from the actual selected options."""
    t = time.t
    scale = float(getattr(model, "inflator_usd_2025", 1.0))
    try:
        rev = _as_values(model.BiocharRealisedRevenueYield[t])
        prod = _as_values(model.BiocharRealisedBiocharProduction[t])
        feed = _as_values(model.BiocharResidueDMUsed[t])
        possible = _as_values(model.yield_revenues)
    except (KeyError, IndexError, AttributeError, ValueError) as exc:
        print(f"  Yield-revenue accounting not yet available: {exc}")
        return

    print()
    print("Yield-revenue accounting (AFTER allocation)")
    print(f"  Deployed cropland cells              {deployed_cells:13,d}")
    print(f"  Total realised revenue              {np.nansum(rev)/1e9:14,.5g} bn USD2025/step")
    print(f"  Realised revenue (annual equivalent) {np.nansum(rev)/float(time.dt.magnitude)/1e9:14,.5g} bn USD2025/yr")
    for tech in range(rev.shape[1]):
        positive = rev[:, tech][rev[:, tech] > 0]
        negative = rev[:, tech][rev[:, tech] < 0]
        print(
            f"  Technology {tech}: revenue +/-/net "
            f"{positive.sum(dtype=np.float64)/1e9:.5g} / "
            f"{negative.sum(dtype=np.float64)/1e9:.5g} / "
            f"{rev[:, tech].sum(dtype=np.float64)/1e9:.5g} bn USD2025/step"
        )
        print(
            f"  Technology {tech}: applied biochar    "
            f"{prod[:, tech].sum(dtype=np.float64)/1e6:,.4f} Mt/step"
        )

    # Positive potential and zero realised revenue can reflect high-lignin
    # feedstock selection; print actual LT feedstock eligibility for diagnosis.
    try:
        from timer.cdr.biochar.feedstock import get_ligneous_feedstock_mask
        mask = np.asarray(get_ligneous_feedstock_mask(model), dtype=bool)
        if feed.ndim == 4 and feed.shape[2] == len(mask):
            lt_used = float(np.nansum(feed[:, 0, :, :]))
            lt_eligible = float(np.nansum(feed[:, 0, ~mask, :]))
            print(
                f"  LT residue use eligible for yield    "
                f"{lt_eligible / 1e6:,.4f} / {lt_used / 1e6:,.4f} MtDM/step "
                f"({100 * lt_eligible / lt_used:.2f}% of LT use)"
                if lt_used > EPS else
                "  LT residue use eligible for yield    0 MtDM (no LT feedstock used)"
            )
    except (ValueError, KeyError, TypeError, AttributeError, ImportError) as exc:
        print(f"  Could not classify LT residue eligibility: {exc}")

    pot = _signed_statistics(possible)
    realised_absolute = float(np.nansum(np.abs(rev)))
    if pot['positive'] > 1.0 and realised_absolute < 1.0:
        if deployed_cells == 0:
            print("  ALERT: Positive POTENTIAL revenues but NO deployment;")
            print("         inspect feasible option NPVs and capacity-investment trigger.")
        else:
            print("  ALERT: Positive POTENTIAL revenues but realised revenues are zero;")
            print("         inspect selected feedstock eligibility and cell-level NPV.")
    elif pot['positive'] <= 1.0:
        print("  NOTE: No positive technology-specific potential revenues reached allocation.")


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

    _print_realised_yield_revenue_diagnostics(
        model=model,
        time=time,
        deployed_cells=deployed_cells,
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
import numpy as np
from dataclasses import dataclass
from timer.globals.constants import EPS
from timer.cdr.cdr_util import (
    BiocharFeedstock,
    PyrolysisTechnology,
    Region26,
    RCCPlus,
)
from timer.cdr.biochar.residue_rationing import (
    get_timer_shared_residue_utilisation,
    read_reference_biochar_residue_use,
    apply_proportional_residue_rationing,
)


BIOCHAR_FEEDSTOCK_NAMES = (
    "temperate_cereals",
    "rice",
    "maize",
    "tropical_cereals",
    "pulses",
    "roots_tubers",
    "oil_crops",
    "fibre",
    "fruit",
    "vegetables",
    "sugarcane",
    "biofuel_sugarcane",
    "biofuel_maize",
    "biofuel_woody",
    "biofuel_nwoody",
    "forestry",
)

WOODY_FEEDSTOCK_NAMES = {
    "biofuel_woody",
    "forestry",
}


@dataclass
class ResidueSupply:
    supply_dm: np.ndarray
    raw_supply_dm: np.ndarray | None = None
    timer_utilisation: np.ndarray | None = None
    timer_waste_diagnostics: dict[str, np.ndarray] | None = None
    biochar_share: np.ndarray | None = None
    timer_share: np.ndarray | None = None
    reference_biochar_residue_use: np.ndarray | None = None

def get_cost_bins():
    cost_bins = len(RCCPlus.coords) - 1
    return cost_bins, slice(0, cost_bins)

def _get_rationing_inputs(
        model,
        time,
):
    year = int( getattr(time, "magnitude", time))

    cache = getattr(
        model,
        "_residue_rationing_input_cache",
        None,
    )

    if cache is None:
        cache = {}
        model._residue_rationing_input_cache = cache

    if year not in cache:
        timer_utilisation, timer_waste_diagnostics = (
            get_timer_shared_residue_utilisation(model=model, time=time)
        )

        reference_biochar_residue_use = (
            read_reference_biochar_residue_use(model=model,time=time)
        )

        cache[year] = (
            timer_utilisation,
            timer_waste_diagnostics,
            reference_biochar_residue_use,
        )

    return cache[year]

def prepare_residue_supply(
        model,
        time,
        ResPotCurveDM,
        copy=False,
        collect_diagnostics=False,
):
    _, cost_bins_slice = get_cost_bins()

    residue_supply_dm = (
        ResPotCurveDM.values[:, :len(BiocharFeedstock.coords), cost_bins_slice].astype(np.float32, copy=copy)
        * time.dt.magnitude
    )

    residue_supply_dm = np.where(
        np.isfinite(residue_supply_dm),
        residue_supply_dm,
        0.0,
    ).astype(np.float32, copy=False)

    if not model.proportional_residue_rationing:
        return ResidueSupply(supply_dm=residue_supply_dm)
    raw_residue_supply_dm = (residue_supply_dm.copy() if collect_diagnostics else None)

    timer_utilisation, timer_waste_diagnostics, reference_biochar_residue_use = _get_rationing_inputs(
        model=model, time=time.t,
    )

    residue_supply_dm, biochar_share, timer_share = apply_proportional_residue_rationing(
        residue_supply_dm=residue_supply_dm,
        timer_utilisation=timer_utilisation,
        reference_biochar_residue_use=reference_biochar_residue_use
    )

    return ResidueSupply(
        supply_dm=residue_supply_dm,
        raw_supply_dm = raw_residue_supply_dm if collect_diagnostics else None,
        timer_utilisation= timer_utilisation if collect_diagnostics else None,
        timer_waste_diagnostics=timer_waste_diagnostics if collect_diagnostics else None,
        biochar_share= biochar_share if collect_diagnostics else None,
        timer_share= timer_share if collect_diagnostics else None,
        reference_biochar_residue_use=reference_biochar_residue_use if collect_diagnostics else None,
    )

def prepare_residue_costs(
        ResCostCurveDM,
):
    _, cost_bins_slice = get_cost_bins()

    residue_cost_dm = (
        ResCostCurveDM.values[:, :len(BiocharFeedstock.coords), cost_bins_slice].astype(np.float32, copy=False)
    )

    return np.where(
        np.isfinite(residue_cost_dm),
        residue_cost_dm,
        np.inf,
    ).astype(np.float32, copy=False)


def get_ligneous_feedstock_mask(
        model,
):
    woody_mask = np.array([name in WOODY_FEEDSTOCK_NAMES for name in BIOCHAR_FEEDSTOCK_NAMES], dtype=bool)

    high_lignin_low_ash_mask = (
        (model.lignin_content.values >= 0.15)
        & (model.ash_content.values <= 0.06)
    )

    return (woody_mask | high_lignin_low_ash_mask)

@dataclass
class SimplifiedFeedstockProperties:
    feedstock_input: np.ndarray
    carbon_fraction: np.ndarray
    decay_rate: np.ndarray

def _time_year(time):
    value = getattr(time, "t", time)
    value = getattr(value, "magnitude", value)
    return int(value)

def _cropland_weighted_stable_fraction_by_region_feedstock(
        model,
        tech,
        cropland_area,
):
    cropland_area = np.asarray(cropland_area, dtype=np.float64)
    regions = np.asarray(model.region_cells.values)

    decay_grid = np.asarray(
        model.decay_rate.values[:, tech, :], dtype=np.float64,
    )
    stable_grid = np.exp(-np.maximum(decay_grid, 0.0) * 100.0)

    nreg = len(Region26.coords)
    nfeed = len(BiocharFeedstock.coords)

    reference_stable = np.exp(
        -np.maximum(
            np.asarray(model.decay_rate_ref.values[tech, :], dtype=np.float64),
            0.0,
        )
        * 100.0
    )

    eligible_global = np.isfinite(cropland_area) & (cropland_area > 1.0)

    global_num = np.sum(
        np.where(
            eligible_global[:, None] & np.isfinite(stable_grid),
            stable_grid * cropland_area[:, None],
            0.0,
        ),
        axis=0,
        dtype=np.float64,
    )
    global_den = np.sum(
        np.where(
            eligible_global[:, None] & np.isfinite(stable_grid),
            cropland_area[:, None],
            0.0,
        ),
        axis=0,
        dtype=np.float64,
    )

    global_cropland_stable = np.divide(
        global_num,
        global_den,
        out=reference_stable.copy(),
        where=global_den > EPS,
    )

    stable_region_feed = np.zeros(
        (nreg, nfeed),
        dtype=np.float64,
    )

    for reg in Region26:
        eligible = (regions == reg)  & eligible_global

        if not np.any(eligible):
            stable_region_feed[reg, :] = global_cropland_stable
            continue

        reg_weights = cropland_area[eligible, None]
        reg_stable = stable_grid[eligible, :]

        valid = np.isfinite(reg_stable)
        numerator = np.sum(
            np.where(valid, reg_stable * reg_weights, 0.0),
            axis=0,
            dtype=np.float64,
        )
        denominator = np.sum(
            np.where(valid, reg_weights, 0.0),
            axis=0,
            dtype=np.float64,
        )

        stable_region_feed[reg, :] = np.divide(
            numerator,
            denominator,
            out=global_cropland_stable.copy(),
            where=denominator > EPS,
        )

    return np.clip(
        stable_region_feed,
        1e-8,
        1.0,
    )


def _aggregate_generic_feedstock_properties(
        model,
        supply_by_feedstock,
        cropland_area,
):
    tech = int(getattr(model, "simplified_technology_index", 1))

    base_input = np.asarray(
        model.BiocharFeedstockInput.values[tech, :],
        dtype=np.float64,
    )
    base_carbon_fraction = np.asarray(
        model.BiocharCarbonFraction.values[tech, :],
        dtype=np.float64,
    )

    valid_input = np.isfinite(base_input) & (base_input > EPS)
    biochar_yield = np.divide(
        1.0,
        base_input,
        out=np.zeros_like(base_input, dtype=np.float64),
        where=valid_input,
    )

    supply_by_feedstock = np.asarray(
        supply_by_feedstock,
        dtype=np.float64,
    )
    supply_by_feedstock = np.where(
        np.isfinite(supply_by_feedstock) & (supply_by_feedstock > 0.0),
        supply_by_feedstock,
        0.0,
    )

    # Soil temperature adjusted 100-year persistence by region/feedstock
    stable_region_feed = (
        _cropland_weighted_stable_fraction_by_region_feedstock(
            model=model,
            tech=tech,
            cropland_area=cropland_area,
        )
    )

    global_supply = np.sum(
        supply_by_feedstock,
        axis=0,
        dtype=np.float64,
    )
    if np.sum(global_supply) <= EPS:
        global_supply = np.ones_like(base_input, dtype=np.float64)

    carbon_output_reg_feed = (
        supply_by_feedstock
        * biochar_yield[None, :]
        * base_carbon_fraction[None, :]
    )

    valid_persistence_weight = (
        np.isfinite(carbon_output_reg_feed)
        & (carbon_output_reg_feed > 0.0)
        & np.isfinite(stable_region_feed)
    )

    persistence_denominator = np.sum(
        np.where(
            valid_persistence_weight,
            carbon_output_reg_feed,
            0.0,
        ), dtype=np.float64,
    )

    if persistence_denominator > EPS:
        global_stable_fraction = (
            np.sum(
                np.where(
                    valid_persistence_weight,
                    carbon_output_reg_feed * stable_region_feed,
                    0.0,
                ), dtype=np.float64,
            ) / persistence_denominator
        )
    else:
        global_stable_fraction = float(
            np.nanmean(stable_region_feed)
        )

    global_stable_fraction = float(np.clip(global_stable_fraction, 1e-8, 1.0))
    global_decay_rate = float(-np.log(global_stable_fraction) / 100.0)

    model._simplified_global_stable_fraction_100y = (
        global_stable_fraction
    )
    model._simplified_global_decay_rate = global_decay_rate

    def aggregate_region(supply):
        total_dm = np.sum(supply, dtype=np.float64)

        if total_dm <= EPS:
            supply = global_supply
            total_dm = np.sum(supply, dtype=np.float64)

        char_output = supply * biochar_yield
        total_char = np.sum(char_output, dtype=np.float64)

        if total_char <= EPS:
            generic_yield = (
                np.mean(biochar_yield[valid_input])
                if np.any(valid_input)
                else 0.0
            )
            generic_input = (
                1.0 / generic_yield
                if generic_yield > EPS
                else np.inf
            )
            generic_c_fraction = float(
                np.nanmean(base_carbon_fraction)
            )
        else:
            generic_yield = total_char / total_dm
            generic_input = 1.0 / generic_yield
            generic_c_fraction = float(
                np.sum(
                    char_output * base_carbon_fraction,
                    dtype=np.float64,
                ) / total_char
            )

        return float(generic_input), float(generic_c_fraction),

    nreg = len(Region26.coords)
    ntech = len(PyrolysisTechnology.coords)
    nfeed = len(BiocharFeedstock.coords)

    feedstock_input = np.zeros((nreg, ntech, nfeed), dtype=np.float32)
    carbon_fraction = np.zeros_like(feedstock_input)
    decay_rate = np.full_like(
        feedstock_input,
        np.float32(global_decay_rate),
    )

    for reg in Region26:
        generic_input, generic_c = aggregate_region(
            supply_by_feedstock[reg, :]
        )
        feedstock_input[reg, :, :] = generic_input
        carbon_fraction[reg, :, :] = generic_c

    return SimplifiedFeedstockProperties(
        feedstock_input=feedstock_input,
        carbon_fraction=carbon_fraction,
        decay_rate=decay_rate,
    )


def get_simplified_feedstock_properties(
        model,
        time,
        residue_supply_dm,
        cropland_area,
):
    year = _time_year(time)
    baseline_year = int(getattr(model, "simplified_baseline_year", 2025))

    cache = getattr(
        model,
        "_simplified_feedstock_property_cache",
        None,
    )
    if cache is None:
        cache = {}
        model._simplified_feedstock_property_cache = cache

    if "baseline" in cache and year >= baseline_year:
        return cache["baseline"]

    key = (
        "baseline"
        if year >= baseline_year
        else f"prebaseline_{year}"
    )

    if key not in cache:
        supply_by_feedstock = np.sum(
            np.asarray(residue_supply_dm, dtype=np.float64),
            axis=2,
            dtype=np.float64,
        )

        cache[key] = _aggregate_generic_feedstock_properties(
            model=model,
            supply_by_feedstock=supply_by_feedstock,
            cropland_area=cropland_area,
        )

        if key == "baseline":
            cache["baseline_year"] = year

    return cache[key]

def get_cached_simplified_feedstock_properties(
        model,
        time,
):
    year = _time_year(time)
    baseline_year = int(
        getattr(model, "simplified_baseline_year", 2025)
    )

    cache = getattr(
        model,
        "_simplified_feedstock_property_cache",
        None,
    )

    if year >= baseline_year and "baseline" in cache:
        return cache["baseline"]

    key = f"prebaseline_{year}"
    if key in cache:
        return cache[key]
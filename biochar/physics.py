import prism
import numpy as np
import pandas as pd
import csv
import warnings
from pathlib import Path
from timer.cdr.biochar.persistence_models import (
    fraction_remaining_100y,
    equivalent_decay_rate_ref,
)
from timer.cdr.biochar.application_management import mean_within_window_aging_multiplier
from timer.globals.constants import EPS
from timer.cdr.cdr_util import (
    biochar_data_dir,
    LandGrid,
    LandCover,
    Crop38,
    Fertiliser,
    SoilDepth,
)

def calculate_biochar_mass_balance(temperatures_C, lignin_percent, feedstock_ash_fraction):
    temp = np.asarray(temperatures_C, dtype=np.float64)
    lignin = np.asarray(lignin_percent, dtype=np.float64)
    ash = np.asarray(feedstock_ash_fraction, dtype=np.float64)

    char_c_daf = 0.93 - 0.92 * np.exp(-0.0042 * temp)
    char_yield_daf = (0.1261
                      + 0.5391 * np.exp(-0.004 * temp[:, None])
                      + 0.2733 * lignin[None, :])
    if np.any((char_yield_daf <= 0.0) | (char_yield_daf > 1.0)):
        raise ValueError("Biochar DAF yield is outside physical range (0, 1]")

    char_daf_mass = (1.0 - ash[None, :]) * char_yield_daf
    char_dry_mass = ash[None, :] + char_daf_mass
    char_ash_fraction = ash[None, :] / char_dry_mass
    char_carbon_fraction_dry = char_c_daf[:, None] * (1.0 - char_ash_fraction)
    feedstock_per_biochar = 1.0 / char_dry_mass

    return (char_ash_fraction.astype(np.float32),
            char_dry_mass.astype(np.float32),
            char_carbon_fraction_dry.astype(np.float32),
            feedstock_per_biochar.astype(np.float32))

def configure_simplified_technology(self):
    if not self._simplification_enabled():
        return

    if getattr(self, "_simplified_technology_configured", False):
        return

    tech = int(getattr(self, "simplified_technology_index", 1))
    ntech = len(self.pyrolysis_temp.values)

    # Save the original endpoints for traceability and for the syngas interpolation before mutating the selected generic tech slot
    original = {
        "temperature": np.asarray(self.pyrolysis_temp.values).copy(),
        "capex": np.asarray(self.pyrolysis_capex.values).copy(),
        "opex": np.asarray(self.pyrolysis_opex.values).copy(),
        "prod_rate": np.asarray(self.pyrolysis_prod_rate.values).copy(),
        "bio_oil_yield": np.asarray(self.pyrolysis_bio_oil_yield.values).copy(),
        "syngas_carbon_flow": np.asarray(
            self.pyrolysis_syngas_carbon_flow.values
        ).copy(),
    }
    self._original_pyrolysis_parameters = original

    # Bahiri et al. 400 deg C configuration
    self.pyrolysis_temp.values[tech] = np.float32(400.0)
    self.pyrolysis_prod_rate.values[tech] = np.float32(3.4794)
    self.pyrolysis_capex.values[tech] = np.float32(34.02e6)
    self.pyrolysis_opex.values[tech] = np.float32(117.0)
    self.pyrolysis_bio_oil_yield.values[tech] = np.float32(13.35)

    # Interpolate between the 300C and 500C carbon flow values using the relative gas yield position of the 400 deg C case
    # (30.89 - 19.70) / (39.41 - 19.70) = 0.568
    syngas_300 = float(original["syngas_carbon_flow"][0])
    syngas_500 = float(original["syngas_carbon_flow"][1])
    self.pyrolysis_syngas_carbon_flow.values[tech] = np.float32(
        syngas_300 + 0.568 * (syngas_500 - syngas_300)
    )

    self.disabled_technologies = tuple(
        idx for idx in range(ntech) if idx != tech
    )

    self._simplified_technology_configured = True

def _weighted_mean(values, weights, mask):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    base_mask = np.asarray(mask, dtype=bool) & np.isfinite(values)

    weighted_mask = (
        base_mask
        & np.isfinite(weights)
        & (weights > 0.0)
    )

    if np.any(weighted_mask):
        return float(
            np.sum(
                values[weighted_mask] * weights[weighted_mask],
                dtype=np.float64,
            )
            / np.sum(weights[weighted_mask], dtype=np.float64)
        )

    if np.any(base_mask):
        return float(np.nanmean(values[base_mask]))

    finite = np.isfinite(values)
    if np.any(finite):
        return float(np.nanmean(values[finite]))

    return 0.0

def compute_carbon_content(self):
    """
    Calculate carbon content of biochar

    References: 
    - Woolf et al. 2021: Greenhouse gas inventory model for biochar additions to soil
    - Phyllis2: Database for biomass and waste
    """
    ash_char, yield_dm, carbon_fraction, feedstock_input = calculate_biochar_mass_balance(
        self.pyrolysis_temp.values,
        self.lignin_content.values,
        self.ash_content.values,
    )
    self.BiocharCarbonFraction.values = carbon_fraction
    self.BiocharFeedstockInput.values = feedstock_input


def compute_decay_rate(
        self,
        time: prism.Time,
        SOILTEMP: prism.Array[LandGrid, SoilDepth, np.float32]
):

    """
    Calculate annual decay rate of biochar 

    References:
    - Azzi et al. 2024: Modelling biochar long-term carbon storage in soil with harmonised analysis of decomposition data
    - Thauront et al. 2024: A critical re-analysis of biochar properties prediction from production parameters and elemental analysis
    - Gridded soil temperature from LPJmL
    """
    # Estimate the molar ratio of H to organic C based on feedstock type and pyrolysis temperature
    hc_ratio = np.exp(
        self.hc_intercept.values[None, :] + self.hc_slope.values[None, :] * self.pyrolysis_temp.values[:, None]
    )

    # Select 100-year retained fraction at 20 C from Azzi et al. 2024
    persistence_model = getattr(self, "carbon_persistence_model", "azzi_power")
    stable_fraction_100y = fraction_remaining_100y(hc_ratio, persistence_model)
    decay_rate_ref = equivalent_decay_rate_ref(hc_ratio, persistence_model)
    self.decay_rate_ref.values = decay_rate_ref.astype(np.float32)

    if not getattr(self, "_persistence_diagnostics_written", False):
        if np.any((hc_ratio < 0.2) | (hc_ratio > 0.7)):
            warnings.warn(
                f"{persistence_model}: predicted H/C extends outside 0.2–0.7",
                UserWarning,
                stacklevel=2,
            )
        path = Path(self.output_dir) / "persistence_reference_20C.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "carbon_persistence_model", "tech_index", "pyrolysis_temp_C",
                "feedstock_index", "H_Corg_molar", "retained_fraction_100y_20C",
                "equivalent_decay_rate_per_year_20C",
            ])
            for tech_index, temperature in enumerate(self.pyrolysis_temp.values):
                for feedstock_index in range(hc_ratio.shape[1]):
                    writer.writerow([
                        persistence_model, tech_index, float(temperature),
                        feedstock_index, float(hc_ratio[tech_index, feedstock_index]),
                        float(stable_fraction_100y[tech_index, feedstock_index]),
                        float(decay_rate_ref[tech_index, feedstock_index]),
                    ])
        self._persistence_diagnostics_written = True

    # Estimate the soil-temperature-dependant decay rates
    temp_ref = 20.0 
    k_ref = 0.9 * np.exp(0.02 * temp_ref) - 0.7

    soil_temp_clean = np.where(
        np.isfinite(SOILTEMP.values[:, 0]),
        SOILTEMP.values[:, 0],
        temp_ref
    ).astype(np.float32)

    k_grid = 0.9 * np.exp(0.02 * soil_temp_clean) - 0.7
    soil_temp_conversion = k_grid / k_ref

    self.decay_rate.values = np.maximum(0.0, soil_temp_conversion[:, None, None] * decay_rate_ref[None, :, :]) 

def estimate_soil_effects(
        self, 
        time: prism.Time,
        GAREA5m: prism.Array[LandGrid, np.float32],
        GLANDCOVERFrac: prism.Array[LandGrid, LandCover, np.float32],
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GFERTILISER: prism.Array[LandGrid, Fertiliser, np.float32],
        GCH4: prism.Array[LandGrid, np.float32],
        GN2O: prism.Array[LandGrid, np.float32],
    ):

    """
    Calculate the gridded values for biochar-induced effects on crop yields and soil GHG emissions

    References:
    - ISRIC World Soil Information: SoilsGrid250m 2.0 database
    - Ye et al. 2020: Biochar effects on crop yields with and without fertiliser: A meta-analysis
        of field studies using separate controls 
    - Soil fertiliser N2O and wetland rice CH4 emissions from IMAGE
    """

    cropland = (
        GLANDCOVERFrac.values[:, 0].astype(np.float32, copy=False) 
        * GAREA5m.values * 100.0 # km2 to ha
    )

    # Categorise cropland into paddy (rice) and upland (all other crops)            
    crop_frac = np.nansum(GFRAC.values[:, 1:], axis=1, dtype=np.float32) # exclude grassland 
    paddy_frac = np.nansum(
        GFRAC.values[:, [2, 23]],
        axis=1,
        dtype=np.float32,
    )        
    paddy_frac = np.maximum(paddy_frac, 0.0)

    paddy_share = np.divide(
        paddy_frac,
        crop_frac,
        out=np.zeros_like(crop_frac, dtype=np.float32),
        where=crop_frac > EPS
    )
    paddy_share = np.clip(paddy_share, 0.0, 1.0)

    upland_share = 1.0 - paddy_share

    # Total area in ha per grid cell 
    paddy_area = cropland * paddy_share
    upland_area = cropland * upland_share

    # Exclude very small areas (less than 1 ha)
    has_paddy = (paddy_area > 1.0) 
    has_upland = (upland_area > 1.0)

    # Categorise cells as upland if upland area is greater than paddy area 
    is_upland = (
        has_upland
        & (upland_area >= paddy_area)
    )

    # Create binned soil pH variable
    is_acidic = np.where(
        self.soil_ph.values < 6.5,
        1, # acidic
        0 # neutral-alkaline
    ).astype(np.int8)

    # Set biochar application rate
    biochar_application_rate = np.zeros(len(LandGrid.coords), dtype=np.float32)
    biochar_application_rate[(cropland > EPS)] = self.application_rate

    #### Estimate crop yield effects
    # Center numeric estimates
    centering_df = pd.read_csv(biochar_data_dir / "yield_centering_means.csv")

    yield_centering_means = dict(
        zip(centering_df["variable"], centering_df["mean"])
    )

    pH_c = self.soil_ph.values - yield_centering_means["pH"]
    soc_c = self.soil_soc.values - yield_centering_means["initial SOC (g/kg)"]
    tn_c = self.soil_n.values - yield_centering_means["initial TN (g/kg)"]

    rate_log = np.log1p(biochar_application_rate).astype(np.float32)
    rate_log_c = rate_log - yield_centering_means["rate_log"]

    beta = getattr(self, "yield_beta_coefficients", None)

    lnrr = (
        beta["Intercept"]
        + beta["C(Land_use)[T.upland]"] * is_upland.astype(np.float32)
        + beta["Q('pH_c')"] * pH_c
        + beta["rate_log_c"] * rate_log_c
        + beta["Q('initial SOC (g/kg)_c'):C(Land_use)[paddy]"] * soc_c * (~is_upland).astype(np.float32)
        + beta["Q('initial SOC (g/kg)_c'):C(Land_use)[upland]"] * soc_c * is_upland.astype(np.float32)
        + beta["Q('initial TN (g/kg)_c'):rate_log_c"] * tn_c * rate_log_c
    )

    # Convert log response ratio to a percentage yield change.
    yield_effects_grid = np.expm1(lnrr) * 100.0
    if not getattr(self, "allow_negative_yield_response", False):
        yield_effects_grid = np.maximum(yield_effects_grid, 0.0)

    if getattr(self, "agronomic_response_mode", "fixed_window") == "aging_within_window":
        years = int(self.application_interval)
        yield_effects_grid *= mean_within_window_aging_multiplier(
            benefit_years=years,
            full_effect_years=int(self.agronomic_full_effect_years),
            final_year_retention=float(self.agronomic_year5_retention),
        )
    yield_effects_grid = yield_effects_grid.astype(np.float32)

    if not self._simplification_enabled():
        self.yield_effects.values[:, 0] = np.where(
            cropland > EPS,
            yield_effects_grid,
            np.nan,
        )
        self.yield_effects.values[:, 1] = np.zeros_like(
            yield_effects_grid,
            dtype=np.float32,
        )

    # Center numeric estimates
    centering_df = pd.read_csv(biochar_data_dir / "n2o_centering_means.csv")

    n2o_centering_means = dict(
        zip(centering_df["variable"], centering_df["mean"])
    )

    pH_c = self.soil_ph.values - n2o_centering_means["pH"]
    clay_c = self.soil_clay.values - n2o_centering_means["Clay (%)"]
    n_rate_c = (GFERTILISER.values[:, 0] * 10.0) - n2o_centering_means["Seasonal N rate (kg/ha)"] # Mg/km2 to kg/ha

    # Build regession equation
    beta_df = pd.read_csv(biochar_data_dir / "n2o_beta.csv")
    beta = dict(zip(beta_df["predictor"], beta_df["beta"]))

    required_n2o = (
        "Intercept", "C(Land_use)[T.upland]", "Q('pH_c')",
        "Q('Clay (%)_c')", "Q('Seasonal N rate (kg/ha)_c')",
    )
    missing_n2o = [key for key in required_n2o if key not in beta]
    if missing_n2o:
        raise ValueError(f"Missing N2O coefficient(s): {missing_n2o}")

    # N2O soil effects are not applied to paddy areas 
    lnrr = (
        beta["Intercept"]
        + beta["C(Land_use)[T.upland]"]
        + beta["Q('pH_c')"] * pH_c
        + beta["Q('Clay (%)_c')"] * clay_c
        + beta["Q('Seasonal N rate (kg/ha)_c')"] * n_rate_c
    ).astype(np.float32)

    lnrr = np.where(
        has_upland,
        lnrr,
        0.0
    )        

    # Convert lnRR to % change
    n2o_effects_grid = (np.expm1(lnrr) * 100.0).astype(np.float32)

    if not self._simplification_enabled():
        self.n2o_effects.values = np.where(
            cropland > EPS,
            n2o_effects_grid,
            np.nan,
        )

    ##### Estimate soil CH4 emission effects
    # Build regession equation
    beta_df = pd.read_csv(biochar_data_dir / "ch4_beta.csv")

    beta = dict(zip(beta_df["predictor"], beta_df["beta"]))

    lnrr = (
        beta["Intercept"]
        + beta["rate_log"] * rate_log
        + beta["is_acidic"] * is_acidic.astype(np.float32)
    )

    # CH4 effects are only included for paddy areas
    lnrr0 = np.where(
        has_paddy,
        lnrr,
        0.0
    ).astype(np.float32)

    # Convert lnRR to % change
    ch4_effects_grid = (np.expm1(lnrr0) * 100.0).astype(np.float32)

    if not self._simplification_enabled():
        self.ch4_effects.values = np.where(
            cropland > EPS,
            ch4_effects_grid,
            np.nan,
        )

    else:
        current_year = int(getattr(time.t, "magnitude", time.t))
        baseline_year = int(
            getattr(self, "simplified_baseline_year", 2025)
        )

        cached_effects = getattr(
            self,
            "_simplified_soil_effects",
            None,
        )

        if (cached_effects is not None
            and current_year >= baseline_year
            and cached_effects.get("year", -1) >= baseline_year
        ):
            effects = cached_effects

        else:
            suitable_cropland = (cropland > 1.0) & np.isfinite(yield_effects_grid)

            yield_mean = _weighted_mean(
                yield_effects_grid,
                cropland,
                suitable_cropland,
            )

            n2o_weight = (
                np.maximum(GN2O.values, 0.0)
                * np.maximum(GAREA5m.values, 0.0)
                * upland_share
            )
            n2o_mean = _weighted_mean(
                n2o_effects_grid,
                n2o_weight,
                suitable_cropland & has_upland,
            )

            ch4_weight = (
                np.maximum(GCH4.values, 0.0)
                * np.maximum(GAREA5m.values, 0.0)
            )
            ch4_mean = _weighted_mean(
                ch4_effects_grid,
                ch4_weight,
                suitable_cropland & has_paddy,
            )

            effects = {
                "year": current_year,
                "yield": np.float32(yield_mean),
                "n2o": np.float32(n2o_mean),
                "ch4": np.float32(ch4_mean),
            }

            if current_year >= baseline_year:
                self._simplified_soil_effects = effects

        generic_tech = int(
            getattr(self, "simplified_technology_index", 1)
        )

        self.yield_effects.values[:, :] = 0.0
        self.yield_effects.values[:, generic_tech] = np.where(
            cropland > EPS,
            effects["yield"],
            np.nan,
        ).astype(np.float32)

        self.n2o_effects.values = np.where(
            cropland > EPS,
            np.where(has_upland, effects["n2o"], 0.0),
            np.nan,
        ).astype(np.float32)

        self.ch4_effects.values = np.where(
            cropland > EPS,
            np.where(has_paddy, effects["ch4"], 0.0),
            np.nan,
        ).astype(np.float32)
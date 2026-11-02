import numpy as np


def eligible_under_full_dose_cap(cumulative_t_ha, dose_t_ha, cap_t_ha):
    if cap_t_ha is None:
        return np.ones_like(cumulative_t_ha, dtype=bool)
    return (np.asarray(cumulative_t_ha, dtype=np.float64) + dose_t_ha
            <= cap_t_ha + 1e-5)


def within_window_aging_profile(
        benefit_years=5, 
        full_effect_years=2,
        final_year_retention=0.5
):
    years = int(benefit_years)
    full = int(full_effect_years)
    retention = float(final_year_retention)
    if years != benefit_years or full != full_effect_years:
        raise ValueError("Year counts must be integers")
    if years < 2 or not (1 <= full < years):
        raise ValueError("Require benefit_years >= 2 and 1 <= full_effect_years < benefit_years")
    if not np.isfinite(retention) or not (0.0 < retention <= 1.0):
        raise ValueError("final_year_retention must lie in (0, 1]")
    ages = np.arange(1, years + 1, dtype=np.float64)
    decay = -np.log(retention) / (years - full)
    return np.exp(-decay * np.maximum(ages - full, 0.0))


def mean_within_window_aging_multiplier(
        benefit_years=5, 
        full_effect_years=2,
        final_year_retention=0.5
):
    return float(np.mean(within_window_aging_profile(
        benefit_years, full_effect_years, final_year_retention
    )))

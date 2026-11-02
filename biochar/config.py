import numpy as np
import pandas as pd
from timer.cdr.cdr_util import biochar_data_dir
from pathlib import Path
import warnings

YIELD_RESPONSE_SCENARIOS = ("mean", "low", "high")

def read_yield_response_coefficients(scenario):
    if scenario not in YIELD_RESPONSE_SCENARIOS:
        raise ValueError(
            f"Unknown yield_response_scenario={scenario!r}; expected one of {YIELD_RESPONSE_SCENARIOS}"
        )

    filename = (
        "yield_beta.csv" if scenario == "mean"
        else f"yield_beta_{scenario}.csv"
    )
    path = biochar_data_dir / filename

    df = pd.read_csv(path)
    values = pd.to_numeric(df["beta"], errors="coerce").to_numpy()

    # Fail if a regression was fitted with a different design matrix
    reference_path = biochar_data_dir / "yield_beta.csv"
    if scenario != "mean" and reference_path.is_file():
        reference = pd.read_csv(reference_path)
        if set(df["predictor"]) != set(reference["predictor"]):
            raise ValueError(
                f"Predictors in {path} differ from the reference; {reference_path}; refit/export them together"
            )

    coefficients = dict(zip(df["predictor"], values))
    required = (
        "Intercept",
        "C(Land_use)[T.upland]",
        "Q('pH_c')",
        "rate_log_c",
        "Q('initial SOC (g/kg)_c'):C(Land_use)[paddy]",
        "Q('initial SOC (g/kg)_c'):C(Land_use)[upland]",
        "Q('initial TN (g/kg)_c'):rate_log_c",
    )
    missing = set(required) - set(coefficients)
    if missing:
        raise ValueError(f"{path} is missing yield predictors: {sorted(missing)}")
    return coefficients, df[["predictor", "beta"]], path

def configure_diffusion_rate_limit(model, settings):
    raw = settings.get("diffusion_rate_limit_per_year", 0.085)
    if isinstance(raw, (bool, np.bool_)):
        raise ValueError(
            "diffusion_rate_limit_per_year must be a fractional annual rate, e.g. 0.04, 0.085, or 0.15 (not 4, 8.5, or 15)"
        )
    try:
        rate = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "diffusion_rate_limit_per_year must be numeric and expressed as a fraction, e.g. 0.04 for 4% per year"
        ) from exc
    if not np.isfinite(rate) or not (0.0 < rate <= 1.0):
        raise ValueError(
            "diffusion_rate_limit_per_year must be finite and in (0, 1], e.g., 0.04, 0.085, or 0.15"
        )
    model.MaxBiocharRate = rate
    
    return rate

def configure_biochar_model(model, run_config):
    settings = run_config.get("settings", {})
    configure_diffusion_rate_limit(model, settings)

    scenario = str(settings.get("yield_response_scenario", "mean")).lower()
    coefficients, beta_table, beta_path = read_yield_response_coefficients(scenario)
    model.yield_response_scenario = scenario
    model.yield_beta_coefficients = coefficients
    model.yield_beta_source = beta_path

    allow_negative = settings.get("allow_negative_yield_response", False)
    model.allow_negative_yield_response = allow_negative

    from timer.cdr.biochar.persistence_models import PERSISTENCE_MODELS
    persistence_model = settings.get("carbon_persistence_model", "azzi_power")
    if not isinstance(persistence_model, str) or persistence_model not in PERSISTENCE_MODELS:
        raise ValueError(
            f"carbon_persistence_model must be one of {PERSISTENCE_MODELS}; received {persistence_model!r}"
        )
    model.carbon_persistence_model = persistence_model
    model._persistence_diagnostics_written = False

    dose = float(settings.get("application_rate_t_ha", 20.0))
    if not np.isfinite(dose) or dose <= 0:
        raise ValueError("application_rate_t_ha must be positive and finite")
    model.application_rate = dose

    cap = settings.get("max_cumulative_application_t_ha", None)
    if cap is not None:
        cap = float(cap)
        if not np.isfinite(cap) or cap < dose:
            raise ValueError("max_cumulative_application_t_ha must be null or >= application_rate_t_ha")
    model.max_cumulative_application_t_ha = cap

    mode = settings.get("agronomic_response_mode", "fixed_window")
    if mode not in ("fixed_window", "aging_within_window"):
        raise ValueError("agronomic_response_mode must be fixed_window or aging_within_window")
    model.agronomic_response_mode = mode

    full = int(settings.get("agronomic_full_effect_years", 2))
    retention = float(settings.get("agronomic_year5_retention", 0.5))
    from timer.cdr.biochar.application_management import within_window_aging_profile
    within_window_aging_profile(5, full, retention)
    model.agronomic_full_effect_years = full
    model.agronomic_year5_retention = retention
    if mode == "aging_within_window" and (
        bool(settings.get("simplify_biochar", False)) or
        bool(settings.get("simplify_biochar_LT", False))
    ):
        raise ValueError("aging_within_window requires spatially detailed biochar (no stylised simplification)")
    model.output_dir = Path(
        run_config["output_directory"]
    )

    model.output_dir.mkdir(parents=True, exist_ok=True)
    beta_table.to_csv(model.output_dir / "yield_beta_selected.csv", index=False)

    model.carbon_tax_directory = Path(
        run_config["carbon_tax_directory"]
    )

    model.residue_directory = Path(
        run_config["residue_directory"]
    )

    model.simplify_biochar = bool(
        settings.get(
            "simplify_biochar",
            False,
        )
    )
    model.simplify_biochar_LT = bool(
        settings.get(
            "simplify_biochar_LT",
            False,
        )
    )

    if model.simplify_biochar and model.simplify_biochar_LT:
        raise ValueError(
            "simplify_biochar and simplify_biochar_LT are mutually exclusive."
        )

    model.simplified_technology_index = (
        0 if model.simplify_biochar_LT else 1
    )

    requested_disabled_technologies = tuple(
        settings.get(
            "disabled_technologies",
            [],
        )
    )

    if model._simplification_enabled() and requested_disabled_technologies:
        raise ValueError(
            "Do not combine simplify_biochar/simplify_biochar_LT with disabled_technologies. Each stylised mode automatically enables only its configured generic 400C tech"
        )

    model.disabled_technologies = requested_disabled_technologies

    model._simplified_feedstock_property_cache = {}
    model._simplified_soil_effects = None
    model._simplified_technology_configured = False
    model._simplified_global_stable_fraction_100y = None
    model._simplified_global_decay_rate = None

    model.print_diagnostics = bool(
        settings.get(
            "print_diagnostics",
            True,
        )
    )

    proportional_residue_rationing = bool(
        settings.get(
            "proportional_residue_rationing",
            False,
        )
    )

    model.proportional_residue_rationing = (
        proportional_residue_rationing
    )

    reference_directory = run_config.get(
        "residue_rationing_reference_directory"
    )

    if reference_directory is None:
        reference_directory = settings.get(
            "residue_rationing_reference_directory"
        )

    if proportional_residue_rationing:

        if reference_directory is None:
            raise ValueError(
                "A residue_rationing_reference_directory is required when proportional_residue_rationing=true"
            )

        model.residue_rationing_reference_directory = Path(
            reference_directory
        )

    else:
        model.residue_rationing_reference_directory = None

    return model
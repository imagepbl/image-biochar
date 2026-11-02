import numpy as np

PERSISTENCE_MODELS = ("azzi_power", "azzi_exponential")

def fraction_remaining_100y(hc_ratio, model="azzi_power"):
    if model not in PERSISTENCE_MODELS:
        raise ValueError(
            f"Unsupported carbon_persistence_model={model!r}; choose one of {PERSISTENCE_MODELS}"
        )

    hc = np.asarray(hc_ratio, dtype=np.float64)
    if not np.all(np.isfinite(hc)) or np.any(hc <= 0):
        raise ValueError("Predicted biochar H/C must be positive and finite")

    with np.errstate(invalid="ignore", over="ignore"):
        if model == "azzi_power":
            retained_percent = 93.8 - 18.0 * np.power(hc, 4.85)
        else:
            retained_percent = 93.0 - 49.1 * np.power(hc, 1.98)

    fraction = retained_percent / 100.0
    if not np.all(np.isfinite(fraction)) or np.any(fraction <= 0) or np.any(fraction > 1):
        raise ValueError(
            f"Azzi {model} predicts an invalid 100-year retained fraction for H/C range {float(np.min(hc)):.4g}–{float(np.max(hc)):.4g}"
            "Check whether feedstock/technology predictions are outside the empirical domain before extrapolating "
        )
    return fraction

def equivalent_decay_rate_ref(hc_ratio, model="azzi_power"):
    return -np.log(fraction_remaining_100y(hc_ratio, model)) / 100.0

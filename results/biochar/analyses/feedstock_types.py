from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None

from timer.cdr.cdr_util import (
    Region26,
    Residue,
    RCCPlus,
    BiocharFeedstock,
    PyrolysisTechnology,
    biochar_data_dir,
)
from timer.cdr.results.biochar.biochar_results import IMAGE_REGIONS


"""
Standalone analysis of regional biochar feedstock availability/use and the
carbon-storage properties of each feedstock.

Purpose
-------
This module answers three related questions for the detailed climate-oriented
biochar scenario:

1. Which residue/feedstock types are potentially available in each IMAGE R26
   region?
2. Which feedstock types are actually used by biochar deployment?
3. Does realised use preferentially draw on feedstocks with favourable
   carbon-storage properties (biochar carbon content, reference decay rate,
   100-year persistence, and retained carbon per unit feedstock)?

The carbon properties are NOT read from model outputs. They are reconstructed
exactly from the equations in physics.py and the external biochar data files:

    feedstock_lignin_content.dat
    feedstock_ash_content.dat
    feedstock_HC_ratio_slope.dat
    feedstock_HC_ratio_intercept.dat
    pyrolysis_temperature.dat

Physics replicated from physics.py
----------------------------------
For pyrolysis temperature T (degC) and feedstock f:

    cc_daf = 0.93 - 0.92 * exp(-0.0042 * T)

    yield_daf_f = (
        0.1261
        + 0.5391 * exp(-0.004 * T)
        + 0.002733 * lignin_f
    )

    ash_char_f = ash_f / (ash_f + yield_daf_f)

    carbon_fraction_f = cc_daf * (1 - ash_char_f)

    yield_dm_f = (
        yield_daf_f * (1 - ash_char_f) / (1 - ash_f)
    )

    feedstock_input_f = 1 / yield_dm_f

For persistence at the model reference soil temperature (20 degC):

    HC_f = exp(intercept_f + slope_f * T)

    F100_f = (93.8 - 18.0 * HC_f**4.85) / 100

    k_ref_f = -log(F100_f) / 100

The analysis therefore reports feedstock-specific *reference-temperature*
decay rates. It deliberately does not include gridded soil temperature; the
objective here is to isolate differences among feedstock types themselves.

Availability versus use
------------------------
Potential availability is taken from ResPotCurveDM and is therefore the ex-ante
residue potential before endogenous biochar use. If proportional TIMER/biochar
residue rationing is enabled, this ex-ante potential can be larger than the
amount actually available to biochar after competition. The module detects
that setting in scenario.yaml and prints a warning rather than silently calling
ResPotCurveDM post-rationing availability.

Realised use is read from:

    BiocharResidueDMUsed.nc / BiocharResidueDMUsed

which is stored by region x technology x feedstock x residue-cost bin.

By default the analysis covers 2030-2100 cumulatively. ResPotCurveDM is
multiplied by the inferred TIMER timestep width before summation, matching the
model's prepare_residue_supply() treatment. BiocharResidueDMUsed is already a
timestep total and is therefore summed directly.

The main figure uses shares of *biochar-equivalent output* rather than dry
matter because this is the weighting basis used by the model when constructing
regional expected retained-carbon properties. Dry-matter shares are also saved
and printed.
"""


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output")
SCENARIO_NAME = "biochar_c400"
SCENARIO_DIR = BASE_DIR / SCENARIO_NAME

# Fallback only. The module first reads residue_directory from scenario.yaml.
RESIDUE_DIR_FALLBACK = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\biochar\SSP2_M"
)

RESIDUE_POTENTIAL_VAR = "ResPotCurveDM"
RESIDUE_USE_VAR = "BiocharResidueDMUsed"

# The climate-oriented scenario is fully high-temperature. Select the model
# technology whose configured pyrolysis temperature is closest to this value.
TARGET_PYROLYSIS_TEMP_C = 500.0

# 'cumulative' -> sum 2030-2100; 'year' -> use ANALYSIS_YEAR only.
ANALYSIS_MODE = "cumulative"
START_YEAR = 2030
END_YEAR = 2100
ANALYSIS_YEAR = 2100

# Figure shares can be based on dry residue mass ('dm') or the amount of
# biochar that the residue could/does produce ('biochar'). 'biochar' is the
# more direct basis for comparison with tC/t biochar carbon properties.
COMPOSITION_BASIS = "biochar"

OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_STEM = "biochar_r26_feedstock_carbon_properties_v2"
SAVE_DPI = 600
SHOW_FIGURE = True

# Heatmap display threshold; values below this remain in calculations but are
# visually near white.
HEATMAP_VMAX_PERCENT = 70.0

# R26 labels used in other manuscript figures.
R26_DISPLAY = {
    "CAN": "Canada",
    "USA": "USA",
    "MEX": "Mexico",
    "RCAM": "Rest of Central America",
    "BRA": "Brazil",
    "RSAM": "Rest of South America",
    "NAF": "North Africa",
    "WAF": "West Africa",
    "EAF": "East Africa",
    "SAF": "South Africa",
    "WEU": "Western Europe",
    "CEU": "Central Europe",
    "TUR": "Turkey",
    "UKR": "Ukraine",
    "STAN": "Central Asia",
    "RUS": "Russia",
    "ME": "Middle East",
    "INDIA": "India",
    "KOR": "Korea",
    "CHN": "China",
    "SEAS": "Southeast Asia",
    "INDO": "Indonesia",
    "JAP": "Japan",
    "OCE": "Oceania",
    "RSAS": "Rest of South Asia",
    "RSAF": "Rest of Southern Africa",
}

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

FEEDSTOCK_DISPLAY = {
    "temperate_cereals": "Temperate cereals",
    "rice": "Rice",
    "maize": "Maize",
    "tropical_cereals": "Tropical cereals",
    "pulses": "Pulses",
    "roots_tubers": "Roots & tubers",
    "oil_crops": "Oil crops",
    "fibre": "Fibre crops",
    "fruit": "Fruit",
    "vegetables": "Vegetables",
    "sugarcane": "Sugarcane",
    "biofuel_sugarcane": "Biofuel sugarcane",
    "biofuel_maize": "Biofuel maize",
    "biofuel_woody": "Biofuel woody",
    "biofuel_nwoody": "Biofuel non-woody",
    "forestry": "Forestry residues",
}


# =============================================================================
# Generic I/O helpers
# =============================================================================

def _resolve_case_insensitive_file(directory, stem):
    directory = Path(directory)
    for suffix in (".nc", ".NC"):
        path = directory / f"{stem}{suffix}"
        if path.exists():
            return path
    return directory / f"{stem}.nc"


def _read_scenario_yaml(scenario_dir):
    path = Path(scenario_dir) / "scenario.yaml"
    if not path.exists():
        return None

    try:
        import yaml
    except Exception:
        warnings.warn(
            "PyYAML is unavailable; using RESIDUE_DIR_FALLBACK instead of "
            "reading scenario.yaml."
        )
        return None

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    return data if isinstance(data, dict) else None


def _resolve_residue_directory(scenario_dir):
    scenario = _read_scenario_yaml(scenario_dir)
    if scenario is not None:
        directory = scenario.get("residue_directory")
        if directory:
            return Path(directory)
    return RESIDUE_DIR_FALLBACK


def _proportional_rationing_enabled(scenario_dir):
    scenario = _read_scenario_yaml(scenario_dir)
    if not scenario:
        return None
    settings = scenario.get("settings", {})
    if not isinstance(settings, dict):
        return None
    return bool(settings.get("proportional_residue_rationing", False))


def _time_years(da):
    if "time" not in da.dims:
        return None

    coord = da["time"]
    values = coord.values

    if np.issubdtype(values.dtype, np.datetime64):
        return coord.dt.year.values.astype(int)

    # Prism exports can be plain integer/float years.
    return np.asarray(values, dtype=float).round().astype(int)


def _infer_timestep_widths(years):
    """
    Return one timestep width per exported year.

    For regular TIMER output this is 5 years. The final width uses the median
    previous difference so the selected 2100 timestep gets the same accounting
    as the model's dt.
    """
    years = np.asarray(years, dtype=float)
    if years.size <= 1:
        return np.ones_like(years, dtype=float)

    diffs = np.diff(years)
    positive = diffs[np.isfinite(diffs) & (diffs > 0)]
    default = float(np.median(positive)) if positive.size else 1.0

    widths = np.full(years.shape, default, dtype=float)
    widths[:-1] = np.where(
        np.isfinite(diffs) & (diffs > 0),
        diffs,
        default,
    )
    return widths


def _load_dat_vector(path, expected_length=None):
    """
    Read a small Prism/TIMER .dat vector robustly.

    Prism data files commonly contain a textual header such as ``real`` plus
    dimension metadata before the numerical payload.  np.loadtxt therefore
    cannot be used directly.

    Strategy
    --------
    1. Read only lines whose tokens are entirely numeric, ignoring textual
       header/comment lines.
    2. If the final ``expected_length`` numeric rows look like indexed data
       (0..N-1 or 1..N in column 1), take the final column as the values.
    3. Otherwise flatten the numeric payload and take the final
       ``expected_length`` values.  This safely drops numeric dimension/header
       entries that precede the actual vector, which is the convention used by
       the Prism input files here.

    This function is intentionally used only for the small static biochar
    parameter files, not for IMAGE/TIMER scenario arrays.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    numeric_rows = []

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            # Strip common inline comments.
            line = raw_line
            for marker in ("#", "//", "!"):
                if marker in line:
                    line = line.split(marker, 1)[0]

            line = (
                line.strip()
                .replace(",", " ")
                .replace(";", " ")
                .replace("\t", " ")
            )

            if not line:
                continue

            tokens = [tok for tok in line.split() if tok]
            row = []
            all_numeric = True

            for token in tokens:
                # Support Fortran-style D exponents.
                token_num = token.replace("D", "E").replace("d", "e")
                try:
                    row.append(float(token_num))
                except ValueError:
                    all_numeric = False
                    break

            if all_numeric and row:
                numeric_rows.append(row)

    if not numeric_rows:
        raise ValueError(
            f"No numerical payload could be read from Prism data file: {path}"
        )

    if expected_length is None:
        return np.asarray(
            [x for row in numeric_rows for x in row],
            dtype=np.float64,
        )

    expected_length = int(expected_length)

    # Detect the possible format:
    #   0 value
    #   1 value
    #   ...
    # or the one-based equivalent.  Use the LAST N numeric rows so any
    # preceding numeric metadata cannot be mistaken for observations.
    if len(numeric_rows) >= expected_length:
        tail_rows = numeric_rows[-expected_length:]

        if all(len(row) >= 2 for row in tail_rows):
            first_col = np.asarray(
                [row[0] for row in tail_rows],
                dtype=np.float64,
            )

            zero_based = np.arange(
                expected_length,
                dtype=np.float64,
            )
            one_based = zero_based + 1.0

            if (
                np.allclose(first_col, zero_based)
                or np.allclose(first_col, one_based)
            ):
                values = np.asarray(
                    [row[-1] for row in tail_rows],
                    dtype=np.float64,
                )
                return values

    flat = np.asarray(
        [x for row in numeric_rows for x in row],
        dtype=np.float64,
    )

    if flat.size < expected_length:
        raise ValueError(
            f"Could read only {flat.size} numeric values from {path}, "
            f"but expected {expected_length}. "
            "Inspect the Prism .dat formatting."
        )

    if flat.size > expected_length:
        # Prism headers can include numerical dimension metadata before the
        # payload.  The parameter vectors themselves are stored at the end.
        values = flat[-expected_length:].copy()
        print(
            f"Read {path.name}: ignored {flat.size - expected_length} "
            f"numeric header/metadata value(s); retained final "
            f"{expected_length} parameter values."
        )
    else:
        values = flat

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f"Non-finite value(s) found in numerical payload of {path}."
        )

    return values


def _feedstock_names():
    nfeed = len(BiocharFeedstock.coords)
    if nfeed != len(BIOCHAR_FEEDSTOCK_NAMES):
        raise ValueError(
            "BiocharFeedstock dimension length no longer matches the explicit "
            f"feedstock-name list: {nfeed} vs {len(BIOCHAR_FEEDSTOCK_NAMES)}. "
            "Update BIOCHAR_FEEDSTOCK_NAMES to match timer.cdr.biochar.feedstock."
        )
    return list(BIOCHAR_FEEDSTOCK_NAMES)


def _region_codes():
    nreg = len(Region26.coords)
    codes = [str(x) for x in IMAGE_REGIONS[:nreg]]
    if len(codes) != nreg:
        raise ValueError(
            f"Expected {nreg} IMAGE R26 region codes; found {len(codes)}."
        )
    return codes


# =============================================================================
# Reconstruct feedstock carbon-storage properties from physics.py equations
# =============================================================================

def calculate_feedstock_properties():
    """
    Reproduce compute_carbon_content() and the reference-temperature portion of
    compute_decay_rate() from physics.py.

    Returns one row per feedstock for the technology closest to
    TARGET_PYROLYSIS_TEMP_C.
    """
    feedstocks = _feedstock_names()
    nfeed = len(feedstocks)

    lignin = _load_dat_vector(
        biochar_data_dir / "feedstock_lignin_content.dat",
        nfeed,
    )
    ash = _load_dat_vector(
        biochar_data_dir / "feedstock_ash_content.dat",
        nfeed,
    )
    hc_slope = _load_dat_vector(
        biochar_data_dir / "feedstock_HC_ratio_slope.dat",
        nfeed,
    )
    hc_intercept = _load_dat_vector(
        biochar_data_dir / "feedstock_HC_ratio_intercept.dat",
        nfeed,
    )
    pyrolysis_temp = _load_dat_vector(
        biochar_data_dir / "pyrolysis_temperature.dat",
        len(PyrolysisTechnology.coords),
    )

    tech_index = int(
        np.nanargmin(
            np.abs(pyrolysis_temp - TARGET_PYROLYSIS_TEMP_C)
        )
    )
    temp = float(pyrolysis_temp[tech_index])

    # Exact physics.py carbon-content equations.
    cc_daf = 0.93 - 0.92 * np.exp(-0.0042 * temp)
    yield_daf = (
        0.1261
        + 0.5391 * np.exp(-0.004 * temp)
        + 0.002733 * lignin
    )
    ash_char = ash / (ash + yield_daf)
    carbon_fraction = cc_daf * (1.0 - ash_char)

    yield_dm = np.divide(
        yield_daf * (1.0 - ash_char),
        (1.0 - ash),
        out=np.full(nfeed, np.nan, dtype=np.float64),
        where=np.isfinite(ash) & ((1.0 - ash) > 0.0),
    )
    feedstock_input = np.divide(
        1.0,
        yield_dm,
        out=np.full(nfeed, np.nan, dtype=np.float64),
        where=np.isfinite(yield_dm) & (yield_dm > 0.0),
    )

    # Exact physics.py reference-decay equations (20 C reference soil temp).
    hc_ratio = np.exp(
        hc_intercept + hc_slope * temp
    )
    stable_fraction_100y = (
        93.8 - 18.0 * hc_ratio ** 4.85
    ) / 100.0

    if np.any(~np.isfinite(stable_fraction_100y)) or np.any(
        stable_fraction_100y <= 0.0
    ):
        raise ValueError(
            "Calculated 100-year stable fractions contain non-positive or "
            "non-finite values; check H/C parameter files."
        )

    decay_rate_ref = -np.log(stable_fraction_100y) / 100.0

    retained_c_per_t_biochar_100y = (
        carbon_fraction * stable_fraction_100y
    )
    retained_co2_per_t_biochar_100y = (
        retained_c_per_t_biochar_100y * (44.0 / 12.0)
    )
    retained_c_per_tdm_100y = (
        yield_dm * retained_c_per_t_biochar_100y
    )
    retained_co2_per_tdm_100y = (
        retained_c_per_tdm_100y * (44.0 / 12.0)
    )

    df = pd.DataFrame(
        {
            "Feedstock": feedstocks,
            "Feedstock display": [
                FEEDSTOCK_DISPLAY.get(x, x)
                for x in feedstocks
            ],
            "Technology index": tech_index,
            "Pyrolysis temperature (C)": temp,
            "Lignin fraction": lignin,
            "Ash fraction": ash,
            "H/Corg": hc_ratio,
            "Biochar yield (t biochar/tDM)": yield_dm,
            "Feedstock input (tDM/t biochar)": feedstock_input,
            "Biochar carbon fraction (tC/t biochar)": carbon_fraction,
            "Reference decay rate (yr-1)": decay_rate_ref,
            "100-y retained fraction at 20C": stable_fraction_100y,
            "100-y retained C (tC/t biochar)": retained_c_per_t_biochar_100y,
            "100-y retained CO2 (tCO2/t biochar)": retained_co2_per_t_biochar_100y,
            "100-y retained C (tC/tDM feedstock)": retained_c_per_tdm_100y,
            "100-y retained CO2 (tCO2/tDM feedstock)": retained_co2_per_tdm_100y,
        }
    ).set_index("Feedstock")

    return df, tech_index, temp


# =============================================================================
# Load regional ex-ante availability
# =============================================================================

def load_available_feedstock_dm(
    scenario_dir,
    mode=ANALYSIS_MODE,
):
    """
    Return ex-ante ResPotCurveDM supply by R26 x feedstock in tDM.

    For cumulative analysis, each year's ResPotCurveDM is multiplied by the
    inferred timestep width, matching prepare_residue_supply() in feedstock.py.
    """
    scenario_dir = Path(scenario_dir)
    residue_dir = _resolve_residue_directory(scenario_dir)
    path = _resolve_case_insensitive_file(
        residue_dir,
        RESIDUE_POTENTIAL_VAR,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {RESIDUE_POTENTIAL_VAR} at {path}"
        )

    with xr.open_dataset(path) as ds:
        if RESIDUE_POTENTIAL_VAR not in ds.data_vars:
            raise KeyError(
                f"{RESIDUE_POTENTIAL_VAR} not found in {path}; "
                f"variables={list(ds.data_vars)}"
            )
        da = ds[RESIDUE_POTENTIAL_VAR]
        years = _time_years(da)
        values = np.asarray(da.values, dtype=np.float64)

    nreg = len(Region26.coords)
    nres = len(Residue.coords)
    nbins_plus = len(RCCPlus.coords)

    if years is None:
        raise ValueError(
            f"{RESIDUE_POTENTIAL_VAR} must contain a time dimension."
        )

    expected_per_time = nreg * nres * nbins_plus
    if values.size != len(years) * expected_per_time:
        raise ValueError(
            f"Unexpected shape/size for {path}: shape={values.shape}; "
            f"expected {len(years)} time steps x {nreg} x {nres} x {nbins_plus}."
        )

    values = values.reshape(
        len(years),
        nreg,
        nres,
        nbins_plus,
    )

    nfeed = len(BiocharFeedstock.coords)
    cost_bins = len(RCCPlus.coords) - 1
    values = values[:, :, :nfeed, :cost_bins]
    values = np.where(
        np.isfinite(values) & (values > 0.0),
        values,
        0.0,
    )

    if mode == "year":
        idx = np.where(years == int(ANALYSIS_YEAR))[0]
        if idx.size == 0:
            raise ValueError(
                f"Year {ANALYSIS_YEAR} not found in {path}; years={years.tolist()}"
            )
        # ResPotCurveDM is a rate-like input; multiply by model dt to obtain
        # the supply available over the selected timestep.
        widths = _infer_timestep_widths(years)
        dm = values[int(idx[0])] * widths[int(idx[0])]
        selected_years = [int(ANALYSIS_YEAR)]

    elif mode == "cumulative":
        selected = (
            (years >= int(START_YEAR))
            & (years <= int(END_YEAR))
        )
        if not np.any(selected):
            raise ValueError(
                f"No years in {START_YEAR}-{END_YEAR} in {path}."
            )
        widths = _infer_timestep_widths(years)
        dm = np.sum(
            values[selected]
            * widths[selected, None, None, None],
            axis=0,
            dtype=np.float64,
        )
        selected_years = years[selected].astype(int).tolist()

    else:
        raise ValueError(
            "ANALYSIS_MODE must be 'year' or 'cumulative'."
        )

    # Sum residue cost bins -> region x feedstock.
    dm = np.sum(dm, axis=2, dtype=np.float64)

    return dm, residue_dir, selected_years


# =============================================================================
# Load realised regional feedstock use
# =============================================================================

def load_used_feedstock_dm(
    scenario_dir,
    tech_index,
    mode=ANALYSIS_MODE,
):
    """
    Return realised BiocharResidueDMUsed by R26 x feedstock in tDM.

    BiocharResidueDMUsed is already a timestep total, so no dt multiplication
    is applied when summing cumulative use.
    """
    scenario_dir = Path(scenario_dir)
    path = _resolve_case_insensitive_file(
        scenario_dir,
        RESIDUE_USE_VAR,
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find realised residue use at {path}. "
            "Ensure BiocharResidueDMUsed is exported for this run."
        )

    with xr.open_dataset(path) as ds:
        if RESIDUE_USE_VAR not in ds.data_vars:
            raise KeyError(
                f"{RESIDUE_USE_VAR} not found in {path}; "
                f"variables={list(ds.data_vars)}"
            )
        da = ds[RESIDUE_USE_VAR]
        years = _time_years(da)

        if years is None:
            raise ValueError(
                f"{RESIDUE_USE_VAR} must contain a time dimension."
            )

        # Prefer named dimensions from output.py.
        required = {"region", "tech", "feed"}
        if required.issubset(set(da.dims)):
            if "bin" in da.dims:
                da = da.sum(dim="bin", skipna=True)

            if mode == "year":
                idx = np.where(years == int(ANALYSIS_YEAR))[0]
                if idx.size == 0:
                    raise ValueError(
                        f"Year {ANALYSIS_YEAR} not found in {path}."
                    )
                da_sel = da.isel(time=int(idx[0]))
                selected_years = [int(ANALYSIS_YEAR)]
            elif mode == "cumulative":
                selected_idx = np.where(
                    (years >= int(START_YEAR))
                    & (years <= int(END_YEAR))
                )[0]
                if selected_idx.size == 0:
                    raise ValueError(
                        f"No years in {START_YEAR}-{END_YEAR} in {path}."
                    )
                da_sel = da.isel(time=selected_idx).sum(
                    dim="time",
                    skipna=True,
                )
                selected_years = years[selected_idx].astype(int).tolist()
            else:
                raise ValueError(
                    "ANALYSIS_MODE must be 'year' or 'cumulative'."
                )

            all_tech = np.asarray(
                da_sel.sum(dim="feed", skipna=True).values,
                dtype=np.float64,
            )
            selected = np.asarray(
                da_sel.isel(tech=int(tech_index)).values,
                dtype=np.float64,
            )

        else:
            values = np.asarray(da.values, dtype=np.float64)
            nreg = len(Region26.coords)
            ntech = len(PyrolysisTechnology.coords)
            nfeed = len(BiocharFeedstock.coords)
            rest = values.size // (
                len(years) * nreg * ntech * nfeed
            )
            if rest <= 0 or (
                len(years) * nreg * ntech * nfeed * rest
                != values.size
            ):
                raise ValueError(
                    f"Cannot reshape {path}; dims={da.dims}, shape={values.shape}."
                )
            values = values.reshape(
                len(years), nreg, ntech, nfeed, rest
            ).sum(axis=4)

            if mode == "year":
                idx = np.where(years == int(ANALYSIS_YEAR))[0]
                if idx.size == 0:
                    raise ValueError(
                        f"Year {ANALYSIS_YEAR} not found in {path}."
                    )
                v = values[int(idx[0])]
                selected_years = [int(ANALYSIS_YEAR)]
            else:
                selected_idx = np.where(
                    (years >= int(START_YEAR))
                    & (years <= int(END_YEAR))
                )[0]
                v = np.sum(values[selected_idx], axis=0)
                selected_years = years[selected_idx].astype(int).tolist()

            all_tech = np.sum(v, axis=2)
            selected = v[:, int(tech_index), :]

    selected = np.where(
        np.isfinite(selected) & (selected > 0.0),
        selected,
        0.0,
    )

    total_all_tech = float(np.nansum(all_tech))
    total_selected = float(np.nansum(selected))
    selected_tech_share = (
        total_selected / total_all_tech
        if total_all_tech > 0.0
        else np.nan
    )

    return selected, selected_tech_share, selected_years


# =============================================================================
# Derived regional/feedstock metrics
# =============================================================================

def _row_shares(values):
    values = np.asarray(values, dtype=np.float64)
    denom = np.sum(values, axis=1, dtype=np.float64)
    return np.divide(
        values,
        denom[:, None],
        out=np.zeros_like(values, dtype=np.float64),
        where=denom[:, None] > 0.0,
    )


def _weighted_mean_by_region(weights, property_values):
    weights = np.asarray(weights, dtype=np.float64)
    prop = np.asarray(property_values, dtype=np.float64)
    denom = np.sum(weights, axis=1, dtype=np.float64)
    numer = np.sum(
        weights * prop[None, :],
        axis=1,
        dtype=np.float64,
    )
    return np.divide(
        numer,
        denom,
        out=np.full(weights.shape[0], np.nan, dtype=np.float64),
        where=denom > 0.0,
    )


def build_analysis_tables(
    properties,
    available_dm,
    used_dm,
):
    feedstocks = list(properties.index)
    regions = _region_codes()

    feedstock_input = properties[
        "Feedstock input (tDM/t biochar)"
    ].to_numpy(dtype=np.float64)

    available_biochar = np.divide(
        available_dm,
        feedstock_input[None, :],
        out=np.zeros_like(available_dm, dtype=np.float64),
        where=np.isfinite(feedstock_input[None, :])
        & (feedstock_input[None, :] > 0.0),
    )
    used_biochar = np.divide(
        used_dm,
        feedstock_input[None, :],
        out=np.zeros_like(used_dm, dtype=np.float64),
        where=np.isfinite(feedstock_input[None, :])
        & (feedstock_input[None, :] > 0.0),
    )

    available_dm_share = _row_shares(available_dm)
    used_dm_share = _row_shares(used_dm)
    available_biochar_share = _row_shares(available_biochar)
    used_biochar_share = _row_shares(used_biochar)

    if COMPOSITION_BASIS == "biochar":
        available_share = available_biochar_share
        used_share = used_biochar_share
    elif COMPOSITION_BASIS == "dm":
        available_share = available_dm_share
        used_share = used_dm_share
    else:
        raise ValueError(
            "COMPOSITION_BASIS must be 'biochar' or 'dm'."
        )

    selection_ratio = np.divide(
        used_share,
        available_share,
        out=np.full_like(used_share, np.nan, dtype=np.float64),
        where=available_share > 0.0,
    )

    availability_df = pd.DataFrame(
        available_share * 100.0,
        index=regions,
        columns=feedstocks,
    )
    use_df = pd.DataFrame(
        used_share * 100.0,
        index=regions,
        columns=feedstocks,
    )
    selection_df = pd.DataFrame(
        selection_ratio,
        index=regions,
        columns=feedstocks,
    )

    # Long table with all mass/output metrics.
    rows = []
    for i, region in enumerate(regions):
        for j, feed in enumerate(feedstocks):
            rows.append(
                {
                    "Region": region,
                    "Region display": R26_DISPLAY.get(region, region),
                    "Feedstock": feed,
                    "Feedstock display": FEEDSTOCK_DISPLAY.get(feed, feed),
                    "Available DM (MtDM)": available_dm[i, j] / 1e6,
                    "Used DM (MtDM)": used_dm[i, j] / 1e6,
                    "Available DM share (%)": 100.0 * available_dm_share[i, j],
                    "Used DM share (%)": 100.0 * used_dm_share[i, j],
                    "Available biochar-equivalent (Mt)": available_biochar[i, j] / 1e6,
                    "Used biochar-equivalent (Mt)": used_biochar[i, j] / 1e6,
                    "Available biochar-equivalent share (%)": 100.0 * available_biochar_share[i, j],
                    "Used biochar-equivalent share (%)": 100.0 * used_biochar_share[i, j],
                    "Selection ratio (used share / available share)": selection_ratio[i, j],
                }
            )
    long_df = pd.DataFrame(rows)

    # Regionally weighted carbon-quality metrics.
    metric_cols = [
        "Biochar carbon fraction (tC/t biochar)",
        "Reference decay rate (yr-1)",
        "100-y retained fraction at 20C",
        "100-y retained C (tC/t biochar)",
        "100-y retained CO2 (tCO2/tDM feedstock)",
    ]

    region_rows = []
    for metric in metric_cols:
        prop = properties[metric].to_numpy(dtype=np.float64)

        # Biochar-output weights for per-t-biochar properties; DM weights for
        # per-tDM properties.
        if "tDM feedstock" in metric:
            avail_w = available_dm
            use_w = used_dm
        else:
            avail_w = available_biochar
            use_w = used_biochar

        avail_mean = _weighted_mean_by_region(avail_w, prop)
        use_mean = _weighted_mean_by_region(use_w, prop)

        for i, region in enumerate(regions):
            region_rows.append(
                {
                    "Region": region,
                    "Region display": R26_DISPLAY.get(region, region),
                    "Metric": metric,
                    "Available-mix mean": avail_mean[i],
                    "Used-mix mean": use_mean[i],
                    "Used minus available": use_mean[i] - avail_mean[i],
                }
            )

    regional_quality = pd.DataFrame(region_rows)

    # Mobilisation by region/feedstock, based on dry matter.
    mobilisation = np.divide(
        used_dm,
        available_dm,
        out=np.full_like(used_dm, np.nan, dtype=np.float64),
        where=available_dm > 0.0,
    )

    mobilisation_df = pd.DataFrame(
        100.0 * mobilisation,
        index=regions,
        columns=feedstocks,
    )

    return {
        "availability_share": availability_df,
        "use_share": use_df,
        "selection_ratio": selection_df,
        "mobilisation_pct": mobilisation_df,
        "long": long_df,
        "regional_quality": regional_quality,
        "available_dm": pd.DataFrame(available_dm, index=regions, columns=feedstocks),
        "used_dm": pd.DataFrame(used_dm, index=regions, columns=feedstocks),
        "available_biochar": pd.DataFrame(available_biochar, index=regions, columns=feedstocks),
        "used_biochar": pd.DataFrame(used_biochar, index=regions, columns=feedstocks),
    }


# =============================================================================
# Quantitative summaries
# =============================================================================

def _top_items(row, n=3):
    row = row.sort_values(ascending=False)
    row = row[row > 0.0]
    return list(row.head(n).items())


def _global_mix(values):
    values = np.asarray(values, dtype=np.float64)
    totals = np.sum(values, axis=0, dtype=np.float64)
    total = float(np.sum(totals, dtype=np.float64))
    if total <= 0.0:
        return np.zeros_like(totals)
    return totals / total


def _weighted_global_mean(weights, prop):
    weights = np.asarray(weights, dtype=np.float64)
    prop = np.asarray(prop, dtype=np.float64)
    w = np.sum(weights, axis=0, dtype=np.float64)
    denom = float(np.sum(w, dtype=np.float64))
    if denom <= 0.0:
        return np.nan
    return float(np.sum(w * prop, dtype=np.float64) / denom)


def print_results_summary(
    properties,
    tables,
    selected_tech_share,
    temp,
    residue_dir,
    available_years,
    used_years,
    rationing_enabled,
):
    availability = tables["availability_share"]
    use = tables["use_share"]
    selection = tables["selection_ratio"]
    available_dm = tables["available_dm"].to_numpy(dtype=np.float64)
    used_dm = tables["used_dm"].to_numpy(dtype=np.float64)
    available_bc = tables["available_biochar"].to_numpy(dtype=np.float64)
    used_bc = tables["used_biochar"].to_numpy(dtype=np.float64)

    print("\n" + "=" * 128)
    print("R26 FEEDSTOCK AVAILABILITY, REALISED USE, AND CARBON-STORAGE QUALITY")
    print("=" * 128)
    print(f"Scenario: {SCENARIO_NAME}")
    print(
        f"Selected technology: {temp:.1f} C "
        f"(target={TARGET_PYROLYSIS_TEMP_C:.1f} C); "
        f"share of realised residue use assigned to this technology = "
        f"{100.0 * selected_tech_share:.2f}%"
    )
    print(f"Residue input directory: {residue_dir}")
    print(f"Analysis mode: {ANALYSIS_MODE}")
    print(f"Availability years: {available_years}")
    print(f"Use years:          {used_years}")
    print(f"Composition basis shown in figure: {COMPOSITION_BASIS}")

    if rationing_enabled:
        print(
            "WARNING: proportional residue rationing is enabled. 'Available' "
            "means ex-ante ResPotCurveDM potential before TIMER/biochar "
            "competition; realised use is post-allocation."
        )

    # Feedstock properties.
    prop_cols = [
        "Feedstock display",
        "Biochar yield (t biochar/tDM)",
        "Biochar carbon fraction (tC/t biochar)",
        "H/Corg",
        "Reference decay rate (yr-1)",
        "100-y retained fraction at 20C",
        "100-y retained C (tC/t biochar)",
        "100-y retained CO2 (tCO2/tDM feedstock)",
    ]
    print("\nFeedstock carbon-storage properties")
    print(properties[prop_cols].round(5).to_string())

    # Global mix comparison.
    feedstocks = list(properties.index)
    if COMPOSITION_BASIS == "biochar":
        global_available = _global_mix(available_bc)
        global_used = _global_mix(used_bc)
    else:
        global_available = _global_mix(available_dm)
        global_used = _global_mix(used_dm)

    global_selection = np.divide(
        global_used,
        global_available,
        out=np.full_like(global_used, np.nan),
        where=global_available > 0.0,
    )

    global_mix_df = pd.DataFrame(
        {
            "Available share (%)": 100.0 * global_available,
            "Used share (%)": 100.0 * global_used,
            "Selection ratio": global_selection,
        },
        index=feedstocks,
    )
    global_mix_df["Feedstock display"] = [
        FEEDSTOCK_DISPLAY.get(x, x)
        for x in feedstocks
    ]
    global_mix_df = global_mix_df.sort_values(
        "Used share (%)",
        ascending=False,
    )

    print("\nGlobal feedstock composition")
    print(
        global_mix_df[
            [
                "Feedstock display",
                "Available share (%)",
                "Used share (%)",
                "Selection ratio",
            ]
        ].round(3).to_string()
    )

    # Property-weighted global means.
    print("\nGlobal carbon-quality of available vs used feedstock mix")
    for metric in [
        "Biochar carbon fraction (tC/t biochar)",
        "Reference decay rate (yr-1)",
        "100-y retained fraction at 20C",
        "100-y retained C (tC/t biochar)",
        "100-y retained CO2 (tCO2/tDM feedstock)",
    ]:
        prop = properties[metric].to_numpy(dtype=np.float64)
        if "tDM feedstock" in metric:
            avail_mean = _weighted_global_mean(available_dm, prop)
            used_mean = _weighted_global_mean(used_dm, prop)
        else:
            avail_mean = _weighted_global_mean(available_bc, prop)
            used_mean = _weighted_global_mean(used_bc, prop)

        pct = (
            100.0 * (used_mean - avail_mean) / abs(avail_mean)
            if np.isfinite(avail_mean) and abs(avail_mean) > 0.0
            else np.nan
        )
        print(
            f"  {metric:<48s} available={avail_mean:.6f}; "
            f"used={used_mean:.6f}; change={pct:+.2f}%"
        )

    # R26 dominant feedstocks.
    print("\nDominant feedstocks by R26 region")
    for region in availability.index:
        a = _top_items(availability.loc[region], n=3)
        u = _top_items(use.loc[region], n=3)
        a_text = ", ".join(
            f"{FEEDSTOCK_DISPLAY.get(f, f)} {v:.1f}%"
            for f, v in a
        ) or "none"
        u_text = ", ".join(
            f"{FEEDSTOCK_DISPLAY.get(f, f)} {v:.1f}%"
            for f, v in u
        ) or "none"
        print(
            f"  {R26_DISPLAY.get(region, region):<24s} "
            f"available: {a_text} | used: {u_text}"
        )

    # Regions with strongest change in 100-y retained carbon quality.
    quality = tables["regional_quality"]
    q = quality[
        quality["Metric"] == "100-y retained C (tC/t biochar)"
    ].copy()
    q = q.sort_values(
        "Used minus available",
        ascending=False,
    )
    print("\nRegions where realised use most changes 100-y retained C per tonne biochar")
    print(
        q[
            [
                "Region display",
                "Available-mix mean",
                "Used-mix mean",
                "Used minus available",
            ]
        ].round(5).to_string(index=False)
    )

    # Feedstock-level global enrichment vs carbon quality.
    if spearmanr is not None:
        valid = np.isfinite(global_selection) & (global_available > 0.0)
        print("\nSpearman association: feedstock enrichment in realised use vs carbon quality")
        for metric in [
            "Biochar carbon fraction (tC/t biochar)",
            "Reference decay rate (yr-1)",
            "100-y retained C (tC/t biochar)",
            "100-y retained CO2 (tCO2/tDM feedstock)",
        ]:
            prop = properties[metric].to_numpy(dtype=np.float64)
            m = valid & np.isfinite(prop)
            if np.count_nonzero(m) >= 3:
                rho, p = spearmanr(
                    global_selection[m],
                    prop[m],
                )
                print(
                    f"  {metric:<48s} rho={rho:+.3f}, p={p:.3g}, n={np.count_nonzero(m)}"
                )

    return global_mix_df


# =============================================================================
# Plotting
# =============================================================================

def _style_heatmap_axis(ax, regions, feedstocks):
    ax.set_yticks(np.arange(len(regions)))
    ax.set_yticklabels(
        [R26_DISPLAY.get(r, r) for r in regions],
        fontsize=7.4,
    )
    ax.set_xticks(np.arange(len(feedstocks)))
    ax.set_xticklabels(
        [FEEDSTOCK_DISPLAY.get(f, f) for f in feedstocks],
        rotation=60,
        ha="right",
        fontsize=7.2,
    )
    ax.tick_params(length=0)


def _annotate_dominant_cells(ax, values, threshold=25.0):
    values = np.asarray(values, dtype=np.float64)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v) and v >= threshold:
                ax.text(
                    j,
                    i,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color="black" if v < 50.0 else "white",
                )


def plot_feedstock_analysis(properties, tables, save=True, show=True):
    regions = list(tables["availability_share"].index)
    feedstocks = list(tables["availability_share"].columns)

    available = tables["availability_share"].to_numpy(dtype=np.float64)
    used = tables["use_share"].to_numpy(dtype=np.float64)
    selection = tables["selection_ratio"].to_numpy(dtype=np.float64)

    fig = plt.figure(figsize=(17.2, 12.0))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=(0.78, 1.22),
        width_ratios=(1.0, 1.0),
        hspace=0.36,
        wspace=0.20,
    )

    # ------------------------------------------------------------------
    # a. Feedstock carbon-storage properties
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])

    x = properties[
        "Biochar carbon fraction (tC/t biochar)"
    ].to_numpy(dtype=np.float64)
    y = properties[
        "Reference decay rate (yr-1)"
    ].to_numpy(dtype=np.float64)
    storage_dm = properties[
        "100-y retained CO2 (tCO2/tDM feedstock)"
    ].to_numpy(dtype=np.float64)
    char_yield = properties[
        "Biochar yield (t biochar/tDM)"
    ].to_numpy(dtype=np.float64)

    sc = ax_a.scatter(
        x,
        y,
        c=storage_dm,
        s=260.0 * char_yield / np.nanmax(char_yield),
        cmap="viridis",
        edgecolor="0.25",
        linewidth=0.5,
        zorder=3,
    )

    for feed, xi, yi in zip(feedstocks, x, y):
        ax_a.annotate(
            FEEDSTOCK_DISPLAY.get(feed, feed),
            (xi, yi),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7.2,
        )

    ax_a.set_xlabel("Biochar carbon fraction (tC t biochar$^{-1}$)")
    ax_a.set_ylabel("Reference decay rate at 20°C (yr$^{-1}$; lower = more persistent)")
    ax_a.set_title(
        r"$\bf{a.}$ Feedstock carbon-storage characteristics",
        loc="left",
        fontweight="normal",
    )
    ax_a.grid(True, linestyle="--", alpha=0.25)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    cbar = fig.colorbar(
        sc,
        ax=ax_a,
        orientation="vertical",
        fraction=0.048,
        pad=0.025,
    )
    cbar.set_label(
        "100-y retained CO$_2$ at 20°C\n(tCO$_2$ tDM feedstock$^{-1}$)",
        fontsize=8.5,
    )

    ax_a.text(
        0.02,
        0.02,
        "Marker size ∝ biochar yield",
        transform=ax_a.transAxes,
        fontsize=7.5,
        va="bottom",
    )

    # ------------------------------------------------------------------
    # b. Used-vs-available enrichment by feedstock (global)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])

    if COMPOSITION_BASIS == "biochar":
        avail_global = _global_mix(
            tables["available_biochar"].to_numpy(dtype=np.float64)
        )
        used_global = _global_mix(
            tables["used_biochar"].to_numpy(dtype=np.float64)
        )
        basis_label = "biochar-equivalent output"
    else:
        avail_global = _global_mix(
            tables["available_dm"].to_numpy(dtype=np.float64)
        )
        used_global = _global_mix(
            tables["used_dm"].to_numpy(dtype=np.float64)
        )
        basis_label = "dry matter"

    delta = 100.0 * (used_global - avail_global)
    order = np.argsort(delta)
    y_pos = np.arange(len(feedstocks))

    ax_b.barh(
        y_pos,
        delta[order],
        color=np.where(delta[order] >= 0.0, "0.25", "0.65"),
        height=0.68,
    )
    ax_b.axvline(0.0, color="black", linewidth=0.8)
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(
        [FEEDSTOCK_DISPLAY.get(feedstocks[i], feedstocks[i]) for i in order],
        fontsize=7.8,
    )
    ax_b.set_xlabel(
        f"Used share − available share\n(percentage points of {basis_label})"
    )
    ax_b.set_title(
        r"$\bf{b.}$ Feedstocks enriched or depleted in realised use",
        loc="left",
        fontweight="normal",
    )
    ax_b.grid(axis="x", linestyle="--", alpha=0.25)
    ax_b.set_axisbelow(True)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # c. R26 potential availability composition
    # ------------------------------------------------------------------
    ax_c = fig.add_subplot(gs[1, 0])
    im_c = ax_c.imshow(
        available,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=HEATMAP_VMAX_PERCENT,
    )
    _style_heatmap_axis(ax_c, regions, feedstocks)
    _annotate_dominant_cells(ax_c, available)
    ax_c.set_title(
        rf"$\bf{{c.}}$ R26 potential feedstock availability ({basis_label} share)",
        loc="left",
        fontweight="normal",
    )
    cb_c = fig.colorbar(
        im_c,
        ax=ax_c,
        orientation="horizontal",
        fraction=0.046,
        pad=0.18,
        aspect=35,
        extend="max",
    )
    cb_c.set_label("Regional feedstock share (%)")

    # ------------------------------------------------------------------
    # d. R26 realised-use composition
    # ------------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 1])
    im_d = ax_d.imshow(
        used,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrBr",
        vmin=0.0,
        vmax=HEATMAP_VMAX_PERCENT,
    )
    _style_heatmap_axis(ax_d, regions, feedstocks)
    _annotate_dominant_cells(ax_d, used)
    ax_d.set_title(
        rf"$\bf{{d.}}$ R26 realised feedstock use ({basis_label} share)",
        loc="left",
        fontweight="normal",
    )
    cb_d = fig.colorbar(
        im_d,
        ax=ax_d,
        orientation="horizontal",
        fraction=0.046,
        pad=0.18,
        aspect=35,
        extend="max",
    )
    cb_d.set_label("Regional feedstock share (%)")

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.96,
        bottom=0.08,
    )

    outputs = {
        "figure": fig,
        "axes": (ax_a, ax_b, ax_c, ax_d),
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        png = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
        pdf = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
        fig.savefig(png, dpi=SAVE_DPI, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        outputs.update({"png_path": png, "pdf_path": pdf})
        print(f"\nSaved figure: {png}")
        print(f"Saved figure: {pdf}")

    if show:
        plt.show()

    return outputs


# =============================================================================
# Save tables
# =============================================================================

def save_tables(properties, tables, global_mix_df):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}

    items = {
        "feedstock_properties": properties,
        "availability_share_r26": tables["availability_share"],
        "use_share_r26": tables["use_share"],
        "selection_ratio_r26": tables["selection_ratio"],
        "mobilisation_pct_r26": tables["mobilisation_pct"],
        "feedstock_region_long": tables["long"],
        "regional_carbon_quality": tables["regional_quality"],
        "global_mix": global_mix_df,
    }

    for suffix, df in items.items():
        path = OUTPUT_DIR / f"{OUTPUT_STEM}_{suffix}.csv"
        if suffix == "feedstock_region_long":
            df.to_csv(path, index=False)
        elif suffix == "regional_carbon_quality":
            df.to_csv(path, index=False)
        else:
            df.to_csv(path)
        paths[f"{suffix}_csv"] = path
        print(f"Saved table: {path}")

    return paths


# =============================================================================
# Main
# =============================================================================

def run_feedstock_carbon_analysis(save=True, show=True):
    scenario_dir = Path(SCENARIO_DIR)
    if not scenario_dir.exists():
        raise FileNotFoundError(
            f"Scenario directory not found: {scenario_dir}"
        )

    properties, tech_index, temp = calculate_feedstock_properties()

    available_dm, residue_dir, available_years = (
        load_available_feedstock_dm(
            scenario_dir=scenario_dir,
            mode=ANALYSIS_MODE,
        )
    )

    used_dm, selected_tech_share, used_years = (
        load_used_feedstock_dm(
            scenario_dir=scenario_dir,
            tech_index=tech_index,
            mode=ANALYSIS_MODE,
        )
    )

    expected_shape = (
        len(Region26.coords),
        len(BiocharFeedstock.coords),
    )
    if available_dm.shape != expected_shape:
        raise ValueError(
            f"Availability shape {available_dm.shape} != {expected_shape}."
        )
    if used_dm.shape != expected_shape:
        raise ValueError(
            f"Use shape {used_dm.shape} != {expected_shape}."
        )

    tables = build_analysis_tables(
        properties=properties,
        available_dm=available_dm,
        used_dm=used_dm,
    )

    rationing_enabled = _proportional_rationing_enabled(
        scenario_dir
    )

    global_mix_df = print_results_summary(
        properties=properties,
        tables=tables,
        selected_tech_share=selected_tech_share,
        temp=temp,
        residue_dir=residue_dir,
        available_years=available_years,
        used_years=used_years,
        rationing_enabled=rationing_enabled,
    )

    figure_outputs = plot_feedstock_analysis(
        properties=properties,
        tables=tables,
        save=save,
        show=show,
    )

    table_paths = {}
    if save:
        table_paths = save_tables(
            properties=properties,
            tables=tables,
            global_mix_df=global_mix_df,
        )

    return {
        "properties": properties,
        "tables": tables,
        "global_mix": global_mix_df,
        "technology_index": tech_index,
        "pyrolysis_temperature_C": temp,
        "selected_technology_use_share": selected_tech_share,
        "figure_outputs": figure_outputs,
        "table_paths": table_paths,
    }


if __name__ == "__main__":
    run_feedstock_carbon_analysis(
        save=True,
        show=SHOW_FIGURE,
    )

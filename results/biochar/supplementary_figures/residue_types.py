from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from timer.cdr.cdr_util import (
    Region26,
    Residue,
    RCCPlus,
    BiocharFeedstock,
    PyrolysisTechnology,
    biochar_data_dir,
)
from timer.cdr.results.biochar.biochar_results import IMAGE_REGIONS
from timer.cdr.biochar.feedstock import BIOCHAR_FEEDSTOCK_NAMES


"""
Standalone figure: cumulative regional residue composition in two bases.

Panels
------
a. R26 potential feedstock availability (dry matter share)
b. R26 potential feedstock availability (biochar-equivalent share)

Definitions
-----------
- Input residue potential is read from ResPotCurveDM.
- Potential is treated cumulatively over the selected analysis horizon
  (default: 2030-2100), matching the model treatment where each timestep's
  ResPotCurveDM is multiplied by its timestep width before summation.
- Biofuel-derived feedstocks are excluded.
- The dry-matter panel shows the composition of cumulative available residue
  dry matter by region.
- The biochar-equivalent panel converts cumulative residue dry matter to
  potential biochar output using the model's biochar-yield equations for the
  technology whose pyrolysis temperature is closest to TARGET_PYROLYSIS_TEMP_C.
  Shares are then calculated from that biochar-equivalent output.
- Layout follows panels c and d of the uploaded analysis module: same region
  labels, heatmap style, and numeric values inside cells.
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

# Choose the pyrolysis technology used to convert residue dry matter into
# potential biochar-equivalent output. For the climate-oriented scenario this
# is typically the technology closest to 500 C.
TARGET_PYROLYSIS_TEMP_C = 500.0

START_YEAR = 2030
END_YEAR = 2100

OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_STEM = "biochar_r26_cumulative_feedstock_availability_two_basis"
SAVE_DPI = 600
SHOW_FIGURE = True

FIGSIZE = (16.8, 8.8)

HEATMAP_VMAX_PERCENT = 70.0
ANNOTATION_THRESHOLD = 10.0
ANNOTATION_DECIMALS = 0

EXCLUDED_FEEDSTOCKS = {
    "biofuel_sugarcane",
    "biofuel_maize",
    "biofuel_woody",
    "biofuel_nwoody",
}


# =============================================================================
# Display labels copied from the latest analysis module
# =============================================================================

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


def _time_years(da):
    if "time" not in da.dims:
        return None

    coord = da["time"]
    values = coord.values

    if np.issubdtype(values.dtype, np.datetime64):
        return coord.dt.year.values.astype(int)

    return np.asarray(values, dtype=float).round().astype(int)


def _infer_timestep_widths(years):
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
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    numeric_rows = []

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
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
                token_num = token.replace("D", "E").replace("d", "e")
                try:
                    row.append(float(token_num))
                except ValueError:
                    all_numeric = False
                    break

            if all_numeric and row:
                numeric_rows.append(row)

    if not numeric_rows:
        raise ValueError(f"No numerical payload could be read from {path}")

    if expected_length is None:
        return np.asarray(
            [x for row in numeric_rows for x in row],
            dtype=np.float64,
        )

    expected_length = int(expected_length)

    if len(numeric_rows) >= expected_length:
        tail_rows = numeric_rows[-expected_length:]
        if all(len(row) >= 2 for row in tail_rows):
            first_col = np.asarray([row[0] for row in tail_rows], dtype=np.float64)
            zero_based = np.arange(expected_length, dtype=np.float64)
            one_based = zero_based + 1.0

            if np.allclose(first_col, zero_based) or np.allclose(first_col, one_based):
                return np.asarray([row[-1] for row in tail_rows], dtype=np.float64)

    flat = np.asarray([x for row in numeric_rows for x in row], dtype=np.float64)

    if flat.size < expected_length:
        raise ValueError(
            f"Could read only {flat.size} numeric values from {path}, "
            f"but expected {expected_length}."
        )

    if flat.size > expected_length:
        values = flat[-expected_length:].copy()
    else:
        values = flat

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in numerical payload of {path}")

    return values


def _feedstock_names():
    nfeed = len(BiocharFeedstock.coords)
    if nfeed != len(BIOCHAR_FEEDSTOCK_NAMES):
        raise ValueError(
            "BiocharFeedstock dimension length no longer matches "
            "BIOCHAR_FEEDSTOCK_NAMES."
        )
    return list(BIOCHAR_FEEDSTOCK_NAMES)


def _region_codes():
    nreg = len(Region26.coords)
    codes = [str(x) for x in IMAGE_REGIONS[:nreg]]
    if len(codes) != nreg:
        raise ValueError(f"Expected {nreg} IMAGE R26 region codes; found {len(codes)}.")
    return codes


# =============================================================================
# Feedstock conversion properties
# =============================================================================

def calculate_feedstock_yields():
    """
    Reproduce the carbon-yield portion of physics.py required to convert
    feedstock dry matter into biochar-equivalent output.

    Returns
    -------
    properties : pandas.DataFrame
        Indexed by feedstock name and containing:
        - Biochar yield (t biochar / tDM)
        - Feedstock input (tDM / t biochar)
    tech_index : int
        Technology index closest to TARGET_PYROLYSIS_TEMP_C.
    temp : float
        Selected pyrolysis temperature in degrees C.
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
    pyrolysis_temp = _load_dat_vector(
        biochar_data_dir / "pyrolysis_temperature.dat",
        len(PyrolysisTechnology.coords),
    )

    tech_index = int(np.nanargmin(np.abs(pyrolysis_temp - TARGET_PYROLYSIS_TEMP_C)))
    temp = float(pyrolysis_temp[tech_index])

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

    properties = pd.DataFrame(
        {
            "Feedstock": feedstocks,
            "Feedstock display": [FEEDSTOCK_DISPLAY.get(f, f) for f in feedstocks],
            "Technology index": tech_index,
            "Pyrolysis temperature (C)": temp,
            "Biochar yield (t biochar/tDM)": yield_dm,
            "Feedstock input (tDM/t biochar)": feedstock_input,
            "Biochar carbon fraction (tC/t biochar)": carbon_fraction,
        }
    ).set_index("Feedstock")

    return properties, tech_index, temp


# =============================================================================
# Load cumulative potential availability
# =============================================================================

def load_cumulative_available_feedstock_dm(scenario_dir):
    """
    Return cumulative ex-ante ResPotCurveDM supply by R26 x feedstock in tDM.

    Each timestep is multiplied by its inferred timestep width before summation,
    matching prepare_residue_supply() in the biochar model.
    """
    scenario_dir = Path(scenario_dir)
    residue_dir = _resolve_residue_directory(scenario_dir)
    path = _resolve_case_insensitive_file(residue_dir, RESIDUE_POTENTIAL_VAR)

    if not path.exists():
        raise FileNotFoundError(f"Could not find {RESIDUE_POTENTIAL_VAR} at {path}")

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
        raise ValueError(f"{RESIDUE_POTENTIAL_VAR} must contain a time dimension.")

    expected_per_time = nreg * nres * nbins_plus
    if values.size != len(years) * expected_per_time:
        raise ValueError(
            f"Unexpected shape/size for {path}: shape={values.shape}; "
            f"expected {len(years)} time steps x {nreg} x {nres} x {nbins_plus}."
        )

    values = values.reshape(len(years), nreg, nres, nbins_plus)

    nfeed = len(BiocharFeedstock.coords)
    cost_bins = len(RCCPlus.coords) - 1
    values = values[:, :, :nfeed, :cost_bins]
    values = np.where(np.isfinite(values) & (values > 0.0), values, 0.0)

    selected = (years >= int(START_YEAR)) & (years <= int(END_YEAR))
    if not np.any(selected):
        raise ValueError(f"No years in {START_YEAR}-{END_YEAR} in {path}.")

    widths = _infer_timestep_widths(years)
    dm = np.sum(
        values[selected] * widths[selected, None, None, None],
        axis=0,
        dtype=np.float64,
    )

    # Sum residue cost bins -> region x feedstock.
    dm = np.sum(dm, axis=2, dtype=np.float64)
    selected_years = years[selected].astype(int).tolist()

    return dm, residue_dir, selected_years


# =============================================================================
# Derived tables
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


def build_availability_tables(properties, available_dm):
    feedstocks = _feedstock_names()
    regions = _region_codes()

    keep = [f not in EXCLUDED_FEEDSTOCKS for f in feedstocks]
    keep_idx = np.where(keep)[0]
    kept_feedstocks = [feedstocks[i] for i in keep_idx]

    # Exclude biofuel-derived feedstocks.
    available_dm_kept = available_dm[:, keep_idx]

    feedstock_input = (
        properties.loc[kept_feedstocks, "Feedstock input (tDM/t biochar)"]
        .to_numpy(dtype=np.float64)
    )

    available_biochar = np.divide(
        available_dm_kept,
        feedstock_input[None, :],
        out=np.zeros_like(available_dm_kept, dtype=np.float64),
        where=np.isfinite(feedstock_input[None, :]) & (feedstock_input[None, :] > 0.0),
    )

    available_dm_share = _row_shares(available_dm_kept)
    available_biochar_share = _row_shares(available_biochar)

    availability_dm_df = pd.DataFrame(
        available_dm_share * 100.0,
        index=regions,
        columns=kept_feedstocks,
    )
    availability_biochar_df = pd.DataFrame(
        available_biochar_share * 100.0,
        index=regions,
        columns=kept_feedstocks,
    )

    return {
        "availability_dm_share": availability_dm_df,
        "availability_biochar_share": availability_biochar_df,
        "available_dm": pd.DataFrame(available_dm_kept, index=regions, columns=kept_feedstocks),
        "available_biochar": pd.DataFrame(available_biochar, index=regions, columns=kept_feedstocks),
    }


# =============================================================================
# Diagnostics
# =============================================================================

def print_results_summary(properties, tables, temp, residue_dir, available_years):
    availability_dm = tables["availability_dm_share"]
    availability_biochar = tables["availability_biochar_share"]

    print("\n" + "=" * 120)
    print("R26 CUMULATIVE FEEDSTOCK AVAILABILITY: DRY MATTER VS BIOCHAR-EQUIVALENT BASIS")
    print("=" * 120)
    print(f"Scenario: {SCENARIO_NAME}")
    print(f"Residue input directory: {residue_dir}")
    print(f"Analysis years: {available_years}")
    print(f"Excluded feedstocks: {', '.join(sorted(EXCLUDED_FEEDSTOCKS))}")
    print(f"Biochar-equivalent conversion temperature: {temp:.1f} C")

    prop_cols = [
        "Feedstock display",
        "Biochar yield (t biochar/tDM)",
        "Feedstock input (tDM/t biochar)",
        "Biochar carbon fraction (tC/t biochar)",
    ]
    prop_df = properties.loc[availability_dm.columns, prop_cols].copy()
    print("\nFeedstock conversion properties used for the biochar-equivalent panel")
    print(prop_df.round(5).to_string())

    print("\nCheck: row sums of dry-matter shares (should be 100)")
    print(availability_dm.sum(axis=1).round(6).to_string())

    print("\nCheck: row sums of biochar-equivalent shares (should be 100)")
    print(availability_biochar.sum(axis=1).round(6).to_string())


# =============================================================================
# Plotting (same layout logic as latest module panels c and d)
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


def _annotate_cells(ax, values, threshold=ANNOTATION_THRESHOLD, decimals=ANNOTATION_DECIMALS):
    values = np.asarray(values, dtype=np.float64)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            if np.isfinite(v) and v >= threshold:
                if decimals <= 0:
                    text = f"{v:.0f}"
                else:
                    text = f"{v:.{decimals}f}"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=6.1,
                    color="black" if v < 50.0 else "white",
                )


def plot_cumulative_availability_two_basis(tables, save=True, show=True):
    regions = list(tables["availability_dm_share"].index)
    feedstocks = list(tables["availability_dm_share"].columns)

    available_dm = tables["availability_dm_share"].to_numpy(dtype=np.float64)
    available_biochar = tables["availability_biochar_share"].to_numpy(dtype=np.float64)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.0, 1.0),
        wspace=0.20,
    )

    # ------------------------------------------------------------------
    # a. R26 potential availability composition (dry matter)
    # ------------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    im_a = ax_a.imshow(
        available_dm,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=HEATMAP_VMAX_PERCENT,
    )
    _style_heatmap_axis(ax_a, regions, feedstocks)
    _annotate_cells(ax_a, available_dm)
    ax_a.set_title(
        r"$\bf{a.}$ Feedstock supply (dry matter share)",
        loc="left",
        fontweight="normal",
    )
    cb_a = fig.colorbar(
        im_a,
        ax=ax_a,
        orientation="horizontal",
        fraction=0.046,
        pad=0.18,
        aspect=35,
        extend="max",
    )
    cb_a.set_label("Regional feedstock share (%)")

    # ------------------------------------------------------------------
    # b. R26 potential availability composition (biochar-equivalent)
    # ------------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    im_b = ax_b.imshow(
        available_biochar,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrBr",
        vmin=0.0,
        vmax=HEATMAP_VMAX_PERCENT,
    )
    _style_heatmap_axis(ax_b, regions, feedstocks)
    _annotate_cells(ax_b, available_biochar)
    ax_b.set_title(
        r"$\bf{b.}$ Feedstock supply (biochar-equivalent share)",
        loc="left",
        fontweight="normal",
    )
    cb_b = fig.colorbar(
        im_b,
        ax=ax_b,
        orientation="horizontal",
        fraction=0.046,
        pad=0.18,
        aspect=35,
        extend="max",
    )
    cb_b.set_label("Regional feedstock share (%)")

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.94,
        bottom=0.22,
    )

    outputs = {
        "figure": fig,
        "axes": (ax_a, ax_b),
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

def save_tables(properties, tables):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}

    items = {
        "feedstock_properties": properties,
        "availability_share_r26_dm": tables["availability_dm_share"],
        "availability_share_r26_biochar": tables["availability_biochar_share"],
        "available_dm": tables["available_dm"],
        "available_biochar": tables["available_biochar"],
    }

    for suffix, df in items.items():
        path = OUTPUT_DIR / f"{OUTPUT_STEM}_{suffix}.csv"
        df.to_csv(path)
        paths[f"{suffix}_csv"] = path
        print(f"Saved table: {path}")

    return paths


# =============================================================================
# Main
# =============================================================================

def run_cumulative_feedstock_availability_analysis(save=True, show=True):
    scenario_dir = Path(SCENARIO_DIR)
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")

    properties, tech_index, temp = calculate_feedstock_yields()
    available_dm, residue_dir, available_years = load_cumulative_available_feedstock_dm(
        scenario_dir=scenario_dir
    )

    expected_shape = (len(Region26.coords), len(BiocharFeedstock.coords))
    if available_dm.shape != expected_shape:
        raise ValueError(f"Availability shape {available_dm.shape} != {expected_shape}.")

    tables = build_availability_tables(
        properties=properties,
        available_dm=available_dm,
    )

    print_results_summary(
        properties=properties,
        tables=tables,
        temp=temp,
        residue_dir=residue_dir,
        available_years=available_years,
    )

    figure_outputs = plot_cumulative_availability_two_basis(
        tables=tables,
        save=save,
        show=show,
    )

    table_paths = {}
    if save:
        table_paths = save_tables(
            properties=properties.loc[tables["availability_dm_share"].columns],
            tables=tables,
        )

    return {
        "properties": properties,
        "tables": tables,
        "selected_tech_index": tech_index,
        "selected_temperature_C": temp,
        "figure_outputs": figure_outputs,
        "table_paths": table_paths,
    }


if __name__ == "__main__":
    run_cumulative_feedstock_availability_analysis(
        save=True,
        show=SHOW_FIGURE,
    )

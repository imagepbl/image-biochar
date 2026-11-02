import prism
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

from timer.cdr.cdr_util import Region
from timer.cdr.results.biochar.biochar_results import (
    R10,
    IMAGE_REGIONS,
    aggregate_r26_to_r10_sum,
)


"""
Two-panel cumulative biochar mitigation-component figure.

Panel a
-------
Biochar for mitigation, 2030-2100
Uses the biochar_c400 scenario and shows:
    - Biochar field CDR
    - Syngas CCS
    - Fossil fuel displacement
    - Direct avoided N2O
    - Direct avoided CH4

Panel b
-------
Biochar for agriculture, 2030-2100
Uses the biochar_c400_LT scenario and shows the same direct components plus:
    - Indirect land-use GHG effect

The indirect land-use GHG effect is calculated from IMAGE land-use emissions
by comparing:
    SSP2_M
against:
    SSP2_M_Biochar_altijd_yield

and excluding the direct biochar CH4/N2O effects from the IMAGE land-emissions
difference so these are not double-counted relative to the PRISM direct
BiocharCH4Mitig / BiocharN2OMitig outputs.

All cumulative mitigation values are plotted in GtCO2-eq.

The module also prints manuscript-oriented quantitative diagnostics covering
global pathway shares, direct-component shifts, regional concentration, and
the decomposition of the indirect land-use GHG effect.
"""


# =============================================================================
# Configuration
# =============================================================================

CDR_OUTPUT_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output"
)

MITIGATION_SCENARIO_DIR = CDR_OUTPUT_DIR / "biochar_c400"
AGRICULTURE_SCENARIO_DIR = CDR_OUTPUT_DIR / "biochar_c400_LT"

IMAGE_LAND_BASE_DIR = Path(
    r"X:\user\orhalem\IMAGE_CDR\3_IMAGE_land"
)

BASELINE_EMIS_DIR = (
    IMAGE_LAND_BASE_DIR
    / "scen"
    / "SSP2_M"
    / "output"
    / "emissions"
)

BIOCHAR_EMIS_DIR = (
    IMAGE_LAND_BASE_DIR
    / "scen"
    / "SSP2_M_Biochar_altijd_yield"
    / "output"
    / "emissions"
)

START_YEAR = 2030
END_YEAR = 2100

OUTPUT_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\figures"
)
OUTPUT_STEM = "biochar_mitigation_components_climate_agriculture_2panel_v4"

SAVE_DPI = 600
SHOW_FIGURE = True

# IMAGE GWP conventions used in the existing LT module.
GWP_CH4 = 30.0
GWP_N2O = 310.0
TG_N_TO_TG_N2O = 44.0 / 28.0

# Figure styling.
FIGSIZE = (15.2, 7.0)
BAR_HEIGHT = 0.72

COLORS = {
    "Biochar field CDR": "#8c510a",
    "Syngas CCS": "#d8b365",
    "Fossil fuel displacement": "#5ab4ac",
    "Direct avoided N₂O": "#4393c3",
    "Direct avoided CH₄": "#d6604d",
    "Indirect land-use GHG effect": "#4daf4a",
}

R10_DISPLAY = {
    "North America": "North America",
    "Latin America": "Latin America",
    "Europe": "Europe",
    "Eurasia": "Eurasia",
    "Middle East & North Africa": "MENA",
    "Sub-Saharan Africa": "SSA",
    "South Asia": "South Asia",
    "East Asia": "East Asia",
    "Southeast Asia": "Southeast Asia",
    "Oceania": "Oceania",
}


# =============================================================================
# IMAGE land-emission variables
# =============================================================================

LandEmisCO2Base = prism.Dimension(
    "land_emis_co2_base",
    list(range(8)),
)
LandEmisCO2Biochar = prism.Dimension(
    "land_emis_co2_biochar",
    list(range(8)),
)

LandEmisCH4Base = prism.Dimension(
    "land_emis_ch4_base",
    list(range(11)),
)
LandEmisCH4Biochar = prism.Dimension(
    "land_emis_ch4_biochar",
    list(range(11)),
)

LandEmisN2OBase = prism.Dimension(
    "land_emis_n2o_base",
    list(range(15)),
)
LandEmisN2OBiochar = prism.Dimension(
    "land_emis_n2o_biochar",
    list(range(15)),
)

simulation_timeline = prism.Timeline(
    start=prism.Q_(2025, "year"),
    end=prism.Q_(2100, "year"),
    stepsize=prism.Q_(1, "year"),
)

LUEMCO2_BASE = prism.TimeVariable[
    LandEmisCO2Base,
    Region,
    np.float32,
](
    timeline=simulation_timeline,
    file=BASELINE_EMIS_DIR / "LUEMCO2.OUT",
)

LUEMCO2_BIOCHAR = prism.TimeVariable[
    LandEmisCO2Biochar,
    Region,
    np.float32,
](
    timeline=simulation_timeline,
    file=BIOCHAR_EMIS_DIR / "LUEMCO2.OUT",
)

LUEMCH4_BASE = prism.TimeVariable[
    LandEmisCH4Base,
    Region,
    np.float32,
](
    timeline=simulation_timeline,
    file=BASELINE_EMIS_DIR / "LUEMCH4.OUT",
)

LUEMCH4_BIOCHAR = prism.TimeVariable[
    LandEmisCH4Biochar,
    Region,
    np.float32,
](
    timeline=simulation_timeline,
    file=BIOCHAR_EMIS_DIR / "LUEMCH4.OUT",
)

LUEMN2O_BASE = prism.TimeVariable[
    LandEmisN2OBase,
    Region,
    np.float32,
](
    timeline=simulation_timeline,
    file=BASELINE_EMIS_DIR / "LUEMN2O.OUT",
)

LUEMN2O_BIOCHAR = prism.TimeVariable[
    LandEmisN2OBiochar,
    Region,
    np.float32,
](
    timeline=simulation_timeline,
    file=BIOCHAR_EMIS_DIR / "LUEMN2O.OUT",
)

EMCH4BIOCHAR = prism.TimeVariable[
    Region,
    np.float32,
](
    timeline=simulation_timeline,
    file=BIOCHAR_EMIS_DIR / "EMCH4BIOCHAR.OUT",
)

EMN2OBIOCHAR = prism.TimeVariable[
    Region,
    np.float32,
](
    timeline=simulation_timeline,
    file=BIOCHAR_EMIS_DIR / "EMN2OBIOCHAR.OUT",
)


# =============================================================================
# Generic helpers
# =============================================================================

def set_publication_style():
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 12.0,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 9.5,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_title(letter, text):
    return rf"$\bf{{{letter}.}}$ {text}"


def _as_region26(arr, label="variable"):
    arr = np.asarray(arr, dtype=np.float64).squeeze()

    if arr.ndim != 1:
        raise ValueError(
            f"{label} should resolve to 1D Region26/27; "
            f"got shape {arr.shape}."
        )

    if arr.size < 26:
        raise ValueError(
            f"{label} has only {arr.size} regional values; expected >=26."
        )

    return arr[:26]


def _as_category_region_array(
    arr,
    expected_categories,
    n_region=26,
    label="variable",
):
    arr = np.asarray(arr, dtype=np.float64).squeeze()

    if arr.ndim != 2:
        raise ValueError(
            f"{label} should resolve to a category x region array; "
            f"got shape {arr.shape}."
        )

    if (
        arr.shape[0] == expected_categories
        and arr.shape[1] >= n_region
    ):
        return arr[:, :n_region]

    if (
        arr.shape[1] == expected_categories
        and arr.shape[0] >= n_region
    ):
        return arr[:n_region, :].T

    raise ValueError(
        f"{label}: could not identify expected "
        f"{expected_categories} categories and >= {n_region} regions "
        f"from shape {arr.shape}."
    )


def _years_array(start_year, end_year, step_years=1):
    return np.arange(
        int(start_year),
        int(end_year) + 1,
        int(step_years),
    )


def integrate_annual_rate(
    values_by_year_r26,
    years,
    integration="step",
    step_years=1,
):
    values_by_year_r26 = np.asarray(
        values_by_year_r26,
        dtype=np.float64,
    )
    years = np.asarray(years, dtype=np.float64)

    if integration == "step":
        return (
            np.nansum(values_by_year_r26, axis=0)
            * float(step_years)
        )

    if integration == "trapezoid":
        try:
            return np.trapezoid(
                values_by_year_r26,
                x=years,
                axis=0,
            )
        except AttributeError:
            return np.trapz(
                values_by_year_r26,
                x=years,
                axis=0,
            )

    raise ValueError(
        "integration must be 'step' or 'trapezoid'."
    )


# =============================================================================
# PRISM cumulative mitigation components
# =============================================================================

def load_region26_cumulative_to_year(
    nc_file,
    var_name,
    year,
    start_year=2030,
):
    nc_file = Path(nc_file)

    if not nc_file.exists():
        raise FileNotFoundError(
            f"Required mitigation component file not found: {nc_file}"
        )

    with xr.open_dataset(nc_file) as ds:
        if var_name not in ds.data_vars:
            raise KeyError(
                f"{var_name} not found in {nc_file}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[var_name]

        if "time" not in da.dims:
            vals = np.asarray(
                da.values,
                dtype=np.float64,
            ).squeeze()
        else:
            time_vals = np.asarray(
                da["time"].values,
            ).astype(int)

            if int(year) not in time_vals:
                raise ValueError(
                    f"Year {year} not found in {var_name}. "
                    f"Available years: {time_vals.tolist()}"
                )

            da = da.sel(
                time=slice(
                    int(start_year),
                    int(year),
                )
            )
            da = da.sum(
                dim="time",
                skipna=True,
            )
            vals = np.asarray(
                da.values,
                dtype=np.float64,
            ).squeeze()

    if vals.ndim != 1:
        raise ValueError(
            f"{var_name} from {nc_file} should be 1D after time sum. "
            f"Got shape {vals.shape}."
        )

    if vals.shape[0] == 26:
        return vals

    if vals.shape[0] >= 27:
        return vals[:26]

    raise ValueError(
        f"{var_name} has shape {vals.shape}; "
        "expected Region26 or Region27."
    )


def load_direct_components_r10(
    scenario_dir,
    start_year=START_YEAR,
    end_year=END_YEAR,
):
    scenario_dir = Path(scenario_dir)

    component_specs = [
        (
            "Biochar field CDR",
            "BiocharFieldCDR.nc",
            "BiocharFieldCDR",
        ),
        (
            "Syngas CCS",
            "BiocharSyngasCCS.nc",
            "BiocharSyngasCCS",
        ),
        (
            "Fossil fuel displacement",
            "BiocharFossilMitig.nc",
            "BiocharFossilMitig",
        ),
        (
            "Direct avoided N₂O",
            "BiocharN2OMitig.nc",
            "BiocharN2OMitig",
        ),
        (
            "Direct avoided CH₄",
            "BiocharCH4Mitig.nc",
            "BiocharCH4Mitig",
        ),
    ]

    out = {}

    for label, filename, var_name in component_specs:
        r26 = load_region26_cumulative_to_year(
            scenario_dir / filename,
            var_name,
            end_year,
            start_year,
        )

        out[label] = (
            aggregate_r26_to_r10_sum(r26)
            / 1e9
        )

    return out


# =============================================================================
# Indirect IMAGE land-use GHG effect
# =============================================================================

def load_luemco2_total_excl_biochar_annual_r26_pgcyr(
    luemco2_var,
    years,
    label="scenario",
    check_biochar_zero=False,
    tolerance=1e-8,
):
    rows = []

    for year in years:
        t = prism.Q_(int(year), "year")

        arr = _as_category_region_array(
            luemco2_var[t].values,
            expected_categories=8,
            n_region=26,
            label=f"{label} LUEMCO2 {year}",
        )

        explicit_biochar = arr[6, :]
        total = arr[7, :]

        if check_biochar_zero:
            max_abs = np.nanmax(
                np.abs(explicit_biochar)
            )

            if max_abs > tolerance:
                raise ValueError(
                    f"{label} LUEMCO2 explicit biochar category "
                    f"is not zero in {year}. "
                    f"Max abs={max_abs:.6e} PgC/yr."
                )

        rows.append(
            total - explicit_biochar
        )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


def load_luemco2_indirect_r26_gtco2(
    luemco2_base,
    luemco2_biochar,
    start_year=START_YEAR,
    end_year=END_YEAR,
    integration="step",
):
    years = _years_array(
        start_year,
        end_year,
        1,
    )

    base = load_luemco2_total_excl_biochar_annual_r26_pgcyr(
        luemco2_var=luemco2_base,
        years=years,
        label="baseline",
        check_biochar_zero=True,
    )

    bio = load_luemco2_total_excl_biochar_annual_r26_pgcyr(
        luemco2_var=luemco2_biochar,
        years=years,
        label="biochar",
        check_biochar_zero=False,
    )

    avoided_pgcyr = base - bio

    cumulative_pgc = integrate_annual_rate(
        avoided_pgcyr,
        years=years,
        integration=integration,
        step_years=1,
    )

    # PgC -> GtCO2
    return cumulative_pgc * (44.0 / 12.0)


def load_luem_total_annual_r26(
    luem_var,
    years,
    expected_categories,
    total_index,
    label,
):
    rows = []

    for year in years:
        t = prism.Q_(int(year), "year")

        arr = _as_category_region_array(
            luem_var[t].values,
            expected_categories=expected_categories,
            n_region=26,
            label=f"{label} {year}",
        )

        rows.append(
            arr[int(total_index), :]
        )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


def load_em_biochar_direct_annual_r26(
    em_biochar_var,
    years,
    label,
):
    rows = []

    for year in years:
        t = prism.Q_(int(year), "year")
        rows.append(
            _as_region26(
                em_biochar_var[t].values,
                label=f"{label} {year}",
            )
        )

    return np.asarray(
        rows,
        dtype=np.float64,
    )


def convert_ch4_tg_to_gtco2e(
    values_tg_ch4,
    gwp_ch4=GWP_CH4,
):
    return (
        np.asarray(
            values_tg_ch4,
            dtype=np.float64,
        )
        * float(gwp_ch4)
        / 1000.0
    )


def convert_n2o_tgn_to_gtco2e(
    values_tg_n,
    gwp_n2o=GWP_N2O,
):
    return (
        np.asarray(
            values_tg_n,
            dtype=np.float64,
        )
        * TG_N_TO_TG_N2O
        * float(gwp_n2o)
        / 1000.0
    )


def load_luemch4_indirect_r26_gtco2e(
    luemch4_base,
    luemch4_biochar,
    emch4biochar,
    start_year=START_YEAR,
    end_year=END_YEAR,
    integration="step",
    gwp_ch4=GWP_CH4,
):
    years = _years_array(
        start_year,
        end_year,
        1,
    )

    base_total_tg = load_luem_total_annual_r26(
        luem_var=luemch4_base,
        years=years,
        expected_categories=11,
        total_index=10,
        label="baseline LUEMCH4",
    )

    bio_total_tg = load_luem_total_annual_r26(
        luem_var=luemch4_biochar,
        years=years,
        expected_categories=11,
        total_index=10,
        label="biochar LUEMCH4",
    )

    # Signed direct biochar emission change from IMAGE.
    direct_tg = load_em_biochar_direct_annual_r26(
        em_biochar_var=emch4biochar,
        years=years,
        label="EMCH4BIOCHAR",
    )

    # Remove direct biochar CH4 from IMAGE totals before calculating
    # the indirect scenario difference.
    bio_total_excl_direct_tg = (
        bio_total_tg - direct_tg
    )

    indirect_avoided_tgyr = (
        base_total_tg
        - bio_total_excl_direct_tg
    )

    cumulative_tg = integrate_annual_rate(
        indirect_avoided_tgyr,
        years=years,
        integration=integration,
        step_years=1,
    )

    return convert_ch4_tg_to_gtco2e(
        cumulative_tg,
        gwp_ch4=gwp_ch4,
    )


def load_luemn2o_indirect_r26_gtco2e(
    luemn2o_base,
    luemn2o_biochar,
    emn2obiochar,
    start_year=START_YEAR,
    end_year=END_YEAR,
    integration="step",
    gwp_n2o=GWP_N2O,
):
    years = _years_array(
        start_year,
        end_year,
        1,
    )

    base_total_tgn = load_luem_total_annual_r26(
        luem_var=luemn2o_base,
        years=years,
        expected_categories=15,
        total_index=14,
        label="baseline LUEMN2O",
    )

    bio_total_tgn = load_luem_total_annual_r26(
        luem_var=luemn2o_biochar,
        years=years,
        expected_categories=15,
        total_index=14,
        label="biochar LUEMN2O",
    )

    direct_tgn = load_em_biochar_direct_annual_r26(
        em_biochar_var=emn2obiochar,
        years=years,
        label="EMN2OBIOCHAR",
    )

    # Remove direct biochar N2O from IMAGE totals before calculating
    # the indirect scenario difference.
    bio_total_excl_direct_tgn = (
        bio_total_tgn - direct_tgn
    )

    indirect_avoided_tgnyr = (
        base_total_tgn
        - bio_total_excl_direct_tgn
    )

    cumulative_tgn = integrate_annual_rate(
        indirect_avoided_tgnyr,
        years=years,
        integration=integration,
        step_years=1,
    )

    return convert_n2o_tgn_to_gtco2e(
        cumulative_tgn,
        gwp_n2o=gwp_n2o,
    )


def load_indirect_landuse_ghg_r10(
    start_year=START_YEAR,
    end_year=END_YEAR,
    integration="step",
):
    """
    Indirect land-use GHG effect for panel b.

    Positive values mean lower cumulative land-use GHG emissions in
    SSP2_M_Biochar_altijd_yield relative to SSP2_M.
    """
    indirect_co2_r26 = (
        load_luemco2_indirect_r26_gtco2(
            luemco2_base=LUEMCO2_BASE,
            luemco2_biochar=LUEMCO2_BIOCHAR,
            start_year=start_year,
            end_year=end_year,
            integration=integration,
        )
    )

    indirect_ch4_r26 = (
        load_luemch4_indirect_r26_gtco2e(
            luemch4_base=LUEMCH4_BASE,
            luemch4_biochar=LUEMCH4_BIOCHAR,
            emch4biochar=EMCH4BIOCHAR,
            start_year=start_year,
            end_year=end_year,
            integration=integration,
        )
    )

    indirect_n2o_r26 = (
        load_luemn2o_indirect_r26_gtco2e(
            luemn2o_base=LUEMN2O_BASE,
            luemn2o_biochar=LUEMN2O_BIOCHAR,
            emn2obiochar=EMN2OBIOCHAR,
            start_year=start_year,
            end_year=end_year,
            integration=integration,
        )
    )

    indirect_total_r26 = (
        indirect_co2_r26
        + indirect_ch4_r26
        + indirect_n2o_r26
    )

    return {
        "Indirect land-use GHG effect":
            aggregate_r26_to_r10_sum(
                indirect_total_r26
            ),
        "Indirect LUEMCO2":
            aggregate_r26_to_r10_sum(
                indirect_co2_r26
            ),
        "Indirect LUEMCH4":
            aggregate_r26_to_r10_sum(
                indirect_ch4_r26
            ),
        "Indirect LUEMN2O":
            aggregate_r26_to_r10_sum(
                indirect_n2o_r26
            ),
    }


# =============================================================================
# Data assembly
# =============================================================================

def build_component_dataframe(
    scenario_dir,
    include_indirect_landuse=False,
):
    direct = load_direct_components_r10(
        scenario_dir=scenario_dir,
        start_year=START_YEAR,
        end_year=END_YEAR,
    )

    data = {
        "Biochar field CDR":
            direct["Biochar field CDR"],
        "Syngas CCS":
            direct["Syngas CCS"],
        "Fossil fuel displacement":
            direct["Fossil fuel displacement"],
        "Direct avoided N₂O":
            direct["Direct avoided N₂O"],
        "Direct avoided CH₄":
            direct["Direct avoided CH₄"],
    }

    indirect_detail = None

    if include_indirect_landuse:
        indirect_detail = (
            load_indirect_landuse_ghg_r10(
                start_year=START_YEAR,
                end_year=END_YEAR,
                integration="step",
            )
        )

        data[
            "Indirect land-use GHG effect"
        ] = indirect_detail[
            "Indirect land-use GHG effect"
        ]
    else:
        data[
            "Indirect land-use GHG effect"
        ] = np.zeros(
            len(R10),
            dtype=np.float64,
        )

    df = pd.DataFrame(
        data,
        index=list(R10.keys()),
    )

    df["Total"] = df[
        list(COLORS.keys())
    ].sum(axis=1)

    return df, indirect_detail


# =============================================================================
# Plotting
# =============================================================================

def plot_stacked_components(
    ax,
    df,
    show_ylabels=True,
):
    regions = list(R10.keys())
    y = np.arange(len(regions))

    positive_left = np.zeros(
        len(regions),
        dtype=np.float64,
    )
    negative_left = np.zeros(
        len(regions),
        dtype=np.float64,
    )

    for label in COLORS:
        values = df[
            label
        ].to_numpy(
            dtype=np.float64,
        )

        pos = np.where(
            values > 0.0,
            values,
            0.0,
        )
        neg = np.where(
            values < 0.0,
            values,
            0.0,
        )

        ax.barh(
            y,
            pos,
            left=positive_left,
            height=BAR_HEIGHT,
            color=COLORS[label],
            edgecolor="none",
        )

        ax.barh(
            y,
            neg,
            left=negative_left,
            height=BAR_HEIGHT,
            color=COLORS[label],
            edgecolor="none",
        )

        positive_left += pos
        negative_left += neg

    ax.axvline(
        0.0,
        color="black",
        linewidth=1.0,
    )

    ax.set_yticks(y)

    if show_ylabels:
        ax.set_yticklabels(
            [
                R10_DISPLAY.get(r, r)
                for r in regions
            ]
        )
    else:
        ax.tick_params(
            axis="y",
            which="both",
            left=False,
            labelleft=False,
        )

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )
    ax.set_axisbelow(True)

    for spine in (
        "top",
        "right",
        "bottom",
        "left",
    ):
        ax.spines[spine].set_linewidth(0.8)


def shared_xlim(*dfs):
    xmin = 0.0
    xmax = 0.0

    for df in dfs:
        values = df[
            list(COLORS.keys())
        ].to_numpy(
            dtype=np.float64,
        )

        positive_extent = np.nansum(
            np.where(
                values > 0.0,
                values,
                0.0,
            ),
            axis=1,
        )
        negative_extent = np.nansum(
            np.where(
                values < 0.0,
                values,
                0.0,
            ),
            axis=1,
        )

        xmin = min(
            xmin,
            float(
                np.nanmin(
                    negative_extent
                )
            ),
        )
        xmax = max(
            xmax,
            float(
                np.nanmax(
                    positive_extent
                )
            ),
        )

    span = xmax - xmin
    pad = (
        0.04 * span
        if span > 0.0
        else 1.0
    )

    return (
        xmin - pad,
        xmax + pad,
    )


def legend_handles():
    return [
        Patch(
            facecolor=COLORS[label],
            edgecolor="none",
            label=label,
        )
        for label in COLORS
    ]


# =============================================================================
# Quantitative manuscript diagnostics
# =============================================================================

DIRECT_COMPONENTS = [
    "Biochar field CDR",
    "Syngas CCS",
    "Fossil fuel displacement",
    "Direct avoided N₂O",
    "Direct avoided CH₄",
]


def _safe_percent(numerator, denominator):
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return np.nan
    if abs(denominator) < 1e-12:
        return np.nan
    return 100.0 * numerator / denominator


def _fmt_percent(value, digits=1):
    if not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}%"


def _global_component_summary(df):
    rows = []
    total = float(df["Total"].sum())

    for component in COLORS:
        value = float(df[component].sum())
        rows.append(
            {
                "Component": component,
                "GtCO2-eq": value,
                "Share of total (%)": _safe_percent(value, total),
            }
        )

    return pd.DataFrame(rows).set_index("Component")


def _regional_summary(df, global_total):
    out = df[["Total"]].copy()
    out["Share of global total (%)"] = (
        100.0 * out["Total"] / float(global_total)
        if abs(float(global_total)) > 1e-12
        else np.nan
    )
    return out.sort_values("Total", ascending=False)


def print_quantitative_results(
    mitigation_df,
    agriculture_df,
    indirect_detail,
):
    """
    Print manuscript-oriented quantitative diagnostics for Fig. 2.

    The diagnostics are descriptive and use the same cumulative 2030-2100
    values that are plotted. They distinguish direct PRISM mitigation from
    the indirect IMAGE land-use GHG effect in the agriculture-oriented case.
    """

    mitigation_total = float(mitigation_df["Total"].sum())
    agriculture_total = float(agriculture_df["Total"].sum())

    mitigation_direct = float(
        mitigation_df[DIRECT_COMPONENTS].to_numpy(dtype=np.float64).sum()
    )
    agriculture_direct = float(
        agriculture_df[DIRECT_COMPONENTS].to_numpy(dtype=np.float64).sum()
    )
    agriculture_indirect = float(
        agriculture_df["Indirect land-use GHG effect"].sum()
    )

    climate_components = _global_component_summary(mitigation_df)
    agriculture_components = _global_component_summary(agriculture_df)

    # Direct-component comparison between deployment strategies.
    comparison_rows = []
    for component in DIRECT_COMPONENTS:
        climate_value = float(mitigation_df[component].sum())
        agriculture_value = float(agriculture_df[component].sum())
        delta = agriculture_value - climate_value
        comparison_rows.append(
            {
                "Component": component,
                "Climate (GtCO2-eq)": climate_value,
                "Agriculture (GtCO2-eq)": agriculture_value,
                "Agriculture - climate (GtCO2-eq)": delta,
                "Relative change (%)": _safe_percent(delta, climate_value),
                "Climate share of direct (%)": _safe_percent(
                    climate_value,
                    mitigation_direct,
                ),
                "Agriculture share of direct (%)": _safe_percent(
                    agriculture_value,
                    agriculture_direct,
                ),
            }
        )
    component_comparison = pd.DataFrame(comparison_rows).set_index("Component")

    # Useful grouped pathway totals.
    climate_energy = float(
        mitigation_df[["Syngas CCS", "Fossil fuel displacement"]]
        .to_numpy(dtype=np.float64)
        .sum()
    )
    agriculture_energy = float(
        agriculture_df[["Syngas CCS", "Fossil fuel displacement"]]
        .to_numpy(dtype=np.float64)
        .sum()
    )
    climate_soil = float(
        mitigation_df[["Direct avoided N₂O", "Direct avoided CH₄"]]
        .to_numpy(dtype=np.float64)
        .sum()
    )
    agriculture_soil = float(
        agriculture_df[["Direct avoided N₂O", "Direct avoided CH₄"]]
        .to_numpy(dtype=np.float64)
        .sum()
    )
    climate_nonfield = mitigation_direct - float(
        mitigation_df["Biochar field CDR"].sum()
    )
    agriculture_nonfield_direct = agriculture_direct - float(
        agriculture_df["Biochar field CDR"].sum()
    )
    agriculture_nonfield_all = agriculture_nonfield_direct + agriculture_indirect

    # Regional summaries and cross-strategy differences.
    climate_regional = _regional_summary(mitigation_df, mitigation_total)
    agriculture_regional = _regional_summary(agriculture_df, agriculture_total)

    regional_comparison = pd.DataFrame(
        {
            "Climate total (GtCO2-eq)": mitigation_df["Total"],
            "Agriculture total (GtCO2-eq)": agriculture_df["Total"],
        }
    )
    regional_comparison["Agriculture - climate (GtCO2-eq)"] = (
        regional_comparison["Agriculture total (GtCO2-eq)"]
        - regional_comparison["Climate total (GtCO2-eq)"]
    )
    regional_comparison["Relative change (%)"] = [
        _safe_percent(delta, base)
        for delta, base in zip(
            regional_comparison["Agriculture - climate (GtCO2-eq)"],
            regional_comparison["Climate total (GtCO2-eq)"],
        )
    ]

    # IMAGE indirect-land-use decomposition.
    indirect_df = None
    if indirect_detail is not None:
        indirect_df = pd.DataFrame(
            {
                key: np.asarray(value, dtype=np.float64)
                for key, value in indirect_detail.items()
            },
            index=list(R10.keys()),
        )

    print("\n" + "=" * 118)
    print(
        f"FIGURE 2 QUANTITATIVE RESULTS OVERVIEW | cumulative {START_YEAR}-{END_YEAR}"
    )
    print("=" * 118)

    print("\n1. GLOBAL MITIGATION TOTALS")
    print("-" * 118)
    print(
        f"Climate-oriented deployment:     {mitigation_total:8.2f} GtCO2-eq"
    )
    print(
        f"Agriculture-oriented deployment: {agriculture_total:8.2f} GtCO2-eq"
    )
    global_delta = agriculture_total - mitigation_total
    print(
        f"Agriculture minus climate:        {global_delta:+8.2f} GtCO2-eq "
        f"({_fmt_percent(_safe_percent(global_delta, mitigation_total))})"
    )
    direct_delta = agriculture_direct - mitigation_direct
    print(
        f"Direct mitigation only, agriculture minus climate: {direct_delta:+.2f} GtCO2-eq "
        f"({_fmt_percent(_safe_percent(direct_delta, mitigation_direct))})"
    )
    print(
        f"Indirect land-use GHG effect in agriculture case:   {agriculture_indirect:+.2f} GtCO2-eq "
        f"({_fmt_percent(_safe_percent(agriculture_indirect, agriculture_total))} of total; "
        f"{_fmt_percent(_safe_percent(agriculture_indirect, agriculture_direct))} relative to direct mitigation)"
    )

    print("\n2. GLOBAL MITIGATION COMPOSITION")
    print("-" * 118)
    print("Climate-oriented deployment")
    for component, row in climate_components.iterrows():
        if component == "Indirect land-use GHG effect":
            continue
        print(
            f"  {component:<31s} {row['GtCO2-eq']:8.2f} GtCO2-eq  "
            f"({_fmt_percent(row['Share of total (%)'])})"
        )
    print(
        f"  {'All non-field direct pathways':<31s} {climate_nonfield:8.2f} GtCO2-eq  "
        f"({_fmt_percent(_safe_percent(climate_nonfield, mitigation_direct))} of direct total)"
    )
    print(
        f"  {'Energy-related (syngas + fossil)':<31s} {climate_energy:8.2f} GtCO2-eq  "
        f"({_fmt_percent(_safe_percent(climate_energy, mitigation_direct))} of direct total)"
    )
    print(
        f"  {'Direct soil GHG (N2O + CH4)':<31s} {climate_soil:8.2f} GtCO2-eq  "
        f"({_fmt_percent(_safe_percent(climate_soil, mitigation_direct))} of direct total)"
    )

    print("\nAgriculture-oriented deployment")
    for component, row in agriculture_components.iterrows():
        print(
            f"  {component:<31s} {row['GtCO2-eq']:8.2f} GtCO2-eq  "
            f"({_fmt_percent(row['Share of total (%)'])})"
        )
    print(
        f"  {'All non-field direct pathways':<31s} {agriculture_nonfield_direct:8.2f} GtCO2-eq  "
        f"({_fmt_percent(_safe_percent(agriculture_nonfield_direct, agriculture_direct))} of direct total)"
    )
    print(
        f"  {'All non-field incl. land-use':<31s} {agriculture_nonfield_all:8.2f} GtCO2-eq  "
        f"({_fmt_percent(_safe_percent(agriculture_nonfield_all, agriculture_total))} of total)"
    )
    print(
        f"  {'Energy-related (syngas + fossil)':<31s} {agriculture_energy:8.2f} GtCO2-eq  "
        f"({_fmt_percent(_safe_percent(agriculture_energy, agriculture_direct))} of direct total)"
    )
    print(
        f"  {'Direct soil GHG (N2O + CH4)':<31s} {agriculture_soil:8.2f} GtCO2-eq  "
        f"({_fmt_percent(_safe_percent(agriculture_soil, agriculture_direct))} of direct total)"
    )

    print("\n3. DIRECT COMPONENT SHIFTS: AGRICULTURE MINUS CLIMATE")
    print("-" * 118)
    print(
        f"{'Component':<31s}{'Climate':>12s}{'Agriculture':>14s}{'Delta':>12s}{'Rel.':>10s}"
    )
    for component, row in component_comparison.iterrows():
        print(
            f"{component:<31s}"
            f"{row['Climate (GtCO2-eq)']:12.2f}"
            f"{row['Agriculture (GtCO2-eq)']:14.2f}"
            f"{row['Agriculture - climate (GtCO2-eq)']:12.2f}"
            f"{_fmt_percent(row['Relative change (%)']):>10s}"
        )

    print("\n4. REGIONAL DISTRIBUTION")
    print("-" * 118)
    for label, regional, total in (
        ("Climate-oriented", climate_regional, mitigation_total),
        ("Agriculture-oriented", agriculture_regional, agriculture_total),
    ):
        top3 = regional.head(3)
        top3_share = float(top3["Share of global total (%)"].sum())
        print(
            f"{label}: top-3 regions provide {top3_share:.1f}% of global cumulative mitigation"
        )
        for region, row in regional.head(5).iterrows():
            print(
                f"  {R10_DISPLAY.get(region, region):<24s} "
                f"{row['Total']:8.2f} GtCO2-eq  "
                f"({_fmt_percent(row['Share of global total (%)'])})"
            )

    print("\nLargest regional shifts in total mitigation (agriculture minus climate)")
    regional_sorted = regional_comparison.sort_values(
        "Agriculture - climate (GtCO2-eq)",
        ascending=False,
    )
    for region, row in regional_sorted.head(3).iterrows():
        print(
            f"  Increase | {R10_DISPLAY.get(region, region):<24s} "
            f"{row['Agriculture - climate (GtCO2-eq)']:+8.2f} GtCO2-eq  "
            f"({_fmt_percent(row['Relative change (%)'])})"
        )
    for region, row in regional_sorted.tail(3).sort_values(
        "Agriculture - climate (GtCO2-eq)"
    ).iterrows():
        print(
            f"  Decrease | {R10_DISPLAY.get(region, region):<24s} "
            f"{row['Agriculture - climate (GtCO2-eq)']:+8.2f} GtCO2-eq  "
            f"({_fmt_percent(row['Relative change (%)'])})"
        )

    if indirect_df is not None:
        print("\n5. INDIRECT LAND-USE GHG EFFECT IN AGRICULTURE CASE")
        print("-" * 118)
        indirect_global = indirect_df.sum(axis=0)
        indirect_total = float(indirect_global["Indirect land-use GHG effect"])
        for key in (
            "Indirect LUEMCO2",
            "Indirect LUEMCH4",
            "Indirect LUEMN2O",
        ):
            value = float(indirect_global[key])
            print(
                f"  {key:<24s} {value:+8.2f} GtCO2-eq  "
                f"({_fmt_percent(_safe_percent(value, indirect_total))} of indirect total)"
            )

        indirect_by_region = indirect_df[
            "Indirect land-use GHG effect"
        ].sort_values(ascending=False)
        n_positive = int((indirect_by_region > 0.0).sum())
        n_negative = int((indirect_by_region < 0.0).sum())
        print(
            f"  Positive indirect effect in {n_positive}/{len(indirect_by_region)} R10 regions; "
            f"negative in {n_negative}/{len(indirect_by_region)}."
        )
        print("  Strongest positive regional effects:")
        for region, value in indirect_by_region.head(3).items():
            print(
                f"    {R10_DISPLAY.get(region, region):<24s} {value:+8.2f} GtCO2-eq"
            )
        if n_negative > 0:
            print("  Most negative regional effects:")
            for region, value in indirect_by_region[indirect_by_region < 0.0].tail(3).items():
                print(
                    f"    {R10_DISPLAY.get(region, region):<24s} {value:+8.2f} GtCO2-eq"
                )

    # Identify dominant global components, useful for concise Results prose.
    climate_dominant = climate_components[
        climate_components.index != "Indirect land-use GHG effect"
    ]["GtCO2-eq"].idxmax()
    agriculture_dominant = agriculture_components["GtCO2-eq"].idxmax()

    print("\n6. RESULTS-WRITING SNAPSHOT")
    print("-" * 118)
    print(
        f"  Largest climate-oriented pathway: {climate_dominant} "
        f"({climate_components.loc[climate_dominant, 'GtCO2-eq']:.2f} GtCO2-eq; "
        f"{_fmt_percent(climate_components.loc[climate_dominant, 'Share of total (%)'])})."
    )
    print(
        f"  Largest agriculture-oriented pathway: {agriculture_dominant} "
        f"({agriculture_components.loc[agriculture_dominant, 'GtCO2-eq']:.2f} GtCO2-eq; "
        f"{_fmt_percent(agriculture_components.loc[agriculture_dominant, 'Share of total (%)'])})."
    )
    print(
        f"  Agriculture-oriented deployment changes total cumulative mitigation by "
        f"{global_delta:+.2f} GtCO2-eq ({_fmt_percent(_safe_percent(global_delta, mitigation_total))}) "
        f"relative to climate-oriented deployment."
    )
    print(
        f"  Of the agriculture-oriented total, {agriculture_indirect:+.2f} GtCO2-eq "
        f"({_fmt_percent(_safe_percent(agriculture_indirect, agriculture_total))}) is attributable to "
        f"the indirect land-use GHG effect."
    )
    print("=" * 118)

    return {
        "climate_components_global": climate_components,
        "agriculture_components_global": agriculture_components,
        "component_comparison": component_comparison,
        "climate_regional": climate_regional,
        "agriculture_regional": agriculture_regional,
        "regional_comparison": regional_comparison,
        "indirect_landuse_r10": indirect_df,
    }


# =============================================================================
# Main figure
# =============================================================================

def plot_two_panel_mitigation_components(
    save=True,
    show=SHOW_FIGURE,
):
    set_publication_style()

    mitigation_df, _ = (
        build_component_dataframe(
            MITIGATION_SCENARIO_DIR,
            include_indirect_landuse=False,
        )
    )

    agriculture_df, indirect_detail = (
        build_component_dataframe(
            AGRICULTURE_SCENARIO_DIR,
            include_indirect_landuse=True,
        )
    )

    xlim = shared_xlim(
        mitigation_df,
        agriculture_df,
    )

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=FIGSIZE,
        sharey=False,
        gridspec_kw={
            "wspace": 0.08,
        },
    )

    ax_a, ax_b = axes

    plot_stacked_components(
        ax_a,
        mitigation_df,
        show_ylabels=True,
    )

    plot_stacked_components(
        ax_b,
        agriculture_df,
        show_ylabels=False,
    )

    ax_a.set_xlim(*xlim)
    ax_b.set_xlim(*xlim)

    xlabel = (
        r"Cumulative mitigation "
        r"(GtCO$_2$-eq)"
    )

    ax_a.set_xlabel(xlabel)
    ax_b.set_xlabel(xlabel)

    ax_a.set_title(
        panel_title(
            "a",
            "Biochar for mitigation, 2030-2100",
        ),
        loc="left",
        fontweight="normal",
        pad=10,
    )

    ax_b.set_title(
        panel_title(
            "b",
            "Biochar for agriculture, 2030-2100",
        ),
        loc="left",
        fontweight="normal",
        pad=10,
    )

    annotation_text = r"SSP2 carbon price 400 \$ t CO$_2^{-1}$"

    for ax in (ax_a, ax_b):
        ax.text(
            25.0,
            len(R10) - 1.0,
            annotation_text,
            ha="left",
            va="center",
            fontsize=9.4,
        )

    # One common legend for both panels.
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
        fontsize=9.8,
        columnspacing=1.5,
        handletextpad=0.55,
    )

    fig.subplots_adjust(
        left=0.145,
        right=0.985,
        top=0.93,
        bottom=0.18,
    )

    outputs = {
        "fig": fig,
        "axes": axes,
        "mitigation_df": mitigation_df,
        "agriculture_df": agriculture_df,
        "indirect_detail": indirect_detail,
        "xlim": xlim,
    }

    if save:
        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        png_path = (
            OUTPUT_DIR
            / f"{OUTPUT_STEM}.png"
        )
        pdf_path = (
            OUTPUT_DIR
            / f"{OUTPUT_STEM}.pdf"
        )
        a_csv = (
            OUTPUT_DIR
            / f"{OUTPUT_STEM}_panel_a.csv"
        )
        b_csv = (
            OUTPUT_DIR
            / f"{OUTPUT_STEM}_panel_b.csv"
        )

        fig.savefig(
            png_path,
            dpi=SAVE_DPI,
            bbox_inches="tight",
        )
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        mitigation_df.to_csv(a_csv)
        agriculture_df.to_csv(b_csv)

        outputs.update(
            {
                "png_path": png_path,
                "pdf_path": pdf_path,
                "panel_a_csv": a_csv,
                "panel_b_csv": b_csv,
            }
        )

        print("\nSaved:")
        print(png_path)
        print(pdf_path)
        print(a_csv)
        print(b_csv)

    quantitative_results = print_quantitative_results(
        mitigation_df=mitigation_df,
        agriculture_df=agriculture_df,
        indirect_detail=indirect_detail,
    )
    outputs["quantitative_results"] = quantitative_results

    if show:
        plt.show()

    return outputs


if __name__ == "__main__":
    outputs = plot_two_panel_mitigation_components(
        save=True,
        show=True,
    )
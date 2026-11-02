from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from timer.cdr.results.biochar.biochar_results import R10, IMAGE_REGIONS


"""
Four-panel R10 realised-cost figure for climate- and agriculture-oriented
biochar scenarios.

Panels
------
a. Biochar for climate, 2100
   Absolute realised cost-component intensities for biochar_c400.

b. Stylised biochar minus main, climate, 2100
   Difference in realised cost-component intensities:
       biochar_c400_simple - biochar_c400

c. Biochar for agriculture, 2100
   Absolute realised cost-component intensities for biochar_c400_LT.

d. Stylised biochar minus main, agriculture, 2100
   Difference in realised cost-component intensities:
       biochar_c400_LT_simple - biochar_c400_LT

Transport-carbon costs are not loaded or plotted.

All panels use realised deployment outputs:
    BiocharRealisedMitigation.nc
    BiocharRealisedCostFeedstock.nc
    BiocharRealisedCostProduction.nc
    BiocharRealisedCostApplication.nc
    BiocharRealisedCostTransport.nc
    BiocharRealisedRevenueYield.nc
    BiocharRealisedRevenueBioenergy.nc
    BiocharRealisedNetCost.nc

For each R10 region:
    component intensity = sum_R26(realised monetary component)
                          / sum_R26(realised mitigation)

Revenues are negative costs.

IMPORTANT
---------
Each main scenario is compared against its matching stylised-biochar scenario:
    panel b = biochar_c400_simple - biochar_c400
    panel d = biochar_c400_LT_simple - biochar_c400_LT
"""


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output")

CLIMATE_SCENARIO = "biochar_c400"
CLIMATE_SIMPLE_SCENARIO = "biochar_c400_simple"

AGRICULTURE_SCENARIO = "biochar_c400_LT"
AGRICULTURE_SIMPLE_SCENARIO = "biochar_c400_LT_simple"

YEAR = 2100
CUMULATIVE = False
START_YEAR = 2030

OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_STEM = "biochar_realised_cost_components_climate_agriculture_4panel_v6"

SAVE_DPI = 600
SHOW_FIGURE = True

# Layout: difference panels intentionally narrower.
FIGSIZE = (15.2, 9.0)
WIDTH_RATIOS = (1.28, 0.86)
HEIGHT_RATIOS = (1.0, 1.0)

BAR_HEIGHT = 0.72
MARKER_SIZE = 42
FRAME_COLOR = "0.35"
FRAME_WIDTH = 0.9
GRID_ALPHA = 0.28
N_X_TICKS = 6

# Sanity-check tolerances.
CLOSURE_TOLERANCE = 1e-3
CLIMATE_SIMPLE_YIELD_INTENSITY_TOLERANCE = 1e-6
YIELD_DECOMPOSITION_TOLERANCE = 1e-8

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

# Same palette in all four panels.
COLORS = {
    "Feedstock cost": "#8C510A",
    "Production cost": "#D95F02",
    "Application cost": "#756BB1",
    "Transport cost": "#4C78A8",
    "Crop yield revenue": "#A6D96A",
    "Bio-oil revenue": "#1B7837",
}

POSITIVE_COMPONENTS = (
    ("Feedstock cost", "BiocharRealisedCostFeedstock"),
    ("Production cost", "BiocharRealisedCostProduction"),
    ("Application cost", "BiocharRealisedCostApplication"),
    ("Transport cost", "BiocharRealisedCostTransport"),
)

NEGATIVE_COMPONENTS = (
    ("Crop yield revenue", "BiocharRealisedRevenueYield"),
    ("Bio-oil revenue", "BiocharRealisedRevenueBioenergy"),
)

ALL_COMPONENT_LABELS = (
    *[label for label, _ in POSITIVE_COMPONENTS],
    *[label for label, _ in NEGATIVE_COMPONENTS],
)

MITIGATION_VAR = "BiocharRealisedMitigation"
NET_VAR = "BiocharRealisedNetCost"

ABSOLUTE_XLABEL = (
    r"Realised cost component "
    r"($\mathrm{US}\$_{2025}\,\mathrm{tCO_2\!-\!eq}^{-1}$)"
)

DIFFERENCE_XLABEL = (
    r"Change in realised cost component "
    r"($\mathrm{US}\$_{2025}\,\mathrm{tCO_2\!-\!eq}^{-1}$)"
)


# =============================================================================
# Publication style
# =============================================================================

def _set_publication_style():
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 10.0,
            "axes.linewidth": FRAME_WIDTH,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_title(letter, text):
    return rf"$\bf{{{letter}.}}$ {text}"


# =============================================================================
# Data loading / R26 -> R10
# =============================================================================

def _load_r26_total(
    scenario_dir,
    var_name,
    year,
    cumulative=CUMULATIVE,
    start_year=START_YEAR,
):
    nc_file = Path(scenario_dir) / f"{var_name}.nc"

    if not nc_file.exists():
        raise FileNotFoundError(f"Required realised output not found: {nc_file}")

    with xr.open_dataset(nc_file) as ds:
        if var_name not in ds:
            raise KeyError(
                f"Variable '{var_name}' not found in {nc_file.name}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[var_name]

        if "time" in da.dims:
            years = np.asarray(da["time"].values, dtype=int)

            if int(year) not in years:
                raise ValueError(
                    f"Year {year} not found in {nc_file.name}. "
                    f"Available years: {years.tolist()}"
                )

            if cumulative:
                da = da.sel(time=slice(int(start_year), int(year))).sum(
                    dim="time",
                    skipna=True,
                )
            else:
                da = da.sel(time=int(year))

        if "region" not in da.dims:
            raise ValueError(
                f"{nc_file.name}:{var_name} has no 'region' dimension."
            )

        if "tech" in da.dims:
            da = da.sum(dim="tech", skipna=True)

        extra_dims = [d for d in da.dims if d != "region"]
        if extra_dims:
            da = da.sum(dim=extra_dims, skipna=True)

        values = np.asarray(da.values, dtype=np.float64).squeeze()

    if values.ndim != 1 or values.size != len(IMAGE_REGIONS):
        raise ValueError(
            f"{nc_file.name}:{var_name} did not resolve to R26. "
            f"Expected {len(IMAGE_REGIONS)} values, got {values.shape}."
        )

    return values


def _aggregate_r26_to_r10_sum(values_r26):
    values_r26 = np.asarray(values_r26, dtype=np.float64)

    return np.asarray(
        [
            np.nansum(
                values_r26[[IMAGE_REGIONS.index(r) for r in members]]
            )
            for members in R10.values()
        ],
        dtype=np.float64,
    )


def _load_r10_total(
    scenario_dir,
    var_name,
    year,
    cumulative=CUMULATIVE,
    start_year=START_YEAR,
):
    return _aggregate_r26_to_r10_sum(
        _load_r26_total(
            scenario_dir=scenario_dir,
            var_name=var_name,
            year=year,
            cumulative=cumulative,
            start_year=start_year,
        )
    )


def _safe_intensity(total, mitigation):
    total = np.asarray(total, dtype=np.float64)
    mitigation = np.asarray(mitigation, dtype=np.float64)

    return np.divide(
        total,
        mitigation,
        out=np.full_like(total, np.nan, dtype=np.float64),
        where=np.isfinite(mitigation) & (mitigation > 0.0),
    )


def load_realised_cost_intensities(
    scenario_dir,
    year=YEAR,
    cumulative=CUMULATIVE,
    start_year=START_YEAR,
):
    """
    Return R10 realised cost intensities in $2025/tCO2-eq.

    Revenues are returned as negative costs.
    """
    scenario_dir = Path(scenario_dir)

    mitigation = _load_r10_total(
        scenario_dir,
        MITIGATION_VAR,
        year,
        cumulative=cumulative,
        start_year=start_year,
    )

    df = pd.DataFrame(
        {
            "Mitigation (tCO2-eq)": mitigation,
            "Mitigation (GtCO2-eq)": mitigation / 1e9,
        },
        index=list(R10.keys()),
    )

    for label, var_name in POSITIVE_COMPONENTS:
        total = _load_r10_total(
            scenario_dir,
            var_name,
            year,
            cumulative=cumulative,
            start_year=start_year,
        )
        df[f"{label} total ($2025)"] = total
        df[label] = _safe_intensity(total, mitigation)

    for label, var_name in NEGATIVE_COMPONENTS:
        total = _load_r10_total(
            scenario_dir,
            var_name,
            year,
            cumulative=cumulative,
            start_year=start_year,
        )
        df[f"{label} total ($2025)"] = total
        df[label] = -_safe_intensity(total, mitigation)

    net_total = _load_r10_total(
        scenario_dir,
        NET_VAR,
        year,
        cumulative=cumulative,
        start_year=start_year,
    )
    df["Net"] = _safe_intensity(net_total, mitigation)

    return df


def calculate_difference_dataframe(main_df, simple_df):
    """
    Difference shown in panels b and d:

        stylised - main

    Positive values therefore mean that the stylised-biochar representation is
    more costly / less favourable for that component.
    """
    out = simple_df[list(ALL_COMPONENT_LABELS)] - main_df[
        list(ALL_COMPONENT_LABELS)
    ]

    out["Net"] = simple_df["Net"] - main_df["Net"]

    return out


# =============================================================================
# Axes / limits
# =============================================================================

def _style_axis(ax):
    ax.grid(
        axis="x",
        linestyle="--",
        linewidth=0.7,
        alpha=GRID_ALPHA,
    )
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=N_X_TICKS))

    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(FRAME_COLOR)
        ax.spines[spine].set_linewidth(FRAME_WIDTH)

    ax.tick_params(width=FRAME_WIDTH)


def _absolute_xlim(dataframes):
    xmin = 0.0
    xmax = 0.0

    positive_labels = [label for label, _ in POSITIVE_COMPONENTS]
    negative_labels = [label for label, _ in NEGATIVE_COMPONENTS]

    for df in dataframes:
        positive_extent = df[positive_labels].sum(axis=1, min_count=1)
        negative_extent = df[negative_labels].sum(axis=1, min_count=1)

        xmin = min(
            xmin,
            float(np.nanmin(negative_extent.values)),
            float(np.nanmin(df["Net"].values)),
        )
        xmax = max(
            xmax,
            float(np.nanmax(positive_extent.values)),
            float(np.nanmax(df["Net"].values)),
        )

    span = xmax - xmin
    pad = 0.055 * span if span > 0 else 10.0

    return xmin - pad, xmax + pad


def _difference_xlim(dataframes):
    xmin = 0.0
    xmax = 0.0

    for df in dataframes:
        values = df[list(ALL_COMPONENT_LABELS)].to_numpy(dtype=np.float64)

        negative_extent = np.nansum(
            np.where(values < 0.0, values, 0.0),
            axis=1,
        )
        positive_extent = np.nansum(
            np.where(values > 0.0, values, 0.0),
            axis=1,
        )

        xmin = min(
            xmin,
            float(np.nanmin(negative_extent)),
            float(np.nanmin(df["Net"].values)),
        )
        xmax = max(
            xmax,
            float(np.nanmax(positive_extent)),
            float(np.nanmax(df["Net"].values)),
        )

    span = xmax - xmin
    pad = 0.07 * span if span > 0 else 5.0

    return xmin - pad, xmax + pad


# =============================================================================
# Plotting
# =============================================================================

def _plot_absolute(ax, df, show_ylabels=True):
    y = np.arange(len(df))

    negative_left = np.zeros(len(df), dtype=np.float64)

    for label, _ in NEGATIVE_COMPONENTS:
        values = df[label].to_numpy(dtype=np.float64)

        ax.barh(
            y,
            values,
            left=negative_left,
            height=BAR_HEIGHT,
            color=COLORS[label],
        )

        negative_left += values

    positive_left = np.zeros(len(df), dtype=np.float64)

    for label, _ in POSITIVE_COMPONENTS:
        values = df[label].to_numpy(dtype=np.float64)

        ax.barh(
            y,
            values,
            left=positive_left,
            height=BAR_HEIGHT,
            color=COLORS[label],
        )

        positive_left += values

    ax.scatter(
        df["Net"].to_numpy(dtype=np.float64),
        y,
        s=MARKER_SIZE,
        marker="D",
        color="black",
        zorder=5,
    )

    ax.axvline(0.0, color="black", linewidth=1.05, zorder=4)
    ax.set_yticks(y)

    if show_ylabels:
        ax.set_yticklabels([R10_DISPLAY.get(r, r) for r in df.index])
    else:
        ax.set_yticklabels([])

    ax.invert_yaxis()
    _style_axis(ax)


def _plot_difference(ax, df, show_ylabels=False):
    y = np.arange(len(df))
    positive_left = np.zeros(len(df), dtype=np.float64)
    negative_left = np.zeros(len(df), dtype=np.float64)

    for label in ALL_COMPONENT_LABELS:
        values = df[label].to_numpy(dtype=np.float64)

        positive = np.where(values > 0.0, values, 0.0)
        negative = np.where(values < 0.0, values, 0.0)

        ax.barh(
            y,
            positive,
            left=positive_left,
            height=BAR_HEIGHT,
            color=COLORS[label],
        )

        ax.barh(
            y,
            negative,
            left=negative_left,
            height=BAR_HEIGHT,
            color=COLORS[label],
        )

        positive_left += positive
        negative_left += negative

    ax.scatter(
        df["Net"].to_numpy(dtype=np.float64),
        y,
        s=MARKER_SIZE,
        marker="D",
        color="black",
        zorder=5,
    )

    ax.axvline(0.0, color="black", linewidth=1.05, zorder=4)
    ax.set_yticks(y)

    if show_ylabels:
        ax.set_yticklabels([R10_DISPLAY.get(r, r) for r in df.index])
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)

    ax.invert_yaxis()
    _style_axis(ax)


def _legend_handles():
    return [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="None",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6,
            label="Net",
        ),
        Patch(
            facecolor=COLORS["Feedstock cost"],
            label="Feedstock cost",
        ),
        Patch(
            facecolor=COLORS["Production cost"],
            label="Production cost",
        ),
        Patch(
            facecolor=COLORS["Application cost"],
            label="Application cost",
        ),
        Patch(
            facecolor=COLORS["Transport cost"],
            label="Transport cost",
        ),
        Patch(
            facecolor=COLORS["Crop yield revenue"],
            label="Crop yield revenue",
        ),
        Patch(
            facecolor=COLORS["Bio-oil revenue"],
            label="Bio-oil revenue",
        ),
    ]


# =============================================================================
# Diagnostics
# =============================================================================

def _closure_error_absolute(df):
    component_sum = df[list(ALL_COMPONENT_LABELS)].sum(
        axis=1,
        min_count=1,
    )
    return float(
        np.nanmax(
            np.abs(
                (df["Net"] - component_sum).to_numpy(dtype=np.float64)
            )
        )
    )


def _closure_error_difference(df):
    component_sum = df[list(ALL_COMPONENT_LABELS)].sum(
        axis=1,
        min_count=1,
    )
    return float(
        np.nanmax(
            np.abs(
                (df["Net"] - component_sum).to_numpy(dtype=np.float64)
            )
        )
    )



# =============================================================================
# Sanity checks
# =============================================================================

def _finite_max_abs(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan
    return float(np.nanmax(np.abs(finite)))


def _scenario_output_sanity(name, df):
    """
    Basic output-level checks for one scenario.

    These checks validate the realised-output accounting used by this plotting
    module. They do not by themselves prove that the internal biophysical
    yield-response field was spatially uniform; that requires a dedicated
    exported yield-effect field from the biochar model.
    """
    issues = []

    mitigation = df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)

    if not np.all(np.isfinite(mitigation)):
        issues.append("non-finite realised mitigation values")

    if np.any(mitigation <= 0.0):
        bad = list(df.index[mitigation <= 0.0])
        issues.append(
            "non-positive realised mitigation in R10 regions: "
            + ", ".join(map(str, bad))
        )

    for label in (*ALL_COMPONENT_LABELS, "Net"):
        values = df[label].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(values)):
            issues.append(f"non-finite values in '{label}'")

    closure = _closure_error_absolute(df)
    if not np.isfinite(closure) or closure > CLOSURE_TOLERANCE:
        issues.append(
            f"component accounting closure error = {closure:.6g} $/tCO2-eq"
        )

    return {
        "scenario": name,
        "closure_error_usd_per_t": closure,
        "min_mitigation_gt": float(
            np.nanmin(df["Mitigation (GtCO2-eq)"].to_numpy(dtype=np.float64))
        ),
        "global_mitigation_gt": float(
            np.nansum(df["Mitigation (GtCO2-eq)"].to_numpy(dtype=np.float64))
        ),
        "issues": issues,
    }


def _yield_revenue_sanity_table(
    agriculture,
    agriculture_simple,
):
    """
    Diagnose the large crop-yield-revenue differences in panel d.

    The plotted cost component is:
        crop-yield cost intensity = - realised yield revenue / realised mitigation

    Therefore:
        panel-d crop-yield component
        = simple intensity - main intensity

    The exact difference is decomposed into:
        numerator effect
            = -(Revenue_simple - Revenue_main) / Mitigation_main

        denominator effect
            = -Revenue_simple * (1/Mitigation_simple - 1/Mitigation_main)

    Their sum exactly reproduces the plotted simple-minus-main intensity
    difference (apart from floating-point error).
    """
    regions = agriculture.index

    m_main = agriculture["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)
    m_simple = agriculture_simple["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)

    r_main = agriculture[
        "Crop yield revenue total ($2025)"
    ].to_numpy(dtype=np.float64)

    r_simple = agriculture_simple[
        "Crop yield revenue total ($2025)"
    ].to_numpy(dtype=np.float64)

    i_main = agriculture[
        "Crop yield revenue"
    ].to_numpy(dtype=np.float64)

    i_simple = agriculture_simple[
        "Crop yield revenue"
    ].to_numpy(dtype=np.float64)

    plotted_difference = i_simple - i_main

    numerator_effect = np.divide(
        -(r_simple - r_main),
        m_main,
        out=np.full_like(r_main, np.nan, dtype=np.float64),
        where=np.isfinite(m_main) & (m_main > 0.0),
    )

    denominator_effect = (
        -r_simple
        * (
            np.divide(
                1.0,
                m_simple,
                out=np.full_like(m_simple, np.nan, dtype=np.float64),
                where=np.isfinite(m_simple) & (m_simple > 0.0),
            )
            - np.divide(
                1.0,
                m_main,
                out=np.full_like(m_main, np.nan, dtype=np.float64),
                where=np.isfinite(m_main) & (m_main > 0.0),
            )
        )
    )

    reconstructed = numerator_effect + denominator_effect
    decomposition_error = reconstructed - plotted_difference

    table = pd.DataFrame(
        {
            "Main mitigation (GtCO2-eq)": m_main / 1e9,
            "Stylised mitigation (GtCO2-eq)": m_simple / 1e9,
            "Stylised / main mitigation": np.divide(
                m_simple,
                m_main,
                out=np.full_like(m_main, np.nan, dtype=np.float64),
                where=np.isfinite(m_main) & (m_main > 0.0),
            ),
            "Main realised crop yield revenue (bn $2025)": r_main / 1e9,
            "Stylised realised crop yield revenue (bn $2025)": r_simple / 1e9,
            "Stylised / main crop yield revenue": np.divide(
                r_simple,
                r_main,
                out=np.full_like(r_main, np.nan, dtype=np.float64),
                where=np.isfinite(r_main) & (np.abs(r_main) > 0.0),
            ),
            "Main crop yield cost intensity ($/tCO2-eq)": i_main,
            "Stylised crop yield cost intensity ($/tCO2-eq)": i_simple,
            "Panel d crop yield difference ($/tCO2-eq)": plotted_difference,
            "Yield numerator effect ($/tCO2-eq)": numerator_effect,
            "Yield mitigation-denominator effect ($/tCO2-eq)": denominator_effect,
            "Yield decomposition error ($/tCO2-eq)": decomposition_error,
        },
        index=regions,
    )

    return table


def run_sanity_checks(
    climate,
    climate_simple,
    agriculture,
    agriculture_simple,
    climate_diff,
    agriculture_diff,
):
    """
    Run and print sanity checks that can be verified from the realised outputs.

    Checks
    ------
    1. Positive/finite mitigation denominators and finite cost components.
    2. Absolute accounting closure for all four scenarios.
    3. Difference accounting closure for panels b and d.
    4. Climate stylised scenario has ~zero crop-yield revenue, as intended.
    5. Agriculture stylised scenario has non-zero crop-yield revenue.
    6. Raw crop-yield-revenue totals and mitigation denominators are reported
       for main vs stylised agriculture.
    7. The crop-yield component difference is exactly decomposed into a
       revenue-numerator effect and mitigation-denominator effect.

    Important
    ---------
    These are output-level checks. They cannot demonstrate that
    simplify_biochar_LT used a spatially uniform *biophysical yield-response
    coefficient*, because this plotting module does not load a dedicated
    exported yield-effect field. `yield_revenues_ha` is not suitable for that
    test because monetary yield value can remain spatially heterogeneous even
    when the percentage yield response is uniform.
    """
    scenario_checks = [
        _scenario_output_sanity("biochar_c400", climate),
        _scenario_output_sanity("biochar_c400_simple", climate_simple),
        _scenario_output_sanity("biochar_c400_LT", agriculture),
        _scenario_output_sanity("biochar_c400_LT_simple", agriculture_simple),
    ]

    difference_checks = {
        "panel_b_difference_closure": _closure_error_difference(climate_diff),
        "panel_d_difference_closure": _closure_error_difference(agriculture_diff),
    }

    climate_simple_yield_max = _finite_max_abs(
        climate_simple["Crop yield revenue"].to_numpy(dtype=np.float64)
    )

    agriculture_simple_yield_total = float(
        np.nansum(
            agriculture_simple[
                "Crop yield revenue total ($2025)"
            ].to_numpy(dtype=np.float64)
        )
    )

    yield_table = _yield_revenue_sanity_table(
        agriculture=agriculture,
        agriculture_simple=agriculture_simple,
    )

    yield_decomp_max_error = _finite_max_abs(
        yield_table[
            "Yield decomposition error ($/tCO2-eq)"
        ].to_numpy(dtype=np.float64)
    )

    print("\n" + "=" * 118)
    print("REALISED-COST SANITY CHECKS")
    print("=" * 118)

    for item in scenario_checks:
        status = "PASS" if not item["issues"] else "CHECK"
        print(
            f"{status:5s}  {item['scenario']:<27s} "
            f"global mitigation={item['global_mitigation_gt']:.3f} GtCO2-eq, "
            f"closure={item['closure_error_usd_per_t']:.6g} $/tCO2-eq"
        )
        for issue in item["issues"]:
            print(f"       - {issue}")

    for name, error in difference_checks.items():
        status = (
            "PASS"
            if np.isfinite(error) and error <= CLOSURE_TOLERANCE
            else "CHECK"
        )
        print(
            f"{status:5s}  {name:<27s} "
            f"closure={error:.6g} $/tCO2-eq"
        )

    print("\nCrop-yield-revenue checks")
    print("-" * 118)

    climate_yield_status = (
        "PASS"
        if np.isfinite(climate_simple_yield_max)
        and climate_simple_yield_max
        <= CLIMATE_SIMPLE_YIELD_INTENSITY_TOLERANCE
        else "CHECK"
    )
    print(
        f"{climate_yield_status:5s}  biochar_c400_simple maximum absolute "
        f"crop-yield cost intensity = {climate_simple_yield_max:.6g} $/tCO2-eq "
        "(expected ~0)"
    )

    agriculture_yield_status = (
        "PASS"
        if np.isfinite(agriculture_simple_yield_total)
        and agriculture_simple_yield_total > 0.0
        else "CHECK"
    )
    print(
        f"{agriculture_yield_status:5s}  biochar_c400_LT_simple total realised "
        f"crop-yield revenue = {agriculture_simple_yield_total / 1e9:.3f} "
        "bn $2025 (expected >0)"
    )

    decomp_status = (
        "PASS"
        if np.isfinite(yield_decomp_max_error)
        and yield_decomp_max_error <= YIELD_DECOMPOSITION_TOLERANCE
        else "CHECK"
    )
    print(
        f"{decomp_status:5s}  maximum crop-yield difference decomposition "
        f"error = {yield_decomp_max_error:.6g} $/tCO2-eq"
    )

    print("\nMain vs stylised agriculture crop-yield diagnostics by R10:")
    print(
        yield_table[
            [
                "Main mitigation (GtCO2-eq)",
                "Stylised mitigation (GtCO2-eq)",
                "Main realised crop yield revenue (bn $2025)",
                "Stylised realised crop yield revenue (bn $2025)",
                "Main crop yield cost intensity ($/tCO2-eq)",
                "Stylised crop yield cost intensity ($/tCO2-eq)",
                "Panel d crop yield difference ($/tCO2-eq)",
                "Yield numerator effect ($/tCO2-eq)",
                "Yield mitigation-denominator effect ($/tCO2-eq)",
            ]
        ].round(3).to_string()
    )

    print(
        "\nInterpretation reminder: a positive crop-yield component in panel d "
        "means the stylised LT run has a less-negative crop-yield cost "
        "component (i.e. lower realised yield revenue per tCO2-eq) than the "
        "main LT run."
    )

    print(
        "\nLimitation: these checks verify realised-output accounting and the "
        "source of the panel-d yield-intensity difference. They do not prove "
        "that the internal simplify_biochar_LT biophysical yield-response "
        "coefficient is spatially uniform; that requires a dedicated exported "
        "yield-effect field."
    )

    return {
        "scenario_checks": scenario_checks,
        "difference_checks": difference_checks,
        "climate_simple_yield_max_abs_intensity":
            climate_simple_yield_max,
        "agriculture_simple_total_yield_revenue_usd":
            agriculture_simple_yield_total,
        "yield_decomposition_max_error":
            yield_decomp_max_error,
        "yield_table": yield_table,
    }



# =============================================================================
# Quantitative Results overview
# =============================================================================

def _infer_timestep_width_years(scenario_dir, year=YEAR):
    """Infer the TIMER timestep width represented by the selected year."""
    nc_file = Path(scenario_dir) / f"{MITIGATION_VAR}.nc"

    with xr.open_dataset(nc_file) as ds:
        da = ds[MITIGATION_VAR]

        if "time" not in da.dims:
            return np.nan

        years = np.asarray(da["time"].values, dtype=int)
        matches = np.where(years == int(year))[0]

        if len(matches) == 0:
            return np.nan

        idx = int(matches[0])

        if idx > 0:
            dt = float(years[idx] - years[idx - 1])
        elif len(years) > 1:
            dt = float(years[1] - years[0])
        else:
            return np.nan

    return dt if np.isfinite(dt) and dt > 0.0 else np.nan


def _mitigation_weighted_intensity(df, column):
    """Global mitigation-weighted realised intensity for one component."""
    values = df[column].to_numpy(dtype=np.float64)
    weights = df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)

    if not np.any(valid):
        return np.nan

    return float(
        np.sum(values[valid] * weights[valid])
        / np.sum(weights[valid])
    )


def _global_scenario_summary(name, scenario_dir, df, year=YEAR):
    positive = {
        label: _mitigation_weighted_intensity(df, label)
        for label, _ in POSITIVE_COMPONENTS
    }
    negative = {
        label: _mitigation_weighted_intensity(df, label)
        for label, _ in NEGATIVE_COMPONENTS
    }

    gross_cost = float(np.nansum(list(positive.values())))
    revenue_total = float(np.nansum(list(negative.values())))
    revenue_offset = -revenue_total
    net = _mitigation_weighted_intensity(df, "Net")

    timestep_mitigation = float(
        np.nansum(df["Mitigation (GtCO2-eq)"].to_numpy(dtype=np.float64))
    )

    dt = _infer_timestep_width_years(
        scenario_dir=scenario_dir,
        year=year,
    )

    annual_mitigation = (
        timestep_mitigation / dt
        if np.isfinite(dt) and dt > 0.0
        else np.nan
    )

    return {
        "Scenario": name,
        "Timestep mitigation (GtCO2-eq)": timestep_mitigation,
        "Timestep width (yr)": dt,
        "Annualised mitigation (GtCO2-eq/yr)": annual_mitigation,
        **positive,
        **negative,
        "Gross positive cost": gross_cost,
        "Revenue offset magnitude": revenue_offset,
        "Revenue offset of gross cost (%)": (
            100.0 * revenue_offset / gross_cost
            if gross_cost > 0.0 else np.nan
        ),
        "Net": net,
    }


def _global_summary_table(
    base_dir,
    climate,
    climate_simple,
    agriculture,
    agriculture_simple,
    year=YEAR,
):
    rows = [
        _global_scenario_summary(
            "Climate main",
            Path(base_dir) / CLIMATE_SCENARIO,
            climate,
            year=year,
        ),
        _global_scenario_summary(
            "Climate stylised",
            Path(base_dir) / CLIMATE_SIMPLE_SCENARIO,
            climate_simple,
            year=year,
        ),
        _global_scenario_summary(
            "Agriculture main",
            Path(base_dir) / AGRICULTURE_SCENARIO,
            agriculture,
            year=year,
        ),
        _global_scenario_summary(
            "Agriculture stylised",
            Path(base_dir) / AGRICULTURE_SIMPLE_SCENARIO,
            agriculture_simple,
            year=year,
        ),
    ]

    return pd.DataFrame(rows).set_index("Scenario")


def _regional_revenue_offset_table(df):
    """Region-level gross costs, revenue offsets and net realised costs."""
    gross = df[
        [label for label, _ in POSITIVE_COMPONENTS]
    ].sum(axis=1, min_count=1)

    revenue = -df[
        [label for label, _ in NEGATIVE_COMPONENTS]
    ].sum(axis=1, min_count=1)

    offset_pct = np.divide(
        100.0 * revenue.to_numpy(dtype=np.float64),
        gross.to_numpy(dtype=np.float64),
        out=np.full(len(df), np.nan, dtype=np.float64),
        where=gross.to_numpy(dtype=np.float64) > 0.0,
    )

    return pd.DataFrame(
        {
            "Gross positive cost ($/tCO2-eq)": gross,
            "Revenue offset ($/tCO2-eq)": revenue,
            "Revenue offset of gross cost (%)": offset_pct,
            "Net cost ($/tCO2-eq)": df["Net"],
            "Crop yield revenue ($/tCO2-eq)": df["Crop yield revenue"],
            "Bio-oil revenue ($/tCO2-eq)": df["Bio-oil revenue"],
            "Mitigation (GtCO2-eq timestep total)": df[
                "Mitigation (GtCO2-eq)"
            ],
        },
        index=df.index,
    )


def _format_region_values(series, n=3, ascending=False, signed=False):
    ordered = series.sort_values(ascending=ascending).head(n)

    parts = []
    for region, value in ordered.items():
        name = R10_DISPLAY.get(region, region)
        fmt = f"{value:+.1f}" if signed else f"{value:.1f}"
        parts.append(f"{name} ({fmt})")

    return ", ".join(parts)


def _print_global_component_breakdown(global_summary):
    print("\nGLOBAL MITIGATION-WEIGHTED COST STRUCTURE")
    print("-" * 118)

    cols = [
        "Annualised mitigation (GtCO2-eq/yr)",
        "Feedstock cost",
        "Production cost",
        "Application cost",
        "Transport cost",
        "Crop yield revenue",
        "Bio-oil revenue",
        "Gross positive cost",
        "Revenue offset magnitude",
        "Revenue offset of gross cost (%)",
        "Net",
    ]

    print(global_summary[cols].round(2).to_string())
    print(
        "\nRevenues are negative cost components in the figure; "
        "'Revenue offset magnitude' reports their positive absolute magnitude."
    )


def _print_main_scenario_comparison(global_summary):
    print("\nMAIN CLIMATE VS AGRICULTURE SCENARIOS")
    print("-" * 118)

    climate = global_summary.loc["Climate main"]
    agriculture = global_summary.loc["Agriculture main"]

    print(
        f"Global net realised cost intensity: climate = {climate['Net']:.1f}, "
        f"agriculture = {agriculture['Net']:.1f} $2025/tCO2-eq "
        f"(agriculture - climate = {agriculture['Net'] - climate['Net']:+.1f})."
    )
    print(
        f"Gross positive cost intensity: climate = "
        f"{climate['Gross positive cost']:.1f}, agriculture = "
        f"{agriculture['Gross positive cost']:.1f} $2025/tCO2-eq."
    )
    print(
        f"Revenues offset {climate['Revenue offset of gross cost (%)']:.1f}% "
        f"of gross positive costs in the climate-oriented case and "
        f"{agriculture['Revenue offset of gross cost (%)']:.1f}% in the "
        f"agriculture-oriented case."
    )
    print(
        f"Annualised realised mitigation: climate = "
        f"{climate['Annualised mitigation (GtCO2-eq/yr)']:.2f}, agriculture = "
        f"{agriculture['Annualised mitigation (GtCO2-eq/yr)']:.2f} "
        f"GtCO2-eq/yr."
    )

    component_deltas = pd.Series(
        {
            label: agriculture[label] - climate[label]
            for label in ALL_COMPONENT_LABELS
        }
    )

    print("\nAgriculture minus climate, global cost-component intensity:")
    print(
        component_deltas
        .sort_values(key=lambda s: np.abs(s), ascending=False)
        .round(1)
        .to_string()
    )


def _print_regional_main_highlights(climate, agriculture):
    print("\nREGIONAL MAIN-SCENARIO HIGHLIGHTS")
    print("-" * 118)

    agriculture_offsets = _regional_revenue_offset_table(agriculture)

    print("Climate-oriented deployment:")
    print(
        "  Lowest net realised costs: "
        + _format_region_values(climate["Net"], n=3, ascending=True)
    )
    print(
        "  Highest net realised costs: "
        + _format_region_values(climate["Net"], n=3, ascending=False)
    )
    print(
        "  Largest bio-oil revenue intensities: "
        + _format_region_values(
            climate["Bio-oil revenue"], n=3, ascending=True
        )
    )

    print("\nAgriculture-oriented deployment:")
    print(
        "  Lowest net realised costs: "
        + _format_region_values(agriculture["Net"], n=3, ascending=True)
    )
    print(
        "  Highest net realised costs: "
        + _format_region_values(agriculture["Net"], n=3, ascending=False)
    )
    print(
        "  Largest crop-yield revenue intensities: "
        + _format_region_values(
            agriculture["Crop yield revenue"], n=4, ascending=True
        )
    )

    negative_net = agriculture["Net"][agriculture["Net"] < 0.0]
    if len(negative_net):
        print(
            "  Regions with negative net realised cost: "
            + ", ".join(
                f"{R10_DISPLAY.get(region, region)} ({value:.1f})"
                for region, value in negative_net.sort_values().items()
            )
        )
    else:
        print("  No R10 region has a negative net realised cost.")

    print(
        "  Largest revenue offsets relative to gross costs: "
        + _format_region_values(
            agriculture_offsets["Revenue offset of gross cost (%)"],
            n=4,
            ascending=False,
        )
    )

    positive_labels = [label for label, _ in POSITIVE_COMPONENTS]
    climate_dominant = climate[positive_labels].idxmax(axis=1)
    agriculture_dominant = agriculture[positive_labels].idxmax(axis=1)

    def counts_text(series):
        counts = series.value_counts()
        return ", ".join(
            f"{component}: {count}/{len(series)} regions"
            for component, count in counts.items()
        )

    print(
        "\nDominant positive cost component by R10 region:\n"
        f"  Climate: {counts_text(climate_dominant)}\n"
        f"  Agriculture: {counts_text(agriculture_dominant)}"
    )


def _global_stylised_minus_main_component_delta(
    global_summary,
    main_name,
    stylised_name,
):
    main = global_summary.loc[main_name]
    stylised = global_summary.loc[stylised_name]

    return pd.Series(
        {
            **{
                label: stylised[label] - main[label]
                for label in ALL_COMPONENT_LABELS
            },
            "Net": stylised["Net"] - main["Net"],
        }
    )


def _print_stylised_sensitivity(
    global_summary,
    climate_diff,
    agriculture_diff,
):
    print("\nSTYLISED-BIOCHAR SENSITIVITY")
    print("-" * 118)

    climate_global_diff = _global_stylised_minus_main_component_delta(
        global_summary,
        "Climate main",
        "Climate stylised",
    )
    agriculture_global_diff = _global_stylised_minus_main_component_delta(
        global_summary,
        "Agriculture main",
        "Agriculture stylised",
    )

    print("Global stylised minus main cost-component intensity ($2025/tCO2-eq):")
    print(
        pd.DataFrame(
            {
                "Climate": climate_global_diff,
                "Agriculture": agriculture_global_diff,
            }
        ).round(1).to_string()
    )

    for name, diff in (
        ("Climate", climate_diff),
        ("Agriculture", agriculture_diff),
    ):
        net = diff["Net"]
        n_higher = int(np.sum(net > 0.0))
        n_lower = int(np.sum(net < 0.0))

        print(f"\n{name} stylised-minus-main regional net-cost differences:")
        print(
            f"  Stylisation raises net cost in {n_higher}/10 R10 regions and "
            f"lowers it in {n_lower}/10."
        )
        print(
            "  Largest increases: "
            + _format_region_values(
                net, n=3, ascending=False, signed=True
            )
        )
        print(
            "  Largest decreases: "
            + _format_region_values(
                net, n=3, ascending=True, signed=True
            )
        )

        drivers = diff[list(ALL_COMPONENT_LABELS)].abs().idxmax(axis=1)
        driver_counts = drivers.value_counts()
        print(
            "  Most frequent largest-magnitude regional difference component: "
            + ", ".join(
                f"{component} ({count}/10)"
                for component, count in driver_counts.items()
            )
        )


def _print_agriculture_yield_hotspots(agriculture, agriculture_simple):
    print("\nAGRICULTURAL-REVENUE HOTSPOTS")
    print("-" * 118)

    for label, df in (
        ("Main agriculture", agriculture),
        ("Stylised agriculture", agriculture_simple),
    ):
        yield_total_bn = df["Crop yield revenue total ($2025)"] / 1e9
        global_total = float(np.nansum(yield_total_bn.values))
        ordered = yield_total_bn.sort_values(ascending=False)
        top3_total = float(ordered.head(3).sum())
        top3_share = (
            100.0 * top3_total / global_total
            if global_total > 0.0 else np.nan
        )

        print(
            f"{label}: total realised crop-yield revenue = "
            f"{global_total:.1f} bn $2025; top 3 R10 regions account for "
            f"{top3_share:.1f}%."
        )
        print(
            "  Largest absolute yield revenues: "
            + ", ".join(
                f"{R10_DISPLAY.get(region, region)} ({value:.1f} bn)"
                for region, value in ordered.head(4).items()
            )
        )



def _component_sign(label):
    """
    Sign used in the plotted intensity.

    Positive costs:  +total / mitigation
    Revenues:        -total / mitigation
    """
    positive_labels = {name for name, _ in POSITIVE_COMPONENTS}
    negative_labels = {name for name, _ in NEGATIVE_COMPONENTS}

    if label in positive_labels:
        return 1.0
    if label in negative_labels:
        return -1.0

    raise KeyError(f"Unknown component label: {label}")


def _component_difference_decomposition(main_df, simple_df, label):
    """
    Exact R10 decomposition of stylised-minus-main component intensity.

    Let plotted intensity I = s*T/M, where:
        T = realised monetary numerator,
        M = realised mitigation,
        s = +1 for costs and -1 for revenues.

    Then:
        I_simple - I_main
        = s*(T_simple - T_main)/M_main
          + s*T_simple*(1/M_simple - 1/M_main)

    The first term is the monetary-numerator effect.
    The second is the mitigation-denominator effect.

    This decomposition is useful because a lower $/tCO2 component intensity
    does NOT necessarily imply a lower unit cost of feedstock, production, etc.
    """
    total_col = f"{label} total ($2025)"
    sign = _component_sign(label)

    m_main = main_df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)
    m_simple = simple_df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)

    t_main = main_df[total_col].to_numpy(dtype=np.float64)
    t_simple = simple_df[total_col].to_numpy(dtype=np.float64)

    i_main = main_df[label].to_numpy(dtype=np.float64)
    i_simple = simple_df[label].to_numpy(dtype=np.float64)

    numerator_effect = np.divide(
        sign * (t_simple - t_main),
        m_main,
        out=np.full_like(t_main, np.nan, dtype=np.float64),
        where=np.isfinite(m_main) & (m_main > 0.0),
    )

    denominator_effect = sign * t_simple * (
        np.divide(
            1.0,
            m_simple,
            out=np.full_like(m_simple, np.nan, dtype=np.float64),
            where=np.isfinite(m_simple) & (m_simple > 0.0),
        )
        - np.divide(
            1.0,
            m_main,
            out=np.full_like(m_main, np.nan, dtype=np.float64),
            where=np.isfinite(m_main) & (m_main > 0.0),
        )
    )

    intensity_difference = i_simple - i_main

    total_change_pct = np.divide(
        100.0 * (t_simple - t_main),
        np.abs(t_main),
        out=np.full_like(t_main, np.nan, dtype=np.float64),
        where=np.isfinite(t_main) & (np.abs(t_main) > 0.0),
    )

    mitigation_change_pct = np.divide(
        100.0 * (m_simple - m_main),
        m_main,
        out=np.full_like(m_main, np.nan, dtype=np.float64),
        where=np.isfinite(m_main) & (m_main > 0.0),
    )

    relative_intensity_shift_pct = np.divide(
        100.0 * intensity_difference,
        np.abs(i_main),
        out=np.full_like(i_main, np.nan, dtype=np.float64),
        where=np.isfinite(i_main) & (np.abs(i_main) > 0.0),
    )

    return pd.DataFrame(
        {
            "Main mitigation (GtCO2-eq)": m_main / 1e9,
            "Stylised mitigation (GtCO2-eq)": m_simple / 1e9,
            "Mitigation change (%)": mitigation_change_pct,
            "Main monetary total (bn $2025)": t_main / 1e9,
            "Stylised monetary total (bn $2025)": t_simple / 1e9,
            "Monetary-total change (%)": total_change_pct,
            "Main intensity ($/tCO2-eq)": i_main,
            "Stylised intensity ($/tCO2-eq)": i_simple,
            "Intensity difference ($/tCO2-eq)": intensity_difference,
            "Relative intensity shift (% of |main|)": relative_intensity_shift_pct,
            "Numerator effect ($/tCO2-eq)": numerator_effect,
            "Mitigation-denominator effect ($/tCO2-eq)": denominator_effect,
            "Decomposition error ($/tCO2-eq)":
                numerator_effect + denominator_effect - intensity_difference,
        },
        index=main_df.index,
    )


def _global_component_difference_decomposition(main_df, simple_df, label):
    """
    Global version of _component_difference_decomposition using R10 totals.
    """
    total_col = f"{label} total ($2025)"
    sign = _component_sign(label)

    m_main = float(np.nansum(
        main_df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)
    ))
    m_simple = float(np.nansum(
        simple_df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)
    ))

    t_main = float(np.nansum(
        main_df[total_col].to_numpy(dtype=np.float64)
    ))
    t_simple = float(np.nansum(
        simple_df[total_col].to_numpy(dtype=np.float64)
    ))

    i_main = sign * t_main / m_main
    i_simple = sign * t_simple / m_simple
    difference = i_simple - i_main

    numerator_effect = sign * (t_simple - t_main) / m_main
    denominator_effect = sign * t_simple * (
        1.0 / m_simple - 1.0 / m_main
    )

    return {
        "Component": label,
        "Main mitigation (GtCO2-eq)": m_main / 1e9,
        "Stylised mitigation (GtCO2-eq)": m_simple / 1e9,
        "Mitigation change (%)": 100.0 * (m_simple - m_main) / m_main,
        "Main monetary total (bn $2025)": t_main / 1e9,
        "Stylised monetary total (bn $2025)": t_simple / 1e9,
        "Monetary-total change (%)": (
            100.0 * (t_simple - t_main) / abs(t_main)
            if abs(t_main) > 0.0 else np.nan
        ),
        "Main intensity ($/tCO2-eq)": i_main,
        "Stylised intensity ($/tCO2-eq)": i_simple,
        "Intensity difference ($/tCO2-eq)": difference,
        "Relative intensity shift (% of |main|)": (
            100.0 * difference / abs(i_main)
            if abs(i_main) > 0.0 else np.nan
        ),
        "Numerator effect ($/tCO2-eq)": numerator_effect,
        "Mitigation-denominator effect ($/tCO2-eq)": denominator_effect,
        "Decomposition error ($/tCO2-eq)":
            numerator_effect + denominator_effect - difference,
    }


def _global_all_component_decomposition(main_df, simple_df):
    rows = [
        _global_component_difference_decomposition(
            main_df,
            simple_df,
            label,
        )
        for label in ALL_COMPONENT_LABELS
    ]

    return pd.DataFrame(rows).set_index("Component")


def _regional_net_change_table(main_df, simple_df):
    """
    Regional relative net-cost changes.

    Relative shift is defined as:
        (Net_simple - Net_main) / |Net_main| * 100

    Using |Net_main| in the denominator keeps the sign interpretable even when
    the main scenario has negative net realised costs:
        positive = stylisation worsens economics (higher net cost)
        negative = stylisation improves economics (lower net cost)
    """
    main_net = main_df["Net"].to_numpy(dtype=np.float64)
    simple_net = simple_df["Net"].to_numpy(dtype=np.float64)
    delta = simple_net - main_net

    relative_shift = np.divide(
        100.0 * delta,
        np.abs(main_net),
        out=np.full_like(delta, np.nan, dtype=np.float64),
        where=np.isfinite(main_net) & (np.abs(main_net) > 0.0),
    )

    m_main = main_df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)
    m_simple = simple_df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)

    mitigation_change = np.divide(
        100.0 * (m_simple - m_main),
        m_main,
        out=np.full_like(m_main, np.nan, dtype=np.float64),
        where=np.isfinite(m_main) & (m_main > 0.0),
    )

    sign_change = np.sign(main_net) != np.sign(simple_net)

    return pd.DataFrame(
        {
            "Main net cost ($/tCO2-eq)": main_net,
            "Stylised net cost ($/tCO2-eq)": simple_net,
            "Net-cost change ($/tCO2-eq)": delta,
            "Relative net-cost shift (% of |main|)": relative_shift,
            "Main mitigation (GtCO2-eq)": m_main / 1e9,
            "Stylised mitigation (GtCO2-eq)": m_simple / 1e9,
            "Mitigation change (%)": mitigation_change,
            "Net-cost sign changes": sign_change,
        },
        index=main_df.index,
    )


def _print_paragraph3_sensitivity_diagnostics(
    climate,
    climate_simple,
    agriculture,
    agriculture_simple,
):
    """
    Print the relative quantitative diagnostics needed for Results paragraph 3.
    """
    print("\n" + "=" * 118)
    print("PARAGRAPH 3: HETEROGENEITY / STYLISED-BIOCHAR DIAGNOSTICS")
    print("=" * 118)

    scenario_pairs = (
        ("CLIMATE-ORIENTED", climate, climate_simple),
        ("AGRICULTURE-ORIENTED", agriculture, agriculture_simple),
    )

    returned = {}

    for scenario_name, main_df, simple_df in scenario_pairs:
        print(f"\n{scenario_name}")
        print("-" * 118)

        component_decomp = _global_all_component_decomposition(
            main_df,
            simple_df,
        )

        regional_net = _regional_net_change_table(
            main_df,
            simple_df,
        )

        m_main = float(np.nansum(
            main_df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)
        ))
        m_simple = float(np.nansum(
            simple_df["Mitigation (tCO2-eq)"].to_numpy(dtype=np.float64)
        ))

        global_main_net = _mitigation_weighted_intensity(main_df, "Net")
        global_simple_net = _mitigation_weighted_intensity(simple_df, "Net")
        global_net_change = global_simple_net - global_main_net

        print(
            f"Global realised mitigation: main={m_main / 1e9:.3f}, "
            f"stylised={m_simple / 1e9:.3f} GtCO2-eq per timestep; "
            f"change={100.0 * (m_simple - m_main) / m_main:+.1f}%."
        )
        print(
            f"Global net realised cost: main={global_main_net:.1f}, "
            f"stylised={global_simple_net:.1f}, "
            f"stylised-main={global_net_change:+.1f} $2025/tCO2-eq "
            f"({100.0 * global_net_change / abs(global_main_net):+.1f}% "
            f"of |main net cost|)."
        )

        print("\nGlobal component changes and exact numerator/denominator decomposition:")
        display_cols = [
            "Main intensity ($/tCO2-eq)",
            "Stylised intensity ($/tCO2-eq)",
            "Intensity difference ($/tCO2-eq)",
            "Relative intensity shift (% of |main|)",
            "Monetary-total change (%)",
            "Mitigation change (%)",
            "Numerator effect ($/tCO2-eq)",
            "Mitigation-denominator effect ($/tCO2-eq)",
        ]
        print(
            component_decomp[display_cols]
            .round(2)
            .to_string()
        )

        # Explicitly answer the feedstock / production mechanism questions.
        for component in ("Feedstock cost", "Production cost"):
            row = component_decomp.loc[component]
            print(
                f"\n{component}: intensity changes by "
                f"{row['Intensity difference ($/tCO2-eq)']:+.1f} $/tCO2-eq "
                f"({row['Relative intensity shift (% of |main|)']:+.1f}%). "
                f"The monetary numerator changes by "
                f"{row['Monetary-total change (%)']:+.1f}%, while mitigation "
                f"changes by {row['Mitigation change (%)']:+.1f}%. "
                f"Exact decomposition: numerator effect "
                f"{row['Numerator effect ($/tCO2-eq)']:+.1f}, denominator "
                f"effect {row['Mitigation-denominator effect ($/tCO2-eq)']:+.1f} "
                f"$/tCO2-eq."
            )

        print(
            "\nNOTE: these diagnostics explain changes in realised cost per unit "
            "mitigation. They do not measure feedstock price per tonne residue "
            "or production cost per tonne biochar; those would require physical "
            "feedstock/char throughput denominators."
        )

        # Regional ranking: absolute and relative changes.
        print("\nRegional net-cost changes, ranked by absolute $/tCO2-eq change:")
        abs_ranked = regional_net.reindex(
            regional_net["Net-cost change ($/tCO2-eq)"]
            .abs()
            .sort_values(ascending=False)
            .index
        )
        print(
            abs_ranked[
                [
                    "Main net cost ($/tCO2-eq)",
                    "Stylised net cost ($/tCO2-eq)",
                    "Net-cost change ($/tCO2-eq)",
                    "Relative net-cost shift (% of |main|)",
                    "Mitigation change (%)",
                    "Net-cost sign changes",
                ]
            ]
            .round(2)
            .to_string()
        )

        print("\nStrongest relative regional net-cost shifts:")
        rel_ranked = regional_net.reindex(
            regional_net["Relative net-cost shift (% of |main|)"]
            .abs()
            .sort_values(ascending=False)
            .index
        )
        for region, row in rel_ranked.head(5).iterrows():
            direction = (
                "worsens"
                if row["Net-cost change ($/tCO2-eq)"] > 0.0
                else "improves"
            )
            print(
                f"  {R10_DISPLAY.get(region, region):<20s}: "
                f"{direction} by "
                f"{row['Net-cost change ($/tCO2-eq)']:+.1f} $/tCO2-eq; "
                f"relative shift="
                f"{row['Relative net-cost shift (% of |main|)']:+.1f}%; "
                f"mitigation change={row['Mitigation change (%)']:+.1f}%."
            )

        # Largest component driver per region, preserving sign of difference.
        component_diff = (
            simple_df[list(ALL_COMPONENT_LABELS)]
            - main_df[list(ALL_COMPONENT_LABELS)]
        )
        driver = component_diff.abs().idxmax(axis=1)

        print("\nLargest-magnitude component driving each regional stylised-minus-main difference:")
        for region in main_df.index:
            component = driver.loc[region]
            value = component_diff.loc[region, component]
            print(
                f"  {R10_DISPLAY.get(region, region):<20s}: "
                f"{component} ({value:+.1f} $/tCO2-eq)"
            )

        returned[scenario_name.lower()] = {
            "global_component_decomposition": component_decomp,
            "regional_net_changes": regional_net,
            "regional_component_differences": component_diff,
        }

    return returned


def print_quantitative_results_overview(
    base_dir,
    climate,
    climate_simple,
    agriculture,
    agriculture_simple,
    climate_diff,
    agriculture_diff,
    year=YEAR,
):
    """Print manuscript-oriented quantitative diagnostics for Figure 2."""
    global_summary = _global_summary_table(
        base_dir=base_dir,
        climate=climate,
        climate_simple=climate_simple,
        agriculture=agriculture,
        agriculture_simple=agriculture_simple,
        year=year,
    )

    print("\n" + "=" * 118)
    print("FIGURE 2: QUANTITATIVE RESULTS OVERVIEW")
    print("=" * 118)

    _print_global_component_breakdown(global_summary)
    _print_main_scenario_comparison(global_summary)
    _print_regional_main_highlights(
        climate=climate,
        agriculture=agriculture,
    )
    _print_stylised_sensitivity(
        global_summary=global_summary,
        climate_diff=climate_diff,
        agriculture_diff=agriculture_diff,
    )
    _print_agriculture_yield_hotspots(
        agriculture=agriculture,
        agriculture_simple=agriculture_simple,
    )

    paragraph3 = _print_paragraph3_sensitivity_diagnostics(
        climate=climate,
        climate_simple=climate_simple,
        agriculture=agriculture,
        agriculture_simple=agriculture_simple,
    )

    print("\nRESULTS-WRITING TAKEAWAYS")
    print("-" * 118)

    climate_main = global_summary.loc["Climate main"]
    agriculture_main = global_summary.loc["Agriculture main"]
    climate_styl = global_summary.loc["Climate stylised"]
    agriculture_styl = global_summary.loc["Agriculture stylised"]

    print(
        f"1. Climate-oriented main deployment has a global mitigation-weighted "
        f"net realised cost of {climate_main['Net']:.1f} $2025/tCO2-eq, "
        f"with revenues offsetting "
        f"{climate_main['Revenue offset of gross cost (%)']:.1f}% of gross costs."
    )
    print(
        f"2. Agriculture-oriented deployment has a global net realised cost of "
        f"{agriculture_main['Net']:.1f} $2025/tCO2-eq; crop-yield plus bio-oil "
        f"revenues offset {agriculture_main['Revenue offset of gross cost (%)']:.1f}% "
        f"of gross costs."
    )
    print(
        f"3. Homogenising biochar changes the global net cost by "
        f"{climate_styl['Net'] - climate_main['Net']:+.1f} $/tCO2-eq in the "
        f"climate-oriented case and "
        f"{agriculture_styl['Net'] - agriculture_main['Net']:+.1f} $/tCO2-eq "
        f"in the agriculture-oriented case."
    )
    print(
        "4. Regional differences can be much larger than the global average; "
        "the ranked regional lists above identify the strongest cases for the "
        "Results text."
    )

    return {
        "global_summary": global_summary,
        "climate_regional_revenue_offsets":
            _regional_revenue_offset_table(climate),
        "agriculture_regional_revenue_offsets":
            _regional_revenue_offset_table(agriculture),
        "climate_global_stylised_minus_main":
            _global_stylised_minus_main_component_delta(
                global_summary,
                "Climate main",
                "Climate stylised",
            ),
        "agriculture_global_stylised_minus_main":
            _global_stylised_minus_main_component_delta(
                global_summary,
                "Agriculture main",
                "Agriculture stylised",
            ),
        "paragraph3": paragraph3,
    }


# =============================================================================
# Main figure
# =============================================================================

def plot_cost_component_figure(
    base_dir=BASE_DIR,
    year=YEAR,
    cumulative=CUMULATIVE,
    start_year=START_YEAR,
    output_dir=OUTPUT_DIR,
    output_stem=OUTPUT_STEM,
    show=SHOW_FIGURE,
):
    _set_publication_style()

    base_dir = Path(base_dir)

    climate = load_realised_cost_intensities(
        base_dir / CLIMATE_SCENARIO,
        year=year,
        cumulative=cumulative,
        start_year=start_year,
    )

    climate_simple = load_realised_cost_intensities(
        base_dir / CLIMATE_SIMPLE_SCENARIO,
        year=year,
        cumulative=cumulative,
        start_year=start_year,
    )

    agriculture = load_realised_cost_intensities(
        base_dir / AGRICULTURE_SCENARIO,
        year=year,
        cumulative=cumulative,
        start_year=start_year,
    )

    agriculture_simple = load_realised_cost_intensities(
        base_dir / AGRICULTURE_SIMPLE_SCENARIO,
        year=year,
        cumulative=cumulative,
        start_year=start_year,
    )

    climate_diff = calculate_difference_dataframe(
        climate,
        climate_simple,
    )

    agriculture_diff = calculate_difference_dataframe(
        agriculture,
        agriculture_simple,
    )

    # Shared x-axis scales by panel type.
    absolute_xlim = _absolute_xlim(
        [climate, agriculture]
    )
    difference_xlim = _difference_xlim(
        [climate_diff, agriculture_diff]
    )

    fig = plt.figure(figsize=FIGSIZE)

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=WIDTH_RATIOS,
        height_ratios=HEIGHT_RATIOS,
        hspace=0.28,
        wspace=0.14,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # Row 1: climate.
    _plot_absolute(
        ax_a,
        climate,
        show_ylabels=True,
    )
    _plot_difference(
        ax_b,
        climate_diff,
        show_ylabels=False,
    )

    # Row 2: agriculture.
    _plot_absolute(
        ax_c,
        agriculture,
        show_ylabels=True,
    )
    _plot_difference(
        ax_d,
        agriculture_diff,
        show_ylabels=False,
    )

    ax_a.set_xlim(*absolute_xlim)
    ax_c.set_xlim(*absolute_xlim)

    ax_b.set_xlim(*difference_xlim)
    ax_d.set_xlim(*difference_xlim)

    ax_a.set_xlabel(ABSOLUTE_XLABEL)
    ax_c.set_xlabel(ABSOLUTE_XLABEL)

    ax_b.set_xlabel(DIFFERENCE_XLABEL)
    ax_d.set_xlabel(DIFFERENCE_XLABEL)

    ax_a.set_title(
        _panel_title("a", f"Biochar for climate, {year}"),
        loc="left",
        pad=8,
        fontweight="normal",
    )

    ax_b.set_title(
        _panel_title("b", f"Stylised biochar minus main, climate, {year}"),
        loc="left",
        pad=8,
        fontweight="normal",
    )

    ax_c.set_title(
        _panel_title("c", f"Biochar for agriculture, {year}"),
        loc="left",
        pad=8,
        fontweight="normal",
    )

    ax_d.set_title(
        _panel_title("d", f"Stylised biochar minus main, agriculture, {year}"),
        loc="left",
        pad=8,
        fontweight="normal",
    )

    annotation_text = r"SSP2 carbon price 400 \$ tCO$_2^{-1}$"

    # Absolute-cost panels have substantial unused space on the lower-left
    # side, so place their scenario annotation there to avoid the positive
    # cost bars. Difference panels retain the lower-right placement.
    for ax in (ax_a, ax_c):
        ax.text(
            0.02,
            0.035,
            annotation_text,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.5,
            },
            zorder=10,
        )

    for ax in (ax_b, ax_d):
        ax.text(
            0.98,
            0.035,
            annotation_text,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8.9,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 1.5,
            },
            zorder=10,
        )

    # One legend for the entire figure.
    fig.legend(
        handles=_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=7,
        frameon=False,
        columnspacing=1.25,
        handletextpad=0.50,
    )

    fig.subplots_adjust(
        left=0.14,
        right=0.985,
        top=0.95,
        bottom=0.16,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"

    fig.savefig(
        png_path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    sanity = run_sanity_checks(
        climate=climate,
        climate_simple=climate_simple,
        agriculture=agriculture,
        agriculture_simple=agriculture_simple,
        climate_diff=climate_diff,
        agriculture_diff=agriculture_diff,
    )

    results_overview = print_quantitative_results_overview(
        base_dir=base_dir,
        climate=climate,
        climate_simple=climate_simple,
        agriculture=agriculture,
        agriculture_simple=agriculture_simple,
        climate_diff=climate_diff,
        agriculture_diff=agriculture_diff,
        year=year,
    )

    # Save panel data for transparent manuscript diagnostics.
    climate.to_csv(
        output_dir / f"{output_stem}_panel_a_climate.csv"
    )
    climate_diff.to_csv(
        output_dir / f"{output_stem}_panel_b_climate_difference.csv"
    )
    agriculture.to_csv(
        output_dir / f"{output_stem}_panel_c_agriculture.csv"
    )
    agriculture_diff.to_csv(
        output_dir / f"{output_stem}_panel_d_agriculture_difference.csv"
    )

    sanity["yield_table"].to_csv(
        output_dir / f"{output_stem}_yield_revenue_sanity_checks.csv"
    )

    results_overview["global_summary"].to_csv(
        output_dir / f"{output_stem}_results_global_summary.csv"
    )
    results_overview["climate_regional_revenue_offsets"].to_csv(
        output_dir / f"{output_stem}_results_climate_regional_offsets.csv"
    )
    results_overview["agriculture_regional_revenue_offsets"].to_csv(
        output_dir / f"{output_stem}_results_agriculture_regional_offsets.csv"
    )
    results_overview["climate_global_stylised_minus_main"].to_csv(
        output_dir / f"{output_stem}_results_climate_stylised_difference.csv",
        header=["Stylised minus main ($/tCO2-eq)"],
    )
    results_overview["agriculture_global_stylised_minus_main"].to_csv(
        output_dir / f"{output_stem}_results_agriculture_stylised_difference.csv",
        header=["Stylised minus main ($/tCO2-eq)"],
    )

    results_overview["paragraph3"]["climate-oriented"][
        "global_component_decomposition"
    ].to_csv(
        output_dir / f"{output_stem}_paragraph3_climate_component_decomposition.csv"
    )
    results_overview["paragraph3"]["agriculture-oriented"][
        "global_component_decomposition"
    ].to_csv(
        output_dir / f"{output_stem}_paragraph3_agriculture_component_decomposition.csv"
    )
    results_overview["paragraph3"]["climate-oriented"][
        "regional_net_changes"
    ].to_csv(
        output_dir / f"{output_stem}_paragraph3_climate_regional_relative_changes.csv"
    )
    results_overview["paragraph3"]["agriculture-oriented"][
        "regional_net_changes"
    ].to_csv(
        output_dir / f"{output_stem}_paragraph3_agriculture_regional_relative_changes.csv"
    )
    results_overview["paragraph3"]["climate-oriented"][
        "regional_component_differences"
    ].to_csv(
        output_dir / f"{output_stem}_paragraph3_climate_regional_component_differences.csv"
    )
    results_overview["paragraph3"]["agriculture-oriented"][
        "regional_component_differences"
    ].to_csv(
        output_dir / f"{output_stem}_paragraph3_agriculture_regional_component_differences.csv"
    )

    diagnostics = {
        "a_climate_closure": _closure_error_absolute(climate),
        "b_climate_difference_closure": _closure_error_difference(
            climate_diff
        ),
        "c_agriculture_closure": _closure_error_absolute(agriculture),
        "d_agriculture_difference_closure": _closure_error_difference(
            agriculture_diff
        ),
    }

    print("\n" + "=" * 110)
    print("Accounting closure diagnostics")
    print("=" * 110)

    for name, error in diagnostics.items():
        print(f"{name:38s}: {error:.6g} $/tCO2-eq")

        if error > CLOSURE_TOLERANCE:
            warnings.warn(
                f"{name} accounting does not close exactly. "
                "Inspect the scenario outputs before using the figure.",
                RuntimeWarning,
            )

    print("\nSaved:")
    print(png_path)
    print(pdf_path)

    if show:
        plt.show()

    return {
        "fig": fig,
        "axes": {
            "a": ax_a,
            "b": ax_b,
            "c": ax_c,
            "d": ax_d,
        },
        "data": {
            "climate": climate,
            "climate_difference": climate_diff,
            "agriculture": agriculture,
            "agriculture_difference": agriculture_diff,
        },
        "absolute_xlim": absolute_xlim,
        "difference_xlim": difference_xlim,
        "diagnostics": diagnostics,
        "sanity": sanity,
        "results_overview": results_overview,
        "png_path": png_path,
        "pdf_path": pdf_path,
    }


if __name__ == "__main__":
    outputs = plot_cost_component_figure()

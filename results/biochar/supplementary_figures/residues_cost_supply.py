from pathlib import Path
import math
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from timer.cdr.cdr_util import (
    Region26,
    Residue,
    RCCPlus,
    BiocharFeedstock,
)
from timer.cdr.biochar.feedstock import BIOCHAR_FEEDSTOCK_NAMES
from timer.cdr.results.biochar.biochar_results import IMAGE_REGIONS, R10


"""
Appendix figures: residue cost-supply curves.

Figures are produced separately for 2050 and 2100:

    1. One panel per IMAGE R26 region.
    2. One panel per R10 region.
    3. One panel per non-biofuel residue type.

Biofuel residue categories are excluded because their supply is zero in the
current model inputs.

For each curve, every region-residue-cost-bin element is treated as an
incremental supply block. Blocks are pooled for the requested aggregation,
sorted by ResCostCurveDM, and ResPotCurveDM is cumulatively summed.

A common high-cost cutoff is determined from the pooled 2050 and 2100 data.
By default, the plotted range retains 99.5% of total non-biofuel residue
supply. This removes extreme-cost bins carrying negligible supply while
preserving essentially the full resource base.

ResPotCurveDM is plotted as annual dry-matter potential; no timestep-width
multiplication is applied.
"""


# =============================================================================
# Configuration
# =============================================================================

RESIDUE_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\biochar\SSP2_M"
)

OUTPUT_DIR = (
    Path(
        r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output\figures"
    )
    / "appendix_residue_cost_supply"
)

YEARS = (2050, 2100)

RESIDUE_POTENTIAL_VAR = "ResPotCurveDM"
RESIDUE_COST_VAR = "ResCostCurveDM"


# -----------------------------------------------------------------------------
# Residue selection
# -----------------------------------------------------------------------------

# These categories are present in the biochar feedstock bookkeeping but have
# zero supply and are therefore omitted from the appendix figures.
EXCLUDED_RESIDUES = {
    "biofuel_sugarcane",
    "biofuel_maize",
    "biofuel_woody",
    "biofuel_nwoody",
}

INCLUDED_RESIDUE_INDICES = np.array(
    [
        i
        for i, name in enumerate(BIOCHAR_FEEDSTOCK_NAMES)
        if name not in EXCLUDED_RESIDUES
    ],
    dtype=int,
)

INCLUDED_RESIDUE_NAMES = tuple(
    BIOCHAR_FEEDSTOCK_NAMES[i]
    for i in INCLUDED_RESIDUE_INDICES
)


# -----------------------------------------------------------------------------
# High-cost-tail treatment
# -----------------------------------------------------------------------------

# Retain this fraction of total non-biofuel residue supply when determining
# the common cost-axis upper limit.
COST_SUPPLY_RETAINED_FRACTION = 0.995


# -----------------------------------------------------------------------------
# Plot units / labels
# -----------------------------------------------------------------------------

# tonnes DM yr-1 -> Mt DM yr-1
SUPPLY_SCALE = 1.0e-6

SUPPLY_AXIS_LABEL = (
    r"Cumulative residue supply "
    r"(Mt DM yr$^{-1}$)"
)

COST_AXIS_LABEL = (
    r"Residue cost "
    r"(US\$ t$_{\mathrm{DM}}^{-1}$)"
)


SAVE_DPI = 600
SAVE_PDF = True
SHOW_FIGURES = True

R26_NCOLS = 5
R10_NCOLS = 5
RESIDUE_NCOLS = 4


# =============================================================================
# Display names
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


RESIDUE_DISPLAY = {
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
    "forestry": "Forestry",
}


# =============================================================================
# Input helpers
# =============================================================================

def _resolve_case_insensitive_file(directory, stem):

    directory = Path(directory)

    for suffix in (".NC", ".nc"):

        candidate = directory / f"{stem}{suffix}"

        if candidate.exists():
            return candidate

    return directory / f"{stem}.NC"


def _time_years(da):

    if "time" not in da.dims:
        return None

    values = da["time"].values

    if np.issubdtype(
        values.dtype,
        np.datetime64,
    ):
        return (
            da["time"]
            .dt.year
            .values
            .astype(int)
        )

    try:
        return np.asarray(
            values,
            dtype=int,
        )

    except (TypeError, ValueError):

        return np.asarray(
            da["time"].dt.year.values,
            dtype=int,
        )


def _select_year(
        da,
        year,
        source,
):

    if "time" not in da.dims:

        warnings.warn(
            f"{source.name} has no time dimension; "
            f"using the same values for requested year {year}.",
            RuntimeWarning,
        )

        return da

    years = _time_years(da)

    matches = np.flatnonzero(
        years == int(year)
    )

    if matches.size == 0:

        raise ValueError(
            f"Year {year} not found in {source}. "
            f"Available years: {years.tolist()}"
        )

    return da.isel(
        time=int(matches[0])
    )


def _read_curve_variable(
        directory,
        variable,
        year,
):

    path = _resolve_case_insensitive_file(
        directory,
        variable,
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Could not find {variable}: {path}"
        )

    with xr.open_dataset(
        path,
        engine="netcdf4",
    ) as ds:

        if variable not in ds:

            raise KeyError(
                f"Variable '{variable}' not found in {path}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = _select_year(
            ds[variable],
            year,
            path,
        )

        attrs = dict(
            ds[variable].attrs
        )

        values = np.asarray(
            da.values,
            dtype=np.float64,
        )

    expected_size = (
        len(Region26.coords)
        * len(Residue.coords)
        * len(RCCPlus.coords)
    )

    if values.size != expected_size:

        raise ValueError(
            f"Unexpected size for {variable} in {path}: "
            f"{values.size}; expected {expected_size} = "
            f"{len(Region26.coords)} regions × "
            f"{len(Residue.coords)} residues × "
            f"{len(RCCPlus.coords)} RCCPlus bins."
        )

    values = values.reshape(
        len(Region26.coords),
        len(Residue.coords),
        len(RCCPlus.coords),
    )

    return (
        values,
        attrs,
        path,
    )


def load_residue_curves(
        directory=RESIDUE_DIR,
        year=2100,
):
    """
    Load the residue potential and cost arrays used by the biochar model.

    Biofuel residues are removed here so all subsequent analysis and plotting
    consistently uses only the residue categories relevant to the appendix.

    Returned arrays have shape:

        [R26, non-biofuel residue, actual cost bin]
    """

    (
        potential,
        potential_attrs,
        potential_path,
    ) = _read_curve_variable(
        directory,
        RESIDUE_POTENTIAL_VAR,
        year,
    )

    (
        cost,
        cost_attrs,
        cost_path,
    ) = _read_curve_variable(
        directory,
        RESIDUE_COST_VAR,
        year,
    )

    # Same convention as timer.cdr.biochar.feedstock.get_cost_bins().
    cost_bins = (
        len(RCCPlus.coords) - 1
    )

    n_regions = len(
        IMAGE_REGIONS
    )

    n_feedstocks = len(
        BiocharFeedstock.coords
    )

    if n_feedstocks != len(
        BIOCHAR_FEEDSTOCK_NAMES
    ):

        raise ValueError(
            "BiocharFeedstock coordinate length does not match "
            "BIOCHAR_FEEDSTOCK_NAMES."
        )

    potential = potential[
        :n_regions,
        :n_feedstocks,
        :cost_bins,
    ]

    cost = cost[
        :n_regions,
        :n_feedstocks,
        :cost_bins,
    ]

    # Exclude the four biofuel-residue categories.
    potential = potential[
        :,
        INCLUDED_RESIDUE_INDICES,
        :,
    ]

    cost = cost[
        :,
        INCLUDED_RESIDUE_INDICES,
        :,
    ]

    potential = np.where(
        np.isfinite(potential)
        & (potential > 0.0),
        potential,
        0.0,
    )

    cost = np.where(
        np.isfinite(cost),
        cost,
        np.nan,
    )

    return {
        "year": int(year),
        "potential": potential,
        "cost": cost,
        "potential_attrs": potential_attrs,
        "cost_attrs": cost_attrs,
        "potential_path": potential_path,
        "cost_path": cost_path,
    }


# =============================================================================
# Cost cutoff
# =============================================================================

def _nice_ceiling(value):
    """
    Round a positive value upward to a convenient plotting limit using
    1, 2, 2.5, 5, 10 × 10^n.
    """

    value = float(value)

    if not np.isfinite(value) or value <= 0.0:
        return 1.0

    exponent = np.floor(
        np.log10(value)
    )

    scale = 10.0 ** exponent
    fraction = value / scale

    nice_values = (
        1.0,
        2.0,
        2.5,
        5.0,
        10.0,
    )

    for nice in nice_values:

        if fraction <= nice:

            return nice * scale

    return 10.0 * scale


def determine_cost_cutoff(
        data_by_year,
        retained_fraction=COST_SUPPLY_RETAINED_FRACTION,
):
    """
    Determine one common cost cutoff from 2050 and 2100.

    Each individual region × residue × cost-bin entry contributes to the
    weighted distribution according to its ResPotCurveDM supply.

    The raw cutoff is the lowest cost at which `retained_fraction` of total
    supply has been accumulated. It is then rounded upward to a convenient
    plotting value.
    """

    if not (
        0.0
        < retained_fraction
        <= 1.0
    ):
        raise ValueError(
            "retained_fraction must lie in (0, 1]."
        )

    all_costs = []
    all_supply = []

    for data in data_by_year.values():

        cost = np.asarray(
            data["cost"],
            dtype=np.float64,
        )

        supply = np.asarray(
            data["potential"],
            dtype=np.float64,
        )

        valid = (
            np.isfinite(cost)
            & np.isfinite(supply)
            & (supply > 0.0)
        )

        if np.any(valid):

            all_costs.append(
                cost[valid]
            )

            all_supply.append(
                supply[valid]
            )

    if not all_costs:

        raise ValueError(
            "No positive residue supply with finite costs was found."
        )

    costs = np.concatenate(
        all_costs
    )

    supply = np.concatenate(
        all_supply
    )

    order = np.argsort(
        costs,
        kind="mergesort",
    )

    costs = costs[order]
    supply = supply[order]

    cumulative = np.cumsum(
        supply,
        dtype=np.float64,
    )

    total_supply = float(
        cumulative[-1]
    )

    target_supply = (
        retained_fraction
        * total_supply
    )

    index = int(
        np.searchsorted(
            cumulative,
            target_supply,
            side="left",
        )
    )

    index = min(
        index,
        len(costs) - 1,
    )

    raw_cutoff = float(
        costs[index]
    )

    rounded_cutoff = float(
        _nice_ceiling(
            raw_cutoff
        )
    )

    retained_at_rounded = float(
        np.sum(
            supply[
                costs <= rounded_cutoff
            ],
            dtype=np.float64,
        )
        / total_supply
    )

    excluded_fraction = (
        1.0
        - retained_at_rounded
    )

    return {
        "raw_cutoff": raw_cutoff,
        "plot_cutoff": rounded_cutoff,
        "retained_fraction": retained_at_rounded,
        "excluded_fraction": excluded_fraction,
        "total_supply": total_supply,
    }


# =============================================================================
# Cost-supply curve construction
# =============================================================================

def build_cost_supply_curve(
        cost_blocks,
        supply_blocks,
        cost_cutoff,
):
    """
    Convert incremental residue-supply blocks into a cumulative cost-supply
    curve.

    Blocks above the common appendix cost cutoff are excluded.
    """

    costs = np.asarray(
        cost_blocks,
        dtype=np.float64,
    ).reshape(-1)

    supply = np.asarray(
        supply_blocks,
        dtype=np.float64,
    ).reshape(-1)

    valid = (
        np.isfinite(costs)
        & np.isfinite(supply)
        & (supply > 0.0)
        & (costs <= cost_cutoff)
    )

    if not np.any(valid):

        return (
            np.array(
                [],
                dtype=np.float64,
            ),
            np.array(
                [],
                dtype=np.float64,
            ),
        )

    costs = costs[valid]
    supply = supply[valid]

    order = np.argsort(
        costs,
        kind="mergesort",
    )

    costs = costs[order]
    supply = supply[order]

    # Combine blocks at identical costs.
    unique_costs, inverse = np.unique(
        costs,
        return_inverse=True,
    )

    supply_at_cost = np.zeros(
        unique_costs.size,
        dtype=np.float64,
    )

    np.add.at(
        supply_at_cost,
        inverse,
        supply,
    )

    cumulative_supply = np.cumsum(
        supply_at_cost,
        dtype=np.float64,
    )

    start_cost = min(
        0.0,
        float(unique_costs[0]),
    )

    x = np.concatenate(
        (
            [start_cost],
            unique_costs,
        )
    )

    y = np.concatenate(
        (
            [0.0],
            cumulative_supply
            * SUPPLY_SCALE,
        )
    )

    return x, y


# =============================================================================
# Plot formatting
# =============================================================================

def _set_publication_style():

    plt.rcParams.update(
        {
            "font.size": 9.0,
            "axes.titlesize": 9.5,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
        }
    )


def _style_axis(
        ax,
        cost_cutoff,
):

    ax.set_xlim(
        0.0,
        cost_cutoff,
    )

    ax.set_ylim(
        bottom=0.0
    )

    ax.grid(
        axis="both",
        alpha=0.22,
        linewidth=0.5,
    )

    ax.tick_params(
        direction="out",
        length=3.0,
        width=0.7,
    )

    for spine in ax.spines.values():

        spine.set_linewidth(
            0.8
        )


def _plot_curve(
        ax,
        costs,
        supply,
        cost_cutoff,
):

    x, y = build_cost_supply_curve(
        costs,
        supply,
        cost_cutoff,
    )

    if x.size:

        ax.step(
            x,
            y,
            where="post",
            linewidth=1.5,
        )

    else:

        ax.text(
            0.5,
            0.5,
            "No positive supply",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.0,
        )

    _style_axis(
        ax,
        cost_cutoff,
    )


def _make_axes(
        n_panels,
        ncols,
        panel_width,
        panel_height,
):

    nrows = int(
        math.ceil(
            n_panels / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            panel_width * ncols,
            panel_height * nrows,
        ),
        squeeze=False,
        sharex=True,
        constrained_layout=True,
    )

    return (
        fig,
        axes.reshape(-1),
    )


def _finish_figure(
        fig,
        axes,
        used,
        title,
        output_stem,
):

    for ax in axes[used:]:

        ax.set_visible(
            False
        )

    # fig.suptitle(
    #     title,
    #     fontsize=12.0,
    #     y=1.015,
    # )

    fig.supxlabel(
        COST_AXIS_LABEL,
        fontsize=10.5,
    )

    fig.supylabel(
        SUPPLY_AXIS_LABEL,
        fontsize=10.5,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    png = (
        OUTPUT_DIR
        / f"{output_stem}.png"
    )

    fig.savefig(
        png,
        dpi=SAVE_DPI,
    )

    if SAVE_PDF:

        pdf = (
            OUTPUT_DIR
            / f"{output_stem}.pdf"
        )

        fig.savefig(
            pdf
        )

    return png


# =============================================================================
# Figure 1: R26 cost-supply curves
# =============================================================================

def plot_r26_figure(
        data,
        cost_cutoff,
):

    year = data["year"]
    potential = data["potential"]
    cost = data["cost"]

    fig, axes = _make_axes(
        n_panels=len(IMAGE_REGIONS),
        ncols=R26_NCOLS,
        panel_width=3.0,
        panel_height=2.25,
    )

    for i, region_code in enumerate(
        IMAGE_REGIONS
    ):

        ax = axes[i]

        _plot_curve(
            ax,
            cost[i, :, :],
            potential[i, :, :],
            cost_cutoff,
        )

        # Region name only; do not add R26 abbreviation in parentheses.
        ax.set_title(
            R26_DISPLAY.get(
                region_code,
                region_code,
            ),
            loc="left",
            pad=4,
        )

    stem = (
        f"appendix_residue_cost_supply_r26_{year}"
    )

    _finish_figure(
        fig,
        axes,
        len(IMAGE_REGIONS),
        (
            "Residue cost-supply curves "
            f"by IMAGE R26 region, {year}"
        ),
        stem,
    )

    return fig


# =============================================================================
# Figure 2: R10 cost-supply curves
# =============================================================================

def plot_r10_figure(
        data,
        cost_cutoff,
):

    year = data["year"]
    potential = data["potential"]
    cost = data["cost"]

    r10_names = list(
        R10.keys()
    )

    fig, axes = _make_axes(
        n_panels=len(r10_names),
        ncols=R10_NCOLS,
        panel_width=3.1,
        panel_height=2.7,
    )

    for i, r10_name in enumerate(
        r10_names
    ):

        ax = axes[i]

        members = R10[
            r10_name
        ]

        idx = [
            IMAGE_REGIONS.index(code)
            for code in members
        ]

        # Pool individual R26 × residue × cost-bin blocks rather than averaging
        # regional costs.
        _plot_curve(
            ax,
            cost[idx, :, :],
            potential[idx, :, :],
            cost_cutoff,
        )

        ax.set_title(
            r10_name,
            loc="left",
            pad=4,
        )

    stem = (
        f"appendix_residue_cost_supply_r10_{year}"
    )

    _finish_figure(
        fig,
        axes,
        len(r10_names),
        (
            "Residue cost-supply curves "
            f"by R10 region, {year}"
        ),
        stem,
    )

    return fig


# =============================================================================
# Figure 3: cost-supply curves by residue type
# =============================================================================

def plot_residue_figure(
        data,
        cost_cutoff,
):

    year = data["year"]
    potential = data["potential"]
    cost = data["cost"]

    residue_names = list(
        INCLUDED_RESIDUE_NAMES
    )

    fig, axes = _make_axes(
        n_panels=len(residue_names),
        ncols=RESIDUE_NCOLS,
        panel_width=3.25,
        panel_height=2.45,
    )

    for i, residue_name in enumerate(
        residue_names
    ):

        ax = axes[i]

        # Pool the selected residue across all R26 regions.
        _plot_curve(
            ax,
            cost[:, i, :],
            potential[:, i, :],
            cost_cutoff,
        )

        ax.set_title(
            RESIDUE_DISPLAY.get(
                residue_name,
                residue_name
                .replace("_", " ")
                .title(),
            ),
            loc="left",
            pad=4,
        )

    stem = (
        f"appendix_residue_cost_supply_residue_{year}"
    )

    _finish_figure(
        fig,
        axes,
        len(residue_names),
        (
            "Residue cost-supply curves "
            f"by residue type, {year}"
        ),
        stem,
    )

    return fig


# =============================================================================
# Main
# =============================================================================

def main():

    _set_publication_style()

    # Read both years before determining the common cost cutoff.
    data_by_year = {
        int(year): load_residue_curves(
            directory=RESIDUE_DIR,
            year=int(year),
        )
        for year in YEARS
    }

    cutoff = determine_cost_cutoff(
        data_by_year,
        retained_fraction=COST_SUPPLY_RETAINED_FRACTION,
    )

    cost_cutoff = cutoff[
        "plot_cutoff"
    ]

    print(
        "\nResidue cost-supply appendix"
    )

    print(
        "-" * 72
    )

    print(
        "Excluded residue categories:"
    )

    for name in sorted(
        EXCLUDED_RESIDUES
    ):

        print(
            f"  - {name}"
        )

    print(
        "\nIncluded residue categories:"
    )

    for name in INCLUDED_RESIDUE_NAMES:

        print(
            f"  - {name}"
        )

    print(
        "\nHigh-cost-tail treatment:"
    )

    print(
        f"  Target retained supply: "
        f"{100.0 * COST_SUPPLY_RETAINED_FRACTION:.2f}%"
    )

    print(
        f"  Raw supply-weighted cutoff: "
        f"{cutoff['raw_cutoff']:.3f} US$/t DM"
    )

    print(
        f"  Rounded plotting cutoff: "
        f"{cost_cutoff:g} US$/t DM"
    )

    print(
        f"  Supply retained after rounding: "
        f"{100.0 * cutoff['retained_fraction']:.4f}%"
    )

    print(
        f"  Supply excluded above cutoff: "
        f"{100.0 * cutoff['excluded_fraction']:.4f}%"
    )

    first = data_by_year[
        int(YEARS[0])
    ]

    print(
        "\nInput files:"
    )

    print(
        f"  ResPotCurveDM: "
        f"{first['potential_path']}"
    )

    print(
        f"  ResCostCurveDM: "
        f"{first['cost_path']}"
    )

    print(
        "\nNetCDF unit metadata:"
    )

    print(
        "  ResPotCurveDM:",
        first[
            "potential_attrs"
        ].get(
            "units",
            "<not provided>",
        ),
    )

    print(
        "  ResCostCurveDM:",
        first[
            "cost_attrs"
        ].get(
            "units",
            "<not provided>",
        ),
    )

    figures = []

    for year in YEARS:

        data = data_by_year[
            int(year)
        ]

        figures.append(
            plot_r26_figure(
                data,
                cost_cutoff,
            )
        )

        figures.append(
            plot_r10_figure(
                data,
                cost_cutoff,
            )
        )

        figures.append(
            plot_residue_figure(
                data,
                cost_cutoff,
            )
        )

    if SHOW_FIGURES:

        plt.show()

    else:

        for fig in figures:

            plt.close(fig)


if __name__ == "__main__":
    main()
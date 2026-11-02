import warnings
import yaml
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
from timer.cdr.results.biochar.biochar_results import R10, IMAGE_REGIONS
from timer.cdr.cdr_util import Region26, Residue, RCCPlus, BiocharFeedstock

BASE_DIR = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output")

CARBON_PRICES = (50, 100, 150, 200, 250, 400)
YEAR = 2100
CUMULATIVE_START_YEAR = 2030
PANEL_B_START_YEAR = 2030
PANEL_B_END_YEAR = YEAR
PANEL_B_CARBON_PRICE = 400
PANEL_B_INCLUDE_DIFFUSION = True
PANEL_C_PRICES = (100, 150, 200, 400)
PANEL_C_LABEL_TOP_DEVIATIONS = 4
PANEL_C_ANNOTATE_SPEARMAN = True

PANEL_C_MANUAL_LABEL_OFFSETS = {
    (150, "USA"): (-12, 8),
    (150, "INDIA"): (12, -9),
    (200, "INDIA"): (10, -14),
    (200, "BRA"): (-12, 10),
    (400, "BRA"): (-12, 11),
    (400, "WAF"): (-8, 10),
    (400, "CHN"): (16, 3),
}

RESIDUE_DIR_FALLBACK = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\biochar\SSP2_M"
)
RESIDUE_POTENTIAL_VAR = "ResPotCurveDM"
RESIDUE_USE_VAR = "BiocharResidueDMUsed"


MAIN_TEMPLATE = "biochar_c{price}"
STYLISED_TEMPLATE = "biochar_c{price}_stylised"

OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_STEM = "biochar_figure1_price_and_sensitivity_timeseries"

SAVE_DPI = 600
SHOW_FIGURE = True

REALIZED_MITIGATION_VAR = "BiocharRealisedMitigation"

# Fallback components for older runs without BiocharRealisedMitigation.
# Field CDR is required; disabled co-service outputs may legitimately be absent.
FALLBACK_COMPONENTS = (
    ("BiocharFieldCDR", True),
    ("BiocharSyngasCCS", False),
    ("BiocharFossilMitig", False),
    ("BiocharN2OMitig", False),
    ("BiocharCH4Mitig", False),
)

# Global scenario colours for panels a and b.
COLOR_MAIN = "#222222"
COLOR_STYLISED = "#2C7FB8"
COLOR_RESIDUE_COMPETITION = "#D07A37"
COLOR_APPLICATION_CAP = "#8561A8"
COLOR_DIFFUSION = "#287C77"

# Main/cap/proportional runs below are present in the uploaded scenario YAML.
# Optional diffusion runs are NOT in that YAML; change these two directory names
# to match your future output directories if you add diffusion scenarios.
# Every entry's 'directory' is relative to BASE_DIR.
PANEL_B_SCENARIOS = (
    dict(key="main", directory="biochar_c400", label="Reference",
         color=COLOR_MAIN, linestyle="-", required=True),
    dict(key="residue_competition", directory="biochar_c400_prop",
         label="Residue competition", color=COLOR_RESIDUE_COMPETITION,
         linestyle="--", required=True),
    dict(key="application_cap", directory="biochar_c400_cap",
         label="Cumulative application cap", color=COLOR_APPLICATION_CAP,
         linestyle="-.", required=True),
    dict(key="diffusion_low", directory="biochar_c400_diffusion_low",
         label="Low diffusion", color=COLOR_DIFFUSION,
         linestyle=(0, (5, 2.5)), required=False),
    dict(key="diffusion_high", directory="biochar_c400_diffusion_high",
         label="High diffusion", color=COLOR_DIFFUSION,
         linestyle=(0, (1.5, 2)), required=False),
)

# Regional colours for panel a.
R10_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#9D755D",
    "#D37295",
    "#7F7F7F",
    "#BCBD22",
    "#B279A2",
)

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

FIGSIZE = (15.4, 9.2)
GRID_ALPHA = 0.24
FRAME_COLOR = "0.35"
FRAME_WIDTH = 0.9


# =============================================================================
# Publication style
# =============================================================================

def _set_publication_style():
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.0,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "axes.linewidth": FRAME_WIDTH,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_title(letter, text):
    return rf"$\bf{{{letter}.}}$ {text}"


def _style_axis(ax, grid_axis="both"):
    ax.grid(
        axis=grid_axis,
        linestyle="--",
        linewidth=0.65,
        alpha=GRID_ALPHA,
    )
    ax.set_axisbelow(True)

    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color(FRAME_COLOR)
        ax.spines[spine].set_linewidth(FRAME_WIDTH)

    ax.tick_params(width=FRAME_WIDTH)


# =============================================================================
# Scenario / time helpers
# =============================================================================

def _scenario_path(base_dir, template, price):
    return Path(base_dir) / template.format(price=int(price))


def _time_years(da):
    if "time" not in da.dims:
        return None

    values = da["time"].values

    if np.issubdtype(values.dtype, np.datetime64):
        return da["time"].dt.year.values.astype(int)

    try:
        return np.asarray(values, dtype=int)
    except (TypeError, ValueError):
        return np.asarray(da["time"].dt.year.values, dtype=int)


def _year_index_and_dt(da, year):
    """
    Return the time-index of `year` and the represented timestep width in years.

    For the final timestep (e.g. 2100), use the distance to the previous
    coordinate. For the first timestep only, use the distance to the next.
    """
    if "time" not in da.dims:
        raise ValueError(
            "Cannot infer annual mitigation because the variable has no "
            "'time' dimension."
        )

    years = _time_years(da)
    matches = np.where(years == int(year))[0]

    if len(matches) == 0:
        raise ValueError(
            f"Year {year} not found. Available years: {years.tolist()}"
        )

    idx = int(matches[0])

    if idx > 0:
        dt = float(years[idx] - years[idx - 1])
    elif len(years) > 1:
        dt = float(years[1] - years[0])
    else:
        raise ValueError(
            "Cannot infer timestep width from a single time coordinate."
        )

    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(
            f"Invalid inferred timestep width {dt} years for year {year}."
        )

    return idx, dt


def _reduce_to_r26(da):
    """
    Sum all dimensions except region and return one value per IMAGE R26 region.
    """
    if "region" not in da.dims:
        raise ValueError(
            f"Expected a 'region' dimension; found dimensions {da.dims}."
        )

    extra_dims = [dim for dim in da.dims if dim != "region"]
    if extra_dims:
        da = da.sum(dim=extra_dims, skipna=True)

    values = np.asarray(da.values, dtype=np.float64).squeeze()

    if values.ndim != 1:
        raise ValueError(
            f"Expected a one-dimensional regional vector; got {values.shape}."
        )

    # Some TIMER outputs may contain an additional aggregate region. Keep the
    # IMAGE R26 entries only, matching the established plotting modules.
    if values.size == len(IMAGE_REGIONS):
        return values

    if values.size > len(IMAGE_REGIONS):
        return values[:len(IMAGE_REGIONS)]

    raise ValueError(
        f"Expected at least {len(IMAGE_REGIONS)} regional values, "
        f"got {values.size}."
    )


def _select_year_r26(da, year):
    idx, dt = _year_index_and_dt(da, year)
    values = _reduce_to_r26(da.isel(time=idx))
    return values, dt


def _sum_period_r26(da, start_year, end_year):
    """
    Sum timestep-total values directly over the requested model years.
    """
    if "time" not in da.dims:
        raise ValueError("Cumulative mitigation requires a 'time' dimension.")

    years = _time_years(da)
    mask = (years >= int(start_year)) & (years <= int(end_year))

    if not np.any(mask):
        raise ValueError(
            f"No timesteps found in {start_year}-{end_year}. "
            f"Available years: {years.tolist()}"
        )

    return _reduce_to_r26(
        da.isel(time=np.where(mask)[0]).sum(dim="time", skipna=True)
    )


# =============================================================================
# Mitigation loading
# =============================================================================

def _load_realized_annual_r26(scenario_dir, year):
    """
    Load realised net mitigation for one model timestep and annualise it.

    Returns
    -------
    annual_r26 : ndarray
        GtCO2-eq yr-1 by R26.
    dt : float
        Inferred timestep width in years.
    """
    scenario_dir = Path(scenario_dir)
    nc_file = scenario_dir / f"{REALIZED_MITIGATION_VAR}.nc"

    if not nc_file.exists():
        return None

    with xr.open_dataset(nc_file) as ds:
        if REALIZED_MITIGATION_VAR not in ds:
            return None

        step_total_r26, dt = _select_year_r26(
            ds[REALIZED_MITIGATION_VAR],
            year,
        )

    # Critical conversion: output is a timestep total, not an annual rate.
    annual_r26 = step_total_r26 / dt / 1e9
    return annual_r26, dt


def _load_realized_cumulative_r26(scenario_dir, start_year, end_year):
    """
    Load realised net mitigation and sum timestep totals over the period.

    Returns GtCO2-eq by R26.
    """
    scenario_dir = Path(scenario_dir)
    nc_file = scenario_dir / f"{REALIZED_MITIGATION_VAR}.nc"

    if not nc_file.exists():
        return None

    with xr.open_dataset(nc_file) as ds:
        if REALIZED_MITIGATION_VAR not in ds:
            return None

        cumulative_r26 = _sum_period_r26(
            ds[REALIZED_MITIGATION_VAR],
            start_year,
            end_year,
        )

    return cumulative_r26 / 1e9


def _load_fallback_annual_r26(scenario_dir, year):
    """
    Fallback annual mitigation reconstructed from explicit component exports.

    Because those exports are also timestep totals, each component is divided
    by the inferred timestep width. Missing optional co-service components are
    treated as zero.
    """
    scenario_dir = Path(scenario_dir)
    total_step = np.zeros(len(IMAGE_REGIONS), dtype=np.float64)
    inferred_dt = None

    for var_name, required in FALLBACK_COMPONENTS:
        nc_file = scenario_dir / f"{var_name}.nc"

        if not nc_file.exists():
            if required:
                raise FileNotFoundError(
                    f"Required mitigation output not found: {nc_file}"
                )

            warnings.warn(
                f"{nc_file.name} not found in {scenario_dir.name}; "
                "treating this optional/disabled component as zero.",
                RuntimeWarning,
            )
            continue

        with xr.open_dataset(nc_file) as ds:
            if var_name not in ds:
                if required:
                    raise KeyError(
                        f"Variable '{var_name}' not found in {nc_file.name}."
                    )
                continue

            component, dt = _select_year_r26(ds[var_name], year)

        if inferred_dt is None:
            inferred_dt = dt
        elif not np.isclose(inferred_dt, dt):
            raise ValueError(
                f"Inconsistent timestep widths in {scenario_dir.name}: "
                f"{inferred_dt} and {dt} years."
            )

        total_step += component

    if inferred_dt is None:
        raise RuntimeError(
            f"Could not infer timestep width in {scenario_dir}."
        )

    warnings.warn(
        f"Using component-sum fallback for {scenario_dir.name}. "
        "Regional transport-emission penalties are not included in this "
        "fallback total.",
        RuntimeWarning,
    )

    return total_step / inferred_dt / 1e9, inferred_dt


def _load_fallback_cumulative_r26(scenario_dir, start_year, end_year):
    """
    Fallback cumulative mitigation from explicit timestep-total components.
    """
    scenario_dir = Path(scenario_dir)
    total = np.zeros(len(IMAGE_REGIONS), dtype=np.float64)

    for var_name, required in FALLBACK_COMPONENTS:
        nc_file = scenario_dir / f"{var_name}.nc"

        if not nc_file.exists():
            if required:
                raise FileNotFoundError(
                    f"Required mitigation output not found: {nc_file}"
                )
            continue

        with xr.open_dataset(nc_file) as ds:
            if var_name not in ds:
                if required:
                    raise KeyError(
                        f"Variable '{var_name}' not found in {nc_file.name}."
                    )
                continue

            total += _sum_period_r26(
                ds[var_name],
                start_year,
                end_year,
            )

    warnings.warn(
        f"Using component-sum fallback for cumulative mitigation in "
        f"{scenario_dir.name}; regional transport-emission penalties are not "
        "included.",
        RuntimeWarning,
    )

    return total / 1e9


def load_annual_mitigation_r26(scenario_dir, year=YEAR):
    """
    Annual net mitigation in GtCO2-eq yr-1 by R26.

    Prefer BiocharRealisedMitigation. Fall back to explicit components.
    """
    realised = _load_realized_annual_r26(scenario_dir, year)

    if realised is not None:
        annual_r26, dt = realised
        return annual_r26, dt, "BiocharRealisedMitigation"

    annual_r26, dt = _load_fallback_annual_r26(scenario_dir, year)
    return annual_r26, dt, "component fallback"


def load_cumulative_mitigation_r26(
    scenario_dir,
    start_year=CUMULATIVE_START_YEAR,
    end_year=YEAR,
):
    """
    Cumulative net mitigation in GtCO2-eq by R26.

    Timestep totals are summed directly; they are NOT multiplied by dt.
    """
    realised = _load_realized_cumulative_r26(
        scenario_dir,
        start_year,
        end_year,
    )

    if realised is not None:
        return realised, "BiocharRealisedMitigation"

    return (
        _load_fallback_cumulative_r26(
            scenario_dir,
            start_year,
            end_year,
        ),
        "component fallback",
    )


def _annualised_mitigation_series_from_dataarray(da, start_year, end_year):
    """Return (years, R26 annual mitigation, timestep widths).

    Each NetCDF entry is a total for one model timestep, in tonnes CO2-eq.
    Model years are treated as period-end labels, matching the original script.
    """
    if "time" not in da.dims or "region" not in da.dims:
        raise ValueError(
            f"Mitigation output needs 'time' and 'region' dimensions; "
            f"got {da.dims}."
        )

    years_all = np.asarray(_time_years(da), dtype=int)
    if years_all.size < 2:
        raise ValueError("At least two time coordinates are required to annualise.")
    intervals = np.diff(years_all).astype(float)
    if np.any(intervals <= 0) or not np.all(np.isfinite(intervals)):
        raise ValueError(f"Invalid or unsorted mitigation years: {years_all.tolist()}")

    dt_all = np.concatenate(([intervals[0]], intervals))
    chosen = np.flatnonzero(
        (years_all >= int(start_year)) & (years_all <= int(end_year))
    )
    if chosen.size == 0:
        raise ValueError(
            f"No mitigation years within {start_year}-{end_year}; "
            f"available years: {years_all.tolist()}"
        )
    years = years_all[chosen]
    if years[0] != int(start_year) or years[-1] != int(end_year):
        raise ValueError(
            f"Missing requested time-series endpoint in {years_all.tolist()}: "
            f"expected {start_year} and {end_year}."
        )

    selected = da.isel(time=chosen)
    other_dims = [d for d in selected.dims if d not in ("time", "region")]
    if other_dims:
        selected = selected.sum(dim=other_dims, skipna=True)

    values = np.asarray(
        selected.transpose("time", "region").values, dtype=np.float64
    )
    n_region = len(IMAGE_REGIONS)
    if values.ndim != 2 or values.shape[1] < n_region:
        raise ValueError(
            f"Expected [time, >= {n_region} regions] from mitigation output; "
            f"got {values.shape}."
        )
    values = values[:, :n_region]
    dt = dt_all[chosen]
    return years, values / dt[:, None] / 1e9, dt


def load_mitigation_timeseries_r26(
    scenario_dir, start_year=PANEL_B_START_YEAR, end_year=PANEL_B_END_YEAR
):
    """Load global mitigation trajectory as [time, R26] GtCO2-eq/yr.

    Prefer the net realised mitigation export; fall back to explicit components
    for older model outputs. Components are added as timestep totals AFTER
    annualisation and validation of matching time grids.
    """
    scenario_dir = Path(scenario_dir)
    preferred_file = scenario_dir / f"{REALIZED_MITIGATION_VAR}.nc"

    if preferred_file.is_file():
        with xr.open_dataset(preferred_file) as ds:
            if REALIZED_MITIGATION_VAR not in ds:
                raise KeyError(
                    f"{preferred_file} does not contain "
                    f"'{REALIZED_MITIGATION_VAR}'."
                )
            years, annual_r26, dt = _annualised_mitigation_series_from_dataarray(
                ds[REALIZED_MITIGATION_VAR], start_year, end_year
            )
        return years, annual_r26, dt, REALIZED_MITIGATION_VAR

    total_annual = None
    expected_years = None
    expected_dt = None
    missing = []
    for variable, required in FALLBACK_COMPONENTS:
        path = scenario_dir / f"{variable}.nc"
        if not path.is_file():
            if required:
                raise FileNotFoundError(
                    f"Missing {preferred_file} and required fallback {path}."
                )
            missing.append(variable)
            continue
        with xr.open_dataset(path) as ds:
            if variable not in ds:
                if required:
                    raise KeyError(f"Variable '{variable}' absent from {path}.")
                missing.append(variable)
                continue
            years, annual_r26, dt = _annualised_mitigation_series_from_dataarray(
                ds[variable], start_year, end_year
            )

        if expected_years is None:
            expected_years, expected_dt = years, dt
            total_annual = np.zeros_like(annual_r26)
        elif not (
            np.array_equal(years, expected_years)
            and np.allclose(dt, expected_dt)
        ):
            raise ValueError(
                f"Incompatible component time axes in {scenario_dir.name}: "
                f"{variable}."
            )
        total_annual += annual_r26

    if total_annual is None:
        raise FileNotFoundError(
            f"No mitigation timeseries could be loaded from {scenario_dir}."
        )
    warnings.warn(
        f"Using component-sum fallback for {scenario_dir.name}. "
        "Transport emissions are not subtracted. "
        + (f"Missing optional components: {', '.join(missing)}." if missing else ""),
        RuntimeWarning,
    )
    return expected_years, total_annual, expected_dt, "component fallback"


# =============================================================================
# Data assembly
# =============================================================================

def build_carbon_price_data(
    base_dir=BASE_DIR,
    prices=CARBON_PRICES,
    year=YEAR,
):
    """Global annual mitigation for the detailed and stylised price curves."""
    main_global = []
    stylised_global = []
    timestep_diagnostics = []

    for price in prices:
        main, dt_main, source_main = load_annual_mitigation_r26(
            _scenario_path(base_dir, MAIN_TEMPLATE, price), year=year
        )
        stylised, dt_stylised, source_stylised = load_annual_mitigation_r26(
            _scenario_path(base_dir, STYLISED_TEMPLATE, price), year=year
        )
        if not np.isclose(dt_main, dt_stylised):
            raise ValueError(
                f"Timestep width mismatch at c{price}: "
                f"main={dt_main}, stylised={dt_stylised}."
            )
        main_global.append(float(np.nansum(main)))
        stylised_global.append(float(np.nansum(stylised)))
        timestep_diagnostics.append(
            dict(price=int(price), dt=float(dt_main),
                 main_source=source_main, stylised_source=source_stylised)
        )

    return {
        "main_global": np.asarray(main_global, dtype=np.float64),
        "stylised_global": np.asarray(stylised_global, dtype=np.float64),
        "timestep_diagnostics": timestep_diagnostics,
    }


def _resolve_case_insensitive_file(directory, stem):
    directory = Path(directory)

    for suffix in (".NC", ".nc"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    return directory / f"{stem}.NC"


def _read_scenario_yaml(scenario_dir):
    scenario_yaml = Path(scenario_dir) / "scenario.yaml"

    if not scenario_yaml.exists():
        return None

    with scenario_yaml.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        return None

    scenario = config.get("scenario", config)
    return scenario if isinstance(scenario, dict) else None


def _resolve_residue_directory(scenario_dir):
    """Resolve the residue-input directory actually used by the model run."""
    scenario = _read_scenario_yaml(scenario_dir)
    if scenario is not None:
        residue_dir = scenario.get("residue_directory")
        if residue_dir:
            return Path(residue_dir)
    return RESIDUE_DIR_FALLBACK


def _panel_b_scenario_label(spec, scenario_dir):
    """Use actual run settings in labels where available; never invent cap/rate."""
    label = spec["label"]
    scenario = _read_scenario_yaml(scenario_dir)
    if scenario is None:
        return label

    settings = scenario.get("settings", {})
    if not isinstance(settings, dict):
        return label

    key = spec["key"]
    if key == "residue_competition":
        if not settings.get("proportional_residue_rationing", False):
            raise ValueError(
                f"{scenario_dir}: configured as residue competition, but "
                "scenario.yaml does not enable proportional_residue_rationing."
            )
    elif key == "application_cap":
        cap = settings.get("max_cumulative_application_t_ha")
        if cap is None or float(cap) <= 0:
            raise ValueError(
                f"{scenario_dir}: no positive cumulative-application cap "
                "recorded in scenario.yaml."
            )
        label = rf"Application cap ({float(cap):g} t ha$^{{-1}}$)"
        if not np.isclose(float(cap), 50.0):
            warnings.warn(
                f"{scenario_dir.name} applies {cap} t/ha; manuscript "
                "describes 50 t/ha. The figure will label the actual value.",
                UserWarning,
            )
    elif key in ("diffusion_low", "diffusion_high"):
        rate = settings.get("diffusion_rate_limit_per_year")
        if rate is not None:
            label = rf"Diffusion ({100 * float(rate):g}% yr$^{{-1}}$)"
    return label


def build_panel_b_data(
    base_dir=BASE_DIR,
    start_year=PANEL_B_START_YEAR,
    end_year=PANEL_B_END_YEAR,
    include_diffusion=PANEL_B_INCLUDE_DIFFUSION,
):
    """Load annual global mitigation and cumulative diagnostics for sensitivity runs.

    Three required runs: reference, proportional rationing and application cap.
    Diffusion low/high are optional and omitted when their output is absent.
    Scenario names are defined in PANEL_B_SCENARIOS.
    """
    series = []
    reference_years = None
    reference_dt = None

    for spec in PANEL_B_SCENARIOS:
        is_diffusion = spec["key"].startswith("diffusion_")
        if is_diffusion and not include_diffusion:
            continue

        scenario_dir = Path(base_dir) / spec["directory"]
        preferred_file = scenario_dir / f"{REALIZED_MITIGATION_VAR}.nc"
        fallback_file = scenario_dir / f"{FALLBACK_COMPONENTS[0][0]}.nc"
        if not preferred_file.is_file() and not fallback_file.is_file():
            message = (
                f"No mitigation output for panel B scenario '{spec['key']}' "
                f"at {scenario_dir}. Expected {preferred_file.name} or "
                f"{fallback_file.name}."
            )
            if spec["required"]:
                raise FileNotFoundError(message)
            warnings.warn(message + " Optional diffusion curve skipped.", RuntimeWarning)
            continue

        years, annual_r26, dt, source = load_mitigation_timeseries_r26(
            scenario_dir, start_year=start_year, end_year=end_year
        )
        if reference_years is None:
            reference_years, reference_dt = years, dt
        elif not (
            np.array_equal(years, reference_years)
            and np.allclose(dt, reference_dt)
        ):
            raise ValueError(
                f"Time axes of {scenario_dir.name} and the reference scenario "
                "do not match, so their annual mitigation cannot be compared."
            )

        annual_global = np.sum(annual_r26, axis=1)
        cumulative_global = float(np.sum(annual_global * dt))
        series.append(
            {
                "key": spec["key"],
                "label": _panel_b_scenario_label(spec, scenario_dir),
                "directory": str(scenario_dir),
                "source": source,
                "color": spec["color"],
                "linestyle": spec["linestyle"],
                "years": years,
                "dt": dt,
                "annual_global": annual_global,
                "cumulative_global": cumulative_global,
            }
        )

    if not series or series[0]["key"] != "main":
        raise RuntimeError("Panel B requires a successfully loaded main scenario.")
    return {"years": reference_years, "dt": reference_dt, "series": series}


def _proportional_rationing_enabled(scenario_dir):
    scenario = _read_scenario_yaml(scenario_dir)

    if scenario is None:
        return None

    settings = scenario.get("settings", {})

    if not isinstance(settings, dict):
        return None

    return bool(settings.get("proportional_residue_rationing", False))


def load_residue_supply_potential_r26(scenario_dir, year=YEAR):
    """
    Load ex-ante residue supply potential by R26 region.

    This reproduces the supply basis used by prepare_residue_supply:
        ResPotCurveDM[:R26, :biochar feedstocks, :cost bins]

    The final RCCPlus element is excluded because the model uses
    len(RCCPlus)-1 actual cost bins.

    Multiplication by timestep width is intentionally omitted because panel c
    plots regional shares. A common timestep multiplier cancels in those
    shares.

    IMPORTANT:
    The plotted x-axis is the ex-ante residue resource endowment, before any
    proportional residue rationing. This is deliberate: it tests whether the
    geography of economic biochar deployment follows the geography of the
    underlying residue resource.
    """
    scenario_dir = Path(scenario_dir)
    residue_dir = _resolve_residue_directory(scenario_dir)

    nc_file = _resolve_case_insensitive_file(
        residue_dir,
        RESIDUE_POTENTIAL_VAR,
    )

    if not nc_file.exists():
        raise FileNotFoundError(
            "Could not find the residue-potential input needed for panel c:\n"
            f"{nc_file}\n\n"
            "The module first reads `residue_directory` from scenario.yaml and "
            "then falls back to RESIDUE_DIR_FALLBACK."
        )

    with xr.open_dataset(nc_file) as ds:
        if RESIDUE_POTENTIAL_VAR not in ds:
            raise KeyError(
                f"Variable '{RESIDUE_POTENTIAL_VAR}' not found in {nc_file}. "
                f"Available variables: {list(ds.data_vars)}"
            )

        da = ds[RESIDUE_POTENTIAL_VAR]

        if "time" in da.dims:
            years = _time_years(da)
            matches = np.where(years == int(year))[0]

            if len(matches) == 0:
                raise ValueError(
                    f"Year {year} not found in {nc_file}. "
                    f"Available years: {years.tolist()}"
                )

            da = da.isel(time=int(matches[0]))

        values = np.asarray(da.values, dtype=np.float64)

    expected_size = (
        len(Region26.coords)
        * len(Residue.coords)
        * len(RCCPlus.coords)
    )

    if values.size != expected_size:
        raise ValueError(
            f"Unexpected {RESIDUE_POTENTIAL_VAR} size in {nc_file}: "
            f"{values.size}; expected {expected_size} = "
            f"{len(Region26.coords)} regions × {len(Residue.coords)} residues × "
            f"{len(RCCPlus.coords)} RCCPlus bins."
        )

    values = values.reshape(
        len(Region26.coords),
        len(Residue.coords),
        len(RCCPlus.coords),
    )

    cost_bins = len(RCCPlus.coords) - 1

    values = values[
        :len(IMAGE_REGIONS),
        :len(BiocharFeedstock.coords),
        :cost_bins,
    ]

    values = np.where(
        np.isfinite(values) & (values > 0.0),
        values,
        0.0,
    )

    return np.nansum(values, axis=(1, 2)), residue_dir


def load_residue_use_r26(scenario_dir, year=YEAR):
    """
    Load realised residue dry-matter use if BiocharResidueDMUsed is available.

    This is not used on either panel-c axis. It is used only to print the
    fraction of each R26 region's ex-ante residue potential that is mobilised.
    """
    scenario_dir = Path(scenario_dir)
    nc_file = _resolve_case_insensitive_file(
        scenario_dir,
        RESIDUE_USE_VAR,
    )

    if not nc_file.exists():
        return None

    with xr.open_dataset(nc_file) as ds:
        if RESIDUE_USE_VAR not in ds:
            return None

        da = ds[RESIDUE_USE_VAR]

        if "time" in da.dims:
            years = _time_years(da)
            matches = np.where(years == int(year))[0]

            if len(matches) == 0:
                return None

            da = da.isel(time=int(matches[0]))

        if "region" in da.dims:
            other_dims = [dim for dim in da.dims if dim != "region"]

            if other_dims:
                da = da.sum(dim=other_dims, skipna=True)

            values = np.asarray(
                da.values,
                dtype=np.float64,
            ).squeeze()

            return values[:len(IMAGE_REGIONS)]

        values = np.asarray(da.values, dtype=np.float64)

    nreg = len(IMAGE_REGIONS)

    if values.size % nreg != 0:
        return None

    return np.nansum(
        values.reshape(nreg, -1),
        axis=1,
    )


def _share_percent(values):
    values = np.asarray(values, dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)

    total = float(np.nansum(values))

    if total <= 0.0:
        return np.zeros_like(values, dtype=np.float64)

    return 100.0 * values / total


def _spearman(x, y):
    """
    Descriptive Spearman rank correlation. No p-value is reported because this
    panel is descriptive rather than an inferential regional sample analysis.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    valid = np.isfinite(x) & np.isfinite(y)

    if np.sum(valid) < 3:
        return np.nan

    if spearmanr is not None:
        result = spearmanr(x[valid], y[valid])
        return float(result.statistic)

    # Dependency-free fallback: Pearson correlation of average ranks.
    def average_ranks(values):
        values = np.asarray(values, dtype=np.float64)
        order = np.argsort(values, kind="mergesort")
        sorted_values = values[order]
        ranks = np.empty(len(values), dtype=np.float64)

        i = 0
        while i < len(values):
            j = i + 1

            while (
                j < len(values)
                and sorted_values[j] == sorted_values[i]
            ):
                j += 1

            average_rank = 0.5 * ((i + 1) + j)
            ranks[order[i:j]] = average_rank
            i = j

        return ranks

    rx = average_ranks(x[valid])
    ry = average_ranks(y[valid])

    return float(np.corrcoef(rx, ry)[0, 1])


def _r26_to_r10_colour():
    mapping = {}

    for r10_index, (r10_name, members) in enumerate(R10.items()):
        colour = R10_COLORS[r10_index]

        for code in members:
            mapping[str(code)] = colour

    return mapping


def build_panel_c_data(
    base_dir=BASE_DIR,
    prices=PANEL_C_PRICES,
    year=YEAR,
):
    """
    Build the four R26 scatter facets.

    x = regional share of global residue supply potential
    y = regional share of global realised biochar mitigation

    The x-axis is deliberately ex ante resource availability, before allocation
    and before any proportional rationing. This makes the 1:1 line interpretable:
        deployment share == underlying resource-endowment share.
    """
    output = {}

    for price in prices:
        scenario_dir = _scenario_path(
            base_dir,
            MAIN_TEMPLATE,
            price,
        )

        mitigation_r26, dt, source = load_annual_mitigation_r26(
            scenario_dir,
            year=year,
        )

        supply_r26, residue_dir = load_residue_supply_potential_r26(
            scenario_dir,
            year=year,
        )

        supply_share = _share_percent(supply_r26)
        mitigation_share = _share_percent(
            np.maximum(mitigation_r26, 0.0)
        )

        residue_used = load_residue_use_r26(
            scenario_dir,
            year=year,
        )

        mobilised_fraction = None

        if residue_used is not None:
            # prepare_residue_supply multiplies ResPotCurveDM by dt. Reproduce
            # that timestep denominator for this diagnostic ratio.
            supply_step = supply_r26 * dt

            mobilised_fraction = np.divide(
                residue_used,
                supply_step,
                out=np.full_like(
                    supply_step,
                    np.nan,
                    dtype=np.float64,
                ),
                where=supply_step > 0.0,
            )

        output[int(price)] = {
            "supply_r26": supply_r26,
            "mitigation_r26": mitigation_r26,
            "supply_share": supply_share,
            "mitigation_share": mitigation_share,
            "spearman_rho": _spearman(
                supply_r26,
                mitigation_r26,
            ),
            "mobilised_fraction": mobilised_fraction,
            "mitigation_source": source,
            "residue_directory": residue_dir,
            "proportional_rationing": _proportional_rationing_enabled(
                scenario_dir
            ),
        }

    return output


# =============================================================================
# Panel plotting
# =============================================================================

def plot_panel_a(ax, price_data, prices=CARBON_PRICES, year=YEAR):
    """Panel a: global price response for main and stylised biochar."""
    prices = np.asarray(prices, dtype=np.float64)
    # Keep the original graphic's zero-price origin; it is an implied zero,
    # not a result loaded from the configured c0 model run.
    x = np.concatenate(([0.0], prices))
    main = np.concatenate(([0.0], price_data["main_global"]))
    stylised = np.concatenate(([0.0], price_data["stylised_global"]))

    ax.plot(x, main, color=COLOR_MAIN, linewidth=2.3, marker="o",
            markersize=4.5, label="Main", zorder=4)
    ax.plot(x, stylised, color=COLOR_STYLISED, linewidth=2.1, marker="o",
            markersize=4.5, label="Stylised biochar", zorder=4)

    ax.set_title(
        _panel_title("a", f"Global carbon price response, {year}"),
        loc="left", pad=7, fontweight="normal"
    )
    ax.set_xlabel(
        r"Carbon price ($\mathrm{US}\$_{2010}\,\mathrm{tCO_2}^{-1}$)"
    )
    ax.set_ylabel(r"Annual mitigation (GtCO$_2$-eq yr$^{-1}$)")
    ax.set_xlim(0, max(prices))
    ax.set_xticks([0, *prices])
    _style_axis(ax)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.245),
        ncol=2, frameon=False, columnspacing=1.4,
        handlelength=2.0, handletextpad=0.5,
        fontsize=9.5, borderaxespad=0.0,
    )


def plot_panel_b(ax, panel_b_data, price=PANEL_B_CARBON_PRICE):
    """Panel b: annual mitigation sensitivity trajectories (2030-2100)."""
    for item in panel_b_data["series"]:
        highlight = item["key"] == "main"
        ax.plot(
            item["years"], item["annual_global"],
            color=item["color"], linestyle=item["linestyle"],
            linewidth=2.5 if highlight else 1.9,
            marker="o" if highlight else None,
            markevery=3, markersize=3.6,
            label=item["label"],
            zorder=5 if highlight else 3,
        )

    ax.set_title(
        _panel_title("b", rf"Mitigation pathways at {price} $\mathrm{{US}}\$_{{2010}}\,\mathrm{{tCO_2}}^{{-1}}$"),
        loc="left", pad=7, fontweight="normal"
    )
    ax.set_xlabel("Year")
    ax.set_ylabel(r"Annual mitigation (GtCO$_2$-eq yr$^{-1}$)")
    ax.set_xlim(int(panel_b_data["years"][0]), int(panel_b_data["years"][-1]))
    years = panel_b_data["years"]
    desired_ticks = (2030, 2050, 2070, 2090, 2100)
    ax.set_xticks([t for t in desired_ticks if years[0] <= t <= years[-1]])
    _style_axis(ax)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.24),
        ncol=2, frameon=False, columnspacing=1.1,
        labelspacing=0.65, handlelength=2.2, handletextpad=0.5,
        fontsize=8.7, borderaxespad=0.0,
    )


def _panel_c_notable_indices(x, y, n=4):
    """
    Select the most notable R26 points in each facet.

    Score combines:
    - absolute deviation from the 1:1 line (disproportionate deployment)
    - deployment share itself (substantive importance)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.size == 0:
        return np.asarray([], dtype=int)

    score = 0.70 * np.abs(y - x) + 0.30 * y
    order = np.argsort(-score)
    return order[:min(n, len(order))]


def _annotate_panel_c_labels(ax, x, y, indices, axis_max, price):
    """
    Place labels with light collision avoidance and keep them inside the axes.
    """
    placed_positions = []
    candidate_offsets = [
        (6, 7), (6, -8), (-6, 7), (-6, -8),
        (8, 0), (-8, 0), (0, 9), (0, -9),
        (12, 7), (12, -8), (-12, 7), (-12, -8),
    ]

    for idx in indices:
        px = float(x[idx])
        py = float(y[idx])
        region_code = str(IMAGE_REGIONS[idx])
        label = R26_DISPLAY.get(region_code, region_code)

        manual_offset = PANEL_C_MANUAL_LABEL_OFFSETS.get(
            (int(price), region_code)
        )

        if manual_offset is not None:
            dx, dy = manual_offset
            ha = "right" if dx < 0 else "left"
            va = "top" if dy < 0 else "bottom"

            ax.annotate(
                label,
                xy=(px, py),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8.0,
                ha=ha,
                va=va,
                color="0.22",
                zorder=5,
                annotation_clip=True,
                clip_on=True,
                arrowprops=dict(
                    arrowstyle="-",
                    color="0.50",
                    lw=0.5,
                    shrinkA=0,
                    shrinkB=2,
                ),
            )

            tx = px + dx * axis_max / 230.0
            ty = py + dy * axis_max / 230.0
            placed_positions.append((tx, ty))
            continue

        preferred = []
        if px > 0.82 * axis_max:
            preferred.extend([(-8, 7), (-8, -8), (-12, 7), (-12, -8), (-8, 0)])
        elif px < 0.18 * axis_max:
            preferred.extend([(8, 7), (8, -8), (12, 7), (12, -8), (8, 0)])

        if py > 0.82 * axis_max:
            preferred = [(dx, -abs(dy)) for dx, dy in preferred] + preferred
        elif py < 0.18 * axis_max:
            preferred = [(dx, abs(dy)) for dx, dy in preferred] + preferred

        offsets = preferred + [item for item in candidate_offsets if item not in preferred]

        chosen = None
        best_penalty = None

        for dx, dy in offsets:
            tx = px + dx * axis_max / 230.0
            ty = py + dy * axis_max / 230.0

            penalty = 0.0
            if tx < 0.8:
                penalty += 100 + (0.8 - tx) * 20
            if tx > axis_max - 0.8:
                penalty += 100 + (tx - (axis_max - 0.8)) * 20
            if ty < 0.8:
                penalty += 100 + (0.8 - ty) * 20
            if ty > axis_max - 0.8:
                penalty += 100 + (ty - (axis_max - 0.8)) * 20

            for ox, oy in placed_positions:
                dist2 = (tx - ox) ** 2 + (ty - oy) ** 2
                if dist2 < (axis_max / 11.5) ** 2:
                    penalty += 200 / max(dist2, 1e-6)

            if best_penalty is None or penalty < best_penalty:
                best_penalty = penalty
                chosen = (dx, dy, tx, ty)

        dx, dy, tx, ty = chosen
        placed_positions.append((tx, ty))

        ha = "right" if dx < 0 else "left"
        va = "top" if dy < 0 else "bottom"

        ax.annotate(
            label,
            xy=(px, py),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.0,
            ha=ha,
            va=va,
            color="0.22",
            zorder=5,
            annotation_clip=True,
            clip_on=True,
            arrowprops=dict(
                arrowstyle="-",
                color="0.50",
                lw=0.5,
                shrinkA=0,
                shrinkB=2,
            ),
        )


def plot_panel_c_facets(
    fig,
    subplot_spec,
    panel_c_data,
    prices=PANEL_C_PRICES,
    year=YEAR,
):
    """
    Plot panel c as four R26 scatter facets.

    A nested GridSpec is used so the outer width of panel c is exactly the
    same as the combined width occupied by panels a and b.
    """
    # Invisible framing axis used only for the common panel title and axis labels.
    frame_ax = fig.add_subplot(subplot_spec)
    frame_ax.set_axis_off()
    frame_ax.set_zorder(0)

    frame_ax.text(
        0.0,
        1.10,
        _panel_title(
            "c",
            f"Regional residue availability and mitigation deployment, {year}",
        ),
        transform=frame_ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
    )

    frame_ax.text(
        0.5,
        -0.16,
        "Share of global residue supply (%)",
        transform=frame_ax.transAxes,
        ha="center",
        va="top",
        fontsize=10.8,
    )

    frame_ax.text(
        -0.032,
        0.50,
        "Share of global biochar mitigation (%)",
        transform=frame_ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=10.8,
    )

    subgs = subplot_spec.subgridspec(
        1,
        len(prices),
        wspace=0.17,
    )

    axes = np.asarray(
        [fig.add_subplot(subgs[0, i]) for i in range(len(prices))]
    )

    max_share = 0.0
    for price in prices:
        data = panel_c_data[int(price)]
        max_share = max(
            max_share,
            float(np.nanmax(data["supply_share"])),
            float(np.nanmax(data["mitigation_share"])),
        )

    axis_max = max(
        10.0,
        5.0 * np.ceil((1.10 * max_share) / 5.0),
    )

    point_colours = _r26_to_r10_colour()

    for ax, price in zip(axes, prices):
        data = panel_c_data[int(price)]
        x = data["supply_share"]
        y = data["mitigation_share"]

        ax.plot(
            [0.0, axis_max],
            [0.0, axis_max],
            linestyle="--",
            linewidth=1.0,
            color="0.45",
            zorder=1,
        )

        for i, code in enumerate(IMAGE_REGIONS):
            ax.scatter(
                x[i],
                y[i],
                s=40,
                color=point_colours.get(str(code), "0.45"),
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

        notable = _panel_c_notable_indices(
            x,
            y,
            n=PANEL_C_LABEL_TOP_DEVIATIONS,
        )

        _annotate_panel_c_labels(
            ax,
            x,
            y,
            notable,
            axis_max,
            price,
        )

        ax.set_title(
            rf"{price} \$ tCO$_2^{{-1}}$",
            fontsize=10.5,
            pad=6,
        )

        if PANEL_C_ANNOTATE_SPEARMAN:
            rho = data["spearman_rho"]

            ax.text(
                0.04,
                0.95,
                (
                    f"Spearman ρ = {rho:.2f}"
                    if np.isfinite(rho)
                    else "Spearman ρ = NA"
                ),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.2,
            )

        ax.set_xlim(0.0, axis_max)
        ax.set_ylim(0.0, axis_max)
        ax.set_aspect("equal", adjustable="box")

        _style_axis(ax)

    # With shared scales, show y tick labels only on the left-most facet.
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)

    return axes


# =============================================================================
# Diagnostics
# =============================================================================

def print_diagnostics(
    price_data,
    panel_b_data,
    panel_c_data,
    prices=CARBON_PRICES,
):
    print("\n" + "=" * 118)
    print("FIGURE 1 DIAGNOSTICS")
    print("=" * 118)

    print("\nPanel A — annualisation and source checks:")
    for item in price_data["timestep_diagnostics"]:
        print(
            f"  c{item['price']:<3d}: dt={item['dt']:.1f} yr | "
            f"main={item['main_source']} | "
            f"stylised={item['stylised_source']}"
        )

    print("\nGlobal annual mitigation in 2100 (GtCO2-eq/yr):")
    print(f"{'Carbon price':>14s}{'Main':>12s}{'Stylised':>13s}")
    for i, price in enumerate(prices):
        print(
            f"{price:14.0f}"
            f"{price_data['main_global'][i]:12.3f}"
            f"{price_data['stylised_global'][i]:13.3f}"
        )

    print("\nPanel B — $400/tCO2 sensitivity pathways:")
    print(
        "  Cumulative values sum timestep totals; individual timepoints "
        "are annualised."
    )
    print(
        f"{'Scenario':<40s} {'2100 (Gt/yr)':>14s} "
        f"{'Change (%)':>12s} {'Cumulative (Gt)':>16s} "
        f"{'Change (%)':>12s}"
    )
    reference = panel_b_data["series"][0]
    ref_final = float(reference["annual_global"][-1])
    ref_cumulative = float(reference["cumulative_global"])
    for item in panel_b_data["series"]:
        final = float(item["annual_global"][-1])
        cumulative = float(item["cumulative_global"])
        final_pc = 100 * (final / ref_final - 1) if ref_final else float("nan")
        cumulative_pc = (
            100 * (cumulative / ref_cumulative - 1)
            if ref_cumulative else float("nan")
        )
        print(
            f"{item['key']:<40s} {final:14.3f} {final_pc:12.1f} "
            f"{cumulative:16.2f} {cumulative_pc:12.1f}"
        )
        print(f"    {item['label']} | {item['source']} | {item['directory']}")


    print("\nPanel-c R26 residue-supply/deployment diagnostics:")
    print(
        f"{'Price':>7s} {'rho':>7s} "
        f"{'most over-represented in mitigation':<48s} "
        f"{'most under-represented in mitigation':<48s}"
    )

    for price in PANEL_C_PRICES:
        data = panel_c_data[int(price)]

        x = data["supply_share"]
        y = data["mitigation_share"]
        deviation = y - x

        over = np.argsort(-deviation)[:3]
        under = np.argsort(deviation)[:3]

        over_text = ", ".join(
            f"{R26_DISPLAY.get(str(IMAGE_REGIONS[i]), str(IMAGE_REGIONS[i]))} "
            f"({deviation[i]:+.1f} pp)"
            for i in over
        )

        under_text = ", ".join(
            f"{R26_DISPLAY.get(str(IMAGE_REGIONS[i]), str(IMAGE_REGIONS[i]))} "
            f"({deviation[i]:+.1f} pp)"
            for i in under
        )

        print(
            f"{price:7.0f} {data['spearman_rho']:7.2f} "
            f"{over_text:<48s} {under_text:<48s}"
        )

        if data["proportional_rationing"]:
            print(
                "         NOTE: proportional residue rationing is enabled. "
                "The plotted x-axis intentionally uses ex-ante ResPotCurveDM "
                "resource potential, before rationing."
            )

        mobilised = data["mobilised_fraction"]

        if mobilised is not None:
            valid = np.isfinite(mobilised)

            if np.any(valid):
                order = np.argsort(
                    -np.where(
                        valid,
                        mobilised,
                        -np.inf,
                    )
                )[:5]

                mobilised_text = ", ".join(
                    f"{R26_DISPLAY.get(str(IMAGE_REGIONS[i]), str(IMAGE_REGIONS[i]))} "
                    f"({100.0 * mobilised[i]:.1f}%)"
                    for i in order
                    if np.isfinite(mobilised[i])
                )

                print(
                    "         Highest fractions of regional residue potential "
                    f"mobilised: {mobilised_text}"
                )

    print("\nPanel-c residue input directories:")
    for price in PANEL_C_PRICES:
        data = panel_c_data[int(price)]
        print(
            f"  c{price}: {data['residue_directory']}"
        )

    print("\n" + "-" * 118)
    print("PANEL C: RESULTS-SECTION QUANTITATIVE SUMMARY")
    print("-" * 118)

    print(
        "\nFor each carbon price, the table reports the R26 concentration of "
        "deployment, its correspondence with residue availability, and the "
        "overall mismatch between the two regional distributions."
    )
    print(
        f"{'Price':>7s} {'Global mitig.':>14s} {'Top-3 deploy.':>14s} "
        f"{'Spearman rho':>13s} {'Mismatch index':>15s}"
    )
    print(
        f"{'$/tCO2':>7s} {'GtCO2-eq/yr':>14s} {'share (%)':>14s} "
        f"{'':>13s} {'pp':>15s}"
    )

    for price in PANEL_C_PRICES:
        data = panel_c_data[int(price)]
        x = np.asarray(data["supply_share"], dtype=np.float64)
        y = np.asarray(data["mitigation_share"], dtype=np.float64)
        mitigation = np.asarray(data["mitigation_r26"], dtype=np.float64)

        global_mitigation = float(np.nansum(mitigation))
        top3_share = float(np.nansum(np.sort(y)[-3:]))
        mismatch_index = 0.5 * float(np.nansum(np.abs(y - x)))

        print(
            f"{price:7.0f} {global_mitigation:14.3f} {top3_share:14.1f} "
            f"{data['spearman_rho']:13.2f} {mismatch_index:15.1f}"
        )

        top_deploy = np.argsort(-y)[:4]
        top_supply = np.argsort(-x)[:4]
        deviation = y - x
        over = np.argsort(-deviation)[:3]
        under = np.argsort(deviation)[:3]

        deploy_text = ", ".join(
            f"{R26_DISPLAY.get(str(IMAGE_REGIONS[i]), str(IMAGE_REGIONS[i]))} "
            f"{y[i]:.1f}% (supply {x[i]:.1f}%)"
            for i in top_deploy
        )
        supply_text = ", ".join(
            f"{R26_DISPLAY.get(str(IMAGE_REGIONS[i]), str(IMAGE_REGIONS[i]))} "
            f"{x[i]:.1f}% (mitigation {y[i]:.1f}%)"
            for i in top_supply
        )
        over_text = ", ".join(
            f"{R26_DISPLAY.get(str(IMAGE_REGIONS[i]), str(IMAGE_REGIONS[i]))} "
            f"{deviation[i]:+.1f} pp"
            for i in over
        )
        under_text = ", ".join(
            f"{R26_DISPLAY.get(str(IMAGE_REGIONS[i]), str(IMAGE_REGIONS[i]))} "
            f"{deviation[i]:+.1f} pp"
            for i in under
        )

        print(f"         Largest mitigation shares: {deploy_text}")
        print(f"         Largest residue shares:    {supply_text}")
        print(f"         Most over-represented:      {over_text}")
        print(f"         Most under-represented:     {under_text}")

        mobilised = data["mobilised_fraction"]
        if mobilised is not None:
            valid = np.isfinite(mobilised)
            if np.any(valid):
                weighted_global_mobilised = np.nansum(
                    np.asarray(data["supply_r26"])[valid]
                    * np.asarray(mobilised)[valid]
                ) / np.nansum(np.asarray(data["supply_r26"])[valid])

                print(
                    "         Supply-weighted residue mobilisation: "
                    f"{100.0 * weighted_global_mobilised:.1f}%"
                )

    # Explicitly quantify how much the geography converges toward the residue
    # endowment as carbon prices rise.
    first = panel_c_data[int(PANEL_C_PRICES[0])]
    last = panel_c_data[int(PANEL_C_PRICES[-1])]

    mismatch_first = 0.5 * float(
        np.nansum(
            np.abs(
                np.asarray(first["mitigation_share"])
                - np.asarray(first["supply_share"])
            )
        )
    )
    mismatch_last = 0.5 * float(
        np.nansum(
            np.abs(
                np.asarray(last["mitigation_share"])
                - np.asarray(last["supply_share"])
            )
        )
    )

    print("\nChange across the carbon-price range:")
    print(
        f"  Spearman rho: {first['spearman_rho']:.2f} at c{PANEL_C_PRICES[0]} "
        f"to {last['spearman_rho']:.2f} at c{PANEL_C_PRICES[-1]}."
    )
    print(
        f"  Distribution mismatch index: {mismatch_first:.1f} pp at "
        f"c{PANEL_C_PRICES[0]} to {mismatch_last:.1f} pp at "
        f"c{PANEL_C_PRICES[-1]}."
    )



# =============================================================================
# Main figure
# =============================================================================

def plot_figure1(
    base_dir=BASE_DIR,
    prices=CARBON_PRICES,
    year=YEAR,
    panel_c_prices=PANEL_C_PRICES,
    panel_b_start_year=PANEL_B_START_YEAR,
    panel_b_include_diffusion=PANEL_B_INCLUDE_DIFFUSION,
    output_dir=OUTPUT_DIR,
    output_stem=OUTPUT_STEM,
    show=SHOW_FIGURE,
):
    _set_publication_style()

    price_data = build_carbon_price_data(
        base_dir=base_dir, prices=prices, year=year
    )
    panel_b_data = build_panel_b_data(
        base_dir=base_dir,
        start_year=panel_b_start_year,
        end_year=year,
        include_diffusion=panel_b_include_diffusion,
    )
    panel_c_data = build_panel_c_data(
        base_dir=base_dir, prices=panel_c_prices, year=year
    )

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(
        2, 2, height_ratios=(1.0, 1.22), width_ratios=(1.0, 1.0),
        hspace=0.86, wspace=0.24,
    )

    # Panel a: global response to carbon price (left).
    ax_a = fig.add_subplot(gs[0, 0])
    # Panel b: temporal sensitivity pathways at $400/tCO2 (right).
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_a(ax_a, price_data=price_data, prices=prices, year=year)
    plot_panel_b(ax_b, panel_b_data=panel_b_data, price=PANEL_B_CARBON_PRICE)

    # Panel c: four regional residue/deployment scatter facets (full bottom row).
    axes_c = plot_panel_c_facets(
        fig, gs[1, :], panel_c_data=panel_c_data,
        prices=panel_c_prices, year=year,
    )

    # Both upper panels show annual mitigation in the same units.
    # Share a y-scale for direct comparison of magnitudes.
    annual_values = [
        price_data["main_global"],
        price_data["stylised_global"],
    ] + [item["annual_global"] for item in panel_b_data["series"]]
    top_ymax = 1.08 * float(np.nanmax(np.concatenate(annual_values)))
    if not np.isfinite(top_ymax) or top_ymax <= 0:
        top_ymax = 1.0
    ax_a.set_ylim(0.0, top_ymax)
    ax_b.set_ylim(0.0, top_ymax)

    fig.subplots_adjust(left=0.085, right=0.975, top=0.965, bottom=0.075)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{output_stem}.png"
    pdf_path = output_dir / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    print_diagnostics(
        price_data=price_data,
        panel_b_data=panel_b_data,
        panel_c_data=panel_c_data,
        prices=prices,
    )

    print("\nSaved:")
    print(png_path)
    print(pdf_path)
    if show:
        plt.show()

    return {
        "fig": fig,
        "axes": {"a": ax_a, "b": ax_b, "c": axes_c},
        "price_data": price_data,
        "panel_b_data": panel_b_data,
        "panel_c_data": panel_c_data,
        "png_path": png_path,
        "pdf_path": pdf_path,
    }


if __name__ == "__main__":
    outputs = plot_figure1()

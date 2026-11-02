"""Standalone, read-only audit of IMAGE/TIMER biochar sensitivity outputs.

Run on the same machine that holds the NetCDF result directories::

    python biochar_sensitivity_audit.py \
        --base-dir "X:\\user\\orhalem\\TIMER_CDR\\timer-prism\\cdr_output"

Requirements::

    pip install numpy pandas xarray netCDF4 matplotlib pyyaml

IMPORTANT SCIENTIFIC CONVENTIONS
--------------------------------
* BiocharRealisedMitigation and its pathway components are 5-YEAR TIMESTEP
  TOTALS in tCO2-eq. Annual values = timestep total / timestep width.
  Cumulative values = SUM of timestep totals (never multiply by dt).
* Realised cost/revenue files are timestep monetary totals in USD 2025.
  Cost intensities = SUM costs / SUM realised mitigation, not mean of ratios.
* Net resource cost excludes BiocharRealisedCostTransportCarbon in the source
  implementation. The latter is reported separately as a policy carbon penalty.
* This audits standalone PRISM biochar output, NOT IMAGE-Land avoided land-use
  emissions or cropland sparing. No causal attribution is implied by SSP comparisons.
* Model completion cannot be proven just from files; PASS means a complete and
  numerically plausible export set, not certified successful simulation.
* No import from 'timer' or 'prism' is needed, and NO scenario files are changed.

Files created under --out-dir:
    run_health.csv, issues.csv, scenario_summary.csv, comparisons.csv,
    global_timeseries.csv, pathway_decomposition.csv, regional_comparisons.csv,
    spatial_diagnostics.csv, agronomic_checks.csv, findings.md,
    01_mitigation_trajectories.png, 02_sensitivity_effects.png,
    03_economic_intensities.png, 04_mitigation_composition.png,
    05_regional_mitigation.png, 06_spatial_overlap.png (if spatial available).

The default --base-dir points to the output root in the user's screenshots.
Optional --skip-spatial makes a substantially faster first-pass audit.
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Safe in terminals, remote sessions and batch jobs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

try:
    import yaml
except ImportError:
    yaml = None


# -----------------------------------------------------------------------------
# CONFIGURATION: these directory names are the names shown in the screenshots.
# -----------------------------------------------------------------------------
BASE_DIR = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output")
REFERENCE = "biochar_c400"
FIRST_YEAR = 2030
LAST_YEAR = 2100
DEFAULT_STEP = 5

@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    group: str
    expected: dict[str, Any] = field(default_factory=dict)
    note: str = ""

SCENARIOS = (
    Scenario(REFERENCE, "Reference (SSP2)", "reference",
             {"land_scenario": "SSP2"}),
    Scenario("biochar_c400_prop", "Residue rationing", "constraints",
             {"proportional_residue_rationing": True},
             "Shared residue resources with modern bioenergy; exogenous rationing."),
    Scenario("biochar_c400_cap", "Application cap", "constraints",
             {"max_cumulative_application_t_ha": "non_null"},
             "Check configured threshold; a 40 or 50 t/ha cap permits two 20 t/ha doses."),
    Scenario("biochar_c400_low_yield", "Low crop-yield response", "agronomy",
             {"yield_response_scenario": "low"}),
    Scenario("biochar_c400_high_yield", "High crop-yield response", "agronomy",
             {"yield_response_scenario": "high"}),
    Scenario("biochar_c400_aging", "Agronomic aging", "agronomy",
             {"agronomic_response_mode": "aging_within_window"},
             "Tests within-window aging in PRISM; not a coupled IMAGE-Land test."),
    Scenario("biochar_c400_exp_carbon", "Exponential persistence", "biophysics",
             {"carbon_persistence_model": "azzi_exponential"}),
    Scenario("biochar_spp1", "SSP1", "context",
             {"land_scenario": "SSP1"},
             "Land context, land rents, residue supply and cropland change together."),
    Scenario("biochar_ssp3", "SSP3", "context",
             {"land_scenario": "SSP3"},
             "Land context, land rents, residue supply and cropland change together."),
)
SCENARIO_LOOKUP = {s.name: s for s in SCENARIOS}

R26 = (
    "CAN", "USA", "MEX", "RCAM", "BRA", "RSAM", "NAF", "WAF", "EAF", "SAF",
    "WEU", "CEU", "TUR", "UKR", "STAN", "RUS", "ME", "INDIA", "KOR", "CHN",
    "SEAS", "INDO", "JAP", "OCE", "RSAS", "RSAF",
)
R10 = {
    "North America": ("CAN", "USA"),
    "Latin America": ("MEX", "RCAM", "BRA", "RSAM"),
    "Europe": ("WEU", "CEU", "TUR"),
    "Eurasia": ("UKR", "STAN", "RUS"),
    "Middle East & North Africa": ("NAF", "ME"),
    "Sub-Saharan Africa": ("WAF", "EAF", "SAF", "RSAF"),
    "South Asia": ("INDIA", "RSAS"),
    "East Asia": ("CHN", "KOR", "JAP"),
    "Southeast Asia": ("SEAS", "INDO"),
    "Oceania": ("OCE",),
}

# Required for a trustworthy realised-economics comparison.
COSTS = (
    "BiocharRealisedCostFeedstock",
    "BiocharRealisedCostProduction",
    "BiocharRealisedCostApplication",
    "BiocharRealisedCostTransport",
)
REVENUES = (
    "BiocharRealisedRevenueYield",
    "BiocharRealisedRevenueBioenergy",
)
COST_PENALTY = "BiocharRealisedCostTransportCarbon"
NET_COST = "BiocharRealisedNetCost"
MITIGATION = "BiocharRealisedMitigation"
BIOCHAR = "BiocharRealisedBiocharProduction"
AREA = "BiocharAreaTech"
CAPACITY = "biochar_capacity"
RESIDUE = "BiocharResidueDMUsed"
COMPONENTS = {
    "Field biochar CDR": "BiocharFieldCDR",
    "Syngas CCS": "BiocharSyngasCCS",
    "Bio-oil fossil displacement": "BiocharFossilMitig",
    "Avoided N2O": "BiocharN2OMitig",
    "Avoided CH4": "BiocharCH4Mitig",
}

REQUIRED_VARS = (MITIGATION, BIOCHAR, AREA, NET_COST, *COSTS, *REVENUES)
OPTIONAL_VARS = (COST_PENALTY, CAPACITY, RESIDUE, *COMPONENTS.values())
REGIONAL_VARS = REQUIRED_VARS + OPTIONAL_VARS

SPATIAL_VARS = (
    "BiocharAlloc_cum",
    "BiocharCumulativeAppliedPerHa",
    "BiocharApplicationCount",
    "yield_effects",
    "biochar_retained_carbon",
)

# Units: most source files use tonnes / timestep, or USD 2025 / timestep.
NINE = 1e9
MILLION = 1e6
CLOSURE_REL_TOL = 0.002  # 0.2% accommodates finite float32 export roundoff
CAP_TOL = 1e-3
COLORS = {
    REFERENCE: "#222222", "biochar_c400_prop": "#c07838",
    "biochar_c400_cap": "#835ca4", "biochar_c400_low_yield": "#4189b2",
    "biochar_c400_high_yield": "#28784e", "biochar_c400_aging": "#9c74a1",
    "biochar_c400_exp_carbon": "#bd5550", "biochar_spp1": "#86a94c",
    "biochar_ssp3": "#d1a237",
}


# -----------------------------------------------------------------------------
# Helpers and data containers
# -----------------------------------------------------------------------------

def file_path(directory: Path, stem: str) -> Path | None:
    for suffix in (".nc", ".NC", ".Nc"):
        p = directory / f"{stem}{suffix}"
        if p.is_file():
            return p
    return None


def safe_percent(current: float, reference: float) -> float:
    if not np.isfinite(current) or not np.isfinite(reference) or abs(reference) < 1e-14:
        return float("nan")
    return 100.0 * (current / reference - 1.0)


def ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) <= 1e-12:
        return float("nan")
    return float(a / b)


def fnum(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}" if np.isfinite(value) else "n/a"


def years_from_coord(coord: xr.DataArray) -> np.ndarray:
    values = np.asarray(coord.values)
    if np.issubdtype(values.dtype, np.datetime64):
        return np.asarray(coord.dt.year.values, dtype=int)
    try:
        return values.astype(int)
    except (ValueError, TypeError):
        return np.asarray(coord.dt.year.values, dtype=int)


def step_widths(years: np.ndarray) -> np.ndarray:
    years = np.asarray(years, dtype=int)
    if len(years) < 2 or np.any(np.diff(years) <= 0):
        raise ValueError("At least two strictly increasing time coordinates are required")
    # A labelled timestep represents the preceding interval (as in the Figure 1 module).
    return np.r_[years[1] - years[0], np.diff(years)].astype(float)


def read_yaml(directory: Path) -> tuple[dict, dict]:
    p = directory / "scenario.yaml"
    if not p.exists():
        return {}, {}
    if yaml is None:
        raise RuntimeError("Install PyYAML to inspect scenario.yaml: pip install pyyaml")
    with p.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}
    if not isinstance(cfg, dict):
        raise ValueError("scenario.yaml does not contain a mapping")
    scenario = cfg.get("scenario", cfg)
    if not isinstance(scenario, dict):
        raise ValueError("scenario.yaml 'scenario' entry is not a mapping")
    return scenario, cfg.get("timeline", {}) or {}


def read_regional(directory: Path, name: str) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Read a small regional export into (time, R26, flattened extra dims)."""
    path = file_path(directory, name)
    if path is None:
        raise FileNotFoundError(f"{name}.nc not found")
    with xr.open_dataset(path) as ds:
        if name not in ds:
            raise KeyError(f"{path.name}: expected {name!r}; variables: {list(ds.data_vars)}")
        da = ds[name]
        if "time" not in da.dims or "region" not in da.dims:
            raise ValueError(f"{path.name}: expected time and region dimensions, got {da.dims}")
        years = years_from_coord(da["time"])
        dims_extra = tuple(d for d in da.dims if d not in ("time", "region"))
        da = da.transpose("time", "region", *dims_extra)
        raw = np.asarray(da.values, dtype=np.float64)
        if raw.shape[1] != len(R26):
            raise ValueError(f"{path.name}: expected {len(R26)} regions, found {raw.shape[1]}")
        raw = raw.reshape(raw.shape[0], raw.shape[1], -1)
    return years, raw, dims_extra


@dataclass
class Result:
    spec: Scenario
    directory: Path
    scenario_yaml: dict = field(default_factory=dict)
    timeline_yaml: dict = field(default_factory=dict)
    years: np.ndarray | None = None
    data: dict[str, np.ndarray] = field(default_factory=dict)
    # By default data contains small (time, region, extra dims) numpy arrays.
    files_found: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    information: list[str] = field(default_factory=list)
    status: str = "NOT_CHECKED"

    def flag(self, level: str, message: str) -> None:
        target = {"ERROR": self.errors, "WARNING": self.warnings,
                  "INFO": self.information}[level]
        if message not in target:
            target.append(message)

    def field(self, name: str) -> np.ndarray | None:
        return self.data.get(name)

    def region(self, name: str) -> np.ndarray | None:
        raw = self.field(name)
        return None if raw is None else np.sum(raw, axis=2)

    def global_step(self, name: str) -> np.ndarray | None:
        region = self.region(name)
        return None if region is None else np.sum(region, axis=1)

    def in_period(self, name: str, start: int, end: int) -> float:
        values = self.global_step(name)
        if values is None or self.years is None:
            return float("nan")
        mask = (self.years >= start) & (self.years <= end)
        return float(np.sum(values[mask])) if mask.any() else float("nan")

    def at(self, name: str, year: int) -> float:
        values = self.global_step(name)
        if values is None or self.years is None or year not in self.years:
            return float("nan")
        return float(values[int(np.where(self.years == year)[0][0])])

    def year_dt(self, year: int) -> float:
        if self.years is None or year not in self.years:
            return float("nan")
        return float(step_widths(self.years)[np.where(self.years == year)[0][0]])

    def annual(self, name: str, year: int) -> float:
        return self.at(name, year) / self.year_dt(year)

    @property
    def is_valid(self) -> bool:
        return self.status in ("PASS", "WARN")


def validate_scenario_config(result: Result) -> None:
    config = result.scenario_yaml
    if not config:
        result.flag("WARNING", "scenario.yaml missing: cannot verify run settings")
        return
    settings = config.get("settings", {})
    if not isinstance(settings, dict):
        result.flag("ERROR", "scenario.yaml settings is not a mapping")
        return
    for key, required in result.spec.expected.items():
        actual = settings.get(key)
        valid = actual is not None if required == "non_null" else actual == required
        if not valid:
            result.flag("ERROR", f"Scenario setting {key}={actual!r}; expected {required!r}")
    if config.get("carbon_tax_name", "c400") != "c400":
        result.flag("ERROR", f"Expected $400/tCO2 experiment, found carbon scenario {config.get('carbon_tax_name')}")
    ssp = settings.get("land_scenario", "SSP2")
    landrent = str(config.get("landrent_directory", "")).replace("\\", "/").rstrip("/").lower()
    expected_suffix = ("biochar/ssp1_redd50" if ssp == "SSP1" else
                       "afforestation/ssp2_redd50")
    if landrent and not landrent.endswith(expected_suffix):
        result.flag("WARNING", f"Land-rent directory {landrent!r} does not end in {expected_suffix!r}")
    cap = settings.get("max_cumulative_application_t_ha")
    if result.spec.name == "biochar_c400_cap":
        if cap is not None:
            if float(cap) != 50.0:
                result.flag("WARNING", f"Cumulative cap is {cap} t/ha, while the manuscript specifies 50 t/ha")
            dose = float(settings.get("application_rate_t_ha", 20.0))
            result.flag("INFO", f"At {dose:g} t/ha per application, a {cap:g} t/ha cap permits {math.floor(float(cap)/dose)} complete applications")
    if result.spec.name == "biochar_c400_aging":
        result.flag("INFO", "Aging diagnostics in BiocharRealisedYieldEffect are not equivalent to whole-system land-use effects")


def audit_run(spec: Scenario, root: Path, start_year: int, end_year: int) -> Result:
    result = Result(spec, root / spec.name)
    directory = result.directory
    if not directory.is_dir():
        result.flag("ERROR", "Run output directory missing")
        result.status = "MISSING"
        return result

    try:
        result.scenario_yaml, result.timeline_yaml = read_yaml(directory)
    except Exception as exc:
        result.flag("ERROR", f"Unable to read scenario.yaml: {exc}")
    validate_scenario_config(result)

    for name in REGIONAL_VARS:
        nc = file_path(directory, name)
        if nc is None:
            if name in REQUIRED_VARS:
                result.flag("ERROR", f"Missing required output {name}.nc")
            else:
                result.flag("WARNING", f"Optional diagnostic output {name}.nc missing")
            continue
        result.files_found += 1
        try:
            years, raw, extra = read_regional(directory, name)
        except Exception as exc:
            result.flag("ERROR" if name in REQUIRED_VARS else "WARNING",
                        f"Cannot read {name}: {type(exc).__name__}: {exc}")
            continue
        if result.years is None:
            result.years = years
        elif not np.array_equal(result.years, years):
            result.flag("ERROR", f"Time coordinate mismatch for {name}: {years.tolist()}")
            continue
        if not np.all(np.isfinite(raw)):
            n = int(np.count_nonzero(~np.isfinite(raw)))
            result.flag("ERROR", f"{name} contains {n} NaN/Inf values in regional outputs")
        if name in (BIOCHAR, AREA, RESIDUE, CAPACITY, *COSTS, *REVENUES, COST_PENALTY):
            if np.any(raw < -1e-5):
                result.flag("WARNING", f"{name} contains negative values (unexpected for this quantity)")
        result.data[name] = raw

    if result.years is None:
        result.flag("ERROR", "No readable regional time-series outputs")
        result.status = "FAIL"
        return result

    years = result.years
    try:
        dt = step_widths(years)
        if not np.all(np.isfinite(dt)) or np.any(dt <= 0):
            result.flag("ERROR", "Non-positive or nonfinite timestep width")
        declared_step = int(result.timeline_yaml.get("step", DEFAULT_STEP))
        declared_start = int(result.timeline_yaml.get("start", 2025))
        declared_end = int(result.timeline_yaml.get("end", end_year))
        expected_years = np.arange(declared_start, declared_end + 1, declared_step)
        if not np.array_equal(years, expected_years):
            result.flag("ERROR", "Model years differ from scenario.yaml timeline: "
                        f"actual {years.tolist()}, expected {expected_years.tolist()}")
        if end_year not in years:
            result.flag("ERROR", f"Final model year {end_year} absent (run may be incomplete)")
        if first_missing := [y for y in range(start_year, end_year + 1, DEFAULT_STEP) if y not in years]:
            result.flag("ERROR", f"Requested comparison years missing: {first_missing}")
    except (ValueError, TypeError) as exc:
        result.flag("ERROR", f"Invalid timeline: {exc}")

    if MITIGATION in result.data and end_year in years:
        try:
            final = result.annual(MITIGATION, end_year)
            if not np.isfinite(final) or final <= 0:
                result.flag("ERROR", f"Annual realised mitigation at {end_year} is {final}; expected > 0 for a c400 run")
        except ValueError as exc:
            result.flag("ERROR", f"Unable to annualise final-year mitigation: {exc}")

    # Reconcile net resource cost with exactly those terms used in allocation.py.
    if all(n in result.data for n in (*COSTS, *REVENUES, NET_COST)):
        expected = sum(result.field(n) for n in COSTS) - sum(result.field(n) for n in REVENUES)
        actual = result.field(NET_COST)
        residual = np.abs(expected - actual)
        # Net cost can be close to zero through cancellation, so normalise
        # closure by gross expenditure, not by the much smaller net cost.
        gross_expenditure = sum(result.field(n) for n in COSTS)
        denominator = np.maximum(np.abs(gross_expenditure), 1.0)
        relative = np.max(residual / denominator)
        if relative > CLOSURE_REL_TOL and float(np.max(residual)) > 5:
            result.flag("ERROR", "Realised net cost fails cost-components closure: "
                        f"max relative discrepancy {100*relative:.3g}%")
        else:
            result.flag("INFO", f"Net resource cost closes (max relative error {100*relative:.3g}%)")

    if MITIGATION in result.data and all(n in result.data for n in COMPONENTS.values()):
        # This residual should approximate positive transport CO2 emissions.
        residual = sum(result.field(n).sum(axis=(1, 2)) for n in COMPONENTS.values()) \
                   - result.global_step(MITIGATION)
        gross = sum(result.field(n).sum(axis=(1, 2)) for n in COMPONENTS.values())
        if np.any(residual < -(1e-4 + 1e-5*np.abs(gross))):
            result.flag("WARNING", "Mitigation component sum is below realised net mitigation in some years; "
                        "check transport emissions / component consistency")
        if end_year in years:
            ix = int(np.where(years == end_year)[0][0])
            result.flag("INFO", f"Inferred {end_year} transport emissions (gross minus net): "
                        f"{residual[ix]/NINE:.5g} GtCO2-eq/timestep")

    if CAPACITY in result.data and BIOCHAR in result.data:
        cap = result.field(CAPACITY)
        production = result.field(BIOCHAR)
        if cap.shape == production.shape and cap.shape[-1] == 2:
            try:
                dt = step_widths(years)[:, None, None]
                excess = production - cap * dt
                max_excess = float(np.nanmax(excess))
                tolerance = max(1e-2, float(np.nanmax(production)) * 1e-4)
                if max_excess > tolerance:
                    result.flag("WARNING", f"Realised production exceeds operational capacity × timestep "
                                f"by up to {max_excess:,.0f} tonnes (inspect capacity timing)")
            except ValueError as exc:
                result.flag("ERROR", f"Cannot check capacity constraint: {exc}")

    # Output directories can contain older files from a previous run. The
    # resolved configuration should not postdate final realised outputs.
    manifest = directory / "scenario.yaml"
    realised_path = file_path(directory, MITIGATION)
    if manifest.is_file() and realised_path is not None:
        if manifest.stat().st_mtime - realised_path.stat().st_mtime > 60.0:
            result.flag("WARNING", "scenario.yaml is newer than the final mitigation export; "
                        "outputs might belong to a previous configuration")

    result.status = "FAIL" if result.errors else "WARN" if result.warnings else "PASS"
    return result


def check_sensitivity_effectiveness(results: dict[str, Result]) -> None:
    """Flag ineffective variants, without assuming differences are guaranteed."""
    ref = results.get(REFERENCE)
    if ref is None or MITIGATION not in ref.data or ref.years is None:
        return
    for spec in SCENARIOS:
        if spec.name == REFERENCE:
            continue
        r = results[spec.name]
        if r.years is None or not np.array_equal(r.years, ref.years):
            continue
        if MITIGATION in r.data:
            a, b = r.global_step(MITIGATION), ref.global_step(MITIGATION)
            if np.allclose(a, b, rtol=1e-7, atol=1e-5):
                r.flag("WARNING", "Realised mitigation trajectory is identical to reference; "
                       "the sensitivity may not bind or its settings may not have taken effect")
        key = None
        if spec.name == "biochar_c400_prop":
            key = RESIDUE
        elif spec.name == "biochar_c400_cap":
            key = AREA
        elif spec.name in ("biochar_c400_low_yield", "biochar_c400_high_yield", "biochar_c400_aging"):
            key = "BiocharRealisedRevenueYield"
        elif spec.name == "biochar_c400_exp_carbon":
            key = "BiocharFieldCDR"
        if key in r.data and key in ref.data:
            x, y = r.global_step(key), ref.global_step(key)
            if np.allclose(x, y, rtol=1e-7, atol=1e-5):
                r.flag("WARNING", f"{key} time series is indistinguishable from the reference; "
                       "verify that the intended scenario mechanism was activated")


def select_spatial(directory: Path, name: str, year: int, tech: int | None = None) -> np.ndarray:
    """Load ONLY the selected year/grid/technology slice, never the whole grid history."""
    p = file_path(directory, name)
    if p is None:
        raise FileNotFoundError(f"Missing {name}.nc")
    with xr.open_dataset(p) as ds:
        if name not in ds:
            raise KeyError(f"Missing {name} variable in {p.name}")
        da = ds[name]
        if "time" in da.dims:
            years = years_from_coord(da["time"])
            at = np.where(years == int(year))[0]
            if at.size == 0:
                raise ValueError(f"Year {year} absent from {p.name}")
            da = da.isel(time=int(at[0]))
        if tech is not None:
            if "tech" not in da.dims:
                raise ValueError(f"{p.name} lacks the 'tech' dimension")
            if tech >= da.sizes["tech"]:
                raise ValueError(f"Technology {tech} absent from {p.name}")
            da = da.isel(tech=tech)
        if "grid" not in da.dims:
            raise ValueError(f"{p.name} is not gridded")
        # A data slice should be [grid], or [grid, tech] only if tech is None.
        out = np.asarray(da.values)
        return out


def crop_beta_coefficients(directory: Path) -> pd.Series | None:
    csv = directory / "yield_beta_selected.csv"
    if not csv.is_file():
        return None
    try:
        frame = pd.read_csv(csv)
        if {"predictor", "beta"}.issubset(frame.columns):
            vals = pd.to_numeric(frame["beta"], errors="coerce")
            return pd.Series(vals.to_numpy(dtype=float), index=frame["predictor"].astype(str)).sort_index()
        return None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Compact results tables (global, temporal, regional, and pathway accounting)
# -----------------------------------------------------------------------------

def build_tables(results: dict[str, Result], start_year: int, end_year: int):
    summary, series_rows, components_rows, region_rows = [], [], [], []
    ref = results.get(REFERENCE)

    for spec in SCENARIOS:
        r = results[spec.name]
        row: dict[str, Any] = {
            "scenario": spec.name, "label": spec.label, "group": spec.group,
            "health": r.status, "settings_validated": bool(r.scenario_yaml),
        }
        if not r.is_valid or r.years is None or MITIGATION not in r.data:
            summary.append(row)
            continue
        years = r.years
        dt = step_widths(years)
        mit = r.global_step(MITIGATION)
        cmask = (years >= start_year) & (years <= end_year)

        for i, year in enumerate(years):
            rec = {"scenario": spec.name, "label": spec.label, "year": int(year),
                   "dt_years": float(dt[i]),
                   "annual_mitigation_GtCO2eq_yr": float(mit[i] / dt[i] / NINE),
                   "step_mitigation_GtCO2eq": float(mit[i] / NINE),
                   "cumulative_since_start_GtCO2eq": float(np.sum(mit[(years >= start_year) & (years <= year)]) / NINE)
                   if year >= start_year else 0.0}
            for var, col, scale in (
                (AREA, "area_treated_Mha_yr", MILLION),
                (BIOCHAR, "production_Mt_biochar_yr", MILLION),
                (RESIDUE, "residue_use_MtDM_yr", MILLION),
            ):
                x = r.global_step(var)
                rec[col] = float(x[i] / dt[i] / scale) if x is not None else np.nan
            net = r.global_step(NET_COST)
            rec["net_cost_USD2025_per_tCO2eq"] = ratio(float(net[i]), float(mit[i])) if net is not None else np.nan
            yieldrev = r.global_step("BiocharRealisedRevenueYield")
            rec["yield_revenue_billion_USD2025_yr"] = float(yieldrev[i] / dt[i] / NINE) if yieldrev is not None else np.nan
            series_rows.append(rec)

        row["annual_mitigation_2100_GtCO2eq_yr"] = r.annual(MITIGATION, end_year) / NINE
        row["cumulative_mitigation_GtCO2eq"] = r.in_period(MITIGATION, start_year, end_year) / NINE
        row["annual_mitigation_2050_GtCO2eq_yr"] = r.annual(MITIGATION, 2050) / NINE
        row["annual_mitigation_2075_GtCO2eq_yr"] = r.annual(MITIGATION, 2075) / NINE
        row["annual_area_treated_2100_Mha_yr"] = r.annual(AREA, end_year) / MILLION
        row["annual_biochar_production_2100_Mt_yr"] = r.annual(BIOCHAR, end_year) / MILLION
        row["annual_residue_use_2100_MtDM_yr"] = r.annual(RESIDUE, end_year) / MILLION
        row["net_cost_2100_USD2025_per_tCO2eq"] = ratio(r.at(NET_COST, end_year), r.at(MITIGATION, end_year))
        row["net_cost_cumulative_USD2025_per_tCO2eq"] = ratio(
            r.in_period(NET_COST, start_year, end_year), r.in_period(MITIGATION, start_year, end_year))
        row["yield_revenue_2100_billion_USD2025_yr"] = r.annual(REVENUES[0], end_year) / NINE
        row["yield_revenue_cumulative_billion_USD2025"] = r.in_period(REVENUES[0], start_year, end_year) / NINE
        row["biooil_revenue_cumulative_billion_USD2025"] = r.in_period(REVENUES[1], start_year, end_year) / NINE
        row["transport_carbon_penalty_cumulative_billion_USD2025"] = r.in_period(COST_PENALTY, start_year, end_year) / NINE
        gross_period = sum(r.in_period(n, start_year, end_year) for n in COSTS)
        row["gross_cost_cumulative_billion_USD2025"] = gross_period / NINE
        row["yield_revenue_as_gross_cost_pct"] = 100 * ratio(
            r.in_period(REVENUES[0], start_year, end_year), gross_period)
        if r.scenario_yaml:
            settings = r.scenario_yaml.get("settings", {})
            row["configured_cumulative_cap_t_ha"] = settings.get("max_cumulative_application_t_ha")
            row["configured_yield_coefficients"] = settings.get("yield_response_scenario")
            row["configured_persistence"] = settings.get("carbon_persistence_model")
            row["configured_land_scenario"] = settings.get("land_scenario")
            row["configured_landrent_directory"] = str(r.scenario_yaml.get("landrent_directory", ""))
        summary.append(row)

        # Report decomposition in absolute terms and as share of net mitigation.
        for period_name, mask in (("2100", years == end_year),
                                  (f"{start_year}-{end_year}", cmask)):
            denominator = float(np.sum(mit[mask]))
            gross = 0.0
            for label, var in COMPONENTS.items():
                v = r.global_step(var)
                absolute = float(np.sum(v[mask])) if v is not None else np.nan
                if np.isfinite(absolute):
                    gross += absolute
                components_rows.append({"scenario": spec.name, "period": period_name,
                                        "component": label, "mitigation_GtCO2eq": absolute / NINE,
                                        "pct_of_net_mitigation": 100 * ratio(absolute, denominator)})
            if all(v in r.data for v in COMPONENTS.values()):
                components_rows.append({"scenario": spec.name, "period": period_name,
                                        "component": "Inferred transport emissions (deduction)",
                                        "mitigation_GtCO2eq": -(gross - denominator) / NINE,
                                        "pct_of_net_mitigation": -100 * ratio(gross-denominator, denominator)})

        for region_name, members in R10.items():
            indices = [R26.index(x) for x in members]
            values = r.region(MITIGATION)[:, indices].sum(axis=1)
            cost = r.region(NET_COST)
            cost_2100 = cost[np.where(years == end_year)[0][0], indices].sum() if cost is not None else np.nan
            mit_2100 = values[np.where(years == end_year)[0][0]]
            region_rows.append({
                "scenario": spec.name, "region": region_name,
                "annual_2100_GtCO2eq_yr": float(mit_2100 / r.year_dt(end_year) / NINE),
                "cumulative_GtCO2eq": float(values[cmask].sum() / NINE),
                "net_cost_2100_USD2025_per_tCO2eq": ratio(cost_2100, mit_2100),
            })

    summary_df = pd.DataFrame(summary)
    series_df = pd.DataFrame(series_rows)
    comp_df = pd.DataFrame(components_rows)
    regional_df = pd.DataFrame(region_rows)

    if not summary_df.empty and ref is not None and ref.is_valid:
        base = summary_df.set_index("scenario").loc[REFERENCE]
        comparison_cols = {
            "annual_mitigation_2100_GtCO2eq_yr": "annual_2100_change_pct",
            "cumulative_mitigation_GtCO2eq": "cumulative_change_pct",
            "annual_mitigation_2050_GtCO2eq_yr": "annual_2050_change_pct",
            "annual_mitigation_2075_GtCO2eq_yr": "annual_2075_change_pct",
            "annual_area_treated_2100_Mha_yr": "treated_area_2100_change_pct",
            "annual_biochar_production_2100_Mt_yr": "production_2100_change_pct",
            "annual_residue_use_2100_MtDM_yr": "residue_use_2100_change_pct",
            "net_cost_2100_USD2025_per_tCO2eq": "net_cost_intensity_2100_change_pct",
            "yield_revenue_2100_billion_USD2025_yr": "yield_revenue_2100_change_pct",
            "yield_revenue_cumulative_billion_USD2025": "yield_revenue_cumulative_change_pct",
        }
        comparison = summary_df[["scenario", "label", "group", "health"]].copy()
        for source, output in comparison_cols.items():
            reference_value = float(base[source]) if source in base else np.nan
            comparison[output] = summary_df[source].map(lambda x: safe_percent(float(x), reference_value))
        comparison["annual_2100_change_GtCO2eq_yr"] = (
            summary_df["annual_mitigation_2100_GtCO2eq_yr"] - float(base["annual_mitigation_2100_GtCO2eq_yr"]))
        comparison["cumulative_change_GtCO2eq"] = (
            summary_df["cumulative_mitigation_GtCO2eq"] - float(base["cumulative_mitigation_GtCO2eq"]))
        if not regional_df.empty:
            ref_regions = regional_df[regional_df.scenario == REFERENCE].set_index("region")
            for i, rec in regional_df.iterrows():
                baseline = ref_regions.loc[rec["region"]]
                regional_df.loc[i, "annual_2100_change_GtCO2eq_yr"] = (
                    rec["annual_2100_GtCO2eq_yr"] - baseline["annual_2100_GtCO2eq_yr"])
                regional_df.loc[i, "annual_2100_change_pct"] = safe_percent(
                    rec["annual_2100_GtCO2eq_yr"], baseline["annual_2100_GtCO2eq_yr"])
                regional_df.loc[i, "cumulative_change_GtCO2eq"] = (
                    rec["cumulative_GtCO2eq"] - baseline["cumulative_GtCO2eq"])
    else:
        comparison = pd.DataFrame()

    return summary_df, comparison, series_df, comp_df, regional_df


# -----------------------------------------------------------------------------
# Spatial checks: only a single final-year slice of each large NetCDF is read.
# -----------------------------------------------------------------------------

def spatial_audit(results: dict[str, Result], end_year: int, enabled: bool):
    rows: list[dict[str, Any]] = []
    agronomic_rows: list[dict[str, Any]] = []
    if not enabled:
        return pd.DataFrame(), pd.DataFrame()

    ref = results.get(REFERENCE)
    base_mask = None
    base_yield = None
    base_retained = None
    beta_ref = None
    if ref is not None and ref.is_valid:
        try:
            base_mask = select_spatial(ref.directory, "BiocharAlloc_cum", end_year) > 0
        except Exception as exc:
            ref.flag("WARNING", f"Cannot read reference spatial footprint: {exc}")
        try:
            base_yield = np.asarray(select_spatial(ref.directory, "yield_effects", end_year, tech=0), dtype=np.float32)
        except Exception as exc:
            ref.flag("WARNING", f"Cannot read reference yield effects: {exc}")
        try:
            base_retained = np.asarray(select_spatial(ref.directory, "biochar_retained_carbon", end_year), dtype=np.float32)
        except Exception as exc:
            ref.flag("WARNING", f"Cannot compare reference carbon retention field: {exc}")
        beta_ref = crop_beta_coefficients(ref.directory)

    for spec in SCENARIOS:
        r = results[spec.name]
        if not r.is_valid:
            continue
        row: dict[str, Any] = {"scenario": spec.name, "label": spec.label}
        current_mask = None
        try:
            current_mask = select_spatial(r.directory, "BiocharAlloc_cum", end_year) > 0
            row["cumulative_deployed_cells_2100"] = int(current_mask.sum())
            if base_mask is not None:
                if len(current_mask) != len(base_mask):
                    r.flag("WARNING", "Spatial grid length differs from reference; cannot calculate Jaccard")
                else:
                    intersection = int((current_mask & base_mask).sum())
                    union = int((current_mask | base_mask).sum())
                    row["jaccard_with_reference_pct"] = 100 * intersection / union if union else np.nan
                    row["cells_only_in_sensitivity"] = int((current_mask & ~base_mask).sum())
                    row["cells_only_in_reference"] = int((base_mask & ~current_mask).sum())
        except Exception as exc:
            r.flag("WARNING", f"Cannot read cumulative spatial deployment: {exc}")

        try:
            amount = select_spatial(r.directory, "BiocharCumulativeAppliedPerHa", end_year)
            valid = np.isfinite(amount)
            if np.any(~valid):
                r.flag("WARNING", f"Cumulative application grid has {(~valid).sum():,} NaNs/Infs")
            pos = valid & (amount > 0)
            row["cells_ever_applied"] = int(pos.sum())
            row["maximum_cumulative_application_t_ha"] = float(np.max(amount[valid])) if valid.any() else np.nan
            row["mean_cumulative_application_on_treated_t_ha"] = float(np.mean(amount[pos])) if pos.any() else np.nan
            if current_mask is not None and amount.shape == current_mask.shape:
                mismatch = int(np.count_nonzero((amount > 0) != current_mask))
                if mismatch:
                    r.flag("WARNING", f"{mismatch:,} cells disagree between cumulative deployment and application history")
            cap = (r.scenario_yaml.get("settings", {}) or {}).get("max_cumulative_application_t_ha")
            if cap is not None:
                over = int(np.sum(amount[valid] > float(cap) + CAP_TOL))
                row["cells_exceeding_application_cap"] = over
                if over:
                    r.flag("ERROR", f"Cumulative application cap of {cap} t/ha exceeded in {over:,} cells")
            if np.any(amount[valid] < -CAP_TOL):
                r.flag("ERROR", "Negative cumulative applications found")
        except Exception as exc:
            r.flag("WARNING", f"Cannot read cumulative application totals: {exc}")

        try:
            counts = select_spatial(r.directory, "BiocharApplicationCount", end_year)
            valid = np.isfinite(counts)
            row["maximum_application_count"] = int(np.max(counts[valid])) if valid.any() else np.nan
            if np.any(counts[valid] < 0):
                r.flag("ERROR", "Negative application counts found")
            cap = (r.scenario_yaml.get("settings", {}) or {}).get("max_cumulative_application_t_ha")
            if cap is not None:
                rate = float((r.scenario_yaml.get("settings", {}) or {}).get("application_rate_t_ha", 20))
                allowed = math.floor(float(cap) / rate)
                invalid_count = int(np.count_nonzero(counts[valid] > allowed))
                row["cells_exceeding_max_applications"] = invalid_count
                if invalid_count:
                    r.flag("ERROR", f"Application-count limit {allowed} exceeded in {invalid_count:,} cells")
        except Exception as exc:
            r.flag("WARNING", f"Cannot read application counts: {exc}")

        # This is an unweighted diagnostic of predicted % effects, not yield
        # revenues or hectares of yield response. Use consistent reference cells.
        try:
            response = np.asarray(select_spatial(r.directory, "yield_effects", end_year, tech=0), dtype=np.float32)
            valid = np.isfinite(response)
            if base_mask is not None and len(response) == len(base_mask) and base_mask.any():
                valid &= base_mask
            row["LT_yield_effect_mean_pct_on_reference_footprint"] = float(np.mean(response[valid])) if valid.any() else np.nan
            row["LT_yield_effect_median_pct_on_reference_footprint"] = float(np.median(response[valid])) if valid.any() else np.nan
            # Yield responses are PERCENT, despite the incorrect 'fraction'
            # metadata on this particular export in the supplied model.
            unphysical = int(np.count_nonzero(response[valid] < -100.0))
            row["LT_yield_effect_cells_below_minus100pct"] = unphysical
            if unphysical:
                r.flag("ERROR", f"Predicted yield responses below -100% in {unphysical:,} reference-treated cells")
            if base_yield is not None and response.shape == base_yield.shape:
                valid &= np.isfinite(base_yield)
                delta = response[valid] - base_yield[valid]
                row["LT_yield_effect_RMS_difference_vs_ref_pct_points"] = float(np.sqrt(np.mean(delta.astype(np.float64)**2))) if delta.size else np.nan
                row["LT_yield_effect_max_absolute_difference_vs_ref_pct_points"] = float(np.max(np.abs(delta))) if delta.size else np.nan
                if spec.name in ("biochar_c400_low_yield", "biochar_c400_high_yield") and (not delta.size or np.max(np.abs(delta)) < 1e-6):
                    r.flag("ERROR", "Selected low/high yield scenario produces indistinguishable LT yield-effect map relative to reference; "
                            "check regression coefficient selection bug")
            # free large slice before loading retained carbon
            del response
        except Exception as exc:
            r.flag("WARNING", f"Cannot inspect crop-yield response grid: {exc}")

        if spec.name in (REFERENCE, "biochar_c400_low_yield", "biochar_c400_high_yield"):
            coeff = crop_beta_coefficients(r.directory)
            arow: dict[str, Any] = {"scenario": spec.name, "coefficient_file_found": coeff is not None}
            if spec.name != REFERENCE:
                if coeff is None:
                    r.flag("WARNING", "yield_beta_selected.csv absent/unreadable; coefficient scenario identity cannot be verified")
                elif beta_ref is not None:
                    union = coeff.index.union(beta_ref.index)
                    delta = (coeff.reindex(union) - beta_ref.reindex(union)).to_numpy(dtype=float)
                    arow["maximum_coefficient_difference_from_mean"] = float(np.nanmax(np.abs(delta)))
                    if not np.any(np.abs(delta[np.isfinite(delta)]) > 1e-10):
                        r.flag("ERROR", "Selected low/high coefficients identical to reference coefficients")
                else:
                    r.flag("WARNING", "Reference coefficient CSV unavailable; coefficients cannot be compared")
            agronomic_rows.append(arow)

        if spec.name in (REFERENCE, "biochar_c400_exp_carbon"):
            try:
                # Temperature-dependent persistent-C fraction: [grid,tech].
                retained = np.asarray(select_spatial(r.directory, "biochar_retained_carbon", end_year), dtype=np.float32)
                if retained.ndim == 2:
                    if spec.name == "biochar_c400_exp_carbon" and base_retained is not None and base_retained.shape == retained.shape:
                        shared = np.isfinite(base_retained) & np.isfinite(retained)
                        if base_mask is not None and retained.shape[0] == len(base_mask):
                            shared &= base_mask[:, None]
                        delta = retained[shared] - base_retained[shared]
                        row["carbon_retention_RMS_change_vs_reference"] = float(np.sqrt(np.mean(delta.astype(np.float64)**2))) if len(delta) else np.nan
                        if not len(delta) or np.max(np.abs(delta)) < 1e-6:
                            r.flag("WARNING", "Exponential persistence setting produced no measurable change "
                                   "in retained-carbon fractions at reference treated cells")
                    valid = np.isfinite(retained)
                    if current_mask is not None and retained.shape[0] == len(current_mask):
                        valid &= current_mask[:, None]
                    physical_bad = valid & ((retained < -1e-5) | (retained > 1.0 + 1e-5))
                    row["retained_carbon_values_outside_0_1"] = int(physical_bad.sum())
                    if physical_bad.any():
                        r.flag("ERROR", f"{int(physical_bad.sum()):,} treated-grid carbon-retention fractions outside [0,1]")
                    for tech in range(min(2, retained.shape[1])):
                        vals = retained[:, tech][valid[:, tech]]
                        row[f"retained_carbon_mean_tech{tech}"] = float(np.mean(vals)) if len(vals) else np.nan
                del retained
            except Exception as exc:
                r.flag("WARNING", f"Cannot inspect gridded carbon persistence: {exc}")

        rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(agronomic_rows)


# -----------------------------------------------------------------------------
# Figures: all are descriptive visualisations, not statistical significance tests.
# -----------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "font.size": 9.5, "axes.labelsize": 10,
        "axes.titlesize": 11, "legend.fontsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 130, "savefig.dpi": 180,
        "pdf.fonttype": 42,
    })


def _save(fig, out: Path, stem: str):
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_trajectories(ts: pd.DataFrame, out: Path):
    if ts.empty:
        return
    _style()
    groups = [
        ("Residue and application constraints", (REFERENCE, "biochar_c400_prop", "biochar_c400_cap")),
        ("Agronomic uncertainties", (REFERENCE, "biochar_c400_low_yield", "biochar_c400_high_yield", "biochar_c400_aging")),
        ("Carbon persistence and land-use context", (REFERENCE, "biochar_c400_exp_carbon", "biochar_spp1", "biochar_ssp3")),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), sharey=True)
    for ax, (title, names) in zip(axes, groups):
        for name in names:
            dat = ts[(ts.scenario == name) & (ts.year >= FIRST_YEAR)].sort_values("year")
            if dat.empty:
                continue
            ax.plot(dat.year, dat.annual_mitigation_GtCO2eq_yr, color=COLORS[name],
                    lw=2.5 if name == REFERENCE else 1.8,
                    linestyle="-" if name == REFERENCE else "--", label=SCENARIO_LOOKUP[name].label)
        ax.set(title=title, xlabel="Year", xlim=(FIRST_YEAR, LAST_YEAR))
        ax.grid(axis="y", alpha=.24, ls="--")
        ax.legend(loc="best", frameon=False)
    axes[0].set_ylabel(r"Annual net mitigation (GtCO$_2$-eq yr$^{-1}$)")
    fig.suptitle("Biochar mitigation trajectories at $400/tCO₂ (SSP varies where indicated)", y=1.02, fontsize=12)
    fig.tight_layout()
    _save(fig, out, "01_mitigation_trajectories")


def plot_effects(comparisons: pd.DataFrame, out: Path):
    if comparisons.empty:
        return
    data = comparisons[(comparisons.scenario != REFERENCE) & (comparisons.health.isin(["PASS", "WARN"]))].copy()
    if data.empty:
        return
    _style()
    data = data.iloc[::-1]
    y = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, max(4.0, len(data)*.47 + 1.3)), sharey=True)
    for ax, col, title in zip(
        axes, ("annual_2100_change_pct", "cumulative_change_pct"),
        ("Annual mitigation in 2100", f"Cumulative mitigation, {FIRST_YEAR}–{LAST_YEAR}")
    ):
        values = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        ax.axvline(0, color=".35", lw=1)
        for i, (name, value) in enumerate(zip(data.scenario, values)):
            if np.isfinite(value):
                ax.plot([0, value], [i, i], color=COLORS[name], lw=2, alpha=.75)
                ax.scatter(value, i, color=COLORS[name], s=48, zorder=4)
                # Move label above the marker to avoid collisions with the
                # scenario names on the left edge for large negative changes.
                ax.annotate(f"{value:+.1f}%", (value, i), xytext=(0, 7),
                            textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set(title=title, xlabel="Change relative to mixed-technology SSP2 reference (%)")
        ax.grid(axis="x", alpha=.23, ls="--")
    axes[0].set_yticks(y, data.label.tolist())
    fig.suptitle("Sensitivity differences; SSP1/SSP3 change several exogenous inputs", fontsize=11.5)
    fig.tight_layout()
    _save(fig, out, "02_sensitivity_effects")


def plot_costs(summary: pd.DataFrame, out: Path):
    needed = ["scenario", "label", "net_cost_2100_USD2025_per_tCO2eq",
              "yield_revenue_as_gross_cost_pct"]
    if summary.empty or any(c not in summary.columns for c in needed):
        return
    data = summary[summary.health.isin(["PASS", "WARN"])].iloc[::-1]
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13.7, max(4.4, len(data)*.46 + 1.4)), sharey=True)
    yy = np.arange(len(data))
    columns = ("net_cost_2100_USD2025_per_tCO2eq", "yield_revenue_as_gross_cost_pct")
    labels = (r"Net realised cost ($2025/tCO₂-eq), 2100",
              "Crop-yield revenues / gross costs, cumulative (%)")
    for ax, col, label in zip(axes, columns, labels):
        x = pd.to_numeric(data[col], errors="coerce").to_numpy(dtype=float)
        for i, (name, val) in enumerate(zip(data.scenario, x)):
            if np.isfinite(val):
                ax.barh(i, val, color=COLORS[name], height=.64)
        ax.axvline(0, color=".4", lw=.8)
        ax.set(xlabel=label)
        ax.grid(axis="x", alpha=.22, ls="--")
    axes[0].set_yticks(yy, data.label.tolist())
    fig.suptitle("Ex-post realised costs; denominator changes with mitigation", fontsize=11.5)
    fig.tight_layout()
    _save(fig, out, "03_economic_intensities")


def plot_components(comp: pd.DataFrame, out: Path, start_year: int, end_year: int):
    if comp.empty:
        return
    period = f"{start_year}-{end_year}"
    select = comp[(comp.period == period) & (comp.component.isin(COMPONENTS))]
    if select.empty:
        return
    pivot = select.pivot(index="scenario", columns="component", values="mitigation_GtCO2eq")
    pivot = pivot.reindex([s.name for s in SCENARIOS if s.name in pivot.index])
    if pivot.empty:
        return
    palette = ["#9d6738", "#d7b26a", "#68a8a7", "#4c7fbd", "#bd6770"]
    _style()
    fig, ax = plt.subplots(figsize=(12.5, max(4.5, len(pivot)*.48 + 1.2)))
    y = np.arange(len(pivot))
    for sign in (1, -1):
        left = np.zeros(len(pivot), dtype=float)
        for label, color in zip(COMPONENTS, palette):
            v = pivot[label].to_numpy(dtype=float)
            sub = np.where(np.isfinite(v) & ((v * sign) > 0), v, 0.0)
            ax.barh(y, sub, left=left, color=color, height=.67,
                    label=label if sign == 1 else None)
            left += sub
    ax.set_yticks(y, [SCENARIO_LOOKUP[name].label for name in pivot.index])
    ax.invert_yaxis()
    ax.set_xlabel(r"Cumulative direct mitigation (GtCO$_2$-eq); gross components before transport emissions")
    ax.set_title(f"Mitigation pathways, {start_year}–{end_year}")
    ax.grid(axis="x", ls="--", alpha=.18)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, frameon=False)
    fig.tight_layout()
    _save(fig, out, "04_mitigation_composition")


def plot_regions(regional: pd.DataFrame, out: Path):
    if regional.empty or "annual_2100_change_GtCO2eq_yr" not in regional.columns:
        return
    pivot = regional.pivot(index="scenario", columns="region", values="annual_2100_change_GtCO2eq_yr")
    pivot = pivot.reindex(index=[s.name for s in SCENARIOS if s.name != REFERENCE and s.name in pivot.index],
                          columns=list(R10))
    if pivot.empty:
        return
    arr = pivot.to_numpy(dtype=float)
    maxabs = float(np.nanmax(np.abs(arr)))
    if not np.isfinite(maxabs) or maxabs <= 0:
        return
    _style()
    fig, ax = plt.subplots(figsize=(13.5, max(4.6, len(pivot)*.55 + 1.4)))
    im = ax.imshow(arr, vmin=-maxabs, vmax=maxabs, cmap="RdBu_r", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)), [SCENARIO_LOOKUP[x].label for x in pivot.index])
    ax.set_title("Regional contribution to 2100 mitigation difference versus SSP2 reference")
    cb = fig.colorbar(im, ax=ax, shrink=.78, pad=.025)
    cb.set_label(r"Difference in annual mitigation (GtCO₂-eq yr$^{-1}$)")
    fig.tight_layout()
    _save(fig, out, "05_regional_mitigation")


def plot_spatial_overlap(spatial: pd.DataFrame, out: Path):
    if spatial.empty or "jaccard_with_reference_pct" not in spatial.columns:
        return
    sub = spatial[(spatial.scenario != REFERENCE) & spatial.jaccard_with_reference_pct.notna()].iloc[::-1]
    if sub.empty:
        return
    _style()
    fig, ax = plt.subplots(figsize=(9.5, max(3.6, len(sub)*.48 + 1.1)))
    bars = ax.barh(np.arange(len(sub)), sub.jaccard_with_reference_pct,
                   color=[COLORS[x] for x in sub.scenario], height=.65)
    ax.bar_label(bars, fmt="%.1f%%", padding=4)
    ax.set_yticks(np.arange(len(sub)), sub.label.tolist())
    ax.set_xlim(0, 105)
    ax.set_xlabel("Jaccard overlap with SSP2 reference deployment footprint (%)")
    ax.set_title("Grid-cell overlap of ever-treated areas through 2100 (not area weighted)")
    ax.grid(axis="x", ls="--", alpha=.2)
    fig.tight_layout()
    _save(fig, out, "06_spatial_overlap")


# -----------------------------------------------------------------------------
# Human-readable quality / publication-interest report
# -----------------------------------------------------------------------------

def _list_issues(results: dict[str, Result]) -> pd.DataFrame:
    items = []
    for r in results.values():
        for level, messages in (("ERROR", r.errors), ("WARNING", r.warnings), ("INFO", r.information)):
            for msg in messages:
                items.append({"scenario": r.spec.name, "level": level, "message": msg})
    return pd.DataFrame(items, columns=["scenario", "level", "message"])


def _health_table(results: dict[str, Result]) -> pd.DataFrame:
    return pd.DataFrame([{
        "scenario": r.spec.name, "label": r.spec.label, "status": r.status,
        "scenario_folder_exists": r.directory.is_dir(),
        "scenario_yaml_found": (r.directory / "scenario.yaml").is_file(),
        "readable_regional_files": r.files_found,
        "time_first": int(r.years.min()) if r.years is not None else np.nan,
        "time_last": int(r.years.max()) if r.years is not None else np.nan,
        "error_count": len(r.errors), "warning_count": len(r.warnings),
        "caveat": "Output-complete PASS does not prove simulation succeeded without logs",
    } for r in results.values()])


def _interesting_score(row: pd.Series, threshold_pct: float) -> tuple[int, list[str]]:
    why = []
    checks = (
        ("annual_2100_change_pct", f"2100 annual mitigation differs by ≥{threshold_pct:g}%"),
        ("cumulative_change_pct", f"cumulative mitigation differs by ≥{threshold_pct:g}%"),
        ("net_cost_intensity_2100_change_pct", "2100 unit net cost differs by ≥10%"),
        ("yield_revenue_cumulative_change_pct", "cumulative yield revenue differs by ≥10%"),
    )
    for column, label in checks:
        value = row.get(column, np.nan)
        thresh = threshold_pct if column in ("annual_2100_change_pct", "cumulative_change_pct") else 10
        if pd.notna(value) and np.isfinite(float(value)) and abs(float(value)) >= thresh:
            why.append(label)
    return len(why), why


def create_report(results: dict[str, Result], summary: pd.DataFrame, comparisons: pd.DataFrame,
                  spatial: pd.DataFrame, start_year: int, end_year: int,
                  threshold_pct: float, skip_spatial: bool, out: Path) -> None:
    health = _health_table(results)
    lines = [
        "# IMAGE biochar sensitivity scenario audit",
        "",
        "**Scope:** Standalone PRISM biochar exports for the $400/tCO₂ scenarios. "
        "No IMAGE-Land avoided emissions or cropland area effects are included.",
        "",
        "## 1. Did the scenario runs produce usable outputs?",
        "",
        "Status **PASS** = expected regional files, timeline and key numerical/closure checks are present. "
        "**WARN** = apparently complete outputs but one or more caveats. **FAIL** = data, "
        "configuration or numerical error. **MISSING** = scenario directory absent. "
        "Even PASS cannot establish that the full original simulation ran successfully without checking the run logs.",
        "",
        "| Scenario | Status | Errors | Warnings | Final year |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in health.iterrows():
        yr = int(row.time_last) if np.isfinite(row.time_last) else "n/a"
        lines.append(f"| {row.scenario} | **{row.status}** | {row.error_count} | {row.warning_count} | {yr} |")

    lines += [
        "", "## 2. Quantitative screening against the SSP2 $400 reference", "",
        "Numbers below are **descriptive scenario differences, not statistical significance tests**. "
        f"A {threshold_pct:g}% threshold is used solely to flag patterns for closer inspection.", "",
    ]
    if comparisons.empty:
        lines.append("Comparisons unavailable because the reference or essential files failed validation.")
    else:
        candidates = []
        for _, row in comparisons.iterrows():
            if row.scenario == REFERENCE:
                continue
            r = results[row.scenario]
            if not r.is_valid:
                lines.append(f"- **{r.spec.label}:** NOT ASSESSABLE ({r.status}).")
                continue
            annual = row.get("annual_2100_change_pct", np.nan)
            cumulative = row.get("cumulative_change_pct", np.nan)
            annual_abs = row.get("annual_2100_change_GtCO2eq_yr", np.nan)
            cum_abs = row.get("cumulative_change_GtCO2eq", np.nan)
            score, reasons = _interesting_score(row, threshold_pct)
            candidates.append((score, r.spec.label))
            lines.append(
                f"- **{r.spec.label}:** 2100 annual mitigation {fnum(annual,1)}% "
                f"({fnum(annual_abs,3)} GtCO₂-eq/yr); cumulative {start_year}–{end_year} "
                f"{fnum(cumulative,1)}% ({fnum(cum_abs,2)} GtCO₂-eq). "
                + ("**Inspect for main text**: " + "; ".join(reasons) + "." if reasons else
                   "No large effect under the screening thresholds; supplementary reporting may suffice.")
            )
        if candidates:
            lines += ["", "**Scenario ranking by number of screening criteria crossed:** "
                      + ", ".join(f"{label} ({score})" for score, label in sorted(candidates, reverse=True)), ""]

    lines += ["", "## 3. Scientific questions to inspect", "",
              "These are prompts for interpretation, not conclusions inferred without actual output:", "",
              "- **Residue rationing:** Does deployment lose mitigation early or only near the resource-constrained ceiling? "
              "Check residue use, biochar production, annual mitigation and cumulative mitigation together.",
              "- **Application cap:** Does mitigation taper in later years? Compare max application count, cumulative dose "
              "and cumulative footprint; a 40 or 50 t/ha cap has the same maximum two full 20 t/ha applications.",
              "- **Low/high crop-yield responses:** Confirm different posterior coefficient sets **and** different "
              "yield-effect maps before interpreting cost/revenue/area changes. Compare realised yield revenues "
              "and net cost per tCO₂, not only mitigation totals.",
              "- **Agronomic aging:** Inspect yield revenues and net mitigation costs; gridded "
              "BiocharRealisedYieldEffect is an aging diagnostic, not equivalent to IMAGE-Land land sparing.",
              "- **Carbon persistence:** Compare Field CDR against syngas CCS and bio-oil mitigation; "
              "the model also changes allocation when persistent carbon changes.",
              "- **SSP1/SSP3:** Compare mitigation and geography with SSP2, but interpret this as a "
              "joint change in land, residue supply and economic context; NOT a one-variable controlled sensitivity.",
              "", "**Suggested manuscript placement:** residue rationing/application cap in Results 1; "
              "yield-response/aging in Results 2; carbon-persistence composition in Results 3; "
              "SSP-dependent economic/spatial effects in Results 5. Land-sparing and indirect "
              "IMAGE-Land emission claims require separate coupled runs.",
              "", "## 4. Interpretation limitations and checks", "",
              "- Reported total mitigation EXCLUDES indirect IMAGE-Land mitigation. Do not compare the totals "
              "directly to whole-system climate + agriculture figures without the same boundary.",
              "- Cost intensities are mitigation-weighted (sums divided by sums). This can change both "
              "because costs change and because the denominator or selected locations change.",
              "- Source NetCDFs label `yield_effects` as 'fraction', but the underlying yield-response "
              "variables represent percent changes. Spatial yield summaries here use percent units.",
              "- Jaccard overlap is unweighted cell overlap of **ever-treated** footprints against the "
              "SSP2 reference, NOT overlap between HT-only and LT-only strategies. "
              "SSP changes alter the eligible cropland domain itself.",
              "- The source's realised net resource cost excludes the separate transport-emission carbon-price "
              "penalty. It is checked separately, not silently included.",
              "- Flagged differences are not statistical significance tests, confidence intervals, or "
              "proof of a mechanism. Any manuscript numbers must be checked against final frozen-run outputs.",
              "- A supplementary output completeness audit cannot replace checking execution logs and input datasets.",
    ]
    if skip_spatial:
        lines.append("- Spatial checks were disabled with `--skip-spatial`.")
    else:
        lines.append("- Spatial checks used only the selected final-year grid slices, not the full grid time history.")

    lines += ["", "## 5. Specific run issues", ""]
    issue_count = 0
    for r in results.values():
        if r.errors or r.warnings:
            issue_count += 1
            lines.append(f"### {r.spec.name} — {r.status}")
            for item in r.errors:
                lines.append(f"- **ERROR:** {item}")
            for item in r.warnings:
                lines.append(f"- **WARNING:** {item}")
            lines.append("")
    if not issue_count:
        lines.append("No errors or warnings detected in the checks implemented here.")
    if not spatial.empty and "jaccard_with_reference_pct" in spatial.columns:
        lines += ["", "## 6. Ever-treated footprint overlap with SSP2 reference", "",
                  "| Scenario | Jaccard (%) | Treated grid cells |", "|---|---:|---:|"]
        for _, row in spatial.iterrows():
            lines.append(f"| {row.scenario} | {fnum(row.get('jaccard_with_reference_pct',np.nan),1)} | "
                         f"{fnum(row.get('cumulative_deployed_cells_2100',np.nan),0)} |")
    (out / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, output_dir: Path, name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df.to_csv(output_dir / name, index=False, float_format="%.9g")


def run_audit(base_dir: Path, out_dir: Path, start_year: int, end_year: int,
              skip_spatial: bool, threshold_pct: float) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Reading {len(SCENARIOS)} scenario directories in {base_dir}", flush=True)
    results = {}
    for index, spec in enumerate(SCENARIOS, 1):
        result = audit_run(spec, base_dir, start_year, end_year)
        results[spec.name] = result
        print(f"  [{index:02}/{len(SCENARIOS):02}] {spec.name:<31} "
              f"{result.status:<7} errors={len(result.errors):2d}, "
              f"warnings={len(result.warnings):2d}", flush=True)

    check_sensitivity_effectiveness(results)
    if not skip_spatial:
        print("Checking selected 2100 grid slices (this can take time for large NetCDFs)...", flush=True)
    spatial, agronomic = spatial_audit(results, end_year, enabled=not skip_spatial)
    for result in results.values():
        if result.status not in ("MISSING", "FAIL"):
            result.status = "FAIL" if result.errors else "WARN" if result.warnings else "PASS"

    summary, comparisons, timeseries, components, regions = build_tables(results, start_year, end_year)
    tables = {
        "run_health.csv": _health_table(results),
        "issues.csv": _list_issues(results),
        "scenario_summary.csv": summary,
        "comparisons.csv": comparisons,
        "global_timeseries.csv": timeseries,
        "pathway_decomposition.csv": components,
        "regional_comparisons.csv": regions,
        "spatial_diagnostics.csv": spatial,
        "agronomic_checks.csv": agronomic,
    }
    for name, frame in tables.items():
        save_table(frame, out_dir, name)
    create_report(results, summary, comparisons, spatial, start_year, end_year,
                  threshold_pct, skip_spatial, out_dir)

    # Main plot expects the study's 2030–2100 display years by default.
    global FIRST_YEAR, LAST_YEAR
    FIRST_YEAR, LAST_YEAR = start_year, end_year
    plot_trajectories(timeseries, out_dir)
    plot_effects(comparisons, out_dir)
    plot_costs(summary, out_dir)
    plot_components(components, out_dir, start_year, end_year)
    plot_regions(regions, out_dir)
    plot_spatial_overlap(spatial, out_dir)

    counts = pd.Series([r.status for r in results.values()]).value_counts().to_dict()
    print("\nRESULT: " + ", ".join(f"{status}={counts.get(status,0)}" for status in
                                    ("PASS", "WARN", "FAIL", "MISSING")), flush=True)
    print(f"Report: {out_dir / 'findings.md'}")
    print(f"Status table: {out_dir / 'run_health.csv'}")
    print(f"Sensitivity comparison: {out_dir / 'comparisons.csv'}")
    return {"results": results, "tables": tables}


def cli():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-dir", type=Path, default=BASE_DIR,
                   help="Directory containing biochar_c400, biochar_c400_cap, etc.")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write the report, tables and figures (default: BASE_DIR/sensitivity_audit)")
    p.add_argument("--start-year", type=int, default=FIRST_YEAR)
    p.add_argument("--end-year", type=int, default=LAST_YEAR)
    p.add_argument("--skip-spatial", action="store_true",
                   help="Skip final-year large gridded files for a fast first-pass audit")
    p.add_argument("--threshold-pct", type=float, default=5.0,
                   help="Descriptive screening percentage for the report (not significance)")
    args = p.parse_args()
    if args.end_year <= args.start_year:
        p.error("--end-year must exceed --start-year")
    if not args.base_dir.is_dir():
        p.error(f"Output root not found: {args.base_dir}")
    output = args.out_dir or (args.base_dir / "sensitivity_audit")
    run_audit(args.base_dir, output, args.start_year, args.end_year,
              skip_spatial=args.skip_spatial, threshold_pct=args.threshold_pct)


if __name__ == "__main__":
    cli()

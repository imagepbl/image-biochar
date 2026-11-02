import warnings
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy import stats
from __future__ import annotations
from pathlib import Path
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from timer.cdr.cdr_util import COUNTRY_NAMES, land_mask

BASE_OUTPUT = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output")

MITIGATION_DIR = BASE_OUTPUT / "biochar_c400"
AGRICULTURE_DIR = BASE_OUTPUT / "biochar_c400_LT"

MITIGATION_ALLOC_FILE = MITIGATION_DIR / "BiocharAlloc_cum.NC"
AGRICULTURE_ALLOC_FILE = AGRICULTURE_DIR / "BiocharAlloc_cum.NC"

YIELD_REVENUE_FILE = AGRICULTURE_DIR / "yield_revenues_ha.NC"

LAND_DIR = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\biochar\SSP2_M_new")
GLANDCOVER_FILE = LAND_DIR / "GLANDCOVER.NC"
GAREA_FILE = LAND_DIR / "GAREA.NC"

GCOUNTRY_FILE = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\data\global\GCOUNTRY.NC")
OUT_DIR = BASE_OUTPUT / "biochar_allocation_maps" / "figures"
OUT_BASENAME = "biochar_allocation_agriculture_benefits_2panel_v5"

YEAR = 2100
N_LAT = 2160
N_LON = 4320
IMAGE_ORIGIN = "upper"
FALLBACK_OUTPUT_YEARS = np.arange(2025, 2101, 5, dtype=int)
FALLBACK_YEAR_AXES = (
    np.arange(2025, 2101, 5, dtype=int),
    np.arange(1970, 2101, 5, dtype=int),
)

CROPLAND_CLASS_INDEX = 0
MIN_CROPLAND_HA_PER_CELL = 1.0
MIN_COUNTRY_CROPLAND_MHA_FOR_OUTPUT = 0.05

MIN_TOTAL_CROPLAND_MHA = 1.0
MIN_ALLOCATED_UNION_MHA = 0.25
MIN_AGRICULTURE_ONLY_MHA = 0.10

TOP_DECILE_PERCENTILE = 0.90
TOP_QUARTILE_PERCENTILE = 0.75

FIGSIZE = (15.2, 6.5)
DPI = 450
SAVE_PNG = True
SAVE_PDF = True
SAVE_SVG = True
SHOW_FIGURE = True

COLOR_OTHER_LAND = "#F5F1E8"
COLOR_CROPLAND = "#D5D5D5"
COLOR_MITIGATION_ONLY = "#D55E00"
COLOR_AGRICULTURE_ONLY = "#0072B2"
COLOR_BOTH = "#009E73"
ALLOCATION_POINT_SIZE = 0.55

COLOR_OTHER_COUNTRY = "#BDBDBD"
COLOR_SSA = "#A50F15"
COLOR_SE_ASIA = "#5E3C99"
COLOR_EXCEPTION = "#B8860B"
COLOR_TREND = "#4D4D4D"
COLOR_REFERENCE = "#777777"
POINT_EDGE = "#333333"

POINT_SIZE = 34
FOCAL_POINT_SIZE = 54

WRITE_RECOMPUTED_CSVS = True

FOCAL_SSA = [
    "Cameroon",
    "Central African Republic",
    "Mozambique",
    "Niger",
    "Côte d'Ivoire",
]

FOCAL_SE_ASIA = [
    "Indonesia",
    "Philippines",
    "Vietnam",
]

FOCAL_EXCEPTIONS = [
    "Chile",
]

LABEL_COUNTRIES = FOCAL_SSA + FOCAL_SE_ASIA + FOCAL_EXCEPTIONS

DISPLAY_NAMES = {
    "Central African Republic": "CAR",
    "Congo (DRC)": "DRC",
    "United States": "USA",
    "Côte d'Ivoire": "Côte d'Ivoire",
}

LABEL_OFFSETS = {
    "Cameroon": (12, 10),
    "Central African Republic": (-10, 10),
    "Mozambique": (8, 8),
    "Niger": (10, -11),
    "Côte d'Ivoire": (8, 8),
    "Indonesia": (8, -9),
    "Philippines": (10, 11),
    "Vietnam": (10, -11),
    "Chile": (8, 8),
}


# =============================================================================
# Generic helpers
# =============================================================================

def _get_main_dataarray(
    ds: xr.Dataset,
    preferred_names: tuple[str, ...] = (),
) -> xr.DataArray:
    for name in preferred_names:
        if name in ds.data_vars:
            return ds[name]

    data_vars = list(ds.data_vars)

    if not data_vars:
        raise ValueError("Dataset contains no data variables.")

    if len(data_vars) == 1:
        return ds[data_vars[0]]

    largest = max(data_vars, key=lambda name: ds[name].size)
    warnings.warn(
        f"Multiple variables found: {data_vars}. Using largest: {largest}"
    )
    return ds[largest]


def _find_time_dim(da: xr.DataArray) -> str | None:
    for dim in da.dims:
        if dim.lower() in {"time", "year", "years", "t"}:
            return dim

    for dim in da.dims:
        if da.sizes[dim] == len(FALLBACK_OUTPUT_YEARS):
            return dim

    return None


def _extract_years(da: xr.DataArray, time_dim: str) -> np.ndarray | None:
    if time_dim not in da.coords:
        return None

    coord = da.coords[time_dim]

    try:
        if np.issubdtype(coord.dtype, np.datetime64):
            return coord.dt.year.values.astype(int)
    except Exception:
        pass

    values = np.asarray(coord.values)

    try:
        values_int = values.astype(int, copy=False)
        if (
            values_int.size
            and np.nanmin(values_int) >= 1800
            and np.nanmax(values_int) <= 2300
        ):
            return values_int.astype(int)
    except Exception:
        pass

    try:
        years = np.array([int(v.year) for v in values], dtype=int)
        if (
            years.size
            and np.nanmin(years) >= 1800
            and np.nanmax(years) <= 2300
        ):
            return years
    except Exception:
        pass

    return None


def _fallback_year_axis(n_time: int) -> np.ndarray | None:
    for years in FALLBACK_YEAR_AXES:
        if len(years) == n_time:
            return years
    return None


def _select_year_or_final(
    da: xr.DataArray,
    year: int | None,
) -> xr.DataArray:
    if da.ndim == 1:
        return da

    time_dim = _find_time_dim(da)

    if time_dim is None:
        return da

    if year is None:
        return da.isel({time_dim: -1})

    years = _extract_years(da, time_dim)

    if years is None:
        years = _fallback_year_axis(da.sizes[time_dim])

    if years is not None and int(year) in years:
        idx = int(np.flatnonzero(years == int(year))[0])
        return da.isel({time_dim: idx})

    warnings.warn(
        f"Could not match year {year} for {da.name}; using final timestep."
    )
    return da.isel({time_dim: -1})


def _select_landcover_class(
    da: xr.DataArray,
    class_index: int = CROPLAND_CLASS_INDEX,
) -> xr.DataArray:
    if da.ndim == 1:
        return da

    candidates = [
        dim
        for dim in da.dims
        if dim.lower()
        in {
            "landcover",
            "land_cover",
            "landcovertype",
            "land_cover_type",
            "cover",
            "class",
            "classes",
        }
    ]

    if not candidates:
        for dim in da.dims:
            if dim.lower() in {"time", "year", "years", "t"}:
                continue
            if da.sizes[dim] <= 20:
                candidates.append(dim)
                break

    if not candidates:
        raise ValueError(
            f"Could not identify the land-cover dimension in {da.name}."
        )

    return da.isel({candidates[0]: int(class_index)})


def _flatten(da: xr.DataArray) -> np.ndarray:
    return np.asarray(da.values).squeeze().reshape(-1)


def _land_mask_flat() -> np.ndarray:
    mask = np.asarray(land_mask("5m"), dtype=bool).reshape(-1)

    expected = N_LAT * N_LON
    if mask.size != expected:
        raise ValueError(
            f"Expected {expected:,} cells in 5-min mask, got {mask.size:,}."
        )

    return mask


def _to_land_vector(
    values: np.ndarray,
    mask_flat: np.ndarray,
    label: str,
) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    n_land = int(mask_flat.sum())

    if values.size == n_land:
        return values

    if values.size == mask_flat.size:
        return values[mask_flat]

    raise ValueError(
        f"{label} has unexpected length {values.size:,}; expected "
        f"{n_land:,} land cells or {mask_flat.size:,} full-grid cells."
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)

    if not np.any(valid):
        return np.nan

    return float(
        np.sum(values[valid] * weights[valid], dtype=np.float64)
        / np.sum(weights[valid], dtype=np.float64)
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or denominator == 0.0:
        return np.nan
    return float(numerator / denominator)


def country_name_from_index(country_index: int) -> str:
    try:
        country_index = int(country_index)
    except Exception:
        return ""

    if COUNTRY_NAMES is not None:
        if 1 <= country_index <= len(COUNTRY_NAMES):
            return str(COUNTRY_NAMES[country_index - 1])

    return f"Country {country_index}"


# =============================================================================
# Input loading
# =============================================================================

def load_allocation(
    path: Path,
    year: int,
    mask_flat: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Allocation file not found: {path}")

    with xr.open_dataset(path) as ds:
        da = _get_main_dataarray(
            ds,
            preferred_names=(
                "BiocharAlloc_cum",
                "BiocharAllocCum",
                "biochar_alloc_cum",
                "BiocharAlloc",
            ),
        )
        da = _select_year_or_final(da, year)
        raw = _flatten(da)

    values = _to_land_vector(raw, mask_flat, path.name)
    values = np.asarray(values, dtype=np.float32)
    return np.where(np.isfinite(values), values, 0.0)


def load_country(
    path: Path,
    mask_flat: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"GCOUNTRY file not found: {path}")

    with xr.open_dataset(path) as ds:
        da = _get_main_dataarray(
            ds,
            preferred_names=("GCOUNTRY", "gcountry", "country", "Country"),
        )
        raw = _flatten(da)

    return _to_land_vector(
        raw,
        mask_flat,
        "GCOUNTRY",
    ).astype(np.int32, copy=False)


def load_yield_revenue_snapshot(
    path: Path,
    year: int,
    mask_flat: np.ndarray,
) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Yield-revenue file not found: {path}")

    with xr.open_dataset(path) as ds:
        da = _get_main_dataarray(
            ds,
            preferred_names=(
                "yield_revenues_ha",
                "YieldRevenuesHa",
                "YieldRevenueHa",
                "yield_revenue_ha",
                "yield_revenues",
            ),
        )
        da = _select_year_or_final(da, year)
        raw = _flatten(da)

    values = _to_land_vector(raw, mask_flat, "yield_revenues_ha")
    values = values.astype(np.float64, copy=False)
    return np.where(np.isfinite(values), values, np.nan)


def load_cropland_area_ha(
    glandcover_file: Path,
    garea_file: Path,
    year: int,
    mask_flat: np.ndarray,
) -> np.ndarray:
    """
    Convert current fractional GLANDCOVER to cropland hectares.
    """
    if not glandcover_file.exists():
        raise FileNotFoundError(f"GLANDCOVER file not found: {glandcover_file}")

    if not garea_file.exists():
        raise FileNotFoundError(f"GAREA file not found: {garea_file}")

    with xr.open_dataset(glandcover_file) as ds:
        da = _get_main_dataarray(
            ds,
            preferred_names=(
                "GLANDCOVER",
                "GLANDCOVERFrac",
                "glandcover",
                "landcover",
            ),
        )
        da = _select_year_or_final(da, year)
        da = _select_landcover_class(da)
        fraction_raw = _flatten(da)

    with xr.open_dataset(garea_file) as ds:
        da = _get_main_dataarray(
            ds,
            preferred_names=("GAREA", "garea", "area"),
        )
        da = _select_year_or_final(da, year)
        area_raw = _flatten(da)

    fraction = _to_land_vector(
        fraction_raw,
        mask_flat,
        "GLANDCOVER cropland fraction",
    ).astype(np.float64, copy=False)

    garea_km2 = _to_land_vector(
        area_raw,
        mask_flat,
        "GAREA",
    ).astype(np.float64, copy=False)

    fraction = np.where(np.isfinite(fraction), fraction, 0.0)
    garea_km2 = np.where(np.isfinite(garea_km2), garea_km2, 0.0)

    fraction = np.clip(fraction, 0.0, 1.0)
    garea_km2 = np.maximum(garea_km2, 0.0)

    return fraction * garea_km2 * 100.0


# =============================================================================
# Allocation classification
# =============================================================================

def classify_allocations(
    mitigation_alloc: np.ndarray,
    agriculture_alloc: np.ndarray,
) -> np.ndarray:
    if mitigation_alloc.size != agriculture_alloc.size:
        raise ValueError("Mitigation and agriculture allocation lengths differ.")

    mitigation = mitigation_alloc > 0.0
    agriculture = agriculture_alloc > 0.0

    category = np.zeros(mitigation.size, dtype=np.uint8)
    category[mitigation & ~agriculture] = 1
    category[~mitigation & agriculture] = 2
    category[mitigation & agriculture] = 3

    return category


def global_overlap_percent(
    category: np.ndarray,
    cropland_area_ha: np.ndarray,
) -> float:
    union = np.sum(
        cropland_area_ha[np.isin(category, [1, 2, 3])],
        dtype=np.float64,
    )
    both = np.sum(
        cropland_area_ha[category == 3],
        dtype=np.float64,
    )

    return 100.0 * _safe_ratio(both, union)


# =============================================================================
# Country-level recomputation
# =============================================================================

def _within_country_percentiles(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """
    Area-weighted midpoint percentile ranks, matching the previous analysis.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    out = np.full(values.size, np.nan, dtype=np.float64)

    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > MIN_CROPLAND_HA_PER_CELL)
    )

    if not np.any(valid):
        return out

    valid_idx = np.flatnonzero(valid)
    order = np.argsort(values[valid], kind="mergesort")

    idx_sorted = valid_idx[order]
    w_sorted = weights[idx_sorted]

    total_weight = np.sum(w_sorted, dtype=np.float64)

    if total_weight <= 0.0:
        return out

    midpoint = np.cumsum(w_sorted) - 0.5 * w_sorted
    out[idx_sorted] = midpoint / total_weight

    return out


def _category_area(
    category: np.ndarray,
    weights: np.ndarray,
    category_id: int,
) -> float:
    mask = category == category_id
    return float(np.sum(weights[mask], dtype=np.float64))


def _category_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    category: np.ndarray,
    category_id: int | None,
) -> float:
    if category_id is None:
        mask = np.ones(category.size, dtype=bool)
    else:
        mask = category == category_id

    return _weighted_mean(values[mask], weights[mask])


def _share_category_area_above_percentile(
    percentile: np.ndarray,
    weights: np.ndarray,
    category: np.ndarray,
    category_id: int,
    threshold: float,
) -> float:
    mask = (
        (category == category_id)
        & np.isfinite(percentile)
        & np.isfinite(weights)
        & (weights > 0.0)
    )

    if not np.any(mask):
        return np.nan

    denominator = np.sum(weights[mask], dtype=np.float64)

    if denominator <= 0.0:
        return np.nan

    numerator = np.sum(
        weights[mask & (percentile >= threshold)],
        dtype=np.float64,
    )

    return float(numerator / denominator)


def build_country_summaries(
    category: np.ndarray,
    cropland_area_ha: np.ndarray,
    country: np.ndarray,
    yield_revenue_ha: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recompute the two country-level inputs previously read from CSV.

    Returns
    -------
    allocation_summary
    yield_summary
    """
    n = category.size

    for name, values in (
        ("cropland area", cropland_area_ha),
        ("country", country),
        ("yield revenue", yield_revenue_ha),
    ):
        if values.size != n:
            raise ValueError(f"{name} length does not match allocation grid.")

    valid = (
        (country > 0)
        & np.isfinite(cropland_area_ha)
        & (cropland_area_ha > 0.0)
    )

    idx = np.flatnonzero(valid)

    if idx.size == 0:
        raise ValueError("No valid cropland-country cells were found.")

    # Sort once by country for efficient country slices.
    order = np.argsort(country[idx], kind="mergesort")
    idx = idx[order]
    country_sorted = country[idx]

    starts = np.r_[0, 1 + np.flatnonzero(np.diff(country_sorted))]
    ends = np.r_[starts[1:], idx.size]

    allocation_rows = []
    yield_rows = []

    for start, end in zip(starts, ends):
        sub_idx = idx[start:end]
        country_id = int(country[sub_idx[0]])
        country_name = country_name_from_index(country_id)

        area = cropland_area_ha[sub_idx]
        cat = category[sub_idx]
        yrev = yield_revenue_ha[sub_idx]

        total_area_ha = float(np.sum(area, dtype=np.float64))

        if total_area_ha <= 0.0:
            continue

        mit_only_ha = _category_area(cat, area, 1)
        agri_only_ha = _category_area(cat, area, 2)
        both_ha = _category_area(cat, area, 3)
        neither_ha = _category_area(cat, area, 0)

        union_ha = mit_only_ha + agri_only_ha + both_ha
        jaccard = _safe_ratio(both_ha, union_ha)

        allocation_rows.append(
            {
                "year": YEAR,
                "country_id": country_id,
                "country": country_name,
                "total_cropland_area_ha": total_area_ha,
                "total_cropland_area_mha": total_area_ha / 1.0e6,
                "neither_cropland_area_ha": neither_ha,
                "neither_cropland_area_mha": neither_ha / 1.0e6,
                "high_only_cropland_area_ha": mit_only_ha,
                "high_only_cropland_area_mha": mit_only_ha / 1.0e6,
                "low_only_cropland_area_ha": agri_only_ha,
                "low_only_cropland_area_mha": agri_only_ha / 1.0e6,
                "both_cropland_area_ha": both_ha,
                "both_cropland_area_mha": both_ha / 1.0e6,
                "allocated_union_cropland_area_ha": union_ha,
                "allocated_union_cropland_area_mha": union_ha / 1.0e6,
                "high_only_share_of_country_cropland": _safe_ratio(
                    mit_only_ha, total_area_ha
                ),
                "low_only_share_of_country_cropland": _safe_ratio(
                    agri_only_ha, total_area_ha
                ),
                "both_share_of_country_cropland": _safe_ratio(
                    both_ha, total_area_ha
                ),
                "overlap_jaccard_area": jaccard,
            }
        )

        percentile = _within_country_percentiles(yrev, area)

        all_mean = _weighted_mean(yrev, area)
        mit_mean = _category_weighted_mean(yrev, area, cat, 1)
        agri_mean = _category_weighted_mean(yrev, area, cat, 2)
        both_mean = _category_weighted_mean(yrev, area, cat, 3)

        mit_pct_mean = _category_weighted_mean(percentile, area, cat, 1)
        agri_pct_mean = _category_weighted_mean(percentile, area, cat, 2)
        both_pct_mean = _category_weighted_mean(percentile, area, cat, 3)

        agri_top25_share = _share_category_area_above_percentile(
            percentile, area, cat, 2, TOP_QUARTILE_PERCENTILE
        )
        agri_top10_share = _share_category_area_above_percentile(
            percentile, area, cat, 2, TOP_DECILE_PERCENTILE
        )
        mit_top25_share = _share_category_area_above_percentile(
            percentile, area, cat, 1, TOP_QUARTILE_PERCENTILE
        )
        mit_top10_share = _share_category_area_above_percentile(
            percentile, area, cat, 1, TOP_DECILE_PERCENTILE
        )

        yield_rows.append(
            {
                "year": YEAR,
                "country_id": country_id,
                "country": country_name,
                "yield_metric": "yield_revenue_ha_snapshot",
                "total_cropland_area_ha": total_area_ha,
                "total_cropland_area_mha": total_area_ha / 1.0e6,
                "high_only_cropland_area_mha": mit_only_ha / 1.0e6,
                "low_only_cropland_area_mha": agri_only_ha / 1.0e6,
                "both_cropland_area_mha": both_ha / 1.0e6,
                "all_cropland_yield_mean": all_mean,
                "high_only_yield_mean": mit_mean,
                "low_only_yield_mean": agri_mean,
                "both_yield_mean": both_mean,
                "low_only_minus_high_only_yield": agri_mean - mit_mean,
                "low_only_minus_all_cropland_yield": agri_mean - all_mean,
                "high_only_minus_all_cropland_yield": mit_mean - all_mean,
                "low_only_to_all_cropland_yield_ratio": _safe_ratio(
                    agri_mean, all_mean
                ),
                "high_only_country_yield_percentile_mean": mit_pct_mean,
                "low_only_country_yield_percentile_mean": agri_pct_mean,
                "both_country_yield_percentile_mean": both_pct_mean,
                "low_only_share_area_country_top25_yield": agri_top25_share,
                "low_only_top25_enrichment": (
                    agri_top25_share / 0.25
                    if np.isfinite(agri_top25_share)
                    else np.nan
                ),
                "low_only_share_area_country_top10_yield": agri_top10_share,
                "low_only_top10_enrichment": (
                    agri_top10_share / 0.10
                    if np.isfinite(agri_top10_share)
                    else np.nan
                ),
                "high_only_share_area_country_top25_yield": mit_top25_share,
                "high_only_top25_enrichment": (
                    mit_top25_share / 0.25
                    if np.isfinite(mit_top25_share)
                    else np.nan
                ),
                "high_only_share_area_country_top10_yield": mit_top10_share,
                "high_only_top10_enrichment": (
                    mit_top10_share / 0.10
                    if np.isfinite(mit_top10_share)
                    else np.nan
                ),
            }
        )

    allocation_summary = pd.DataFrame(allocation_rows)
    yield_summary = pd.DataFrame(yield_rows)

    allocation_summary = allocation_summary[
        allocation_summary["total_cropland_area_mha"]
        >= MIN_COUNTRY_CROPLAND_MHA_FOR_OUTPUT
    ].copy()

    yield_summary = yield_summary[
        yield_summary["total_cropland_area_mha"]
        >= MIN_COUNTRY_CROPLAND_MHA_FOR_OUTPUT
    ].copy()

    allocation_summary = allocation_summary.sort_values(
        ["low_only_cropland_area_mha", "high_only_cropland_area_mha"],
        ascending=False,
    ).reset_index(drop=True)

    yield_summary = yield_summary.sort_values(
        ["low_only_cropland_area_mha", "low_only_yield_mean"],
        ascending=False,
    ).reset_index(drop=True)

    return allocation_summary, yield_summary


def build_panel_b_data(
    allocation_summary: pd.DataFrame,
    yield_summary: pd.DataFrame,
) -> pd.DataFrame:
    allocation_cols = [
        "country",
        "total_cropland_area_mha",
        "high_only_cropland_area_mha",
        "low_only_cropland_area_mha",
        "both_cropland_area_mha",
        "overlap_jaccard_area",
    ]

    yield_cols = [
        "country",
        "low_only_top10_enrichment",
        "low_only_share_area_country_top10_yield",
        "low_only_country_yield_percentile_mean",
        "high_only_country_yield_percentile_mean",
        "both_country_yield_percentile_mean",
        "low_only_to_all_cropland_yield_ratio",
    ]

    merged = allocation_summary[allocation_cols].merge(
        yield_summary[yield_cols],
        on="country",
        how="inner",
        validate="one_to_one",
    )

    merged["allocated_union_mha"] = (
        merged["high_only_cropland_area_mha"]
        + merged["low_only_cropland_area_mha"]
        + merged["both_cropland_area_mha"]
    )

    merged["jaccard_pct"] = merged["overlap_jaccard_area"] * 100.0
    merged["agriculture_top10_enrichment"] = merged[
        "low_only_top10_enrichment"
    ]

    return merged.sort_values("country").reset_index(drop=True)


def panel_b_sample(df: pd.DataFrame) -> pd.DataFrame:
    finite = (
        np.isfinite(df["jaccard_pct"])
        & np.isfinite(df["agriculture_top10_enrichment"])
        & np.isfinite(df["allocated_union_mha"])
        & np.isfinite(df["low_only_cropland_area_mha"])
        & np.isfinite(df["total_cropland_area_mha"])
    )

    return df[
        finite
        & (df["total_cropland_area_mha"] >= MIN_TOTAL_CROPLAND_MHA)
        & (df["allocated_union_mha"] >= MIN_ALLOCATED_UNION_MHA)
        & (df["low_only_cropland_area_mha"] >= MIN_AGRICULTURE_ONLY_MHA)
    ].copy().reset_index(drop=True)


# =============================================================================
# Statistics
# =============================================================================

def spearman_summary(sample: pd.DataFrame) -> tuple[float, float]:
    if len(sample) < 3:
        return np.nan, np.nan

    rho, p_value = stats.spearmanr(
        sample["jaccard_pct"].to_numpy(dtype=float),
        sample["agriculture_top10_enrichment"].to_numpy(dtype=float),
        nan_policy="omit",
    )

    return float(rho), float(p_value)


def _pvalue_text(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "p = NA"
    if p_value < 0.001:
        return r"$p < 0.001$"
    return rf"$p = {p_value:.3f}$"


# =============================================================================
# Figure style
# =============================================================================

def _publication_rcparams() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "legend.fontsize": 8.5,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_title(letter: str, text: str) -> str:
    return rf"$\bf{{{letter}.}}$ {text}"


# =============================================================================
# Panel a
# =============================================================================

def _full_grid_background(
    mask_flat: np.ndarray,
    cropland_area_ha: np.ndarray,
) -> np.ndarray:
    background = np.full(mask_flat.size, np.nan, dtype=np.float32)

    land_values = np.zeros(cropland_area_ha.size, dtype=np.float32)
    land_values[cropland_area_ha > MIN_CROPLAND_HA_PER_CELL] = 1.0

    background[mask_flat] = land_values
    return background.reshape(N_LAT, N_LON)


def _land_lon_lat(mask_flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Coordinates of 5 arc-min IMAGE land-cell centres in mask order.
    """
    flat = np.flatnonzero(mask_flat)

    row = flat // N_LON
    col = flat % N_LON

    lats = 90.0 - (row + 0.5) * (180.0 / N_LAT)
    lons = -180.0 + (col + 0.5) * (360.0 / N_LON)

    return lons.astype(np.float32), lats.astype(np.float32)




def plot_panel_a(
    ax,
    category: np.ndarray,
    cropland_area_ha: np.ndarray,
    mask_flat: np.ndarray,
) -> None:
    projection_data = ccrs.PlateCarree()

    ax.set_global()

    ax.add_feature(
        cfeature.OCEAN.with_scale("110m"),
        facecolor="white",
        edgecolor="none",
        zorder=0,
    )

    # Other land = light neutral; cropland = grey.
    background = _full_grid_background(mask_flat, cropland_area_ha)

    bg_cmap = ListedColormap([COLOR_OTHER_LAND, COLOR_CROPLAND])
    bg_norm = BoundaryNorm([-0.5, 0.5, 1.5], bg_cmap.N)

    ax.imshow(
        np.ma.masked_invalid(background),
        origin=IMAGE_ORIGIN,
        extent=(-180.0, 180.0, -90.0, 90.0),
        transform=projection_data,
        cmap=bg_cmap,
        norm=bg_norm,
        interpolation="nearest",
        zorder=1,
    )

    lons, lats = _land_lon_lat(mask_flat)

    category_styles = (
        (1, COLOR_MITIGATION_ONLY, "Mitigation"),
        (2, COLOR_AGRICULTURE_ONLY, "Agriculture"),
        (3, COLOR_BOTH, "Both"),
    )

    for category_id, color, _ in category_styles:
        selected = category == category_id

        ax.scatter(
            lons[selected],
            lats[selected],
            s=ALLOCATION_POINT_SIZE,
            marker="s",
            color=color,
            linewidths=0,
            alpha=0.88,
            transform=projection_data,
            rasterized=True,
            zorder=2,
        )

    ax.add_feature(
        cfeature.BORDERS.with_scale("110m"),
        edgecolor="0.70",
        linewidth=0.30,
        zorder=3,
    )

    ax.coastlines(
        resolution="110m",
        color="0.30",
        linewidth=0.45,
        zorder=4,
    )

    # Scenario annotation, using Unicode sub/superscripts to avoid
    # mathtext parsing artefacts around the dollar sign.
    ax.text(
        0.53,
        0.19,
        r"SSP2 carbon price 400 \$ tCO$_2^{-1}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.0,
        color="#222222",
        zorder=6,
    )

    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for _, color, label in category_styles
    ]

    # Place the legend inside the map, in a compact white-backed block.
    # The opaque backing keeps the Robinson frame/map detail from crossing
    # through the legend, matching the reference-map treatment.
    legend = ax.legend(
        handles=handles,
        title="Deployment scenario",
        loc="lower left",
        bbox_to_anchor=(0.055, 0.115),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.96,
        fancybox=False,
        handlelength=1.05,
        handletextpad=0.45,
        borderaxespad=0.0,
        borderpad=0.30,
        labelspacing=0.42,
        prop={"size": 10.4},
        title_fontsize=11.0,
    )
    legend._legend_box.align = "left"


# =============================================================================
# Panel b
# =============================================================================

def _point_group(country: str) -> str:
    if country in FOCAL_SSA:
        return "Selected Sub-Saharan Africa"
    if country in FOCAL_SE_ASIA:
        return "Southeast Asia"
    if country in FOCAL_EXCEPTIONS:
        return "Chile and Peru"
    return "Other countries"


def _group_style(group: str) -> tuple[str, float, int]:
    if group == "Selected Sub-Saharan Africa":
        return COLOR_SSA, 0.95, FOCAL_POINT_SIZE
    if group == "Southeast Asia":
        return COLOR_SE_ASIA, 0.95, FOCAL_POINT_SIZE
    if group == "Chile and Peru":
        return COLOR_EXCEPTION, 0.95, FOCAL_POINT_SIZE

    return COLOR_OTHER_COUNTRY, 0.66, POINT_SIZE


def _annotate_countries(ax, sample: pd.DataFrame) -> None:
    labelled = sample[sample["country"].isin(LABEL_COUNTRIES)]

    for _, row in labelled.iterrows():
        country = row["country"]
        x = float(row["jaccard_pct"])
        y = float(row["agriculture_top10_enrichment"])
        dx, dy = LABEL_OFFSETS.get(country, (6, 6))

        annotation = ax.annotate(
            DISPLAY_NAMES.get(country, country),
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            fontsize=8.1,
            color="#222222",
            arrowprops={
                "arrowstyle": "-",
                "color": "#777777",
                "linewidth": 0.45,
                "alpha": 0.75,
                "shrinkA": 0,
                "shrinkB": 3,
            },
            annotation_clip=True,
            zorder=8,
        )

        annotation.set_path_effects(
            [pe.withStroke(linewidth=2.0, foreground="white")]
        )


def _plot_theil_sen(ax, sample: pd.DataFrame) -> None:
    if len(sample) < 3:
        return

    x = sample["jaccard_pct"].to_numpy(dtype=float)
    y = sample["agriculture_top10_enrichment"].to_numpy(dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    if len(x) < 3 or np.nanmax(x) <= np.nanmin(x):
        return

    slope, intercept, _, _ = stats.theilslopes(y, x, alpha=0.95)

    xx = np.linspace(np.nanmin(x), np.nanmax(x), 300)
    yy = intercept + slope * xx

    ax.plot(
        xx,
        yy,
        color=COLOR_TREND,
        linewidth=1.25,
        alpha=0.88,
        zorder=2,
    )




def plot_panel_b(
    ax,
    sample: pd.DataFrame,
    global_overlap_pct: float,
    rho: float,
    p_value: float,
) -> None:
    # Reference lines.
    ax.axhline(
        1.0,
        color=COLOR_REFERENCE,
        linewidth=0.75,
        linestyle=(0, (1.5, 2.5)),
        alpha=0.65,
        zorder=0,
    )

    ax.axhline(
        5.0,
        color=COLOR_REFERENCE,
        linewidth=0.75,
        linestyle=(0, (3, 3)),
        alpha=0.60,
        zorder=0,
    )

    if np.isfinite(global_overlap_pct):
        ax.axvline(
            global_overlap_pct,
            color=COLOR_REFERENCE,
            linewidth=0.9,
            linestyle=(0, (4, 3)),
            alpha=0.85,
            zorder=1,
        )

    sample = sample.copy()
    sample["plot_group"] = sample["country"].map(_point_group)

    plot_order = [
        "Other countries",
        "Selected Sub-Saharan Africa",
        "Southeast Asia",
        "Chile and Peru",
    ]

    for group in plot_order:
        group_df = sample[sample["plot_group"] == group]

        if group_df.empty:
            continue

        color, alpha, size = _group_style(group)

        ax.scatter(
            group_df["jaccard_pct"],
            group_df["agriculture_top10_enrichment"],
            s=size,
            facecolor=color,
            edgecolor=POINT_EDGE,
            linewidth=0.5,
            alpha=alpha,
            zorder=4 if group != "Other countries" else 3,
        )

    _plot_theil_sen(ax, sample)
    _annotate_countries(ax, sample)

    ax.set_xlim(-2.0, 102.0)
    ax.set_ylim(-0.15, 10.35)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_yticks(np.arange(0, 11, 1))

    ax.set_xlabel("Spatial allocation overlap (Jaccard index, %)")
    ax.set_ylabel(
        "Biochar for agriculture top-decile\n"
        "yield-revenue enrichment (×)"
    )

    ax.grid(
        True,
        which="major",
        linewidth=0.4,
        color="#E6E6E6",
        zorder=0,
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    stats_text = (
        rf"Spearman $\rho$ = {rho:.2f}"
        + "\n"
        + _pvalue_text(p_value)
        + rf", $n$ = {len(sample)}"
    )

    ax.text(
        0.985,
        0.975,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#D0D0D0",
            "linewidth": 0.55,
            "alpha": 0.94,
        },
        zorder=9,
    )

    if np.isfinite(global_overlap_pct):
        ax.text(
            global_overlap_pct + 1.2,
            9.45,
            f"Global overlap {global_overlap_pct:.1f}%",
            ha="left",
            va="top",
            fontsize=8.0,
            color="#555555",
        )

    ax.text(
        99.0,
        1.0 + 0.13,
        "No enrichment (1×)",
        ha="right",
        va="bottom",
        fontsize=7.7,
        color="#666666",
    )

    ax.text(
        99.0,
        5.0 + 0.13,
        "50% of area in national top decile (5×)",
        ha="right",
        va="bottom",
        fontsize=7.7,
        color="#666666",
    )


# =============================================================================
# Main workflow
# =============================================================================


def make_figure(
    category: np.ndarray,
    cropland_area_ha: np.ndarray,
    mask_flat: np.ndarray,
    panel_b_data: pd.DataFrame,
    sample: pd.DataFrame,
    global_overlap_pct: float,
) -> tuple[plt.Figure, dict[str, plt.Axes], float, float]:
    _publication_rcparams()

    rho, p_value = spearman_summary(sample)

    fig = plt.figure(figsize=FIGSIZE)

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=(1.20, 1.0),
        wspace=0.14,
    )

    ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    ax_scatter = fig.add_subplot(gs[0, 1])

    plot_panel_a(
        ax_map,
        category=category,
        cropland_area_ha=cropland_area_ha,
        mask_flat=mask_flat,
    )

    plot_panel_b(
        ax_scatter,
        sample=sample,
        global_overlap_pct=global_overlap_pct,
        rho=rho,
        p_value=p_value,
    )

    # Lower the plotting area slightly so the figure-level panel headings
    # have clear, equal spacing above both panels.
    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.865,
        bottom=0.105,
    )

    # Figure-level headings guarantee identical vertical alignment despite
    # the different Cartopy and Cartesian axes geometries.
    map_box = ax_map.get_position()
    scatter_box = ax_scatter.get_position()
    heading_y = 0.925

    fig.text(
        map_box.x0,
        heading_y,
        _panel_title(
            "a",
            "Cumulative deployment footprint, 2030–2100",
        ),
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="normal",
    )

    fig.text(
        scatter_box.x0,
        heading_y,
        _panel_title(
            "b",
            "Relationship between spatial allocation and agricultural benefits",
        ),
        ha="left",
        va="bottom",
        fontsize=11.5,
        fontweight="normal",
    )

    return (
        fig,
        {"map": ax_map, "scatter": ax_scatter},
        rho,
        p_value,
    )


def save_outputs(
    fig: plt.Figure,
    allocation_summary: pd.DataFrame,
    yield_summary: pd.DataFrame,
    panel_b_data: pd.DataFrame,
    sample: pd.DataFrame,
    rho: float,
    p_value: float,
    global_overlap_pct: float,
) -> dict[str, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = {}

    if SAVE_PNG:
        path = OUT_DIR / f"{OUT_BASENAME}.png"
        fig.savefig(path, dpi=DPI, bbox_inches="tight")
        paths["png"] = path

    if SAVE_PDF:
        path = OUT_DIR / f"{OUT_BASENAME}.pdf"
        fig.savefig(path, bbox_inches="tight")
        paths["pdf"] = path

    if SAVE_SVG:
        path = OUT_DIR / f"{OUT_BASENAME}.svg"
        fig.savefig(path, bbox_inches="tight")
        paths["svg"] = path

    if WRITE_RECOMPUTED_CSVS:
        allocation_path = OUT_DIR / (
            "biochar_allocation_agreement_HT_vs_LT_2100_"
            "country_allocation_summary_recomputed.csv"
        )
        yield_path = OUT_DIR / (
            "biochar_allocation_agreement_HT_vs_LT_2100_"
            "yield_hypothesis_summary_recomputed.csv"
        )
        plot_data_path = OUT_DIR / (
            "biochar_allocation_agreement_HT_vs_LT_2100_"
            "panel_b_plot_data_recomputed.csv"
        )
        sample_path = OUT_DIR / (
            "biochar_allocation_agreement_HT_vs_LT_2100_"
            "panel_b_analysis_sample_recomputed.csv"
        )

        allocation_summary.to_csv(allocation_path, index=False)
        yield_summary.to_csv(yield_path, index=False)
        panel_b_data.to_csv(plot_data_path, index=False)
        sample.to_csv(sample_path, index=False)

        paths["allocation_csv"] = allocation_path
        paths["yield_csv"] = yield_path
        paths["plot_data_csv"] = plot_data_path
        paths["sample_csv"] = sample_path

    diagnostics_path = OUT_DIR / f"{OUT_BASENAME}_diagnostics.txt"

    with diagnostics_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "Biochar spatial allocation / agricultural-benefit figure\n"
        )
        handle.write("=" * 70 + "\n")
        handle.write(f"Year: {YEAR}\n")
        handle.write(
            f"Global area-weighted Jaccard overlap: "
            f"{global_overlap_pct:.6f}%\n"
        )
        handle.write(f"Panel-b countries: {len(sample)}\n")
        handle.write(f"Spearman rho: {rho:.8f}\n")
        handle.write(f"Spearman p: {p_value:.12g}\n")
        handle.write("\nPanel-b filters\n")
        handle.write(
            f"MIN_TOTAL_CROPLAND_MHA = {MIN_TOTAL_CROPLAND_MHA}\n"
        )
        handle.write(
            f"MIN_ALLOCATED_UNION_MHA = {MIN_ALLOCATED_UNION_MHA}\n"
        )
        handle.write(
            f"MIN_AGRICULTURE_ONLY_MHA = "
            f"{MIN_AGRICULTURE_ONLY_MHA}\n"
        )
        handle.write(
            "\nCropland area convention:\n"
            "GLANDCOVER fraction * GAREA km2 * 100 ha/km2\n"
        )

    paths["diagnostics"] = diagnostics_path

    return paths


def main() -> None:
    mask_flat = _land_mask_flat()

    mitigation_alloc = load_allocation(
        MITIGATION_ALLOC_FILE,
        YEAR,
        mask_flat,
    )
    agriculture_alloc = load_allocation(
        AGRICULTURE_ALLOC_FILE,
        YEAR,
        mask_flat,
    )

    category = classify_allocations(
        mitigation_alloc,
        agriculture_alloc,
    )

    cropland_area_ha = load_cropland_area_ha(
        GLANDCOVER_FILE,
        GAREA_FILE,
        YEAR,
        mask_flat,
    )

    country = load_country(
        GCOUNTRY_FILE,
        mask_flat,
    )

    yield_revenue = load_yield_revenue_snapshot(
        YIELD_REVENUE_FILE,
        YEAR,
        mask_flat,
    )

    allocation_summary, yield_summary = build_country_summaries(
        category=category,
        cropland_area_ha=cropland_area_ha,
        country=country,
        yield_revenue_ha=yield_revenue,
    )

    panel_b_data = build_panel_b_data(
        allocation_summary,
        yield_summary,
    )

    sample = panel_b_sample(panel_b_data)

    if len(sample) < 3:
        raise RuntimeError(
            "Too few countries remain after Panel B filtering. "
            "Relax the area thresholds in the User settings section."
        )

    global_overlap_pct = global_overlap_percent(
        category,
        cropland_area_ha,
    )

    fig, _, rho, p_value = make_figure(
        category=category,
        cropland_area_ha=cropland_area_ha,
        mask_flat=mask_flat,
        panel_b_data=panel_b_data,
        sample=sample,
        global_overlap_pct=global_overlap_pct,
    )

    paths = save_outputs(
        fig=fig,
        allocation_summary=allocation_summary,
        yield_summary=yield_summary,
        panel_b_data=panel_b_data,
        sample=sample,
        rho=rho,
        p_value=p_value,
        global_overlap_pct=global_overlap_pct,
    )

    print("\n" + "=" * 90)
    print("Recomputed Panel B diagnostics")
    print("=" * 90)
    print(f"Global overlap: {global_overlap_pct:.2f}%")
    print(f"Panel B n:      {len(sample)}")
    print(f"Spearman rho:   {rho:.3f}")
    print(f"Spearman p:     {p_value:.4g}")

    print("\nSaved outputs:")
    for key, path in paths.items():
        print(f"  {key}: {path}")

    if SHOW_FIGURE:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()

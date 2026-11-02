"""Standalone IMAGE cropland and ruminant-livestock opportunity-cost model.

This module separates the agricultural land-rent calculation into two economic
components:

1. Cropland rent:
   MAGNET ``LandPricesReal - ARABLE&PERMCR`` downscaled within each IMAGE26
   region using a crop-production-value proxy per hectare of crop land.

2. Ruminant/pasture rent:
   MAGNET ``LandPricesReal - livestock_rum`` downscaled within each IMAGE26
   region using IMAGE grass productivity per hectare of pasture.

The two per-hectare sector rents are then combined with IMAGE land-use fractions
to obtain a whole-cell agricultural opportunity cost.

Runtime outputs
---------------
- crop_value_proxy.nc
- crop_rent_scaler.nc
- pasture_productivity_proxy.nc
- pasture_rent_scaler.nc
- CropLandRent.nc
- PastureLandRent.nc
- LandRent.nc

``CropLandRent`` is the rent per hectare of crop land where crop land occurs.
``PastureLandRent`` is the rent per hectare of pasture where pasture occurs.
``LandRent`` is the cell-average opportunity cost:
    crop_fraction * CropLandRent + pasture_fraction * PastureLandRent.

For ``GFORMAN == 4`` cells outside current agricultural land, the historical
IMAGE/A-R grass-based counterfactual is retained: the cell receives a prospective
pasture rent, evaluated with the baseline MAGNET LUT column, but it is excluded
from calibration of the regional pasture scaler.

The MAGNET LUT can be rebuilt from full M2I scenario files. The resulting .mat
contains separate crop and pasture LUTs and retains AGRI_PRIM only as a
reconciliation/diagnostic series. The same afforestation-induced land-supply
reduction is used to select both sectoral LUTs; no extra livestock scarcity
multiplier is applied.
"""


import argparse
import re
import time
import warnings
from pathlib import Path
from typing import List
from scipy.io import loadmat

import numpy as np
import pandas as pd
import prism
import xarray as xr
from pint.errors import UnitStrippedWarning
from scipy.io import savemat

from timer.cdr.cdr_util import (
    Crop17,
    Crop22,
    Crop38,
    GrassLand,
    Grid,
    HarvestGroup3,
    LandGrid,
    Region,
    Region26,
    export_variable_to_netcdf,
    land_data_dir,
    land_mask,
    read_excel_column_F,
    region_to_grid,
    time_step,
)


warnings.simplefilter("ignore", UnitStrippedWarning)
warnings.filterwarnings(
    "ignore",
    message="The specified chunks separate the stored chunks",
    category=UserWarning,
)


# -----------------------------------------------------------------------------
# Defaults copied from the current standalone afforestation workflow.
# They can all be overridden from the command line.
# -----------------------------------------------------------------------------
DEFAULT_SCENARIO_ROOT = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\afforestation"
)
DEFAULT_OUTPUT_ROOT = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output"
)
DEFAULT_CROP_PRICE_FILE = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\data\landrent"
) / "IMAGE_crop_prices.xlsx"

# Canonical order for the FAOSTAT workbook. These are the MAGNET short codes;
# the first 26 IMAGE regions use the corresponding positions. Do not match
# cells on these text labels; region_to_grid('5m') provides zero-based IDs.
IMAGE26_PRICE_REGION_CODES = (
    "CAN", "USA", "MEX", "CAM", "BRA", "RSM", "NAF", "WAF",
    "EAF", "ZAF", "WEU", "CEU", "TUR", "UKR", "CAS", "RUR",
    "ME", "IND", "KOR", "CHR", "SEA", "IDN", "JAP", "OCE",
    "RSA", "RSF",
)

IMAGE16_PRICE_COLUMNS = (
    "01_Wheat", "02_Rice", "03_Maize", "04_Tropical cereals",
    "05_Other temperate cereals", "06_Pulses", "07_Soybeans",
    "08_Temperate oil crops", "09_Tropical oil crops",
    "10_Temperate roots & tubers", "11_Tropical roots & tubers",
    "12_Sugar Crops", "13_Oil & palm fruit", "14_Vegetables & fruits",
    "15_Other non-food & luxury & spices", "16_Plant based fibres",

)

# Raw MAGNET uses WER/CER for the two European regions whereas the IMAGE/price
# workbook uses WEU/CEU. All other short codes align directly.
MAGNET_TO_IMAGE_REGION = {
    "WER": "WEU",
    "CER": "CEU",
}

MAGNET_CROP_LAND_ITEM = "ARABLE&PERMCR"
MAGNET_PASTURE_LAND_ITEM = "livestock_rum"
MAGNET_AGRIPRIM_ITEM = "AGRI_PRIM"

# The original IMAGE/MATLAB LUT maps M2I values to a 1970-2100, five-year
# IMAGE timeline. 2017 is not used; 2019 is treated as the pre-2020 anchor.
M2I_LUT_SOURCE_YEARS = (2019, 2025, 2030, 2035, 2040, 2045, 2050,
                        2060, 2070, 2080, 2090, 2100)
M2I_LUT_TARGET_1BASED = tuple(list(range(11, 18)) + list(range(19, 28, 2)))


def load_image26_crop_prices(workbook_path: Path | str) -> np.ndarray:
    """Read/validate FAOSTAT regional prices from `RegionalWide16`.

    Returns
    -------
    np.ndarray, shape (26, 16), dtype float32
        Regional prices in constant 2010 USD/tonne, in zero-based IMAGE
        region order and crop IDs 1..16. All agricultural values must be
        finite and positive. Grass and bioenergy are NOT in this workbook.
    """
    path = Path(workbook_path)
    if not path.is_file():
        raise FileNotFoundError(f"FAOSTAT crop-price workbook not found: {path}")

    # Work with both the original multi-sheet validation workbook and a
    # stand-alone 26 x 16 worksheet exported as IMAGE_crop_prices.xlsx.
    with pd.ExcelFile(path) as xl:
        if "RegionalWide16" in xl.sheet_names:
            price_sheet = "RegionalWide16"
        elif len(xl.sheet_names) == 1:
            price_sheet = xl.sheet_names[0]
            print(f"Reading single-sheet IMAGE price workbook: {price_sheet!r}")
        else:
            raise ValueError(
                f"{path} has no 'RegionalWide16' sheet and contains multiple "
                f"worksheets: {xl.sheet_names}. Set the price sheet name "
                "to RegionalWide16 or supply a one-sheet price workbook."
            )
        df = pd.read_excel(xl, sheet_name=price_sheet)
    required = ["image_region_number", "image_region_index0", "image_region", *IMAGE16_PRICE_COLUMNS]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Crop-price sheet 'RegionalWide16' is missing columns: {missing}")
    if len(df) != 26:
        raise ValueError(f"Expected 26 IMAGE regions; got {len(df)} in {path}")

    ids0 = pd.to_numeric(df["image_region_index0"], errors="raise")
    ids1 = pd.to_numeric(df["image_region_number"], errors="raise")
    if not ids0.notna().all() or not ids1.notna().all():
        raise ValueError("IMAGE region indices contain missing values")
    if not np.array_equal(np.sort(ids0.to_numpy()), np.arange(26)):
        raise ValueError("image_region_index0 must enumerate every integer from 0 to 25 exactly once")
    if not np.array_equal(ids1.to_numpy(), ids0.to_numpy() + 1):
        raise ValueError("image_region_number must equal image_region_index0 + 1")

    df = df.assign(_index0=ids0.to_numpy(dtype=int)).sort_values("_index0")
    labels = tuple(df["image_region"].astype(str).str.strip())
    if labels != IMAGE26_PRICE_REGION_CODES:
        wrong = [(i, got, exp) for i, (got, exp) in enumerate(zip(labels, IMAGE26_PRICE_REGION_CODES))
                 if got != exp]
        raise ValueError(f"Crop-price region codes do not match expected IMAGE26 order: {wrong}")

    # Series are parsed numerically so a blank, text, Excel error or NaN
    # cannot silently become a zero economic price.
    matrix = df.loc[:, list(IMAGE16_PRICE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float64)
    invalid = ~np.isfinite(matrix) | (matrix <= 0)
    if invalid.any():
        r, c = np.argwhere(invalid)[0]
        raise ValueError(
            f"Invalid regional crop price: region={IMAGE26_PRICE_REGION_CODES[r]}, "
            f"crop={IMAGE16_PRICE_COLUMNS[c]}, value={matrix[r, c]}"
        )
    return matrix.astype(np.float32)


def agricultural_prices_to_crop38(regional_prices: np.ndarray) -> np.ndarray:
    """Return IMAGE Crop38 x Region26 crop-price array.

    Crop IDs 1..16 are rainfed; IDs 22..37 are their irrigated equivalents.
    Grass (ID 0) and dedicated bioenergy crops (17..21) deliberately remain
    zero because pasture opportunity cost is calibrated directly to MAGNET's
    ``livestock_rum`` land price rather than through a synthetic grass price.
    """
    regional_prices = np.asarray(regional_prices, dtype=np.float32)
    if regional_prices.shape != (26, 16):
        raise ValueError(
            f"Expected regional_prices (26, 16), got {regional_prices.shape}"
        )
    if not np.isfinite(regional_prices).all() or (regional_prices <= 0).any():
        raise ValueError("All agricultural crop prices must be finite and positive")

    result = np.zeros((38, 26), dtype=np.float32)
    result[1:17, :] = regional_prices.T
    result[22:38, :] = regional_prices.T
    return result


DEFAULT_MAGNET_SCENS = [
    "M2I_SSP2_nocap_2024-12-02",
    "M2I_SSP2_affor7p5_2024-12-02",
    "M2I_SSP2_affor15_2024-12-02",
    "M2I_SSP2_affor22p5_2024-12-02",
    "M2I_SSP2_affor30_2024-12-02",
]


# =============================================================================
# Optional MAGNET land-rent LUT preprocessing
# =============================================================================

def _expected_m2i_scenario_name(basename: str) -> str | None:
    """Return the internal scenario name implied by an M2I file basename."""
    match = re.match(r"^M2I_(.+?)_\d{4}-\d{2}-\d{2}$", basename)
    return match.group(1) if match else None


def _read_full_m2i_table(path: Path) -> pd.DataFrame:
    """Read the columns needed for sectoral land-price extraction.

    CSV is preferred when available because it is much faster than Excel. For
    an Excel workbook, inspect worksheets and select a full M2I sheet containing
    Variable/Item/Region/Year/Value, not the old AGRI_PRIM-only
    ``landprice_selected`` helper sheet.
    """
    required = {"Variable", "Item", "Region", "Year", "Value"}

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(
            path,
            usecols=lambda c: c in required | {"Scenario", "Type"},
            low_memory=False,
        )
    elif path.suffix.lower() in (".xlsx", ".xlsm"):
        with pd.ExcelFile(path) as xl:
            candidate_sheet = None
            for sheet in xl.sheet_names:
                if sheet.lower() in {"landprice_selected", "mapping"}:
                    continue
                header = pd.read_excel(xl, sheet_name=sheet, nrows=0)
                if required.issubset(set(header.columns)):
                    candidate_sheet = sheet
                    break
            if candidate_sheet is None:
                raise ValueError(
                    f"{path} has no full M2I worksheet with columns "
                    f"{sorted(required)}. The AGRI_PRIM-only landprice_selected "
                    "sheet is insufficient for separate crop/pasture rents."
                )
            df = pd.read_excel(
                xl,
                sheet_name=candidate_sheet,
                usecols=lambda c: c in required | {"Scenario", "Type"},
            )
    else:
        raise ValueError(f"Unsupported M2I file format: {path}")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required M2I columns {sorted(missing)}")

    df = df.copy()
    df["Variable"] = df["Variable"].astype(str).str.strip()
    df["Item"] = df["Item"].astype(str).str.strip()
    df["Region"] = (
        df["Region"].astype(str).str.strip().replace(MAGNET_TO_IMAGE_REGION)
    )
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df[df["Year"].notna() & np.isfinite(df["Value"])].copy()
    df["Year"] = df["Year"].astype(int)
    return df


def _load_m2i_for_scenario(scen_dir: Path, basename: str) -> tuple[pd.DataFrame, Path]:
    """Load a complete M2I scenario file.

    Exact basename matches are tried first (CSV, then Excel). If those are
    incomplete or internally mislabeled, all M2I files in the directory are
    scanned as a fallback and the file whose internal ``Scenario`` matches the
    requested scenario is used. This handles historical exports where an XLSX
    filename and its internal scenario label disagree.
    """
    required_items = {
        MAGNET_CROP_LAND_ITEM,
        MAGNET_PASTURE_LAND_ITEM,
        MAGNET_AGRIPRIM_ITEM,
    }
    expected_scen = _expected_m2i_scenario_name(basename)
    errors: list[str] = []
    tried: set[Path] = set()

    def inspect(path: Path, *, require_internal_match: bool) -> tuple[pd.DataFrame, Path]:
        tried.add(path)
        df = _read_full_m2i_table(path)
        lp_items = set(
            df.loc[df["Variable"].eq("LandPricesReal"), "Item"].dropna()
        )
        missing = required_items - lp_items
        if missing:
            raise ValueError(
                f"missing LandPricesReal item(s) {sorted(missing)}"
            )

        if expected_scen and "Scenario" in df.columns:
            actual = {
                str(v).strip()
                for v in df["Scenario"].dropna().unique()
                if str(v).strip()
            }
            if actual and actual != {expected_scen}:
                raise ValueError(
                    f"internal Scenario={sorted(actual)} does not match "
                    f"requested {expected_scen!r}"
                )
        elif require_internal_match and expected_scen:
            raise ValueError(
                "fallback candidate has no Scenario column, so its internal "
                "scenario identity cannot be verified"
            )
        return df, path

    # Fast path: exact expected filename.
    for suffix in (".csv", ".xlsx", ".xlsm"):
        path = scen_dir / f"{basename}{suffix}"
        if not path.is_file():
            continue
        try:
            return inspect(path, require_internal_match=False)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    # Robust fallback for historical/mislabeled M2I exports.
    if expected_scen:
        candidates = sorted(
            set(scen_dir.glob("M2I_*.csv"))
            | set(scen_dir.glob("M2I_*.xlsx"))
            | set(scen_dir.glob("M2I_*.xlsm"))
        )
        for path in candidates:
            if path in tried:
                continue
            try:
                df, source = inspect(path, require_internal_match=True)
                print(
                    f"WARNING: requested {basename!r} but using {source.name!r} "
                    f"because its internal Scenario is {expected_scen!r}."
                )
                return df, source
            except Exception as exc:
                errors.append(f"{path.name}: {exc}")

    raise FileNotFoundError(
        f"Could not load a complete M2I dataset for {basename!r} in {scen_dir}. "
        + (" Attempts: " + " | ".join(errors) if errors else "")
    )


def _extract_m2i_landprice_matrix(
    df: pd.DataFrame,
    item: str,
    *,
    years: tuple[int, ...] = M2I_LUT_SOURCE_YEARS,
) -> np.ndarray:
    """Return [year, IMAGE26 region] LandPricesReal values for one MAGNET item."""
    sub = df[
        df["Variable"].eq("LandPricesReal")
        & df["Item"].eq(item)
        & df["Year"].isin(years)
        & df["Region"].isin(IMAGE26_PRICE_REGION_CODES)
    ][["Region", "Year", "Value"]].copy()

    if sub.duplicated(["Region", "Year"]).any():
        dup = sub[sub.duplicated(["Region", "Year"], keep=False)]
        raise ValueError(
            f"Duplicate LandPricesReal observations for {item}:\n"
            + dup.head(20).to_string(index=False)
        )

    pivot = sub.pivot(index="Year", columns="Region", values="Value")
    pivot = pivot.reindex(index=list(years), columns=list(IMAGE26_PRICE_REGION_CODES))
    if pivot.isna().any().any():
        where = np.argwhere(pivot.isna().to_numpy())
        iy, ir = where[0]
        raise ValueError(
            f"Incomplete MAGNET {item} land-price series: "
            f"year={years[iy]}, region={IMAGE26_PRICE_REGION_CODES[ir]}"
        )

    values = pivot.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"MAGNET {item} contains nonfinite/nonpositive land prices")
    return values.astype(np.float32)


def _legacy_image_time_expand(source: np.ndarray) -> np.ndarray:
    """Map 2019-2100 M2I values to the 27-step legacy IMAGE LUT timeline.

    This reproduces the current A/R/MATLAB convention exactly: the 2019 value is
    placed at the IMAGE 2020 anchor and extended backward; intermediate years
    are linearly interpolated.
    """
    source = np.asarray(source, dtype=np.float32)
    if source.shape[0] != len(M2I_LUT_SOURCE_YEARS):
        raise ValueError(
            f"Expected {len(M2I_LUT_SOURCE_YEARS)} source years, got {source.shape}"
        )

    out = np.zeros((27,) + source.shape[1:], dtype=np.float32)
    target0 = np.asarray(M2I_LUT_TARGET_1BASED, dtype=int) - 1
    out[target0, ...] = source

    # IMAGE/MATLAB convention: pre-2020 years use the 2019/2020 anchor.
    out[0, ...] = out[target0[0], ...]

    x_known = np.asarray((1,) + M2I_LUT_TARGET_1BASED, dtype=float)
    xi = np.arange(1, 28, dtype=float)
    flat = out.reshape(27, -1)
    for col in range(flat.shape[1]):
        flat[:, col] = np.interp(xi, x_known, flat[x_known.astype(int) - 1, col])
    return flat.reshape(out.shape)


def _check_agriprim_reconciliation(df: pd.DataFrame, label: str) -> None:
    """Check that AGRI_PRIM is the land-demand-weighted crop/ruminant price."""
    keys = ["Region", "Year"]

    def series(variable: str, item: str, name: str) -> pd.DataFrame:
        x = df[
            df["Variable"].eq(variable)
            & df["Item"].eq(item)
            & df["Region"].isin(IMAGE26_PRICE_REGION_CODES)
        ][keys + ["Value"]].copy()
        if x.empty:
            return x
        if x.duplicated(keys).any():
            raise ValueError(f"{label}: duplicate {variable}/{item} records")
        return x.rename(columns={"Value": name})

    p_agri = series("LandPricesReal", MAGNET_AGRIPRIM_ITEM, "p_agri")
    p_crop = series("LandPricesReal", MAGNET_CROP_LAND_ITEM, "p_crop")
    p_rum = series("LandPricesReal", MAGNET_PASTURE_LAND_ITEM, "p_rum")
    d_crop = series("LandDemand", MAGNET_CROP_LAND_ITEM, "d_crop")
    d_rum = series("LandDemand", MAGNET_PASTURE_LAND_ITEM, "d_rum")

    if any(x.empty for x in (p_agri, p_crop, p_rum, d_crop, d_rum)):
        print(
            f"WARNING: {label}: LandDemand or AGRI_PRIM records are incomplete; "
            "skipping MAGNET aggregate-rent reconciliation."
        )
        return

    m = p_agri.merge(p_crop, on=keys).merge(p_rum, on=keys)
    m = m.merge(d_crop, on=keys).merge(d_rum, on=keys)
    denom = m["d_crop"] + m["d_rum"]
    valid = np.isfinite(denom) & (denom > 0)
    reconstructed = (
        m.loc[valid, "d_crop"] * m.loc[valid, "p_crop"]
        + m.loc[valid, "d_rum"] * m.loc[valid, "p_rum"]
    ) / denom[valid]
    observed = m.loc[valid, "p_agri"]
    rel = np.abs(reconstructed - observed) / np.maximum(np.abs(observed), 1e-30)
    max_rel = float(rel.max()) if len(rel) else np.nan
    print(
        f"MAGNET AGRI_PRIM reconciliation {label}: "
        f"max relative error={max_rel:.3e} over {len(rel)} region-years"
    )
    if np.isfinite(max_rel) and max_rel > 1e-4:
        raise ValueError(
            f"{label}: AGRI_PRIM does not reconcile with ARABLE&PERMCR and "
            f"livestock_rum (max relative error {max_rel:.3e})."
        )


def build_landrent_lut(
    magnet_scens: List[str],
    land_supply_file: str,
    scen_dir: Path | str,
    output_file: Path | str | None = None,
) -> Path:
    """Build separate cropland and ruminant/pasture MAGNET land-rent LUTs.

    The same five land-supply-reduction scenarios are used for both sectors.
    ``AGRI_PRIM`` is retained as a diagnostic/legacy series only.

    Saved arrays
    ------------
    cropland_lut : [27, 26, 401]
        LandPricesReal - ARABLE&PERMCR.
    pasture_lut : [27, 26, 401]
        LandPricesReal - livestock_rum.
    agriprim_lut / landrent_lut : [27, 26, 401]
        Aggregate agricultural rent retained for reconciliation/backward
        compatibility, not used to calculate the new gridded opportunity costs.
    aff_land_reds_lut : [27, 401]
        Afforestation-induced agricultural land-supply reduction coordinate.
    """
    scen_dir = Path(scen_dir)
    if output_file is None:
        output_file = scen_dir / "sector_landrent_LUT_input_data.mat"
    else:
        output_file = Path(output_file)

    nscens = len(magnet_scens)
    if nscens != 5:
        raise ValueError(
            "The current land-supply interpolation expects exactly five MAGNET "
            f"scenarios; got {nscens}."
        )

    nregions = len(Region26.coords)
    raw_crop = np.zeros(
        (len(M2I_LUT_SOURCE_YEARS), nregions, nscens), dtype=np.float32
    )
    raw_pasture = np.zeros_like(raw_crop)
    raw_agriprim = np.zeros_like(raw_crop)

    for sc_idx, scen in enumerate(magnet_scens):
        df, source_path = _load_m2i_for_scenario(scen_dir, scen)
        print(f"Reading sectoral MAGNET land prices: {source_path}")
        _check_agriprim_reconciliation(df, scen)

        raw_crop[:, :, sc_idx] = _extract_m2i_landprice_matrix(
            df, MAGNET_CROP_LAND_ITEM
        )
        raw_pasture[:, :, sc_idx] = _extract_m2i_landprice_matrix(
            df, MAGNET_PASTURE_LAND_ITEM
        )
        raw_agriprim[:, :, sc_idx] = _extract_m2i_landprice_matrix(
            df, MAGNET_AGRIPRIM_ITEM
        )

    # Preserve the established M2I -> IMAGE land-rent unit conversion.
    raw_crop *= 1e6
    raw_pasture *= 1e6
    raw_agriprim *= 1e6

    crop_time = _legacy_image_time_expand(raw_crop)
    pasture_time = _legacy_image_time_expand(raw_pasture)
    agriprim_time = _legacy_image_time_expand(raw_agriprim)

    aff_land_reds = np.zeros((27, 5), dtype=np.float32)
    land_supply_file_path = land_data_dir / land_supply_file
    df_reds = pd.read_excel(
        land_supply_file_path,
        sheet_name="land_supply_reductions_v2024",
        usecols="B:F",
        header=None,
        skiprows=1,
        nrows=20,
        engine="openpyxl",
    )
    arr_reds = df_reds.to_numpy(dtype=np.float32)
    if arr_reds.shape != (20, 5) or not np.all(np.isfinite(arr_reds)):
        raise ValueError(
            f"Unexpected land-supply-reduction table shape/data: {arr_reds.shape}"
        )
    aff_land_reds[7:27, :] = arr_reds

    aff_land_reds_lut = np.empty((27, 401), dtype=np.float32)
    cropland_lut = np.empty((27, nregions, 401), dtype=np.float32)
    pasture_lut = np.empty_like(cropland_lut)
    agriprim_lut = np.empty_like(cropland_lut)

    vars1 = np.arange(1, 402, 100)
    xi_lut = np.arange(1, 402)

    for yr in range(27):
        aff_land_reds_lut[yr, :] = np.interp(
            xi_lut, vars1, aff_land_reds[yr, :]
        )
        for reg in range(nregions):
            cropland_lut[yr, reg, :] = np.interp(
                xi_lut, vars1, crop_time[yr, reg, :]
            )
            pasture_lut[yr, reg, :] = np.interp(
                xi_lut, vars1, pasture_time[yr, reg, :]
            )
            agriprim_lut[yr, reg, :] = np.interp(
                xi_lut, vars1, agriprim_time[yr, reg, :]
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        output_file,
        {
            "cropland_lut": cropland_lut,
            "pasture_lut": pasture_lut,
            "agriprim_lut": agriprim_lut,
            # Keep the legacy key so old diagnostic code can still inspect it.
            "landrent_lut": agriprim_lut,
            "aff_land_reds_lut": aff_land_reds_lut,
        },
        do_compression=True,
    )
    return output_file


# =============================================================================
# Spatial MAGNET land-rent downscaling (pure numerical helper)
# =============================================================================

def downscale_magnet_landrent(
    scaler: np.ndarray,
    regions: np.ndarray,
    forest_management: np.ndarray,
    regional_lut: np.ndarray,
    reduction_lut: np.ndarray,
    land_reduction_percent: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the MAGNET rent lookup and grid downscaling in AfforAllocSA.

    For each IMAGE region, find the nearest land-reduction LUT percentage,
    multiply that region's MAGNET land rent by the spatial ``scaler``.
    In GFORMAN == 4 cells use the baseline (first) MAGNET LUT column,
    matching the current afforestation allocation code.

    Returns
    -------
    grid_rent : np.ndarray
        Spatial MAGNET rent (same units as the MAGNET LUT).
    selected_lut_index : np.ndarray
        Zero-based selected LUT column, one per region. GFORMAN == 4
        cells still use column zero regardless of the returned index.
    """
    regional_lut = np.asarray(regional_lut)
    reduction_lut = np.asarray(reduction_lut).reshape(-1)
    land_reduction_percent = np.asarray(land_reduction_percent).reshape(-1)
    scaler = np.asarray(scaler)
    regions = np.asarray(regions)
    forest_management = np.asarray(forest_management)

    if regional_lut.ndim != 2:
        raise ValueError(f"Expected regional LUT [region, bin]; got {regional_lut.shape}")
    n_regions, n_bins = regional_lut.shape
    if reduction_lut.shape != (n_bins,):
        raise ValueError("Reduction and land-rent LUT bins have inconsistent sizes")
    if land_reduction_percent.shape != (n_regions,):
        raise ValueError(f"Expected {n_regions} land-reduction percentages")
    if scaler.shape != regions.shape or scaler.shape != forest_management.shape:
        raise ValueError("Cell scaler, region and forest-management shapes differ")
    if not np.all(np.isfinite(land_reduction_percent)):
        raise ValueError("Nonfinite land-reduction percentages")
    if not np.all(np.isfinite(reduction_lut)) or not np.all(np.isfinite(regional_lut)):
        raise ValueError("Nonfinite MAGNET LUT data")

    # This nearest-bin selection is intentionally the same as AfforAllocSA.
    selected = np.argmin(
        np.abs(reduction_lut[None, :] - land_reduction_percent[:, None]),
        axis=1,
    ).astype(np.intp)

    grid_rent = np.zeros(scaler.shape, dtype=np.float32)
    valid = (regions >= 0) & (regions < n_regions)
    if np.any(valid):
        rr = regions[valid].astype(np.intp, copy=False)
        actual_rent = regional_lut[rr, selected[rr]]
        reference_rent = regional_lut[rr, 0]
        grid_rent[valid] = (
            scaler[valid]
            * np.where(forest_management[valid] == 4, reference_rent, actual_rent)
        ).astype(np.float32)

    return grid_rent, selected


# =============================================================================
# Standalone land-rent PRISM model
# =============================================================================

@prism.interface
class LandRentSA(prism.Model):
    """Calculate separate gridded cropland and ruminant/pasture opportunity costs."""

    def configure_paths(
        self,
        scen_dir: Path | str,
        output_dir: Path | str | None = None,
    ) -> None:
        self.scen_dir = Path(scen_dir)
        self.output_dir = Path.cwd() if output_dir is None else Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def configure_crop_prices(
        self,
        crop_price_file: Path | str = DEFAULT_CROP_PRICE_FILE,
    ) -> None:
        """Select the FAOSTAT IMAGE26 crop-price workbook."""
        self.crop_price_file = Path(crop_price_file)

    def configure_land_reduction(
        self,
        lut_file: Path | str,
        land_reduction_percent: float = 0.0,
        affor_area_nc: Path | str | None = None,
    ) -> None:
        """Select the sectoral MAGNET LUT and exogenous/A-R land reduction."""
        self.lut_file = Path(lut_file)
        self.land_reduction_percent = float(land_reduction_percent)
        self.affor_area_nc = Path(affor_area_nc) if affor_area_nc else None
        if not np.isfinite(self.land_reduction_percent):
            raise ValueError("--land-reduction-percent must be finite")
        if self.land_reduction_percent < 0:
            raise ValueError("--land-reduction-percent must not be negative")

    # Spatial production/productivity proxies and normalized sector scalers.
    crop_value_proxy: prism.TimeVariable[LandGrid, np.float32] = prism.export()
    crop_rent_scaler: prism.TimeVariable[LandGrid, np.float32] = prism.export()
    pasture_productivity_proxy: prism.TimeVariable[LandGrid, np.float32] = prism.export()
    pasture_rent_scaler: prism.TimeVariable[LandGrid, np.float32] = prism.export()

    # Sector-specific per-sector-hectare rents and whole-cell opportunity cost.
    CropLandRent: prism.TimeVariable[LandGrid, np.float32] = prism.export()
    PastureLandRent: prism.TimeVariable[LandGrid, np.float32] = prism.export()
    LandRent: prism.TimeVariable[LandGrid, np.float32] = prism.export()

    # Static/reference inputs retained from the existing crop-value downscaler.
    MF_2010: prism.Array[Crop38, Region26, np.float32] = prism.private()
    frharvtot_2015: prism.Array[Crop38, Region26, np.float32] = prism.private()
    cropprice: prism.Array[Crop38, Region26, np.float32] = prism.private()
    region_cells: prism.Array[LandGrid, np.int16] = prism.private()

    def compute_initial_values(self, _: prism.Timeline) -> None:
        """Load sectoral MAGNET LUTs, crop calibration inputs and optional A/R area."""
        if not hasattr(self, "lut_file"):
            raise RuntimeError("Call configure_land_reduction() before simulate()")
        if not self.lut_file.is_file():
            raise FileNotFoundError(
                f"Sectoral MAGNET land-rent LUT not found: {self.lut_file}. "
                "Use --rebuild-lut to build one from full M2I scenario files."
            )

        lut_data = loadmat(self.lut_file)
        required = {
            "cropland_lut",
            "pasture_lut",
            "aff_land_reds_lut",
        }
        missing = required - set(lut_data)
        if missing:
            raise KeyError(
                f"{self.lut_file} is an old/unsupported LUT and lacks "
                f"{sorted(missing)}. Rebuild it with this module using --rebuild-lut."
            )

        self._lut_crop = np.asarray(lut_data["cropland_lut"], dtype=np.float32)
        self._lut_pasture = np.asarray(lut_data["pasture_lut"], dtype=np.float32)
        self._lut_agriprim = (
            np.asarray(lut_data["agriprim_lut"], dtype=np.float32)
            if "agriprim_lut" in lut_data
            else None
        )
        self._lut_reduction = np.asarray(
            lut_data["aff_land_reds_lut"], dtype=np.float32
        )

        nregions = len(Region26.coords)
        expected_shape = self._lut_crop.shape
        if (
            self._lut_crop.ndim != 3
            or self._lut_crop.shape[1] != nregions
            or self._lut_pasture.shape != expected_shape
            or self._lut_reduction.shape
            != (self._lut_crop.shape[0], self._lut_crop.shape[2])
        ):
            raise ValueError(
                "Unexpected sectoral MAGNET LUT shapes: "
                f"crop={self._lut_crop.shape}, pasture={self._lut_pasture.shape}, "
                f"reductions={self._lut_reduction.shape}"
            )
        if self._lut_agriprim is not None and self._lut_agriprim.shape != expected_shape:
            raise ValueError(
                f"AGRI_PRIM diagnostic LUT shape {self._lut_agriprim.shape} "
                f"does not match {expected_shape}"
            )

        self._affor_area_by_year = None
        if self.affor_area_nc is not None:
            if not self.affor_area_nc.is_file():
                raise FileNotFoundError(f"A/R area NetCDF not found: {self.affor_area_nc}")
            with xr.open_dataset(self.affor_area_nc) as ds:
                if "AfforArea" not in ds:
                    raise KeyError("A/R NetCDF must contain variable 'AfforArea'")
                da = ds["AfforArea"]
                if "time" not in da.dims:
                    raise ValueError("AfforArea must contain a 'time' dimension")
                region_dims = [d for d in da.dims if d != "time"]
                if len(region_dims) != 1:
                    raise ValueError("Expected AfforArea with time and region dimensions")
                da = da.transpose("time", region_dims[0])
                values = np.asarray(da.values, dtype=np.float64)
                if values.shape[1] < nregions or not np.all(np.isfinite(values)):
                    raise ValueError("AfforArea must have at least 26 finite region values")
                time_coords = np.asarray(da["time"].values)
                if np.issubdtype(time_coords.dtype, np.datetime64):
                    years = np.asarray(da["time"].dt.year.values, dtype=int)
                else:
                    years = np.asarray(time_coords, dtype=int)
                self._affor_area_by_year = {
                    int(year): values[i, :nregions]
                    for i, year in enumerate(years)
                }

        var_timeline = prism.Timeline(
            start=prism.Q_(1970, "year"),
            end=prism.Q_(2100, "year"),
            stepsize=prism.Q_(1, "year"),
        )
        mf_tmp = prism.TimeVariable[Crop22, Region26, np.float32](
            file=self.scen_dir / "MF.scn",
            timeline=var_timeline,
        )
        frharv = prism.TimeVariable[Region, np.float32](
            file=self.scen_dir / "FRHARV.SCN",
            timeline=var_timeline,
        )
        frharvmapspam = prism.Array[HarvestGroup3, Crop17, Region, np.float32](
            file=self.scen_dir / "FRHARVMAPSPAM.SCN"
        )

        t_mf = prism.Q_(2010, "year")
        t_frharv = prism.Q_(2015, "year")
        self.region_cells.values, _ = region_to_grid("5m")

        self.MF_2010.values[:] = 0.0
        self.frharvtot_2015.values[:] = 0.0
        self.cropprice.values[:] = 0.0

        self.MF_2010.values[1:17, :] = mf_tmp[t_mf].values[1:17, :]
        self.MF_2010.values[22:38, :] = mf_tmp[t_mf].values[1:17, :]

        frharv_values = np.asarray(
            frharv[t_frharv].values[:26], dtype=np.float32
        )
        frharvmapspam_values = np.asarray(
            frharvmapspam.values[:, :, :26], dtype=np.float32
        )
        self.frharvtot_2015.values[1:17, :] = (
            frharvmapspam_values[0, 1:17, :] * frharv_values[None, :]
        )
        self.frharvtot_2015.values[22:38, :] = (
            frharvmapspam_values[1, 1:17, :] * frharv_values[None, :]
        )

        if not hasattr(self, "crop_price_file"):
            raise RuntimeError("Call configure_crop_prices() before simulate()")
        regional_prices = load_image26_crop_prices(self.crop_price_file)
        self.cropprice.values[:] = agricultural_prices_to_crop38(regional_prices)
        print(f"Regional FAOSTAT crop prices loaded: {self.crop_price_file}")
        print(
            "Pasture price: no synthetic grass commodity price is used; "
            "the regional economic level comes directly from MAGNET "
            f"LandPricesReal-{MAGNET_PASTURE_LAND_ITEM}."
        )

    @staticmethod
    def _sanitize_fraction(values: np.ndarray) -> np.ndarray:
        """Convert nonfinite/negative land fractions to zero without hiding >1 values."""
        x = np.asarray(values, dtype=np.float32)
        return np.where(np.isfinite(x) & (x > 0.0), x, 0.0).astype(np.float32)

    def _normalize_sector_proxy(
        self,
        proxy: np.ndarray,
        fraction: np.ndarray,
        areas: np.ndarray,
        valid_region_mask: np.ndarray,
        *,
        label: str,
        extra_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Normalize a per-sector-ha proxy with sector hectares as weights.

        ``extra_mask`` can receive the calibrated scaler without contributing to
        the regional calibration mean. It is used for the GFORMAN==4
        counterfactual pasture cells.
        """
        region_cells = self.region_cells.values
        proxy = np.asarray(proxy, dtype=np.float64)
        fraction = np.asarray(fraction, dtype=np.float64)
        areas = np.asarray(areas, dtype=np.float64)
        scaler = np.zeros(proxy.shape, dtype=np.float32)

        for reg in Region26:
            r = int(reg)
            calibration = (
                valid_region_mask
                & (region_cells == r)
                & np.isfinite(proxy)
                & (proxy >= 0.0)
                & np.isfinite(fraction)
                & (fraction > 0.0)
                & np.isfinite(areas)
                & (areas > 0.0)
            )
            if np.any(calibration):
                weights = areas[calibration] * fraction[calibration]
                mean_val = np.average(proxy[calibration], weights=weights)
                if np.isfinite(mean_val) and mean_val > 0.0:
                    scaler[calibration] = (
                        proxy[calibration] / mean_val
                    ).astype(np.float32)

                    if extra_mask is not None:
                        extra = (
                            valid_region_mask
                            & (region_cells == r)
                            & extra_mask
                            & np.isfinite(proxy)
                            & (proxy > 0.0)
                        )
                        scaler[extra] = (proxy[extra] / mean_val).astype(np.float32)
            else:
                extra = (
                    extra_mask is not None
                    and np.any(valid_region_mask & (region_cells == r) & extra_mask)
                )
                if extra:
                    # No actual sector hectares exist to calibrate a spatial
                    # pattern. Preserve the regional MAGNET level on the
                    # counterfactual cells rather than assigning zero.
                    fallback = valid_region_mask & (region_cells == r) & extra_mask
                    scaler[fallback] = 1.0
                    print(
                        f"WARNING: {label}: region {r} has counterfactual cells "
                        "but no positive actual sector hectares; using scaler=1."
                    )
                else:
                    print(
                        f"WARNING: {label}: no positive sector proxy in region {r}."
                    )
        return scaler

    def compute_values(
        self,
        time: prism.Time,
        GLCT: prism.Array[LandGrid, np.int8],
        GFORMAN: prism.Array[LandGrid, np.int8],
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GRPPC: prism.Array[LandGrid, Crop38, np.float32],
        GAREA5m: prism.Array[LandGrid, np.float32],
        GrazIntens: prism.Array[GrassLand, Region26, np.float32],
    ) -> None:
        crop_fraction, pasture_fraction, counterfactual_pasture = (
            self._calculate_spatial_scalers(
                time=time,
                GLCT=GLCT,
                GFORMAN=GFORMAN,
                GFRAC=GFRAC,
                GRPPC=GRPPC,
                GAREA5m=GAREA5m,
                GrazIntens=GrazIntens,
            )
        )
        self._calculate_magnet_landrent(
            time=time,
            GLCT=GLCT,
            GFORMAN=GFORMAN,
            GAREA5m=GAREA5m,
            crop_fraction=crop_fraction,
            pasture_fraction=pasture_fraction,
            counterfactual_pasture=counterfactual_pasture,
        )
        self._write_results_to_files(time=time)

    def _calculate_spatial_scalers(
        self,
        time: prism.Time,
        GLCT: prism.Array[LandGrid, np.int8],
        GFORMAN: prism.Array[LandGrid, np.int8],
        GFRAC: prism.Array[LandGrid, Crop38, np.float32],
        GRPPC: prism.Array[LandGrid, Crop38, np.float32],
        GAREA5m: prism.Array[LandGrid, np.float32],
        GrazIntens: prism.Array[GrassLand, Region26, np.float32],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build crop-value and pasture-productivity scalers per sector hectare."""
        t = time.t
        region_cells = self.region_cells.values
        ngrid = len(LandGrid.coords)
        valid_region = (
            (region_cells >= 0) & (region_cells < len(Region26.coords))
        )
        agri_land = valid_region & (GLCT.values == 1)
        extensive_grass = valid_region & (GLCT.values == 2)

        print(f"Starting sectoral land-rent scalers for year {t.magnitude}")

        # ------------------------------------------------------------------
        # CROPLAND: production value per hectare of crop land.
        # ------------------------------------------------------------------
        crop_area_fraction = np.zeros(ngrid, dtype=np.float32)
        crop_value_cell = np.zeros(ngrid, dtype=np.float64)

        crop_ids = tuple(range(1, 17)) + tuple(range(22, 38))
        for c in crop_ids:
            frac = self._sanitize_fraction(GFRAC.values[:, c])
            mask = agri_land & (frac > 0.0)
            if not np.any(mask):
                continue

            reg = region_cells
            mf = np.take(self.MF_2010.values[c, :], reg, mode="clip")
            frh = np.take(self.frharvtot_2015.values[c, :], reg, mode="clip")
            mf[reg < 0] = 0.0
            frh[reg < 0] = 0.0
            yld = np.asarray(GRPPC.values[:, c], dtype=np.float64)
            price = self.cropprice.values[c, reg[mask]]

            # The crop fraction is a land-area quantity and is retained even
            # where a productivity input is zero/missing. Invalid productivity
            # therefore contributes zero value rather than silently removing
            # crop hectares from the MAGNET calibration population.
            crop_area_fraction[mask] += frac[mask]

            valid = (
                mask
                & np.isfinite(yld) & (yld >= 0.0)
                & np.isfinite(mf) & (mf >= 0.0)
                & np.isfinite(frh) & (frh >= 0.0)
            )
            if np.any(valid):
                crop_value_cell[valid] += (
                    frac[valid]
                    * yld[valid]
                    * mf[valid]
                    * frh[valid]
                    * self.cropprice.values[c, reg[valid]]
                )

        crop_value_per_ha = np.zeros(ngrid, dtype=np.float32)
        crop_positive = crop_area_fraction > 0.0
        crop_value_per_ha[crop_positive] = np.divide(
            crop_value_cell[crop_positive],
            crop_area_fraction[crop_positive],
        ).astype(np.float32)

        # ------------------------------------------------------------------
        # PASTURE: productivity per hectare of pasture.
        # GLCT==1 uses its grass fraction; GLCT==2 is full-cell pasture.
        # ------------------------------------------------------------------
        pasture_fraction = np.zeros(ngrid, dtype=np.float32)
        grass_frac = self._sanitize_fraction(GFRAC.values[:, 0])
        pasture_fraction[agri_land] = grass_frac[agri_land]
        pasture_fraction[extensive_grass] = 1.0

        # GFORMAN==4 outside current agricultural land retains the historical
        # grass-based prospective opportunity-cost treatment, but does not
        # influence calibration of the actual pasture scaler.
        counterfactual_pasture = (
            valid_region
            & (GFORMAN.values == 4)
            & ~(agri_land | extensive_grass)
        )

        grazing_intensity = np.asarray(
            GrazIntens.values[0, :], dtype=np.float64
        )
        if grazing_intensity.shape != (len(Region26.coords),):
            raise ValueError(
                f"Expected grazing intensity for 26 regions, got "
                f"{grazing_intensity.shape}"
            )
        if np.any(~np.isfinite(grazing_intensity)) or np.any(grazing_intensity < 0):
            raise ValueError("GRAZINTENS contains nonfinite/negative values")

        gi_cell = np.take(grazing_intensity, region_cells, mode="clip")
        gi_cell[region_cells < 0] = 0.0
        grass_yield = np.asarray(GRPPC.values[:, 0], dtype=np.float64)
        pasture_productivity = grass_yield * gi_cell
        pasture_productivity = np.where(
            np.isfinite(pasture_productivity) & (pasture_productivity >= 0.0),
            pasture_productivity,
            0.0,
        )

        # Fractions are land shares. A substantial sum > 1 indicates a
        # classification mismatch and should be visible rather than silently
        # renormalized.
        priced_fraction = crop_area_fraction + pasture_fraction
        bad_fraction = agri_land & np.isfinite(priced_fraction) & (priced_fraction > 1.01)
        if np.any(bad_fraction):
            print(
                "WARNING: crop + pasture fractions exceed 1.01 in "
                f"{int(bad_fraction.sum())} GLCT==1 cells; maximum="
                f"{float(np.nanmax(priced_fraction[bad_fraction])):.4f}. "
                "These fractions are left unchanged for auditability."
            )

        crop_scaler = self._normalize_sector_proxy(
            crop_value_per_ha,
            crop_area_fraction,
            GAREA5m.values,
            valid_region,
            label="cropland",
        )
        pasture_scaler = self._normalize_sector_proxy(
            pasture_productivity,
            pasture_fraction,
            GAREA5m.values,
            valid_region,
            label="pasture",
            extra_mask=counterfactual_pasture,
        )

        crop_proxy_out = np.zeros(ngrid, dtype=np.float32)
        crop_proxy_out[crop_positive] = crop_value_per_ha[crop_positive]

        pasture_proxy_out = np.zeros(ngrid, dtype=np.float32)
        pasture_proxy_mask = (
            (pasture_fraction > 0.0) | counterfactual_pasture
        ) & np.isfinite(pasture_productivity)
        pasture_proxy_out[pasture_proxy_mask] = (
            pasture_productivity[pasture_proxy_mask].astype(np.float32)
        )

        self.crop_value_proxy[t] = crop_proxy_out
        self.crop_rent_scaler[t] = crop_scaler
        self.pasture_productivity_proxy[t] = pasture_proxy_out
        self.pasture_rent_scaler[t] = pasture_scaler

        return crop_area_fraction, pasture_fraction, counterfactual_pasture

    def _current_reduction_percent(
        self,
        time: prism.Time,
        GLCT: prism.Array[LandGrid, np.int8],
        GAREA5m: prism.Array[LandGrid, np.float32],
    ) -> np.ndarray:
        """Return the regional agricultural land-supply reduction used by both LUTs."""
        nregions = len(Region26.coords)
        reduction_percent = np.full(
            nregions, self.land_reduction_percent, dtype=np.float64
        )

        if self._affor_area_by_year is None:
            return reduction_percent

        if time.t == time.start:
            prev_area = np.zeros(nregions, dtype=np.float64)
        else:
            prev_year = int((time.t - time.dt).magnitude)
            if prev_year not in self._affor_area_by_year:
                raise KeyError(
                    f"AfforArea does not contain previous model year {prev_year}"
                )
            prev_area = self._affor_area_by_year[prev_year]

        valid = (
            (self.region_cells.values >= 0)
            & (self.region_cells.values < nregions)
        )
        # Preserve the existing IMAGE/MAGNET coupling definition of
        # agricultural area: both agricultural land and extensive grassland.
        agricultural = (
            valid
            & (GLCT.values <= 2)
            & np.isfinite(GAREA5m.values)
            & (GAREA5m.values > 0)
        )
        agricultural_area = np.bincount(
            self.region_cells.values[agricultural],
            weights=GAREA5m.values[agricultural],
            minlength=nregions,
        )
        if np.any((agricultural_area <= 0) & (prev_area > 0)):
            raise ValueError(
                "Cannot calculate A/R percentage: positive AfforArea in a "
                "region without agricultural area"
            )
        return np.divide(
            100.0 * prev_area,
            agricultural_area,
            out=np.zeros(nregions, dtype=np.float64),
            where=agricultural_area > 0,
        )

    def _calculate_magnet_landrent(
        self,
        time: prism.Time,
        GLCT: prism.Array[LandGrid, np.int8],
        GFORMAN: prism.Array[LandGrid, np.int8],
        GAREA5m: prism.Array[LandGrid, np.float32],
        crop_fraction: np.ndarray,
        pasture_fraction: np.ndarray,
        counterfactual_pasture: np.ndarray,
    ) -> None:
        """Apply sectoral MAGNET rents and combine them by IMAGE land fraction."""
        idx_year = time_step(time.t)
        if not 0 <= idx_year < self._lut_crop.shape[0]:
            raise IndexError(
                f"LUT has {self._lut_crop.shape[0]} time steps; "
                f"year {time.t.magnitude} corresponds to index {idx_year}"
            )

        reduction_percent = self._current_reduction_percent(
            time=time, GLCT=GLCT, GAREA5m=GAREA5m
        )

        crop_rent, crop_bins = downscale_magnet_landrent(
            scaler=self.crop_rent_scaler[time.t],
            regions=self.region_cells.values,
            forest_management=GFORMAN.values,
            regional_lut=self._lut_crop[idx_year, :, :],
            reduction_lut=self._lut_reduction[idx_year, :],
            land_reduction_percent=reduction_percent,
        )
        pasture_rent, pasture_bins = downscale_magnet_landrent(
            scaler=self.pasture_rent_scaler[time.t],
            regions=self.region_cells.values,
            forest_management=GFORMAN.values,
            regional_lut=self._lut_pasture[idx_year, :, :],
            reduction_lut=self._lut_reduction[idx_year, :],
            land_reduction_percent=reduction_percent,
        )

        if not np.array_equal(crop_bins, pasture_bins):
            raise RuntimeError("Crop and pasture LUTs selected different reduction bins")

        # Sector files contain per-sector-hectare rents. LandRent is per total
        # grid-cell hectare, hence the explicit land fractions.
        combined = (
            np.asarray(crop_fraction, dtype=np.float32) * crop_rent
            + np.asarray(pasture_fraction, dtype=np.float32) * pasture_rent
        ).astype(np.float32)

        # Historical additional-deforestation treatment: a GFORMAN==4 cell
        # outside current agriculture is assigned a full-cell prospective
        # pasture opportunity cost using the baseline MAGNET column.
        combined[counterfactual_pasture] = pasture_rent[counterfactual_pasture]

        self.CropLandRent[time.t] = crop_rent
        self.PastureLandRent[time.t] = pasture_rent
        self.LandRent[time.t] = combined

        print(
            f"MAGNET sectoral rents {time.t.magnitude}: reduction "
            f"{reduction_percent.min():.3f}-{reduction_percent.max():.3f}% ; "
            f"selected LUT bins {crop_bins.min()}-{crop_bins.max()}"
        )

    def _write_results_to_files(self, time: prism.Time) -> None:
        t, dt, t_start, t_end = time.t, time.dt, time.start, time.end
        if t != t_end:
            return

        times = np.arange(
            t_start.magnitude,
            t_end.magnitude + dt.magnitude,
            dt.magnitude,
            dtype=int,
        )

        outputs = {
            "crop_value_proxy": (
                self.crop_value_proxy,
                self.output_dir / "crop_value_proxy.nc",
                "Crop production-value proxy per hectare of crop land; "
                "used only to construct the within-region cropland scaler.",
            ),
            "crop_rent_scaler": (
                self.crop_rent_scaler,
                self.output_dir / "crop_rent_scaler.nc",
                "Dimensionless cropland rent scaler; crop-hectare-weighted "
                "regional mean equals one.",
            ),
            "pasture_productivity_proxy": (
                self.pasture_productivity_proxy,
                self.output_dir / "pasture_productivity_proxy.nc",
                "Grass productivity proxy GRPPC(grass)*regional grazing intensity "
                "per hectare of pasture; no grass commodity price is used.",
            ),
            "pasture_rent_scaler": (
                self.pasture_rent_scaler,
                self.output_dir / "pasture_rent_scaler.nc",
                "Dimensionless ruminant/pasture rent scaler; pasture-hectare-"
                "weighted regional mean equals one.",
            ),
            "CropLandRent": (
                self.CropLandRent,
                self.output_dir / "CropLandRent.nc",
                "MAGNET ARABLE&PERMCR land rent downscaled within IMAGE regions; "
                "value is per hectare of crop land where the crop scaler is defined.",
            ),
            "PastureLandRent": (
                self.PastureLandRent,
                self.output_dir / "PastureLandRent.nc",
                "MAGNET livestock_rum land rent downscaled within IMAGE regions; "
                "value is per hectare of pasture. GFORMAN==4 counterfactual cells "
                "use the baseline MAGNET LUT column.",
            ),
            "LandRent": (
                self.LandRent,
                self.output_dir / "LandRent.nc",
                "Whole-cell agricultural opportunity cost: crop fraction times "
                "CropLandRent plus pasture fraction times PastureLandRent; "
                "counterfactual GFORMAN==4 cells use full-cell pasture rent.",
            ),
        }

        for _, path, _ in outputs.values():
            if path.exists():
                path.unlink()

        for name, (var, path, description) in outputs.items():
            export_variable_to_netcdf(
                prism_var=var,
                filename=path,
                var_name=name,
                dims=("grid",),
                coords={"grid": np.array(LandGrid.coords)},
                times=times,
                units="-",
                description=description,
            )

        print("Wrote sectoral land-rent outputs:")
        for _, path, _ in outputs.values():
            print(f"  {path}")


# Backward-compatible class alias. Note that the economic meaning is now
# sector-separated; downstream A/R allocation code must use LandRent directly
# or be updated to consume CropLandRent/PastureLandRent rather than the old
# single glandrent_scaler.
AfforLandRentSA = LandRentSA


# =============================================================================
# Post-simulation gridded map and regional-diagnostic reporting
# =============================================================================

DEFAULT_LANDGRID_COORDINATES = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\sa\biomass\IMAGE_grid_5min.nc"
)


def read_landgrid_coordinates(
    grid_coordinates_file: Path | str,
    n_land_cells: int,
    land_mask_5m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return lon/lat ordered EXACTLY like IMAGE's one-dimensional LandGrid.

    The IMAGE coordinate file may contain:
    1. Exactly the LandGrid cell coordinates;
    2. Global Grid coordinates requiring the 5-minute land mask; or
    3. LandGrid coordinates followed by 17,221 padding coordinates (the
       layout used in the original plotting snippet supplied by the user).
    Any other layout fails explicitly rather than silently misplacing cells.
    """
    path = Path(grid_coordinates_file)
    if not path.is_file():
        raise FileNotFoundError(
            f"IMAGE 5-minute coordinate file not found: {path}. "
            "Pass --grid-coordinates PATH to the correct file."
        )
    with xr.open_dataset(path) as ds:
        if "lat" not in ds or "lon" not in ds:
            raise KeyError(f"{path} must contain 'lat' and 'lon'")
        lat = np.asarray(ds["lat"].values, dtype=np.float64).reshape(-1)
        lon = np.asarray(ds["lon"].values, dtype=np.float64).reshape(-1)

    if lat.size != lon.size:
        raise ValueError("Latitude and longitude arrays have different lengths")
    if land_mask_5m is not None:
        land_mask_5m = np.asarray(land_mask_5m, dtype=bool).reshape(-1)

    if lat.size == n_land_cells:
        pass  # Already 1-D LandGrid coordinates.
    elif land_mask_5m is not None and lat.size == land_mask_5m.size:
        lat, lon = lat[land_mask_5m], lon[land_mask_5m]
    elif lat.size == n_land_cells + 17221:
        # Match the user's existing IMAGE_grid_5min.nc plotting convention.
        print("Coordinate file: dropping 17,221 trailing non-LandGrid records.")
        lat, lon = lat[:n_land_cells], lon[:n_land_cells]
    else:
        raise ValueError(
            f"Cannot align coordinate-file length {lat.size} with "
            f"LandGrid length {n_land_cells} "
            f"(full Grid length {None if land_mask_5m is None else land_mask_5m.size}). "
            "Check coordinate-file ordering; do not truncate arbitrarily."
        )

    if lat.size != n_land_cells or not np.all(np.isfinite(lat)) or not np.all(np.isfinite(lon)):
        raise ValueError("LandGrid coordinates have wrong size or non-finite values")
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon, lat


def read_rent_year(nc_path: Path | str, variable: str, year: int) -> np.ndarray:
    """Load exactly one saved simulation year from a gridded NetCDF output."""
    nc_path = Path(nc_path)
    with xr.open_dataset(nc_path) as ds:
        if variable not in ds:
            raise KeyError(f"Variable {variable!r} not in {nc_path}")
        da = ds[variable]
        if "time" not in da.dims:
            raise ValueError(f"{variable} in {nc_path} has no 'time' dimension")
        raw_time = np.asarray(da["time"].values)
        if np.issubdtype(raw_time.dtype, np.datetime64):
            years = np.asarray(da["time"].dt.year.values, dtype=int)
        elif np.issubdtype(raw_time.dtype, np.number):
            years = np.asarray(raw_time, dtype=int)
        else:
            raise ValueError(f"Unrecognized time coordinate type: {raw_time.dtype}")
        matches = np.flatnonzero(years == int(year))
        if matches.size != 1:
            raise ValueError(
                f"Expected exactly one {year} record in {nc_path.name}; "
                f"available years: {years.tolist()}"
            )
        return np.asarray(da.isel(time=int(matches[0])).values, dtype=np.float32).reshape(-1)


def extract_landgrid_area(
    garea_values: np.ndarray,
    n_land_cells: int,
    mask_land_5m: np.ndarray,
) -> np.ndarray:
    """Reproduce the GAREA flatten-and-mask convention in build_input_fun."""
    raw = np.asarray(garea_values, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask_land_5m, dtype=bool).reshape(-1)
    if raw.size == mask.size:
        area = raw[mask]
    elif raw.size == n_land_cells:
        area = raw
    else:
        raise ValueError(
            f"GAREA has {raw.size} entries; expected {mask.size} full-grid "
            f"entries or {n_land_cells} land-grid entries"
        )
    if area.size != n_land_cells:
        raise ValueError("GAREA and land-rent grids cannot be aligned")
    return area


def compute_region_rent_diagnostics(
    values: np.ndarray,
    regions: np.ndarray,
    areas: np.ndarray,
    *,
    year: int,
    variable: str,
) -> pd.DataFrame:
    """Compute gridded statistics, including zero rents, for IMAGE26 regions.

    Regular mean/min/max and median describe all valid LandGrid cells;
    weighted_mean uses GAREA in the same year as the land-rent output.
    positive_area_pct flags regions with many structurally zero rent cells.
    """
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    regions = np.asarray(regions, dtype=np.int64).reshape(-1)
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)
    if values.shape != regions.shape or areas.shape != values.shape:
        raise ValueError("Rent, region and GAREA arrays must share LandGrid shape")

    rows = []
    base_valid = np.isfinite(values) & np.isfinite(areas) & (areas > 0)
    for r, label in enumerate(IMAGE26_PRICE_REGION_CODES):
        idx = base_valid & (regions == r)
        v, a = values[idx], areas[idx]
        positive = v > 0
        rows.append({
            "variable": variable,
            "year": int(year),
            "image_region_number": r + 1,
            "image_region": label,
            "n_valid_cells": int(v.size),
            "area_km2": float(a.sum()) if a.size else 0.0,
            "mean": float(v.mean()) if v.size else np.nan,
            "area_weighted_mean": float(np.average(v, weights=a)) if v.size else np.nan,
            "min": float(v.min()) if v.size else np.nan,
            "max": float(v.max()) if v.size else np.nan,
            "median": float(np.median(v)) if v.size else np.nan,
            "p05": float(np.percentile(v, 5)) if v.size else np.nan,
            "p95": float(np.percentile(v, 95)) if v.size else np.nan,
            "positive_area_pct": (
                float(100 * a[positive].sum() / a.sum()) if a.size else np.nan
            ),
            "positive_area_weighted_mean": (
                float(np.average(v[positive], weights=a[positive]))
                if np.any(positive) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def choose_shared_map_norm(
    arrays: dict[int, np.ndarray],
    *,
    robust: bool = True,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[float, float]:
    """Return a common color range for all three years of one variable."""
    # Sampling limits the temporary memory used for large global grids.
    samples = []
    for year in sorted(arrays):
        arr = np.asarray(arrays[year]).reshape(-1)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            step = max(1, int(np.ceil(finite.size / 1_000_000)))
            samples.append(finite[::step])
    if not samples:
        raise ValueError("No finite values available for gridded maps")
    combined = np.concatenate(samples)
    if robust:
        lo, hi = np.percentile(combined, percentiles)
    else:
        lo, hi = float(combined.min()), float(combined.max())
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError("Invalid color limits")
    if hi <= lo:
        hi = lo + max(abs(lo) * 0.01, 1.0)
    return float(lo), float(hi)


def plot_gridded_landrent_map(
    values: np.ndarray,
    lons: np.ndarray,
    lats: np.ndarray,
    *,
    variable: str,
    year: int,
    output_path: Path | str,
    vmin: float,
    vmax: float,
    point_size: float = 0.8,
    dpi: int = 200,
    show: bool = False,
) -> Path:
    """Save a Robinson-projection scatter map, using Cartopy if available."""
    import matplotlib as mpl
    if not show:
        # Non-interactive figures avoid blocking a long automated simulation.
        mpl.use("Agg")
    import matplotlib.pyplot as plt

    values = np.asarray(values, dtype=np.float32).reshape(-1)
    lon = np.asarray(lons).reshape(-1)
    lat = np.asarray(lats).reshape(-1)
    if values.size != lon.size or values.size != lat.size:
        raise ValueError(f"{variable} map values and coordinates are misaligned")
    valid = np.isfinite(values) & np.isfinite(lon) & np.isfinite(lat)
    if not np.any(valid):
        raise ValueError(f"No valid {variable} map values for {year}")

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        ccrs, cfeature = None, None
        print("WARNING: Cartopy not installed; saving simple lon/lat map without boundaries.")

    fig = plt.figure(figsize=(10.5, 5.6))
    if ccrs is not None:
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.set_global()
        ax.add_feature(cfeature.LAND, facecolor="0.92", zorder=0)
        ax.add_feature(cfeature.OCEAN, facecolor="white", zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.4, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, alpha=0.5, zorder=2)
        transformation = {"transform": ccrs.PlateCarree()}
    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        transformation = {}

    cmap = "Greens" if variable == "LandRent" else "viridis"
    sc = ax.scatter(
        lon[valid], lat[valid], c=values[valid],
        s=point_size, cmap=cmap,
        norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax),
        linewidths=0, rasterized=True, zorder=1,
        **transformation,
    )
    cb = fig.colorbar(
        sc, ax=ax, orientation="horizontal", fraction=0.045,
        pad=0.06, aspect=35,
    )
    cb.set_label(f"{variable}")
    ax.set_title(f"IMAGE {variable} ({year})", fontsize=16, pad=10)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return output_path


def compute_sector_weighted_diagnostics(
    crop_rent: np.ndarray,
    pasture_rent: np.ndarray,
    combined_rent: np.ndarray,
    crop_fraction: np.ndarray,
    pasture_fraction: np.ndarray,
    regions: np.ndarray,
    areas: np.ndarray,
    *,
    year: int,
) -> pd.DataFrame:
    """Report sector-hectare-weighted means used to validate MAGNET calibration."""
    crop_rent = np.asarray(crop_rent, dtype=np.float64).reshape(-1)
    pasture_rent = np.asarray(pasture_rent, dtype=np.float64).reshape(-1)
    combined_rent = np.asarray(combined_rent, dtype=np.float64).reshape(-1)
    crop_fraction = np.asarray(crop_fraction, dtype=np.float64).reshape(-1)
    pasture_fraction = np.asarray(pasture_fraction, dtype=np.float64).reshape(-1)
    regions = np.asarray(regions, dtype=np.int64).reshape(-1)
    areas = np.asarray(areas, dtype=np.float64).reshape(-1)

    rows = []
    for r, label in enumerate(IMAGE26_PRICE_REGION_CODES):
        base = (
            (regions == r)
            & np.isfinite(areas)
            & (areas > 0)
        )

        cw = np.where(
            base & np.isfinite(crop_fraction) & (crop_fraction > 0),
            areas * crop_fraction,
            0.0,
        )
        pw = np.where(
            base & np.isfinite(pasture_fraction) & (pasture_fraction > 0),
            areas * pasture_fraction,
            0.0,
        )
        aw = np.where(base, areas, 0.0)

        crop_valid = (cw > 0) & np.isfinite(crop_rent)
        pasture_valid = (pw > 0) & np.isfinite(pasture_rent)
        combined_valid = (aw > 0) & np.isfinite(combined_rent)

        rows.append({
            "year": int(year),
            "image_region_number": r + 1,
            "image_region": label,
            "crop_area_equivalent": float(cw.sum()),
            "pasture_area_equivalent": float(pw.sum()),
            "crop_hectare_weighted_mean_rent": (
                float(np.average(crop_rent[crop_valid], weights=cw[crop_valid]))
                if np.any(crop_valid) else np.nan
            ),
            "pasture_hectare_weighted_mean_rent": (
                float(np.average(
                    pasture_rent[pasture_valid], weights=pw[pasture_valid]
                ))
                if np.any(pasture_valid) else np.nan
            ),
            "whole_cell_area_weighted_mean_opportunity_cost": (
                float(np.average(
                    combined_rent[combined_valid], weights=aw[combined_valid]
                ))
                if np.any(combined_valid) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def report_landrent_outputs(
    output_dir: Path | str,
    scen_dir: Path | str,
    *,
    years_to_plot: tuple[int, ...] = (2030, 2050, 2100),
    years_to_diagnose: tuple[int, ...] = (2050, 2100),
    grid_coordinates_file: Path | str = DEFAULT_LANDGRID_COORDINATES,
    map_point_size: float = 0.8,
    map_dpi: int = 200,
    make_maps: bool = True,
    show_maps: bool = False,
) -> pd.DataFrame:
    """Generate maps and diagnostics for crop, pasture and combined opportunity cost."""
    output_dir, scen_dir = Path(output_dir), Path(scen_dir)
    region_cells, _ = region_to_grid("5m")
    regions = np.asarray(region_cells, dtype=np.int16).reshape(-1)
    land_mask_5m = np.asarray(land_mask("5m"), dtype=bool).reshape(-1)
    n_land = regions.size

    print("\n" + "=" * 80)
    print("Post-simulation sectoral land-rent maps and regional diagnostics")
    print("=" * 80)

    variables = ("CropLandRent", "PastureLandRent", "LandRent")
    requested_years = sorted(
        set(years_to_plot if make_maps else ()) | set(years_to_diagnose)
    )
    grids: dict[str, dict[int, np.ndarray]] = {}
    for variable in variables:
        path = output_dir / f"{variable}.nc"
        if not path.is_file():
            raise FileNotFoundError(f"Postprocessing expects completed file {path}")
        grids[variable] = {}
        for year in requested_years:
            v = read_rent_year(path, variable, year)
            if v.size != n_land:
                raise ValueError(
                    f"{path.name}, year={year}: {v.size} grid cells, "
                    f"expected {n_land} from region_to_grid('5m')"
                )
            grids[variable][year] = v

    # Read land area plus IMAGE crop/pasture fractions for the diagnostic years.
    area_by_year: dict[int, np.ndarray] = {}
    crop_fraction_by_year: dict[int, np.ndarray] = {}
    pasture_fraction_by_year: dict[int, np.ndarray] = {}

    with (
        xr.open_dataset(scen_dir / "GAREA.NC") as ds_garea,
        xr.open_dataset(scen_dir / "GLCT.NC") as ds_glct,
        xr.open_dataset(scen_dir / "GFRAC.NC") as ds_gfrac,
    ):
        for year in years_to_diagnose:
            index = time_step(prism.Q_(int(year), "year"))

            garea = ds_garea["GAREA"].isel(time=index).values
            area_by_year[year] = extract_landgrid_area(
                garea, n_land, land_mask_5m
            )

            glct_raw = np.asarray(
                ds_glct["GLCT"].isel(time=index).values, dtype=np.float32
            ).reshape(-1)
            if glct_raw.size == land_mask_5m.size:
                glct = glct_raw[land_mask_5m]
            elif glct_raw.size == n_land:
                glct = glct_raw
            else:
                raise ValueError("Cannot align GLCT with LandGrid in diagnostics")

            gfrac_raw = np.asarray(
                ds_gfrac["GFRAC"].isel(time=index).values, dtype=np.float32
            ).reshape(len(Grid.coords), len(Crop38.coords))
            gfrac = gfrac_raw[land_mask_5m, :]

            crop_ids = list(range(1, 17)) + list(range(22, 38))
            crop_fraction = np.nansum(
                np.where(np.isfinite(gfrac[:, crop_ids]) & (gfrac[:, crop_ids] > 0),
                         gfrac[:, crop_ids], 0.0),
                axis=1,
            ).astype(np.float32)
            crop_fraction[glct != 1] = 0.0

            pasture_fraction = np.zeros(n_land, dtype=np.float32)
            grass = np.where(
                np.isfinite(gfrac[:, 0]) & (gfrac[:, 0] > 0),
                gfrac[:, 0],
                0.0,
            )
            pasture_fraction[glct == 1] = grass[glct == 1]
            pasture_fraction[glct == 2] = 1.0

            crop_fraction_by_year[year] = crop_fraction
            pasture_fraction_by_year[year] = pasture_fraction

    diagnostic_frames = []
    for variable in variables:
        for year in years_to_diagnose:
            frame = compute_region_rent_diagnostics(
                grids[variable][year],
                regions,
                area_by_year[year],
                year=year,
                variable=variable,
            )
            diagnostic_frames.append(frame)
            print(f"\n{variable} | {year} | all valid cells including zeros")
            print(
                frame[[
                    "image_region", "n_valid_cells", "mean",
                    "area_weighted_mean", "min", "max", "positive_area_pct",
                ]].to_string(index=False, float_format=lambda v: f"{v:,.3f}")
            )

    combined = pd.concat(diagnostic_frames, ignore_index=True)
    csv_path = output_dir / "sector_landrent_regional_diagnostics_2050_2100.csv"
    combined.to_csv(csv_path, index=False)
    print(f"\nSaved general regional diagnostics: {csv_path}")

    sector_frames = []
    for year in years_to_diagnose:
        sec = compute_sector_weighted_diagnostics(
            grids["CropLandRent"][year],
            grids["PastureLandRent"][year],
            grids["LandRent"][year],
            crop_fraction_by_year[year],
            pasture_fraction_by_year[year],
            regions,
            area_by_year[year],
            year=year,
        )
        sector_frames.append(sec)
        print(f"\nSector-hectare-weighted calibration diagnostics | {year}")
        print(
            sec[[
                "image_region",
                "crop_hectare_weighted_mean_rent",
                "pasture_hectare_weighted_mean_rent",
                "whole_cell_area_weighted_mean_opportunity_cost",
            ]].to_string(index=False, float_format=lambda v: f"{v:,.3f}")
        )

    sector_diag = pd.concat(sector_frames, ignore_index=True)
    sector_csv = output_dir / "sector_landrent_calibration_diagnostics_2050_2100.csv"
    sector_diag.to_csv(sector_csv, index=False)
    print(f"\nSaved sector calibration diagnostics: {sector_csv}")

    if make_maps:
        lons, lats = read_landgrid_coordinates(
            grid_coordinates_file, n_land, land_mask_5m
        )
        plot_dir = output_dir / "landrent_maps"
        for variable in variables:
            maps = {year: grids[variable][year] for year in years_to_plot}
            vmin, vmax = choose_shared_map_norm(maps, robust=True)
            print(
                f"\n{variable}: same 2nd-98th percentile color scale across "
                f"all map years: {vmin:,.3f} to {vmax:,.3f}"
            )
            for year in years_to_plot:
                path = plot_gridded_landrent_map(
                    maps[year],
                    lons,
                    lats,
                    variable=variable,
                    year=year,
                    output_path=plot_dir / f"{variable}_{year}.png",
                    vmin=vmin,
                    vmax=vmax,
                    point_size=map_point_size,
                    dpi=map_dpi,
                    show=show_maps,
                )
                print(f"Saved map: {path}")

    return combined


# =============================================================================
# Standalone input loader / runner
# =============================================================================

def build_input_fun(
    scen_dir: Path,
    mask_land_5m: np.ndarray,
):
    """Open the time-varying IMAGE inputs needed by the sectoral rent model."""

    ds_glct = xr.open_dataset(
        scen_dir / "GLCT.NC",
        engine="netcdf4",
        chunks={"time": 1},
    )
    ds_gforman = xr.open_dataset(
        scen_dir / "GFORMAN.NC",
        engine="netcdf4",
        chunks={"time": 1},
    )
    ds_gfrac = xr.open_dataset(
        scen_dir / "GFRAC.NC",
        engine="netcdf4",
        chunks={"time": 1},
    )
    ds_grppc = xr.open_dataset(
        scen_dir / "GRPPC.NC",
        engine="netcdf4",
        chunks={"time": 1},
    )
    ds_garea = xr.open_dataset(
        scen_dir / "GAREA.NC",
        engine="netcdf4",
        chunks={"time": 1},
    )

    glct_da = ds_glct["GLCT"]
    gforman_da = ds_gforman["GFORMAN"]
    gfrac_da = ds_gfrac["GFRAC"]
    grppc_da = ds_grppc["GRPPC"]
    garea_da = ds_garea["GAREA"]

    var_timeline = prism.Timeline(
        start=prism.Q_(1970, "year"),
        end=prism.Q_(2100, "year"),
        stepsize=prism.Q_(1, "year"),
    )
    grazintens_tv = prism.TimeVariable[GrassLand, Region26, np.float32](
        file=scen_dir / "GRAZINTENS.out",
        timeline=var_timeline,
    )

    def input_fun(t):
        ts = time_step(t)

        sub = gfrac_da.isel(time=ts)
        sub_np = np.asarray(sub, dtype=np.float32)
        gfrac_flat = sub_np.reshape(len(Grid.coords), len(Crop38.coords))
        gfrac_flat = gfrac_flat[mask_land_5m, :]
        GFRAC = prism.array(gfrac_flat, dims=[LandGrid, Crop38])
        del sub, sub_np, gfrac_flat

        sub = grppc_da.isel(time=ts)
        sub_np = np.asarray(sub, dtype=np.float32)
        grppc_flat = sub_np.reshape(len(Grid.coords), len(Crop38.coords))
        grppc_flat = grppc_flat[mask_land_5m, :]
        GRPPC = prism.array(grppc_flat, dims=[LandGrid, Crop38])
        del sub, sub_np, grppc_flat

        sub = glct_da.isel(time=ts)
        sub_np = np.asarray(sub, dtype=np.float32)
        glct_flat = sub_np.reshape(len(Grid.coords))
        glct_flat = glct_flat[mask_land_5m]
        glct_flat = np.where(
            np.isfinite(glct_flat), glct_flat, 99
        ).astype(np.int8)
        GLCT = prism.array(glct_flat, dims=[LandGrid])
        del sub, sub_np, glct_flat

        sub = gforman_da.isel(time=ts, NFORMID=0)
        sub_np = np.asarray(sub, dtype=np.float32)
        gforman_flat = sub_np.reshape(len(Grid.coords))
        gforman_flat = gforman_flat[mask_land_5m]
        gforman_flat = np.where(
            np.isfinite(gforman_flat), gforman_flat, -99
        ).astype(np.int8)
        GFORMAN = prism.array(gforman_flat, dims=[LandGrid])
        del sub, sub_np, gforman_flat

        sub = garea_da.isel(time=ts)
        sub_np = np.asarray(sub, dtype=np.float32)
        garea_flat = sub_np.reshape(len(Grid.coords))
        garea_flat = garea_flat[mask_land_5m]
        GAREA5m = prism.array(garea_flat, dims=[LandGrid])
        del sub, sub_np, garea_flat

        gi_values = np.asarray(grazintens_tv[t].values, dtype=np.float32)
        if gi_values.shape[1] < len(Region26.coords):
            raise ValueError(
                f"GRAZINTENS has {gi_values.shape[1]} regions; expected at least 26"
            )
        GrazIntens = prism.array(
            gi_values[:, :len(Region26.coords)],
            dims=[GrassLand, Region26],
        )

        return {
            "GFRAC": GFRAC,
            "GRPPC": GRPPC,
            "GLCT": GLCT,
            "GFORMAN": GFORMAN,
            "GAREA5m": GAREA5m,
            "GrazIntens": GrazIntens,
        }

    datasets = [ds_glct, ds_gforman, ds_gfrac, ds_grppc, ds_garea]
    return input_fun, datasets


def run_landrent_scenario(
    scen_dir: Path,
    output_dir: Path,
    simulation_timeline: prism.Timeline,
    lut_file: Path,
    land_reduction_percent: float = 0.0,
    affor_area_nc: Path | None = None,
    crop_price_file: Path = DEFAULT_CROP_PRICE_FILE,
) -> None:
    print("\n" + "=" * 80)
    print(f"Running standalone IMAGE sectoral land-rent model: {scen_dir.name}")
    print("=" * 80)

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_land_5m = land_mask("5m")
    input_fun, datasets = build_input_fun(
        scen_dir=scen_dir,
        mask_land_5m=mask_land_5m,
    )

    landrent = LandRentSA(simulation_timeline)
    landrent.configure_paths(
        scen_dir=scen_dir,
        output_dir=output_dir,
    )
    landrent.configure_crop_prices(
        crop_price_file=crop_price_file,
    )
    landrent.configure_land_reduction(
        lut_file=lut_file,
        land_reduction_percent=land_reduction_percent,
        affor_area_nc=affor_area_nc,
    )

    try:
        landrent.simulate(
            simulation_timeline,
            inputs=input_fun,
        )
    finally:
        for ds in datasets:
            ds.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the IMAGE standalone sectoral agricultural opportunity-cost "
            "calculation without afforestation allocation. Cropland uses MAGNET "
            "ARABLE&PERMCR rent; pasture uses MAGNET livestock_rum rent."
        )
    )

    parser.add_argument(
        "--scen",
        type=str,
        required=True,
        help="Name of the IMAGE land scenario folder.",
    )
    parser.add_argument(
        "--scenario-root",
        type=Path,
        default=DEFAULT_SCENARIO_ROOT,
        help=(
            "Root directory containing the IMAGE land scenario folder. "
            f"Default: {DEFAULT_SCENARIO_ROOT}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Default: "
            "<cdr_output>/landrent_sectoral_<scenario>."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Root used when --output-dir is omitted. "
            f"Default: {DEFAULT_OUTPUT_ROOT}"
        ),
    )
    parser.add_argument(
        "--crop-price-file",
        type=Path,
        default=DEFAULT_CROP_PRICE_FILE,
        help=(
            "RegionalWide16 workbook with 26 IMAGE regions and 16 crop prices "
            "(constant 2010 USD/tonne). Grass price is not used."
        ),
    )

    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--end-year", type=int, default=2100)
    parser.add_argument("--step-years", type=int, default=5)

    parser.add_argument(
        "--lut-file",
        type=Path,
        default=None,
        help=(
            "Sectoral MAGNET LUT .mat file. Default: "
            "<scenario>/sector_landrent_LUT_input_data.mat."
        ),
    )
    reductions = parser.add_mutually_exclusive_group()
    reductions.add_argument(
        "--land-reduction-percent",
        type=float,
        default=0.0,
        help=(
            "Fixed agricultural land-supply reduction percentage used for BOTH "
            "cropland and pasture LUTs (default: 0)."
        ),
    )
    reductions.add_argument(
        "--affor-area-nc",
        type=Path,
        default=None,
        help=(
            "Existing AfforArea NetCDF. Uses previous-step cumulative A/R area "
            "divided by current GLCT<=2 agricultural area for the shared MAGNET "
            "LUT reduction, matching AfforAllocSA."
        ),
    )

    parser.add_argument(
        "--rebuild-lut",
        action="store_true",
        help=(
            "Build separate ARABLE&PERMCR and livestock_rum LUTs from the five "
            "full M2I MAGNET scenario files."
        ),
    )
    parser.add_argument(
        "--magnet-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the M2I CSV/XLSX files when rebuilding the "
            "LUT. Default: the IMAGE scenario directory."
        ),
    )
    parser.add_argument(
        "--land-supply-file",
        type=str,
        default="land_supply_reductions.xlsx",
        help="Land-supply-reduction workbook under land_data_dir.",
    )
    parser.add_argument(
        "--magnet-scens",
        nargs="+",
        default=DEFAULT_MAGNET_SCENS,
        help=(
            "Five M2I scenario basenames used to rebuild the sectoral LUT. "
            "Each may be supplied as CSV or XLSX."
        ),
    )

    # Reporting options.
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help=(
            "Create maps/diagnostics from existing CropLandRent, "
            "PastureLandRent and LandRent NetCDF outputs without rerunning PRISM."
        ),
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Print/export regional diagnostics without producing PNG maps.",
    )
    parser.add_argument(
        "--show-maps",
        action="store_true",
        help="Also display each map interactively; normally PNGs are only saved.",
    )
    parser.add_argument(
        "--grid-coordinates",
        type=Path,
        default=DEFAULT_LANDGRID_COORDINATES,
        help="IMAGE 5-minute lat/lon coordinate NetCDF file.",
    )
    parser.add_argument(
        "--map-dpi",
        type=int,
        default=200,
        help="PNG output DPI (default: 200).",
    )
    parser.add_argument(
        "--map-point-size",
        type=float,
        default=0.8,
        help="Scatter marker area, in points squared (default: 0.8).",
    )
    return parser.parse_args()


def main() -> None:
    print("Running IMAGE standalone cropland + ruminant opportunity-cost model")
    args = parse_args()

    scen_dir = args.scenario_root / args.scen
    if not scen_dir.exists():
        raise FileNotFoundError(f"Land scenario folder not found: {scen_dir}")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.output_root / f"landrent_sectoral_{args.scen}"
    )

    start_wall = time.perf_counter()

    if not args.postprocess_only:
        simulation_timeline = prism.Timeline(
            start=prism.Q_(args.start_year, "year"),
            end=prism.Q_(args.end_year, "year"),
            stepsize=prism.Q_(args.step_years, "year"),
        )

        lut_path = (
            args.lut_file
            if args.lut_file is not None
            else scen_dir / "sector_landrent_LUT_input_data.mat"
        )

        if args.rebuild_lut:
            magnet_dir = args.magnet_dir if args.magnet_dir is not None else scen_dir
            print(f"Rebuilding sectoral MAGNET land-rent LUT from: {magnet_dir}")
            lut_path = build_landrent_lut(
                magnet_scens=args.magnet_scens,
                land_supply_file=args.land_supply_file,
                scen_dir=magnet_dir,
                output_file=lut_path,
            )
            print(f"Wrote sectoral LUT: {lut_path}")

        if not lut_path.is_file():
            raise FileNotFoundError(
                f"Missing sectoral MAGNET land-rent LUT: {lut_path}. "
                "An old affor_LUT_input_data.mat containing only AGRI_PRIM is "
                "not sufficient. Run with --rebuild-lut and point --magnet-dir "
                "to the five full M2I scenario files, or supply --lut-file."
            )

        run_landrent_scenario(
            scen_dir=scen_dir,
            output_dir=output_dir,
            simulation_timeline=simulation_timeline,
            lut_file=lut_path,
            land_reduction_percent=args.land_reduction_percent,
            affor_area_nc=args.affor_area_nc,
            crop_price_file=args.crop_price_file,
        )

    report_landrent_outputs(
        output_dir=output_dir,
        scen_dir=scen_dir,
        years_to_plot=(2030, 2050, 2100),
        years_to_diagnose=(2050, 2100),
        grid_coordinates_file=args.grid_coordinates,
        map_point_size=args.map_point_size,
        map_dpi=args.map_dpi,
        make_maps=not args.no_maps,
        show_maps=args.show_maps,
    )

    elapsed = time.perf_counter() - start_wall
    print(f"\nFinished sectoral land-rent model in {elapsed:.1f} s")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()

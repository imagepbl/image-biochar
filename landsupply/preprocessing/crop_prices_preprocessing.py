import argparse
import math
import re
import unicodedata
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_DATA_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\data\agrilandrent\crop_prices_preprocessing"
)
DEFAULT_MAPPING_FILENAME = "IMAGE_country_to_region.tsv"
DEFAULT_OUTPUT_FILENAME = "FAOSTAT_IMAGE26_crop_prices_2010USD_v5.xlsx"
DEFAULT_SUPPLEMENTAL_FILENAME = "IMAGE_additional_crop_prices_input.xlsx"

BASE_YEAR = 2010
CENTRAL_START = 2010
CENTRAL_END = 2019
SUPPORT_START = 2000
SUPPORT_END = 2009

MIN_PRICE_OBS = 3
PREFERRED_REGIONAL_MODEL_COVERAGE = 0.50
MIN_REGIONAL_MODEL_COVERAGE = 0.33
IMPUTATION_MIN_DONORS = 3
OUTLIER_MIN_COUNTRIES = 8
OUTLIER_ROBUST_Z = 5.0

EXCLUDE_FLAGGED_OUTLIERS = False

IMAGE26_REGIONS: List[Tuple[int, str]] = [
    (1, "CAN"),
    (2, "USA"),
    (3, "MEX"),
    (4, "CAM"),
    (5, "BRA"),
    (6, "RSM"),
    (7, "NAF"),
    (8, "WAF"),
    (9, "EAF"),
    (10, "ZAF"),
    (11, "WEU"),
    (12, "CEU"),
    (13, "TUR"),
    (14, "UKR"),
    (15, "CAS"),
    (16, "RUR"),
    (17, "ME"),
    (18, "IND"),
    (19, "KOR"),
    (20, "CHR"),
    (21, "SEA"),
    (22, "IDN"),
    (23, "JAP"),
    (24, "OCE"),
    (25, "RSA"),
    (26, "RSF"),
]
REGION_TO_NUMBER = {code: n for n, code in IMAGE26_REGIONS}
REGION_TO_INDEX0 = {code: n - 1 for n, code in IMAGE26_REGIONS}

# Non-ISO / aggregate FAOSTAT entities needing explicit treatment.
EXCLUDED_FAOSTAT_M49 = {
    159: "China aggregate; excluded to avoid double counting China mainland, Taiwan, Hong Kong and Macao."
}
REGION_OVERRIDE_BY_M49 = {
    736: "EAF",  # Sudan (former), safe at regional scale because both successors are EAF.
}

# Sudan (former) producer prices (M49=736) have a usable 2010 exchange rate in
# the FAOSTAT inputs but no GDP-deflator series.  Use current Sudan (M49=729)
# only as a deflator-ratio proxy.  The former-Sudan 2010 exchange rate is kept.
DEFLATOR_PROXY_BY_M49 = {
    736: 729,
}

IMAGE_GROUPS: Dict[int, Dict[str, str]] = {
    1: {"image_crop": "Wheat", "magnet_crop_group": "Wheat"},
    2: {"image_crop": "Rice", "magnet_crop_group": "Paddy rice"},
    3: {"image_crop": "Maize", "magnet_crop_group": "Cereal grains"},
    4: {"image_crop": "Tropical cereals", "magnet_crop_group": "Cereal grains"},
    5: {"image_crop": "Other temperate cereals", "magnet_crop_group": "Cereal grains"},
    6: {"image_crop": "Pulses", "magnet_crop_group": "Fruit, vegetables, nuts"},
    7: {"image_crop": "Soybeans", "magnet_crop_group": "Oil seeds"},
    8: {"image_crop": "Temperate oil crops", "magnet_crop_group": "Oil seeds"},
    9: {"image_crop": "Tropical oil crops", "magnet_crop_group": "Oil seeds"},
    10: {"image_crop": "Temperate roots & tubers", "magnet_crop_group": "Fruit, vegetables, nuts"},
    11: {"image_crop": "Tropical roots & tubers", "magnet_crop_group": "Fruit, vegetables, nuts"},
    12: {"image_crop": "Sugar Crops", "magnet_crop_group": "Sugarcrops"},
    13: {"image_crop": "Oil & palm fruit", "magnet_crop_group": "Oil seeds"},
    14: {"image_crop": "Vegetables & fruits", "magnet_crop_group": "Fruit, vegetables, nuts"},
    15: {"image_crop": "Other non-food & luxury & spices", "magnet_crop_group": "Crops nec"},
    16: {"image_crop": "Plant based fibres", "magnet_crop_group": "Plant based fibres"},
}

FAOSTAT_ITEMS_BY_IMAGE_GROUP: Dict[int, List[str]] = {
    1: [
        "Wheat",
    ],
    2: [
        "Rice",
    ],
    3: [
        "Maize (corn)",
    ],
    4: [
        "Fonio",
        "Millet",
        "Sorghum",
    ],
    5: [
        "Barley",
        "Buckwheat",
        "Canary seed",
        "Mixed grain",
        "Oats",
        "Quinoa",
        "Rye",
        "Triticale",
    ],
    6: [
        "Bambara beans, dry",
        "Beans, dry",
        "Broad beans and horse beans, dry",
        "Chick peas, dry",
        "Cow peas, dry",
        "Lentils, dry",
        "Lupins",
        "Other pulses n.e.c.",
        "Peas, dry",
        "Pigeon peas, dry",
        "Vetches",
    ],
    7: [
        "Soya beans",
    ],
    8: [
        "Hempseed",
        "Linseed",
        "Mustard seed",
        "Olives",
        "Poppy seed",
        "Rape or colza seed",
        "Safflower seed",
        "Sunflower seed",
    ],
    9: [
        "Castor oil seeds",
        "Coconuts, in shell",
        "Groundnuts, excluding shelled",
        "Jojoba seeds",
        "Karite nuts (sheanuts)",
        "Melonseed",
        "Sesame seed",
        "Tallowtree seeds",
        "Tung nuts",
    ],
    10: [
        "Potatoes",
    ],
    11: [
        "Cassava, fresh",
        "Edible roots and tubers with high starch or inulin content, n.e.c., fresh",
        "Sweet potatoes",
        "Taro",
        "Yams",
        "Yautia",
    ],
    12: [
        "Other sugar crops n.e.c.",
        "Sugar beet",
        "Sugar cane",
    ],
    13: [
        "Oil palm fruit",
    ],
    14: [
        "Almonds, in shell",
        "Apples",
        "Apricots",
        "Artichokes",
        "Asparagus",
        "Avocados",
        "Bananas",
        "Blueberries",
        "Brazil nuts, in shell",
        "Broad beans and horse beans, green",
        "Cabbages",
        "Cantaloupes and other melons",
        "Carrots and turnips",
        "Cashew nuts, in shell",
        "Cashewapple",
        "Cauliflowers and broccoli",
        "Cherries",
        "Chestnuts, in shell",
        "Chicory roots",
        "Chillies and peppers, green (Capsicum spp. and Pimenta spp.)",
        "Cranberries",
        "Cucumbers and gherkins",
        "Currants",
        "Dates",
        "Eggplants (aubergines)",
        "Figs",
        "Gooseberries",
        "Grapes",
        "Green corn (maize)",
        "Green garlic",
        "Hazelnuts, in shell",
        "Kiwi fruit",
        "Leeks and other alliaceous vegetables",
        "Lemons and limes",
        "Lettuce and chicory",
        "Locust beans (carobs)",
        "Mangoes, guavas and mangosteens",
        "Okra",
        "Onions and shallots, dry (excluding dehydrated)",
        "Onions and shallots, green",
        "Oranges",
        "Other beans, green",
        "Other berries and fruits of the genus vaccinium n.e.c.",
        "Other citrus fruit, n.e.c.",
        "Other fruits, n.e.c.",
        "Other nuts (excluding wild edible nuts and groundnuts), in shell, n.e.c.",
        "Other pome fruits",
        "Other stone fruits",
        "Other tropical fruits, n.e.c.",
        "Other vegetables, fresh n.e.c.",
        "Papayas",
        "Peaches and nectarines",
        "Pears",
        "Peas, green",
        "Persimmons",
        "Pineapples",
        "Pistachios, in shell",
        "Plantains and cooking bananas",
        "Plums and sloes",
        "Pomelos and grapefruits",
        "Pumpkins, squash and gourds",
        "Quinces",
        "Raspberries",
        "Sour cherries",
        "Spinach",
        "Strawberries",
        "String beans",
        "Tangerines, mandarins, clementines",
        "Tomatoes",
        "Walnuts, in shell",
        "Watermelons",
    ],
    15: [
        "Anise, badian, coriander, cumin, caraway, fennel and juniper berries, raw",
        "Areca nuts",
        "Chillies and peppers, dry (Capsicum spp., Pimenta spp.), raw",
        "Cinnamon and cinnamon-tree flowers, raw",
        "Cloves (whole stems), raw",
        "Cocoa beans",
        "Coffee, green",
        "Ginger, raw",
        "Hop cones",
        "Kola nuts",
        "Maté leaves",
        "Natural rubber in primary forms",
        "Nutmeg, mace, cardamoms, raw",
        "Other stimulant, spice and aromatic crops, n.e.c.",
        "Pepper (Piper spp.), raw",
        "Peppermint, spearmint",
        "Pyrethrum, dried flowers",
        "Tea leaves",
        "Unmanufactured tobacco",
        "Vanilla, raw",
    ],
    16: [
        "Abaca, manila hemp, raw",
        "Agave fibres, raw, n.e.c.",
        "Flax, raw or retted",
        "Jute, raw or retted",
        "Kapok fruit",
        "Kenaf, and other textile bast fibres, raw or retted",
        "Other fibre crops, raw, n.e.c.",
        "Ramie, raw or retted",
        "Seed cotton, unginned",
        "Sisal, raw",
        "True hemp, raw or retted",
    ],
}

# These production items are intentionally not used in the central mapping.
# They are retained in the output crosswalk so that exclusions are auditable.
AMBIGUOUS_EXCLUDED: Dict[str, str] = {
    "Cereals n.e.c.": (
        "Residual cereal category cannot be reliably split between IMAGE "
        "tropical and other temperate cereals."
    ),
    "Other oil seeds, n.e.c.": (
        "Residual oilseed category cannot be reliably split between IMAGE "
        "temperate and tropical oil crops."
    ),
}

EXCLUDED_ITEMS: Dict[str, str] = {
    "Cassava leaves": (
        "Secondary/by-product commodity; excluded to avoid double counting "
        "land already represented by cassava roots."
    ),
    "Coir, raw": (
        "Coconut by-product; excluded to avoid double counting land already "
        "represented by coconuts."
    ),
    "Mushrooms and truffles": (
        "Excluded from the cropland opportunity-cost basket because production "
        "is not reliably tied to open agricultural land area."
    ),
}

# A few included assignments deserve an explicit note because they are broader
# classification choices rather than one-to-one commodity matches.
SPECIAL_MAPPING_NOTES: Dict[str, str] = {
    "Edible roots and tubers with high starch or inulin content, n.e.c., fresh": (
        "Assigned to IMAGE tropical roots & tubers as a pragmatic residual "
        "starchy-root category; retain as a sensitivity/check item."
    ),
    "Hempseed": "Assigned to temperate oil crops; fibre hemp is mapped separately.",
    "Olives": "Assigned by economic use to temperate oil crops.",
    "Kapok fruit": "Assigned to plant-based fibres.",
}


# Full IMAGE NGFBFC/Crop38 labels supplied by the model. Price values are only
# defined for incumbent agricultural crop classes 1-16 and their irrigated
# counterparts 22-37. Grass and dedicated bioenergy categories are left NaN.
NGFBFC_LABELS: Dict[int, str] = {
    0: "grass",
    1: "Wheat (rainfed)",
    2: "Rice (rainfed)",
    3: "Maize (rainfed)",
    4: "Tropical cereals (rainfed)",
    5: "Other temperate cereals (rainfed)",
    6: "Pulses (rainfed)",
    7: "Soybeans (rainfed)",
    8: "Temperate oil crops (rainfed)",
    9: "Tropical oil crops (rainfed)",
    10: "Temperate roots & tubers (rainfed)",
    11: "Tropical roots & tubers (rainfed)",
    12: "Sugar Crops (rainfed)",
    13: "Oil & palm fruit (rainfed)",
    14: "Vegetables & fruits (rainfed)",
    15: "Other non-food & luxury & spices (rainfed)",
    16: "Plant based fibres (rainfed)",
    17: "Grains (biofuel)",
    18: "Oil crops (biofuel)",
    19: "Sugarcrops (biofuel)",
    20: "Woody biofuel",
    21: "Non-woody biofuel",
    22: "Wheat (irrigated)",
    23: "Rice (irrigated)",
    24: "Maize (irrigated)",
    25: "Tropical cereals (irrigated)",
    26: "Other temperate cereals (irrigated)",
    27: "Pulses (irrigated)",
    28: "Soybeans (irrigated)",
    29: "Temperate oil crops (irrigated)",
    30: "Tropical oil crops (irrigated)",
    31: "Temperate roots & tubers (irrigated)",
    32: "Tropical roots & tubers (irrigated)",
    33: "Sugar Crops (irrigated)",
    34: "Oil & palm fruit (irrigated)",
    35: "Vegetables & fruits (irrigated)",
    36: "Other non-food & luxury & spices (irrigated)",
    37: "Plant based fibres (irrigated)",
}


# =============================================================================
# GENERIC UTILITIES
# =============================================================================

def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    return text.replace("\xa0", " ").strip()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_text(c) for c in out.columns]
    for col in ("Area", "Item", "Element", "Unit", "Months", "Domain", "Flag", "Flag Description"):
        if col in out.columns:
            out[col] = out[col].map(normalize_text)
    if "Year" in out.columns:
        out["Year"] = pd.to_numeric(out["Year"], errors="coerce").astype("Int64")
    if "Value" in out.columns:
        out["Value"] = pd.to_numeric(out["Value"], errors="coerce")
    if "Area Code (M49)" in out.columns:
        out["Area Code (M49)"] = pd.to_numeric(out["Area Code (M49)"], errors="coerce").astype("Int64")
    return out


def require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing columns {missing}; available={list(df.columns)}")


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".xls":
        try:
            return pd.read_excel(path, sheet_name=0, engine="xlrd")
        except ImportError as exc:
            raise ImportError(
                f"Reading {path.name} requires xlrd>=2.0.1. Install with: pip install xlrd>=2.0.1"
            ) from exc
    if suffix in (".xlsx", ".xlsm"):
        return pd.read_excel(path, sheet_name=0)
    if suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def discover_files(data_dir: Path, stem: str) -> List[Path]:
    matches: List[Path] = []
    for ext in ("xls", "xlsx", "xlsm"):
        matches.extend(data_dir.glob(f"{stem}_*.{ext}"))
    # Avoid accidentally ingesting generated output or old all-in-one files
    # unless they follow the requested chunk naming pattern.
    return sorted(set(matches))


def read_concat(files: Sequence[Path], label: str) -> pd.DataFrame:
    if not files:
        raise FileNotFoundError(f"No {label} files found.")
    frames = []
    for path in files:
        x = clean_columns(read_table(path))
        x["_source_file"] = path.name
        frames.append(x)
    return pd.concat(frames, ignore_index=True, sort=False)


def validate_element(df: pd.DataFrame, expected: str, label: str) -> pd.DataFrame:
    require_columns(df, ["Element"], label)
    vals = sorted(v for v in df["Element"].dropna().unique() if v)
    if expected not in vals:
        raise ValueError(f"{label}: expected Element={expected!r}; found {vals[:20]}")
    return df[df["Element"] == expected].copy()


def year_set(df: pd.DataFrame) -> set[int]:
    if "Year" not in df.columns:
        return set()
    return {int(y) for y in df["Year"].dropna().unique()}



def check_reference_years(
    pp: pd.DataFrame,
    prod: pd.DataFrame,
    defl: pd.DataFrame,
    xr: pd.DataFrame,
    allow_partial: bool,
) -> List[str]:
    """Check that the downloaded chunks cover the intended processing windows.

    The model production weights use 2010-2019 only. Producer prices may use
    2000-2009 as fallback, so the deflator must span the full 2000-2019 price
    window. Only the 2010 SLC/USD exchange rate is required by the conversion.
    """
    main_years = set(range(CENTRAL_START, CENTRAL_END + 1))
    full_price_years = set(range(SUPPORT_START, CENTRAL_END + 1))
    messages: List[str] = []

    checks = [
        ("Producer prices", pp, main_years),
        ("Production (model-weight period)", prod, main_years),
        ("GDP deflator (full price window)", defl, full_price_years),
        ("Exchange rates", xr, {BASE_YEAR}),
    ]
    for label, df, req in checks:
        missing = sorted(req - year_set(df))
        if missing:
            msg = f"{label}: missing required year(s) {missing}"
            if not allow_partial:
                raise ValueError(msg)
            messages.append("WARNING: " + msg)

    fallback_missing = sorted(set(range(SUPPORT_START, SUPPORT_END + 1)) - year_set(pp))
    if fallback_missing:
        messages.append(
            "WARNING: Producer-price fallback period is incomplete; missing global year(s) "
            + str(fallback_missing)
        )
    return messages


def build_item_crosswalk(production_items: Iterable[str], pp_items: Iterable[str]) -> pd.DataFrame:
    item_to_group: Dict[str, int] = {}
    for gid, items in FAOSTAT_ITEMS_BY_IMAGE_GROUP.items():
        for item in items:
            item = normalize_text(item)
            if item in item_to_group:
                raise ValueError(f"Duplicate item in explicit crosswalk: {item}")
            item_to_group[item] = gid

    prod_items = sorted({normalize_text(x) for x in production_items if normalize_text(x)})
    pp_set = {normalize_text(x) for x in pp_items if normalize_text(x)}
    rows = []

    for item in prod_items:
        if item in item_to_group:
            gid = item_to_group[item]
            status = "included"
            note = SPECIAL_MAPPING_NOTES.get(item, "")
            include = True
        elif item in AMBIGUOUS_EXCLUDED:
            gid = np.nan
            status = "ambiguous_excluded"
            note = AMBIGUOUS_EXCLUDED[item]
            include = False
        elif item in EXCLUDED_ITEMS:
            gid = np.nan
            status = "excluded"
            note = EXCLUDED_ITEMS[item]
            include = False
        else:
            gid = np.nan
            status = "unmapped"
            note = "Not in explicit crosswalk; review before using."
            include = False

        if include:
            crop = IMAGE_GROUPS[int(gid)]["image_crop"]
            magnet = IMAGE_GROUPS[int(gid)]["magnet_crop_group"]
        else:
            crop = ""
            magnet = ""

        rows.append(
            {
                "faostat_item": item,
                "image_crop_id": gid,
                "image_crop": crop,
                "magnet_crop_group": magnet,
                "include_in_group_price": include,
                "mapping_status": status,
                "mapping_note": note,
                "producer_price_item_present": item in pp_set,
            }
        )
    return pd.DataFrame(rows)


def load_geo_mapping(path: Path) -> pd.DataFrame:
    m = clean_columns(read_table(path))
    require_columns(m, ["ISO numeric-3", "ISO short name", "IMAGE region"], "IMAGE country-region mapping")
    m["m49"] = pd.to_numeric(m["ISO numeric-3"], errors="coerce").astype("Int64")
    m["image_region"] = m["IMAGE region"].map(normalize_text)
    m["iso_short_name"] = m["ISO short name"].map(normalize_text)

    # Keep one row per M49. The supplied mapping should already satisfy this.
    dup = m[m["m49"].notna()].duplicated("m49", keep=False)
    if dup.any():
        d = m.loc[dup, ["m49", "iso_short_name", "image_region"]].sort_values("m49")
        raise ValueError(f"Duplicate M49 codes in region mapping:\n{d.to_string(index=False)}")

    return m[["m49", "iso_short_name", "image_region"]].copy()


def attach_region(df: pd.DataFrame, geo: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["m49"] = pd.to_numeric(out["Area Code (M49)"], errors="coerce").astype("Int64")
    out = out.merge(geo, on="m49", how="left")

    # Explicit safe historic override.
    for m49, region in REGION_OVERRIDE_BY_M49.items():
        out.loc[out["m49"] == m49, "image_region"] = region
        out.loc[out["m49"] == m49, "iso_short_name"] = "Sudan (former)"

    out["geo_status"] = "mapped"
    for m49, reason in EXCLUDED_FAOSTAT_M49.items():
        mask = out["m49"] == m49
        out.loc[mask, "geo_status"] = "excluded_aggregate"
        out.loc[mask, "image_region"] = np.nan

    out.loc[out["image_region"] == "GRL", "geo_status"] = "excluded_grl"
    out.loc[out["image_region"] == "GRL", "image_region"] = np.nan

    valid_regions = set(REGION_TO_NUMBER)
    invalid = out["image_region"].notna() & ~out["image_region"].isin(valid_regions)
    out.loc[invalid, "geo_status"] = "invalid_region"
    out.loc[invalid, "image_region"] = np.nan

    unmatched = out["image_region"].isna() & (out["geo_status"] == "mapped")
    out.loc[unmatched, "geo_status"] = "unmatched_m49"

    diag_cols = [c for c in ["m49", "Area", "iso_short_name", "geo_status", "_source_file"] if c in out.columns]
    diag = out[diag_cols].drop_duplicates().sort_values(["geo_status", "m49"], na_position="last")
    return out, diag


# =============================================================================
# FAOSTAT DATA PREPARATION
# =============================================================================


def load_supplemental_inputs(
    supplemental_file: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load user-supplied producer-price rows and explicit N=2 exception rules.

    Expected workbook sheets
    ------------------------
    ProducerPrices:
        FAOSTAT-like rows with at least Area Code (M49), Area, Item, Year,
        Value, and Element='Producer Price (SLC/tonne)'.

    N2Exceptions:
        Area Code (M49), Area, Item, Minimum observations, Reason.
        The only supported relaxed minimum is 2.

    A missing supplemental file is allowed; the base FAOSTAT-only workflow then
    behaves as before apart from the Sudan deflator repair and 33% residual
    regional-coverage gate.
    """
    pp_cols = [
        "Area Code (M49)", "Area", "Item", "Year", "Value", "Element",
        "Unit", "Months", "Flag", "Flag Description", "_source_file",
    ]
    n2_cols = ["Area Code (M49)", "Area", "Item", "Minimum observations", "Reason"]

    if supplemental_file is None:
        return pd.DataFrame(columns=pp_cols), pd.DataFrame(columns=n2_cols)

    if not supplemental_file.exists():
        raise FileNotFoundError(f"Supplemental input file not found: {supplemental_file}")
    if supplemental_file.suffix.lower() not in (".xlsx", ".xlsm"):
        raise ValueError(
            "Supplemental input must be an .xlsx/.xlsm workbook with sheets "
            "'ProducerPrices' and 'N2Exceptions'."
        )

    prices = clean_columns(pd.read_excel(supplemental_file, sheet_name="ProducerPrices"))
    require_columns(
        prices,
        ["Area Code (M49)", "Area", "Item", "Year", "Value", "Element"],
        "Supplemental producer prices",
    )
    prices = validate_element(
        prices, "Producer Price (SLC/tonne)", "Supplemental producer prices"
    )
    prices["Item"] = prices["Item"].map(normalize_text)
    if "Months" not in prices.columns:
        prices["Months"] = "Annual value"
    if "Unit" not in prices.columns:
        prices["Unit"] = "SLC/tonne"
    if "Flag" not in prices.columns:
        prices["Flag"] = "SUP"
    if "Flag Description" not in prices.columns:
        prices["Flag Description"] = "User-supplied supplemental observation"

    prices.dropna(subset=["Area Code (M49)", "Year", "Value"], inplace=True)
    prices = prices[
        np.isfinite(prices["Value"]) & (prices["Value"] > 0)
    ].copy()
    prices["_source_file"] = supplemental_file.name + "#ProducerPrices"
    prices["_supplemental_input"] = True

    n2 = pd.read_excel(supplemental_file, sheet_name="N2Exceptions")
    n2.columns = [normalize_text(c) for c in n2.columns]
    require_columns(
        n2,
        ["Area Code (M49)", "Area", "Item", "Minimum observations"],
        "N2 exception rules",
    )
    n2["Area Code (M49)"] = pd.to_numeric(
        n2["Area Code (M49)"], errors="coerce"
    ).astype("Int64")
    n2["Item"] = n2["Item"].map(normalize_text)
    n2["Minimum observations"] = pd.to_numeric(
        n2["Minimum observations"], errors="coerce"
    ).astype("Int64")
    if "Reason" not in n2.columns:
        n2["Reason"] = ""
    n2["Reason"] = n2["Reason"].map(normalize_text)
    n2.dropna(
        subset=["Area Code (M49)", "Item", "Minimum observations"], inplace=True
    )
    if not n2["Minimum observations"].eq(2).all():
        bad = n2.loc[
            ~n2["Minimum observations"].eq(2),
            ["Area Code (M49)", "Item", "Minimum observations"],
        ]
        raise ValueError(
            "Only N=2 exceptions are supported. Invalid rules:\n"
            + bad.to_string(index=False)
        )
    if n2.duplicated(["Area Code (M49)", "Item"]).any():
        dup = n2[n2.duplicated(["Area Code (M49)", "Item"], keep=False)]
        raise ValueError(
            "Duplicate N2 exception rules:\n"
            + dup[["Area Code (M49)", "Area", "Item"]].to_string(index=False)
        )
    return prices, n2


def n2_exception_key_set(n2_exceptions: Optional[pd.DataFrame]) -> set[Tuple[int, str]]:
    if n2_exceptions is None or n2_exceptions.empty:
        return set()
    return {
        (int(row["Area Code (M49)"]), normalize_text(row["Item"]))
        for _, row in n2_exceptions.iterrows()
    }


def prepare_inputs(
    data_dir: Path,
    mapping_file: Path,
    supplemental_file: Optional[Path] = None,
    allow_partial: bool = False,
):
    """Read and harmonise FAOSTAT chunks plus optional supplemental price inputs."""
    defl_files = sorted(set(
        discover_files(data_dir, "FAOSTAT_defl")
        + discover_files(data_dir, "FAOSTAT_def")
    ))
    files = {
        "pp": discover_files(data_dir, "FAOSTAT_pp"),
        "prod": discover_files(data_dir, "FAOSTAT_prod"),
        "defl": defl_files,
        "xr": discover_files(data_dir, "FAOSTAT_exchange"),
    }

    for key, paths in files.items():
        if not paths:
            raise FileNotFoundError(f"No files found for {key} in {data_dir}")

    pp = read_concat(files["pp"], "producer-price")
    prod = read_concat(files["prod"], "production")
    defl = read_concat(files["defl"], "deflator")
    xr = read_concat(files["xr"], "exchange-rate")

    supplemental_pp, n2_exceptions = load_supplemental_inputs(supplemental_file)
    if not supplemental_pp.empty:
        # Align columns without discarding provenance columns present only in one source.
        pp = pd.concat([pp, supplemental_pp], ignore_index=True, sort=False)

    pp = validate_element(pp, "Producer Price (SLC/tonne)", "Producer prices")
    prod = validate_element(prod, "Production", "Production")
    defl = validate_element(defl, "Value Standard Local Currency, 2015 prices", "Deflators")
    xr = validate_element(xr, "Standard local currency units per USD", "Exchange rates")

    require_columns(pp, ["Area Code (M49)", "Area", "Item", "Year", "Value"], "Producer prices")
    require_columns(prod, ["Area Code (M49)", "Area", "Item", "Year", "Value"], "Production")
    require_columns(defl, ["Area Code (M49)", "Area", "Item", "Year", "Value"], "Deflators")
    require_columns(xr, ["Area Code (M49)", "Area", "Year", "Value"], "Exchange rates")

    if "Item" in defl.columns:
        defl = defl[defl["Item"] == "GDP Deflator"].copy()
    if "Months" in pp.columns:
        pp = pp[(pp["Months"] == "") | (pp["Months"] == "Annual value")].copy()
    if "Months" in xr.columns:
        xr = xr[(xr["Months"] == "") | (xr["Months"] == "Annual value")].copy()

    for df in (pp, prod, defl, xr):
        df.dropna(subset=["Year", "Value", "Area Code (M49)"], inplace=True)
    pp = pp[np.isfinite(pp["Value"]) & (pp["Value"] > 0)].copy()
    prod = prod[np.isfinite(prod["Value"]) & (prod["Value"] >= 0)].copy()
    defl = defl[np.isfinite(defl["Value"]) & (defl["Value"] > 0)].copy()
    xr = xr[np.isfinite(xr["Value"]) & (xr["Value"] > 0)].copy()

    messages = check_reference_years(pp, prod, defl, xr, allow_partial)
    if supplemental_file is not None:
        messages.append(
            f"INFO: loaded {len(supplemental_pp)} supplemental producer-price rows "
            f"and {len(n2_exceptions)} N=2 exception rules from {supplemental_file.name}."
        )

    geo = load_geo_mapping(mapping_file)
    pp, pp_geo_diag = attach_region(pp, geo)
    prod, prod_geo_diag = attach_region(prod, geo)

    input_rows = [
        {"dataset": k, "file": p.name, "path": str(p)}
        for k, paths in files.items() for p in paths
    ]
    input_rows.append(
        {"dataset": "geo_mapping", "file": mapping_file.name, "path": str(mapping_file)}
    )
    if supplemental_file is not None:
        input_rows.extend([
            {
                "dataset": "supplemental_producer_prices",
                "file": supplemental_file.name,
                "path": str(supplemental_file) + "#ProducerPrices",
            },
            {
                "dataset": "n2_exception_rules",
                "file": supplemental_file.name,
                "path": str(supplemental_file) + "#N2Exceptions",
            },
        ])
    input_files = pd.DataFrame(input_rows)

    geo_diag = pd.concat(
        [pp_geo_diag.assign(dataset="producer_prices"), prod_geo_diag.assign(dataset="production")],
        ignore_index=True,
    ).drop_duplicates()

    return pp, prod, defl, xr, geo, input_files, geo_diag, messages, n2_exceptions

def estimate_country_item_prices(
    pp_real: pd.DataFrame,
    n2_exceptions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Estimate one representative real price per M49 x detailed commodity.

    The default minimum remains three observations.  Country-item pairs listed
    in ``N2Exceptions`` use a minimum of two observations.  The period hierarchy
    is otherwise unchanged: use all 2010-2019 observations when the applicable
    minimum is met, or add only the most recent 2000-2009 observations needed to
    reach that minimum.
    """
    x = pp_real.copy()
    x = x[
        x["image_region"].isin(REGION_TO_NUMBER)
        & x["price_2010usd_t"].notna()
        & np.isfinite(x["price_2010usd_t"])
        & (x["price_2010usd_t"] > 0)
    ].copy()

    n2_keys = n2_exception_key_set(n2_exceptions)
    main = x[x["Year"].between(CENTRAL_START, CENTRAL_END)].copy()
    fallback = x[x["Year"].between(SUPPORT_START, SUPPORT_END)].copy()

    keycols = ["m49", "image_region", "Item"]
    main_groups = {
        key: g.sort_values("Year").copy()
        for key, g in main.groupby(keycols, sort=False, dropna=False)
    }
    fallback_groups = {
        key: g.sort_values("Year", ascending=False).copy()
        for key, g in fallback.groupby(keycols, sort=False, dropna=False)
    }
    keys = sorted(
        set(main_groups) | set(fallback_groups),
        key=lambda z: (str(z[1]), str(z[0]), str(z[2])),
    )

    empty = x.iloc[0:0].copy()
    rows = []

    for key in keys:
        m49, image_region, item = key
        m = main_groups.get(key, empty)
        f = fallback_groups.get(key, empty)
        n_main_available = len(m)
        n2_allowed = (int(m49), normalize_text(item)) in n2_keys
        required_obs = 2 if n2_allowed else MIN_PRICE_OBS

        if n_main_available >= required_obs:
            used_main = m
            used_fallback = f.iloc[0:0]
            used = used_main
            source = "main_2010_2019"
        else:
            need = max(0, required_obs - n_main_available)
            used_main = m
            used_fallback = f.head(need)
            used = pd.concat([used_main, used_fallback], ignore_index=True)
            if len(used) >= required_obs:
                source = (
                    "fallback_2000_2009"
                    if n_main_available == 0
                    else "main_plus_fallback_2000_2009"
                )
            else:
                source = "insufficient_observations"

        accepted = len(used) >= required_obs
        price = float(used["price_2010usd_t"].median()) if accepted else np.nan

        main_years_available = sorted(int(y) for y in m["Year"].dropna().unique())
        main_years_used = sorted(int(y) for y in used_main["Year"].dropna().unique())
        fallback_years_used = sorted(int(y) for y in used_fallback["Year"].dropna().unique())
        years_used = sorted(main_years_used + fallback_years_used)

        flags = []
        if "Flag" in used.columns:
            flags = sorted({
                fcode
                for cell in used["Flag"].fillna("")
                for fcode in str(cell).split("|")
                if fcode
            })

        if len(m):
            area_label = m.sort_values("Year")["Area"].iloc[-1]
        elif len(f):
            area_label = f.sort_values("Year")["Area"].iloc[-1]
        else:
            area_label = ""

        n_fallback_used = len(used_fallback)
        n_used = len(used)
        rows.append({
            "m49": int(m49),
            "Area": area_label,
            "image_region": image_region,
            "faostat_item": item,
            "country_item_price_2010usd_t": price,
            "price_source": source,
            "accepted_price": accepted,
            "min_price_obs_required": required_obs,
            "n2_exception_allowed": n2_allowed,
            "n2_exception_applied": bool(
                accepted and n2_allowed and n_used < MIN_PRICE_OBS
            ),
            "uses_fallback": bool(accepted and n_fallback_used > 0),
            "fallback_only": bool(
                accepted and n_main_available == 0 and n_fallback_used >= required_obs
            ),
            "n_main_obs_available": n_main_available,
            "n_main_obs_used": len(used_main),
            "n_fallback_obs_used": n_fallback_used,
            "n_total_obs_used": n_used,
            "fallback_observation_fraction": (
                n_fallback_used / n_used if accepted and n_used > 0 else np.nan
            ),
            "main_years_available": ",".join(map(str, main_years_available)),
            "main_years_used": ",".join(map(str, main_years_used)),
            "fallback_years_used": ",".join(map(str, fallback_years_used)),
            "years_used": ",".join(map(str, years_used)),
            "oldest_year_used": min(years_used) if years_used else np.nan,
            "newest_year_used": max(years_used) if years_used else np.nan,
            "faostat_flags_used": "|".join(flags),
        })

    return pd.DataFrame(rows)

def representative_production(prod: pd.DataFrame, crosswalk: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = prod.copy()
    x = x[(x["Year"] >= CENTRAL_START) & (x["Year"] <= CENTRAL_END)].copy()
    x["m49"] = x["Area Code (M49)"].astype("Int64")
    x["Item"] = x["Item"].map(normalize_text)

    # Remove overlapping aggregates / non-IMAGE26 rows, but keep Sudan former override.
    x = x[x["image_region"].isin(REGION_TO_NUMBER)].copy()

    # Sum duplicates within country-item-year. This also makes the denominator
    # invariant to duplicated rows in an export.
    annual = (
        x.groupby(["m49", "image_region", "Item", "Year"], as_index=False)
        .agg(
            production_t=("Value", "sum"),
            Area=("Area", "last"),
        )
    )

    n_ref_years = CENTRAL_END - CENTRAL_START + 1
    q = (
        annual.groupby(["m49", "image_region", "Item"], as_index=False)
        .agg(
            Area=("Area", "last"),
            production_sum_2010_2019_t=("production_t", "sum"),
            n_production_year_rows=("Year", "nunique"),
            first_production_year=("Year", "min"),
            last_production_year=("Year", "max"),
        )
    )
    q["q_ref_t_per_year"] = q["production_sum_2010_2019_t"] / n_ref_years

    cw = crosswalk.rename(columns={"faostat_item": "Item"})
    q = q.merge(cw, on="Item", how="left")

    ambiguous = q[q["mapping_status"].isin(["ambiguous_excluded", "excluded", "unmapped"])].copy()
    included = q[q["include_in_group_price"] == True].copy()  # noqa: E712
    return included, ambiguous


def build_region_wide(regional: pd.DataFrame, value_col: str = "final_price_2010usd_t") -> pd.DataFrame:
    p = regional.pivot(index=["image_region_number", "image_region_index0", "image_region"],
                       columns="image_crop_id", values=value_col)
    p = p.reindex(columns=range(1, 17))
    p.columns = [f"{gid:02d}_{IMAGE_GROUPS[gid]['image_crop']}" for gid in p.columns]
    return p.reset_index().sort_values("image_region_number")


def build_ngfbfc38(regional: pd.DataFrame) -> pd.DataFrame:
    """Build Crop38 prices from final regional prices.

    Agricultural rainfed and irrigated classes use final_price_2010usd_t. The
    latter equals the original data-derived model price wherever one existed and
    contains regional crop-class imputation only for formerly missing cells.
    """
    price_lookup = {
        (row.image_region, int(row.image_crop_id)): row.final_price_2010usd_t
        for row in regional.itertuples()
    }
    quality_lookup = {
        (row.image_region, int(row.image_crop_id)): row.final_price_quality
        for row in regional.itertuples()
    }
    source_lookup = {
        (row.image_region, int(row.image_crop_id)): row.final_price_source
        for row in regional.itertuples()
    }
    cov_lookup = {
        (row.image_region, int(row.image_crop_id)): row.production_coverage
        for row in regional.itertuples()
    }

    rows = []
    for region_no, region in IMAGE26_REGIONS:
        for ngid in range(38):
            if 1 <= ngid <= 16:
                base_gid = ngid
                price = price_lookup.get((region, base_gid), np.nan)
                quality = quality_lookup.get((region, base_gid), "")
                coverage = cov_lookup.get((region, base_gid), np.nan)
                source = source_lookup.get((region, base_gid), "")
            elif 22 <= ngid <= 37:
                base_gid = ngid - 21
                price = price_lookup.get((region, base_gid), np.nan)
                quality = quality_lookup.get((region, base_gid), "")
                coverage = cov_lookup.get((region, base_gid), np.nan)
                base_source = source_lookup.get((region, base_gid), "")
                source = (
                    f"same_as_rainfed_crop:{base_source}"
                    if base_source else "same_as_rainfed_crop"
                )
            else:
                base_gid = np.nan
                price = np.nan
                quality = "not_applicable"
                coverage = np.nan
                source = "grass_or_dedicated_bioenergy_not_priced_here"

            rows.append(
                {
                    "image_region_number": region_no,
                    "image_region_index0": region_no - 1,
                    "image_region": region,
                    "ngfbfc_id": ngid,
                    "ngfbfc_label": NGFBFC_LABELS[ngid],
                    "base_image_crop_id": base_gid,
                    "price_2010usd_t": price,
                    "production_coverage": coverage,
                    "quality": quality,
                    "price_source": source,
                }
            )
    return pd.DataFrame(rows)


def ambiguous_production_summary(ambiguous: pd.DataFrame) -> pd.DataFrame:
    if ambiguous.empty:
        return ambiguous
    return (
        ambiguous.groupby(
            ["image_region", "Item", "mapping_status", "mapping_note"], as_index=False, dropna=False
        )
        .agg(
            q_ref_t_per_year=("q_ref_t_per_year", "sum"),
            n_entities=("m49", "nunique"),
        )
        .sort_values(["image_region", "mapping_status", "q_ref_t_per_year"], ascending=[True, True, False])
    )


# =============================================================================
# OUTPUT / METADATA
# =============================================================================

def autosize_excel(path: Path) -> None:
    """Apply light formatting without scanning entire large worksheets."""
    from openpyxl import load_workbook
    from openpyxl.styles import Font, PatternFill

    wb = load_workbook(path)
    for ws in wb.worksheets:
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        for cell in ws[1]:
            cell.font = Font(bold=True)
            cell.fill = PatternFill("solid", fgColor="D9EAF7")

        # Estimate widths from header + first 200 data rows only.
        sample_end = min(ws.max_row, 201)
        for col_idx in range(1, ws.max_column + 1):
            max_len = 0
            for row_idx in range(1, sample_end + 1):
                val = ws.cell(row=row_idx, column=col_idx).value
                txt = "" if val is None else str(val)
                max_len = max(max_len, len(txt))
            letter = ws.cell(row=1, column=col_idx).column_letter
            ws.column_dimensions[letter].width = min(max(max_len + 2, 10), 42)
    wb.save(path)




def build_fallback_dependence_priority(regional: pd.DataFrame) -> pd.DataFrame:
    """Rank otherwise usable region-crop prices by production dependent on fallback data.

    This is diagnostic only: fallback dependence does not make a price non-model-ready.
    """
    x = regional[
        (regional["q_total_ref_t_per_year"] > 0)
        & (regional["q_fallback_dependent_priced_ref_t_per_year"] > 0)
    ].copy()
    if x.empty:
        return x
    x = x.sort_values(
        [
            "q_fallback_dependent_priced_ref_t_per_year",
            "fallback_dependent_share_of_priced_production",
        ],
        ascending=[False, False],
    ).reset_index(drop=True)
    x["fallback_priority_rank"] = np.arange(1, len(x) + 1)
    return x


def _join_reasons(*parts: pd.Series) -> pd.Series:
    """Join non-empty reason strings row-wise without duplicate separators."""
    frame = pd.concat(parts, axis=1).fillna("")
    return frame.apply(
        lambda row: "|".join(dict.fromkeys([str(v) for v in row if str(v)])), axis=1
    )


# =============================================================================
# ROBUSTNESS CONFIGURATION: OUTLIERS, MACRO AUDIT, MODEL GATING
# =============================================================================

# Same-item cross-country outlier diagnostics.
OUTLIER_MIN_COUNTRIES = 5
OUTLIER_ROBUST_Z_FLAG = 5.0
OUTLIER_ROBUST_Z_EXCLUDE = 5.0
ITEM_RATIO_FLAG = 10.0
ITEM_RATIO_EXCLUDE = 20.0
ITEM_RATIO_CATASTROPHIC = 100.0

# Systemic country-scale diagnostics catch currency/unit discontinuities even
# when individual commodities have sparse cross-country coverage.
SYSTEMIC_MIN_COMPARABLE_ITEMS = 5
SYSTEMIC_COUNTRY_RATIO_FLAG = 20.0
SYSTEMIC_COUNTRY_RATIO_EXCLUDE = 100.0
SYSTEMIC_MIN_SHARE_EXTREME = 0.50

# Final region-level plausibility fail-safe.
REGIONAL_RATIO_FLAG = 20.0
REGIONAL_RATIO_CATASTROPHIC = 100.0
REGIONAL_BENCHMARK_MIN_REGIONS = 5


def macro_tables(defl: pd.DataFrame, xr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build annual/base macro tables and repair Sudan (former) deflator gaps.

    For M49=736 (Sudan former), only the GDP-deflator ratio is proxied from
    current Sudan (M49=729), and only where the former entity has no deflator
    observation for that year.  The former-Sudan 2010 exchange rate is retained.
    Source-M49 columns make the bridge fully auditable.
    """
    d = defl.copy()
    d["m49"] = d["Area Code (M49)"].astype("Int64")
    d = d[np.isfinite(d["Value"]) & (d["Value"] > 0)].copy()
    d = (
        d.groupby(["m49", "Year"], as_index=False)
        .agg(deflator=("Value", "median"), deflator_n=("Value", "size"))
    )
    d["deflator_source_m49"] = d["m49"]

    for target_m49, source_m49 in DEFLATOR_PROXY_BY_M49.items():
        proxy = d[d["m49"] == source_m49].copy()
        if proxy.empty:
            continue
        existing_years = set(
            int(y) for y in d.loc[d["m49"] == target_m49, "Year"].dropna()
        )
        proxy = proxy[~proxy["Year"].isin(existing_years)].copy()
        if proxy.empty:
            continue
        proxy["m49"] = target_m49
        proxy["deflator_source_m49"] = source_m49
        d = pd.concat([d, proxy], ignore_index=True, sort=False)

    x = xr.copy()
    x["m49"] = x["Area Code (M49)"].astype("Int64")
    x = x[np.isfinite(x["Value"]) & (x["Value"] > 0)].copy()
    x = (
        x.groupby(["m49", "Year"], as_index=False)
        .agg(xr_slc_per_usd=("Value", "median"), xr_n=("Value", "size"))
    )
    x["xr_source_m49"] = x["m49"]

    d0 = d[d["Year"] == BASE_YEAR][
        ["m49", "deflator", "deflator_source_m49"]
    ].rename(columns={
        "deflator": "deflator_base",
        "deflator_source_m49": "deflator_base_source_m49",
    })
    x0 = x[x["Year"] == BASE_YEAR][
        ["m49", "xr_slc_per_usd", "xr_source_m49"]
    ].rename(columns={
        "xr_slc_per_usd": "xr_base",
        "xr_source_m49": "xr_base_source_m49",
    })
    base = d0.merge(x0, on="m49", how="outer")
    return d, x, base

def convert_pp_to_real_2010usd(
    pp: pd.DataFrame, defl: pd.DataFrame, xr: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert SLC/t producer prices to constant 2010 USD/t with explicit audit reasons."""
    d, _, base = macro_tables(defl, xr)
    out = pp.copy()
    out["m49"] = out["Area Code (M49)"].astype("Int64")
    out = out.merge(
        d[["m49", "Year", "deflator", "deflator_source_m49"]],
        on=["m49", "Year"], how="left"
    )
    out = out.merge(base, on="m49", how="left")

    out["uses_deflator_proxy"] = (
        out["deflator_source_m49"].notna()
        & (out["deflator_source_m49"] != out["m49"])
    )
    out["uses_deflator_base_proxy"] = (
        out["deflator_base_source_m49"].notna()
        & (out["deflator_base_source_m49"] != out["m49"])
    )
    out["macro_proxy_note"] = np.where(
        out["uses_deflator_proxy"] | out["uses_deflator_base_proxy"],
        "GDP_deflator_proxy_from_M49_"
        + out["deflator_source_m49"].fillna(
            out["deflator_base_source_m49"]
        ).astype("Int64").astype(str),
        "",
    )

    out["missing_deflator_year"] = out["deflator"].isna() | ~(out["deflator"] > 0)
    out["missing_deflator_base"] = out["deflator_base"].isna() | ~(out["deflator_base"] > 0)
    out["missing_exchange_rate_base"] = out["xr_base"].isna() | ~(out["xr_base"] > 0)

    reason = np.full(len(out), "", dtype=object)
    for col, label in [
        ("missing_deflator_year", "missing_or_invalid_year_deflator"),
        ("missing_deflator_base", "missing_or_invalid_2010_deflator"),
        ("missing_exchange_rate_base", "missing_or_invalid_2010_slc_usd"),
    ]:
        mask = out[col].to_numpy(bool)
        reason[mask] = [(r + "|" + label).strip("|") for r in reason[mask]]

    out["macro_failure_reason"] = reason
    out["macro_ok"] = out["macro_failure_reason"].eq("")

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        converted = out["Value"] * out["deflator_base"] / out["deflator"] / out["xr_base"]
    out["price_2010usd_t"] = np.where(
        out["macro_ok"] & np.isfinite(converted) & (converted > 0), converted, np.nan
    )

    bad_calc = out["macro_ok"] & ~(
        np.isfinite(out["price_2010usd_t"]) & (out["price_2010usd_t"] > 0)
    )
    out.loc[bad_calc, "macro_ok"] = False
    out.loc[bad_calc, "macro_failure_reason"] = "nonfinite_or_nonpositive_converted_price"

    out["macro_period"] = np.select(
        [
            out["Year"].between(CENTRAL_START, CENTRAL_END),
            out["Year"].between(SUPPORT_START, SUPPORT_END),
        ],
        ["main_2010_2019", "fallback_2000_2009"],
        default="outside_processing_window",
    )
    out["mapped_image26"] = out["image_region"].isin(REGION_TO_NUMBER)

    summary = (
        out.groupby(["m49", "Area", "image_region", "macro_period"], dropna=False)
        .agg(
            n_pp_rows=("Value", "size"),
            n_macro_ok=("macro_ok", "sum"),
            n_macro_missing=("macro_ok", lambda x: int((~x).sum())),
            deflator_2010=("deflator_base", "first"),
            deflator_2010_source_m49=("deflator_base_source_m49", "first"),
            xr_2010_slc_per_usd=("xr_base", "first"),
            xr_2010_source_m49=("xr_base_source_m49", "first"),
            n_rows_using_deflator_proxy=("uses_deflator_proxy", "sum"),
        )
        .reset_index()
    )
    return out, summary

def collapse_duplicate_pp(pp_real: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate annual observations while preserving macro/flag diagnostics."""
    keys = ["m49", "Area", "image_region", "Item", "Year"]
    agg = {
        "price_2010usd_t": "median",
        "Value": "median",
        "macro_ok": "max",
        "macro_failure_reason": lambda s: "|".join(
            sorted({r for cell in s.fillna("") for r in str(cell).split("|") if r})
        ),
    }
    for col in ["deflator", "deflator_base", "xr_base", "macro_period", "mapped_image26"]:
        if col in pp_real.columns:
            agg[col] = "first"
    if "Flag" in pp_real.columns:
        agg["Flag"] = lambda s: "|".join(sorted({normalize_text(x) for x in s if normalize_text(x)}))
    if "Flag Description" in pp_real.columns:
        agg["Flag Description"] = lambda s: "|".join(sorted({normalize_text(x) for x in s if normalize_text(x)}))
    x = pp_real.groupby(keys, as_index=False, dropna=False).agg(agg)
    counts = pp_real.groupby(keys, as_index=False, dropna=False).size().rename(columns={"size": "duplicate_count"})
    return x.merge(counts, on=keys, how="left")


def add_outlier_diagnostics(country_prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnose and conservatively exclude catastrophic/systemic price-scale outliers.

    Three complementary checks are used:
      1) same-item robust z-score of log prices;
      2) ratio to the cross-country median for the exact same item;
      3) systemic country price-scale factor across many different items.

    The third check is crucial for currency redenomination/unit discontinuities: an
    entire country can be rejected even when a sparse individual item has too few
    peers for an item-specific robust z-score.
    """
    out = country_prices.copy()
    valid = out["accepted_price"] & np.isfinite(out["country_item_price_2010usd_t"]) & (out["country_item_price_2010usd_t"] > 0)
    out["log_price"] = np.where(valid, np.log(out["country_item_price_2010usd_t"]), np.nan)

    # Exact-item benchmark statistics.
    bench = (
        out[valid]
        .groupby("faostat_item", as_index=False)
        .agg(
            item_median_price_2010usd_t=("country_item_price_2010usd_t", "median"),
            item_peer_country_count=("m49", "nunique"),
        )
    )
    out = out.merge(bench, on="faostat_item", how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        out["item_price_ratio_to_median"] = (
            out["country_item_price_2010usd_t"] / out["item_median_price_2010usd_t"]
        )
        out["item_log10_ratio_to_median"] = np.log10(out["item_price_ratio_to_median"])

    out["robust_z_log_price"] = np.nan
    for item, idx in out[out["log_price"].notna()].groupby("faostat_item").groups.items():
        vals = out.loc[idx, "log_price"]
        if len(vals) < OUTLIER_MIN_COUNTRIES:
            continue
        med = float(vals.median())
        mad = float(np.median(np.abs(vals - med)))
        if not np.isfinite(mad) or mad <= 0:
            continue
        out.loc[idx, "robust_z_log_price"] = 0.67448975 * (vals - med) / mad

    ratio = out["item_price_ratio_to_median"]
    peers = out["item_peer_country_count"].fillna(0)
    out["item_ratio_flag"] = valid & (peers >= 3) & (
        (ratio >= ITEM_RATIO_FLAG) | (ratio <= 1.0 / ITEM_RATIO_FLAG)
    )
    out["item_catastrophic_flag"] = valid & (peers >= 3) & (
        (ratio >= ITEM_RATIO_CATASTROPHIC) | (ratio <= 1.0 / ITEM_RATIO_CATASTROPHIC)
    )
    out["item_robust_z_flag"] = out["robust_z_log_price"].abs() >= OUTLIER_ROBUST_Z_FLAG
    out["item_extreme_statistical_exclusion"] = valid & (
        out["robust_z_log_price"].abs() >= OUTLIER_ROBUST_Z_EXCLUDE
    ) & (peers >= OUTLIER_MIN_COUNTRIES) & (
        (ratio >= ITEM_RATIO_EXCLUDE) | (ratio <= 1.0 / ITEM_RATIO_EXCLUDE)
    )

    # Systemic country-scale diagnostic: median same-item log10 ratio across items.
    comparable = out[
        valid
        & (out["item_peer_country_count"].fillna(0) >= OUTLIER_MIN_COUNTRIES)
        & out["item_log10_ratio_to_median"].notna()
    ].copy()
    if comparable.empty:
        country_diag = pd.DataFrame(columns=[
            "m49", "Area", "image_region", "n_comparable_items",
            "country_median_log10_ratio", "country_median_ratio_to_item_median",
            "share_items_beyond_10x", "systemic_country_scale_flag",
            "systemic_country_scale_exclusion",
        ])
    else:
        comparable["beyond_10x"] = comparable["item_log10_ratio_to_median"].abs() >= math.log10(ITEM_RATIO_FLAG)
        country_diag = (
            comparable.groupby(["m49", "Area", "image_region"], as_index=False)
            .agg(
                n_comparable_items=("faostat_item", "nunique"),
                country_median_log10_ratio=("item_log10_ratio_to_median", "median"),
                share_items_beyond_10x=("beyond_10x", "mean"),
            )
        )
        country_diag["country_median_ratio_to_item_median"] = np.power(
            10.0, country_diag["country_median_log10_ratio"]
        )
        country_diag["systemic_country_scale_flag"] = (
            (country_diag["n_comparable_items"] >= SYSTEMIC_MIN_COMPARABLE_ITEMS)
            & (country_diag["share_items_beyond_10x"] >= SYSTEMIC_MIN_SHARE_EXTREME)
            & (country_diag["country_median_log10_ratio"].abs() >= math.log10(SYSTEMIC_COUNTRY_RATIO_FLAG))
        )
        country_diag["systemic_country_scale_exclusion"] = (
            (country_diag["n_comparable_items"] >= SYSTEMIC_MIN_COMPARABLE_ITEMS)
            & (country_diag["share_items_beyond_10x"] >= SYSTEMIC_MIN_SHARE_EXTREME)
            & (country_diag["country_median_log10_ratio"].abs() >= math.log10(SYSTEMIC_COUNTRY_RATIO_EXCLUDE))
        )

    out = out.merge(
        country_diag[[
            "m49", "systemic_country_scale_flag", "systemic_country_scale_exclusion",
            "country_median_ratio_to_item_median", "country_median_log10_ratio",
            "n_comparable_items", "share_items_beyond_10x",
        ]] if not country_diag.empty else pd.DataFrame(columns=[
            "m49", "systemic_country_scale_flag", "systemic_country_scale_exclusion",
            "country_median_ratio_to_item_median", "country_median_log10_ratio",
            "n_comparable_items", "share_items_beyond_10x",
        ]),
        on="m49", how="left"
    )
    for col in ["systemic_country_scale_flag", "systemic_country_scale_exclusion"]:
        out[col] = out[col].eq(True)

    out["price_outlier_flag"] = (
        out["item_ratio_flag"] | out["item_robust_z_flag"] | out["systemic_country_scale_flag"]
    )
    out["price_exclusion_flag"] = (
        out["item_catastrophic_flag"]
        | out["item_extreme_statistical_exclusion"]
        | out["systemic_country_scale_exclusion"]
    )

    reasons = []
    for row in out.itertuples():
        r = []
        if getattr(row, "systemic_country_scale_exclusion", False):
            r.append("systemic_country_price_scale_anomaly")
        if getattr(row, "item_catastrophic_flag", False):
            r.append("catastrophic_same_item_ratio")
        if getattr(row, "item_extreme_statistical_exclusion", False):
            r.append("extreme_same_item_statistical_outlier")
        reasons.append("|".join(r))
    out["price_exclusion_reason"] = reasons
    out["valid_for_regional_aggregation"] = out["accepted_price"] & ~out["price_exclusion_flag"]

    if not country_diag.empty:
        country_diag = country_diag.sort_values(
            ["systemic_country_scale_exclusion", "country_median_log10_ratio"],
            ascending=[False, False]
        )
    return out, country_diag



def build_country_item_macro_audit(
    pp_real_raw: pd.DataFrame,
    q: pd.DataFrame,
    n2_exceptions: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Audit PP availability and macro conversion success for every produced country-item."""
    p = pp_real_raw[
        pp_real_raw["macro_period"].isin(["main_2010_2019", "fallback_2000_2009"])
        & pp_real_raw["image_region"].isin(REGION_TO_NUMBER)
    ].copy()
    p["is_main"] = p["macro_period"].eq("main_2010_2019")
    p["is_fallback"] = p["macro_period"].eq("fallback_2000_2009")
    p["macro_ok_main"] = p["macro_ok"] & p["is_main"]
    p["macro_ok_fallback"] = p["macro_ok"] & p["is_fallback"]
    p["failure_reason_cell"] = np.where(~p["macro_ok"], p["macro_failure_reason"], "")

    if p.empty:
        stats = pd.DataFrame(columns=["m49", "image_region", "Item"])
    else:
        stats = (
            p.groupby(["m49", "image_region", "Item"], as_index=False, dropna=False)
            .agg(
                n_pp_rows_total=("Value", "size"),
                n_pp_rows_main=("is_main", "sum"),
                n_pp_rows_fallback=("is_fallback", "sum"),
                n_macro_ok_total=("macro_ok", "sum"),
                n_macro_ok_main=("macro_ok_main", "sum"),
                n_macro_ok_fallback=("macro_ok_fallback", "sum"),
                n_macro_failed=("macro_ok", lambda x: int((~x).sum())),
                macro_failure_reasons=("failure_reason_cell", lambda s: "|".join(
                    sorted({r for cell in s if cell for r in str(cell).split("|") if r})
                )),
            )
        )

    qa = q[[
        "m49", "image_region", "Item", "Area", "image_crop_id", "image_crop",
        "magnet_crop_group", "q_ref_t_per_year"
    ]].copy()
    audit = qa.merge(stats, on=["m49", "image_region", "Item"], how="left")
    count_cols = [
        "n_pp_rows_total", "n_pp_rows_main", "n_pp_rows_fallback",
        "n_macro_ok_total", "n_macro_ok_main", "n_macro_ok_fallback", "n_macro_failed",
    ]
    for col in count_cols:
        audit[col] = audit[col].fillna(0).astype(int)
    audit["macro_failure_reasons"] = audit["macro_failure_reasons"].fillna("")

    n2_keys = n2_exception_key_set(n2_exceptions)
    audit["min_price_obs_required"] = [
        2 if (int(r.m49), normalize_text(r.Item)) in n2_keys else MIN_PRICE_OBS
        for r in audit.itertuples()
    ]
    audit["n2_exception_allowed"] = audit["min_price_obs_required"].eq(2)

    def classify(row) -> str:
        required = int(row.min_price_obs_required)
        if row.n_pp_rows_total == 0:
            return "no_producer_price_rows"
        if row.n_macro_ok_total == 0:
            return "all_price_rows_fail_macro_conversion"
        if row.n_macro_ok_total < required and row.n_pp_rows_total >= required:
            return "macro_failures_contribute_to_insufficient_observations"
        if row.n_macro_failed > 0:
            return "some_macro_failures_but_convertible_observations_exist"
        return "no_macro_conversion_problem"

    audit["macro_problem_class"] = [classify(r) for r in audit.itertuples()]
    audit["q_at_risk_due_macro_t_per_year"] = np.where(
        audit["macro_problem_class"].isin([
            "all_price_rows_fail_macro_conversion",
            "macro_failures_contribute_to_insufficient_observations",
        ]),
        audit["q_ref_t_per_year"], 0.0
    )
    return audit

def build_macro_failure_outputs(
    pp_real_raw: pd.DataFrame,
    macro_country_summary: pd.DataFrame,
    country_item_macro_audit: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return failed PP rows, country/reason summary, and region-crop production impact."""
    fail_cols = [c for c in [
        "m49", "Area", "image_region", "Item", "Year", "Value", "macro_period",
        "macro_failure_reason", "deflator", "deflator_base", "xr_base", "Flag",
        "Flag Description", "_source_file"
    ] if c in pp_real_raw.columns]
    failed = pp_real_raw[
        (~pp_real_raw["macro_ok"])
        & pp_real_raw["macro_period"].isin(["main_2010_2019", "fallback_2000_2009"])
    ][fail_cols].copy()

    if failed.empty:
        failure_summary = pd.DataFrame(columns=[
            "image_region", "m49", "Area", "macro_period",
            "macro_failure_reason", "n_failed_rows"
        ])
    else:
        failure_summary = (
            failed.groupby(
                ["image_region", "m49", "Area", "macro_period", "macro_failure_reason"],
                as_index=False, dropna=False
            ).size().rename(columns={"size": "n_failed_rows"})
            .sort_values("n_failed_rows", ascending=False)
        )

    impact = (
        country_item_macro_audit.groupby(
            ["image_region", "image_crop_id", "image_crop", "magnet_crop_group"],
            as_index=False,
        )
        .agg(
            q_total_ref_t_per_year=("q_ref_t_per_year", "sum"),
            q_at_risk_due_macro_t_per_year=("q_at_risk_due_macro_t_per_year", "sum"),
            n_country_items=("Item", "size"),
            n_country_items_all_macro_fail=(
                "macro_problem_class",
                lambda s: int((s == "all_price_rows_fail_macro_conversion").sum()),
            ),
            n_country_items_macro_insufficient=(
                "macro_problem_class",
                lambda s: int((s == "macro_failures_contribute_to_insufficient_observations").sum()),
            ),
        )
    )
    impact["macro_at_risk_share"] = np.where(
        impact["q_total_ref_t_per_year"] > 0,
        impact["q_at_risk_due_macro_t_per_year"] / impact["q_total_ref_t_per_year"],
        np.nan,
    )
    impact = impact.sort_values("q_at_risk_due_macro_t_per_year", ascending=False)
    return failed, failure_summary, impact



def aggregate_regional_prices(
    q: pd.DataFrame,
    country_prices: pd.DataFrame,
    country_item_macro_audit: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate valid prices and decompose genuinely unpriced production by cause.

    Fallback-assisted country-item estimates are fully eligible for regional
    aggregation. Their dependence is reported diagnostically, not used as a
    separate model-readiness gate.
    """
    pcols = [
        "m49", "image_region", "faostat_item", "country_item_price_2010usd_t",
        "price_source", "accepted_price", "valid_for_regional_aggregation",
        "price_outlier_flag", "price_exclusion_flag", "price_exclusion_reason",
        "item_price_ratio_to_median", "robust_z_log_price",
        "systemic_country_scale_exclusion", "country_median_ratio_to_item_median",
        "uses_fallback", "fallback_only", "fallback_observation_fraction",
        "n_main_obs_available", "n_main_obs_used", "n_fallback_obs_used",
        "n_total_obs_used", "years_used", "min_price_obs_required",
        "n2_exception_allowed", "n2_exception_applied",
    ]
    pcols = [c for c in pcols if c in country_prices.columns]
    p = country_prices[pcols].rename(columns={"faostat_item": "Item"})

    c = q.merge(p, on=["m49", "image_region", "Item"], how="left")
    if country_item_macro_audit is not None and not country_item_macro_audit.empty:
        mac_cols = [
            "m49", "image_region", "Item", "n_pp_rows_total", "n_pp_rows_main",
            "n_pp_rows_fallback", "n_macro_ok_total", "n_macro_ok_main",
            "n_macro_ok_fallback", "n_macro_failed", "macro_problem_class",
            "macro_failure_reasons",
        ]
        mac_cols = [col for col in mac_cols if col in country_item_macro_audit.columns]
        c = c.merge(
            country_item_macro_audit[mac_cols],
            on=["m49", "image_region", "Item"],
            how="left",
        )
    else:
        c["n_pp_rows_total"] = 0
        c["n_macro_ok_total"] = 0
        c["macro_problem_class"] = ""

    c["accepted_price"] = c["accepted_price"].eq(True) if "accepted_price" in c.columns else False
    c["valid_for_regional_aggregation"] = (
        c["valid_for_regional_aggregation"].eq(True)
        if "valid_for_regional_aggregation" in c.columns else False
    )
    c["price_exclusion_flag"] = (
        c["price_exclusion_flag"].eq(True)
        if "price_exclusion_flag" in c.columns else False
    )
    c["uses_fallback"] = c["uses_fallback"].eq(True) if "uses_fallback" in c.columns else False
    c["fallback_only"] = c["fallback_only"].eq(True) if "fallback_only" in c.columns else False
    c["fallback_observation_fraction"] = pd.to_numeric(
        c.get("fallback_observation_fraction", np.nan), errors="coerce"
    ).fillna(0.0)
    for col in ["n_pp_rows_total", "n_macro_ok_total"]:
        c[col] = c[col].fillna(0).astype(int)

    def unpriced_reason(row) -> str:
        if row.valid_for_regional_aggregation:
            return "priced"
        if getattr(row, "price_exclusion_flag", False):
            return "excluded_price_outlier"
        mpc = getattr(row, "macro_problem_class", "")
        if mpc in {
            "all_price_rows_fail_macro_conversion",
            "macro_failures_contribute_to_insufficient_observations",
        }:
            return "macro_conversion_failure"
        if getattr(row, "n_pp_rows_total", 0) <= 0:
            return "no_producer_price_data"
        if getattr(row, "accepted_price", False) is False:
            return "insufficient_price_observations"
        return "other_unpriced_reason"

    c["unpriced_reason"] = [unpriced_reason(r) for r in c.itertuples()]
    c["priced_q"] = np.where(
        c["valid_for_regional_aggregation"], c["q_ref_t_per_year"], 0.0
    )
    c["main_only_priced_q"] = np.where(
        c["valid_for_regional_aggregation"] & ~c["uses_fallback"],
        c["q_ref_t_per_year"], 0.0
    )
    c["fallback_dependent_priced_q"] = np.where(
        c["valid_for_regional_aggregation"] & c["uses_fallback"],
        c["q_ref_t_per_year"], 0.0
    )
    c["fallback_only_priced_q"] = np.where(
        c["valid_for_regional_aggregation"] & c["fallback_only"],
        c["q_ref_t_per_year"], 0.0
    )
    c["fallback_obs_weight_q"] = np.where(
        c["valid_for_regional_aggregation"],
        c["q_ref_t_per_year"] * c["fallback_observation_fraction"],
        0.0,
    )
    c["value_weight"] = np.where(
        c["valid_for_regional_aggregation"],
        c["q_ref_t_per_year"] * c["country_item_price_2010usd_t"],
        0.0,
    )

    unpriced_reasons = [
        "no_producer_price_data",
        "insufficient_price_observations",
        "macro_conversion_failure",
        "excluded_price_outlier",
        "other_unpriced_reason",
    ]
    for reason in unpriced_reasons:
        c[f"q_unpriced_{reason}"] = np.where(
            c["unpriced_reason"].eq(reason), c["q_ref_t_per_year"], 0.0
        )

    rows = []
    for region_no, region in IMAGE26_REGIONS:
        for gid in range(1, 17):
            sub = c[(c["image_region"] == region) & (c["image_crop_id"] == gid)]
            total_q = float(sub["q_ref_t_per_year"].sum())
            priced_q = float(sub["priced_q"].sum())
            main_only_q = float(sub["main_only_priced_q"].sum())
            fallback_dep_q = float(sub["fallback_dependent_priced_q"].sum())
            fallback_only_q = float(sub["fallback_only_priced_q"].sum())
            fallback_obs_weight_q = float(sub["fallback_obs_weight_q"].sum())
            value = float(sub["value_weight"].sum())

            if total_q <= 0:
                raw_price, coverage, quality = np.nan, np.nan, "no_production"
            elif priced_q <= 0:
                raw_price, coverage, quality = np.nan, 0.0, "no_price_coverage"
            else:
                raw_price = value / priced_q
                coverage = priced_q / total_q
                if coverage >= 0.90:
                    quality = "high"
                elif coverage >= 0.75:
                    quality = "good"
                elif coverage >= PREFERRED_REGIONAL_MODEL_COVERAGE:
                    quality = "usable"
                elif coverage >= MIN_REGIONAL_MODEL_COVERAGE:
                    quality = "relaxed_usable"
                else:
                    quality = "low"

            fallback_share_priced = fallback_dep_q / priced_q if priced_q > 0 else np.nan
            fallback_only_share_priced = fallback_only_q / priced_q if priced_q > 0 else np.nan
            fallback_obs_weighted_share = (
                fallback_obs_weight_q / priced_q if priced_q > 0 else np.nan
            )

            if total_q <= 0:
                fallback_dependence = "not_applicable"
            elif priced_q <= 0:
                fallback_dependence = "no_price_coverage"
            elif fallback_dep_q <= 0:
                fallback_dependence = "none"
            elif fallback_share_priced <= 0.25:
                fallback_dependence = "low"
            elif fallback_share_priced <= 0.50:
                fallback_dependence = "moderate"
            elif fallback_share_priced < 1.0 - 1e-12:
                fallback_dependence = "high"
            else:
                fallback_dependence = "all_priced_estimates_use_fallback"

            crop = IMAGE_GROUPS[gid]
            row = {
                "image_region_number": region_no,
                "image_region_index0": region_no - 1,
                "image_region": region,
                "image_crop_id": gid,
                "image_crop": crop["image_crop"],
                "magnet_crop_group": crop["magnet_crop_group"],
                "regional_price_2010usd_t": raw_price,
                "production_coverage": coverage,
                "main_only_price_coverage": (
                    main_only_q / total_q if total_q > 0 else np.nan
                ),
                "fallback_dependent_price_coverage": (
                    fallback_dep_q / total_q if total_q > 0 else np.nan
                ),
                "fallback_only_price_coverage": (
                    fallback_only_q / total_q if total_q > 0 else np.nan
                ),
                "fallback_dependent_share_of_priced_production": fallback_share_priced,
                "fallback_only_share_of_priced_production": fallback_only_share_priced,
                "fallback_observation_weighted_share": fallback_obs_weighted_share,
                "q_total_ref_t_per_year": total_q,
                "q_priced_ref_t_per_year": priced_q,
                "q_main_only_priced_ref_t_per_year": main_only_q,
                "q_fallback_dependent_priced_ref_t_per_year": fallback_dep_q,
                "q_fallback_only_priced_ref_t_per_year": fallback_only_q,
                "q_unpriced_ref_t_per_year": max(total_q - priced_q, 0.0),
                "n_country_item_contributors": int(
                    sub["valid_for_regional_aggregation"].sum()
                ),
                "n_country_item_total": int(len(sub)),
                "coverage_quality": quality,
                "regional_quality": quality,
                "fallback_dependence": fallback_dependence,
            }
            for reason in unpriced_reasons:
                row[f"q_unpriced_{reason}_t_per_year"] = float(
                    sub[f"q_unpriced_{reason}"].sum()
                )
            rows.append(row)

    return pd.DataFrame(rows), c



def add_regional_plausibility_and_model_gate(regional: pd.DataFrame) -> pd.DataFrame:
    """Add crop-wise plausibility diagnostics and two-tier production coverage gating.

    A 50% production-coverage threshold remains the preferred standard.  For
    residual gaps, a regional estimate may be used down to 33% coverage, provided
    a price can actually be calculated and the catastrophic plausibility fail-safe
    is passed.  The output explicitly records which threshold admitted each price.
    """
    out = regional.copy()
    out["regional_crop_median_price_2010usd_t"] = np.nan
    out["regional_price_ratio_to_crop_median"] = np.nan
    out["regional_price_plausibility_flag"] = False
    out["regional_price_catastrophic_flag"] = False

    for gid, idx in out[out["regional_price_2010usd_t"].notna()].groupby("image_crop_id").groups.items():
        candidates = out.loc[idx]
        strong = candidates[
            candidates["production_coverage"].fillna(0)
            >= PREFERRED_REGIONAL_MODEL_COVERAGE
        ]
        bench = strong if len(strong) >= REGIONAL_BENCHMARK_MIN_REGIONS else candidates
        med = float(bench["regional_price_2010usd_t"].median()) if len(bench) else np.nan
        if not np.isfinite(med) or med <= 0:
            continue

        ratio = out.loc[idx, "regional_price_2010usd_t"] / med
        out.loc[idx, "regional_crop_median_price_2010usd_t"] = med
        out.loc[idx, "regional_price_ratio_to_crop_median"] = ratio
        out.loc[idx, "regional_price_plausibility_flag"] = (
            (ratio >= REGIONAL_RATIO_FLAG) | (ratio <= 1.0 / REGIONAL_RATIO_FLAG)
        )
        out.loc[idx, "regional_price_catastrophic_flag"] = (
            (ratio >= REGIONAL_RATIO_CATASTROPHIC)
            | (ratio <= 1.0 / REGIONAL_RATIO_CATASTROPHIC)
        )

    coverage = out["production_coverage"].fillna(0)
    out["passes_preferred_production_coverage_gate"] = (
        coverage >= PREFERRED_REGIONAL_MODEL_COVERAGE
    )
    out["passes_relaxed_production_coverage_gate"] = (
        coverage >= MIN_REGIONAL_MODEL_COVERAGE
    )
    # Backward-compatible column name now represents the active model gate.
    out["passes_production_coverage_gate"] = (
        out["passes_relaxed_production_coverage_gate"]
    )
    out["passes_plausibility_gate"] = ~out["regional_price_catastrophic_flag"]

    basis = np.full(len(out), "below_relaxed_33pct", dtype=object)
    basis[out["coverage_quality"].eq("no_production").to_numpy()] = "no_production"
    basis[out["coverage_quality"].eq("no_price_coverage").to_numpy()] = "no_price_coverage"
    preferred = out["passes_preferred_production_coverage_gate"].to_numpy()
    relaxed_only = (
        out["passes_relaxed_production_coverage_gate"]
        & ~out["passes_preferred_production_coverage_gate"]
    ).to_numpy()
    basis[preferred] = "preferred_50pct"
    basis[relaxed_only] = "residual_relaxation_33pct"
    out["coverage_gate_basis"] = basis

    out["model_ready"] = (
        out["regional_price_2010usd_t"].notna()
        & out["passes_relaxed_production_coverage_gate"]
        & out["passes_plausibility_gate"]
    )
    out["model_price_2010usd_t"] = np.where(
        out["model_ready"], out["regional_price_2010usd_t"], np.nan
    )

    statuses = []
    for row in out.itertuples():
        if row.coverage_quality == "no_production":
            statuses.append("no_production")
            continue
        problems = []
        if row.coverage_quality == "no_price_coverage":
            problems.append("no_price_coverage")
        elif not row.passes_relaxed_production_coverage_gate:
            problems.append("insufficient_production_coverage_below_33pct")
        if not row.passes_plausibility_gate:
            problems.append("catastrophic_regional_price_plausibility_failure")
        if problems:
            statuses.append("|".join(problems))
        elif row.coverage_gate_basis == "residual_relaxation_33pct":
            statuses.append("model_ready_relaxed_33pct_coverage")
        else:
            statuses.append("model_ready")
    out["model_status"] = statuses
    return out


def add_regional_crop_class_imputation(regional: pd.DataFrame) -> pd.DataFrame:
    """Fill only missing final region x crop-class prices.

    The method is deliberately simple and operates only after all data-derived
    regional prices and model gates have been calculated.

    1. For each IMAGE crop class c, calculate a benchmark B[c] as the median of
       existing data-derived ``model_price_2010usd_t`` values from regions that
       pass the preferred >=50% production-coverage gate. If fewer than
       ``IMPUTATION_MIN_DONORS`` preferred donors exist, use all existing
       data-derived model prices for that crop class instead.

    2. For each IMAGE region r, calculate a regional relative price factor S[r]
       as the median of ``model_price_2010usd_t / B[c]`` across the region's
       already-populated crop classes. Prefer >=50% coverage donor cells when at
       least ``IMPUTATION_MIN_DONORS`` are available; otherwise use all existing
       data-derived model prices in the region.

    3. For a missing region x crop-class cell only:

           final_price_2010usd_t[r,c] = B[c] * S[r]

    Imputed values are never donors: all benchmarks and regional factors are
    calculated from a frozen snapshot of the pre-imputation data-derived model
    prices. Existing populated model prices are copied unchanged into the final
    price column and verified unchanged before return.

    The same rule is used for missing cells with historical production gaps and
    for structural ``no_production`` cells so RegionalWide16 is complete. These
    cases remain separately tagged in ``imputation_target_class``.
    """
    out = regional.copy()
    if "model_price_2010usd_t" not in out.columns:
        raise ValueError("Regional table is missing model_price_2010usd_t")

    observed = pd.to_numeric(out["model_price_2010usd_t"], errors="coerce")
    observed_valid = observed.notna() & np.isfinite(observed) & (observed > 0)

    # Preserve the exact pre-imputation values. This column is intentionally
    # redundant with model_price_2010usd_t for auditability.
    out["data_derived_model_price_2010usd_t"] = observed
    out["final_price_2010usd_t"] = observed
    out["price_is_imputed"] = False
    out["final_price_source"] = np.where(
        observed_valid, "data_derived_regional_price", ""
    )
    out["final_price_quality"] = np.where(
        observed_valid, out["regional_quality"].fillna(""), ""
    )
    out["imputation_method"] = ""
    out["imputation_reason"] = ""
    out["imputation_target_class"] = ""
    out["imputation_crop_benchmark_2010usd_t"] = np.nan
    out["imputation_crop_benchmark_n_regions"] = 0
    out["imputation_crop_benchmark_basis"] = ""
    out["imputation_region_price_factor"] = np.nan
    out["imputation_region_factor_n_crops"] = 0
    out["imputation_region_factor_basis"] = ""

    # -------------------------------------------------------------------------
    # Crop-class benchmark B[c]. Donors are DATA-DERIVED values only.
    # -------------------------------------------------------------------------
    crop_benchmark: Dict[int, float] = {}
    crop_n: Dict[int, int] = {}
    crop_basis: Dict[int, str] = {}

    for gid in range(1, 17):
        crop_mask = out["image_crop_id"].eq(gid) & observed_valid
        preferred_mask = crop_mask & out[
            "passes_preferred_production_coverage_gate"
        ].fillna(False)

        if int(preferred_mask.sum()) >= IMPUTATION_MIN_DONORS:
            donor_mask = preferred_mask
            basis = "preferred_ge50pct_data_derived"
        else:
            donor_mask = crop_mask
            basis = "all_data_derived_fallback"

        vals = observed[donor_mask]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            crop_benchmark[gid] = float(vals.median())
            crop_n[gid] = int(len(vals))
            crop_basis[gid] = basis

    out["imputation_crop_benchmark_2010usd_t"] = out["image_crop_id"].map(
        crop_benchmark
    )
    out["imputation_crop_benchmark_n_regions"] = (
        out["image_crop_id"].map(crop_n).fillna(0).astype(int)
    )
    out["imputation_crop_benchmark_basis"] = (
        out["image_crop_id"].map(crop_basis).fillna("")
    )

    # -------------------------------------------------------------------------
    # Regional factor S[r]. Again, use only the frozen data-derived snapshot.
    # -------------------------------------------------------------------------
    bench = out["imputation_crop_benchmark_2010usd_t"]
    comparable = observed_valid & bench.notna() & np.isfinite(bench) & (bench > 0)
    ratios = pd.Series(np.nan, index=out.index, dtype=float)
    ratios.loc[comparable] = observed.loc[comparable] / bench.loc[comparable]

    region_factor: Dict[str, float] = {}
    region_n: Dict[str, int] = {}
    region_basis: Dict[str, str] = {}

    for _, region in IMAGE26_REGIONS:
        region_mask = out["image_region"].eq(region) & comparable
        preferred_mask = region_mask & out[
            "passes_preferred_production_coverage_gate"
        ].fillna(False)

        if int(preferred_mask.sum()) >= IMPUTATION_MIN_DONORS:
            donor_mask = preferred_mask
            basis = "preferred_ge50pct_data_derived"
        else:
            donor_mask = region_mask
            basis = "all_data_derived_fallback"

        vals = ratios[donor_mask]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            region_factor[region] = float(vals.median())
            region_n[region] = int(len(vals))
            region_basis[region] = basis
        else:
            # Defensive fallback only. With the IMAGE26 dataset each region should
            # have at least one data-derived crop price, but 1.0 is the transparent
            # neutral factor if a future sparse input does not.
            region_factor[region] = 1.0
            region_n[region] = 0
            region_basis[region] = "neutral_1.0_no_region_donors"

    out["imputation_region_price_factor"] = out["image_region"].map(region_factor)
    out["imputation_region_factor_n_crops"] = (
        out["image_region"].map(region_n).fillna(0).astype(int)
    )
    out["imputation_region_factor_basis"] = (
        out["image_region"].map(region_basis).fillna("")
    )

    # -------------------------------------------------------------------------
    # Fill missing values ONLY. No blending and no adjustment of populated cells.
    # -------------------------------------------------------------------------
    target = ~observed_valid
    candidate = (
        out["imputation_crop_benchmark_2010usd_t"]
        * out["imputation_region_price_factor"]
    )
    can_impute = target & candidate.notna() & np.isfinite(candidate) & (candidate > 0)

    out.loc[can_impute, "final_price_2010usd_t"] = candidate.loc[can_impute]
    out.loc[can_impute, "price_is_imputed"] = True
    out.loc[can_impute, "final_price_source"] = "regional_crop_class_imputation"
    out.loc[can_impute, "final_price_quality"] = "imputed_region_crop_class"
    out.loc[can_impute, "imputation_method"] = (
        "crop_class_median_x_region_relative_price_factor"
    )
    out.loc[can_impute, "imputation_reason"] = out.loc[
        can_impute, "model_status"
    ].fillna("")
    out.loc[
        can_impute & out["coverage_quality"].eq("no_production"),
        "imputation_target_class",
    ] = "no_reference_production"
    out.loc[
        can_impute & ~out["coverage_quality"].eq("no_production"),
        "imputation_target_class",
    ] = "residual_data_gap"

    final = pd.to_numeric(out["final_price_2010usd_t"], errors="coerce")
    out["final_price_ready"] = final.notna() & np.isfinite(final) & (final > 0)
    out["final_price_status"] = out["model_status"].astype(str)
    out.loc[can_impute, "final_price_status"] = np.where(
        out.loc[can_impute, "imputation_target_class"].eq("no_reference_production"),
        "imputed_region_crop_class_no_reference_production",
        "imputed_region_crop_class_residual_data_gap",
    )
    out.loc[target & ~can_impute, "final_price_status"] = (
        "imputation_unavailable_missing_crop_benchmark_or_region_factor"
    )

    # Hard invariant requested for v5: every populated pre-imputation value must
    # remain numerically identical in the final price column.
    if not np.allclose(
        out.loc[observed_valid, "final_price_2010usd_t"].to_numpy(float),
        observed.loc[observed_valid].to_numpy(float),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    ):
        raise AssertionError(
            "Regional imputation altered an already-populated model price."
        )

    return out

def build_gap_priority(regional: pd.DataFrame) -> pd.DataFrame:
    """Rank residual non-model-ready region-crop combinations by unpriced production."""
    g = regional[
        (regional["q_total_ref_t_per_year"] > 0) & (~regional["model_ready"])
    ].copy()
    if g.empty:
        return g

    g["unpriced_share"] = np.where(
        g["q_total_ref_t_per_year"] > 0,
        g["q_unpriced_ref_t_per_year"] / g["q_total_ref_t_per_year"],
        np.nan,
    )
    g["gap_priority_t_per_year"] = g["q_unpriced_ref_t_per_year"]
    g = g.sort_values(
        ["gap_priority_t_per_year", "q_total_ref_t_per_year"],
        ascending=[False, False],
    ).reset_index(drop=True)
    g["gap_priority_rank"] = np.arange(1, len(g) + 1)

    def action(row) -> str:
        if getattr(row, "regional_price_catastrophic_flag", False):
            return "audit surviving country-item price scale/outliers before any imputation"

        q_un = row.q_unpriced_ref_t_per_year
        if q_un <= 0:
            return "no missing-production gap; inspect model-readiness/plausibility diagnostics"

        components = {
            "macro": row.q_unpriced_macro_conversion_failure_t_per_year,
            "outlier": row.q_unpriced_excluded_price_outlier_t_per_year,
            "no_pp": row.q_unpriced_no_producer_price_data_t_per_year,
            "few_obs": row.q_unpriced_insufficient_price_observations_t_per_year,
            "other": row.q_unpriced_other_unpriced_reason_t_per_year,
        }
        dominant = max(components, key=components.get)
        share = components[dominant] / q_un if q_un > 0 else 0.0

        if share >= 0.50:
            return {
                "macro": "repair macro/currency conversion first; do not impute a monetary-unit error",
                "outlier": "audit dominant excluded producers; prefer official/alternative price data",
                "no_pp": "target dominant missing producers; then use region-scaled item imputation only if needed",
                "few_obs": "seek additional same-country observations; otherwise use a transparent regional/peer fallback",
                "other": "inspect top unpriced contributors and source economically material prices",
            }[dominant]
        return "inspect top unpriced contributors; address the largest missing-production sources first"

    g["recommended_next_action"] = [action(r) for r in g.itertuples()]
    return g



def build_gap_contributors(
    contributions: pd.DataFrame, gap_priority: pd.DataFrame
) -> pd.DataFrame:
    """List country-item observations responsible for genuine residual unpriced production."""
    if gap_priority.empty:
        return pd.DataFrame()

    keys = gap_priority[
        ["image_region", "image_crop_id", "gap_priority_rank", "model_status"]
    ]
    x = contributions.merge(keys, on=["image_region", "image_crop_id"], how="inner")
    x = x[x["unpriced_reason"].ne("priced")].copy()
    x["gap_component"] = x["unpriced_reason"]
    x["gap_relevant_q_t_per_year"] = x["q_ref_t_per_year"]
    return x.sort_values(
        ["gap_priority_rank", "gap_relevant_q_t_per_year"],
        ascending=[True, False],
    )



def build_region_summary(regional: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region_no, region in IMAGE26_REGIONS:
        x = regional[regional["image_region"] == region]
        with_prod = x[x["coverage_quality"] != "no_production"]
        fallback_vals = with_prod[
            with_prod["fallback_dependent_share_of_priced_production"].notna()
        ]
        rows.append({
            "image_region_number": region_no,
            "image_region_index0": region_no - 1,
            "image_region": region,
            "crop_groups_with_production": int(len(with_prod)),
            "high_coverage": int((x["coverage_quality"] == "high").sum()),
            "good_coverage": int((x["coverage_quality"] == "good").sum()),
            "usable_coverage": int((x["coverage_quality"] == "usable").sum()),
            "relaxed_usable_coverage": int((x["coverage_quality"] == "relaxed_usable").sum()),
            "low_coverage": int((x["coverage_quality"] == "low").sum()),
            "no_price_coverage": int((x["coverage_quality"] == "no_price_coverage").sum()),
            "model_ready_prices": int(x["model_ready"].sum()),
            "imputed_prices": int(x.get("price_is_imputed", False).sum()),
            "imputed_residual_data_gaps": int(
                (x.get("imputation_target_class", "") == "residual_data_gap").sum()
            ),
            "imputed_no_reference_production": int(
                (x.get("imputation_target_class", "") == "no_reference_production").sum()
            ),
            "final_price_ready": int(x.get("final_price_ready", x["model_ready"]).sum()),
            "fallback_none": int((x["fallback_dependence"] == "none").sum()),
            "fallback_low": int((x["fallback_dependence"] == "low").sum()),
            "fallback_moderate": int((x["fallback_dependence"] == "moderate").sum()),
            "fallback_high": int((x["fallback_dependence"] == "high").sum()),
            "fallback_all_priced": int(
                (x["fallback_dependence"] == "all_priced_estimates_use_fallback").sum()
            ),
            "regional_plausibility_flags": int(
                x["regional_price_plausibility_flag"].sum()
            ),
            "median_production_coverage": (
                float(with_prod["production_coverage"].median())
                if len(with_prod) else np.nan
            ),
            "median_fallback_share_of_priced_production": (
                float(fallback_vals["fallback_dependent_share_of_priced_production"].median())
                if len(fallback_vals) else 0.0
            ),
        })
    return pd.DataFrame(rows)



def metadata_table(
    input_files: pd.DataFrame,
    validation_messages: List[str],
    regional: pd.DataFrame,
    country_prices: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        ("purpose", "Regional IMAGE26 crop-price reference data for the agricultural land-rent spatial scaler."),
        ("currency", "constant 2010 USD per tonne"),
        ("preferred_price_period", f"{CENTRAL_START}-{CENTRAL_END}"),
        ("fallback_price_period", f"{SUPPORT_START}-{SUPPORT_END}"),
        ("production_weight_period", f"{CENTRAL_START}-{CENTRAL_END}"),
        ("production_2000_2009", "may be loaded if present but is not used in q_ref model weights"),
        ("base_year", str(BASE_YEAR)),
        ("default_minimum_country_item_price_observations", str(MIN_PRICE_OBS)),
        (
            "country_item_price_rule",
            "Default: median of all valid 2010-2019 observations when >=3 exist; otherwise add only the most recent valid 2000-2009 observations needed to reach 3. Explicit N2Exceptions use the same hierarchy with a minimum of 2.",
        ),
        (
            "n2_exception_count_allowed",
            str(int(country_prices.get("n2_exception_allowed", pd.Series(dtype=bool)).fillna(False).sum())),
        ),
        (
            "n2_exception_count_applied",
            str(int(country_prices.get("n2_exception_applied", pd.Series(dtype=bool)).fillna(False).sum())),
        ),
        ("fallback_model_gate", "none; fallback dependence is diagnostic only"),
        (
            "regional_aggregation",
            "production-weighted mean of valid country-item prices using mean annual 2010-2019 production weights",
        ),
        ("regional_preferred_production_coverage", f"{PREFERRED_REGIONAL_MODEL_COVERAGE:.0%}"),
        ("regional_residual_relaxed_min_production_coverage", f"{MIN_REGIONAL_MODEL_COVERAGE:.0%}"),
        (
            "regional_coverage_gate_rule",
            "50% remains preferred. Remaining regional gaps may be admitted at 33-<50% coverage if a regional price exists and the catastrophic plausibility gate passes; coverage_gate_basis records the route.",
        ),
        (
            "regional_imputation_rule",
            "Only missing final regional prices are imputed. Crop benchmark = median of >=50% coverage data-derived regional prices for that crop (fallback: all data-derived regional prices if fewer than 3 donors). Region factor = median of observed regional price / crop benchmark across already-populated crop classes, preferring >=50% coverage donors when at least 3 exist. Imputed price = crop benchmark * region factor.",
        ),
        ("regional_imputation_min_donors", str(IMPUTATION_MIN_DONORS)),
        ("regional_imputation_uses_imputed_values_as_donors", "no"),
        ("regional_imputation_adjusts_existing_model_prices", "no"),
        (
            "regional_imputation_no_production_rule",
            "Missing no-production cells are filled with the same regional crop-class imputation so RegionalWide16 is complete; they are tagged no_reference_production.",
        ),
        ("regionalwide16_value_column", "final_price_2010usd_t"),
        ("fixed_exchange_rate", "2010 Standard Local Currency units per USD used for every price year"),
        ("deflator", "GDP Deflator, Value Standard Local Currency, 2015 prices; ratios D2010/Dy used"),
        (
            "sudan_former_macro_repair",
            "For FAOSTAT M49=736, missing GDP deflators are proxied year-by-year from current Sudan M49=729; the M49=736 2010 exchange rate is retained. Proxy source M49 is recorded in macro audit columns.",
        ),
        (
            "outlier_item_flag",
            f"same-item robust |z|>={OUTLIER_ROBUST_Z_FLAG} and/or price ratio outside {1/ITEM_RATIO_FLAG:g}-{ITEM_RATIO_FLAG:g}x",
        ),
        (
            "outlier_item_exclusion",
            f"ratio outside {1/ITEM_RATIO_CATASTROPHIC:g}-{ITEM_RATIO_CATASTROPHIC:g}x, or robust outlier with ratio outside {1/ITEM_RATIO_EXCLUDE:g}-{ITEM_RATIO_EXCLUDE:g}x",
        ),
        (
            "outlier_systemic_country_exclusion",
            f">={SYSTEMIC_MIN_COMPARABLE_ITEMS} comparable items, >= {SYSTEMIC_MIN_SHARE_EXTREME:.0%} beyond 10x, median country scale outside {1/SYSTEMIC_COUNTRY_RATIO_EXCLUDE:g}-{SYSTEMIC_COUNTRY_RATIO_EXCLUDE:g}x",
        ),
        (
            "regional_plausibility_fail_safe",
            f"flag outside {1/REGIONAL_RATIO_FLAG:g}-{REGIONAL_RATIO_FLAG:g}x crop median; exclude from model if outside {1/REGIONAL_RATIO_CATASTROPHIC:g}-{REGIONAL_RATIO_CATASTROPHIC:g}x",
        ),
        ("country_join", "FAOSTAT M49 -> IMAGE ISO numeric-3"),
        ("china_aggregate", "FAOSTAT M49=159 excluded to prevent double counting."),
        ("sudan_former", "FAOSTAT M49=736 mapped to EAF at regional scale."),
        ("greenland", "GRL excluded from IMAGE26 agricultural output."),
        ("model_ready_regional_prices", str(int(regional["model_ready"].sum()))),
        (
            "model_ready_via_relaxed_33pct_coverage",
            str(int(((regional["coverage_gate_basis"] == "residual_relaxation_33pct") & regional["model_ready"]).sum())),
        ),
        (
            "regional_prices_imputed",
            str(int(regional.get("price_is_imputed", pd.Series(dtype=bool)).fillna(False).sum())),
        ),
        (
            "regional_residual_data_gaps_imputed",
            str(int((regional.get("imputation_target_class", pd.Series(dtype=str)) == "residual_data_gap").sum())),
        ),
        (
            "regional_no_reference_production_cells_imputed",
            str(int((regional.get("imputation_target_class", pd.Series(dtype=str)) == "no_reference_production").sum())),
        ),
        (
            "final_regional_prices_ready",
            str(int(regional.get("final_price_ready", regional["model_ready"]).sum())),
        ),
        ("country_item_prices_initially_accepted", str(int(country_prices["accepted_price"].sum()))),
        ("country_item_prices_using_any_fallback", str(int(country_prices["uses_fallback"].fillna(False).sum()))),
        ("country_item_prices_fallback_only", str(int(country_prices["fallback_only"].fillna(False).sum()))),
        ("country_item_prices_excluded_outlier", str(int(country_prices["price_exclusion_flag"].sum()))),
        ("country_item_prices_used_regionally", str(int(country_prices["valid_for_regional_aggregation"].sum()))),
    ]
    for msg in validation_messages:
        rows.append(("validation_message", msg))
    return pd.DataFrame(rows, columns=["parameter", "value"])

def write_output(
    output_path: Path,
    regional: pd.DataFrame,
    regional_wide: pd.DataFrame,
    region_summary: pd.DataFrame,
    ngfbfc38: pd.DataFrame,
    country_prices: pd.DataFrame,
    country_scale_diag: pd.DataFrame,
    contributions: pd.DataFrame,
    gap_priority: pd.DataFrame,
    gap_contributors: pd.DataFrame,
    fallback_priority: pd.DataFrame,
    crosswalk: pd.DataFrame,
    ambiguous_summary: pd.DataFrame,
    pp_real_raw: pd.DataFrame,
    macro_country_summary: pd.DataFrame,
    country_item_macro_audit: pd.DataFrame,
    macro_failure_rows: pd.DataFrame,
    macro_failure_summary: pd.DataFrame,
    macro_impact: pd.DataFrame,
    geo: pd.DataFrame,
    geo_diag: pd.DataFrame,
    input_files: pd.DataFrame,
    metadata: pd.DataFrame,
    write_pp_audit: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    outliers = country_prices[country_prices["price_outlier_flag"]].copy()
    excluded = country_prices[country_prices["price_exclusion_flag"]].copy()
    unused_pp = sorted(set(pp_real_raw["Item"].dropna()) - set(crosswalk["faostat_item"]))
    unused_pp_df = pd.DataFrame({
        "producer_price_item_not_in_production_crosswalk": unused_pp
    })

    pp_audit_cols = [c for c in [
        "m49", "Area", "image_region", "Item", "Year", "Value", "macro_period",
        "deflator", "deflator_source_m49", "deflator_base", "deflator_base_source_m49",
        "xr_base", "xr_base_source_m49", "uses_deflator_proxy", "macro_proxy_note",
        "price_2010usd_t", "macro_ok", "macro_failure_reason", "Flag", "Flag Description", "_source_file"
    ] if c in pp_real_raw.columns]
    pp_audit = pp_real_raw[
        pp_real_raw["Year"].between(SUPPORT_START, CENTRAL_END)
    ][pp_audit_cols].copy()
    imputation_diag = regional[
        regional.get("price_is_imputed", pd.Series(False, index=regional.index)).fillna(False)
    ].copy()

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        regional.to_excel(writer, sheet_name="RegionalPrices", index=False)
        regional_wide.to_excel(writer, sheet_name="RegionalWide16", index=False)
        imputation_diag.to_excel(writer, sheet_name="ImputationDiagnostics", index=False)
        region_summary.to_excel(writer, sheet_name="RegionSummary", index=False)
        ngfbfc38.to_excel(writer, sheet_name="NGFBFC38", index=False)
        gap_priority.to_excel(writer, sheet_name="GapPriority", index=False)
        gap_contributors.to_excel(writer, sheet_name="GapContributors", index=False)
        fallback_priority.to_excel(writer, sheet_name="FallbackDependence", index=False)
        country_prices.to_excel(writer, sheet_name="CountryItemPrices", index=False)
        country_scale_diag.to_excel(writer, sheet_name="CountryScaleDiagnostics", index=False)
        contributions.to_excel(writer, sheet_name="Contributions", index=False)
        outliers.to_excel(writer, sheet_name="Outliers", index=False)
        excluded.to_excel(writer, sheet_name="OutliersExcluded", index=False)
        country_item_macro_audit.to_excel(writer, sheet_name="CountryItemMacroAudit", index=False)
        macro_country_summary.to_excel(writer, sheet_name="MacroCountrySummary", index=False)
        macro_failure_summary.to_excel(writer, sheet_name="MacroFailureSummary", index=False)
        macro_impact.to_excel(writer, sheet_name="MacroImpactRegionCrop", index=False)
        macro_failure_rows.to_excel(writer, sheet_name="MacroFailureRows", index=False)
        crosswalk.to_excel(writer, sheet_name="ItemCrosswalk", index=False)
        ambiguous_summary.to_excel(writer, sheet_name="AmbiguousProduction", index=False)
        unused_pp_df.to_excel(writer, sheet_name="UnusedPPItems", index=False)
        geo.to_excel(writer, sheet_name="GeoMapping", index=False)
        geo_diag.to_excel(writer, sheet_name="GeoDiagnostics", index=False)
        input_files.to_excel(writer, sheet_name="InputFiles", index=False)
        metadata.to_excel(writer, sheet_name="Metadata", index=False)
        if write_pp_audit:
            pp_audit.to_excel(writer, sheet_name="PPConvertedAudit", index=False)


def resolve_mapping_file(data_dir: Path, mapping_arg: Optional[str]) -> Path:
    if mapping_arg:
        p = Path(mapping_arg)
        if not p.is_absolute():
            p = data_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Mapping file not found: {p}")
        return p

    direct = data_dir / DEFAULT_MAPPING_FILENAME
    if direct.exists():
        return direct
    candidates = []
    for pattern in ("*country*region*.tsv", "*country*region*.txt", "*country*region*.csv",
                    "*country*region*.xlsx", "*region*mapping*.tsv", "*region*mapping*.txt"):
        candidates.extend(data_dir.glob(pattern))
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No IMAGE country-region mapping found in {data_dir}. "
            f"Save the supplied mapping as {DEFAULT_MAPPING_FILENAME} or pass --mapping-file."
        )
    raise ValueError(
        "Multiple possible country-region mapping files found; pass --mapping-file explicitly:\n"
        + "\n".join(str(p) for p in candidates)
    )



def resolve_supplemental_file(
    data_dir: Path, supplemental_arg: Optional[str]
) -> Optional[Path]:
    """Resolve the optional supplemental workbook.

    If --supplemental-file is omitted, the default filename is used when present;
    otherwise the module runs without supplemental rows/N=2 rules.
    """
    if supplemental_arg:
        p = Path(supplemental_arg)
        if not p.is_absolute():
            p = data_dir / p
        if not p.exists():
            raise FileNotFoundError(f"Supplemental input file not found: {p}")
        return p

    default = data_dir / DEFAULT_SUPPLEMENTAL_FILENAME
    return default if default.exists() else None


def run(
    data_dir: Path,
    mapping_file: Path,
    output_file: str = DEFAULT_OUTPUT_FILENAME,
    supplemental_file: Optional[Path] = None,
    allow_partial: bool = False,
    write_pp_audit: bool = False,
) -> Path:
    (
        pp, prod, defl, xr, geo, input_files, geo_diag, messages, n2_exceptions
    ) = prepare_inputs(
        data_dir,
        mapping_file,
        supplemental_file=supplemental_file,
        allow_partial=allow_partial,
    )

    crosswalk = build_item_crosswalk(prod["Item"].unique(), pp["Item"].unique())
    unmapped = crosswalk[crosswalk["mapping_status"] == "unmapped"]
    if not unmapped.empty:
        warnings.warn(
            f"{len(unmapped)} production item(s) are unmapped. Review ItemCrosswalk sheet."
        )

    pp_real_raw, macro_country_summary = convert_pp_to_real_2010usd(pp, defl, xr)
    pp_real = collapse_duplicate_pp(pp_real_raw)

    country_prices = estimate_country_item_prices(
        pp_real, n2_exceptions=n2_exceptions
    )
    country_prices, country_scale_diag = add_outlier_diagnostics(country_prices)

    # Core production weights remain 2010-2019. Any 2000-2009 production chunks
    # present in the folder are loaded but intentionally filtered out here.
    q, ambiguous = representative_production(prod, crosswalk)

    country_item_macro_audit = build_country_item_macro_audit(
        pp_real_raw, q, n2_exceptions=n2_exceptions
    )
    macro_failure_rows, macro_failure_summary, macro_impact = build_macro_failure_outputs(
        pp_real_raw, macro_country_summary, country_item_macro_audit
    )

    regional, contributions = aggregate_regional_prices(
        q, country_prices, country_item_macro_audit
    )
    regional = add_regional_plausibility_and_model_gate(regional)

    # Preserve residual gap diagnostics BEFORE imputation. GapPriority and
    # GapContributors therefore continue to describe the underlying data gaps,
    # rather than disappearing merely because a final model backstop was applied.
    gap_priority = build_gap_priority(regional)
    gap_contributors = build_gap_contributors(contributions, gap_priority)
    fallback_priority = build_fallback_dependence_priority(regional)

    # Final backstop: fill only missing regional model values. Existing populated
    # model_price_2010usd_t values remain untouched and are copied exactly into
    # final_price_2010usd_t.
    regional = add_regional_crop_class_imputation(regional)

    regional_wide = build_region_wide(regional)
    region_summary = build_region_summary(regional)
    ngfbfc38 = build_ngfbfc38(regional)
    ambiguous_summary = ambiguous_production_summary(ambiguous)
    metadata = metadata_table(input_files, messages, regional, country_prices)

    output_path = data_dir / output_file
    write_output(
        output_path=output_path,
        regional=regional,
        regional_wide=regional_wide,
        region_summary=region_summary,
        ngfbfc38=ngfbfc38,
        country_prices=country_prices,
        country_scale_diag=country_scale_diag,
        contributions=contributions,
        gap_priority=gap_priority,
        gap_contributors=gap_contributors,
        fallback_priority=fallback_priority,
        crosswalk=crosswalk,
        ambiguous_summary=ambiguous_summary,
        pp_real_raw=pp_real_raw,
        macro_country_summary=macro_country_summary,
        country_item_macro_audit=country_item_macro_audit,
        macro_failure_rows=macro_failure_rows,
        macro_failure_summary=macro_failure_summary,
        macro_impact=macro_impact,
        geo=geo,
        geo_diag=geo_diag,
        input_files=input_files,
        metadata=metadata,
        write_pp_audit=write_pp_audit,
    )

    print("FAOSTAT -> IMAGE26 regional crop-price preprocessing complete (supplemental/N2/Sudan/33%/regional-imputation v5)")
    print(f"Output: {output_path}")
    if supplemental_file is not None:
        print(f"Supplemental input: {supplemental_file}")
        print(f"N=2 exception rules loaded: {len(n2_exceptions)}")
    else:
        print("Supplemental input: none")
    print(f"Regional crop cells: {len(regional)} (= 26 regions x 16 crops)")
    print(f"Data-derived model-ready regional prices: {int(regional['model_ready'].sum())}")
    print(f"Region x crop-class prices imputed: {int(regional['price_is_imputed'].sum())}")
    print(f"Final regional prices ready: {int(regional['final_price_ready'].sum())} / {len(regional)}")
    final_missing = int((~regional["final_price_ready"]).sum())
    print(f"Final RegionalWide16 values still missing: {final_missing}")
    relaxed_ready = int(
        ((regional["coverage_gate_basis"] == "residual_relaxation_33pct")
         & regional["model_ready"]).sum()
    )
    print(f"Model-ready prices admitted via 33% residual relaxation: {relaxed_ready}")
    print("Coverage quality:")
    print(regional["coverage_quality"].value_counts(dropna=False).to_string())
    print("Fallback dependence:")
    print(regional["fallback_dependence"].value_counts(dropna=False).to_string())

    print(f"Initially accepted country-item prices: {int(country_prices['accepted_price'].sum())}")
    print(
        "Country-item estimates accepted using N=2 exception: "
        f"{int(country_prices['n2_exception_applied'].fillna(False).sum())}"
    )
    print(f"Country-item estimates using any fallback: {int(country_prices['uses_fallback'].fillna(False).sum())}")
    print(f"Country-item estimates using fallback only: {int(country_prices['fallback_only'].fillna(False).sum())}")
    print(f"Excluded country-item price outliers: {int(country_prices['price_exclusion_flag'].sum())}")
    print(f"Country-item prices used regionally: {int(country_prices['valid_for_regional_aggregation'].sum())}")

    systemic = (
        country_scale_diag[
            country_scale_diag.get("systemic_country_scale_exclusion", False) == True
        ]
        if not country_scale_diag.empty else pd.DataFrame()
    )
    print(f"Systemic country price-scale exclusions: {len(systemic)}")
    if len(systemic):
        print(
            systemic[[
                "m49", "Area", "image_region",
                "country_median_ratio_to_item_median", "n_comparable_items"
            ]].to_string(index=False)
        )

    failed = (
        (~pp_real_raw["macro_ok"])
        & pp_real_raw["macro_period"].isin(["main_2010_2019", "fallback_2000_2009"])
    )
    print(f"PP rows failing macro conversion in processing window: {int(failed.sum())}")
    for period in ["main_2010_2019", "fallback_2000_2009"]:
        n = int(
            ((~pp_real_raw["macro_ok"]) & (pp_real_raw["macro_period"] == period)).sum()
        )
        print(f"  {period}: {n}")

    sudan_proxy_rows = int(
        (
            (pp_real_raw["m49"] == 736)
            & pp_real_raw.get("uses_deflator_proxy", False)
        ).sum()
    )
    print(f"Sudan (former) PP rows using current-Sudan deflator proxy: {sudan_proxy_rows}")

    print(f"Unmapped production items: {(crosswalk['mapping_status'] == 'unmapped').sum()}")
    print(f"Ambiguous excluded production items: {(crosswalk['mapping_status'] == 'ambiguous_excluded').sum()}")

    if not gap_priority.empty:
        print("Top residual regional gaps by genuinely unpriced production:")
        cols = [
            "gap_priority_rank", "image_region", "image_crop",
            "gap_priority_t_per_year", "production_coverage", "model_status"
        ]
        print(gap_priority[cols].head(10).to_string(index=False))

    if not fallback_priority.empty:
        print("Largest model/diagnostic exposure to fallback-assisted price estimates:")
        cols = [
            "fallback_priority_rank", "image_region", "image_crop",
            "q_fallback_dependent_priced_ref_t_per_year",
            "fallback_dependent_share_of_priced_production",
            "production_coverage",
        ]
        print(fallback_priority[cols].head(10).to_string(index=False))

    return output_path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Directory containing FAOSTAT chunk files."
    )
    parser.add_argument(
        "--mapping-file", type=str, default=None,
        help=f"IMAGE country-region mapping. Default/search name: {DEFAULT_MAPPING_FILENAME}"
    )
    parser.add_argument(
        "--supplemental-file", type=str, default=None,
        help=(
            f"Optional supplemental workbook. If omitted, "
            f"{DEFAULT_SUPPLEMENTAL_FILENAME} is used automatically when present in data-dir."
        )
    )
    parser.add_argument(
        "--output-file", type=str, default=DEFAULT_OUTPUT_FILENAME,
        help="Output .xlsx filename."
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Allow missing central reference years (useful only for testing)."
    )
    parser.add_argument(
        "--write-pp-audit", action="store_true",
        help="Also write all converted PP observations for 2000-2019; workbook can become large."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mapping = resolve_mapping_file(args.data_dir, args.mapping_file)
    supplemental = resolve_supplemental_file(
        args.data_dir, args.supplemental_file
    )
    run(
        data_dir=args.data_dir,
        mapping_file=mapping,
        output_file=args.output_file,
        supplemental_file=supplemental,
        allow_partial=args.allow_partial,
        write_pp_audit=args.write_pp_audit,
    )
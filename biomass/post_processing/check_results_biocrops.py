"""Thorough diagnostics for outputs of the standalone IMAGE/TIMER bioenergy-crop module.

This script is intentionally independent of the BioCrop model itself: it reads the
NetCDF output files written by biocrops.py, checks internal accounting identities,
prints diagnostics, writes CSV reports, and produces spatial maps and distributions.

Typical use (from the timer-prism environment):

    python review_biocrops_outputs.py \
        --input-dir X:\\path\\to\\biocrops_outputs \
        --report-dir X:\\path\\to\\biocrops_review

Recommended, if GAREA.NC and the IMAGE grid coordinate file are not in their default
locations:

    python review_biocrops_outputs.py \
        --input-dir X:\\path\\to\\biocrops_outputs \
        --report-dir X:\\path\\to\\biocrops_review \
        --garea X:\\user\\orhalem\\TIMER_CDR\\timer-prism\\timer\\data\\scen\\biomass\\SSP2_M\\GAREA.NC \
        --grid-coordinates X:\\user\\orhalem\\TIMER_CDR\\timer-prism\\timer\\data\\sa\\biomass\\IMAGE_grid_5min.nc \
        --years 2030 2050 2100

What is checked
---------------
1. File/variable/dimension consistency.
2. BioPotFrac and BioPotFracByLand bounds and land-source accounting.
3. Non-negative yield/productivity/potential and support consistency.
4. BiomassPotential = BiomassProductivity * eligible area (when GAREA is available).
5. BioAnnualRevenue = BiomassProductivity * max(BioFeedstockValue, 0) at the final
   simulation year (because BiomassProductivity is currently exported as a final-state
   static field).
6. BioNPV = area-weighted BioNPVByLand at the final simulation year (same reason).
7. NPV availability/NaN support, negative/positive NPV shares, annual margins, and
   best-crop selection.
8. Optional check that TIMER BioFeedstockValue is spatially constant within IMAGE
   regions, if timer.cdr.cdr_util is importable.
9. Spatial maps of physical potential, economics, best crop, and source-land NPV.

Important limitation of the current BioCrop output design
----------------------------------------------------------
BioPotFrac, BioPotFracByLand, BiomassProductivity, and BiomassPotential are currently
written as static arrays at the end of the simulation. They therefore represent the
FINAL simulation state only. Time-varying BioFeedstockValue, BioAnnualRevenue,
BioAnnualNonLandCost, BioNPV, and BioNPVByLand contain the full time series.
Accordingly, cross-file identities involving the static fields are only tested at the
final year. The script prints a warning about this explicitly.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_SSP2M_DIR = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\scen\biomass\SSP2_M"
)
DEFAULT_GRID_COORDINATES = Path(
    r"X:\user\orhalem\TIMER_CDR\timer-prism\timer\data\sa\biomass\IMAGE_grid_5min.nc"
)

EPS = 1.0e-8
ABS_TOL = 1.0e-5
REL_TOL = 1.0e-5

CROP_NAMES = {
    0: "Woody",
    1: "Maize",
    2: "Sugar cane",
    3: "Oil crops",
    4: "Non-woody",
}

LAND_NAMES = {
    0: "Cropland",
    1: "Pasture",
    2: "Forest",
    3: "Other land",
    4: "Urban",
}

EXPECTED = {
    "BioPotFrac": "BioPotFrac.nc",
    "BioPotFracByLand": "BioPotFracByLand.nc",
    "BiomassYield": "BiomassYield.nc",
    "BiomassProductivity": "BiomassProductivity.nc",
    "BiomassPotential": "BiomassPotential.nc",
    "BioFeedstockValue": "BioFeedstockValue.nc",
    "BioAnnualRevenue": "BioAnnualRevenue.nc",
    "BioAnnualNonLandCost": "BioAnnualNonLandCost.nc",
    "BioNPV": "BioNPV.nc",
    "BioNPVByLand": "BioNPVByLand.nc",
}

TIME_VARYING = {
    "BiomassYield",
    "BioFeedstockValue",
    "BioAnnualRevenue",
    "BioAnnualNonLandCost",
    "BioNPV",
    "BioNPVByLand",
}

STATIC_FINAL_STATE = {
    "BioPotFrac",
    "BioPotFracByLand",
    "BiomassProductivity",
    "BiomassPotential",
}


# =============================================================================
# Helpers
# =============================================================================


def _label_crop(value: int | float | str) -> str:
    try:
        return CROP_NAMES.get(int(value), f"Crop {value}")
    except (TypeError, ValueError):
        return str(value)


def _label_land(value: int | float | str) -> str:
    try:
        return LAND_NAMES.get(int(value), f"Land {value}")
    except (TypeError, ValueError):
        return str(value)


def _safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text).strip("_")


def _finite_stats(values: np.ndarray) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(a)
    n = a.size
    nf = int(finite.sum())
    out = {
        "n": int(n),
        "n_finite": nf,
        "finite_pct": 100.0 * nf / n if n else np.nan,
        "nan_pct": 100.0 * np.isnan(a).sum() / n if n else np.nan,
        "pos_pct": np.nan,
        "zero_pct": np.nan,
        "neg_pct": np.nan,
        "min": np.nan,
        "p01": np.nan,
        "p05": np.nan,
        "p25": np.nan,
        "median": np.nan,
        "mean": np.nan,
        "p75": np.nan,
        "p95": np.nan,
        "p99": np.nan,
        "max": np.nan,
    }
    if nf == 0:
        return out
    x = a[finite]
    out.update(
        {
            "pos_pct": 100.0 * np.sum(x > EPS) / nf,
            "zero_pct": 100.0 * np.sum(np.abs(x) <= EPS) / nf,
            "neg_pct": 100.0 * np.sum(x < -EPS) / nf,
            "min": float(np.min(x)),
            "mean": float(np.mean(x)),
            "max": float(np.max(x)),
        }
    )
    q = np.quantile(x, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    for key, val in zip(("p01", "p05", "p25", "median", "p75", "p95", "p99"), q):
        out[key] = float(val)
    return out


def _error_metrics(actual: np.ndarray, expected: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(e)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    n = int(valid.sum())
    if n == 0:
        return {
            "n_compared": 0,
            "max_abs_error": np.nan,
            "mean_abs_error": np.nan,
            "max_rel_error": np.nan,
            "mean_rel_error": np.nan,
            "fail_count": 0,
            "fail_pct": np.nan,
        }
    diff = np.abs(a[valid] - e[valid])
    denom = np.maximum(np.abs(e[valid]), 1.0e-12)
    rel = diff / denom
    fail = diff > (ABS_TOL + REL_TOL * np.abs(e[valid]))
    return {
        "n_compared": n,
        "max_abs_error": float(np.max(diff)),
        "mean_abs_error": float(np.mean(diff)),
        "max_rel_error": float(np.max(rel)),
        "mean_rel_error": float(np.mean(rel)),
        "fail_count": int(fail.sum()),
        "fail_pct": 100.0 * float(fail.sum()) / n,
    }


def _resolve_year_indices(available: Sequence[float], requested: Sequence[int] | None) -> list[int]:
    years = np.asarray(available, dtype=float)
    if years.size == 0:
        return []
    if requested:
        indices = []
        for year in requested:
            j = int(np.argmin(np.abs(years - year)))
            if j not in indices:
                indices.append(j)
        return indices
    # Good default review slices: first, early-middle, middle, late-middle, final.
    raw = [0, years.size // 4, years.size // 2, 3 * years.size // 4, years.size - 1]
    return sorted(set(raw))


def _time_values(da: xr.DataArray) -> np.ndarray:
    if "time" not in da.dims:
        return np.array([], dtype=float)
    if "time" in da.coords:
        vals = np.asarray(da["time"].values).reshape(-1)
        try:
            return vals.astype(float)
        except (TypeError, ValueError):
            pass
    return np.arange(da.sizes["time"], dtype=float)


def _extract_slice(
    da: xr.DataArray,
    *,
    time_index: int | None = None,
    crop_index: int | None = None,
    land_index: int | None = None,
) -> np.ndarray:
    x = da
    if time_index is not None and "time" in x.dims:
        x = x.isel(time=time_index)
    if crop_index is not None and "biocrop" in x.dims:
        x = x.isel(biocrop=crop_index)
    if land_index is not None and "landcover" in x.dims:
        x = x.isel(landcover=land_index)
    return np.asarray(x.values)


def _read_landgrid_coordinates(path: Path, n_land: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        if "lon" not in ds or "lat" not in ds:
            raise KeyError(f"{path} must contain 'lon' and 'lat'")
        lon = np.asarray(ds["lon"].values, dtype=np.float64).reshape(-1)
        lat = np.asarray(ds["lat"].values, dtype=np.float64).reshape(-1)

    if lon.size == n_land:
        pass
    elif lon.size == n_land + 17221:
        # Layout used by the current IMAGE_grid_5min.nc helper in this project.
        lon = lon[:n_land]
        lat = lat[:n_land]
    else:
        # Try the project land mask if the coordinate file contains the full grid.
        try:
            from timer.cdr.cdr_util import land_mask

            mask = np.asarray(land_mask("5m"), dtype=bool).reshape(-1)
            if lon.size == mask.size:
                lon = lon[mask]
                lat = lat[mask]
            else:
                raise ValueError
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ValueError(
                f"Cannot align {path} ({lon.size} points) with LandGrid ({n_land} cells)."
            ) from exc

    if lon.size != n_land or lat.size != n_land:
        raise ValueError("Coordinate alignment failed")
    if not np.all(np.isfinite(lon)) or not np.all(np.isfinite(lat)):
        raise ValueError("Non-finite longitude/latitude values")
    lon = ((lon + 180.0) % 360.0) - 180.0
    return lon, lat


def _read_garea(path: Path, n_land: int, final_time_index: int = -1) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with xr.open_dataset(path) as ds:
        var_name = "GAREA" if "GAREA" in ds.data_vars else next(iter(ds.data_vars))
        da = ds[var_name]
        if "time" in da.dims:
            da = da.isel(time=final_time_index)
        arr = np.asarray(da.values, dtype=np.float64).reshape(-1)

    if arr.size == n_land:
        return arr

    try:
        from timer.cdr.cdr_util import land_mask

        mask = np.asarray(land_mask("5m"), dtype=bool).reshape(-1)
        if arr.size == mask.size:
            return arr[mask]
    except Exception:
        pass

    raise ValueError(f"Cannot align GAREA array length {arr.size} with LandGrid length {n_land}")


def _aggregate_mean_map(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    resolution: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(values) & np.isfinite(lon) & np.isfinite(lat)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        valid &= np.isfinite(w) & (w > 0)
        ww = w[valid]
    else:
        ww = np.ones(int(valid.sum()), dtype=np.float64)

    lon_edges = np.arange(-180.0, 180.0 + resolution, resolution)
    lat_edges = np.arange(-90.0, 90.0 + resolution, resolution)
    weighted_sum, _, _ = np.histogram2d(
        lat[valid], lon[valid], bins=(lat_edges, lon_edges), weights=values[valid] * ww
    )
    weight_sum, _, _ = np.histogram2d(
        lat[valid], lon[valid], bins=(lat_edges, lon_edges), weights=ww
    )
    grid = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full_like(weighted_sum, np.nan, dtype=np.float64),
        where=weight_sum > 0,
    )
    return grid, lon_edges, lat_edges


def _aggregate_category_map(
    lon: np.ndarray,
    lat: np.ndarray,
    categories: np.ndarray,
    n_categories: int,
    resolution: float,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cat = np.asarray(categories).reshape(-1)
    valid_base = np.isfinite(cat) & np.isfinite(lon) & np.isfinite(lat)
    lon_edges = np.arange(-180.0, 180.0 + resolution, resolution)
    lat_edges = np.arange(-90.0, 90.0 + resolution, resolution)
    counts = []
    for k in range(n_categories):
        valid = valid_base & (cat == k)
        if weights is None:
            wk = None
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            valid &= np.isfinite(w) & (w > 0)
            wk = w[valid]
        h, _, _ = np.histogram2d(lat[valid], lon[valid], bins=(lat_edges, lon_edges), weights=wk)
        counts.append(h)
    stack = np.stack(counts, axis=0)
    total = np.sum(stack, axis=0)
    dominant = np.argmax(stack, axis=0).astype(float)
    dominant[total <= 0] = np.nan
    return dominant, lon_edges, lat_edges


def _robust_limits(values: np.ndarray, symmetric: bool = False) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    lo, hi = np.quantile(x, [0.02, 0.98])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    if symmetric:
        lim = max(abs(lo), abs(hi))
        return -lim, lim
    return float(lo), float(hi)


def _plot_continuous_map(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    resolution: float,
    area_weights: np.ndarray | None = None,
    symmetric: bool = False,
    log1p: bool = False,
) -> None:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    label = cbar_label
    if log1p:
        vals = np.where(vals >= 0, np.log10(1.0 + vals), np.nan)
        label = f"log10(1 + {cbar_label})"
    grid, lon_edges, lat_edges = _aggregate_mean_map(
        lon, lat, vals, resolution=resolution, weights=area_weights
    )
    vmin, vmax = _robust_limits(grid, symmetric=symmetric)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    mesh = ax.pcolormesh(lon_edges, lat_edges, grid, shading="auto", vmin=vmin, vmax=vmax)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
    cb.set_label(label)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_category_map(
    path: Path,
    lon: np.ndarray,
    lat: np.ndarray,
    categories: np.ndarray,
    labels: dict[int, str],
    title: str,
    resolution: float,
    area_weights: np.ndarray | None = None,
) -> None:
    dominant, lon_edges, lat_edges = _aggregate_category_map(
        lon, lat, categories, len(labels), resolution=resolution, weights=area_weights
    )
    # Discrete categorical plot; default colormap is sufficient for diagnostics.
    fig, ax = plt.subplots(figsize=(13, 6.5))
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        dominant,
        shading="auto",
        vmin=-0.5,
        vmax=len(labels) - 0.5,
        cmap=plt.get_cmap("tab10", len(labels)),
    )
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    cb = fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.05, pad=0.08)
    cb.set_ticks(list(labels))
    cb.set_ticklabels([labels[k] for k in labels])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_histogram(path: Path, values: np.ndarray, title: str, xlabel: str) -> None:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return
    lo, hi = np.quantile(x, [0.005, 0.995])
    x = x[(x >= lo) & (x <= hi)]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(x, bins=80)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Grid-cell count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@dataclass
class CheckResult:
    check: str
    status: str
    details: str
    max_abs_error: float | None = None
    max_rel_error: float | None = None
    fail_count: int | None = None


class BioCropReview:
    def __init__(
        self,
        input_dir: Path,
        report_dir: Path,
        grid_coordinates: Path | None,
        garea_path: Path | None,
        requested_years: Sequence[int] | None,
        map_resolution: float,
        all_years: bool,
        make_maps: bool,
    ) -> None:
        self.input_dir = input_dir
        self.report_dir = report_dir
        self.grid_coordinates = grid_coordinates
        self.garea_path = garea_path
        self.requested_years = requested_years
        self.map_resolution = map_resolution
        self.all_years = all_years
        self.make_maps = make_maps

        self.datasets: dict[str, xr.Dataset] = {}
        self.data: dict[str, xr.DataArray] = {}
        self.checks: list[CheckResult] = []
        self.summary_rows: list[dict] = []
        self.econ_rows: list[dict] = []
        self.best_rows: list[dict] = []
        self.land_rows: list[dict] = []

        self.n_land = 0
        self.n_crop = 0
        self.n_landcover = 0
        self.time_years = np.array([], dtype=float)
        self.year_indices: list[int] = []
        self.lon: np.ndarray | None = None
        self.lat: np.ndarray | None = None
        self.garea: np.ndarray | None = None

    def close(self) -> None:
        for ds in self.datasets.values():
            ds.close()

    def _record_check(
        self,
        check: str,
        ok: bool,
        details: str,
        *,
        warning: bool = False,
        metrics: dict[str, float] | None = None,
    ) -> None:
        status = "WARN" if warning and not ok else ("PASS" if ok else "FAIL")
        self.checks.append(
            CheckResult(
                check=check,
                status=status,
                details=details,
                max_abs_error=None if not metrics else metrics.get("max_abs_error"),
                max_rel_error=None if not metrics else metrics.get("max_rel_error"),
                fail_count=None if not metrics else int(metrics.get("fail_count", 0)),
            )
        )
        print(f"[{status:4s}] {check}: {details}")

    def open(self) -> None:
        print("\n" + "=" * 88)
        print("Opening BioCrop output NetCDF files")
        print("=" * 88)
        missing = []
        for var, filename in EXPECTED.items():
            path = self.input_dir / filename
            if not path.is_file():
                missing.append(str(path))
                continue
            ds = xr.open_dataset(path)
            if var not in ds:
                ds.close()
                raise KeyError(f"Expected variable {var!r} in {path}; found {list(ds.data_vars)}")
            self.datasets[var] = ds
            self.data[var] = ds[var]
            print(f"{var:24s} {str(self.data[var].dims):38s} shape={self.data[var].shape}")

        if missing:
            raise FileNotFoundError("Missing BioCrop output files:\n  " + "\n  ".join(missing))

        # Basic dimensions from BioNPVByLand, the richest output.
        rich = self.data["BioNPVByLand"]
        for required_dim in ("grid", "biocrop", "landcover"):
            if required_dim not in rich.dims:
                raise ValueError(f"BioNPVByLand is missing dimension {required_dim!r}")
        self.n_land = int(rich.sizes["grid"])
        self.n_crop = int(rich.sizes["biocrop"])
        self.n_landcover = int(rich.sizes["landcover"])

        # Pick canonical time axis from BioNPV.
        self.time_years = _time_values(self.data["BioNPV"])
        if self.time_years.size == 0:
            raise ValueError("BioNPV has no time coordinate")
        self.year_indices = (
            list(range(self.time_years.size))
            if self.all_years
            else _resolve_year_indices(self.time_years, self.requested_years)
        )

        self._validate_dimensions()
        self._load_optional_spatial_inputs()

    def _validate_dimensions(self) -> None:
        print("\nDimension checks")
        for name, da in self.data.items():
            ok = "grid" in da.dims and int(da.sizes["grid"]) == self.n_land
            self._record_check(
                f"{name}: grid dimension",
                ok,
                f"grid={da.sizes.get('grid')} expected={self.n_land}",
            )
            if "biocrop" in da.dims:
                okc = int(da.sizes["biocrop"]) == self.n_crop
                self._record_check(
                    f"{name}: biocrop dimension",
                    okc,
                    f"biocrop={da.sizes['biocrop']} expected={self.n_crop}",
                )
            if name in TIME_VARYING:
                okt = "time" in da.dims and int(da.sizes["time"]) == len(self.time_years)
                self._record_check(
                    f"{name}: time dimension",
                    okt,
                    f"time={da.sizes.get('time')} expected={len(self.time_years)}",
                )

        if self.n_crop != 5:
            self._record_check(
                "Bioenergy crop count",
                False,
                f"Expected 5 crops, found {self.n_crop}",
            )
        else:
            self._record_check("Bioenergy crop count", True, "5 crop classes")

        if self.n_landcover < 4:
            self._record_check(
                "Reference land-cover count",
                False,
                f"Expected at least 4 reference land classes, found {self.n_landcover}",
            )
        else:
            self._record_check(
                "Reference land-cover count",
                True,
                f"{self.n_landcover} land-cover classes",
            )

        print(
            "\nNOTE: BioPotFrac, BioPotFracByLand, BiomassProductivity and BiomassPotential "
            "are static final-state outputs in the current BioCrop implementation. Cross-file "
            "identities involving these fields are therefore checked at the final year only."
        )

    def _load_optional_spatial_inputs(self) -> None:
        if self.grid_coordinates and self.grid_coordinates.is_file():
            try:
                self.lon, self.lat = _read_landgrid_coordinates(self.grid_coordinates, self.n_land)
                print(f"Loaded LandGrid coordinates: {self.grid_coordinates}")
            except Exception as exc:
                print(f"WARNING: Could not load grid coordinates: {exc}")
        elif self.make_maps:
            print("WARNING: Grid coordinate file not found; maps will be skipped.")

        garea_candidates = []
        if self.garea_path is not None:
            garea_candidates.append(self.garea_path)
        garea_candidates.append(DEFAULT_SSP2M_DIR / "GAREA.NC")
        for path in garea_candidates:
            if path and path.is_file():
                try:
                    self.garea = _read_garea(path, self.n_land)
                    print(f"Loaded GAREA: {path}")
                    break
                except Exception as exc:
                    print(f"WARNING: Could not use GAREA {path}: {exc}")
        if self.garea is None:
            print("WARNING: GAREA unavailable. Area-weighted totals and potential identity check will be skipped.")

    def _add_stats(
        self,
        variable: str,
        values: np.ndarray,
        *,
        year: float | None = None,
        crop: int | None = None,
        land: int | None = None,
    ) -> None:
        row = {
            "variable": variable,
            "year": year,
            "crop": None if crop is None else _label_crop(crop),
            "landcover": None if land is None else _label_land(land),
        }
        row.update(_finite_stats(values))
        self.summary_rows.append(row)

    def static_diagnostics(self) -> None:
        print("\n" + "=" * 88)
        print("Static/final-state physical diagnostics")
        print("=" * 88)

        frac = _extract_slice(self.data["BioPotFrac"])
        frac_land = _extract_slice(self.data["BioPotFracByLand"])
        prod = _extract_slice(self.data["BiomassProductivity"])
        potential = _extract_slice(self.data["BiomassPotential"])

        # Stats by crop.
        for c in range(self.n_crop):
            self._add_stats("BioPotFrac", frac[:, c], crop=c)
            self._add_stats("BiomassProductivity", prod[:, c], crop=c)
            self._add_stats("BiomassPotential", potential[:, c], crop=c)
            for l in range(self.n_landcover):
                self._add_stats("BioPotFracByLand", frac_land[:, c, l], crop=c, land=l)

        # Fraction bounds.
        finite_frac = frac[np.isfinite(frac)]
        bad = np.sum((finite_frac < -ABS_TOL) | (finite_frac > 1.0 + ABS_TOL))
        self._record_check(
            "BioPotFrac in [0, 1]",
            bad == 0,
            f"out-of-range values={bad:,}; min={np.nanmin(frac):.6g}, max={np.nanmax(frac):.6g}",
        )
        finite_fl = frac_land[np.isfinite(frac_land)]
        bad_fl = np.sum((finite_fl < -ABS_TOL) | (finite_fl > 1.0 + ABS_TOL))
        self._record_check(
            "BioPotFracByLand in [0, 1]",
            bad_fl == 0,
            f"out-of-range values={bad_fl:,}; min={np.nanmin(frac_land):.6g}, max={np.nanmax(frac_land):.6g}",
        )

        # Land-source sum identity.
        sum_land = np.nansum(frac_land, axis=2)
        metrics = _error_metrics(frac, sum_land)
        self._record_check(
            "BioPotFrac = sum(BioPotFracByLand)",
            metrics["fail_count"] == 0,
            f"max abs err={metrics['max_abs_error']:.3e}; failures={metrics['fail_count']:,}",
            metrics=metrics,
        )

        # Urban must be zero if present.
        if self.n_landcover >= 5:
            urban = frac_land[:, :, 4]
            nbad = int(np.sum(np.isfinite(urban) & (np.abs(urban) > ABS_TOL)))
            self._record_check(
                "Urban bioenergy potential is zero",
                nbad == 0,
                f"non-zero urban crop-land entries={nbad:,}",
            )

        # Non-negativity.
        for name, arr in (("BiomassProductivity", prod), ("BiomassPotential", potential)):
            finite = arr[np.isfinite(arr)]
            nneg = int(np.sum(finite < -ABS_TOL))
            self._record_check(
                f"{name} non-negative",
                nneg == 0,
                f"negative values={nneg:,}; min={np.nanmin(arr):.6g}",
            )

        # Productivity support is explicitly zero where no eligible fraction.
        nbad = int(np.sum((frac <= EPS) & np.isfinite(prod) & (np.abs(prod) > ABS_TOL)))
        self._record_check(
            "No productivity where BioPotFrac=0",
            nbad == 0,
            f"violations={nbad:,}",
        )

        # Potential identity, if area is available. GAREA is km2; 1 km2 = 100 ha.
        if self.garea is not None:
            expected = prod * frac * self.garea[:, None] * 100.0
            metrics = _error_metrics(potential, expected, mask=np.isfinite(expected))
            self._record_check(
                "BiomassPotential accounting identity",
                metrics["fail_count"] == 0,
                "Potential = productivity × eligible fraction × GAREA × 100 ha/km2; "
                f"max rel err={metrics['max_rel_error']:.3e}; failures={metrics['fail_count']:,}",
                metrics=metrics,
            )

            for c in range(self.n_crop):
                eligible_area_mha = float(np.nansum(frac[:, c] * self.garea) * 100.0 / 1.0e6)
                total_potential_ej = float(np.nansum(potential[:, c]) / 1.0e9)
                print(
                    f"{_label_crop(c):12s} final eligible area = {eligible_area_mha:10.3f} Mha | "
                    f"primary potential = {total_potential_ej:10.4f} EJ/yr"
                )

        # Final-year maps for physical fields.
        if self.make_maps and self.lon is not None and self.lat is not None:
            mapdir = self.report_dir / "maps" / "physical_final"
            for c in range(self.n_crop):
                name = _safe_slug(_label_crop(c))
                _plot_continuous_map(
                    mapdir / f"BioPotFrac_{name}.png",
                    self.lon,
                    self.lat,
                    frac[:, c],
                    f"Eligible bioenergy land fraction — {_label_crop(c)} (final state)",
                    "Fraction",
                    self.map_resolution,
                    area_weights=self.garea,
                )
                _plot_continuous_map(
                    mapdir / f"BiomassProductivity_{name}.png",
                    self.lon,
                    self.lat,
                    prod[:, c],
                    f"Primary biomass productivity — {_label_crop(c)} (final state)",
                    "GJ/ha/yr",
                    self.map_resolution,
                    area_weights=self.garea,
                )
                _plot_continuous_map(
                    mapdir / f"BiomassPotential_{name}.png",
                    self.lon,
                    self.lat,
                    potential[:, c],
                    f"Cell biomass potential — {_label_crop(c)} (final state)",
                    "GJ/yr",
                    self.map_resolution,
                    area_weights=None,
                    log1p=True,
                )

    def time_diagnostics(self) -> None:
        print("\n" + "=" * 88)
        print("Time-varying economic diagnostics")
        print("=" * 88)

        feed_da = self.data["BioFeedstockValue"]
        rev_da = self.data["BioAnnualRevenue"]
        cost_da = self.data["BioAnnualNonLandCost"]
        npv_da = self.data["BioNPV"]
        npv_land_da = self.data["BioNPVByLand"]
        yield_da = self.data["BiomassYield"]

        # Static final-state arrays for final-year identities.
        prod_final = _extract_slice(self.data["BiomassProductivity"])
        frac_land_final = _extract_slice(self.data["BioPotFracByLand"])
        final_i = int(len(self.time_years) - 1)

        for ti in self.year_indices:
            year = float(self.time_years[ti])
            feed = _extract_slice(feed_da, time_index=ti)
            rev = _extract_slice(rev_da, time_index=ti)
            cost = _extract_slice(cost_da, time_index=ti)
            npv = _extract_slice(npv_da, time_index=ti)
            npv_land = _extract_slice(npv_land_da, time_index=ti)
            byield = _extract_slice(yield_da, time_index=ti)
            margin = rev - cost

            print(f"\n--- Year {year:g} ---")
            for c in range(self.n_crop):
                crop = _label_crop(c)
                self._add_stats("BiomassYield", byield[:, c], year=year, crop=c)
                self._add_stats("BioFeedstockValue", feed[:, c], year=year, crop=c)
                self._add_stats("BioAnnualRevenue", rev[:, c], year=year, crop=c)
                self._add_stats("BioAnnualNonLandCost", cost[:, c], year=year, crop=c)
                self._add_stats("BioAnnualMarginBeforeLand", margin[:, c], year=year, crop=c)
                self._add_stats("BioNPV", npv[:, c], year=year, crop=c)

                valid = np.isfinite(npv[:, c])
                pos = valid & (npv[:, c] > 0)
                neg = valid & (npv[:, c] < 0)
                if self.garea is not None:
                    pos_area_mha = float(np.sum(self.garea[pos]) * 100.0 / 1.0e6)
                else:
                    pos_area_mha = np.nan
                self.econ_rows.append(
                    {
                        "year": year,
                        "crop": crop,
                        "mean_feedstock_value": float(np.nanmean(feed[:, c])),
                        "mean_revenue": float(np.nanmean(rev[:, c])),
                        "mean_nonland_cost": float(np.nanmean(cost[:, c])),
                        "mean_margin_before_land": float(np.nanmean(margin[:, c])),
                        "median_npv": float(np.nanmedian(npv[:, c])),
                        "positive_npv_pct_of_finite": 100.0 * float(pos.sum()) / max(1, int(valid.sum())),
                        "negative_npv_pct_of_finite": 100.0 * float(neg.sum()) / max(1, int(valid.sum())),
                        "positive_npv_cell_area_mha_unadjusted": pos_area_mha,
                    }
                )
                print(
                    f"{crop:12s} netback mean={np.nanmean(feed[:, c]):9.3f} $/GJ | "
                    f"revenue={np.nanmean(rev[:, c]):10.2f} | cost={np.nanmean(cost[:, c]):10.2f} "
                    f"$/ha/yr | NPV median={np.nanmedian(npv[:, c]):12.2f} $/ha | "
                    f"positive={100.0 * pos.sum() / max(1, valid.sum()):6.2f}%"
                )

                for l in range(self.n_landcover):
                    vals = npv_land[:, c, l]
                    self._add_stats("BioNPVByLand", vals, year=year, crop=c, land=l)
                    v = np.isfinite(vals)
                    p = v & (vals > 0)
                    self.land_rows.append(
                        {
                            "year": year,
                            "crop": crop,
                            "landcover": _label_land(l),
                            "n_finite": int(v.sum()),
                            "positive_pct": 100.0 * float(p.sum()) / max(1, int(v.sum())),
                            "median_npv": float(np.nanmedian(vals)) if np.any(v) else np.nan,
                            "mean_npv": float(np.nanmean(vals)) if np.any(v) else np.nan,
                        }
                    )

            # Annual nonland costs should not be negative under the present formulation.
            finite_cost = cost[np.isfinite(cost)]
            nneg_cost = int(np.sum(finite_cost < -ABS_TOL))
            self._record_check(
                f"{year:g}: BioAnnualNonLandCost non-negative",
                nneg_cost == 0,
                f"negative values={nneg_cost:,}; min={np.nanmin(cost):.6g}",
            )

            # Best crop based on area-weighted BioNPV.
            finite_any = np.any(np.isfinite(npv), axis=1)
            npv_safe = np.where(np.isfinite(npv), npv, -np.inf)
            best_crop = np.argmax(npv_safe, axis=1).astype(float)
            best_npv = np.max(npv_safe, axis=1)
            best_crop[~finite_any] = np.nan
            best_npv[~finite_any] = np.nan
            profitable = finite_any & (best_npv > 0)

            for c in range(self.n_crop):
                selected = profitable & (best_crop == c)
                if self.garea is not None:
                    area = float(np.sum(self.garea[selected]) * 100.0 / 1.0e6)
                else:
                    area = np.nan
                self.best_rows.append(
                    {
                        "year": year,
                        "crop": _label_crop(c),
                        "selected_profitable_cells": int(selected.sum()),
                        "share_of_profitable_cells_pct": 100.0 * float(selected.sum()) / max(1, int(profitable.sum())),
                        "gridcell_area_mha_unadjusted": area,
                    }
                )

            print(
                f"Best-crop result: {int(profitable.sum()):,} cells have max NPV > 0 "
                f"({100.0 * profitable.sum() / self.n_land:.2f}% of LandGrid cells)."
            )

            # Final-year identities against static outputs.
            if ti == final_i:
                expected_rev = prod_final * np.maximum(feed, 0.0)
                metrics = _error_metrics(rev, expected_rev, mask=np.isfinite(expected_rev))
                self._record_check(
                    f"{year:g}: annual revenue identity",
                    metrics["fail_count"] == 0,
                    "Revenue = final productivity × max(TIMER netback, 0); "
                    f"max rel err={metrics['max_rel_error']:.3e}; failures={metrics['fail_count']:,}",
                    metrics=metrics,
                )

                # BioNPV is weighted by source-specific eligible fractions.
                valid_weight = (frac_land_final > EPS) & np.isfinite(npv_land)
                weights = np.where(valid_weight, frac_land_final, 0.0)
                weighted_sum = np.sum(
                    np.where(valid_weight, frac_land_final * npv_land, 0.0), axis=2
                )
                weight_sum = np.sum(weights, axis=2)
                expected_npv = np.divide(
                    weighted_sum,
                    weight_sum,
                    out=np.full_like(weighted_sum, np.nan, dtype=np.float64),
                    where=weight_sum > EPS,
                )
                metrics = _error_metrics(npv, expected_npv, mask=np.isfinite(expected_npv))
                self._record_check(
                    f"{year:g}: BioNPV weighted-source identity",
                    metrics["fail_count"] == 0,
                    "BioNPV = weighted mean of BioNPVByLand using BioPotFracByLand; "
                    f"max rel err={metrics['max_rel_error']:.3e}; failures={metrics['fail_count']:,}",
                    metrics=metrics,
                )

            if self.make_maps and self.lon is not None and self.lat is not None:
                self._maps_for_year(year, feed, rev, cost, npv, npv_land, best_crop, best_npv)

    def _maps_for_year(
        self,
        year: float,
        feed: np.ndarray,
        rev: np.ndarray,
        cost: np.ndarray,
        npv: np.ndarray,
        npv_land: np.ndarray,
        best_crop: np.ndarray,
        best_npv: np.ndarray,
    ) -> None:
        year_tag = f"{year:g}"
        mapdir = self.report_dir / "maps" / f"year_{year_tag}"
        margin = rev - cost
        for c in range(self.n_crop):
            cname = _safe_slug(_label_crop(c))
            for varname, vals, label, symmetric in (
                ("BioFeedstockValue", feed[:, c], "$/GJ-primary", False),
                ("BioAnnualRevenue", rev[:, c], "$/ha/yr", False),
                ("BioAnnualNonLandCost", cost[:, c], "$/ha/yr", False),
                ("AnnualMarginBeforeLand", margin[:, c], "$/ha/yr", True),
                ("BioNPV", npv[:, c], "$/ha", True),
            ):
                _plot_continuous_map(
                    mapdir / f"{varname}_{cname}_{year_tag}.png",
                    self.lon,
                    self.lat,
                    vals,
                    f"{varname} — {_label_crop(c)} — {year_tag}",
                    label,
                    self.map_resolution,
                    area_weights=self.garea,
                    symmetric=symmetric,
                )

            _plot_histogram(
                self.report_dir / "histograms" / f"BioNPV_{cname}_{year_tag}.png",
                npv[:, c],
                f"BioNPV distribution — {_label_crop(c)} — {year_tag}",
                "$/ha",
            )

        _plot_category_map(
            mapdir / f"BestCrop_{year_tag}.png",
            self.lon,
            self.lat,
            best_crop,
            CROP_NAMES,
            f"Highest-NPV bioenergy crop — {year_tag}",
            self.map_resolution,
            area_weights=self.garea,
        )
        _plot_continuous_map(
            mapdir / f"BestNPV_{year_tag}.png",
            self.lon,
            self.lat,
            best_npv,
            f"Maximum bioenergy NPV across crops — {year_tag}",
            "$/ha",
            self.map_resolution,
            area_weights=self.garea,
            symmetric=True,
        )

        # Source-land-specific best NPV/crop maps.
        for l in range(min(self.n_landcover, 4)):  # Urban is not an eligible source.
            land_name = _safe_slug(_label_land(l))
            vals = npv_land[:, :, l]
            valid_any = np.any(np.isfinite(vals), axis=1)
            safe = np.where(np.isfinite(vals), vals, -np.inf)
            best_l_crop = np.argmax(safe, axis=1).astype(float)
            best_l_npv = np.max(safe, axis=1)
            best_l_crop[~valid_any] = np.nan
            best_l_npv[~valid_any] = np.nan
            _plot_category_map(
                mapdir / f"BestCrop_{land_name}_{year_tag}.png",
                self.lon,
                self.lat,
                best_l_crop,
                CROP_NAMES,
                f"Highest-NPV crop on reference {_label_land(l).lower()} — {year_tag}",
                self.map_resolution,
                area_weights=self.garea,
            )
            _plot_continuous_map(
                mapdir / f"BestNPV_{land_name}_{year_tag}.png",
                self.lon,
                self.lat,
                best_l_npv,
                f"Best bioenergy NPV on reference {_label_land(l).lower()} — {year_tag}",
                "$/ha",
                self.map_resolution,
                area_weights=self.garea,
                symmetric=True,
            )

    def regional_netback_diagnostic(self) -> None:
        print("\n" + "=" * 88)
        print("Regional TIMER-netback mapping diagnostic")
        print("=" * 88)
        try:
            from timer.cdr.cdr_util import region_to_grid

            region_cells, region_mask = region_to_grid("5m")
            regions = np.asarray(region_cells, dtype=np.int16).reshape(-1)
            mask = np.asarray(region_mask, dtype=bool).reshape(-1)
            if regions.size != self.n_land:
                raise ValueError(
                    f"region_to_grid returned {regions.size} cells; outputs contain {self.n_land}"
                )
        except Exception as exc:
            print(f"Skipping regional netback check: {exc}")
            return

        da = self.data["BioFeedstockValue"]
        rows = []
        for ti in self.year_indices:
            year = float(self.time_years[ti])
            feed = _extract_slice(da, time_index=ti)
            for r in np.unique(regions[mask]):
                rm = mask & (regions == r)
                for c in range(self.n_crop):
                    x = feed[rm, c]
                    x = x[np.isfinite(x)]
                    if x.size == 0:
                        continue
                    rows.append(
                        {
                            "year": year,
                            "region_index": int(r),
                            "crop": _label_crop(c),
                            "n_cells": int(x.size),
                            "min": float(np.min(x)),
                            "max": float(np.max(x)),
                            "range": float(np.max(x) - np.min(x)),
                            "mean": float(np.mean(x)),
                        }
                    )
        frame = pd.DataFrame(rows)
        if frame.empty:
            return
        max_range = float(frame["range"].max())
        nbad = int((frame["range"] > ABS_TOL).sum())
        self._record_check(
            "BioFeedstockValue constant within IMAGE region",
            nbad == 0,
            f"region/crop/year groups with range>{ABS_TOL:g}: {nbad}; max range={max_range:.3e}",
        )
        frame.to_csv(self.report_dir / "regional_netback_diagnostics.csv", index=False)

    def write_reports(self) -> None:
        self.report_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([c.__dict__ for c in self.checks]).to_csv(
            self.report_dir / "consistency_checks.csv", index=False
        )
        pd.DataFrame(self.summary_rows).to_csv(
            self.report_dir / "variable_summary_stats.csv", index=False
        )
        pd.DataFrame(self.econ_rows).to_csv(
            self.report_dir / "crop_year_economics.csv", index=False
        )
        pd.DataFrame(self.best_rows).to_csv(
            self.report_dir / "best_crop_shares.csv", index=False
        )
        pd.DataFrame(self.land_rows).to_csv(
            self.report_dir / "source_land_npv_summary.csv", index=False
        )

        failed = [c for c in self.checks if c.status == "FAIL"]
        warned = [c for c in self.checks if c.status == "WARN"]
        print("\n" + "=" * 88)
        print("Review summary")
        print("=" * 88)
        print(f"Checks: {len(self.checks)} total | {len(failed)} FAIL | {len(warned)} WARN")
        if failed:
            print("\nFAILED CHECKS:")
            for c in failed:
                print(f"  - {c.check}: {c.details}")
        print(f"\nReports written to: {self.report_dir}")
        if self.make_maps:
            print(f"Maps written to:    {self.report_dir / 'maps'}")

    def run(self) -> None:
        try:
            self.open()
            self.report_dir.mkdir(parents=True, exist_ok=True)
            self.static_diagnostics()
            self.time_diagnostics()
            self.regional_netback_diagnostic()
            self.write_reports()
        finally:
            self.close()


# =============================================================================
# Command-line interface
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review IMAGE/TIMER BioCrop NetCDF outputs with diagnostics and maps."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory containing BioPotFrac.nc, BioNPV.nc, etc. Default: current directory.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for CSV diagnostics and PNGs. Default: <input-dir>/biocrops_review.",
    )
    parser.add_argument(
        "--grid-coordinates",
        type=Path,
        default=DEFAULT_GRID_COORDINATES,
        help="IMAGE LandGrid coordinate NetCDF containing lon and lat.",
    )
    parser.add_argument(
        "--garea",
        type=Path,
        default=None,
        help=(
            "Optional GAREA.NC path. If omitted, the script tries "
            f"{DEFAULT_SSP2M_DIR / 'GAREA.NC'}"
        ),
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="*",
        default=None,
        help="Years to diagnose/map, e.g. --years 2030 2050 2100. Nearest available year is used.",
    )
    parser.add_argument(
        "--all-years",
        action="store_true",
        help="Run time-varying diagnostics and maps for every model year (can be slow).",
    )
    parser.add_argument(
        "--map-resolution",
        type=float,
        default=1.0,
        help="Diagnostic map aggregation resolution in degrees. Default: 1.0.",
    )
    parser.add_argument(
        "--no-maps",
        action="store_true",
        help="Run numerical diagnostics only; do not create PNG maps/histograms.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    report_dir = (args.report_dir or (input_dir / "biocrops_review")).resolve()

    print(f"BioCrop output directory: {input_dir}")
    print(f"Review output directory:  {report_dir}")

    review = BioCropReview(
        input_dir=input_dir,
        report_dir=report_dir,
        grid_coordinates=args.grid_coordinates,
        garea_path=args.garea,
        requested_years=args.years,
        map_resolution=args.map_resolution,
        all_years=args.all_years,
        make_maps=not args.no_maps,
    )
    try:
        review.run()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path
import warnings

import numpy as np
import xarray as xr

from timer.cdr.results.biochar.biochar_results import R10, IMAGE_REGIONS


"""
Quantitative analysis for Figure 1 of the biochar manuscript.

Prints the main quantities needed to write the Results subsection:
  1) global 2100 carbon-price response;
  2) R10 regional dynamics and ranking changes;
  3) R26 cumulative 2030-2100 sensitivity losses at c400;
  4) temporal dynamics at c400;
  5) a compact writing-oriented summary.

Accounting follows the Figure 1 plotting module:
  - mitigation exports are timestep totals;
  - annual values = timestep total / inferred timestep width;
  - cumulative values = sum of timestep totals;
  - BiocharRealisedMitigation is preferred;
  - explicit mitigation components are used as a fallback.
"""


# =============================================================================
# Configuration
# =============================================================================

BASE_DIR = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output")

CARBON_PRICES = (50, 100, 150, 200, 250, 400)
YEAR = 2100
START_YEAR = 2030
FOCUS_PRICE = 400

SCENARIOS = {
    "Main": "biochar_c{price}",
    "Stylised biochar": "biochar_c{price}_simple",
    "No co-services": "biochar_c{price}_no_coservices",
}

REALIZED_VAR = "BiocharRealisedMitigation"
FALLBACK_COMPONENTS = (
    ("BiocharFieldCDR", True),
    ("BiocharSyngasCCS", False),
    ("BiocharFossilMitig", False),
    ("BiocharN2OMitig", False),
    ("BiocharCH4Mitig", False),
)

R10_DISPLAY = {
    "Middle East & North Africa": "MENA",
    "Sub-Saharan Africa": "SSA",
}

R26_DISPLAY = {
    "CAN": "Canada", "USA": "United States", "MEX": "Mexico",
    "RCAM": "Rest of Central America", "BRA": "Brazil",
    "RSAM": "Rest of South America", "NAF": "North Africa",
    "WAF": "West Africa", "EAF": "East Africa", "SAF": "South Africa",
    "WEU": "Western Europe", "CEU": "Central Europe", "TUR": "Turkey",
    "UKR": "Ukraine", "STAN": "Central Asia", "RUS": "Russia",
    "ME": "Middle East", "INDIA": "India", "KOR": "Korea",
    "CHN": "China", "SEAS": "Southeast Asia", "INDO": "Indonesia",
    "JAP": "Japan", "OCE": "Oceania", "RSAS": "Rest of South Asia",
    "RSAF": "Rest of Southern Africa",
}

REPORT_YEARS = (2030, 2050, 2075, 2100)
TOP_R10 = 5
TOP_R26 = 10
_SERIES_CACHE = {}


# =============================================================================
# Helpers / loading
# =============================================================================

def scenario_dir(base_dir, template, price):
    return Path(base_dir) / template.format(price=int(price))


def display_r10(name):
    return R10_DISPLAY.get(name, name)


def display_r26(code):
    return R26_DISPLAY.get(str(code), str(code))


def safe_pct(num, den):
    if not np.isfinite(den) or abs(den) < 1e-15:
        return np.nan
    return 100.0 * num / den


def section(title):
    print("\n" + "=" * 116)
    print(title)
    print("=" * 116)


def time_years(da):
    vals = da["time"].values
    if np.issubdtype(vals.dtype, np.datetime64):
        return da["time"].dt.year.values.astype(int)
    try:
        return np.asarray(vals, dtype=int)
    except (TypeError, ValueError):
        return da["time"].dt.year.values.astype(int)


def timestep_widths(years):
    years = np.asarray(years, dtype=int)
    if years.size < 2:
        raise ValueError("Need at least two time coordinates to infer timestep width.")
    dt = np.empty(years.size, dtype=float)
    dt[0] = years[1] - years[0]
    dt[1:] = np.diff(years)
    if np.any(dt <= 0):
        raise ValueError(f"Invalid time coordinates: {years.tolist()}")
    return dt


def reduce_time_r26(da):
    if "time" not in da.dims or "region" not in da.dims:
        raise ValueError(f"Expected time and region dimensions; got {da.dims}.")
    extra = [d for d in da.dims if d not in ("time", "region")]
    if extra:
        da = da.sum(dim=extra, skipna=True)
    da = da.transpose("time", "region")
    arr = np.asarray(da.values, dtype=float)
    if arr.shape[1] < len(IMAGE_REGIONS):
        raise ValueError(
            f"Expected at least {len(IMAGE_REGIONS)} regions; got {arr.shape[1]}."
        )
    return arr[:, :len(IMAGE_REGIONS)]


def load_series(path):
    """Return years, dt, timestep-total Gt, annualised Gt/yr, source."""
    path = Path(path)
    key = str(path)
    if key in _SERIES_CACHE:
        return _SERIES_CACHE[key]

    realised_file = path / f"{REALIZED_VAR}.nc"
    if realised_file.exists():
        with xr.open_dataset(realised_file) as ds:
            if REALIZED_VAR in ds:
                da = ds[REALIZED_VAR]
                years = time_years(da)
                step = reduce_time_r26(da) / 1e9
                source = REALIZED_VAR
            else:
                step = None
        if step is not None:
            dt = timestep_widths(years)
            out = {
                "years": years,
                "dt": dt,
                "step": step,
                "annual": step / dt[:, None],
                "source": source,
            }
            _SERIES_CACHE[key] = out
            return out

    years_ref = None
    total = None
    for var, required in FALLBACK_COMPONENTS:
        fn = path / f"{var}.nc"
        if not fn.exists():
            if required:
                raise FileNotFoundError(fn)
            warnings.warn(f"{fn.name} missing in {path.name}; treating as zero.")
            continue
        with xr.open_dataset(fn) as ds:
            if var not in ds:
                if required:
                    raise KeyError(f"{var} missing in {fn}")
                continue
            da = ds[var]
            years = time_years(da)
            vals = reduce_time_r26(da)
        if years_ref is None:
            years_ref = years
            total = np.zeros_like(vals)
        elif not np.array_equal(years_ref, years):
            raise ValueError(f"Component time axes differ in {path.name}.")
        total += vals

    if years_ref is None:
        raise RuntimeError(f"No mitigation outputs found in {path}.")

    warnings.warn(
        f"Using component fallback for {path.name}; regional transport-emission "
        "penalty is not included."
    )
    step = total / 1e9
    dt = timestep_widths(years_ref)
    out = {
        "years": years_ref,
        "dt": dt,
        "step": step,
        "annual": step / dt[:, None],
        "source": "component fallback",
    }
    _SERIES_CACHE[key] = out
    return out


def annual_r26(path, year):
    s = load_series(path)
    idx = np.where(s["years"] == int(year))[0]
    if len(idx) == 0:
        raise ValueError(f"Year {year} not available in {path}.")
    i = int(idx[0])
    return s["annual"][i], float(s["dt"][i]), s["source"]


def cumulative_r26(path, start_year, end_year):
    s = load_series(path)
    mask = (s["years"] >= start_year) & (s["years"] <= end_year)
    if not np.any(mask):
        raise ValueError(f"No years {start_year}-{end_year} in {path}.")
    return np.nansum(s["step"][mask], axis=0), s["source"]


def r26_to_r10(v):
    v = np.asarray(v, dtype=float)
    return np.asarray([
        np.nansum(v[[IMAGE_REGIONS.index(r) for r in members]])
        for members in R10.values()
    ])


def ranks_desc(v):
    order = np.argsort(-np.asarray(v), kind="stable")
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


# =============================================================================
# Build 2100 carbon-price data
# =============================================================================

def build_price_data(base_dir=BASE_DIR, prices=CARBON_PRICES, year=YEAR):
    out = {}
    for scen, template in SCENARIOS.items():
        glob, r10, r26, dts, src = [], [], [], [], []
        for price in prices:
            v, dt, source = annual_r26(scenario_dir(base_dir, template, price), year)
            r26.append(v)
            r10.append(r26_to_r10(v))
            glob.append(float(np.nansum(v)))
            dts.append(dt)
            src.append(source)
        out[scen] = {
            "global": np.asarray(glob),
            "r10": np.asarray(r10),
            "r26": np.asarray(r26),
            "dt": np.asarray(dts),
            "source": src,
        }
    return out


# =============================================================================
# 1. Global carbon-price response
# =============================================================================

def analyse_global(price_data, prices=CARBON_PRICES):
    section("1. GLOBAL 2100 CARBON-PRICE RESPONSE")
    main = price_data["Main"]["global"]
    styl = price_data["Stylised biochar"]["global"]
    nocos = price_data["No co-services"]["global"]

    print(
        f"{'Price':>6s} {'Main':>8s} {'Stylised':>9s} {'NoCo':>8s} "
        f"{'ΔStyl':>8s} {'Styl%':>7s} {'ΔNoCo':>8s} {'NoCo%':>7s}"
    )
    for i, p in enumerate(prices):
        ds, dn = main[i] - styl[i], main[i] - nocos[i]
        print(
            f"{p:6.0f} {main[i]:8.3f} {styl[i]:9.3f} {nocos[i]:8.3f} "
            f"{ds:8.3f} {safe_pct(ds, main[i]):6.1f}% "
            f"{dn:8.3f} {safe_pct(dn, main[i]):6.1f}%"
        )

    print("\nLargest sampled annual sensitivity gaps:")
    for name, vals in (("Stylised biochar", styl), ("No co-services", nocos)):
        gap = main - vals
        i = int(np.nanargmax(np.abs(gap)))
        print(
            f"  {name:<17s}: c{prices[i]}, {gap[i]:.3f} Gt/yr "
            f"({safe_pct(gap[i], main[i]):.1f}% of main)"
        )

    print("\nApproach to each scenario's c400 value:")
    for name in SCENARIOS:
        vals = price_data[name]["global"]
        ceiling = vals[-1]
        pieces = []
        for frac in (0.80, 0.90, 0.95):
            idx = np.where(vals >= frac * ceiling)[0]
            pieces.append(
                f"{int(frac*100)}%=" + (f"c{prices[idx[0]]}" if len(idx) else "NA")
            )
        print(f"  {name:<17s}: c400={ceiling:.3f} Gt/yr | " + ", ".join(pieces))

    if 200 in prices and 250 in prices:
        i200, i250 = list(prices).index(200), list(prices).index(250)
        c400 = main[-1]
        print("\nMain-scenario saturation:")
        print(f"  c200 = {main[i200]:.3f} Gt/yr ({100*main[i200]/c400:.1f}% of c400)")
        print(f"  c250 = {main[i250]:.3f} Gt/yr ({100*main[i250]/c400:.1f}% of c400)")
        print(f"  c250→c400 gain = {c400-main[i250]:.3f} Gt/yr")


# =============================================================================
# 2. R10 regional dynamics
# =============================================================================

def gap_decomposition(price_data, prices, sensitivity, price, top_n=TOP_R10):
    i = list(prices).index(price)
    main = price_data["Main"]["r10"][i]
    sens = price_data[sensitivity]["r10"][i]
    loss = main - sens
    total = float(np.nansum(loss))
    order = np.argsort(-loss)
    regions = list(R10.keys())

    print(f"\nTop R10 contributors to the {sensitivity} gap at c{price} (global gap={total:.3f} Gt/yr):")
    print(f"{'Region':<22s} {'Main':>8s} {'Sens.':>8s} {'Loss':>8s} {'Share':>8s}")
    for j in order[:top_n]:
        print(
            f"{display_r10(regions[j]):<22s} {main[j]:8.3f} {sens[j]:8.3f} "
            f"{loss[j]:8.3f} {safe_pct(loss[j], total):7.1f}%"
        )


def analyse_r10(price_data, prices=CARBON_PRICES, focus_price=FOCUS_PRICE):
    section("2. REGIONAL 2100 DYNAMICS (R10)")
    i = list(prices).index(focus_price)
    regions = list(R10.keys())

    main = price_data["Main"]["r10"][i]
    styl = price_data["Stylised biochar"]["r10"][i]
    nocos = price_data["No co-services"]["r10"][i]

    mr, sr, nr = ranks_desc(main), ranks_desc(styl), ranks_desc(nocos)
    order = np.argsort(-main)
    total = float(np.nansum(main))

    print(f"R10 outcomes at c{focus_price}, sorted by main mitigation:")
    print(
        f"{'Region':<22s} {'Main':>7s} {'Share':>7s} {'Styl':>7s} {'ΔStyl':>7s} "
        f"{'rank':>7s} {'NoCo':>7s} {'ΔNoCo':>7s} {'rank':>7s}"
    )
    for j in order:
        print(
            f"{display_r10(regions[j]):<22s} {main[j]:7.3f} {safe_pct(main[j], total):6.1f}% "
            f"{styl[j]:7.3f} {main[j]-styl[j]:7.3f} {mr[j]}→{sr[j]} "
            f"{nocos[j]:7.3f} {main[j]-nocos[j]:7.3f} {mr[j]}→{nr[j]}"
        )

    print("\nRegional ordering at c400:")
    for name in SCENARIOS:
        vals = price_data[name]["r10"][i]
        names = [display_r10(regions[j]) for j in np.argsort(-vals)]
        print(f"  {name:<17s}: " + " > ".join(names))

    main_g = price_data["Main"]["global"]
    for name in ("Stylised biochar", "No co-services"):
        gap = main_g - price_data[name]["global"]
        imax = int(np.nanargmax(np.abs(gap)))
        gap_decomposition(price_data, prices, name, prices[imax])

    print("\nTop regional patterns across carbon prices (values in Gt/yr):")
    print(
        f"{'Price':>6s} | {'Top 3 main':<48s} | {'Top 3 stylised losses':<47s} | {'Top 3 no-co-services losses':<47s}"
    )
    for ip, p in enumerate(prices):
        m = price_data["Main"]["r10"][ip]
        ds = m - price_data["Stylised biochar"]["r10"][ip]
        dn = m - price_data["No co-services"]["r10"][ip]
        top_m, top_s, top_n = np.argsort(-m)[:3], np.argsort(-ds)[:3], np.argsort(-dn)[:3]
        fmt = lambda inds, vals: ", ".join(f"{display_r10(regions[j])} ({vals[j]:.2f})" for j in inds)
        print(f"{p:6.0f} | {fmt(top_m,m):<48s} | {fmt(top_s,ds):<47s} | {fmt(top_n,dn):<47s}")


# =============================================================================
# 3. R26 cumulative losses at c400
# =============================================================================

def print_top_r26(main, sens, name, region_names, top_n=TOP_R26):
    loss = main - sens
    total = float(np.nansum(loss))
    order = np.argsort(-loss)
    print(f"\nLargest cumulative losses: {name}")
    print(f"{'Region':<28s} {'Main':>8s} {'Sens.':>8s} {'Loss':>8s} {'Loss/main':>10s} {'Share':>8s}")
    for j in order[:top_n]:
        print(
            f"{region_names[j]:<28s} {main[j]:8.3f} {sens[j]:8.3f} {loss[j]:8.3f} "
            f"{safe_pct(loss[j],main[j]):9.1f}% {safe_pct(loss[j],total):7.1f}%"
        )
    positives = [j for j in order if loss[j] > 0]
    for n in (3, 5, 10):
        top = positives[:n]
        if top and abs(total) > 1e-15:
            print(f"  Top {len(top)} regions account for {100*np.nansum(loss[top])/total:.1f}% of global loss.")


def analyse_r26_cumulative(base_dir=BASE_DIR, price=FOCUS_PRICE, start_year=START_YEAR, end_year=YEAR, price_data=None):
    section(f"3. CUMULATIVE REGIONAL EFFECTS (R26), {start_year}-{end_year}, c{price}")
    vectors, sources = {}, {}
    for name, template in SCENARIOS.items():
        vectors[name], sources[name] = cumulative_r26(scenario_dir(base_dir, template, price), start_year, end_year)

    main, styl, nocos = vectors["Main"], vectors["Stylised biochar"], vectors["No co-services"]
    gm, gs, gn = map(float, (np.nansum(main), np.nansum(styl), np.nansum(nocos)))

    print(f"Global cumulative mitigation:")
    print(f"  Main             : {gm:.3f} GtCO2-eq")
    print(f"  Stylised biochar : {gs:.3f} | loss={gm-gs:.3f} ({safe_pct(gm-gs,gm):.1f}%)")
    print(f"  No co-services   : {gn:.3f} | loss={gm-gn:.3f} ({safe_pct(gm-gn,gm):.1f}%)")

    names = [display_r26(code) for code in IMAGE_REGIONS]
    print_top_r26(main, styl, "Stylised biochar", names)
    print_top_r26(main, nocos, "No co-services", names)

    if price_data is not None:
        i = list(CARBON_PRICES).index(price)
        annual_main = price_data["Main"]["global"][i]
        print("\nCumulative versus 2100 annual gap:")
        for name, cum in (("Stylised biochar", gs), ("No co-services", gn)):
            annual_s = price_data[name]["global"][i]
            print(
                f"  {name:<17s}: 2100 gap={annual_main-annual_s:.3f} Gt/yr "
                f"({safe_pct(annual_main-annual_s,annual_main):.1f}%); "
                f"cumulative gap={gm-cum:.3f} Gt ({safe_pct(gm-cum,gm):.1f}%)"
            )

    print("\nData sources:")
    for name, source in sources.items():
        print(f"  {name:<17s}: {source}")

    return vectors


# =============================================================================
# 4. Temporal dynamics at c400
# =============================================================================

def global_annual(series, year):
    i = int(np.where(series["years"] == year)[0][0])
    return float(np.nansum(series["annual"][i]))


def global_cumulative(series, start_year, end_year):
    mask = (series["years"] >= start_year) & (series["years"] <= end_year)
    return float(np.nansum(series["step"][mask]))


def analyse_time(base_dir=BASE_DIR, price=FOCUS_PRICE, start_year=START_YEAR):
    section(f"4. TEMPORAL DYNAMICS AT c{price}")
    series = {
        name: load_series(scenario_dir(base_dir, template, price))
        for name, template in SCENARIOS.items()
    }
    common = sorted(set.intersection(*(set(s["years"].tolist()) for s in series.values())))
    report = [y for y in REPORT_YEARS if y in common]
    if not report:
        report = [common[0], common[len(common)//2], common[-1]]

    print("Annual global mitigation:")
    print(f"{'Year':>6s} {'Main':>9s} {'Styl':>9s} {'NoCo':>9s} {'Styl loss':>10s} {'NoCo loss':>10s}")
    for y in report:
        m, s, n = (global_annual(series[k], y) for k in ("Main", "Stylised biochar", "No co-services"))
        print(f"{y:6d} {m:9.3f} {s:9.3f} {n:9.3f} {safe_pct(m-s,m):9.1f}% {safe_pct(m-n,m):9.1f}%")

    print(f"\nCumulative sensitivity gaps from {start_year}:")
    print(f"{'Through':>7s} {'Main':>9s} {'ΔStyl':>9s} {'Styl%':>8s} {'ΔNoCo':>9s} {'NoCo%':>8s}")
    for y in report:
        if y < start_year:
            continue
        m, s, n = (global_cumulative(series[k], start_year, y) for k in ("Main", "Stylised biochar", "No co-services"))
        print(f"{y:7d} {m:9.3f} {m-s:9.3f} {safe_pct(m-s,m):7.1f}% {m-n:9.3f} {safe_pct(m-n,m):7.1f}%")

    print("\nLargest annual gap over time:")
    for name in ("Stylised biochar", "No co-services"):
        candidates = []
        for y in common:
            if y < start_year:
                continue
            m = global_annual(series["Main"], y)
            s = global_annual(series[name], y)
            candidates.append((abs(m-s), y, m-s, m))
        _, y, gap, m = max(candidates, key=lambda x: x[0])
        print(f"  {name:<17s}: {y}, gap={gap:.3f} Gt/yr ({safe_pct(gap,m):.1f}% of main)")

    print("\nLeading R10 contributors in the main scenario over time:")
    regions = list(R10.keys())
    for y in report:
        s = series["Main"]
        i = int(np.where(s["years"] == y)[0][0])
        r10 = r26_to_r10(s["annual"][i])
        order = np.argsort(-r10)[:3]
        total = float(np.nansum(r10))
        text = "; ".join(f"{display_r10(regions[j])} {r10[j]:.2f} ({safe_pct(r10[j],total):.0f}%)" for j in order)
        print(f"  {y}: {text}")


# =============================================================================
# 5. Compact writing-oriented summary
# =============================================================================

def writing_summary(price_data, cumulative_vectors, prices=CARBON_PRICES, focus_price=FOCUS_PRICE):
    section("5. COMPACT RESULTS-WRITING SUMMARY")
    main = price_data["Main"]["global"]
    styl = price_data["Stylised biochar"]["global"]
    nocos = price_data["No co-services"]["global"]
    i = list(prices).index(focus_price)

    ds, dn = main-styl, main-nocos
    istyl, inoco = int(np.nanargmax(np.abs(ds))), int(np.nanargmax(np.abs(dn)))

    print(f"• Main at c{focus_price}: {main[i]:.3f} GtCO2-eq/yr in {YEAR}.")
    print(f"• Largest stylised annual gap: c{prices[istyl]}, {ds[istyl]:.3f} Gt/yr ({safe_pct(ds[istyl],main[istyl]):.1f}%).")
    print(f"• Stylised gap at c{focus_price}: {ds[i]:.3f} Gt/yr ({safe_pct(ds[i],main[i]):.1f}%).")
    print(f"• No-co-services gap at c{focus_price}: {dn[i]:.3f} Gt/yr ({safe_pct(dn[i],main[i]):.1f}%).")

    r10 = price_data["Main"]["r10"][i]
    regions = list(R10.keys())
    order = np.argsort(-r10)[:3]
    print("• Top R10 contributors at c400: " + ", ".join(
        f"{display_r10(regions[j])} ({safe_pct(r10[j],np.nansum(r10)):.1f}%)" for j in order
    ) + ".")

    main_c = cumulative_vectors["Main"]
    names = [display_r26(code) for code in IMAGE_REGIONS]
    for label in ("Stylised biochar", "No co-services"):
        loss = main_c - cumulative_vectors[label]
        order = np.argsort(-loss)[:5]
        print(f"• Largest R26 cumulative losses for {label.lower()}: " + ", ".join(
            f"{names[j]} ({loss[j]:.2f} Gt)" for j in order
        ) + ".")


# =============================================================================
# Main
# =============================================================================

def run_analysis(
    base_dir=BASE_DIR,
    prices=CARBON_PRICES,
    year=YEAR,
    start_year=START_YEAR,
    focus_price=FOCUS_PRICE,
):
    data = build_price_data(base_dir, prices, year)

    section("0. ACCOUNTING / SOURCE CHECK")
    for name in SCENARIOS:
        print(
            f"{name:<17s}: sources={sorted(set(data[name]['source']))}; "
            f"2100 dt={sorted(set(data[name]['dt'].tolist()))} yr"
        )

    analyse_global(data, prices)
    analyse_r10(data, prices, focus_price)
    cumulative_vectors = analyse_r26_cumulative(
        base_dir, focus_price, start_year, year, price_data=data
    )
    analyse_time(base_dir, focus_price, start_year)
    writing_summary(data, cumulative_vectors, prices, focus_price)

    return {"price_data": data, "cumulative_vectors": cumulative_vectors}


if __name__ == "__main__":
    outputs = run_analysis()

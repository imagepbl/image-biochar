from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from timer.cdr.cdr_util import (
    BiocharFeedstock,
    PyrolysisTechnology,
    biochar_data_dir,
)

try:
    from adjustText import adjust_text
except Exception:
    adjust_text = None


"""
Standalone figure of feedstock carbon-storage properties for low- and
high-temperature pyrolysis.

Panels
------
a. Low-temperature pyrolysis
b. High-temperature pyrolysis

For each non-biofuel feedstock, the figure shows:
- x-axis: biochar carbon fraction (tC t biochar^-1)
- y-axis: 100-year retained carbon fraction at 20 °C
- colour: residue-to-biochar conversion efficiency
           (t biochar tDM feedstock^-1)

The equations are reconstructed directly from physics.py:
    cc_daf = 0.93 - 0.92 * exp(-0.0042 * T)
    yield_daf_f = 0.1261 + 0.5391 * exp(-0.004 * T) + 0.002733 * lignin_f
    ash_char_f = ash_f / (ash_f + yield_daf_f)
    carbon_fraction_f = cc_daf * (1 - ash_char_f)
    yield_dm_f = yield_daf_f * (1 - ash_char_f) / (1 - ash_f)
    HC_f = exp(intercept_f + slope_f * T)
    F100_f = (93.8 - 18.0 * HC_f**4.85) / 100
"""


# =============================================================================
# Configuration
# =============================================================================

OUTPUT_DIR = Path(r"X:\user\orhalem\TIMER_CDR\timer-prism\cdr_output\figures")
OUTPUT_STEM = "biochar_feedstock_properties_low_high_temp_v2"

SAVE_DPI = 600
SHOW_FIGURE = True

FIGSIZE = (14.6, 6.8)

# Use the model-configured technologies closest to these nominal temperatures.
LOW_TEMP_TARGET_C = 300.0
HIGH_TEMP_TARGET_C = 500.0

CMAP = "viridis"

# Axis padding and annotation settings.
X_PAD_FRAC = 0.10
Y_PAD_FRAC = 0.12
MIN_X_PAD = 0.015
MIN_Y_PAD = 0.015

LABEL_FONTSIZE = 7.1
POINT_SIZE = 125
MAX_LABEL_MOVE_PIXELS = 120

FEEDSTOCK_NAMES = (
    "temperate_cereals",
    "rice",
    "maize",
    "tropical_cereals",
    "pulses",
    "roots_tubers",
    "oil_crops",
    "fibre",
    "fruit",
    "vegetables",
    "sugarcane",
    "biofuel_sugarcane",
    "biofuel_maize",
    "biofuel_woody",
    "biofuel_nwoody",
    "forestry",
)

FEEDSTOCK_DISPLAY = {
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
    "biofuel_sugarcane": "Biofuel sugarcane",
    "biofuel_maize": "Biofuel maize",
    "biofuel_woody": "Biofuel woody",
    "biofuel_nwoody": "Biofuel non-woody",
    "forestry": "Forestry residues",
}

EXCLUDED_FEEDSTOCKS = {
    "biofuel_sugarcane",
    "biofuel_maize",
    "biofuel_woody",
    "biofuel_nwoody",
}


# =============================================================================
# Helpers
# =============================================================================

def _load_dat_vector(path, expected_length=None):
    """
    Load a numerical vector from one of the .dat files used by the biochar
    module. The parser is tolerant to comments and simple two-column formats.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    numeric_rows = []

    with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line
            for marker in ("#", "//", "!"):
                if marker in line:
                    line = line.split(marker, 1)[0]

            line = (
                line.strip()
                .replace(",", " ")
                .replace(";", " ")
                .replace("\t", " ")
            )

            if not line:
                continue

            tokens = [tok for tok in line.split() if tok]
            row = []
            all_numeric = True

            for token in tokens:
                token_num = token.replace("D", "E").replace("d", "e")
                try:
                    row.append(float(token_num))
                except ValueError:
                    all_numeric = False
                    break

            if all_numeric and row:
                numeric_rows.append(row)

    if not numeric_rows:
        raise ValueError(f"No numeric payload could be read from {path}")

    if expected_length is None:
        return np.asarray(
            [x for row in numeric_rows for x in row],
            dtype=np.float64,
        )

    expected_length = int(expected_length)

    # Support index + value style tables.
    if len(numeric_rows) >= expected_length:
        tail_rows = numeric_rows[-expected_length:]
        if all(len(row) >= 2 for row in tail_rows):
            first_col = np.asarray([row[0] for row in tail_rows], dtype=np.float64)
            zero_based = np.arange(expected_length, dtype=np.float64)
            one_based = zero_based + 1.0

            if np.allclose(first_col, zero_based) or np.allclose(first_col, one_based):
                return np.asarray([row[-1] for row in tail_rows], dtype=np.float64)

    flat = np.asarray([x for row in numeric_rows for x in row], dtype=np.float64)

    if flat.size < expected_length:
        raise ValueError(
            f"Could read only {flat.size} numeric values from {path}, "
            f"but expected {expected_length}."
        )

    if flat.size > expected_length:
        values = flat[-expected_length:].copy()
    else:
        values = flat

    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values found in {path}")

    return values


def _technology_index(pyrolysis_temp, target_c):
    return int(np.nanargmin(np.abs(pyrolysis_temp - float(target_c))))


def _panel_title(letter, label, temp):
    return rf"$\bf{{{letter}.}}$ {label} ({temp:.0f} °C)"


def _compute_axis_limits(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    ymin = float(np.nanmin(y))
    ymax = float(np.nanmax(y))

    xrange = xmax - xmin
    yrange = ymax - ymin

    xpad = max(MIN_X_PAD, X_PAD_FRAC * xrange if xrange > 0 else MIN_X_PAD)
    ypad = max(MIN_Y_PAD, Y_PAD_FRAC * yrange if yrange > 0 else MIN_Y_PAD)

    xlim = (xmin - xpad, xmax + xpad)
    ylim = (max(0.0, ymin - ypad), min(1.0, ymax + ypad))

    return xlim, ylim


def _fallback_adjust_texts(ax, texts, x, y):
    """
    Lightweight fallback when adjustText is unavailable.

    It spreads labels vertically in data coordinates while keeping them close to
    their anchor points and inside the panel bounds.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xspan = xlim[1] - xlim[0]
    yspan = ylim[1] - ylim[0]

    if xspan <= 0 or yspan <= 0 or len(texts) <= 1:
        return

    order = np.argsort(y)
    min_dy = 0.035 * yspan

    adjusted_y = np.array(y, dtype=np.float64)[order].copy()

    for i in range(1, len(adjusted_y)):
        if adjusted_y[i] - adjusted_y[i - 1] < min_dy:
            adjusted_y[i] = adjusted_y[i - 1] + min_dy

    upper = ylim[1] - 0.015 * yspan
    if adjusted_y[-1] > upper:
        adjusted_y -= adjusted_y[-1] - upper

    lower = ylim[0] + 0.015 * yspan
    if adjusted_y[0] < lower:
        adjusted_y += lower - adjusted_y[0]

    x_offsets = np.array([0.016, 0.026, 0.020, 0.030, 0.014]) * xspan

    for rank, idx in enumerate(order):
        new_x = min(x[idx] + x_offsets[rank % len(x_offsets)], xlim[1] - 0.01 * xspan)
        new_y = np.clip(adjusted_y[rank], ylim[0] + 0.01 * yspan, ylim[1] - 0.01 * yspan)
        texts[idx].set_position((new_x, new_y))


def _place_non_overlapping_labels(ax, x, y, labels):
    """
    Create labels and automatically adjust them to minimise overlap.
    """
    texts = []
    for xi, yi, label in zip(x, y, labels):
        txt = ax.text(
            xi,
            yi,
            label,
            fontsize=LABEL_FONTSIZE,
            ha="left",
            va="bottom",
            zorder=4,
            clip_on=True,
        )
        texts.append(txt)

    if adjust_text is not None and len(texts) > 1:
        ax.figure.canvas.draw()
        try:
            adjust_text(
                texts,
                x=x,
                y=y,
                ax=ax,
                expand_points=(1.20, 1.35),
                expand_text=(1.12, 1.25),
                force_points=(0.20, 0.35),
                force_text=(0.22, 0.42),
                only_move={"points": "y", "text": "xy"},
                ensure_inside_axes=True,
                lim=400,
                arrowprops=dict(
                    arrowstyle="-",
                    color="0.45",
                    lw=0.4,
                    shrinkA=0,
                    shrinkB=0,
                    alpha=0.7,
                ),
            )
        except Exception:
            _fallback_adjust_texts(ax, texts, x, y)
    else:
        _fallback_adjust_texts(ax, texts, x, y)

    return texts


# =============================================================================
# Property calculations
# =============================================================================

def calculate_feedstock_properties():
    nfeed = len(BiocharFeedstock.coords)
    ntech = len(PyrolysisTechnology.coords)

    if nfeed != len(FEEDSTOCK_NAMES):
        raise ValueError(
            "BiocharFeedstock dimension length does not match FEEDSTOCK_NAMES."
        )

    lignin = _load_dat_vector(
        biochar_data_dir / "feedstock_lignin_content.dat",
        nfeed,
    )
    ash = _load_dat_vector(
        biochar_data_dir / "feedstock_ash_content.dat",
        nfeed,
    )
    hc_slope = _load_dat_vector(
        biochar_data_dir / "feedstock_HC_ratio_slope.dat",
        nfeed,
    )
    hc_intercept = _load_dat_vector(
        biochar_data_dir / "feedstock_HC_ratio_intercept.dat",
        nfeed,
    )
    pyrolysis_temp = _load_dat_vector(
        biochar_data_dir / "pyrolysis_temperature.dat",
        ntech,
    )

    # Carbon fraction and conversion efficiency.
    cc_daf = 0.93 - 0.92 * np.exp(-0.0042 * pyrolysis_temp[:, None])

    yield_daf = (
        0.1261
        + 0.5391 * np.exp(-0.004 * pyrolysis_temp[:, None])
        + 0.002733 * lignin[None, :]
    )

    ash_char = ash[None, :] / (ash[None, :] + yield_daf)

    carbon_fraction = cc_daf * (1.0 - ash_char)

    conversion_efficiency = np.divide(
        yield_daf * (1.0 - ash_char),
        (1.0 - ash[None, :]),
        out=np.full((ntech, nfeed), np.nan, dtype=np.float64),
        where=(1.0 - ash[None, :]) > 0.0,
    )

    # 100-year retained carbon fraction at 20 C.
    hc_ratio = np.exp(
        hc_intercept[None, :] + hc_slope[None, :] * pyrolysis_temp[:, None]
    )

    retained_fraction_100y = (93.8 - 18.0 * hc_ratio**4.85) / 100.0
    retained_fraction_100y = np.clip(retained_fraction_100y, 0.0, 1.0)

    rows = []
    for tech_idx, temp in enumerate(pyrolysis_temp):
        for feed_idx, feed in enumerate(FEEDSTOCK_NAMES):
            rows.append(
                {
                    "Technology index": tech_idx,
                    "Pyrolysis temperature (C)": float(temp),
                    "Feedstock": feed,
                    "Feedstock display": FEEDSTOCK_DISPLAY.get(feed, feed),
                    "Biochar carbon fraction (tC/t biochar)": float(
                        carbon_fraction[tech_idx, feed_idx]
                    ),
                    "Residue-to-biochar conversion efficiency (t biochar/tDM)": float(
                        conversion_efficiency[tech_idx, feed_idx]
                    ),
                    "100-year retained carbon fraction at 20C": float(
                        retained_fraction_100y[tech_idx, feed_idx]
                    ),
                }
            )

    df = pd.DataFrame(rows)

    # Exclude biofuel-derived residues.
    df = df.loc[
        ~df["Feedstock"].isin(EXCLUDED_FEEDSTOCKS)
    ].copy()

    return df, pyrolysis_temp


# =============================================================================
# Plotting
# =============================================================================

def _plot_panel(ax, df, title, norm, cmap):
    x = df["Biochar carbon fraction (tC/t biochar)"].to_numpy(dtype=np.float64)
    y = df["100-year retained carbon fraction at 20C"].to_numpy(dtype=np.float64)
    c = df["Residue-to-biochar conversion efficiency (t biochar/tDM)"].to_numpy(
        dtype=np.float64
    )
    labels = df["Feedstock display"].tolist()

    sc = ax.scatter(
        x,
        y,
        c=c,
        s=POINT_SIZE,
        cmap=cmap,
        norm=norm,
        edgecolor="0.25",
        linewidth=0.5,
        zorder=3,
    )

    xlim, ylim = _compute_axis_limits(x, y)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    _place_non_overlapping_labels(ax, x, y, labels)

    ax.set_xlabel("Biochar carbon fraction (tC t biochar$^{-1}$)")
    ax.set_ylabel("100-year retained carbon fraction at 20°C")
    ax.set_title(title, loc="left", fontweight="normal")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return sc


def plot_feedstock_property_figure(save=True, show=True):
    properties, pyrolysis_temp = calculate_feedstock_properties()

    low_idx = _technology_index(pyrolysis_temp, LOW_TEMP_TARGET_C)
    high_idx = _technology_index(pyrolysis_temp, HIGH_TEMP_TARGET_C)

    low_temp = float(pyrolysis_temp[low_idx])
    high_temp = float(pyrolysis_temp[high_idx])

    low_df = properties.loc[properties["Technology index"] == low_idx].copy()
    high_df = properties.loc[properties["Technology index"] == high_idx].copy()

    combined_eff = properties["Residue-to-biochar conversion efficiency (t biochar/tDM)"]
    norm = Normalize(
        vmin=float(np.nanmin(combined_eff)),
        vmax=float(np.nanmax(combined_eff)),
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=FIGSIZE,
        sharex=False,
        sharey=False,
    )

    sc = _plot_panel(
        axes[0],
        low_df,
        _panel_title("a", "Low-temperature pyrolysis", low_temp),
        norm=norm,
        cmap=CMAP,
    )
    _plot_panel(
        axes[1],
        high_df,
        _panel_title("b", "High-temperature pyrolysis", high_temp),
        norm=norm,
        cmap=CMAP,
    )

    cbar = fig.colorbar(
        sc,
        ax=axes,
        orientation="vertical",
        fraction=0.035,
        pad=0.03,
    )
    cbar.set_label(
        "Residue-to-biochar conversion efficiency\n"
        "(t biochar tDM feedstock$^{-1}$)",
        fontsize=9.0,
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.90,
        top=0.91,
        bottom=0.11,
        wspace=0.20,
    )

    outputs = {
        "figure": fig,
        "axes": axes,
        "properties": properties,
        "low_temp_df": low_df,
        "high_temp_df": high_df,
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        png_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
        pdf_path = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
        csv_path = OUTPUT_DIR / f"{OUTPUT_STEM}_properties.csv"

        fig.savefig(png_path, dpi=SAVE_DPI, bbox_inches="tight")
        fig.savefig(pdf_path, bbox_inches="tight")
        properties.to_csv(csv_path, index=False)

        outputs.update(
            {
                "png_path": png_path,
                "pdf_path": pdf_path,
                "csv_path": csv_path,
            }
        )

        print(f"Saved figure: {png_path}")
        print(f"Saved figure: {pdf_path}")
        print(f"Saved table:  {csv_path}")

    if show:
        plt.show()

    return outputs


if __name__ == "__main__":
    plot_feedstock_property_figure(
        save=True,
        show=SHOW_FIGURE,
    )

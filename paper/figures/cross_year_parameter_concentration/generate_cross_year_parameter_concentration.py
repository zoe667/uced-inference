#!/usr/bin/env python3
"""Plot cross-year parameter concentration at epsilon = 10%.

The visual design follows figure_poster/generate_temporal_shift_median_plot.py,
while the plotted summaries come only from the verified no-EI epsilon-10 set.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
INPUT = HERE / "eps10_parameter_concentration.csv"
OUTPUT_STEM = "fig_cross_year_parameter_concentration_eps10"

PARAMETERS = [
    ("MLT_Band", "(a) MLT band"),
    (
        "CHP_NonRetrofitted_Min_Power_300-660",
        "(b) CHP min. output, 300–660 MW",
    ),
    (
        "CHP_NonRetrofitted_Min_Power_0-300",
        "(c) CHP min. output, 0–300 MW",
    ),
]
YEARS = [2016, 2021]

TEXT = "#182B49"
BLUE = "#1f66b1"
ORANGE = "#d9654f"
BLUE_LIGHT = "#9bbfe3"
ORANGE_LIGHT = "#e9aa9c"
GRAY = "#5f6670"
GRID = "#d9d9d9"
COLORS = {2016: BLUE, 2021: ORANGE}
IQR_COLORS = {2016: BLUE_LIGHT, 2021: ORANGE_LIGHT}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Aptos", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.edgecolor": TEXT,
            "axes.linewidth": 0.78,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    expected = {
        "year",
        "epsilon_pct",
        "parameter",
        "p05",
        "p25",
        "median",
        "p75",
        "p95",
        "surrogate_selected_minimum",
        "n_points",
    }
    missing = expected.difference(data.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    if len(data) != 6 or set(data["year"]) != set(YEARS):
        raise ValueError("Expected exactly two years for each of three parameters")
    if set(data["epsilon_pct"]) != {10}:
        raise ValueError("This figure must use epsilon = 10% only")
    if data.groupby("parameter")["year"].nunique().to_dict() != {
        parameter: 2 for parameter, _ in PARAMETERS
    }:
        raise ValueError("Each plotted parameter must contain both years")
    return data


def draw_interval(
    ax: plt.Axes,
    x: float,
    row: pd.Series,
    color: str,
    iqr_color: str,
) -> None:
    # Thin capped line: descriptive 5th–95th percentile range.
    ax.errorbar(
        [x],
        [row["median"]],
        yerr=[
            [row["median"] - row["p05"]],
            [row["p95"] - row["median"]],
        ],
        fmt="none",
        ecolor=color,
        elinewidth=0.85,
        capsize=3.2,
        capthick=0.85,
        alpha=0.32,
        zorder=2,
    )

    # Thick central line: interquartile range.
    ax.vlines(
        x,
        row["p25"],
        row["p75"],
        color=iqr_color,
        linewidth=4.0,
        alpha=0.58,
        zorder=3,
    )

    # Filled circle: median.
    ax.scatter(
        [x],
        [row["median"]],
        s=47,
        color=color,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.82,
        zorder=5,
    )

    # Open diamond: surrogate-selected minimum.
    ax.scatter(
        [x],
        [row["surrogate_selected_minimum"]],
        s=31,
        marker="D",
        facecolor="white",
        edgecolor=color,
        linewidth=1.15,
        zorder=6,
    )


def draw_panel(ax: plt.Axes, rows: pd.DataFrame, title: str) -> None:
    rows = rows.set_index("year").loc[YEARS]
    x = np.array([0.0, 1.0])
    medians = rows["median"].to_numpy(float)

    plotted_values = rows[
        ["p05", "p25", "median", "p75", "p95", "surrogate_selected_minimum"]
    ].to_numpy(float)
    y_min = float(np.min(plotted_values))
    y_max = float(np.max(plotted_values))
    y_span = y_max - y_min
    pad = max(0.13 * y_span, 0.006 if y_max < 0.2 else 0.025)
    ax.set_ylim(y_min - pad, y_max + pad)

    # Undirected temporal median-to-median connector.
    ax.plot(
        x,
        medians,
        color=GRAY,
        linewidth=0.95,
        linestyle=(0, (4, 2.5)),
        dash_capstyle="round",
        zorder=1,
    )

    for xpos, year in zip(x, YEARS):
        draw_interval(
            ax,
            xpos,
            rows.loc[year],
            COLORS[year],
            IQR_COLORS[year],
        )

    delta = medians[1] - medians[0]
    ax.text(
        0.50,
        0.965,
        rf"$\Delta_{{\mathrm{{median}}}} = {delta:+.3f}$",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
        fontweight="semibold",
        color=TEXT,
        bbox={
            "boxstyle": "round,pad=0.16",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
        zorder=8,
    )

    ax.set_title(title, fontsize=8.2, pad=4.5, fontweight="normal")
    ax.set_xlim(-0.34, 1.34)
    ax.set_xticks(x, ["2016", "2021"])
    ax.tick_params(axis="both", labelsize=7.7, width=0.8, length=3.2)
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.76)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", linestyle="none", markerfacecolor=TEXT,
            markeredgecolor="white", markeredgewidth=0.8, markersize=6.2,
            label="Median",
        ),
        Line2D([0], [0], color="#a8b8c8", linewidth=4.0, alpha=0.65, label="IQR"),
        Line2D([0], [0], color=TEXT, linewidth=0.85, alpha=0.32, label="5–95% range"),
        Line2D(
            [0], [0], marker="D", linestyle="none", markerfacecolor="white",
            markeredgecolor=TEXT, markeredgewidth=1.0, markersize=5.2,
            label="Surrogate minimum",
        ),
    ]


def main() -> None:
    configure_style()
    data = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.72), dpi=300)
    fig.patch.set_alpha(0)

    for ax, (parameter, title) in zip(axes, PARAMETERS):
        draw_panel(ax, data.loc[data["parameter"] == parameter], title)

    axes[0].set_ylabel("Parameter value", fontsize=8.6, fontweight="semibold")
    legend = fig.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=7.0,
        bbox_to_anchor=(0.5, 0.068),
        columnspacing=1.35,
        handlelength=1.8,
        handletextpad=0.48,
    )
    fig.subplots_adjust(left=0.078, right=0.992, top=0.895, bottom=0.255, wspace=0.34)
    pdf = HERE / f"{OUTPUT_STEM}.pdf"
    png = HERE / f"{OUTPUT_STEM}.png"
    fig.savefig(pdf, transparent=True, bbox_inches="tight", pad_inches=0.035)
    fig.savefig(png, dpi=600, transparent=True, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)
    print(pdf)
    print(png)


if __name__ == "__main__":
    main()

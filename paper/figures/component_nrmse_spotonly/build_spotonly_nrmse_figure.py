#!/usr/bin/env python3
"""Build the SpotOnly reference NRMSE comparison for Results B.

The weekly UCED outputs contain 52 * 168 = 8,736 modeled hours.  The script
does not invent the missing 24 (or 48 in leap-year 2016) calendar hours.
Instead, it assigns the modeled hours sequentially to calendar months, clips
December to the remaining modeled hours, and compares monthly-average power:

    simulated monthly energy / modeled hours represented in that month
    historical monthly energy / calendar hours in that month

Generation and MLT NRMSE definitions reproduce the logic in
experiment/03_aggregate_and_analyze.jl.  Only implementation defects in that
script (undefined month-hour variables and integer parsing of named paths) are
avoided here.
"""

from __future__ import annotations

import calendar
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "uced-inference-matplotlib-cache")
)
os.environ.setdefault(
    "XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "uced-inference-xdg-cache")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
import numpy as np
import pandas as pd


def find_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "experiment").is_dir() and (parent / "model").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root")


ROOT = find_repo_root()
OUT = Path(__file__).resolve().parent
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
REGIONS = ["HL", "IME", "LN", "JL"]
TECH_RESOURCES = {
    "Coal": ["cogen_conventional_steam_coal", "conventional_steam_coal"],
    "Wind": ["onshore_wind_turbine"],
    "Solar": ["solar_photovoltaic"],
}
TECH_COLUMNS = {
    "Coal": "nrmse_coal",
    "Wind": "nrmse_wind",
    "Solar": "nrmse_solar",
    "MLT": "nrmse_mlt",
}
SCENARIOS = ["Reference", "Representative inferred", "Best LHS run"]
PLOT_SCENARIOS = ["Reference", "Representative inferred"]
SCENARIO_MARKERS = {
    "Reference": "o",
    "Representative inferred": "D",
    "Best LHS run": "s",
}
YEAR_COLORS = {2016: "#1768AC", 2021: "#D55E00"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


@dataclass(frozen=True)
class YearConfig:
    year: int
    spotonly_dir: Path
    noei_dir: Path
    base_data_dir: Path
    hist_data_dir: Path


CONFIGS = [
    YearConfig(
        year=2016,
        spotonly_dir=ROOT / "model" / "Batch" / "Results_ne_2016_SpotOnly",
        noei_dir=ROOT / "experiment" / "runs_2016_100",
        base_data_dir=ROOT / "data" / "ne_2016_maininput",
        hist_data_dir=ROOT / "data" / "hist_data_2016",
    ),
    YearConfig(
        year=2021,
        spotonly_dir=ROOT / "model" / "Batch" / "Results_ne_2021_SpotOnly",
        noei_dir=ROOT / "experiment" / "runs_2021_100",
        base_data_dir=ROOT / "data" / "ne_2021_maininput",
        hist_data_dir=ROOT / "data" / "hist_data_2021",
    ),
]


def modeled_month_slices(year: int, total_hours: int) -> tuple[dict[str, slice], dict[str, int]]:
    """Assign sequential modeled hours to calendar months, clipping December."""
    calendar_hours = {
        month: calendar.monthrange(year, i + 1)[1] * 24
        for i, month in enumerate(MONTHS)
    }
    slices: dict[str, slice] = {}
    modeled_hours: dict[str, int] = {}
    start = 0
    for month in MONTHS:
        stop = min(start + calendar_hours[month], total_hours)
        slices[month] = slice(start, stop)
        modeled_hours[month] = max(stop - start, 0)
        start = stop
    if start != total_hours:
        raise AssertionError(f"Only assigned {start} of {total_hours} modeled hours")
    if any(modeled_hours[m] == 0 for m in MONTHS):
        raise AssertionError("At least one month received no modeled hours")
    return slices, modeled_hours


def read_weekly_outputs(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Horizontally concatenate the 52 weekly dispatch and flow files."""
    dispatch_meta = ["Index", "Zone", "Region", "Resource"]
    flow_meta = ["Index", "Path"]
    dispatch_base: pd.DataFrame | None = None
    flow_base: pd.DataFrame | None = None
    dispatch_blocks: list[pd.DataFrame] = []
    flow_blocks: list[pd.DataFrame] = []

    for week in range(1, 53):
        dispatch_path = result_dir / str(week) / "vGENDISPATCH_results.csv"
        flow_path = result_dir / str(week) / "vFLOW_results.csv"
        if not dispatch_path.is_file() or not flow_path.is_file():
            raise FileNotFoundError(f"Missing SpotOnly output in week {week}: {result_dir}")

        dispatch = pd.read_csv(dispatch_path)
        flow = pd.read_csv(flow_path)
        hour_cols = [f"x{i}" for i in range(1, 169)]
        if dispatch.columns.tolist() != dispatch_meta + hour_cols:
            raise ValueError(f"Unexpected dispatch schema: {dispatch_path}")
        if flow.columns.tolist() != flow_meta + hour_cols:
            raise ValueError(f"Unexpected flow schema: {flow_path}")

        if dispatch_base is None:
            dispatch_base = dispatch[dispatch_meta].copy()
            flow_base = flow[flow_meta].copy()
        else:
            if not dispatch[dispatch_meta].equals(dispatch_base):
                raise ValueError(f"Dispatch row identities/order changed in week {week}")
            if not flow[flow_meta].equals(flow_base):
                raise ValueError(f"Flow row identities/order changed in week {week}")

        first_hour = (week - 1) * 168 + 1
        renamed = {f"x{i}": f"h{first_hour + i - 1}" for i in range(1, 169)}
        dispatch_blocks.append(dispatch[hour_cols].rename(columns=renamed))
        flow_blocks.append(flow[hour_cols].rename(columns=renamed))

    assert dispatch_base is not None and flow_base is not None
    dispatch_year = pd.concat([dispatch_base, *dispatch_blocks], axis=1)
    flow_year = pd.concat([flow_base, *flow_blocks], axis=1)
    expected_hours = [f"h{i}" for i in range(1, 8737)]
    if dispatch_year.columns.tolist() != dispatch_meta + expected_hours:
        raise AssertionError("Dispatch extraction does not contain h1..h8736 exactly once")
    if flow_year.columns.tolist() != flow_meta + expected_hours:
        raise AssertionError("Flow extraction does not contain h1..h8736 exactly once")
    return dispatch_year, flow_year


def monthly_average_outputs(
    year: int,
    dispatch: pd.DataFrame,
    flow: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    """Aggregate hourly MW to monthly-average GW."""
    hour_cols = [f"h{i}" for i in range(1, 8737)]
    slices, modeled_hours = modeled_month_slices(year, len(hour_cols))
    outputs: dict[str, pd.DataFrame] = {}

    for tech, resources in TECH_RESOURCES.items():
        rows = []
        for region in REGIONS:
            mask = dispatch["Region"].eq(region) & dispatch["Resource"].isin(resources)
            if not mask.any():
                raise ValueError(f"No {tech} dispatch found for {year} {region}")
            hourly_mw = dispatch.loc[mask, hour_cols].to_numpy(float).sum(axis=0)
            row = {"Region": region}
            for month in MONTHS:
                block = hourly_mw[slices[month]]
                row[month] = block.mean() / 1000.0
            rows.append(row)
        outputs[tech] = pd.DataFrame(rows)

    flow_rows = []
    for _, source in flow.iterrows():
        hourly_mw = source[hour_cols].to_numpy(float)
        row = {"Path": str(source["Path"])}
        for month in MONTHS:
            row[month] = hourly_mw[slices[month]].mean() / 1000.0
        flow_rows.append(row)
    monthly_flow = pd.DataFrame(flow_rows).sort_values("Path").reset_index(drop=True)
    return outputs, monthly_flow, modeled_hours


def historical_monthly_average(config: YearConfig) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, int]]:
    """Convert historical monthly energy totals to monthly-average GW."""
    calendar_hours = {
        month: calendar.monthrange(config.year, i + 1)[1] * 24
        for i, month in enumerate(MONTHS)
    }
    generation = {}
    for tech, stem in (("Coal", "coal"), ("Wind", "wind"), ("Solar", "solar")):
        frame = pd.read_csv(
            config.hist_data_dir / f"hist_{stem}_gen_monthly.csv",
            encoding="utf-8-sig",
        )
        frame = frame.rename(columns={frame.columns[0]: "Region"})
        for month in MONTHS:
            frame[month] = frame[month].astype(float) / calendar_hours[month]
        generation[tech] = frame

    mlt = pd.read_csv(
        config.hist_data_dir / "hist_mlt_trans_monthly.csv",
        encoding="utf-8-sig",
    )
    mlt = mlt.rename(columns={mlt.columns[0]: "Path"})
    for month in MONTHS:
        # Historical MLT is MWh; /1000 -> GWh; /hours -> average GW.
        mlt[month] = mlt[month].astype(float) / 1000.0 / calendar_hours[month]
    return generation, mlt, calendar_hours


def load_capacities(config: YearConfig) -> dict[str, dict[str, float]]:
    """Reproduce 03: sum Cap_Size by technology/region and convert MW to GW."""
    generators = pd.read_csv(config.base_data_dir / "Generators_data.csv")
    capacities: dict[str, dict[str, float]] = {}
    for tech, resources in TECH_RESOURCES.items():
        capacities[tech] = {}
        for region in REGIONS:
            mask = generators["region"].eq(region) & generators["technology"].isin(resources)
            capacity = generators.loc[mask, "Cap_Size"].astype(float).sum() / 1000.0
            if capacity <= 0:
                raise ValueError(f"Non-positive {tech} capacity for {config.year} {region}")
            capacities[tech][region] = capacity
    return capacities


def generation_nrmse(
    simulated: pd.DataFrame,
    historical: pd.DataFrame,
    capacities: dict[str, float],
) -> tuple[float, list[dict[str, float]]]:
    """Capacity-normalized regional RMSE, averaged with four equal weights."""
    regional = []
    for region in REGIONS:
        sim = simulated.loc[simulated["Region"].eq(region), MONTHS]
        hist = historical.loc[historical["Region"].eq(region), MONTHS]
        if len(sim) != 1 or len(hist) != 1:
            raise ValueError(f"Expected one simulated/historical row for {region}")
        sim_values = sim.iloc[0].to_numpy(float)
        hist_values = hist.iloc[0].to_numpy(float)
        rmse = float(np.sqrt(np.mean((sim_values - hist_values) ** 2)))
        nrmse = rmse / capacities[region]
        regional.append({"region": region, "rmse_GW": rmse, "nrmse": nrmse})
    return float(np.mean([x["nrmse"] for x in regional])), regional


def mlt_nrmse(
    year: int,
    simulated: pd.DataFrame,
    historical: pd.DataFrame,
) -> tuple[float, float, list[dict[str, float]]]:
    """Mean path RMSE divided by mean absolute historical monthly flow."""
    baseline = float(np.mean(np.abs(historical[MONTHS].to_numpy(float))))
    excluded = {"IME_to_SD"} if year == 2016 else set()
    path_rows = []
    for path in simulated["Path"]:
        if path in excluded:
            continue
        sim = simulated.loc[simulated["Path"].eq(path), MONTHS]
        hist = historical.loc[historical["Path"].eq(path), MONTHS]
        if len(hist) != 1:
            raise ValueError(f"No unique historical MLT path for {year}: {path}")
        rmse = float(np.sqrt(np.mean(
            (sim.iloc[0].to_numpy(float) - hist.iloc[0].to_numpy(float)) ** 2
        )))
        path_rows.append({"path": path, "rmse_GW": rmse})
    if not path_rows:
        raise ValueError("No matched MLT paths")
    global_rmse = float(np.mean([x["rmse_GW"] for x in path_rows]))
    return global_rmse / baseline, baseline, path_rows


def write_year_outputs(
    config: YearConfig,
    dispatch: pd.DataFrame,
    flow: pd.DataFrame,
    monthly_generation: dict[str, pd.DataFrame],
    monthly_flow: pd.DataFrame,
    modeled_hours: dict[str, int],
    calendar_hours: dict[str, int],
    summary: pd.DataFrame,
    regional_details: pd.DataFrame,
) -> None:
    year_dir = OUT / str(config.year)
    year_dir.mkdir(parents=True, exist_ok=True)
    dispatch.to_csv(year_dir / "spotonly_dispatch_hourly_wide.csv", index=False, float_format="%.10g")
    flow.to_csv(year_dir / "spotonly_flow_hourly_wide.csv", index=False, float_format="%.10g")
    for tech, frame in monthly_generation.items():
        frame.to_csv(
            year_dir / f"spotonly_monthly_average_{tech.lower()}_gen.csv",
            index=False,
            float_format="%.10g",
        )
    monthly_flow.to_csv(
        year_dir / "spotonly_monthly_average_mlt_flow.csv",
        index=False,
        float_format="%.10g",
    )
    pd.DataFrame({
        "month": MONTHS,
        "modeled_hours": [modeled_hours[m] for m in MONTHS],
        "calendar_hours": [calendar_hours[m] for m in MONTHS],
    }).to_csv(year_dir / "month_hour_audit.csv", index=False)
    summary.to_csv(year_dir / "spotonly_nrmse_summary.csv", index=False, float_format="%.12g")
    regional_details.to_csv(
        year_dir / "spotonly_nrmse_diagnostics.csv", index=False, float_format="%.12g"
    )


def load_comparison_cases(config: YearConfig, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join the new reference to the no-EI medoid and minimum-loss LHS run."""
    lhs = pd.read_csv(config.noei_dir / "final_performance_summary.csv")
    best = lhs.sort_values("total_loss", ascending=True).iloc[0]
    medoid_path = (
        config.noei_dir / "back_check_archive" /
        "backcheck_medoid_eps10" / "backcheck_result.json"
    )
    medoid_record = json.loads(medoid_path.read_text(encoding="utf-8"))
    if medoid_record.get("role") != "medoid":
        raise ValueError(f"Unexpected medoid record: {medoid_path}")
    medoid = medoid_record["result"]

    rows = []
    for tech, column in TECH_COLUMNS.items():
        reference_value = float(reference.loc[reference["technology"].eq(tech), "nrmse"].iloc[0])
        rows.extend([
            {
                "year": config.year,
                "technology": tech,
                "scenario": "Reference",
                "nrmse": reference_value,
                "run_id_or_role": "SpotOnly",
            },
            {
                "year": config.year,
                "technology": tech,
                "scenario": "Representative inferred",
                "nrmse": float(medoid[f"raw_{tech.lower()}_error"]),
                "run_id_or_role": "medoid (S10%)",
            },
            {
                "year": config.year,
                "technology": tech,
                "scenario": "Best LHS run",
                "nrmse": float(best[column]),
                "run_id_or_role": f"run {int(best['run_id'])}",
            },
        ])

    reference_loss = 0.25 * float(np.sum(reference["nrmse"].to_numpy(float) ** 2))
    meta = {
        "year": config.year,
        "reference_total_L": reference_loss,
        "representative_role": "medoid (S10%)",
        "representative_original_uced_L": float(medoid["performance_score"]),
        "best_lhs_run_id": int(best["run_id"]),
        "best_lhs_total_L": float(best["total_loss"]),
        "modeled_hours": 8736,
    }
    return pd.DataFrame(rows), meta


def relative_change_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return representative change relative to the SpotOnly reference."""
    rows = []
    for (year, technology), group in data.groupby(["year", "technology"]):
        values = group.set_index("scenario")["nrmse"]
        reference = float(values["Reference"])
        representative = float(values["Representative inferred"])
        rows.append({
            "year": int(year),
            "technology": technology,
            "reference_nrmse": reference,
            "representative_nrmse": representative,
            "change_relative_to_reference_pct":
                (representative - reference) / reference * 100.0,
        })
    return pd.DataFrame(rows).sort_values(["year", "technology"]).reset_index(drop=True)


def plot_cleveland_legacy(data: pd.DataFrame, output: Path) -> None:
    """Science-style relative-change plot; zero is the SpotOnly reference."""
    tech_order = ["Coal", "Wind", "Solar", "MLT"]
    years = [2016, 2021]
    base_y = {tech: len(tech_order) - 1 - i for i, tech in enumerate(tech_order)}
    offsets = {2016: 0.14, 2021: -0.14}
    year_markers = {2016: "o", 2021: "D"}

    fig, ax = plt.subplots(figsize=(3.45, 2.55))
    fig.subplots_adjust(left=0.19, right=0.975, bottom=0.20, top=0.86)
    for year in years:
        color = YEAR_COLORS[year]
        for tech in tech_order:
            subset = data[(data["year"] == year) & (data["technology"] == tech)]
            if len(subset) != 1:
                raise ValueError(f"Expected one relative-change row for {year} {tech}")
            change = float(subset.iloc[0]["change_relative_to_reference_pct"])
            y = base_y[tech] + offsets[year]
            ax.plot(
                [0, change], [y, y],
                color=color, linewidth=0.8, alpha=0.38,
                solid_capstyle="round", zorder=1,
            )
            ax.scatter(
                change, y,
                s=28,
                marker=year_markers[year],
                color=color,
                alpha=0.84,
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
            label_offset = 2.0
            label_x = change - label_offset if change < 0 else change + label_offset
            ax.text(
                label_x,
                y,
                f"{change:+.0f}%".replace("+0%", "0%"),
                ha="right" if change < 0 else "left",
                va="center",
                fontsize=5.8,
                color="#3F3F3F",
                zorder=4,
            )

    # Horizontal category separators plus one explicit zero reference line.
    for y in [0.5, 1.5, 2.5]:
        ax.axhline(y, color="#D8D8D8", linewidth=0.45,
                   linestyle=(0, (2, 2)), alpha=0.75, zorder=0)
    ax.axvline(0, color="#555555", linewidth=0.7, alpha=0.75, zorder=0)
    ax.set_xlabel("Change in NRMSE relative to reference (%)", fontsize=7.0)
    ax.set_yticks([base_y[t] for t in tech_order])
    ax.set_yticklabels(tech_order, fontsize=7.2)
    ax.tick_params(axis="x", labelsize=6.8)
    ax.set_ylim(-0.45, 3.45)
    ax.set_xlim(-102, 10)
    ax.set_xticks([-100, -80, -60, -40, -20, 0])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.text(
        0.995, 1.015, "Negative = lower mismatch",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.6, color="#555555",
    )

    handles = [
        Line2D([0], [0], marker=year_markers[2016], color=YEAR_COLORS[2016],
               linewidth=0.8, alpha=0.78, markeredgecolor="white",
               markersize=4.8, label="2016"),
        Line2D([0], [0], marker=year_markers[2021], color=YEAR_COLORS[2021],
               linewidth=0.8, alpha=0.78, markeredgecolor="white",
               markersize=4.8, label="2021"),
    ]
    fig.legend(
        handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
        ncol=2, frameon=False, fontsize=6.1, handletextpad=0.28,
        columnspacing=0.9, borderaxespad=0.0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_cleveland(data: pd.DataFrame, output: Path) -> None:
    """Compact journal panel: position encodes change; aligned columns give values."""
    from matplotlib.ticker import FuncFormatter

    fig = plt.figure(figsize=(3.1, 1.82))
    ax = fig.add_axes([0.16, 0.13, 0.53, 0.74])
    values = fig.add_axes([0.715, 0.13, 0.275, 0.74], sharey=ax)
    order = ["Coal", "Wind", "Solar", "MLT"]
    for i in range(4):
        if i % 2 == 0:
            for panel in (ax, values):
                panel.axhspan(i - 0.43, i + 0.43, color="#F4F5F6", lw=0, zorder=0)
    ax.axvline(0, color="#969696", lw=0.6, zorder=1)
    for year, offset, marker, xpos in [(2016, -0.12, "o", 0.23), (2021, 0.12, "D", 0.76)]:
        color = YEAR_COLORS[year]
        for i, tech in enumerate(order):
            row = data.loc[(data.year == year) & (data.technology == tech)]
            if len(row) != 1:
                raise ValueError(f"Expected one row: {year} {tech}")
            change = float(row.iloc[0].change_relative_to_reference_pct)
            ax.scatter(change, i + offset, s=19, marker=marker, color=color,
                       edgecolor="white", linewidth=0.35, zorder=3)
            values.text(xpos, i, f"{change:+.1f}%", ha="center", va="center",
                        color=color, fontsize=7.0)
        values.text(xpos, 1.09, str(year), transform=values.transAxes,
                    ha="center", color=color, fontsize=7.3, fontweight="bold")
    ax.set_ylim(3.48, -0.48)
    ax.set_xlim(-100, 8)
    ax.set_yticks(range(4), order, fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_xticks([-100, -50, 0])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax.tick_params(axis="x", labelsize=7, width=0.55, length=2.5)
    ax.set_xlabel("")
    for spine in ["top", "left", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.55)
    ax.spines["bottom"].set_color("#777777")
    values.set_xlim(0, 1)
    values.set_axis_off()
    ax.text(0, 1.09, "Lower mismatch  \u2190", transform=ax.transAxes,
            fontsize=7.0, color="#555555")
    # Preserve the tight horizontal crop while leaving a small caption-safe
    # buffer below the x-axis labels.
    fig.canvas.draw()
    tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    caption_safe_bbox = Bbox.from_extents(
        tight_bbox.x0,
        tight_bbox.y0 - 0.14,
        tight_bbox.x1,
        tight_bbox.y1,
    )
    fig.savefig(output, dpi=600, facecolor="white", bbox_inches=caption_safe_bbox)
    fig.savefig(output.with_suffix(".pdf"), facecolor="white",
                bbox_inches=caption_safe_bbox)
    plt.close(fig)


def process_reference(config: YearConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    dispatch, flow = read_weekly_outputs(config.spotonly_dir)
    monthly_gen, monthly_flow, modeled_hours = monthly_average_outputs(
        config.year, dispatch, flow
    )
    hist_gen, hist_flow, calendar_hours = historical_monthly_average(config)
    capacities = load_capacities(config)

    summary_rows = []
    diagnostic_rows = []
    for tech in ("Coal", "Wind", "Solar"):
        value, regional = generation_nrmse(
            monthly_gen[tech], hist_gen[tech], capacities[tech]
        )
        summary_rows.append({"year": config.year, "technology": tech, "nrmse": value})
        for row in regional:
            diagnostic_rows.append({"year": config.year, "technology": tech, **row})

    mlt_value, mlt_baseline, paths = mlt_nrmse(config.year, monthly_flow, hist_flow)
    summary_rows.append({"year": config.year, "technology": "MLT", "nrmse": mlt_value})
    for row in paths:
        diagnostic_rows.append({
            "year": config.year,
            "technology": "MLT",
            "region": row["path"],
            "rmse_GW": row["rmse_GW"],
            "nrmse": row["rmse_GW"] / mlt_baseline,
        })

    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    write_year_outputs(
        config, dispatch, flow, monthly_gen, monthly_flow,
        modeled_hours, calendar_hours, summary, diagnostics,
    )
    return summary, diagnostics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison_frames = []
    meta_rows = []
    for config in CONFIGS:
        reference, _ = process_reference(config)
        comparison, meta = load_comparison_cases(config, reference)
        comparison_frames.append(comparison)
        meta_rows.append(meta)

    comparison = pd.concat(comparison_frames, ignore_index=True)
    comparison.to_csv(
        OUT / "tech_nrmse_cleveland_data.csv", index=False, float_format="%.12g"
    )
    relative = relative_change_data(comparison)
    relative.to_csv(
        OUT / "tech_nrmse_relative_change.csv", index=False, float_format="%.12g"
    )
    pd.DataFrame(meta_rows).to_csv(
        OUT / "tech_nrmse_cleveland_meta.csv", index=False, float_format="%.12g"
    )
    plot_cleveland(relative, OUT / "fig_tech_nrmse_cleveland_spotonly.png")

    manifest = {
        "modeled_hours_per_year": 8736,
        "missing_calendar_hours_not_imputed": True,
        "reference_source": {
            str(c.year): str(c.spotonly_dir.relative_to(ROOT)) for c in CONFIGS
        },
        "inference_source": {str(c.year): str(c.noei_dir.relative_to(ROOT)) for c in CONFIGS},
        "representative_case": "medoid of reduced-GP S_10%, original-UCED revalidation",
        "best_lhs_rule": "minimum total_loss in final_performance_summary.csv",
        "comparison_data_scenarios": SCENARIOS,
        "figure_quantity": "representative NRMSE change relative to SpotOnly reference (%)",
        "figure_scenarios": PLOT_SCENARIOS,
        "vertical_grid_lines": False,
    }
    (OUT / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(comparison.pivot_table(
        index=["year", "technology"], columns="scenario", values="nrmse"
    ).to_string())
    print("\n", pd.DataFrame(meta_rows).to_string(index=False), sep="")
    print(f"\nSaved outputs to {OUT}")


if __name__ == "__main__":
    main()

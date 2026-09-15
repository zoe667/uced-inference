# SpotOnly reference and component-level NRMSE figure

This folder contains the new reference-case extraction, monthly aggregation,
NRMSE calculation, and Cleveland plot for Results B. The two SpotOnly model
years are compared with the S_10% medoid revalidation and the lowest-loss LHS
run from the corresponding no-EI experiment.

## Reproduce

From the repository root:

```bash
python3 paper/figures/component_nrmse_spotonly/build_spotonly_nrmse_figure.py
```

No new UCED simulations are launched. The script reads the completed results folder.

## NRMSE definitions

Coal, wind, and solar follow `03_aggregate_and_analyze.jl`:

1. compute the 12-month RMSE of monthly-average GW separately for each region;
2. divide each regional RMSE by that technology's regional installed capacity;
3. average the four regional NRMSE values with equal 0.25 weights.

MLT also follows `03`:

1. compute a 12-month RMSE for each matched transmission path;
2. average the path RMSE values;
3. divide by the mean absolute historical monthly-average flow across the full
   historical MLT table.

The aggregate loss reported in the metadata is
`0.25 * (coal^2 + wind^2 + solar^2 + MLT^2)`.

## Figure cases

- `Reference` (circle): newly calculated SpotOnly result.
- `Representative inferred` (diamond): original-UCED result for the medoid of
  the reduced-GP S_10% set.
- `Best LHS run` remains in the exported comparison data for auditability, but
  is intentionally omitted from the final figure.

Inference inputs are represented by the archived medoid UCED results in `results/2016` and `results/2021`.

The final figure reports the percentage change from `Reference` to
`Representative inferred`, with the reference fixed at zero. Negative values
mean lower mismatch. The compact 3.5-by-2.15-inch figure uses blue circles
for 2016 and orange diamonds for 2021, a zero-reference line, and aligned
one-decimal percentage columns. Alternating faint row bands guide reading;
long connectors and point-adjacent labels are omitted. These values are
relative changes, not absolute NRMSE percentages or uncertainty intervals.

## Outputs

At the folder root:

- `fig_tech_nrmse_cleveland_spotonly.pdf/.png`
- `tech_nrmse_cleveland_data.csv`
- `tech_nrmse_relative_change.csv`
- `tech_nrmse_cleveland_meta.csv`

Under `2016/` and `2021/`:

- concatenated 8,736-hour dispatch and flow tables;
- monthly-average coal, wind, solar, and MLT tables;
- month-hour audit;
- NRMSE summary and regional/path diagnostics.

## Replot from frozen data

The repository excludes the large weekly SpotOnly and inferred-run outputs. Recreate the final panel directly from the frozen plotting table with:

```bash
python3 replot_from_frozen_data.py
```

The full builder is retained to document how the plotting table was constructed.

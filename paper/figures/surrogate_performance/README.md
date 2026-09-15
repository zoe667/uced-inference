# Surrogate performance figure group

This folder contains the surrogate-performance plotting data, figures, and scripts.

## Rebuild

From the repository root:

```bash
julia --project=. paper/figures/surrogate_performance/build_surrogate_results_package.jl
Rscript paper/figures/surrogate_performance/plot_surrogate_performance.R
```

The Julia step reads the frozen records in `results/2016` and
`results/2021` and regenerates the three CSV files. The R step
reads only those local CSV files and regenerates all PDF and PNG figures in
this folder.

## Files

- `surrogate_validation_summary.csv`: CV, held-out, and full-versus-reduced metrics.
- `surrogate_holdout_predictions.csv`: original-UCED held-out losses and final reduced-GP predictions.
- `ard_fold_screening.csv`: fold-level ARD length scales and retain/exclude decisions.
- `fig_gp_holdout_validation.*`: two-panel wide held-out parity plot.
- `fig_surrogate_performance_active_parameters_ieee.*`: compact two-column figure with four horizontal panels: two held-out parity plots and two year-specific ARD decision summaries.
- `supp_ard_fold_stability.*`: supplementary fold-stability figure.

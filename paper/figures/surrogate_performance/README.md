# Surrogate performance figure group

This folder contains every asset used for the Results subsection `Surrogate Performance and Relevant Dimensions`.

## Rebuild

From the repository root:

```bash
julia --project=. paper_assets/noEI/surrogate_performance/build_surrogate_results_package.jl
Rscript paper_assets/noEI/surrogate_performance/plot_surrogate_performance.R
```

The Julia step reads `experiment/runs_2016_100` and
`experiment/runs_2021_100` and regenerates the three CSV files. The R step
reads only those local CSV files and regenerates all PDF and PNG figures in
this folder.

## Files

- `surrogate_validation_summary.csv`: CV, held-out, and full-versus-reduced metrics.
- `surrogate_holdout_predictions.csv`: original-UCED held-out losses and final reduced-GP predictions.
- `ard_fold_screening.csv`: fold-level ARD length scales and retain/exclude decisions.
- `fig_gp_holdout_validation.*`: two-panel wide held-out parity plot.
- `fig_gp_holdout_validation_stacked.*`: one-column stacked alternative.
- `fig_surrogate_performance_active_parameters_ieee.*`: compact IEEE
  two-column figure with four horizontal panels: two held-out parity plots and
  two year-specific ARD decision summaries.
- `supp_ard_fold_stability.*`: supplementary fold-stability figure.
- `surrogate_performance_subsection.tex`: three-paragraph manuscript draft and main-figure caption.

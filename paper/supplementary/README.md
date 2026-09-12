# Revised supplementary material — S1–S5 working revision

`Supplementary_Methodological_Framework_Revised.pdf` contains a compact forward formulation, the candidate domain and preliminary-screening account, final 100-run GP/ARD validation, and updated low-mismatch-region diagnostics. S5 reports six original-UCED evaluations for each of 2016 and 2021, including surrogate best, with GP predictions recomputed at the exact evaluated parameters and component NRMSE values. All twelve points match the final candidate records and satisfy their year-specific S10% cutoffs. S6 briefly describes reproduction materials.

The final 100-point design and 80/20 development/test split are used consistently. The alternative-bound and cross-fitted ARD robustness subsection uses the completed `04b` outputs from the final `_100` folders.

## Overleaf

Upload `Supplementary_Overleaf.zip` into a new Overleaf project and select `Supplementary_Methodological_Framework_Revised.tex` as the main document. The archive includes all referenced figure PDFs and generated table sources. Compile with pdfLaTeX or XeLaTeX.

## Reproduce the report

From the repository root, with the repository Julia environment and Python's numpy/matplotlib installed:

```sh
julia --project=. paper/supplementary/scripts/extract_gp_validation.jl
julia --project=. paper/supplementary/scripts/refresh_dense_predictions.jl
python3 paper/supplementary/scripts/build_tables_figures.py
julia --project=. paper/supplementary/scripts/extract_revalidation.jl
python3 paper/supplementary/scripts/build_revalidation.py
```

Then compile the main `.tex` in this directory twice with pdfLaTeX, or once with Tectonic. No GP optimization or original UCED solves are performed by these reporting scripts.

## Files

- `tables/`: auto-generated numerical tables and result macros.
- `figures/`: scientific figures, in PDF and PNG.
- `data/`: holdout predictions, full/reduced metrics, ARD rows, refreshed 20,000-point predictions for each year, region summaries, correlations, and source hashes.
- `scripts/`: read-only model extraction and report generation.
- `MAIN_PAPER_SYNC_NOTES.md`: author-facing list of exact main-paper replacements and the final `04b` interpretation.

Source directories are `results/2016` and `results/2021`. The package recomputes predictions at the stored dense-pool coordinates. Historical experiment outputs and the main manuscript are maintained outside this release repository.

The historical method PDF supplied by the user provides the qualitative preliminary three-month screening account; no new numerical screening analysis is claimed. Historical design boundaries come from frozen year-specific domain files, which document their reconstruction provenance.

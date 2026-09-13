# UCED Inference

[![CI](https://github.com/zoe667/uced-inference/actions/workflows/ci.yml/badge.svg)](https://github.com/zoe667/uced-inference/actions/workflows/ci.yml)

This repository contains the code and data package for surrogate-assisted inverse inference of policy-shaped operating parameters in a unit commitment and economic dispatch (UCED) model. It includes the model code, the final 100-point experimental designs for 2016 and 2021, frozen surrogate and revalidation outputs, scripts used to produce the reported figures and supplementary material, and lightweight verification checks.

## Quick start

The primary verification uses only Python's standard library. It does not refit a Gaussian process or solve the UCED model.

```bash
git clone https://github.com/zoe667/uced-inference.git
cd uced-inference
python3 reproduction/verify_results.py
```

Expected final line:

```text
PASS: all archived 2016 and 2021 results are internally consistent.
```

The check verifies the 100-point designs, frozen 80/20 splits, loss construction, reported GP metrics, active parameter sets, low-mismatch region sizes, and all twelve original-UCED revalidations. Immutable data, result, and paper artifacts also have SHA-256 entries in `reproduction/checksums.sha256`.

## Repository contents

- `model/`: Julia UCED model and output processing.
- `experiment/`: canonical scripts for design generation, batch UCED simulation, aggregation, GP construction, ARD robustness, search, and revalidation.
- `data/`: model inputs and observed monthly targets for 2016 and 2021.
- `results/`: frozen outputs from the final 100-point designs.
- `paper/figures/`: final paper figures, plotting data, and plotting scripts.
- `paper/supplementary/`: revised supplementary PDF, LaTeX source, generated tables, and reporting scripts.
- `reproduction/`: verification and reproduction entry points.

## Software environment

The analyses were finalized with Julia 1.11.3. Install the locked Julia environment with:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Python reporting scripts use the packages in `requirements.txt`:

```bash
python3 -m pip install -r requirements.txt
```

The held-out result verifier itself has no third-party Python dependencies.

## Reproduce surrogate fitting and search

Stages 04, 04b, and 05 can be rerun from the frozen final performance summaries without UCED solves. The wrapper copies archived inputs to a disposable work directory before writing new outputs:

```bash
bash reproduction/reproduce_surrogate.sh 2016
bash reproduction/reproduce_surrogate.sh 2021
```

Outputs are written under `experiment/reproduction_<year>/`; the frozen files in `results/` are never overwritten. GP optimization can produce small platform-level numerical differences.

## Reproduce figures

```bash
bash reproduction/reproduce_figures.sh
```

This rebuilds figures from frozen plotting tables. The surrogate performance panel requires base R; the other released plotting scripts require Python. Final PDF and PNG files are included for direct inspection.

To rebuild the supplementary reporting data and figures from the frozen Julia model objects:

```bash
julia --project=. paper/supplementary/scripts/extract_gp_validation.jl
julia --project=. paper/supplementary/scripts/refresh_dense_predictions.jl
python3 paper/supplementary/scripts/build_tables_figures.py
julia --project=. paper/supplementary/scripts/extract_revalidation.jl
python3 paper/supplementary/scripts/build_revalidation.py
```

These reporting commands perform no GP optimization and no UCED solve.

## Optional UCED smoke test

The full UCED model uses Gurobi and contains nonconvex quadratic terms. A working Gurobi installation and license are therefore required. A one-design, one-week execution test is provided:

```bash
bash reproduction/smoke_uced.sh 2021
```

The complete experiment uses 100 parameter designs and 52 modeled weeks for each year. It is intended for a multicore workstation or computing cluster and is not part of the automated CI job.

## Workflow and paper mapping

| Paper output | Source stage | Frozen artifact |
|---|---|---|
| Full and reduced GP validation | `04_build_surrogate.jl` | `results/<year>/validation_summary.json` and CV prediction files |
| ARD screening robustness | `04b_active_parameter_robustness.jl` | `results/<year>/active_selection_robustness/` |
| Low-mismatch parameter region | `05_surrogate_search.jl` | dense search and search summary files |
| Original-UCED revalidation | `05b_run_revalidation_candidate.jl` | `results/<year>/back_check_archive/` |
| Main figures | plotting scripts | `paper/figures/` |
| Supplementary Sections S1--S5 | reporting scripts | `paper/supplementary/` |

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete workflow, resource expectations, and artifact definitions.

## Data and licensing

Code is released under the MIT License. Input-data provenance and redistribution terms are documented in [data/README.md](data/README.md). The data are not covered by the software license.

## Citation

Citation metadata are provided in `CITATION.cff` and can be updated with the article citation and archived release DOI when those records are available.

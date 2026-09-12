# Reproducibility guide

## Computational stages

1. `experiment/01_generate_inputs.jl` creates a Latin-hypercube design and run-specific generator/configuration files.
2. `experiment/02_run_batch.jl` launches the UCED forward model for each design.
3. `experiment/03_aggregate_and_analyze.jl` reconstructs monthly outputs and calculates the equally weighted loss from coal, wind, solar, and interregional-exchange NRMSE.
4. `experiment/04_build_surrogate.jl` fits full and ARD-screened Gaussian-process surrogates using a frozen 80/20 development/test split.
5. `experiment/04b_active_parameter_robustness.jl` evaluates bound sensitivity and cross-fitted screening stability without changing the production models.
6. `experiment/05_surrogate_search.jl` searches 20,000 points and characterizes low-mismatch regions.
7. `experiment/05b_run_revalidation_candidate.jl` evaluates selected candidates in the original UCED model.

## Reproduction levels

### Level 1: archived-result verification

`python3 reproduction/verify_results.py` completes quickly and requires no solver. This is the default reviewer and CI check.

### Level 2: surrogate and reporting reproduction

`bash reproduction/reproduce_surrogate.sh <year>` refits the GP and reruns the search from the final UCED performance table. `bash reproduction/reproduce_figures.sh` redraws the main figures from frozen plotting data. Julia and Python environments must first be installed.

### Level 3: forward-model execution

`bash reproduction/smoke_uced.sh <year>` runs one sampled design for one week. Gurobi is required. Full reconstruction uses `UCED_NUM_RUNS=100` and all 52 weeks and should be run on a suitable multicore system.

## Configuration

The main environment variables are:

| Variable | Default | Meaning |
|---|---:|---|
| `RUNS_SAVE_DIR` | `runs_2021_work` | Run directory below `experiment/`; its name must contain the four-digit year |
| `UCED_NUM_RUNS` | `100` | Number of sampled parameter designs |
| `UCED_NUM_WEEKS` | `52` | Number of weeks expected by aggregation |
| `UCED_SELECTED_WEEKS` | empty | Comma-separated weeks solved by `model/Run.jl`; used for smoke testing |
| `UCED_SOLVER_THREADS` | `8` | Gurobi threads per UCED process |
| `N_SEARCH_SAMPLES` | `20000` | Dense surrogate-search sample size |

## Frozen artifacts

The files under `results/2016` and `results/2021` are the records used for the paper. Reproduction scripts copy them to a working directory before running any write-producing stage. JLD2 files are supplied for exact reporting extraction; the CSV and JSON artifacts remain the portable record if binary deserialization changes across Julia versions.

## Numerical tolerance

The result verifier uses strict tolerances for arithmetic identities and stored outputs. A newly optimized GP can differ slightly across operating systems and library builds, so a refit should be assessed against the reported predictive behavior and retained parameter set rather than byte equality of model files.

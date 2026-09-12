# 04_build_and_validate_surrogate.jl
# Single metric_type configuration: "NRMSE", "RMSE", or "MAE"
#
# Purpose:
#   1. Load the LHS run results (training data).
#   2. Run k-fold cross-validation of the GP surrogate.
#   3. Report R2/RMSE/MAE, predictive NLL and interval coverage.
#   4. Screen ARD length scales for weakly relevant parameters.
#
# Run after 03_aggregate_and_analyze.jl; the constrained version runs after this.
#   julia experiment/04_build_and_validate_surrogate.jl 2>&1 | tee metamodel_validation_$(date +%Y%m%d_%H%M).log
#
# ============================================================================
# GP parameterization (2026 Aug correction)
# ----------------------------------------------------------------------------
#   SEArd/Mat32Ard/Mat52Ard(ll, lsigma) and Noise(lsigma) all take LOG-scale
#   arguments. GPE(x, y, mean, kernel[, logNoise]) carries its own observation
#   noise (default logNoise = -2.0) which optimize! estimates by default, so a
#   `+ Noise(...)` kernel term must never be added on top of it.
#
#   Initialization is ell_j = 1, sigma_f = 1 in standardized coordinates; a
#   single nugget is estimated by bounded ML-II from 3 restarts. The noise_frac
#   grid is gone: every grid point was re-optimized anyway, so it only supplied
#   starting points and never identified a noise level.
#
# Production surrogate workflow
# ----------------------------------------------------------------------------
#   development set (80%) : kernel selection -> fixed-kernel 5-fold ARD screening
#     -> baseline 4/5 candidate-inactive rule -> refit the GP on retained dimensions.
#   independent test set (20%) : evaluated exactly once after the final baseline
#     surrogate has been chosen; test metrics never enter feature/kernel selection.
#
#   This script intentionally does NOT run the expensive active-selection robustness
#   suite. Cross-fitted ablation, outer-fold active-set stability, low-mismatch OOF
#   diagnostics, and 0.5x/1x/2x ARD-bound sensitivity are in:
#       04b_active_parameter_robustness.jl
#
#   The active set produced here is the baseline production set used by 05.
#   04b provides the paper robustness evidence and must be checked before results
#   are treated as methodologically final.
# Migration: delete any schema-v1 optimal_hyperparameters_*.json; replace all
# `kernel.kleft` access downstream with get_ard_kernel()/get_length_scales().
# ============================================================================

module ValidateMetamodel
export select_best_kernel_with_holdout_validation, select_best_kernel_comprehensive, build_and_validate_single_model, analyze_length_scales, reduce_dimensions_if_needed, standardize_features, apply_scaling, validate_metamodel, validate_with_external_test, grid_search_with_cv, save_optimal_hyperparameters, load_optimal_hyperparameters, build_gp, optimize_gp_bounded!, fit_gp_multistart, get_ard_kernel, get_length_scales, get_log_length_scales, fitted_noise_std, fitted_signal_std, screen_parameters, verify_gp_parameterization, SAVE_DIR, USE_FEATURE_SELECTION, METRIC_TYPE

include("FoldManager.jl")
include("PathConfig.jl")
using .PathConfig
using .FoldManager: apply_y_transform, learn_y_transform_params
using DataFrames, CSV, GaussianProcesses, Statistics, Printf, StatsBase, LinearAlgebra, Random, JLD2, Dates, JSON

# --- Configuration ---
NUM_FOLDS = 5
CONFIDENCE_LEVEL = 0.95
METRIC_TYPE = "NRMSE"  # 🔧 "NRMSE", "RMSE", or "MAE"
AUTO_TRANSFORM = true
USE_EXTERNAL_TEST = false
FORCE_HYPERPARAMETER_SEARCH = true
SAVE_DIR = get(ENV, "SAVE_DIR_OVERRIDE", PathConfig.SAVE_DIR)
TIMESTAMP = Dates.format(Dates.now(), "yyyymmdd_HHMM")
USE_FEATURE_SELECTION = false

# ============================================================================
# GP hyperparameter initialization / bounds  (2026 correction)
# ============================================================================
#   ll = log(length scale), lsigma = log(signal standard deviation);
#   Noise(lsigma) stores sigma^2 = exp(2*lsigma), also a log std.
#   GPE carries its own observation noise, so no `+ Noise(...)` term is used.
#
#   The old code passed sqrt(d), 0.7(1-f)*var(y) and log(noise_variance) into
#   these log slots, which set ell = exp(sqrt(d)), sigma_f^2 = exp(1.4(1-f))
#   and sigma_n^2 = f^2, on top of a second noise term. All corrected here.



# Standardized coordinates (X_j ~ O(1), y ~ O(1)): ell_j = 1, sigma_f = 1
const INIT_LOG_LENGTHSCALE = 0.0            # ell = 1
const INIT_LOG_SIGNAL_STD  = 0.0            # sigma_f = 1
const INIT_LOG_NOISE_STD   = log(0.05)      # nugget sigma_n = 0.05

# ML-II bounds (log scale): keep ell finite and stop the nugget absorbing signal
const LOG_LS_LO,     LOG_LS_HI     = -3.0, 4.0            # ell in [0.05, 54.6]
const LOG_SIGNAL_LO, LOG_SIGNAL_HI = -3.0, 3.0            # sigma_f in [0.05, 20.1]
const LOG_NOISE_LO,  LOG_NOISE_HI  = log(1e-3), log(0.5)  # sigma_n in [1e-3, 0.5]

# Multi-start ML-II: the small-sample ARD likelihood is typically multimodal
const LS_RESTARTS = [log(0.5), 0.0, log(2.0)]   # initial ell in {0.5, 1, 2}

# ARD screening (replaces the old absolute LENGTHSCALE_THRESHOLD).
# Candidate inactive: log ell_j at the upper bound in >= SCREEN_MIN_FRAC of folds.
const SCREEN_CEILING_TOL = 0.10
const SCREEN_MIN_FRAC    = 0.80

# Deprecated: the old absolute length-scale threshold (20 for 2016 / 50 for 2021)
# was equivalent to "ell never left its exp(sqrt(d)) initial value". Kept for compat.
const LENGTHSCALE_THRESHOLD = nothing

println("📁 Validate_metamodel Configuration: Save directory = $SAVE_DIR")
println("📊 Metric type: $METRIC_TYPE")

# ============================================================================
# Utility Functions
# ============================================================================

function get_metric_columns(metric_type::String)
    """Column names for the given metric_type"""
    if metric_type == "NRMSE"
        return ["nrmse_coal", "nrmse_wind", "nrmse_solar", "nrmse_mlt"]
    elseif metric_type == "RMSE"
        return ["raw_rmse_coal", "raw_rmse_wind", "raw_rmse_solar", "raw_rmse_mlt"]
    elseif metric_type == "MAE"
        return ["raw_mae_coal", "raw_mae_wind", "raw_mae_solar", "raw_mae_mlt"]
    else
        error("Unknown metric_type: $metric_type. Must be 'NRMSE', 'RMSE', or 'MAE'")
    end
end

function extract_metric_data(df::DataFrame, metric_type::String)
    """Extract the metric columns from a DataFrame"""
    cols = get_metric_columns(metric_type)

    for col in cols
        if !hasproperty(df, Symbol(col))
            error("Column '$col' not found in data for metric_type='$metric_type'")
        end
    end

    return (
        coal = Vector{Float64}(df[!, cols[1]]),
        wind = Vector{Float64}(df[!, cols[2]]),
        solar = Vector{Float64}(df[!, cols[3]]),
        mlt = Vector{Float64}(df[!, cols[4]])
    )
end

function standardize_features(X::Matrix, feature_names::Union{Nothing, Vector{String}}=nothing, verbose::Bool=false)
    """Z-score standardization of the feature matrix"""
    X_std = similar(X, Float64)
    scaling_params = []

    if verbose
        println("\n📊 FEATURE STANDARDIZATION:")
        println("-"^60)
        println("$(rpad("Feature", 25)) | $(rpad("Mean", 10)) | $(rpad("Std", 10)) | $(rpad("Min", 10)) | $(rpad("Max", 10)) | Status")
        println("-"^70)
    end

    for i in 1:size(X, 2)
        μ = mean(X[:, i])
        σ = std(X[:, i])

        push!(scaling_params, (mean=μ, std=σ))

        if σ > 1e-10
            X_std[:, i] = (X[:, i] .- μ) ./ σ
            status = "✅ Scaled"
        else
            X_std[:, i] = X[:, i] .- μ
            status = "⚠️  Centered only"
        end

        if verbose
            feature_name = feature_names !== nothing && i <= length(feature_names) ?
                          feature_names[i] : "Feature_$i"

            min_val = minimum(X[:, i])
            max_val = maximum(X[:, i])

            println("$(rpad(feature_name, 25)) | $(rpad(round(μ, digits=3), 10)) | $(rpad(round(σ, digits=3), 10)) | $(rpad(round(min_val, digits=3), 10)) | $(rpad(round(max_val, digits=3), 10)) | $status")
        end
    end

    if verbose
        println("-"^70)
        println("✅ Standardization completed for $(size(X, 2)) features")

        X_std_means = [mean(X_std[:, i]) for i in 1:size(X_std, 2)]
        X_std_stds = [std(X_std[:, i]) for i in 1:size(X_std, 2)]

        mean_check = all(abs.(X_std_means) .< 1e-10)
        std_check = all(abs.(X_std_stds .- 1.0) .< 1e-10)

        println("📋 Post-standardization validation:")
        println("   Mean ≈ 0: $(mean_check ? "✅ PASS" : "❌ FAIL")")
        println("   Std ≈ 1:  $(std_check ? "✅ PASS" : "⚠️  PARTIAL (some features have σ=0)")")
        println()
    end

    return X_std, scaling_params
end

function apply_scaling(X_raw, scaling_params)
    """Apply learned standardization parameters to new data"""
    X_scaled = similar(X_raw, Float64)

    for i in 1:size(X_raw, 2)
        μ = scaling_params[i].mean
        σ = scaling_params[i].std

        if σ > 1e-10
            X_scaled[:, i] = (X_raw[:, i] .- μ) ./ σ
        else
            X_scaled[:, i] = X_raw[:, i] .- μ
        end
    end

    return X_scaled
end

function calculate_adjusted_r2(r2::Float64, n_samples::Int, n_features::Int)
    """Adjusted R-squared"""
    if n_samples <= n_features + 1
        return NaN
    end

    adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    return adjusted_r2
end

"""
    raw_aggregate_loss(df, metric_type)

Aggregate the raw component mismatches into the scalar UCED loss used for
stratification and for defining low-mismatch subsets. Never a surrogate prediction.
"""
function raw_aggregate_loss(df::DataFrame, metric_type::String)
    hasproperty(df, :total_loss) && return Vector{Float64}(df[!, :total_loss])
    md = extract_metric_data(df, metric_type)
    if metric_type == "NRMSE"
        return PathConfig.WEIGHTS["coal_gen"] .* md.coal.^2 .+
               PathConfig.WEIGHTS["wind_gen"] .* md.wind.^2 .+
               PathConfig.WEIGHTS["solar_gen"] .* md.solar.^2 .+
               PathConfig.WEIGHTS["mlt_flow"] .* md.mlt.^2
    else
        return PathConfig.WEIGHTS["coal_gen"] .* abs.(md.coal) .+
               PathConfig.WEIGHTS["wind_gen"] .* abs.(md.wind) .+
               PathConfig.WEIGHTS["solar_gen"] .* abs.(md.solar) .+
               PathConfig.WEIGHTS["mlt_flow"] .* abs.(md.mlt)
    end
end


function analyze_length_scales(metamodel, feature_columns)
    """Report GP length scales (log scale is the one to read for ARD relevance)"""
    length_scales = get_length_scales(metamodel)
    log_length_scales = log.(length_scales)
    at_ceiling = log_length_scales .>= (LOG_LS_HI - SCREEN_CEILING_TOL)

    length_scale_df = DataFrame(
        Parameter=feature_columns,
        LengthScale=length_scales,
        LogLengthScale=log_length_scales,
        AtCeiling=at_ceiling
    )

    sort!(length_scale_df, :LengthScale)

    println("\n--- Optimized Metamodel Length-Scales ---")
    println(" (A shorter length-scale indicates a more influential parameter)")
    println(" (AtCeiling = pushed to the bound log ell = $(LOG_LS_HI) in this fit;")
    println("  weak ARD relevance only - removal needs fold stability and development-set ablation)")
    println("---------------------------------------------------------")
    for row in eachrow(length_scale_df)
        @printf("%-30s: l = %10.4f   log l = %8.4f  %s\n",
                row.Parameter, row.LengthScale, row.LogLengthScale,
                row.AtCeiling ? "⚠️ at ceiling" : "")
    end
    println("---------------------------------------------------------")

    n_ceiling = sum(at_ceiling)
    if n_ceiling > 0
        println(" ⚠️  $(n_ceiling)/$(length(feature_columns)) parameters hit the upper bound in this single fit.")
        println("     This is a weak-ARD-relevance flag, not a removal decision:")
        println("     candidates come from screen_parameters(); removal is decided by ablation.")
    end

    return length_scales
end

function parse_kernel_type(kernel_name::String)
    """Convert a kernel name string to its type"""
    normalized_name = replace(kernel_name, "Type{" => "", "}" => "")
    normalized_name = replace(normalized_name, "GaussianProcesses." => "")
    normalized_name = split(normalized_name, ".")[end]

    if normalized_name == "SEArd"
        return SEArd
    elseif normalized_name == "Mat32Ard"
        return Mat32Ard
    elseif normalized_name == "Mat52Ard"
        return Mat52Ard
    else
        error("Unknown kernel type: $kernel_name")
    end
end

# ============================================================================
# GP Construction — single source of truth
# ============================================================================
# The old version duplicated the kernel/GPE construction in four places
# (grid search / kernel selection / fold model / final model), which is why the
# same parameterization error appeared four times.

"""
    get_ard_kernel(gp)

Return the ARD kernel carrying the `iL2` field.
Handles both a bare kernel (this version) and a SumKernel (old `kernel + Noise`).
"""
function get_ard_kernel(gp)
    k = hasfield(typeof(gp), :kernel) ? gp.kernel : gp
    hasfield(typeof(k), :iℓ2) && return k
    if isa(k, GaussianProcesses.SumKernel)
        for sub in (k.kleft, k.kright)
            hasfield(typeof(sub), :iℓ2) && return sub
        end
    end
    error("No ARD kernel with iℓ2 field found in $(typeof(k))")
end

"""ARD length scales ell_j (linear scale)"""
get_length_scales(gp) = 1.0 ./ sqrt.(get_ard_kernel(gp).iℓ2)

"""ARD log length scales - the scale to use when reading relevance"""
get_log_length_scales(gp) = log.(get_length_scales(gp))

"""Fitted signal standard deviation sigma_f"""
function fitted_signal_std(gp)
    k = get_ard_kernel(gp)
    return hasfield(typeof(k), :σ2) ? sqrt(k.σ2) : NaN
end

"""Fitted nugget / observation noise standard deviation sigma_n (the only one)"""
function fitted_noise_std(gp)
    ln = gp.logNoise
    v = isa(ln, Real) ? float(ln) : float(first(GaussianProcesses.get_params(ln)))
    return exp(v)
end

"""
    build_gp(KernelType, X_std, y_std; log_ls, log_sf, log_sn)

Build the GP: ARD kernel (both arguments on the log scale) plus the single
observation noise carried by GPE. A `+ Noise(...)` term is never added: it would
make sigma_1^2 + sigma_2^2 unidentifiable and double-count the nugget in predict_y.
"""
function build_gp(KernelType, X_std::AbstractMatrix, y_std::AbstractVector;
                  log_ls::Union{Nothing, Vector{Float64}}=nothing,
                  log_sf::Float64=INIT_LOG_SIGNAL_STD,
                  log_sn::Float64=INIT_LOG_NOISE_STD)
    d = size(X_std, 2)
    ls = log_ls === nothing ? fill(INIT_LOG_LENGTHSCALE, d) : log_ls
    if length(ls) != d
        error("build_gp: log_ls length $(length(ls)) != num_dims $d")
    end
    kernel = KernelType(ls, log_sf)
    return GPE(Matrix(X_std'), Vector{Float64}(y_std), MeanZero(), kernel, log_sn)
end

"""
    optimize_gp_bounded!(gp; num_dims)

Bounded type-II maximum marginal likelihood.
kernbounds must follow the order of `get_params(kernel)`, i.e. [ll..., lsigma]
for the *Ard kernels (check with verify_gp_parameterization()).
"""
function optimize_gp_bounded!(
    gp;
    num_dims::Int,
    log_ls_hi::Float64=LOG_LS_HI
    )

    kernbounds = [
        vcat(fill(LOG_LS_LO, num_dims), LOG_SIGNAL_LO),
        vcat(fill(log_ls_hi, num_dims), LOG_SIGNAL_HI),
    ]
    GaussianProcesses.optimize!(gp;
        kernbounds  = kernbounds,
        noisebounds = [LOG_NOISE_LO, LOG_NOISE_HI])
    return gp
end

"""
    fit_gp_multistart(KernelType, X_std, y_std; num_dims, ...)

Multi-start ML-II over ell_0 in {0.5, 1, 2}, keeping the fit with the highest
`gp.target` (the mll when no priors are set). Errors if every restart fails.
"""
function fit_gp_multistart(KernelType, X_std::AbstractMatrix, y_std::AbstractVector;
                           num_dims::Int,
                           ls_starts::Vector{Float64}=LS_RESTARTS,
                           log_sn::Float64=INIT_LOG_NOISE_STD,
                           log_ls_hi::Float64=LOG_LS_HI,
                           allow_unbounded_fallback::Bool=false,
                           verbose::Bool=false)
    best_gp = nothing
    best_target = -Inf
    n_ok = 0

    for ls0 in ls_starts
        gp = build_gp(KernelType, X_std, y_std;
                      log_ls=fill(ls0, num_dims),
                      log_sf=INIT_LOG_SIGNAL_STD,
                      log_sn=log_sn)
        ok = false
        try
            optimize_gp_bounded!(gp; num_dims=num_dims, log_ls_hi=log_ls_hi)
            ok = true
        catch e
            verbose && @warn "restart log_ls=$(round(ls0, digits=3)) bounded optimization failed: $e"
            # 不做自动 unbounded fallback：无边界拟合会让 log ell 越过 log_ls_hi，
            # 于是 `log ell >= log_ls_hi - tol` 仍然记为 "at ceiling" 而实际上没有
            # ceiling，ARD 的 ceiling vote 和 bound sensitivity 都会失去意义。
            # A silent unbounded fit would change the estimand: log ell can exceed
            # log_ls_hi, yet still be counted "at ceiling", which voids both the
            # ARD ceiling vote and the bound-sensitivity analysis.
            if allow_unbounded_fallback
                try
                    gp = build_gp(KernelType, X_std, y_std;
                                  log_ls=fill(ls0, num_dims),
                                  log_sf=INIT_LOG_SIGNAL_STD,
                                  log_sn=log_sn)
                    GaussianProcesses.optimize!(gp)
                    ok = true
                catch e2
                    verbose && @warn "restart log_ls=$(round(ls0, digits=3)) unbounded fallback failed: $e2"
                end
            end
        end

        ok || continue
        n_ok += 1
        tgt = gp.target
        if isfinite(tgt) && tgt > best_target
            best_target = tgt
            best_gp = gp
        end
    end

    best_gp === nothing && error("All $(length(ls_starts)) GP restarts failed for $(KernelType)")
    return best_gp, best_target, n_ok
end

"""
    screen_parameters(fold_log_lengthscales, feature_columns)

ARD screening from per-fold log length scales (replaces the absolute threshold).

Produces CANDIDATE inactive dimensions (weak ARD relevance / candidate for
removal). In 04 these candidates define the baseline reduced production set;
formal stability/ablation evidence is evaluated separately in 04b.
A dimension is a candidate when both hold:
  (1) log ell_j reaches the upper bound (>= log_ls_hi - ceiling_tol) in a fold;
  (2) that happens in >= min_frac of the folds.
Median / IQR and the largest gap in the sorted medians are reported as diagnostics.
"""
function screen_parameters(fold_log_lengthscales::Vector{Vector{Float64}},
                           feature_columns::Vector{String};
                           ceiling_tol::Float64=SCREEN_CEILING_TOL,
                           min_frac::Float64=SCREEN_MIN_FRAC,
                           log_ls_hi::Float64=LOG_LS_HI)
    isempty(fold_log_lengthscales) && error("screen_parameters: no per-fold length scales provided")

    L = hcat(fold_log_lengthscales...)          # d × n_folds
    d, nf = size(L)
    d == length(feature_columns) ||
        error("screen_parameters: dimension mismatch $d vs $(length(feature_columns))")

    at_ceiling = L .>= (log_ls_hi - ceiling_tol)
    votes = vec(sum(at_ceiling, dims=2))
    need = ceil(Int, min_frac * nf)
    candidate_inactive = votes .>= need

    med = vec(mapslices(median, L, dims=2))
    iqr = vec(mapslices(x -> quantile(x, 0.75) - quantile(x, 0.25), L, dims=2))
    order = sortperm(med)

    println("\n" * "="^80)
    println("ARD RELEVANCE ACROSS FOLDS (log length scale)")
    println("="^80)
    println(" Rule: log ell at the bound ($(log_ls_hi)) in >= $(need)/$(nf) folds => candidate inactive")
    println(" candidate inactive = weak ARD relevance / candidate for removal (not final)")
    println(" Shorter log ell = more relevant; large IQR = unstable, ranking not quotable")
    println("-"^80)
    @printf("%-30s | %-12s | %-8s | %-10s | %s\n",
            "Parameter", "median logl", "IQR", "ceiling", "verdict")
    println("-"^80)
    for i in order
        @printf("%-30s | %-12.4f | %-8.4f | %-10s | %s\n",
                feature_columns[i], med[i], iqr[i],
                "$(votes[i])/$(nf)",
                candidate_inactive[i] ? "⚠️ candidate inactive" : "✅ retain")
    end
    println("-"^80)

    gaps = length(order) > 1 ? diff(med[order]) : Float64[]
    if !isempty(gaps)
        gi = argmax(gaps)
        @printf(" Largest gap in sorted median log l: rank %d -> %d  (delta = %.4f)\n",
                gi, gi + 1, gaps[gi])
        println(" A gap much larger than the others is a natural retain / candidate split.")
    end

    n_unstable = sum(iqr .> 1.0)
    if n_unstable > 0
        println(" WARNING: $(n_unstable)/$(d) parameters have across-fold IQR > 1.0 (log scale):")
        println("     the likelihood may still be multimodal; add restarts or use MAP (set_priors!).")
    end
    println("="^80)

    return Dict(
        "median_log_ls" => med,
        "iqr_log_ls" => iqr,
        "ceiling_votes" => votes,
        "n_folds" => nf,
        "votes_needed" => need,
        "log_ls_hi" => log_ls_hi,
        "ceiling_tol" => ceiling_tol,
        "min_frac" => min_frac,
        "candidate_inactive_mask" => candidate_inactive,
        "inactive_mask" => candidate_inactive,   # backward compatibility
        "sorted_order" => order,
        "fold_log_lengthscales" => fold_log_lengthscales,
        "feature_columns" => feature_columns
    )
end


"""
    run_fixed_kernel_ard_screening(df_train, metric_type, apply_transform,
                                   fold_assignments, KernelType, feature_columns)

Production 5-fold ARD screening on the development set only. The kernel family is
held fixed after kernel selection. A parameter is flagged as candidate inactive when
its fitted log length scale reaches the prescribed upper bound in at least 4/5 folds.

This is the baseline active-set construction used by 04. Robustness to the upper
bound and cross-fitted active-set stability are evaluated separately in 04b.
"""
function run_fixed_kernel_ard_screening(
    df_train::DataFrame,
    metric_type::String,
    apply_transform::Bool,
    fold_assignments::Vector{Int},
    KernelType::Type,
    feature_columns::Vector{String};
    ceiling_tol::Float64=SCREEN_CEILING_TOL,
    min_frac::Float64=SCREEN_MIN_FRAC,
    log_ls_hi::Float64=LOG_LS_HI
    )

    X_raw = Matrix{Float64}(df_train[!, feature_columns])
    md = extract_metric_data(df_train, metric_type)

    num_dims = size(X_raw, 2)
    nf = maximum(fold_assignments)
    nf == 5 || @warn "Baseline ARD screening expected 5 folds; got $nf"

    fold_log_ls = Vector{Float64}[]

    println("\n" * "="^80)
    println("BASELINE FIXED-KERNEL ARD SCREENING")
    println("="^80)
    println("Kernel fixed to: $(KernelType)")
    println("Development data only; independent test set is untouched.")
    println("Rule: candidate inactive if at ceiling in >= 4/5 folds.")
    println("="^80)

    for k in 1:nf
        tr = findall(!=(k), fold_assignments)

        X_tr, _ = standardize_features(X_raw[tr, :])

        tp = FoldManager.learn_y_transform_params(
            md.coal[tr], md.wind[tr], md.solar[tr], md.mlt[tr],
            metric_type=metric_type,
            apply_transform=apply_transform,
            verbose=false
        )

        y_tr_t, _ = FoldManager.apply_y_transform(
            md.coal[tr], md.wind[tr], md.solar[tr], md.mlt[tr],
            tp,
            fold_train_indices=nothing,
            apply_standardization=false,
            verbose=false
        )

        y_tr, _ = FoldManager.standardize_y(
            y_tr_t,
            nothing,
            verbose=false
        )

        model, _mll, _nok = fit_gp_multistart(
            KernelType,
            X_tr,
            y_tr;
            num_dims=num_dims,
            log_ls_hi=log_ls_hi,
            allow_unbounded_fallback=false,
            verbose=false
        )

        push!(fold_log_ls, get_log_length_scales(model))
    end

    length(fold_log_ls) == nf ||
        error("Baseline ARD screening requires all $nf folds to fit successfully")

    sc = screen_parameters(
        fold_log_ls,
        feature_columns;
        ceiling_tol=ceiling_tol,
        min_frac=min_frac,
        log_ls_hi=log_ls_hi
    )

    sc["screening_source"] = "fixed_kernel_5fold_ARD_baseline_bound"
    return sc
end

"""
    verify_gp_parameterization(num_dims)

Startup check: (a) *Ard parameter order is [ll..., lsigma]; (b) initial ell = 1,
not exp(sqrt(d)); (c) initial sigma_n; (d) no second noise term in the kernel.
"""
function verify_gp_parameterization(num_dims::Int=3; verbose::Bool=true)
    k = Mat32Ard(fill(INIT_LOG_LENGTHSCALE, num_dims), INIT_LOG_SIGNAL_STD)
    p = GaussianProcesses.get_params(k)
    ok_order = (length(p) == num_dims + 1)

    X = randn(20, num_dims)
    y = randn(20)
    gp = build_gp(Mat32Ard, X, y)

    ls = get_length_scales(gp)
    ok_ls = all(isapprox.(ls, exp(INIT_LOG_LENGTHSCALE); rtol=1e-6))
    ok_noise = isapprox(fitted_noise_std(gp), exp(INIT_LOG_NOISE_STD); rtol=1e-6)
    ok_single = !isa(gp.kernel, GaussianProcesses.SumKernel)

    if verbose
        println("\n🔎 GP parameterization self-check:")
        println("   get_params(kernel) length = $(length(p)) (expect $(num_dims + 1))  $(ok_order ? "✅" : "❌")")
        println("   initial length scales     = $(round(ls[1], digits=6)) (expect 1.0)      $(ok_ls ? "✅" : "❌")")
        println("   initial sigma_n           = $(round(fitted_noise_std(gp), digits=6)) (expect $(round(exp(INIT_LOG_NOISE_STD), digits=6)))  $(ok_noise ? "✅" : "❌")")
        println("   single noise term         = $(ok_single ? "yes" : "NO — SumKernel detected")  $(ok_single ? "✅" : "❌")")
    end

    all_ok = ok_order && ok_ls && ok_noise && ok_single
    all_ok || @warn "GP parameterization self-check FAILED - verify the GaussianProcesses.jl version and parameter order"
    return all_ok
end

# ============================================================================
# Hyperparameter Management
# ============================================================================

function save_optimal_hyperparameters(
    kernel_type::Type,
    fitted_noise_std_value::Float64;
    model_type::String="full",
    features::Union{Nothing, Vector{String}}=nothing,
    save_path::String="optimal_hyperparameters.json",
    metric_type::String=METRIC_TYPE
    )
    """Save hyperparameters to JSON.

    The second argument is the ML-II estimate of the nugget standard deviation in
    standardized coordinates, NOT the old "noise_fraction" (an initialization label
    """
    hp_data = Dict(
        "kernel_type" => string(kernel_type),
        "fitted_noise_std" => fitted_noise_std_value,
        "model_type" => model_type,
        "n_features" => features === nothing ? nothing : length(features),
        "feature_names" => features,
        "timestamp" => string(Dates.now()),
        "metric_type" => metric_type,

        # Provenance: keep this parameterization traceable in the outputs
        "schema_version" => 2,
        "parameterization" => "log-scale (corrected 2026)",
        "init_log_lengthscale" => INIT_LOG_LENGTHSCALE,
        "init_log_signal_std" => INIT_LOG_SIGNAL_STD,
        "init_log_noise_std" => INIT_LOG_NOISE_STD,
        "log_ls_bounds" => [LOG_LS_LO, LOG_LS_HI],
        "log_signal_bounds" => [LOG_SIGNAL_LO, LOG_SIGNAL_HI],
        "log_noise_bounds" => [LOG_NOISE_LO, LOG_NOISE_HI],
        "n_restarts" => length(LS_RESTARTS)
    )

    open(save_path, "w") do f
        JSON.print(f, hp_data, 2)
    end

    println("✅ Hyperparameters saved to: $save_path")
end

function load_optimal_hyperparameters(load_path::String="optimal_hyperparameters.json")
    """Load saved hyperparameters (schema v2 only; v1 noise_fraction is void)"""
    if !isfile(load_path)
        error("Hyperparameter file not found: $load_path")
    end

    hp_data = JSON.parsefile(load_path)

    if !haskey(hp_data, "fitted_noise_std")
        error("""
        $(load_path) uses the old schema (v1).
        Its "noise_fraction" was passed into the log-std slot of Noise(log(v)),
        giving sigma_n^2 = noise_fraction^2 on top of the GPE default logNoise=-2.0.
        The value is not reusable - delete the file and re-run the search.""")
    end

    kernel_type = parse_kernel_type(hp_data["kernel_type"])

    println("✅ Loaded hyperparameters:")
    println("   Kernel: $(hp_data["kernel_type"])")
    println("   Fitted sigma_n: $(hp_data["fitted_noise_std"])")
    println("   Model type: $(hp_data["model_type"])")
    println("   Metric type: $(hp_data["metric_type"])")

    return Dict(
        "kernel_type" => kernel_type,
        "kernel" => hp_data["kernel_type"],
        "fitted_noise_std" => hp_data["fitted_noise_std"],
        "model_type" => hp_data["model_type"],
        "features" => get(hp_data, "feature_names", nothing),
        "metric_type" => hp_data["metric_type"]
    )
end

# ============================================================================
# Kernel Selection Functions
# ============================================================================

function grid_search_with_cv(
    df_train::DataFrame,
    metric_type::String,
    apply_transform::Bool,
    fold_assignments::Vector{Int},
    num_folds::Int=5;
    selection_criterion="composite"
    )
    """
    Kernel selection by K-fold CV.

    Difference from the old version: `noise_fraction` is no longer searched.
    Every grid point was re-optimized by optimize!, so the grid only supplied
    different starting points and never identified a noise level. The nugget is
    now a single parameter estimated by bounded ML-II.
    """

    println("\n" * "="^70)
    println("KERNEL SELECTION WITH K-FOLD CV")
    println("="^70)
    println("Metric type: $metric_type")
    println("Selection criterion: $selection_criterion")

    candidate_kernel_types = [SEArd, Mat32Ard, Mat52Ard]

    # Features
    all_names = names(df_train)
    feature_columns = all_names[findfirst(==("run_id"), all_names)+1:findfirst(==("Description"), all_names)-1]
    X_raw = Matrix{Float64}(df_train[!, feature_columns])

    # Targets
    metric_data = extract_metric_data(df_train, metric_type)
    coal_raw = metric_data.coal
    wind_raw = metric_data.wind
    solar_raw = metric_data.solar
    mlt_raw = metric_data.mlt

    num_dims = size(X_raw, 2)

    fold_sizes = [count(==(k), fold_assignments) for k in 1:num_folds]
    use_calibration = minimum(fold_sizes) >= 30

    println("Training samples: $(size(X_raw, 1))")
    println("Features: $(num_dims)")
    println("Folds: $(num_folds)  (test sizes: $fold_sizes)")
    println("Configurations: $(length(candidate_kernel_types)) kernels x $(num_folds) folds x $(length(LS_RESTARTS)) restarts")
    if !use_calibration
        println("WARNING: smallest fold < 30 test points; calibration term disabled in the composite score")
    end

    results = []
    best_score = Inf
    best_kernel_type = Mat32Ard
    best_noise_std = exp(INIT_LOG_NOISE_STD)

    println("\n$(rpad("Kernel", 12)) | $(rpad("RMSE", 8)) | $(rpad("NLL", 8)) | $(rpad("CalibErr", 9)) | $(rpad("sigma_n", 8)) | $(rpad("Score", 8)) | Status")
    println("-"^82)

    for KernelType in candidate_kernel_types
        kernel_name = replace(string(KernelType), "GaussianProcesses." => "")

        fold_rmses = Float64[]
        fold_nlls = Float64[]
        fold_calibration_errors = Float64[]
        fold_r2s = Float64[]
        fold_noise_stds = Float64[]

        for k in 1:num_folds
            test_indices = findall(==(k), fold_assignments)
            train_indices = findall(!=(k), fold_assignments)

            # Split
            X_train_raw = X_raw[train_indices, :]
            X_test_raw = X_raw[test_indices, :]

            coal_train_raw = coal_raw[train_indices]
            wind_train_raw = wind_raw[train_indices]
            solar_train_raw = solar_raw[train_indices]
            mlt_train_raw = mlt_raw[train_indices]
            coal_test_raw = coal_raw[test_indices]
            wind_test_raw = wind_raw[test_indices]
            solar_test_raw = solar_raw[test_indices]
            mlt_test_raw = mlt_raw[test_indices]

            # Standardize features
            X_train, scaling_params = standardize_features(X_train_raw)
            X_test = apply_scaling(X_test_raw, scaling_params)

            # Transform targets
            transform_params = FoldManager.learn_y_transform_params(
                coal_train_raw, wind_train_raw, solar_train_raw, mlt_train_raw,
                metric_type=metric_type,
                apply_transform=apply_transform,
                verbose=false
            )

            y_train_transformed, _ = FoldManager.apply_y_transform(
                coal_train_raw, wind_train_raw, solar_train_raw, mlt_train_raw,
                transform_params,
                fold_train_indices=nothing,
                apply_standardization=false,
                verbose=false
            )

            y_test_transformed, _ = FoldManager.apply_y_transform(
                coal_test_raw, wind_test_raw, solar_test_raw, mlt_test_raw,
                transform_params,
                fold_train_indices=nothing,
                apply_standardization=false,
                verbose=false
            )

            # Standardize targets
            y_train, std_params = FoldManager.standardize_y(
                y_train_transformed,
                nothing,
                verbose=false
            )

            y_test = FoldManager.apply_y_standardization(
                y_test_transformed,
                std_params
            )

            # Fit (log-scale init + single nugget + bounded multi-start)
            local model
            try
                model, _mll, _nok = fit_gp_multistart(
                    KernelType, X_train, y_train; num_dims=num_dims)
            catch e
                continue
            end

            # Predict
            μ_pred, σ²_pred = GaussianProcesses.predict_y(model, X_test')
            σ²_pred = max.(σ²_pred, 1e-6)

            if any(isnan, μ_pred) || any(isinf, μ_pred)
                continue
            end

            # Metrics
            fold_rmse = sqrt(mean((y_test .- μ_pred) .^ 2))
            fold_nll = 0.5 * mean(log.(2π * σ²_pred) + (y_test .- μ_pred).^2 ./ σ²_pred)

            z_scores = abs.((y_test .- μ_pred) ./ sqrt.(σ²_pred))
            coverage = mean(z_scores .<= 1.96)
            fold_calibration_error = abs(coverage - 0.95)

            ss_res = sum((y_test .- μ_pred) .^ 2)
            ss_tot = sum((y_test .- mean(y_test)) .^ 2)
            fold_r2 = ss_tot > 1e-9 ? 1 - (ss_res / ss_tot) : 0.0

            push!(fold_rmses, fold_rmse)
            push!(fold_nlls, fold_nll)
            push!(fold_calibration_errors, fold_calibration_error)
            push!(fold_r2s, fold_r2)
            push!(fold_noise_stds, fitted_noise_std(model))
        end

        if isempty(fold_rmses)
            println("$(rpad(kernel_name, 12)) | all folds failed")
            continue
        end

        # Fold means
        avg_rmse = mean(fold_rmses)
        avg_nll = mean(fold_nlls)
        avg_calibration_error = mean(fold_calibration_errors)
        avg_r2 = mean(fold_r2s)
        avg_noise_std = mean(fold_noise_stds)

        # Score
        local score::Float64

        if selection_criterion == "rmse"
            score = avg_rmse
        elseif selection_criterion == "nll"
            score = avg_nll
        elseif selection_criterion == "composite"
            if use_calibration
                score = 0.5 * avg_rmse + 0.3 * avg_nll + 0.2 * avg_calibration_error
            else
                # Renormalize onto RMSE/NLL
                score = 0.625 * avg_rmse + 0.375 * avg_nll
            end
        else
            error("Unknown selection criterion: $selection_criterion")
        end

        # Store
        result_entry = Dict(
            "kernel" => kernel_name,
            "kernel_type" => KernelType,
            "avg_rmse" => avg_rmse,
            "avg_nll" => avg_nll,
            "avg_calibration_error" => avg_calibration_error,
            "avg_r2" => avg_r2,
            "avg_fitted_noise_std" => avg_noise_std,
            "fitted_noise_stds" => fold_noise_stds,
            "score" => score,
            "selection_criterion" => selection_criterion,
            "calibration_used" => use_calibration,
            "fold_rmses" => fold_rmses,
            "fold_nlls" => fold_nlls,
            "fold_calibration_errors" => fold_calibration_errors,
            "fold_r2s" => fold_r2s
        )

        push!(results, result_entry)

        if score < best_score
            best_score = score
            best_kernel_type = KernelType
            best_noise_std = avg_noise_std
        end

        status = score == best_score ? "✅ BEST" : "      "

        @printf("%-12s | %-8.4f | %-8.4f | %-9.4f | %-8.4f | %-8.4f | %s\n",
                kernel_name[1:min(11, end)], avg_rmse, avg_nll,
                avg_calibration_error, avg_noise_std, score, status)
    end

    if isempty(results)
        error("All kernel configurations failed!")
    end

    # Report
    println("\n" * "="^70)
    println("KERNEL SELECTION RESULTS")
    println("="^70)

    sorted_results = sort(results, by=x->x["score"])

    println("\nRank | Kernel     | RMSE   | NLL    | CalibErr | R2     | sigma_n | Score")
    println("-"^78)
    for (i, res) in enumerate(sorted_results)
        marker = i == 1 ? " 🏆" : "   "
        @printf("%s%2d | %-10s | %-6.4f | %-6.4f | %-8.4f | %-6.4f | %-7.4f | %-6.4f\n",
                marker, i, res["kernel"],
                res["avg_rmse"], res["avg_nll"], res["avg_calibration_error"],
                res["avg_r2"], res["avg_fitted_noise_std"], res["score"])
    end

    best_result = sorted_results[1]

    # Are the kernels actually distinguishable?
    if length(sorted_results) > 1
        gap = sorted_results[2]["score"] - sorted_results[1]["score"]
        rmse_sd = std(best_result["fold_rmses"]) / sqrt(length(best_result["fold_rmses"]))
        if gap < rmse_sd
            println("\nWARNING: top-2 score gap ($(round(gap, digits=4))) is below the across-fold")
            println("    standard error of the best kernel RMSE ($(round(rmse_sd, digits=4))): not distinguishable.")
        end
    end

    println("\n" * "="^70)
    println("SELECTED CONFIGURATION")
    println("="^70)
    println("Kernel: $(best_kernel_type)")
    println("Fitted sigma_n (mean over folds): $(round(best_noise_std, digits=4))")
    println("Selection criterion: $(selection_criterion)")
    println("Final score: $(round(best_score, digits=4))")
    println("  RMSE: $(round(best_result["avg_rmse"], digits=4))")
    println("  NLL: $(round(best_result["avg_nll"], digits=4))")
    println("  Calibration error: $(round(best_result["avg_calibration_error"], digits=4))$(use_calibration ? "" : "  (not used in score)")")
    println("="^70)

    return best_kernel_type, best_noise_std, Dict(
        "all_results" => results,
        "best_score" => best_score,
        "best_result" => best_result,
        "selection_criterion" => selection_criterion,
        "calibration_used" => use_calibration
    )
end

function select_best_kernel_comprehensive(
    X_train_raw, y_train_raw,
    coal_train_raw, wind_train_raw, solar_train_raw, mlt_train_raw,
    num_dims;
    val_ratio=0.2,                 # deprecated, kept for backward compatibility
    inner_folds::Int=3,
    apply_transform=true,
    selection_criterion="composite",
    metric_type::String=METRIC_TYPE
    )
    """
    Inner-loop kernel selection for nested CV.

    Two differences from the old version:
      1. noise_fraction is no longer searched - the nugget is a single bounded ML-II estimate;
      2. the single hold-out is replaced by inner K-fold CV. The old code split off
         about 16 validation points with val_ratio=0.2 and picked among 21 configs,
         where selection noise dominates; coverage on 16 points also has a resolution
         of only 1/16 = 0.0625, making calibration_error essentially quantization noise.
    """

    n_total = size(X_train_raw, 1)

    println("\n   🔍 Kernel Selection with Inner $(inner_folds)-Fold CV")
    println("      Total samples: $n_total")
    println("      Selection criterion: $selection_criterion")
    println("      Metric type: $metric_type")

    # Stratification variable (raw targets, leak-free)
    use_l2 = (metric_type == "NRMSE")
    y_for_stratification_raw = if use_l2
        PathConfig.WEIGHTS["coal_gen"] .* coal_train_raw.^2 .+
        PathConfig.WEIGHTS["wind_gen"] .* wind_train_raw.^2 .+
        PathConfig.WEIGHTS["solar_gen"] .* solar_train_raw.^2 .+
        PathConfig.WEIGHTS["mlt_flow"] .* mlt_train_raw.^2
    else
        PathConfig.WEIGHTS["coal_gen"] .* abs.(coal_train_raw) .+
        PathConfig.WEIGHTS["wind_gen"] .* abs.(wind_train_raw) .+
        PathConfig.WEIGHTS["solar_gen"] .* abs.(solar_train_raw) .+
        PathConfig.WEIGHTS["mlt_flow"] .* abs.(mlt_train_raw)
    end

    println("      Stratification: raw $(use_l2 ? "L2" : "L1") aggregation (leak-free)")

    # Deterministic stratification: sort by target, then round-robin into inner folds
    sorted_indices = sortperm(y_for_stratification_raw)
    inner_assignments = zeros(Int, n_total)
    for (rank, idx) in enumerate(sorted_indices)
        inner_assignments[idx] = mod1(rank, inner_folds)
    end
    inner_sizes = [count(==(f), inner_assignments) for f in 1:inner_folds]
    println("      Inner fold sizes: $inner_sizes")

    use_calibration = minimum(inner_sizes) >= 30
    if !use_calibration
        println("      WARNING: inner folds < 30 points; calibration term disabled in the score")
    end

    candidate_kernel_types = [SEArd, Mat32Ard, Mat52Ard]

    results = []
    best_val_score = Inf
    best_kernel_type = Mat32Ard
    best_noise_std = exp(INIT_LOG_NOISE_STD)

    println("      " * "-"^72)
    println("      $(rpad("Kernel", 12)) $(rpad("RMSE", 8)) $(rpad("NLL", 8)) $(rpad("CalibErr", 9)) $(rpad("sigma_n", 8)) $(rpad("Score", 8)) Status")
    println("      " * "-"^72)

    for KernelType in candidate_kernel_types
        kernel_name = replace(string(KernelType), "GaussianProcesses." => "")

        f_rmse = Float64[]
        f_nll = Float64[]
        f_calib = Float64[]
        f_r2 = Float64[]
        f_noise = Float64[]

        for f in 1:inner_folds
            val_idx = findall(==(f), inner_assignments)
            tr_idx = findall(!=(f), inner_assignments)

            if isempty(val_idx) || length(tr_idx) < num_dims + 2
                continue
            end

            X_tr_raw = X_train_raw[tr_idx, :]
            X_va_raw = X_train_raw[val_idx, :]

            X_tr, sp = standardize_features(X_tr_raw)
            X_va = apply_scaling(X_va_raw, sp)

            tp = FoldManager.learn_y_transform_params(
                coal_train_raw[tr_idx], wind_train_raw[tr_idx],
                solar_train_raw[tr_idx], mlt_train_raw[tr_idx],
                metric_type=metric_type,
                apply_transform=apply_transform,
                verbose=false
            )

            y_tr_t, _ = FoldManager.apply_y_transform(
                coal_train_raw[tr_idx], wind_train_raw[tr_idx],
                solar_train_raw[tr_idx], mlt_train_raw[tr_idx],
                tp, fold_train_indices=nothing, apply_standardization=false, verbose=false
            )

            y_va_t, _ = FoldManager.apply_y_transform(
                coal_train_raw[val_idx], wind_train_raw[val_idx],
                solar_train_raw[val_idx], mlt_train_raw[val_idx],
                tp, fold_train_indices=nothing, apply_standardization=false, verbose=false
            )

            y_tr, stp = FoldManager.standardize_y(y_tr_t, nothing, verbose=false)
            y_va = FoldManager.apply_y_standardization(y_va_t, stp)

            if var(y_tr) < 1e-10
                @warn "$(kernel_name) inner fold $f: target variance too small"
                continue
            end

            local model
            try
                model, _mll, _nok = fit_gp_multistart(
                    KernelType, X_tr, y_tr; num_dims=num_dims)
            catch e
                @warn "$(kernel_name) inner fold $f: optimization failed - $e"
                continue
            end

            local μ_va, σ²_va
            try
                μ_va, σ²_va = GaussianProcesses.predict_y(model, X_va')
            catch pred_err
                @warn "$(kernel_name) inner fold $f: prediction failed - $pred_err"
                continue
            end

            if any(isnan, μ_va) || any(isinf, μ_va)
                @warn "$(kernel_name) inner fold $f: invalid predictions"
                continue
            end

            σ²_va = max.(σ²_va, 1e-6)

            rmse_f = sqrt(mean((y_va .- μ_va) .^ 2))
            nll_f = 0.5 * mean(log.(2π * σ²_va) + (y_va .- μ_va).^2 ./ σ²_va)
            cov_f = mean(abs.((y_va .- μ_va) ./ sqrt.(σ²_va)) .<= 1.96)
            r2_f = 1 - sum((y_va .- μ_va).^2) / max(sum((y_va .- mean(y_va)).^2), 1e-12)

            if !isfinite(rmse_f) || !isfinite(nll_f)
                @warn "$(kernel_name) inner fold $f: invalid metrics"
                continue
            end

            push!(f_rmse, rmse_f)
            push!(f_nll, nll_f)
            push!(f_calib, abs(cov_f - 0.95))
            push!(f_r2, r2_f)
            push!(f_noise, fitted_noise_std(model))
        end

        if isempty(f_rmse)
            println("      $(rpad(kernel_name, 12)) all inner folds failed")
            continue
        end

        val_rmse = mean(f_rmse)
        val_nll = mean(f_nll)
        calibration_error = mean(f_calib)
        val_r2 = mean(f_r2)
        noise_std = mean(f_noise)

        local score::Float64
        if selection_criterion == "rmse"
            score = val_rmse
        elseif selection_criterion == "nll"
            score = val_nll
        elseif selection_criterion == "composite"
            score = use_calibration ?
                (0.4 * val_rmse + 0.4 * val_nll + 0.2 * calibration_error) :
                (0.5 * val_rmse + 0.5 * val_nll)
        else
            error("Unknown selection criterion: $selection_criterion")
        end

        push!(results, Dict(
            "kernel" => kernel_name,
            "kernel_type" => KernelType,
            "val_rmse" => val_rmse,
            "val_r2" => val_r2,
            "val_nll" => val_nll,
            "calibration_error" => calibration_error,
            "fitted_noise_std" => noise_std,
            "score" => score,
            "selection_criterion" => selection_criterion,
            "calibration_used" => use_calibration,
            "inner_folds" => inner_folds
        ))

        if score < best_val_score
            best_val_score = score
            best_kernel_type = KernelType
            best_noise_std = noise_std
        end

        status = score == best_val_score ? "✅ BEST" : "      "
        @printf("      %-12s %-8.4f %-8.4f %-9.4f %-8.4f %-8.4f %s\n",
                kernel_name[1:min(11, end)], val_rmse, val_nll,
                calibration_error, noise_std, score, status)
    end

    if isempty(results)
        error("All kernel configurations failed!")
    end

    sorted_results = sort(results, by=x->x["score"])
    best_result = sorted_results[1]

    println("\n      🏆 Selected kernel: $(best_kernel_type)")
    println("         Inner-CV $(selection_criterion) score: $(round(best_val_score, digits=4))")
    println("         RMSE: $(round(best_result["val_rmse"], digits=4))  " *
            "NLL: $(round(best_result["val_nll"], digits=4))  " *
            "sigma_n: $(round(best_result["fitted_noise_std"], digits=4))")

    if length(sorted_results) > 1
        gap = sorted_results[2]["score"] - sorted_results[1]["score"]
        if gap < 0.01
            println("         WARNING: score gap to the runner-up is only $(round(gap, digits=5)): not distinguishable")
        end
    end

    selection_info = Dict(
        "all_results" => results,
        "best_score" => best_val_score,
        "best_result" => best_result,
        "best_kernel" => string(best_kernel_type),
        "fitted_noise_std" => best_noise_std,
        "n_total" => n_total,
        "inner_folds" => inner_folds,
        "inner_fold_sizes" => inner_sizes,
        "selection_criterion" => selection_criterion,
        "calibration_used" => use_calibration
    )

    return best_kernel_type, best_noise_std, selection_info
end

# ============================================================================
# Main Validation Function
# ============================================================================

function build_and_validate_single_model(
    df::DataFrame,
    metric_type::String,
    apply_transform::Bool=true,
    external_fold_assignments=nothing,
    fixed_hyperparameters::Union{Nothing, Dict}=nothing
    )
    """
    Cross-validation and model construction on a single dataset.

    Arguments:
    - df: data frame
    - metric_type: "NRMSE", "RMSE", or "MAE"
    - apply_transform: whether to apply the Box-Cox transform
    - external_fold_assignments: externally supplied fold assignment
    - fixed_hyperparameters: fixed kernel (skips the nested search)

    Returns:
    - dictionary of validation results
    """

    # ========================================================================
    # 1. Mode banner
    # ========================================================================
    println("--- Starting Complete Metamodel Validation (Internal Split) ---")
    println("---------------------------------------------------------------")
    println("🔄 REPRODUCIBILITY: Using deterministic stratified CV")
    println("   No random seed needed - stratified assignment is deterministic")
    println("   Results should be identical across runs")
    println("   Metric type: $metric_type")
    println("---------------------------------------------------------------")

    # ========================================================================
    # 2. Extract features and targets
    # ========================================================================
    all_names = names(df)
    feature_columns = all_names[findfirst(==("run_id"), all_names)+1:findfirst(==("Description"), all_names)-1]
    X_raw = Matrix{Float64}(df[!, feature_columns])

    # Metric extraction
    metric_data = extract_metric_data(df, metric_type)
    coal_raw = metric_data.coal
    wind_raw = metric_data.wind
    solar_raw = metric_data.solar
    mlt_raw = metric_data.mlt

    num_dims = size(X_raw, 2)
    n_samples = size(X_raw, 1)

    println("Training data loaded successfully:")
    println("   Samples: $n_samples")
    println("   Features: $num_dims")
    println("   Metric type: $metric_type")

    # ========================================================================
    # 3. Create or reuse fold assignments
    # ========================================================================
    println("\n--- Preparing for Stratified K-Fold ---")
    local fold_assignments::Vector{Int}

    if external_fold_assignments !== nothing
        println("🔄 Using provided external fold assignments")

        if length(external_fold_assignments) != nrow(df)
            error("External fold assignments length ($(length(external_fold_assignments))) doesn't match data size ($nrow(df))")
        end

        fold_assignments = external_fold_assignments
        fold_counts = [sum(fold_assignments .== k) for k in 1:NUM_FOLDS]

        if any(fold_counts .== 0)
            error("❌ Empty folds detected: $fold_counts")
        end

        println("   📊 Using $(NUM_FOLDS) folds with distribution: $fold_counts")
    else
        println("🔄 Creating new deterministic fold assignments")
        shared_folds, _ = FoldManager.create_shared_stratified_folds(
            df, metric_type, NUM_FOLDS, apply_transform
        )
        fold_assignments = shared_folds

        fold_counts = [sum(fold_assignments .== k) for k in 1:NUM_FOLDS]
        println("   📊 Created $(NUM_FOLDS) folds with distribution: $fold_counts")
    end

    # ========================================================================
    # 4. K-fold cross-validation
    # ========================================================================
    println("\n--- Performing $(NUM_FOLDS)-Fold Cross-Validation ---")

    if fixed_hyperparameters !== nothing
        println("   ⚙️  Hyperparameter mode: FIXED KERNEL")
        println("   Kernel: $(fixed_hyperparameters["kernel"])")
        println("   (the nugget sigma_n is always a bounded ML-II estimate, never fixed)")
    else
        println("   🔍 Hyperparameter mode: NESTED CV (kernel selection only)")
        println("   Each fold independently selects the kernel by inner K-fold CV")
    end

    all_true_y = Float64[]
    all_pred_y = Float64[]
    all_pred_var = Float64[]
    fold_hyperparams = []
    fold_r2 = Float64[]
    fold_adj_r2 = Float64[]
    fold_rmse = Float64[]
    fold_mae = Float64[]
    fold_selected_kernels = String[]
    fold_fitted_noise_stds = Float64[]
    fold_log_lengthscales = Vector{Float64}[]
    fold_mlls = Float64[]
    fold_coverages = Float64[]
    fold_nlls = Float64[]

    # ========================================================================
    # 5. Fold loop
    # ========================================================================
    for k in 1:NUM_FOLDS
        println("\n" * "="^70)
        println("FOLD $(k)/$(NUM_FOLDS)")
        println("="^70)

        # 5.1 Train/test indices
        test_indices = findall(==(k), fold_assignments)
        train_indices = findall(!=(k), fold_assignments)

        n_train_fold = length(train_indices)
        n_test_fold = length(test_indices)

        println("   Train samples: $n_train_fold")
        println("   Test samples: $n_test_fold")

        # 5.2 Standardize features on the training fold
        X_train_raw = X_raw[train_indices, :]
        X_test_raw = X_raw[test_indices, :]

        if k == 1
            println("\n   🔧 Standardizing features (detailed output for fold 1):")
            X_train, scaling_params = standardize_features(X_train_raw, feature_columns, true)
        else
            X_train, scaling_params = standardize_features(X_train_raw, feature_columns, false)
        end
        X_test = apply_scaling(X_test_raw, scaling_params)

        # 5.3 Raw targets
        coal_train_raw = coal_raw[train_indices]
        wind_train_raw = wind_raw[train_indices]
        solar_train_raw = solar_raw[train_indices]
        mlt_train_raw = mlt_raw[train_indices]

        coal_test_raw = coal_raw[test_indices]
        wind_test_raw = wind_raw[test_indices]
        solar_test_raw = solar_raw[test_indices]
        mlt_test_raw = mlt_raw[test_indices]

        # 5.4 Learn the target transform on the training fold only
        if k == 1
            println("\n   📦 Learning target transformation parameters:")
        end

        transform_params = FoldManager.learn_y_transform_params(
            coal_train_raw, wind_train_raw, solar_train_raw, mlt_train_raw,
            metric_type=metric_type,
            apply_transform=apply_transform,
            verbose=(k == 1)
        )

        # 5.5 Apply the transform to train and test
        y_train_transformed, _ = FoldManager.apply_y_transform(
            coal_train_raw, wind_train_raw, solar_train_raw, mlt_train_raw,
            transform_params,
            fold_train_indices=nothing,
            apply_standardization=false,  # standardize separately below
            verbose=false
        )

        y_test_transformed, _ = FoldManager.apply_y_transform(
            coal_test_raw, wind_test_raw, solar_test_raw, mlt_test_raw,
            transform_params,
            fold_train_indices=nothing,
            apply_standardization=false,
            verbose=false
        )

        # 5.6 Standardize targets using training-fold parameters
        y_train, std_params = FoldManager.standardize_y(
            y_train_transformed,
            nothing,
            verbose=(k == 1)
        )

        y_test = FoldManager.apply_y_standardization(
            y_test_transformed,
            std_params
        )

        # 5.7 Kernel selection (noise is not a grid dimension; it is ML-II estimated)
        local selected_kernel_type::Type

        if fixed_hyperparameters !== nothing
            selected_kernel_type = fixed_hyperparameters["kernel_type"]
            println("\n   ⚙️  Using fixed kernel: $(fixed_hyperparameters["kernel"])")
        else
            println("\n   🔍 Performing nested kernel selection (inner K-fold CV)...")

            selected_kernel_type, _inner_noise_std, selection_info = select_best_kernel_comprehensive(
                X_train, y_train,
                coal_train_raw, wind_train_raw, solar_train_raw, mlt_train_raw,
                num_dims,
                apply_transform=apply_transform,
                selection_criterion="composite",
                metric_type=metric_type
            )

            push!(fold_hyperparams, selection_info)
        end

        push!(fold_selected_kernels, string(selected_kernel_type))

        # 5.8 Fit the fold model (bounded multi-start ML-II, single nugget)
        println("\n   🎯 Training GP model on fold $k training set...")

        y_var_fold = var(y_train)

        if y_var_fold < 1e-10
            @warn "Fold $k: Target variance extremely small ($y_var_fold), may cause numerical issues"
        end

        local model
        local fold_mll_val::Float64 = NaN

        try
            model, fold_mll_val, n_ok = fit_gp_multistart(
                selected_kernel_type, X_train, y_train;
                num_dims=num_dims, verbose=true)
            println("      ✅ Fold $k optimization successful " *
                    "($(n_ok)/$(length(LS_RESTARTS)) restarts OK, mll = $(round(fold_mll_val, digits=3)))")
        catch e
            error("Fold $k: all bounded GP optimization restarts failed. " *
                  "Validation is aborted rather than using an unoptimized fallback. Original error: $e")
        end

        # Record ARD and nugget - the quantities that are interpretable across folds
        fold_ls = get_length_scales(model)
        fold_log_ls = log.(fold_ls)
        fold_sigma_n = fitted_noise_std(model)

        push!(fold_log_lengthscales, fold_log_ls)
        push!(fold_fitted_noise_stds, fold_sigma_n)
        push!(fold_mlls, fold_mll_val)

        n_at_ceiling = sum(fold_log_ls .>= (LOG_LS_HI - SCREEN_CEILING_TOL))
        @printf("      sigma_n = %.4f | sigma_f = %.4f | log l in [%.2f, %.2f] | at ceiling: %d/%d\n",
                fold_sigma_n, fitted_signal_std(model),
                minimum(fold_log_ls), maximum(fold_log_ls), n_at_ceiling, num_dims)

        # 5.9 Predict
        μ_pred, σ²_pred = GaussianProcesses.predict_y(model, X_test')
        σ²_pred = max.(σ²_pred, 1e-6)  # numerical stability

        # 5.10 Metrics
        # RMSE
        fold_rmse_val = sqrt(mean((y_test .- μ_pred) .^ 2))
        push!(fold_rmse, fold_rmse_val)

        # MAE
        fold_mae_val = mean(abs.(y_test .- μ_pred))
        push!(fold_mae, fold_mae_val)

        # R²
        ss_res = sum((y_test .- μ_pred) .^ 2)
        ss_tot = sum((y_test .- mean(y_test)) .^ 2)
        fold_r2_val = ss_tot > 1e-9 ? 1 - (ss_res / ss_tot) : 0.0
        push!(fold_r2, fold_r2_val)

        # Adjusted R²
        # Meaningful only when n_test is large enough: num_dims is not the GP degrees
        # of freedom, and at n_test ~ 16, p ~ 14 the correction factor flips the sign.
        fold_adj_r2_val = n_test_fold > 3 * num_dims ?
            calculate_adjusted_r2(fold_r2_val, n_test_fold, num_dims) : NaN
        push!(fold_adj_r2, fold_adj_r2_val)

        # Negative Log-Likelihood
        fold_nll_val = 0.5 * mean(log.(2π * σ²_pred) + (y_test .- μ_pred).^2 ./ σ²_pred)
        push!(fold_nlls, fold_nll_val)

        # Coverage (95% prediction interval)
        z_scores = abs.((y_test .- μ_pred) ./ sqrt.(σ²_pred))
        coverage = mean(z_scores .<= 1.96)
        push!(fold_coverages, coverage)

        # Store predictions
        append!(all_true_y, y_test)
        append!(all_pred_y, μ_pred)
        append!(all_pred_var, σ²_pred)

        # 5.11 Fold report
        println("\n   📊 Fold $k Results:")
        println("      RMSE: $(round(fold_rmse_val, digits=4))")
        println("      MAE: $(round(fold_mae_val, digits=4))")
        println("      R²: $(round(fold_r2_val, digits=4))")
        println("      Adjusted R²: $(round(fold_adj_r2_val, digits=4))")
        println("      NLL: $(round(fold_nll_val, digits=4))")
        println("      Coverage: $(round(coverage, digits=4)) (target: 0.95)")

        calibration_quality = abs(coverage - 0.95) < 0.05 ? "✅ Excellent" :
                             abs(coverage - 0.95) < 0.10 ? "⚠️  Good" : "❌ Poor"
        println("      Calibration: $calibration_quality")
    end

    # ========================================================================
    # 5.5 Final model on the complete training set
    # ========================================================================
    println("\n" * "="^70)
    println("TRAINING FINAL MODEL ON COMPLETE TRAINING SET")
    println("="^70)

    # Standardize features on all training data
    X_train_full, scaling_params_final = standardize_features(X_raw, feature_columns, false)

    # Transform targets on all training data
    transform_params_final = FoldManager.learn_y_transform_params(
        coal_raw, wind_raw, solar_raw, mlt_raw,
        metric_type=metric_type,
        apply_transform=apply_transform,
        verbose=true
    )

    y_train_transformed_final, _ = FoldManager.apply_y_transform(
        coal_raw, wind_raw, solar_raw, mlt_raw,
        transform_params_final,
        fold_train_indices=nothing,
        apply_standardization=false,
        verbose=false
    )

    y_train_final, std_params_final = FoldManager.standardize_y(
        y_train_transformed_final,
        nothing,
        verbose=true
    )

    # Final kernel (the nugget is ML-II estimated; no averaging of "noise fraction",
    # which was not even a value the grid had tested)
    if fixed_hyperparameters !== nothing
        final_kernel_type = fixed_hyperparameters["kernel_type"]
    else
        kernel_counts = Dict{String, Int}()
        for k in fold_selected_kernels
            kernel_counts[k] = get(kernel_counts, k, 0) + 1
        end
        most_common_kernel = mode(fold_selected_kernels)
        final_kernel_type = parse_kernel_type(most_common_kernel)

        println("   Using most common kernel: $most_common_kernel")
    end

    # Final fit (bounded multi-start)
    y_var_final = var(y_train_final)
    if y_var_final < 1e-10
        @warn "Final model: target variance extremely small ($y_var_final)"
    end

    local model_final
    local final_mll::Float64 = NaN

    try
        model_final, final_mll, n_ok_final = fit_gp_multistart(
            final_kernel_type, X_train_full, y_train_final;
            num_dims=num_dims, verbose=true)
        println("   ✅ Final model optimization successful " *
                "($(n_ok_final)/$(length(LS_RESTARTS)) restarts OK, mll = $(round(final_mll, digits=3)))")
    catch e
        error("Final GP fit failed for all bounded restarts. " *
              "The pipeline is aborted rather than saving an unoptimized surrogate. Original error: $e")
    end

    final_fitted_noise = fitted_noise_std(model_final)
    println("   sigma_n (fitted, final model) = $(round(final_fitted_noise, digits=4))")
    println("   sigma_f (fitted, final model) = $(round(fitted_signal_std(model_final), digits=4))")

    println("="^70)

    # ========================================================================
    # 6. Aggregate fold results
    # ========================================================================
    println("\n" * "="^70)
    println("CROSS-VALIDATION SUMMARY")
    println("="^70)

    avg_rmse = mean(fold_rmse)
    std_rmse = std(fold_rmse)
    avg_mae = mean(fold_mae)
    std_mae = std(fold_mae)
    avg_r2 = mean(fold_r2)
    std_r2 = std(fold_r2)
    _valid_adj = filter(isfinite, fold_adj_r2)
    avg_adj_r2 = isempty(_valid_adj) ? NaN : mean(_valid_adj)
    std_adj_r2 = length(_valid_adj) > 1 ? std(_valid_adj) : NaN
    avg_nll = mean(fold_nlls)
    std_nll = std(fold_nlls)
    avg_coverage = mean(fold_coverages)
    std_coverage = std(fold_coverages)

    println("\n📈 Performance Metrics (Mean ± Std):")
    println("-"^70)
    @printf("   RMSE:         %.4f ± %.4f\n", avg_rmse, std_rmse)
    @printf("   MAE:          %.4f ± %.4f\n", avg_mae, std_mae)
    @printf("   R²:           %.4f ± %.4f\n", avg_r2, std_r2)
    @printf("   Adjusted R²:  %.4f ± %.4f\n", avg_adj_r2, std_adj_r2)
    @printf("   NLL:          %.4f ± %.4f\n", avg_nll, std_nll)
    @printf("   Coverage:     %.4f ± %.4f (target: 0.95)\n", avg_coverage, std_coverage)
    println("-"^70)

    # Overall R2 from pooled predictions
    overall_ss_res = sum((all_true_y .- all_pred_y) .^ 2)
    overall_ss_tot = sum((all_true_y .- mean(all_true_y)) .^ 2)
    overall_r2 = overall_ss_tot > 1e-9 ? 1 - (overall_ss_res / overall_ss_tot) : 0.0
    overall_adj_r2 = calculate_adjusted_r2(overall_r2, length(all_true_y), num_dims)

    println("\n📊 Overall Metrics (Pooled Predictions):")
    @printf("   Overall R²:          %.4f\n", overall_r2)
    @printf("   Overall Adjusted R²: %.4f\n", overall_adj_r2)
    @printf("   Overall RMSE:        %.4f\n", sqrt(mean((all_true_y .- all_pred_y) .^ 2)))

    # Kernel selection consistency
    if fixed_hyperparameters === nothing
        println("\n🔧 Kernel Selection Summary:")
        println("-"^70)

        kernel_counts = Dict{String, Int}()
        for k in fold_selected_kernels
            kernel_counts[k] = get(kernel_counts, k, 0) + 1
        end

        println("   Kernel selection frequency:")
        for (kernel, count) in sort(collect(kernel_counts), by=x->x[2], rev=true)
            @printf("      %s: %d/%d folds (%.1f%%)\n",
                    kernel, count, NUM_FOLDS, 100*count/NUM_FOLDS)
        end
    end

    # Kernel stability (report only). Mixed kernel families across folds make the
    # per-fold ARD length scales non-comparable, so the official screening uses the
    if fixed_hyperparameters === nothing && !isempty(fold_selected_kernels)
        kernel_counts_report = Dict{String, Int}()
        for kname in fold_selected_kernels
            kernel_counts_report[kname] = get(kernel_counts_report, kname, 0) + 1
        end

        dominant_count = maximum(values(kernel_counts_report))
        dominant_frac = dominant_count / length(fold_selected_kernels)

        println("   Dominant-kernel fraction: " *
                "$(dominant_count)/$(length(fold_selected_kernels)) " *
                "($(round(100*dominant_frac, digits=1))%)")

        if dominant_frac < 0.80
            println("   ⚠️ Fold-wise kernel choice is not stable.")
            println("      Official ARD screening will therefore use the dedicated")
            println("      fixed-kernel screening run in main().")
        end
    end

    # Nugget: NUM_FOLDS independent ML-II estimates, interpretable in the paper
    # (the old "noise fraction consistency" was not - it was an initialization label)
    if !isempty(fold_fitted_noise_stds)
        println("\n🔊 Fitted nugget sigma_n across folds (standardized units):")
        @printf("   median = %.4f   range = [%.4f, %.4f]\n",
                median(fold_fitted_noise_stds),
                minimum(fold_fitted_noise_stds),
                maximum(fold_fitted_noise_stds))
        if maximum(fold_fitted_noise_stds) >= exp(LOG_NOISE_HI) - 1e-6
            println("   WARNING: sigma_n is at the upper bound $(round(exp(LOG_NOISE_HI), digits=3)):")
            println("       the response surface may be genuinely rough at commitment switches;")
            println("       discuss this rather than calling it observation noise.")
        end
    end

    # Multi-start consistency (is the likelihood multimodal?)
    if !isempty(fold_mlls) && all(isfinite, fold_mlls)
        @printf("\n📐 Per-fold marginal log-likelihood: median = %.3f, range = [%.3f, %.3f]\n",
                median(fold_mlls), minimum(fold_mlls), maximum(fold_mlls))
    end

    # Across-fold ARD stability and screening (replaces the absolute threshold)
    screening_info = nothing
    if !isempty(fold_log_lengthscales)
        screening_info = screen_parameters(fold_log_lengthscales, Vector{String}(feature_columns))
    end

    # ========================================================================
    # 7. Quality assessment
    # ========================================================================
    println("\n" * "="^70)
    println("QUALITY ASSESSMENT")
    println("="^70)

    # RMSE
    if avg_rmse < 0.1
        println("   ✅ Prediction Accuracy: EXCELLENT (RMSE < 0.1)")
    elseif avg_rmse < 0.2
        println("   ⚠️  Prediction Accuracy: GOOD (0.1 ≤ RMSE < 0.2)")
    elseif avg_rmse < 0.3
        println("   ⚠️  Prediction Accuracy: FAIR (0.2 ≤ RMSE < 0.3)")
    else
        println("   ❌ Prediction Accuracy: POOR (RMSE ≥ 0.3)")
    end

    # R2
    if avg_r2 > 0.9
        println("   ✅ Explained Variance: EXCELLENT (R² > 0.9)")
    elseif avg_r2 > 0.8
        println("   ⚠️  Explained Variance: GOOD (0.8 < R² ≤ 0.9)")
    elseif avg_r2 > 0.7
        println("   ⚠️  Explained Variance: FAIR (0.7 < R² ≤ 0.8)")
    else
        println("   ❌ Explained Variance: POOR (R² ≤ 0.7)")
    end

    # Calibration
    calibration_error = abs(avg_coverage - 0.95)
    if calibration_error < 0.05
        println("   ✅ Uncertainty Calibration: EXCELLENT (error < 0.05)")
    elseif calibration_error < 0.10
        println("   ⚠️  Uncertainty Calibration: GOOD (0.05 ≤ error < 0.10)")
    else
        println("   ❌ Uncertainty Calibration: POOR (error ≥ 0.10)")
    end

    # NLL (lower is better)
    if avg_nll < 0
        println("   ✅ Log-Likelihood: EXCELLENT (NLL < 0)")
    elseif avg_nll < 0.5
        println("   ⚠️  Log-Likelihood: GOOD (0 ≤ NLL < 0.5)")
    else
        println("   ⚠️  Log-Likelihood: FAIR (NLL ≥ 0.5)")
    end

    println("="^70)

    # ========================================================================
    # 8. Return
    # ========================================================================
    return Dict(
        "metric_type" => metric_type,
        "n_samples" => n_samples,
        "n_features" => num_dims,
        "n_folds" => NUM_FOLDS,
        "fold_assignments" => fold_assignments,

        "feature_columns" => feature_columns,
        "scaling_params" => scaling_params_final,
        "final_transform_params" => transform_params_final,
        "final_std_params" => std_params_final,
        "full_metamodel" => model_final,
        "X_standardized" => X_train_full,
        "y_standardized" => y_train_final,
        "y_transformed" => y_train_transformed_final,

        # Fold-level results
        "fold_rmse" => fold_rmse,
        "fold_mae" => fold_mae,
        "fold_r2" => fold_r2,
        "fold_adj_r2" => fold_adj_r2,
        "fold_nll" => fold_nlls,
        "fold_coverage" => fold_coverages,
        "fold_kernels" => fold_selected_kernels,

        # GP hyperparameters (corrected, interpretable)
        "fold_fitted_noise_stds" => fold_fitted_noise_stds,
        "fold_log_lengthscales" => fold_log_lengthscales,
        "fold_mlls" => fold_mlls,
        "screening_info" => screening_info,
        "final_fitted_noise_std" => final_fitted_noise,
        "final_fitted_signal_std" => fitted_signal_std(model_final),
        "final_mll" => final_mll,
        "final_log_lengthscales" => log.(get_length_scales(model_final)),

        # Fold means
        "avg_rmse" => avg_rmse,
        "std_rmse" => std_rmse,
        "avg_mae" => avg_mae,
        "std_mae" => std_mae,
        "r2_mean" => avg_r2,
        "r2_std" => std_r2,
        "rmse_mean" => avg_rmse,
        "rmse_std" => std_rmse,
        "avg_r2" => avg_r2,
        "std_r2" => std_r2,
        "avg_adj_r2" => avg_adj_r2,
        "std_adj_r2" => std_adj_r2,
        "adjusted_r2_cv" => (isfinite(avg_adj_r2) && abs(avg_adj_r2) > 1e-6) ?
                            std_adj_r2 / avg_adj_r2 : NaN,
        "avg_nll" => avg_nll,
        "std_nll" => std_nll,
        "avg_coverage" => avg_coverage,
        "std_coverage" => std_coverage,

        # Overall metrics
        "overall_r2" => overall_r2,
        "overall_adj_r2" => overall_adj_r2,
        "overall_rmse" => sqrt(mean((all_true_y .- all_pred_y) .^ 2)),
        "overall_mae" => mean(abs.(all_true_y .- all_pred_y)),
        "coverage_probability" => avg_coverage,
        "stability_cv" => abs(avg_r2) > 1e-6 ? std_r2 / avg_r2 : NaN,

        # Predictions
        "all_true_y" => all_true_y,
        "all_pred_y" => all_pred_y,
        "all_pred_var" => all_pred_var,

        # Prediction intervals
        "avg_interval_width" => mean(sqrt.(all_pred_var)) * 2 * 1.96,

        # Hyperparameter info
        "hyperparameter_selection_info" => fixed_hyperparameters === nothing ? fold_hyperparams : nothing,
        "used_fixed_hyperparameters" => fixed_hyperparameters !== nothing,
        "final_kernel_name" => string(final_kernel_type),
        "final_kernel_selection_method" => fixed_hyperparameters === nothing ? "Nested_CV" : "Fixed",
        "method_used" => fixed_hyperparameters === nothing ? "Nested_CV" : "Fixed",

        # GP configuration (provenance)
        "gp_config" => Dict(
            "init_log_lengthscale" => INIT_LOG_LENGTHSCALE,
            "init_log_signal_std" => INIT_LOG_SIGNAL_STD,
            "init_log_noise_std" => INIT_LOG_NOISE_STD,
            "log_ls_bounds" => [LOG_LS_LO, LOG_LS_HI],
            "log_signal_bounds" => [LOG_SIGNAL_LO, LOG_SIGNAL_HI],
            "log_noise_bounds" => [LOG_NOISE_LO, LOG_NOISE_HI],
            "n_restarts" => length(LS_RESTARTS),
            "single_noise_term" => true,
            "parameterization" => "log-scale (corrected 2026)"
        ),

        # Configuration
        "apply_transform" => apply_transform,
    )
end

function validate_metamodel(
    metric_type::String=METRIC_TYPE,
    external_fold_assignments=nothing,
    df_training_only=nothing,
    fixed_hyperparameters::Union{Nothing, Dict}=nothing
    )
    """
    Main validation entry point - two modes.

    Arguments:
    - metric_type: "NRMSE", "RMSE", or "MAE"
    - external_fold_assignments: externally supplied fold assignment
    - df_training_only: externally supplied training data (external-test mode)
    - fixed_hyperparameters: fixed kernel configuration

    Returns:
    - dictionary of validation results
    """

    println("--- Running $(metric_type)-based Metamodel Validation ---")

    if df_training_only !== nothing
        # ====================================================================
        # Case 1: complete training data supplied externally
        # ====================================================================
        println("📊 Using provided training data: $(nrow(df_training_only)) samples")
        println("   Mode: External test (cross-validation only)")

        validation_results = build_and_validate_single_model(
            df_training_only,
            metric_type,
            AUTO_TRANSFORM,
            external_fold_assignments,
            fixed_hyperparameters
        )

        # Result metadata
        validation_results["df_train"] = df_training_only
        validation_results["df_test"] = nothing
        validation_results["mode"] = "external_test"
        validation_results["test_r2"] = nothing
        validation_results["test_rmse"] = nothing
        validation_results["test_coverage"] = nothing

        return validation_results

    else
        # ====================================================================
        # Case 2: internal split
        # ====================================================================
        println("📊 Loading data and performing internal train/test split")

        script_dir = @__DIR__
        results_summary_path = joinpath(script_dir, SAVE_DIR, "final_performance_summary.csv")

        if !isfile(results_summary_path)
            error("Performance summary file not found: $results_summary_path")
        end

        df = CSV.read(results_summary_path, DataFrame)

        # ================================================================
        # Step 1: stratification variable
        # ================================================================
        n_total = nrow(df)
        n_test = Int(round(0.2 * n_total))

        local y_for_split_raw::Vector{Float64}

        if hasproperty(df, :total_loss)
            println("✅ Using pre-computed total_loss column for stratification")
            y_for_split_raw = Vector{Float64}(df[!, "total_loss"])
            println("   No transformation needed - direct L2 aggregation")
        else
            println("⚠️  total_loss column not found, computing on-the-fly...")

        # Raw metric data
            metric_data = extract_metric_data(df, metric_type)
            coal_raw_all = metric_data.coal
            wind_raw_all = metric_data.wind
            solar_raw_all = metric_data.solar
            mlt_raw_all = metric_data.mlt

        # Aggregation by metric type
            use_l2 = (metric_type == "NRMSE")

            y_for_split_raw = if use_l2
                # L2 (sum of squares)
                PathConfig.WEIGHTS["coal_gen"] .* coal_raw_all.^2 .+
                PathConfig.WEIGHTS["wind_gen"] .* wind_raw_all.^2 .+
                PathConfig.WEIGHTS["solar_gen"] .* solar_raw_all.^2 .+
                PathConfig.WEIGHTS["mlt_flow"] .* mlt_raw_all.^2
            else
                # L1 (sum of absolute values)
                PathConfig.WEIGHTS["coal_gen"] .* abs.(coal_raw_all) .+
                PathConfig.WEIGHTS["wind_gen"] .* abs.(wind_raw_all) .+
                PathConfig.WEIGHTS["solar_gen"] .* abs.(solar_raw_all) .+
                PathConfig.WEIGHTS["mlt_flow"] .* abs.(mlt_raw_all)
            end

            println("   Computed $(use_l2 ? "L2" : "L1") aggregation for stratification")
        end

        # ================================================================
        # Step 2: stratified sampling
        # ================================================================
        sorted_indices = sortperm(y_for_split_raw)
        test_indices_set = Set{Int}()

        # Even-interval sampling
        for i in 1:n_test
            test_idx = sorted_indices[1 + (i-1) * (n_total ÷ n_test)]
            push!(test_indices_set, test_idx)
        end

        # Enforce the exact test-set size
        while length(test_indices_set) < n_test
            # Deterministic fallback: take the first not-yet-selected index in
            # loss-sorted order. This branch is rarely needed, but it preserves
            # the script's reproducibility claim if interval sampling collides.
            added = false
            for idx in sorted_indices
                if !(idx in test_indices_set)
                    push!(test_indices_set, idx)
                    added = true
                    break
                end
            end
            added || break
        end

        while length(test_indices_set) > n_test
            pop!(test_indices_set)
        end

        test_indices = sort!(Vector{Int}(collect(test_indices_set)))
        train_indices = sort!(Vector{Int}(setdiff(1:n_total, test_indices)))

        # Build train and test sets
        df_train = df[train_indices, :]
        df_test = df[test_indices, :]

        println("✅ Data split completed:")
        println("   Training set: $(nrow(df_train)) samples")
        println("   Test set: $(nrow(df_test)) samples")
        println("   Stratification based on: $(metric_type) total loss")

        # ================================================================
        # Step 3: cross-validation on the training set
        # ================================================================
        println("\n" * "="^60)
        println("CROSS-VALIDATION ON TRAINING SET (VALIDATION)")
        println("="^60)

        validation_results = build_and_validate_single_model(
            df_train,
            metric_type,
            AUTO_TRANSFORM,
            external_fold_assignments,
            fixed_hyperparameters
        )

        validation_results["df_train"] = df_train
        validation_results["df_test"] = df_test
        validation_results["train_indices"] = train_indices
        validation_results["test_indices"] = test_indices
        validation_results["mode"] = "internal_split"

        # ================================================================
        # Step 4: DO NOT TOUCH THE INDEPENDENT TEST SET HERE
        # ================================================================
        # All kernel selection, ARD screening, dimensionality reduction, and the
        # full-vs-reduced decision are completed using df_train only.  df_test is
        # carried forward untouched and is evaluated exactly once in main() after
        # the final model (full or reduced) has been selected.
        println("\n🔒 Independent test set reserved: $(nrow(df_test)) samples")
        println("   No predictions or metrics are computed before model selection.")

        return validation_results
    end
end


"""
    evaluate_selected_model_on_test!(model_results, df_test; metric_type=METRIC_TYPE)

Evaluate the already-selected final surrogate exactly once on the independent test set.
This function is called only AFTER kernel/feature/model selection is complete.  The
resulting metrics are reporting diagnostics and never feed back into model selection.
"""
function evaluate_selected_model_on_test!(
    model_results::Dict,
    df_test::DataFrame;
    metric_type::String=METRIC_TYPE
    )

    feature_columns = Vector{String}(model_results["feature_columns"])
    X_test_raw = Matrix{Float64}(df_test[!, feature_columns])

    md = extract_metric_data(df_test, metric_type)
    X_test = apply_scaling(X_test_raw, model_results["scaling_params"])

    y_test_t, _ = FoldManager.apply_y_transform(
        md.coal, md.wind, md.solar, md.mlt,
        model_results["final_transform_params"],
        fold_train_indices=nothing,
        apply_standardization=false,
        verbose=false
    )
    y_test = FoldManager.apply_y_standardization(
        y_test_t,
        model_results["final_std_params"]
    )

    model = model_results["full_metamodel"]
    y_pred, y_var = GaussianProcesses.predict_y(model, X_test')
    y_var = max.(y_var, 1e-6)

    ss_res = sum((y_test .- y_pred).^2)
    ss_tot = sum((y_test .- mean(y_test)).^2)
    test_r2 = ss_tot > 1e-9 ? 1 - ss_res / ss_tot : 0.0
    test_rmse = sqrt(mean((y_test .- y_pred).^2))
    test_mae = mean(abs.(y_test .- y_pred))
    test_nll = 0.5 * mean(log.(2π .* y_var) + (y_test .- y_pred).^2 ./ y_var)

    pred_std = sqrt.(y_var)
    lower = y_pred .- 1.96 .* pred_std
    upper = y_pred .+ 1.96 .* pred_std
    test_coverage = mean((y_test .>= lower) .& (y_test .<= upper))

    n_test = length(y_test)
    p = length(feature_columns)
    test_adj_r2 = calculate_adjusted_r2(test_r2, n_test, p)

    # Reporting-only comparison with development OOF performance.
    cv_r2 = model_results["overall_r2"]
    cv_rmse = model_results["overall_rmse"]
    cv_coverage = model_results["coverage_probability"]
    r2_gap = abs(test_r2 - cv_r2)
    rmse_gap = abs(test_rmse - cv_rmse)
    coverage_gap = abs(test_coverage - cv_coverage)
    rmse_relative_gap = rmse_gap / max(cv_rmse, 1e-12)

    generalization_quality = r2_gap < 0.15 && rmse_relative_gap < 0.30

    model_results["df_test"] = df_test
    model_results["test_r2"] = test_r2
    model_results["test_adj_r2"] = test_adj_r2
    model_results["test_rmse"] = test_rmse
    model_results["test_mae"] = test_mae
    model_results["test_nll"] = test_nll
    model_results["test_coverage"] = test_coverage
    model_results["test_true_y"] = y_test
    model_results["test_pred_y"] = y_pred
    model_results["test_pred_var"] = y_var
    model_results["r2_gap"] = r2_gap
    model_results["rmse_gap"] = rmse_gap
    model_results["coverage_gap"] = coverage_gap
    model_results["generalization_quality"] = generalization_quality
    model_results["independent_test_evaluated_after_selection"] = true

    println("\n" * "="^70)
    println("INDEPENDENT TEST EVALUATION — FINAL SELECTED MODEL ONLY")
    println("="^70)
    model_type_label = get(model_results, "model_type", "full")
    println("Model type: $model_type_label")
    println("Features: $(p)")
    @printf("Test R²:       %.4f\n", test_r2)
    @printf("Test RMSE:     %.4f\n", test_rmse)
    @printf("Test MAE:      %.4f\n", test_mae)
    @printf("Test NLL:      %.4f\n", test_nll)
    @printf("Test coverage: %.4f (target 0.95)\n", test_coverage)
    println("These metrics are reporting-only and did not enter model selection.")
    println("="^70)

    return model_results
end

function reduce_dimensions_if_needed(
    validation_results,
    threshold=nothing,
    apply_transform::Bool=true,
    reduced_hyperparameters::Union{Nothing, Dict}=nothing;
    metric_type::String=METRIC_TYPE
    )
    """
    Screen features by the baseline 5-fold ARD rule and refit the production GP on retained dimensions.

    Arguments:
    - validation_results: results of the full model (fold assignments + screening_info)
    - threshold: DEPRECATED and ignored. The old absolute length-scale threshold
      (20 for 2016 / 50 for 2021) was equivalent to "ell never left its exp(sqrt(d))
      initial value". Screening now uses the across-fold ceiling vote.
    - apply_transform: whether to apply the target transform
    - reduced_hyperparameters: kernel for the reduced model (nothing = fresh search)
    - metric_type: "NRMSE", "RMSE", or "MAE"

    Returns:
    - validation results of the reduced model plus reduction metadata

    Candidate dimensions come from the official fixed-kernel 4/5 ARD screening on
    the development set. If candidates exist, the production reduced GP is refitted
    using the same kernel family. Formal robustness of this active set is assessed in
    04b; the independent test set is not read by this function.
    """

    if threshold !== nothing
        @warn "reduce_dimensions_if_needed: `threshold` is deprecated and ignored (old absolute length-scale threshold)." *
              " Screening uses the across-fold ceiling vote: tol=$(SCREEN_CEILING_TOL), min_frac=$(SCREEN_MIN_FRAC), ceiling=$(LOG_LS_HI)."
    end

    # ============================================================
    # Step 1: extract inputs
    # ============================================================

    # 1. Data source
    if !haskey(validation_results, "df_train") || validation_results["df_train"] === nothing
        error("Cannot find training data in validation_results")
    end

    df_train = validation_results["df_train"]
    original_fold_assignments = validation_results["fold_assignments"]

    # 2. Size consistency
    if length(original_fold_assignments) != nrow(df_train)
        error("Data size mismatch: fold_assignments ($(length(original_fold_assignments))) != df_train rows ($(nrow(df_train)))")
    end

    # 3. Mode
    mode = get(validation_results, "mode", "unknown")

    println("\n🔍 Data consistency check:")
    println("   Mode: $mode")
    println("   Metric type: $metric_type")
    println("   Training samples: $(nrow(df_train))")
    println("   Fold assignments: $(length(original_fold_assignments))")
    println("   ✅ Data safety checks passed")

    # 4. The independent test set is intentionally not read here.
    if mode == "internal_split"
        println("   🔒 Independent test set remains untouched during reduction")
    end

    feature_columns = validation_results["feature_columns"]
    metamodel = validation_results["full_metamodel"]

    println("\n" * "="^60)
    println("DIMENSION REDUCTION ANALYSIS")
    println("="^60)
    println("Current mode: $mode")
    println("Metric type: $metric_type")
    println("Training samples: $(nrow(df_train))")
    println("Independent test: reserved; not accessed in this function")

    # ============================================================
    # Step 2: candidate inactive dimensions from across-fold ARD stability
    # ============================================================
    local feature_columns_new::Vector{String}
    local removed_features::Vector{String}
    local inactive_mask
    local screening_source::String

    try
        screening_info = get(validation_results, "screening_info", nothing)

        if screening_info !== nothing
            inactive_mask = get(screening_info, "candidate_inactive_mask",
                                screening_info["inactive_mask"])
            screening_source = get(validation_results, "screening_source",
                                   "fold_stable_ceiling_vote")
            println("\n📏 Fold-stable ARD screening (candidate inactive dimensions):")
            println("   Source: $(screening_source)")
            println("   Rule: log ell at the bound $(get(screening_info, "log_ls_hi", LOG_LS_HI)) in >= $(screening_info["votes_needed"])/$(screening_info["n_folds"]) folds")
            for (i, param) in enumerate(feature_columns)
                status = inactive_mask[i] ? "⚠️ candidate" : "✅ retain   "
                @printf("   %s %-30s median log l = %8.4f  IQR = %6.4f  ceiling %d/%d\n",
                        status, param,
                        screening_info["median_log_ls"][i],
                        screening_info["iqr_log_ls"][i],
                        screening_info["ceiling_votes"][i],
                        screening_info["n_folds"])
            end
        else
            @warn "No screening_info (per-fold log ell) in validation_results." *
                  " Falling back to the single final-model fit - this biases the reduced-model CV metrics optimistically."
            screening_source = "single_fit_fallback"
            ls_single = get_length_scales(metamodel)
            log_ls_single = log.(ls_single)
            inactive_mask = log_ls_single .>= (LOG_LS_HI - SCREEN_CEILING_TOL)
            println("\n📏 Fallback screening (single fit, log l at ceiling):")
            for (i, param) in enumerate(feature_columns)
                status = inactive_mask[i] ? "❌ Remove" : "✅ Keep  "
                @printf("   %s %-30s log l = %8.4f\n", status, param, log_ls_single[i])
            end
        end

        important_indices = findall(.!inactive_mask)

        # Is the reduction meaningful?
        if length(important_indices) < 2
            println("\n⚠️  Too few features would remain ($(length(important_indices)))")
            println("   Keeping all features.")
            return validation_results
        end

        if length(important_indices) == length(feature_columns)
            println("\n✅ No candidate inactive dimension (no parameter reached the length-scale ceiling). No reduction tested.")
            return validation_results
        end

        feature_columns_new = feature_columns[important_indices]
        removed_features = setdiff(feature_columns, feature_columns_new)

        println("\n🔄 Baseline dimensionality reduction:")
        println("   $(length(feature_columns)) → $(length(feature_columns_new)) features")
        println("   Candidate for removal: $(join(removed_features, ", "))")

    catch e
        println("❌ ARD screening failed: $e")
        return validation_results
    end

    # ============================================================
    # Step 3: reduced training data
    # ============================================================
    error_cols = get_metric_columns(metric_type)

    df_train_reduced = select(df_train,
        "run_id",
        feature_columns_new...,
        "Description",
        error_cols...
    )

    println("\n📊 Reduced training data prepared: $(nrow(df_train_reduced)) samples")

    # ============================================================
    # Step 4: hyperparameter strategy for the reduced model
    # ============================================================
    println("\n" * "="^60)
    println("RE-VALIDATING REDUCED MODEL")
    println("="^60)

    local hp_for_reduced::Union{Nothing, Dict}

    if reduced_hyperparameters !== nothing
        # Case 1: kernel supplied explicitly
        println("   Strategy: Using provided kernel")
        println("   Kernel: $(reduced_hyperparameters["kernel"])")
        hp_for_reduced = reduced_hyperparameters

    else
        # Case 2: follow the original model
        original_method = validation_results["method_used"]
        force_search = get(ENV, "FORCE_REDUCED_HYPERPARAMETER_SEARCH", "false") == "true"

        should_search = (
            force_search ||
            original_method == "Nested_CV" ||
            FORCE_HYPERPARAMETER_SEARCH
        )

        if should_search
            println("   Strategy: Fresh kernel selection for reduced feature set")
            search_strategy = mode == "cv_only" ? "grid_search" : "nested_cv"
            println("   Search method: $(search_strategy)")

            reason = if force_search
                "Forced by environment variable"
            elseif original_method == "Nested_CV"
                "Original model used nested CV"
            elseif FORCE_HYPERPARAMETER_SEARCH
                "Global FORCE_HYPERPARAMETER_SEARCH=true"
            else
                "Features changed significantly"
            end
            println("   Reason: $reason")

            if search_strategy == "grid_search"
                println("   Using K-fold kernel selection (CV-only mode)")

                best_kernel_type, best_noise_std, search_info = grid_search_with_cv(
                    df_train_reduced,
                    metric_type,
                    apply_transform,
                    original_fold_assignments,
                    NUM_FOLDS;
                    selection_criterion="composite"
                )

                hp_for_reduced = Dict(
                    "kernel_type" => best_kernel_type,
                    "kernel" => replace(string(best_kernel_type), "GaussianProcesses." => ""),
                    "fitted_noise_std" => best_noise_std,
                    "source" => "kernel_search_reduced"
                )

                println("   Kernel selection completed:")
                println("     Best kernel: $(hp_for_reduced["kernel"])")
                println("     Mean fitted sigma_n: $(round(best_noise_std, digits=4))")

            else
                println("   Using nested CV (internal split mode)")
                hp_for_reduced = nothing
            end

        else
            println("   Strategy: Reusing the original model's kernel")
            println("   Reason: Original used saved/provided hyperparameters")
            println("   ⚠️  Features have changed - consider a fresh kernel search")

            hp_for_reduced = Dict(
                "kernel" => validation_results["final_kernel_name"],
                "kernel_type" => parse_kernel_type(validation_results["final_kernel_name"]),
                "fitted_noise_std" => get(validation_results, "final_fitted_noise_std", NaN),
                "source" => "adapted_from_original"
            )
        end

        println("   Kernel: $(hp_for_reduced === nothing ? "nested CV search" : hp_for_reduced["kernel"])")
    end

    # ============================================================
    # Step 5: re-validate the reduced model
    # ============================================================
    println("\n" * "="^60)
    println("REDUCED MODEL CROSS-VALIDATION")
    println("="^60)

    reduced_validation_results = build_and_validate_single_model(
        df_train_reduced,
        metric_type,
        apply_transform,
        original_fold_assignments,
        hp_for_reduced
    )

    # ============================================================
    # Step 6: length scales of the reduced model
    # ============================================================
    println("\n" * "="^60)
    println("REDUCED MODEL LENGTH SCALE ANALYSIS")
    println("="^60)

    try
        reduced_model = reduced_validation_results["full_metamodel"]
        reduced_features = reduced_validation_results["feature_columns"]

        println("Analyzing length scales for $(length(reduced_features)) remaining features...")

        reduced_length_scales = analyze_length_scales(reduced_model, reduced_features)
        reduced_log_ls = log.(reduced_length_scales)

        # Across-fold stability, preferred for interpretation when available
        reduced_screening = get(reduced_validation_results, "screening_info", nothing)

        # ARD is used for SCREENING only; no importance ranking is claimed here.
        # 这里只列出 log ell 与折间稳定性，不做 importance 分级 —— 方法学上 ARD 只用于
        # 筛选不可分辨维度，不用于给出正式的参数重要性排序。
        println("\n📏 REDUCED MODEL ARD LENGTH SCALES (screening diagnostic, not a ranking):")
        println("-"^78)
        @printf("%-30s | %-12s | %-12s | %s\n", "Feature", "log l", "fold IQR", "at ceiling")
        println("-"^78)

        for i in sortperm(reduced_log_ls)
            iqr_str = reduced_screening !== nothing ?
                      @sprintf("%.4f", reduced_screening["iqr_log_ls"][i]) : "n/a"
            @printf("%-30s | %-12.4f | %-12s | %s\n",
                    reduced_features[i], reduced_log_ls[i], iqr_str,
                    reduced_log_ls[i] >= LOG_LS_HI - SCREEN_CEILING_TOL ? "yes" : "no")
        end
        println("-"^78)

        lengthscale_order = [(reduced_features[i], reduced_length_scales[i])
                             for i in 1:length(reduced_features)]
        sort!(lengthscale_order, by=x->x[2])

        # Statistics on the log scale
        min_ls = minimum(reduced_length_scales)
        max_ls = maximum(reduced_length_scales)
        mean_ls = mean(reduced_length_scales)
        std_ls = std(reduced_length_scales)

        println("\n📈 REDUCED MODEL LENGTH SCALE STATISTICS (log scale):")
        println("   min log l:    $(round(minimum(reduced_log_ls), digits=3)) (shortest fitted scale)")
        println("   max log l:    $(round(maximum(reduced_log_ls), digits=3)) (longest fitted scale)")
        println("   median log l: $(round(median(reduced_log_ls), digits=3))")
        println("   spread (max-min): $(round(maximum(reduced_log_ls) - minimum(reduced_log_ls), digits=3))")

        sorted_log = sort(reduced_log_ls)
        if length(sorted_log) > 1
            gaps = diff(sorted_log)
            gi = argmax(gaps)
            println("   largest gap:  $(round(gaps[gi], digits=3)) (between rank $gi and $(gi+1))")
        end

        # Comparison with the original model
        if haskey(validation_results, "full_metamodel")
            println("\n🔄 COMPARISON WITH ORIGINAL MODEL (log scale):")
            try
                original_model = validation_results["full_metamodel"]
                original_features = validation_results["feature_columns"]
                original_log_ls = log.(get_length_scales(original_model))

                println("   Original model features: $(length(original_features))")
                println("   Reduced model features:  $(length(reduced_features))")
                println("   Features removed:        $(length(original_features) - length(reduced_features))")

                println("\n   $(rpad("Feature", 30)) | $(rpad("Orig log l", 12)) | $(rpad("Red. log l", 12)) | $(rpad("Change", 10)) | Status")
                println("   " * "-"^80)

                for (i, feature) in enumerate(reduced_features)
                    original_idx = findfirst(==(feature), original_features)
                    if original_idx !== nothing
                        o = original_log_ls[original_idx]
                        r = reduced_log_ls[i]
                        change = r - o

                        status = if abs(change) < 0.2
                            "✅ Stable"
                        elseif abs(change) < 0.7
                            "⚠️  Changed"
                        else
                            "🔄 Major shift"
                        end

                        @printf("   %-30s | %-12.4f | %-12.4f | %-10.4f | %s\n",
                                feature, o, r, change, status)
                    end
                end

            catch e
                println("   ⚠️  Could not compare with original model: $e")
            end
        end

        # Reduction effectiveness
        println("\n🎯 DIMENSION REDUCTION EFFECTIVENESS:")

        still_at_ceiling = sum(reduced_log_ls .>= LOG_LS_HI - SCREEN_CEILING_TOL)
        println("   Retained dimensions:                          $(length(reduced_features))")
        println("   Still at ceiling (candidate weak relevance): $still_at_ceiling")

        if still_at_ceiling > 0
            println("   Some parameters are still at the bound; a further round is possible,")
        else
            println("   All retained parameters are below the ARD ceiling in this fit")
        end

        # Store length-scale information
        reduced_validation_results["reduced_length_scales"] = reduced_length_scales
        reduced_validation_results["reduced_log_length_scales"] = reduced_log_ls
        reduced_validation_results["reduced_lengthscale_order"] = lengthscale_order
        reduced_validation_results["length_scale_stats"] = Dict(
            "min" => min_ls,
            "max" => max_ls,
            "mean" => mean_ls,
            "std" => std_ls,
            "min_log" => minimum(reduced_log_ls),
            "max_log" => maximum(reduced_log_ls),
            "median_log" => median(reduced_log_ls),
            "log_spread" => maximum(reduced_log_ls) - minimum(reduced_log_ls),
            "n_at_ceiling" => still_at_ceiling
        )

    catch e
        println("❌ Length scale analysis failed for reduced model: $e")
        println("   This might indicate kernel optimization issues")
    end

    println("="^60)

    # ============================================================
    # Step 7: independent test evaluation is intentionally deferred to main()
    # ============================================================
    println("\n🔒 Reduced candidate built on development data only; independent test remains untouched.")

    # ============================================================
    # Step 8: reduction metadata
    # ============================================================
    reduced_validation_results["model_type"] = "reduced"
    reduced_validation_results["dimension_reduced"] = true
    reduced_validation_results["original_dimensions"] = length(feature_columns)
    reduced_validation_results["reduced_dimensions"] = length(feature_columns_new)
    reduced_validation_results["threshold_used"] =
        "fold_stable_ceiling_screening(source=$(screening_source), tol=$(SCREEN_CEILING_TOL), " *
        "min_frac=$(SCREEN_MIN_FRAC), log_ls_ceiling=$(LOG_LS_HI))"
    reduced_validation_results["screening_source"] = screening_source
    reduced_validation_results["removed_features"] = removed_features
    reduced_validation_results["df_train"] = df_train_reduced
    reduced_validation_results["metric_type"] = metric_type

    # ============================================================
    # Step 9: performance comparison
    # ============================================================
    println("\n" * "="^60)
    println("DIMENSION REDUCTION SUMMARY")
    println("="^60)

    cv_r2_original = validation_results["overall_r2"]
    cv_r2_reduced = reduced_validation_results["overall_r2"]
    cv_r2_change = cv_r2_reduced - cv_r2_original

    cv_adj_r2_original = validation_results["overall_adj_r2"]
    cv_adj_r2_reduced = reduced_validation_results["overall_adj_r2"]
    cv_adj_r2_change = cv_adj_r2_reduced - cv_adj_r2_original

    cv_coverage_original = validation_results["coverage_probability"]
    cv_coverage_reduced = reduced_validation_results["coverage_probability"]
    cv_coverage_change = cv_coverage_reduced - cv_coverage_original

    println("\n📊 Cross-Validation Performance:")
    @printf("   Original model (%2d features):\n", length(feature_columns))
    @printf("      R² = %.4f, Adj R² = %.4f\n", cv_r2_original, cv_adj_r2_original)
    @printf("      Coverage = %.4f (%.1f%%)\n", cv_coverage_original, cv_coverage_original*100)

    @printf("\n   Reduced model  (%2d features):\n", length(feature_columns_new))
    @printf("      R² = %.4f, Adj R² = %.4f\n", cv_r2_reduced, cv_adj_r2_reduced)
    @printf("      Coverage = %.4f (%.1f%%)\n", cv_coverage_reduced, cv_coverage_reduced*100)

    println("\n   📈 Changes:")
    if cv_r2_change > 0
        @printf("      R² IMPROVEMENT:      +%.4f (+%.1f%%)\n",
                cv_r2_change, abs(cv_r2_change/cv_r2_original)*100)
    else
        @printf("      R² drop:             %.4f (%.1f%%)\n",
                cv_r2_change, abs(cv_r2_change/cv_r2_original)*100)
    end

    if cv_adj_r2_change > 0
        @printf("      Adj R² IMPROVEMENT:  +%.4f (+%.1f%%)\n",
                cv_adj_r2_change, abs(cv_adj_r2_change/cv_adj_r2_original)*100)
    else
        @printf("      Adj R² drop:         %.4f (%.1f%%)\n",
                cv_adj_r2_change, abs(cv_adj_r2_change/cv_adj_r2_original)*100)
    end

    if cv_coverage_change > 0
        @printf("      Coverage IMPROVEMENT: +%.4f (+%.1f pp)\n",
                cv_coverage_change, cv_coverage_change*100)
    else
        @printf("      Coverage drop:        %.4f (%.1f pp)\n",
                cv_coverage_change, cv_coverage_change*100)
    end

    # Independent-test full-vs-reduced comparison is deliberately absent:
    # only the selected final model is evaluated on the test set.

    # ============================================================
    # Step 10: baseline production selection
    # ============================================================
    # The expensive robustness decision is intentionally outside this script.
    # Here, a stable 4/5 baseline ARD screen defines the active set and the GP is
    # refitted on those retained dimensions. 04b tests whether this choice is robust.
    should_use_reduced = true
    decision_reason =
        "Baseline fixed-kernel 4/5 ARD screening identified weak-relevance " *
        "dimensions; the production GP was refitted on the retained dimensions. " *
        "Run 04b_active_parameter_robustness.jl for robustness evidence."
    decision_factors = [
        "baseline ARD screening: $(length(feature_columns)) → $(length(feature_columns_new)) dimensions",
        "removed: $(join(removed_features, ", "))",
        "robustness status: pending 04b"
    ]

    println("\n🎯 BASELINE PRODUCTION ACTIVE SET")
    println("-"^70)
    for factor in decision_factors
        println("   $factor")
    end
    println("   Decision: USE REDUCED BASELINE MODEL")
    println("   Robustness evidence is produced separately by 04b.")
    println("-"^70)

    reduced_validation_results["should_use_reduced"] = should_use_reduced
    reduced_validation_results["decision_reason"] = decision_reason
    reduced_validation_results["decision_factors"] = decision_factors
    reduced_validation_results["decision_source"] = "baseline_fixed_kernel_4of5_ARD"
    reduced_validation_results["robustness_status"] = "PENDING_04B"

    println("="^60)

    return reduced_validation_results
end

function main()
    """Main pipeline (internal split only)"""
    println("🚀 Starting Production Surrogate Build Pipeline")
    println("="^70)

    # Startup check: log parameterization, initial ell = 1, single nugget
    if !verify_gp_parameterization(3)
        error("GP parameterization self-check failed — aborting before any fitting.")
    end

    # Configuration
    METRIC_TYPE = ValidateMetamodel.METRIC_TYPE
    AUTO_TRANSFORM = ValidateMetamodel.AUTO_TRANSFORM
    NUM_FOLDS = ValidateMetamodel.NUM_FOLDS
    SAVE_DIR = ValidateMetamodel.SAVE_DIR
    FORCE_HYPERPARAMETER_SEARCH = ValidateMetamodel.FORCE_HYPERPARAMETER_SEARCH

    println("📊 Target metric: $METRIC_TYPE")

    script_dir = @__DIR__
    save_dir = joinpath(script_dir, SAVE_DIR)

    # Hyperparameter file paths
    hp_full_path = joinpath(save_dir, "optimal_hyperparameters_full.json")
    hp_reduced_path = joinpath(save_dir, "optimal_hyperparameters_reduced.json")
    println("📁 Using save directory: $save_dir")

    # ================================================================
    # Phase 1: full-feature model validation (internal split)
    # ================================================================
    println("\n🎯 MODE: Internal train/test split validation")

    # Reuse saved hyperparameters?
    local fixed_hyperparameters::Union{Nothing, Dict} = nothing

    if isfile(hp_full_path) && !FORCE_HYPERPARAMETER_SEARCH
        println("\n✅ Found saved hyperparameters for full model")
        try
            fixed_hyperparameters = ValidateMetamodel.load_optimal_hyperparameters(hp_full_path)
            println("   Using kernel: $(fixed_hyperparameters["kernel"])")
        catch e
            println("   ⚠️  Failed to load hyperparameters: $e")
            println("   Will perform hyperparameter search")
            fixed_hyperparameters = nothing
        end
    else
        if isfile(hp_full_path)
            println("\n🔄 Found saved hyperparameters but FORCE_HYPERPARAMETER_SEARCH=true")
        else
            println("\n🔍 No saved hyperparameters found")
        end
        println("   Will use nested CV for hyperparameter selection")
    end

    # validate_metamodel handles the train/test split internally
    cv_results = validate_metamodel(METRIC_TYPE, nothing, nothing, fixed_hyperparameters)

    # ================================================================
    # Phase 1b: baseline fixed-kernel ARD screening (development set only)
    # ================================================================
    println("\n🔍 Running baseline fixed-kernel 5-fold ARD screening...")

    official_kernel_type = parse_kernel_type(cv_results["final_kernel_name"])

    # Keep the nested-CV per-fold screening only as a diagnostic because folds
    # may have selected different kernel families.
    cv_results["screening_info_cv_diagnostic"] =
        get(cv_results, "screening_info", nothing)

    baseline_screening = run_fixed_kernel_ard_screening(
        cv_results["df_train"],
        METRIC_TYPE,
        AUTO_TRANSFORM,
        Vector{Int}(cv_results["fold_assignments"]),
        official_kernel_type,
        Vector{String}(cv_results["feature_columns"])
    )

    cv_results["screening_info"] = baseline_screening
    cv_results["screening_source"] = "fixed_kernel_5fold_ARD_baseline_bound"
    cv_results["robustness_status"] = "PENDING_04B"

    # Save hyperparameters if a search was performed
    if fixed_hyperparameters === nothing &&
       haskey(cv_results, "final_kernel_name") &&
       haskey(cv_results, "final_fitted_noise_std")

        kernel_type = parse_kernel_type(cv_results["final_kernel_name"])

        save_optimal_hyperparameters(
            kernel_type,
            cv_results["final_fitted_noise_std"];
            model_type="full",
            features=cv_results["feature_columns"],
            save_path=hp_full_path
        )
    end

    # ================================================================
    # Phase 2: ARD screening and dimension reduction
    # ================================================================
    local final_model_to_save = cv_results
    local model_type = "full"

    println("\n🔍 Inspecting full-model ARD diagnostics...")
    analyze_length_scales(cv_results["full_metamodel"], cv_results["feature_columns"])

    println("\n🔄 Building baseline reduced surrogate from the 4/5 ARD active set...")

    # Keep the kernel FAMILY fixed after it has been selected. Length scales,
    # signal variance, and nugget are still re-estimated on the reduced inputs.
    reduced_kernel_spec = Dict(
        "kernel_type" => official_kernel_type,
        "kernel" => replace(string(official_kernel_type), "GaussianProcesses." => ""),
        "source" => "official_full_model_kernel"
    )

    reduced_results = reduce_dimensions_if_needed(
        cv_results,
        nothing,
        AUTO_TRANSFORM,
        reduced_kernel_spec;
        metric_type=METRIC_TYPE
    )

    # Record the baseline production choice. Robustness is appended by 04b,
    # not recomputed every time 04/05 is run.
    cv_results["dimension_reduction_trial"] = Dict(
        "attempted" => get(reduced_results, "dimension_reduced", false),
        "candidate_features" => get(reduced_results, "feature_columns", String[]),
        "removed_features" => get(reduced_results, "removed_features", String[]),
        "screening_source" => get(cv_results, "screening_source", "unknown"),
        "should_use_reduced" => get(reduced_results, "should_use_reduced", false),
        "decision_reason" => get(reduced_results, "decision_reason", ""),
        "robustness_status" => "PENDING_04B"
    )


        if haskey(reduced_results, "dimension_reduced") &&
           reduced_results["dimension_reduced"]

            if get(reduced_results, "should_use_reduced", false)
                final_model_to_save = reduced_results
                model_type = "reduced"

                println("\n💡 FINAL DECISION: ✅ Using reduced model")
                println("   Features: $(length(cv_results["feature_columns"])) → " *
                        "$(length(reduced_results["feature_columns"]))")
                println("   Reason: $(get(reduced_results, "decision_reason", "ablation passed"))")

                # Save reduced-model hyperparameters only when the reduction is accepted
                if haskey(reduced_results, "final_kernel_name") &&
                   haskey(reduced_results, "final_fitted_noise_std")

                    save_optimal_hyperparameters(
                        parse_kernel_type(reduced_results["final_kernel_name"]),
                        reduced_results["final_fitted_noise_std"];
                        model_type="reduced",
                        features=reduced_results["feature_columns"],
                        save_path=hp_reduced_path
                    )
                end
            else
                final_model_to_save = cv_results
                model_type = "full"

                println("\n💡 FINAL DECISION: KEEP FULL MODEL")
                println("   Candidate reduction was tested but not accepted.")
                _why = get(reduced_results, "decision_reason", "reduced model did not pass ablation")
                println("   Reason: $(_why)")
            end
        else
            println("\n✅ No dimension reduction needed/performed - using original model")
            final_model_to_save = cv_results
            model_type = "full"
        end

    # ================================================================
    # Phase 3: independent test evaluation — exactly once, after selection
    # ================================================================
    independent_test = cv_results["df_test"]
    evaluate_selected_model_on_test!(
        final_model_to_save,
        independent_test;
        metric_type=METRIC_TYPE
    )

    # Reporting status only. This value is computed after selection and cannot
    # alter whether the full or reduced model was chosen.
    validation_passed = get(final_model_to_save, "generalization_quality", true)

    # ================================================================
    # Phase 4: save results
    # ================================================================
    save_all_results(save_dir, final_model_to_save, validation_passed, model_type, cv_results)

    println("\n🎉 Production surrogate build completed successfully!")
    println("="^70)

    return final_model_to_save, validation_passed, model_type
end

function save_all_results(
    output_dir::String,
    final_model_to_save::Dict,
    validation_passed::Bool,
    model_type::String,
    cv_results::Union{Nothing, Dict}=nothing
    )
    """
    Save all validation results and model files.

    Arguments:
    - output_dir: output directory
    - final_model_to_save: validation results of the final model
    - validation_passed: whether validation passed
    - model_type: "full" or "reduced"
    """
    println("\n💾 Saving results to: $output_dir")

    # Ensure the directory exists
    if !isdir(output_dir)
        mkpath(output_dir)
        println("   📁 Created directory: $output_dir")
    end

    # Metric type
    metric_type = get(final_model_to_save, "metric_type", METRIC_TYPE)

    # ================================================================
    # 0. Data source of the full model
    # ================================================================
    local full_model_data_source::Dict

    if model_type == "reduced" && cv_results !== nothing
        # Reduced mode: cv_results holds the full model
        full_model_data_source = cv_results
        println("\n📦 Preparing to save FULL model (before reduction)...")
    else
        # Non-reduced mode: final_model_to_save is the full model
        full_model_data_source = final_model_to_save
        println("\n📦 Preparing to save FULL model (final selected)...")
    end

    # ================================================================
    # 0b. Save the exact train/test split and the production active set
    # ================================================================
    split_source = cv_results === nothing ? final_model_to_save : cv_results

    split_path = joinpath(output_dir, "master_data_split.json")
    open(split_path, "w") do f
        JSON.print(f, Dict(
            "train_indices" => get(split_source, "train_indices", Int[]),
            "test_indices" => get(split_source, "test_indices", Int[]),
            "n_train" => length(get(split_source, "train_indices", Int[])),
            "n_test" => length(get(split_source, "test_indices", Int[])),
            "test_role" => "independent_reporting_only",
            "timestamp" => string(Dates.now())
        ), 2)
    end
    println("   ✅ Data split saved: $(split_path)")

    active_path = joinpath(output_dir, "active_parameters.json")
    original_features = Vector{String}(full_model_data_source["feature_columns"])
    active_features = Vector{String}(final_model_to_save["feature_columns"])
    removed_features = setdiff(original_features, active_features)
    open(active_path, "w") do f
        JSON.print(f, Dict(
            "active_parameters" => active_features,
            "removed_parameters" => removed_features,
            "original_parameters" => original_features,
            "model_type" => model_type,
            "model_file" => "trained_metamodel_final.jld2",
            "kernel_family" => final_model_to_save["final_kernel_name"],
            "screening_rule" => "fixed-kernel 5-fold ARD; candidate inactive at upper bound in >=4/5 folds",
            "screening_source" => get(full_model_data_source, "screening_source", "unknown"),
            "robustness_status" => get(final_model_to_save, "robustness_status",
                                       get(full_model_data_source, "robustness_status", "PENDING_04B")),
            "robustness_file" => "active_selection_robustness.json",
            "timestamp" => string(Dates.now())
        ), 2)
    end
    println("   ✅ Production active set saved: $(active_path)")

    # ================================================================
    # 1. Always save the full model
    # ================================================================
    println("\n📦 Saving FULL model...")

    full_model_path = joinpath(output_dir, "trained_metamodel_full.jld2")
    full_model_data = Dict(
        "metamodel" => full_model_data_source["full_metamodel"],
        "scaling_params" => full_model_data_source["scaling_params"],
        "transform_params" => full_model_data_source["final_transform_params"],
        "final_std_params" => full_model_data_source["final_std_params"],
        "feature_columns" => full_model_data_source["feature_columns"],

        "X_standardized" => full_model_data_source["X_standardized"],
        "y_standardized" => full_model_data_source["y_standardized"],
        "y_transformed" => get(full_model_data_source, "y_transformed", nothing),

        "schema_version" => 2,
        "gp_config" => full_model_data_source["gp_config"],

        "model_type" => "full",
        "final_dimensions" => length(full_model_data_source["feature_columns"]),
        "timestamp" => string(Dates.now())
    )

    JLD2.save(full_model_path, full_model_data)
    println("   ✅ Full model saved: $(full_model_path)")
    println("      Features: $(length(full_model_data_source["feature_columns"]))")


    # ================================================================
    # 2. Always save the full-model CV predictions
    # ================================================================
    full_cv_predictions_path = joinpath(output_dir, "cv_predictions_full.json")
    open(full_cv_predictions_path, "w") do f
        JSON.print(f, Dict(
            "all_true_y" => full_model_data_source["all_true_y"],
            "all_pred_y" => full_model_data_source["all_pred_y"],
            "all_pred_var" => full_model_data_source["all_pred_var"],
            "fold_assignments" => full_model_data_source["fold_assignments"],

            "overall_r2" => full_model_data_source["overall_r2"],
            "overall_rmse" => full_model_data_source["overall_rmse"],
            "overall_mae" => full_model_data_source["overall_mae"],
            "coverage_probability" => full_model_data_source["coverage_probability"],
            "fold_r2" => full_model_data_source["fold_r2"],
            "fold_rmse" => full_model_data_source["fold_rmse"],
            "fold_mae" => full_model_data_source["fold_mae"],

            "timestamp" => string(Dates.now())
        ), 2)
    end
    println("   ✅ Full model CV predictions saved")
    println("      CV R² = $(round(full_model_data_source["overall_r2"], digits=4))")
    println("      CV RMSE = $(round(full_model_data_source["overall_rmse"], digits=4))")


    # ================================================================
    # 3. Save the selected model (an extra file when it is the reduced one)
    # ================================================================
    if model_type == "reduced"
        println("\n📦 Saving REDUCED model (final selected)...")

        reduced_model_path = joinpath(output_dir, "trained_metamodel_reduced.jld2")
        reduced_model_data = Dict(
            "metamodel" => final_model_to_save["full_metamodel"],
            "scaling_params" => final_model_to_save["scaling_params"],
            "transform_params" => final_model_to_save["final_transform_params"],
            "final_std_params" => final_model_to_save["final_std_params"],
            "feature_columns" => final_model_to_save["feature_columns"],

            "X_standardized" => final_model_to_save["X_standardized"],
            "y_standardized" => final_model_to_save["y_standardized"],
            "y_transformed" => get(final_model_to_save, "y_transformed", nothing),

            "schema_version" => 2,
            "gp_config" => final_model_to_save["gp_config"],

            "model_type" => "reduced",
            "final_dimensions" => length(final_model_to_save["feature_columns"]),

            # Reduction metadata
            "original_dimensions" => final_model_to_save["original_dimensions"],
            "removed_features" => final_model_to_save["removed_features"],
            "threshold_used" => final_model_to_save["threshold_used"],

            "timestamp" => string(Dates.now())
        )

        JLD2.save(reduced_model_path, reduced_model_data)
        println("   ✅ Reduced model saved: $(reduced_model_path)")
        println("      Features: $(length(final_model_to_save["feature_columns"]))")

        # Reduced-model CV predictions
        reduced_cv_path = joinpath(output_dir, "cv_predictions_reduced.json")
        open(reduced_cv_path, "w") do f
            JSON.print(f, Dict(
                "all_true_y" => final_model_to_save["all_true_y"],
                "all_pred_y" => final_model_to_save["all_pred_y"],
                "all_pred_var" => final_model_to_save["all_pred_var"],
                "fold_assignments" => final_model_to_save["fold_assignments"],

                "overall_r2" => final_model_to_save["overall_r2"],
                "overall_rmse" => final_model_to_save["overall_rmse"],
                "overall_mae" => final_model_to_save["overall_mae"],
                "coverage_probability" => final_model_to_save["coverage_probability"],
                "fold_r2" => final_model_to_save["fold_r2"],
                "fold_rmse" => final_model_to_save["fold_rmse"],
                "fold_mae" => final_model_to_save["fold_mae"],

                "timestamp" => string(Dates.now())
            ), 2)
        end
        println("   ✅ Reduced CV predictions: R²=$(round(final_model_to_save["overall_r2"],digits=4)), RMSE=$(round(final_model_to_save["overall_rmse"],digits=4))")

    else
        println("\n📦 Full model is the final selected model")
        println("   (No additional save needed - already saved above)")
    end

    # ================================================================
    # 3b. Save a stable final-model alias for downstream 05
    # ================================================================
    # 05 should not need to know whether the baseline production model is full
    # or reduced. This file always points to the selected surrogate.
    final_alias_path = joinpath(output_dir, "trained_metamodel_final.jld2")
    final_alias_data = Dict(
        "metamodel" => final_model_to_save["full_metamodel"],
        "scaling_params" => final_model_to_save["scaling_params"],
        "transform_params" => final_model_to_save["final_transform_params"],
        "final_std_params" => final_model_to_save["final_std_params"],
        "feature_columns" => final_model_to_save["feature_columns"],
        "X_standardized" => final_model_to_save["X_standardized"],
        "y_standardized" => final_model_to_save["y_standardized"],
        "y_transformed" => get(final_model_to_save, "y_transformed", nothing),
        "schema_version" => 2,
        "gp_config" => final_model_to_save["gp_config"],
        "model_type" => model_type,
        "final_dimensions" => length(final_model_to_save["feature_columns"]),
        "robustness_status" => get(final_model_to_save, "robustness_status", "PENDING_04B"),
        "timestamp" => string(Dates.now())
    )
    JLD2.save(final_alias_path, final_alias_data)
    println("   ✅ Final downstream model alias saved: $(final_alias_path)")

    # ================================================================
    # 4. Save fold assignments (always the full-model assignment)
    # ================================================================
    fold_save_path = joinpath(output_dir, "master_fold_assignments.json")

    # Always the full-model fold assignment
    fold_data_source = model_type == "reduced" && cv_results !== nothing ?
                      cv_results : final_model_to_save

    open(fold_save_path, "w") do f
        JSON.print(f, Dict(
            "fold_assignments" => fold_data_source["fold_assignments"],
            "n_folds" => NUM_FOLDS,
            "training_size" => length(fold_data_source["fold_assignments"]),
            "stratification_method" => "deterministic_stratified",
            "metric_type" => metric_type,
            "timestamp" => string(Dates.now())
        ), 2)
    end
    println("   ✅ Fold assignments saved: $(fold_save_path)")

    # ================================================================
    # 3. Save transform parameters
    # ================================================================
    transform_save_path = joinpath(output_dir, "master_transform_params.json")
    open(transform_save_path, "w") do f
        JSON.print(f, Dict(
            "final_transform_params" => final_model_to_save["final_transform_params"],
            "apply_transform" => AUTO_TRANSFORM,
            "metric_type" => metric_type,
            "weights" => PathConfig.WEIGHTS,
            "timestamp" => string(Dates.now())
        ), 2)
    end
    println("   ✅ Transform parameters saved: $(transform_save_path)")

    # ================================================================
    # 4. Save scaling parameters
    # ================================================================
    scaling_save_path = joinpath(output_dir, "master_scaling_params.json")
    open(scaling_save_path, "w") do f
        JSON.print(f, Dict(
            "scaling_params" => final_model_to_save["scaling_params"],
            "feature_columns" => final_model_to_save["feature_columns"],
            "n_features" => length(final_model_to_save["feature_columns"]),
            "timestamp" => string(Dates.now())
        ), 2)
    end
    println("   ✅ Scaling parameters saved: $(scaling_save_path)")

    # ================================================================
    # 5. Save length scales
    # ================================================================
    lengthscale_save_path = joinpath(output_dir, "master_lengthscales.json")
    try
        metamodel = final_model_to_save["full_metamodel"]
        feature_columns = final_model_to_save["feature_columns"]

        # Handles a bare kernel (this version) and a SumKernel (old kernel + Noise)
        local actual_kernel
        try
            actual_kernel = get_ard_kernel(metamodel)
        catch
            actual_kernel = nothing
        end

        if actual_kernel !== nothing && hasfield(typeof(actual_kernel), :iℓ2)
            inverse_squared_length_scales = actual_kernel.iℓ2
            length_scales = 1.0 ./ sqrt.(inverse_squared_length_scales)

            feature_lengthscale_map = Dict{String, Float64}()
            for (i, feature_name) in enumerate(feature_columns)
                if i <= length(length_scales)
                    feature_lengthscale_map[feature_name] = length_scales[i]
                end
            end

            lengthscale_order = [(feature_columns[i], length_scales[i]) for i in 1:length(feature_columns)]
            sort!(lengthscale_order, by=x->x[2])

            lengthscale_data = Dict(
                "feature_lengthscales" => feature_lengthscale_map,
                "feature_columns" => feature_columns,
                "length_scales_array" => length_scales,
                "log_length_scales_array" => log.(length_scales),
                "statistics" => Dict(
                    "min_lengthscale" => minimum(length_scales),
                    "max_lengthscale" => maximum(length_scales),
                    "mean_lengthscale" => mean(length_scales),
                    "std_lengthscale" => std(length_scales),
                    "range" => maximum(length_scales) - minimum(length_scales),
                    "ratio_max_min" => maximum(length_scales) / minimum(length_scales),
                    "min_log_lengthscale" => minimum(log.(length_scales)),
                    "max_log_lengthscale" => maximum(log.(length_scales)),
                    "median_log_lengthscale" => median(log.(length_scales)),
                    "log_spread" => maximum(log.(length_scales)) - minimum(log.(length_scales)),
                    "n_at_ceiling" => sum(log.(length_scales) .>= LOG_LS_HI - SCREEN_CEILING_TOL)
                ),
                "lengthscale_ordering_diagnostic" => [
                    Dict("feature" => feat, "lengthscale" => ls,
                         "log_lengthscale" => log(ls), "order" => i)
                    for (i, (feat, ls)) in enumerate(lengthscale_order)
                ],
                "model_type" => model_type,
                "metric_type" => metric_type,
                "kernel_type" => string(typeof(actual_kernel)),
                "n_features" => length(feature_columns),
                "fitted_noise_std" => get(final_model_to_save, "final_fitted_noise_std", NaN),
                "fitted_signal_std" => get(final_model_to_save, "final_fitted_signal_std", NaN),
                "log_ls_ceiling" => LOG_LS_HI,
                "screening_ceiling_tol" => SCREEN_CEILING_TOL,
                "screening_min_frac" => SCREEN_MIN_FRAC,
                "parameterization" => "log-scale (corrected 2026)",
                "timestamp" => string(Dates.now())
            )

            # P4: keep the two screenings apart. The official screening that decided
            # which dimensions to drop is the full-dimensional one; a reduced model's own
            # ARD is only a diagnostic and must not be read as the removal evidence.
            official_sc = get(full_model_data_source, "screening_info", nothing)
            if official_sc !== nothing
                lengthscale_data["official_full_dimension_screening"] = Dict(
                    "source" => get(full_model_data_source, "screening_source", "unknown"),
                    "median_log_ls" => official_sc["median_log_ls"],
                    "iqr_log_ls" => official_sc["iqr_log_ls"],
                    "ceiling_votes" => official_sc["ceiling_votes"],
                    "n_folds" => official_sc["n_folds"],
                    "votes_needed" => official_sc["votes_needed"],
                    "log_ls_hi" => get(official_sc, "log_ls_hi", LOG_LS_HI),
                    "candidate_inactive_mask" => collect(get(official_sc, "candidate_inactive_mask",
                                                             official_sc["inactive_mask"])),
                    "feature_columns" => get(official_sc, "feature_columns", nothing),
                    "fold_log_lengthscales" => official_sc["fold_log_lengthscales"]
                )
            end

            if model_type == "reduced"
                red_sc = get(final_model_to_save, "screening_info", nothing)
                if red_sc !== nothing
                    lengthscale_data["reduced_model_ard_diagnostic"] = Dict(
                        "median_log_ls" => red_sc["median_log_ls"],
                        "iqr_log_ls" => red_sc["iqr_log_ls"],
                        "ceiling_votes" => red_sc["ceiling_votes"],
                        "n_folds" => red_sc["n_folds"],
                        "feature_columns" => get(red_sc, "feature_columns", nothing)
                    )
                end
            end

            if model_type == "reduced" && haskey(final_model_to_save, "removed_features")
                lengthscale_data["dimension_reduction_info"] = Dict(
                    "threshold_used" => final_model_to_save["threshold_used"],
                    "removed_features" => final_model_to_save["removed_features"],
                    "original_dimensions" => final_model_to_save["original_dimensions"]
                )
            end

            open(lengthscale_save_path, "w") do f
                JSON.print(f, lengthscale_data, 2)
            end

            println("   ✅ Length scales saved: $(lengthscale_save_path)")
        else
            println("   ⚠️  Could not extract length scales from kernel")
        end

    catch e
        println("   ❌ Length scales saving failed: $e")
    end

    # ================================================================
    # 6. Save the validation summary
    # ================================================================
    validation_save_path = joinpath(output_dir, "validation_summary.json")
    open(validation_save_path, "w") do f
        validation_info = Dict(
            # Model info
            "model_type" => model_type,
            "metric_type" => metric_type,
            "feature_columns" => final_model_to_save["feature_columns"],
            "final_dimensions" => length(final_model_to_save["feature_columns"]),

            # CV metrics
            "cv_r2" => final_model_to_save["overall_r2"],
            "cv_adjusted_r2" => final_model_to_save["overall_adj_r2"],
            "cv_rmse" => final_model_to_save["overall_rmse"],
            "cv_mae" => final_model_to_save["overall_mae"],
            "cv_coverage" => final_model_to_save["coverage_probability"],
            "avg_interval_width" => final_model_to_save["avg_interval_width"],

            # Fold-level info
            "fold_r2" => final_model_to_save["fold_r2"],
            "fold_rmse" => final_model_to_save["fold_rmse"],
            "fold_mae" => final_model_to_save["fold_mae"],
            "fold_adj_r2" => final_model_to_save["fold_adj_r2"],

            # Stability
            "stability_cv" => final_model_to_save["stability_cv"],
            "adjusted_r2_cv" => get(final_model_to_save, "adjusted_r2_cv", NaN),
            "r2_mean" => final_model_to_save["r2_mean"],
            "r2_std" => final_model_to_save["r2_std"],
            "rmse_mean" => final_model_to_save["rmse_mean"],
            "rmse_std" => final_model_to_save["rmse_std"],

            # Hyperparameters
            "final_kernel_name" => final_model_to_save["final_kernel_name"],
            "final_fitted_noise_std" => get(final_model_to_save, "final_fitted_noise_std", NaN),
            "final_fitted_signal_std" => get(final_model_to_save, "final_fitted_signal_std", NaN),
            "final_mll" => get(final_model_to_save, "final_mll", NaN),
            "fold_fitted_noise_stds" => get(final_model_to_save, "fold_fitted_noise_stds", Float64[]),
            "gp_config" => get(final_model_to_save, "gp_config", nothing),
            "final_kernel_selection_method" => final_model_to_save["final_kernel_selection_method"],

            # Validation status
            "validation_passed" => validation_passed,
            "validation_mode" => "internal_split",

            # Configuration
            "auto_transform" => AUTO_TRANSFORM,
            "n_folds" => NUM_FOLDS,

            # File references
            "model_file" => "trained_metamodel_final.jld2",
            "model_specific_file" => "trained_metamodel_$(model_type).jld2",
            "fold_assignments_file" => "master_fold_assignments.json",
            "transform_params_file" => "master_transform_params.json",
            "scaling_params_file" => "master_scaling_params.json",
            "cv_predictions_file" => "cv_predictions.json",
            "plotting_data_file" => "plotting_data_$(model_type).json",

            "timestamp" => string(Dates.now())
        )

        # Test results, if any
        if haskey(final_model_to_save, "test_r2") && final_model_to_save["test_r2"] !== nothing
            validation_info["test_r2"] = final_model_to_save["test_r2"]
            validation_info["test_adjusted_r2"] = get(final_model_to_save, "test_adj_r2", NaN)
            validation_info["test_rmse"] = final_model_to_save["test_rmse"]
            validation_info["test_mae"] = final_model_to_save["test_mae"]
            validation_info["test_coverage"] = get(final_model_to_save, "test_coverage", NaN)
            validation_info["test_nll"] = get(final_model_to_save, "test_nll", NaN)

            # Generalization metrics
            validation_info["r2_gap"] = get(final_model_to_save, "r2_gap", NaN)
            validation_info["rmse_gap"] = get(final_model_to_save, "rmse_gap", NaN)
            validation_info["coverage_gap"] = get(final_model_to_save, "coverage_gap", NaN)
            validation_info["generalization_quality"] = get(final_model_to_save, "generalization_quality", false)
        end

        # Reduction info
        if model_type == "reduced" && haskey(final_model_to_save, "removed_features")
            validation_info["dimension_reduction_info"] = Dict(
                "original_dimensions" => final_model_to_save["original_dimensions"],
                "removed_features" => final_model_to_save["removed_features"],
                "threshold_used" => final_model_to_save["threshold_used"]
            )
        end

        # Baseline active-set construction metadata. Expensive robustness evidence
        # is written separately by 04b_active_parameter_robustness.jl.
        if cv_results !== nothing
            validation_info["dimension_reduction_trial"] =
                get(cv_results, "dimension_reduction_trial", nothing)
            validation_info["screening_source"] =
                get(cv_results, "screening_source", nothing)
            validation_info["robustness_status"] = "PENDING_04B"
            validation_info["robustness_file"] = "active_selection_robustness.json"
        end

        JSON.print(f, validation_info, 2)
    end
    println("   ✅ Validation summary saved: $(validation_save_path)")
    println("      Includes: All CV metrics, fold details, hyperparameters")



    # ================================================================
    # 7. Generic CV prediction file (backward compatibility)
    # ================================================================
    # Always points to the predictions of the finally selected model
    cv_predictions_generic_path = joinpath(output_dir, "cv_predictions.json")
    open(cv_predictions_generic_path, "w") do f
        JSON.print(f, Dict(
            "all_true_y" => final_model_to_save["all_true_y"],
            "all_pred_y" => final_model_to_save["all_pred_y"],
            "all_pred_var" => final_model_to_save["all_pred_var"],
            "fold_assignments" => final_model_to_save["fold_assignments"],

            "overall_r2" => final_model_to_save["overall_r2"],
            "overall_rmse" => final_model_to_save["overall_rmse"],
            "overall_mae" => final_model_to_save["overall_mae"],
            "coverage_probability" => final_model_to_save["coverage_probability"],
            "fold_r2" => final_model_to_save["fold_r2"],
            "fold_rmse" => final_model_to_save["fold_rmse"],
            "fold_mae" => final_model_to_save["fold_mae"],

            "model_type" => model_type,  # provenance
            "timestamp" => string(Dates.now())
        ), 2)
    end
    println("   ✅ Generic CV predictions saved: $(cv_predictions_generic_path)")
    println("      (Points to $(model_type) model)")

    # ================================================================
    # 8. Plotting data file
    # ================================================================
    plotting_data_path = joinpath(output_dir, "plotting_data_$(model_type).json")

    plotting_data = Dict(
        # Model metadata
        "model_type" => model_type,
        "metric_type" => metric_type,
        "feature_columns" => final_model_to_save["feature_columns"],
        "n_features" => length(final_model_to_save["feature_columns"]),

        # CV results
        "cv" => Dict(
            "true_y" => final_model_to_save["all_true_y"],
            "pred_y" => final_model_to_save["all_pred_y"],
            "pred_var" => final_model_to_save["all_pred_var"],
            "fold_assignments" => final_model_to_save["fold_assignments"],
            "overall_r2" => final_model_to_save["overall_r2"],
            "overall_adj_r2" => final_model_to_save["overall_adj_r2"],
            "overall_rmse" => final_model_to_save["overall_rmse"],
            "overall_mae" => final_model_to_save["overall_mae"],
            "coverage" => final_model_to_save["coverage_probability"],
            "fold_r2" => final_model_to_save["fold_r2"],
            "fold_rmse" => final_model_to_save["fold_rmse"],
            "stability_cv" => final_model_to_save["stability_cv"]
        ),

        # Hyperparameters
        "hyperparameters" => Dict(
            "final_kernel" => final_model_to_save["final_kernel_name"],
            "final_fitted_noise_std" => get(final_model_to_save, "final_fitted_noise_std", NaN),
            "final_fitted_signal_std" => get(final_model_to_save, "final_fitted_signal_std", NaN),
            "fold_fitted_noise_stds" => get(final_model_to_save, "fold_fitted_noise_stds", Float64[]),
            "fold_log_lengthscales" => get(final_model_to_save, "fold_log_lengthscales", nothing),
            "selection_method" => final_model_to_save["final_kernel_selection_method"],
            "gp_config" => get(final_model_to_save, "gp_config", nothing)
        ),

        # Preprocessing parameters
        "preprocessing" => Dict(
            "scaling_params" => final_model_to_save["scaling_params"],
            "transform_params" => final_model_to_save["final_transform_params"],
            "std_params" => final_model_to_save["final_std_params"]
        ),

        "timestamp" => string(Dates.now()),

        # File references
        "model_file" => "trained_metamodel_final.jld2",
        "model_specific_file" => "trained_metamodel_$(model_type).jld2",
        "data_info" => "Training data (X_standardized, y_standardized) stored in .jld2"
    )

    # Test-set data, if any
    if haskey(final_model_to_save, "test_r2") && final_model_to_save["test_r2"] !== nothing
        plotting_data["internal_test"] = Dict(
            "true_y" => get(final_model_to_save, "test_true_y", nothing),
            "pred_y" => get(final_model_to_save, "test_pred_y", nothing),
            "pred_var" => get(final_model_to_save, "test_pred_var", nothing),
            "r2" => final_model_to_save["test_r2"],
            "adj_r2" => get(final_model_to_save, "test_adj_r2", NaN),
            "rmse" => final_model_to_save["test_rmse"],
            "mae" => final_model_to_save["test_mae"],
            "coverage" => get(final_model_to_save, "test_coverage", NaN),
            "nll" => get(final_model_to_save, "test_nll", NaN)
        )
    end

    # Reduction info
    if model_type == "reduced" && haskey(final_model_to_save, "removed_features")
        plotting_data["dimension_reduction"] = Dict(
            "original_dims" => final_model_to_save["original_dimensions"],
            "reduced_dims" => length(final_model_to_save["feature_columns"]),
            "removed_features" => final_model_to_save["removed_features"],
            "threshold" => final_model_to_save["threshold_used"]
        )
    end

    open(plotting_data_path, "w") do f
        JSON.print(f, plotting_data, 2)
    end

    println("   ✅ Plotting data saved: $(plotting_data_path)")
    println("      Complete dataset for all visualization needs")

    # ================================================================
    # Final summary
    # ================================================================
    println("\n📊 SAVE SUMMARY:")
    println("   1. Model (.jld2):             ✅ Includes X_standardized, y_standardized")
    println("   2. Fold assignments (.json):  ✅")
    println("   3. Transform params (.json):  ✅")
    println("   4. Scaling params (.json):    ✅")
    println("   5. Length scales (.json):     ✅")
    println("   6. Validation summary (.json):✅ All metrics")
    println("   7. CV predictions (.json):    ✅ Includes all_pred_var")
    println("   8. Plotting data (.json):     ✅ Complete visualization dataset")
    println("\n   Core model/validation files saved; plus active_parameters.json and master_data_split.json")
    println("   Mode: Internal split (train/test)")
    println("   Metric type: $metric_type")
end

end

if abspath(PROGRAM_FILE) == @__FILE__
    include("FoldManager.jl")
    using .ValidateMetamodel
    using .FoldManager
    using DataFrames, CSV, GaussianProcesses, Statistics, Printf, StatsBase, LinearAlgebra, Random, JLD2, Dates, JSON

    # Entry point
    final_model, results, model_type = ValidateMetamodel.main()
end
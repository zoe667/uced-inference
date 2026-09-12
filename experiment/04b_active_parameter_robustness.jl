# 04b_active_parameter_robustness.jl
#
# Purpose:
#   Paper robustness checks for the active-parameter set produced by
#   04_build_and_validate_surrogate.jl.
#
# This script DOES NOT change or overwrite the production GP used by 05.
# It reads the exact development split and baseline active set saved by 04 and runs:
#   1. ARD upper-bound sensitivity at 0.5x / 1x / 2x l_max;
#   2. cross-fitted full-vs-reduced ablation on development data only;
#   3. outer-fold stability of the specific baseline removed parameters;
#   4. low-mismatch OOF diagnostic (warning only).
#
# Output (all under SAVE_DIR/active_selection_robustness/):
#   robustness_summary.json
#   ard_bound_sensitivity.csv
#   cross_fitted_selection_stability.csv
#
# The production outputs in SAVE_DIR remain untouched; 05 does not depend on
# anything written by this script.
#
# No UCED runs are performed.

include("04_build_surrogate.jl")
using .ValidateMetamodel
using DataFrames, CSV, GaussianProcesses, Statistics, Printf, JSON, Dates

const ROBUST_ABS_R2_DROP = 0.05
const ROBUST_REL_RMSE_INCREASE = 0.10
const ROBUST_REL_MAE_INCREASE = 0.10
const ROBUST_LOW_RMSE_WARNING = 0.25
const ROBUST_LOW_FRAC = 0.25

function run_ard_bound_sensitivity(
    df_train::DataFrame,
    metric_type::String,
    apply_transform::Bool,
    fold_assignments::Vector{Int},
    KernelType::Type,
    feature_columns::Vector{String};
    multipliers::Vector{Float64}=[0.5, 1.0, 2.0]
    )

    X_raw = Matrix{Float64}(df_train[!, feature_columns])
    md = ValidateMetamodel.extract_metric_data(df_train, metric_type)
    num_dims = size(X_raw, 2)
    nf = maximum(fold_assignments)
    nf == 5 || error("Robustness screening requires 5 development folds; got $nf")

    results = Dict{String, Any}()

    println("\n" * "="^80)
    println("ARD UPPER-BOUND SENSITIVITY — FIXED KERNEL")
    println("="^80)
    println("Kernel: $(KernelType)")
    println("Development set only")
    println("="^80)

    for mult in multipliers
        log_hi = ValidateMetamodel.LOG_LS_HI + log(mult)
        linear_hi = exp(log_hi)
        fold_log_ls = Vector{Float64}[]

        println("\n--- lmax = $(mult)x baseline " *
                "(log upper=$(round(log_hi,digits=4)), linear=$(round(linear_hi,digits=3))) ---")

        for k in 1:nf
            tr = findall(!=(k), fold_assignments)
            X_tr, _ = ValidateMetamodel.standardize_features(X_raw[tr, :])

            tp = ValidateMetamodel.FoldManager.learn_y_transform_params(
                md.coal[tr], md.wind[tr], md.solar[tr], md.mlt[tr],
                metric_type=metric_type,
                apply_transform=apply_transform,
                verbose=false
            )
            y_t, _ = ValidateMetamodel.FoldManager.apply_y_transform(
                md.coal[tr], md.wind[tr], md.solar[tr], md.mlt[tr],
                tp,
                fold_train_indices=nothing,
                apply_standardization=false,
                verbose=false
            )
            y, _ = ValidateMetamodel.FoldManager.standardize_y(y_t, nothing, verbose=false)

            m, _, _ = ValidateMetamodel.fit_gp_multistart(
                KernelType, X_tr, y;
                num_dims=num_dims,
                log_ls_hi=log_hi,
                allow_unbounded_fallback=false,
                verbose=false
            )
            push!(fold_log_ls, ValidateMetamodel.get_log_length_scales(m))
        end

        length(fold_log_ls) == nf ||
            error("Bound sensitivity requires all $nf folds to fit successfully")

        sc = ValidateMetamodel.screen_parameters(
            fold_log_ls,
            feature_columns;
            ceiling_tol=ValidateMetamodel.SCREEN_CEILING_TOL,
            min_frac=ValidateMetamodel.SCREEN_MIN_FRAC,
            log_ls_hi=log_hi
        )

        key = @sprintf("%.1fx", mult)
        results[key] = Dict(
            "multiplier" => mult,
            "log_ls_hi" => log_hi,
            "linear_ls_hi" => linear_hi,
            "screening_info" => sc
        )
    end

    return Dict(
        "kernel" => string(KernelType),
        "multipliers" => multipliers,
        "results" => results
    )
end


function run_cross_fitted_ablation(
    df_train::DataFrame,
    metric_type::String,
    apply_transform::Bool,
    fold_assignments::Vector{Int},
    KernelType::Type,
    feature_columns::Vector{String};
    inner_folds::Int=5,
    low_frac::Float64=ROBUST_LOW_FRAC
    )

    inner_folds == 5 || error("Cross-fitted ARD screening must use 5 inner folds for the 4/5 rule")

    X_raw = Matrix{Float64}(df_train[!, feature_columns])
    md = ValidateMetamodel.extract_metric_data(df_train, metric_type)
    raw_loss = ValidateMetamodel.raw_aggregate_loss(df_train, metric_type)

    d = length(feature_columns)
    nf = maximum(fold_assignments)
    nf == 5 || error("Cross-fitted ablation requires 5 outer development folds; got $nf")

    oof_true = Float64[]
    oof_full = Float64[]
    oof_red = Float64[]
    oof_rows = Int[]

    drop_counts = zeros(Int, d)
    per_fold_removed = Vector{Vector{String}}()

    println("\n" * "="^80)
    println("CROSS-FITTED ACTIVE-SELECTION ABLATION")
    println("="^80)
    println("Outer folds: 5 | inner ARD folds: 5 | screening rule: 4/5")
    println("Independent test set is not used.")
    println("="^80)

    for k in 1:nf
        tr = findall(!=(k), fold_assignments)
        va = findall(==(k), fold_assignments)

        X_tr, sp = ValidateMetamodel.standardize_features(X_raw[tr, :])
        X_va = ValidateMetamodel.apply_scaling(X_raw[va, :], sp)

        tp = ValidateMetamodel.FoldManager.learn_y_transform_params(
            md.coal[tr], md.wind[tr], md.solar[tr], md.mlt[tr],
            metric_type=metric_type, apply_transform=apply_transform, verbose=false
        )
        y_tr_t, _ = ValidateMetamodel.FoldManager.apply_y_transform(
            md.coal[tr], md.wind[tr], md.solar[tr], md.mlt[tr], tp,
            fold_train_indices=nothing, apply_standardization=false, verbose=false
        )
        y_va_t, _ = ValidateMetamodel.FoldManager.apply_y_transform(
            md.coal[va], md.wind[va], md.solar[va], md.mlt[va], tp,
            fold_train_indices=nothing, apply_standardization=false, verbose=false
        )
        y_tr, stp = ValidateMetamodel.FoldManager.standardize_y(y_tr_t, nothing, verbose=false)
        y_va = ValidateMetamodel.FoldManager.apply_y_standardization(y_va_t, stp)

        # Deterministic 5-fold inner screening on the outer-training rows only.
        order = sortperm(raw_loss[tr])
        inner_assign = zeros(Int, length(tr))
        for (rank, idx) in enumerate(order)
            inner_assign[idx] = mod1(rank, inner_folds)
        end

        inner_log_ls = Vector{Float64}[]
        for f in 1:inner_folds
            itr = findall(!=(f), inner_assign)
            rows = tr[itr]

            X_i, _ = ValidateMetamodel.standardize_features(X_raw[rows, :])
            tp_i = ValidateMetamodel.FoldManager.learn_y_transform_params(
                md.coal[rows], md.wind[rows], md.solar[rows], md.mlt[rows],
                metric_type=metric_type, apply_transform=apply_transform, verbose=false
            )
            y_i_t, _ = ValidateMetamodel.FoldManager.apply_y_transform(
                md.coal[rows], md.wind[rows], md.solar[rows], md.mlt[rows], tp_i,
                fold_train_indices=nothing, apply_standardization=false, verbose=false
            )
            y_i, _ = ValidateMetamodel.FoldManager.standardize_y(y_i_t, nothing, verbose=false)

            m_i, _, _ = ValidateMetamodel.fit_gp_multistart(
                KernelType, X_i, y_i;
                num_dims=d,
                log_ls_hi=ValidateMetamodel.LOG_LS_HI,
                allow_unbounded_fallback=false,
                verbose=false
            )
            push!(inner_log_ls, ValidateMetamodel.get_log_length_scales(m_i))
        end

        length(inner_log_ls) == inner_folds ||
            error("Outer fold $k: not all 5 inner ARD fits succeeded")

        sc = ValidateMetamodel.screen_parameters(
            inner_log_ls,
            feature_columns;
            ceiling_tol=ValidateMetamodel.SCREEN_CEILING_TOL,
            min_frac=ValidateMetamodel.SCREEN_MIN_FRAC,
            log_ls_hi=ValidateMetamodel.LOG_LS_HI
        )

        candidate = Vector{Bool}(sc["candidate_inactive_mask"])
        keep = findall(.!candidate)

        # If the fold proposes no safe reduction, compare full against full.
        if length(keep) < 2 || length(keep) == d
            keep = collect(1:d)
        end
        dropped = setdiff(1:d, keep)

        m_full, _, _ = ValidateMetamodel.fit_gp_multistart(
            KernelType, X_tr, y_tr;
            num_dims=d,
            allow_unbounded_fallback=false
        )
        m_red, _, _ = ValidateMetamodel.fit_gp_multistart(
            KernelType, X_tr[:, keep], y_tr;
            num_dims=length(keep),
            allow_unbounded_fallback=false
        )

        μ_full, _ = GaussianProcesses.predict_y(m_full, X_va')
        μ_red, _ = GaussianProcesses.predict_y(m_red, X_va[:, keep]')

        append!(oof_true, y_va)
        append!(oof_full, μ_full)
        append!(oof_red, μ_red)
        append!(oof_rows, va)

        push!(per_fold_removed, feature_columns[dropped])
        for j in dropped
            drop_counts[j] += 1
        end

        @printf("outer fold %d: removed %d/%d (%s)\n",
                k, length(dropped), d,
                isempty(dropped) ? "none" : join(feature_columns[dropped], ", "))
    end

    _r2(y,p) = 1 - sum((y .- p).^2) / max(sum((y .- mean(y)).^2), 1e-12)
    _rmse(y,p) = sqrt(mean((y .- p).^2))
    _mae(y,p) = mean(abs.(y .- p))

    full_r2, red_r2 = _r2(oof_true,oof_full), _r2(oof_true,oof_red)
    full_rmse, red_rmse = _rmse(oof_true,oof_full), _rmse(oof_true,oof_red)
    full_mae, red_mae = _mae(oof_true,oof_full), _mae(oof_true,oof_red)

    n_low = max(3, ceil(Int, low_frac * length(oof_true)))
    low_idx = sortperm(raw_loss[oof_rows])[1:n_low]

    full_low_rmse = _rmse(oof_true[low_idx], oof_full[low_idx])
    red_low_rmse = _rmse(oof_true[low_idx], oof_red[low_idx])
    full_low_mae = _mae(oof_true[low_idx], oof_full[low_idx])
    red_low_mae = _mae(oof_true[low_idx], oof_red[low_idx])

    return Dict(
        "usable" => true,
        "n_folds_used" => nf,
        "n_folds_expected" => nf,
        "kernel" => string(KernelType),
        "inner_folds" => inner_folds,
        "feature_columns" => feature_columns,
        "drop_counts" => drop_counts,
        "per_fold_removed" => per_fold_removed,
        "full_r2" => full_r2, "reduced_r2" => red_r2,
        "full_rmse" => full_rmse, "reduced_rmse" => red_rmse,
        "full_mae" => full_mae, "reduced_mae" => red_mae,
        "n_low" => n_low,
        "full_low_rmse" => full_low_rmse, "reduced_low_rmse" => red_low_rmse,
        "full_low_mae" => full_low_mae, "reduced_low_mae" => red_low_mae,
        "oof_true" => oof_true, "oof_full" => oof_full, "oof_reduced" => oof_red,
        "oof_rows" => oof_rows, "low_indices" => collect(low_idx)
    )
end


function assess_robustness(
    cv_ablation::Dict,
    baseline_removed::Vector{String}
    )
    factors = String[]

    n = cv_ablation["n_folds_used"]
    required = ceil(Int, ValidateMetamodel.SCREEN_MIN_FRAC * n)
    features = Vector{String}(cv_ablation["feature_columns"])
    counts = Vector{Int}(cv_ablation["drop_counts"])

    selection_stable = true
    stability_rows = Dict{String, Any}[]

    for name in baseline_removed
        j = findfirst(==(name), features)
        j === nothing && error("Baseline removed parameter $name not found in cross-fit features")
        votes = counts[j]
        stable = votes >= required
        selection_stable &= stable
        push!(stability_rows, Dict(
            "parameter" => name,
            "outer_drop_votes" => votes,
            "outer_folds" => n,
            "votes_required" => required,
            "stable" => stable
        ))
        push!(factors, "$(stable ? "✅" : "❌") $name outer-fold selection = $votes/$n (need $required/$n)")
    end

    abs_r2_drop = cv_ablation["full_r2"] - cv_ablation["reduced_r2"]
    rel_rmse = (cv_ablation["reduced_rmse"] - cv_ablation["full_rmse"]) /
               max(cv_ablation["full_rmse"], 1e-12)
    rel_mae = (cv_ablation["reduced_mae"] - cv_ablation["full_mae"]) /
              max(cv_ablation["full_mae"], 1e-12)

    r2_ok = abs_r2_drop <= ROBUST_ABS_R2_DROP
    rmse_ok = rel_rmse <= ROBUST_REL_RMSE_INCREASE
    mae_ok = rel_mae <= ROBUST_REL_MAE_INCREASE

    push!(factors, "$(r2_ok ? "✅" : "❌") cross-fitted absolute R2 drop = $(round(abs_r2_drop,digits=4))")
    push!(factors, "$(rmse_ok ? "✅" : "❌") cross-fitted RMSE increase = $(round(100*rel_rmse,digits=1))%")
    push!(factors, "$(mae_ok ? "✅" : "❌") cross-fitted MAE increase = $(round(100*rel_mae,digits=1))%")

    rel_low = (cv_ablation["reduced_low_rmse"] - cv_ablation["full_low_rmse"]) /
              max(cv_ablation["full_low_rmse"], 1e-12)
    low_warning = rel_low > ROBUST_LOW_RMSE_WARNING
    push!(factors, "$(low_warning ? "⚠️" : "ℹ️") low-mismatch OOF RMSE increase = $(round(100*rel_low,digits=1))% (diagnostic only)")

    pass = selection_stable && r2_ok && rmse_ok && mae_ok

    return Dict(
        "pass" => pass,
        "selection_stable" => selection_stable,
        "stability_rows" => stability_rows,
        "r2_ok" => r2_ok,
        "rmse_ok" => rmse_ok,
        "mae_ok" => mae_ok,
        "low_mismatch_warning" => low_warning,
        "factors" => factors
    )
end


function main()
    metric_type = ValidateMetamodel.METRIC_TYPE
    apply_transform = ValidateMetamodel.AUTO_TRANSFORM

    script_dir = @__DIR__
    save_dir = joinpath(script_dir, ValidateMetamodel.SAVE_DIR)

    # Read 04 production outputs from save_dir, but keep all 04b artifacts
    # in a separate directory so downstream production code (05) remains
    # completely decoupled from robustness diagnostics.
    robustness_dir = joinpath(save_dir, "active_selection_robustness")
    mkpath(robustness_dir)

    summary_path = joinpath(save_dir, "final_performance_summary.csv")
    split_path = joinpath(save_dir, "master_data_split.json")
    fold_path = joinpath(save_dir, "master_fold_assignments.json")
    active_path = joinpath(save_dir, "active_parameters.json")

    for p in (summary_path, split_path, fold_path, active_path)
        isfile(p) || error("Required 04 output not found: $p")
    end

    df = CSV.read(summary_path, DataFrame)
    split = JSON.parsefile(split_path)
    folds = JSON.parsefile(fold_path)
    active = JSON.parsefile(active_path)

    train_idx = Int.(split["train_indices"])
    df_train = df[train_idx, :]
    fold_assignments = Int.(folds["fold_assignments"])

    full_features = String.(active["original_parameters"])
    active_features = String.(active["active_parameters"])
    removed_features = String.(active["removed_parameters"])
    KernelType = ValidateMetamodel.parse_kernel_type(String(active["kernel_family"]))

    println("\n🚀 Active-parameter robustness suite")
    println("Baseline active dimensions: $(length(active_features))/$(length(full_features))")
    println("Baseline removed: $(isempty(removed_features) ? "none" : join(removed_features, ", "))")
    println("Kernel family: $(KernelType)")

    bound = run_ard_bound_sensitivity(
        df_train, metric_type, apply_transform, fold_assignments,
        KernelType, full_features
    )

    cf = run_cross_fitted_ablation(
        df_train, metric_type, apply_transform, fold_assignments,
        KernelType, full_features
    )

    assessment = assess_robustness(cf, removed_features)

    # Bound-set stability: compare each sensitivity active set with the baseline 1x set.
    baseline_mask = Vector{Bool}(bound["results"]["1.0x"]["screening_info"]["candidate_inactive_mask"])
    reproduced_removed = sort(full_features[findall(baseline_mask)])
    saved_removed = sort(removed_features)
    baseline_reproduced = reproduced_removed == saved_removed

    bound_rows = Dict{String, Any}[]
    bound_stable = true

    for key in ["0.5x", "1.0x", "2.0x"]
        sc = bound["results"][key]["screening_info"]
        mask = Vector{Bool}(sc["candidate_inactive_mask"])
        same = mask == baseline_mask
        key == "1.0x" || (bound_stable &= same)
        push!(bound_rows, Dict(
            "bound_multiplier" => key,
            "candidate_inactive" => full_features[findall(mask)],
            "same_as_1x" => same
        ))
    end

    overall_pass = baseline_reproduced && assessment["pass"] && bound_stable

    println("\n" * "="^80)
    println("ACTIVE-SELECTION ROBUSTNESS RESULT")
    println("="^80)
    for f in assessment["factors"]
        println("   $f")
    end
    println("   $(baseline_reproduced ? "✅" : "❌") 1.0x screening reproduces the active set saved by 04")
    println("   $(bound_stable ? "✅" : "❌") 0.5x / 1x / 2x bound candidate sets stable")
    println("\nOverall status: $(overall_pass ? "PASS" : "FAIL / REVIEW")")
    println("04 production files are not modified by this script.")
    println("="^80)

    out = Dict(
        "artifact_role" => "active_parameter_selection_robustness",
        "production_outputs_modified" => false,
        "production_save_dir" => save_dir,
        "robustness_output_dir" => robustness_dir,

        "overall_status" => overall_pass ? "PASS" : "FAIL_REVIEW",
        "overall_pass" => overall_pass,

        "baseline_active_parameters" => active_features,
        "baseline_removed_parameters" => removed_features,
        "kernel_family" => String(active["kernel_family"]),
        "screening_rule" => "fixed-kernel 5-fold ARD; candidate inactive at upper bound in >=4/5 folds",

        "baseline_reproduced" => baseline_reproduced,
        "baseline_reproduced_removed" => reproduced_removed,
        "bound_sensitivity_pass" => bound_stable,
        "bound_sensitivity" => bound,
        "cross_fitted_ablation" => cf,
        "selection_assessment" => assessment,

        "supplementary_tables" => Dict(
            "ard_bound_sensitivity" => "ard_bound_sensitivity.csv",
            "cross_fitted_selection_stability" => "cross_fitted_selection_stability.csv"
        ),
        "timestamp" => string(Dates.now())
    )

    out_path = joinpath(robustness_dir, "robustness_summary.json")
    open(out_path, "w") do f
        JSON.print(f, out, 2)
    end

    # Compact CSVs for supplementary tables.
    bound_df = DataFrame(
        Parameter = full_features
    )
    for key in ["0.5x", "1.0x", "2.0x"]
        sc = bound["results"][key]["screening_info"]
        bound_df[!, Symbol("votes_" * replace(key, "." => "p"))] = sc["ceiling_votes"]
        bound_df[!, Symbol("candidate_" * replace(key, "." => "p"))] = sc["candidate_inactive_mask"]
    end
    CSV.write(joinpath(robustness_dir, "ard_bound_sensitivity.csv"), bound_df)

    stab_df = DataFrame(
        Parameter = full_features,
        OuterDropVotes = cf["drop_counts"],
        OuterFolds = fill(cf["n_folds_used"], length(full_features)),
        BaselineRemoved = [p in removed_features for p in full_features]
    )
    CSV.write(joinpath(robustness_dir, "cross_fitted_selection_stability.csv"), stab_df)

    println("✅ Robustness outputs saved to: $robustness_dir")
    println("   - robustness_summary.json")
    println("   - ard_bound_sensitivity.csv")
    println("   - cross_fitted_selection_stability.csv")
    println("   Production outputs in $save_dir were not modified.")
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

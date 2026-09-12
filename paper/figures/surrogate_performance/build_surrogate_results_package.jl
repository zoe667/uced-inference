#!/usr/bin/env julia

# Reporting-only extraction for the Results subsection
# "Surrogate Performance and Relevant Dimensions".
#
# This script does not fit, select, or screen a model. It reads the split and
# models already produced by 04_build_surrogate.jl, evaluates both saved models
# on the same frozen held-out UCED rows, returns predictions to the original
# aggregate-loss scale, and exports the fold-wise ARD screening saved by 04.

using CSV
using DataFrames
using GaussianProcesses
using JLD2
using JSON
using Statistics
using StatsBase

const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))

include(joinpath(EXPERIMENT_DIR, "FoldManager.jl"))
using .FoldManager

const OUTPUT_DIR = @__DIR__

const RUNS = [
    (year=2016, directory=joinpath(ROOT, "results", "2016")),
    (year=2021, directory=joinpath(ROOT, "results", "2021")),
]

function value_mean_std(x)
    if x isa AbstractDict
        return Float64(x["mean"]), Float64(x["std"])
    end
    return Float64(getproperty(x, :mean)), Float64(getproperty(x, :std))
end

function standardize_features(df::DataFrame, features::Vector{String}, params)
    X = Matrix{Float64}(df[!, features])
    Z = similar(X)
    for j in axes(X, 2)
        mu, sigma = value_mean_std(params[j])
        Z[:, j] = sigma > 1e-10 ? (X[:, j] .- mu) ./ sigma : X[:, j] .- mu
    end
    return Z
end

function inverse_prediction(z::AbstractVector, transform_params, std_params)
    mu, sigma = value_mean_std(std_params)
    fold_params = Dict("fold_mean" => mu, "fold_std" => sigma)
    params = transform_params isa Dict ? transform_params : Dict(transform_params)
    return FoldManager.inverse_y_transform(Vector{Float64}(z), params, fold_params)
end

r2_score(y, yhat) = 1 - sum((y .- yhat) .^ 2) / sum((y .- mean(y)) .^ 2)
rmse(y, yhat) = sqrt(mean((y .- yhat) .^ 2))

function spearman_score(y, yhat)
    return cor(tiedrank(Vector{Float64}(y)), tiedrank(Vector{Float64}(yhat)))
end

function predict_original_scale(model_path::String, test_df::DataFrame)
    model_data = JLD2.load(model_path)
    features = Vector{String}(model_data["feature_columns"])
    X_test = standardize_features(test_df, features, model_data["scaling_params"])
    pred_std, _ = GaussianProcesses.predict_y(model_data["metamodel"], X_test')
    pred_loss = inverse_prediction(
        pred_std,
        model_data["transform_params"],
        model_data["final_std_params"],
    )
    return pred_loss, features
end

function extract_year(run)
    year = run.year
    directory = run.directory

    split = JSON.parsefile(joinpath(directory, "master_data_split.json"))
    validation = JSON.parsefile(joinpath(directory, "validation_summary.json"))
    active = JSON.parsefile(joinpath(directory, "active_parameters.json"))
    cv_reduced = JSON.parsefile(joinpath(directory, "cv_predictions_reduced.json"))
    cv_full = JSON.parsefile(joinpath(directory, "cv_predictions_full.json"))
    lengthscales = JSON.parsefile(joinpath(directory, "master_lengthscales.json"))

    full_df = CSV.read(joinpath(directory, "final_performance_summary.csv"), DataFrame)
    test_indices = Int.(split["test_indices"])
    test_df = full_df[test_indices, :]
    true_loss = Vector{Float64}(test_df[!, :total_loss])

    full_pred, full_features = predict_original_scale(
        joinpath(directory, "trained_metamodel_full.jld2"), test_df)
    reduced_pred, reduced_features = predict_original_scale(
        joinpath(directory, "trained_metamodel_reduced.jld2"), test_df)

    # Check the reduced predictions against the exact reporting predictions
    # saved by 04. This also verifies row ordering and inverse transformation.
    plotting = JSON.parsefile(joinpath(directory, "plotting_data_reduced.json"))
    saved_test = plotting["internal_test"]
    saved_true = inverse_prediction(
        Float64.(saved_test["true_y"]),
        plotting["preprocessing"]["transform_params"],
        plotting["preprocessing"]["std_params"],
    )
    saved_pred = inverse_prediction(
        Float64.(saved_test["pred_y"]),
        plotting["preprocessing"]["transform_params"],
        plotting["preprocessing"]["std_params"],
    )

    maximum(abs.(saved_true .- true_loss)) < 1e-12 ||
        error("$year: saved held-out truths do not match master_data_split.json")
    maximum(abs.(saved_pred .- reduced_pred)) < 1e-12 ||
        error("$year: regenerated reduced-GP predictions do not match 04 output")
    length(full_pred) == length(reduced_pred) == length(test_indices) ||
        error("$year: full/reduced test sizes differ")
    reduced_features == String.(active["active_parameters"]) ||
        error("$year: reduced model features differ from active_parameters.json")

    full_r2 = r2_score(true_loss, full_pred)
    reduced_r2 = r2_score(true_loss, reduced_pred)
    full_rmse = rmse(true_loss, full_pred)
    reduced_rmse = rmse(true_loss, reduced_pred)

    summary_row = (
        year=year,
        cv_r2=Float64(cv_reduced["overall_r2"]),
        cv_rmse_standardized=Float64(cv_reduced["overall_rmse"]),
        cv_fold_r2=join(Float64.(cv_reduced["fold_r2"]), ";"),
        n_folds=Int(validation["n_folds"]),
        test_r2=reduced_r2,
        test_rmse=reduced_rmse,
        test_spearman=spearman_score(true_loss, reduced_pred),
        n_test=length(test_indices),
        full_test_r2=full_r2,
        reduced_test_r2=reduced_r2,
        delta_test_r2=reduced_r2 - full_r2,
        full_test_rmse=full_rmse,
        reduced_test_rmse=reduced_rmse,
        delta_test_rmse=reduced_rmse - full_rmse,
        full_cv_r2=Float64(cv_full["overall_r2"]),
        reduced_cv_r2=Float64(cv_reduced["overall_r2"]),
        delta_cv_r2=Float64(cv_reduced["overall_r2"]) - Float64(cv_full["overall_r2"]),
        selected_kernel=String(active["kernel_family"]),
    )

    prediction_rows = DataFrame(
        year=fill(year, length(test_indices)),
        observation_id=string.(test_df[!, :run_id]),
        source_row=test_indices,
        uced_loss=true_loss,
        gp_predicted_loss=reduced_pred,
    )

    screening = lengthscales["official_full_dimension_screening"]
    parameters = String.(screening["feature_columns"])
    fold_log_lengthscales = screening["fold_log_lengthscales"]
    votes = Int.(screening["ceiling_votes"])
    inactive = Bool.(screening["candidate_inactive_mask"])
    log_upper = Float64(screening["log_ls_hi"])
    tolerance = Float64(lengthscales["screening_ceiling_tol"])
    upper = exp(log_upper)

    ard_rows = DataFrame(
        year=Int[], parameter=String[], fold=Int[],
        log_length_scale=Float64[], length_scale=Float64[], upper_bound=Float64[],
        hit_threshold=Float64[], hit_upper_bound=Bool[], total_hits=Int[], retained=Bool[],
    )
    for fold in eachindex(fold_log_lengthscales)
        fold_values = Float64.(fold_log_lengthscales[fold])
        for j in eachindex(parameters)
            log_ls = fold_values[j]
            push!(ard_rows, (
                year, parameters[j], fold, log_ls, exp(log_ls), upper,
                exp(log_upper - tolerance), log_ls >= log_upper - tolerance,
                votes[j], !inactive[j],
            ))
        end
    end

    return summary_row, prediction_rows, ard_rows
end

function main()
    mkpath(OUTPUT_DIR)

    summaries = NamedTuple[]
    predictions = DataFrame()
    ard = DataFrame()

    for run in RUNS
        summary, pred, screening = extract_year(run)
        push!(summaries, summary)
        predictions = isempty(predictions) ? pred : vcat(predictions, pred)
        ard = isempty(ard) ? screening : vcat(ard, screening)
    end

    summary_df = DataFrame(summaries)
    CSV.write(joinpath(OUTPUT_DIR, "surrogate_validation_summary.csv"), summary_df)
    CSV.write(joinpath(OUTPUT_DIR, "surrogate_holdout_predictions.csv"), predictions)
    CSV.write(joinpath(OUTPUT_DIR, "ard_fold_screening.csv"), ard)

    println(summary_df)
    println("\nWrote reporting package to: $OUTPUT_DIR")
end

main()

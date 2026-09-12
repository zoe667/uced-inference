#!/usr/bin/env julia

# Run exactly one candidate from revalidation_candidates_eps10.csv. This entry
# point is safe to launch concurrently because every role receives a separate
# UCED working directory. After all roles finish, rerun 05 with the six role
# names so it can reuse the caches and write the combined validation table.

using CSV
using DataFrames
using Dates
using JLD2
using JSON
using SHA

include("Back_validation.jl")
using .BackValidation

const METADATA_COLUMNS = Set([
    "role", "draw", "predicted_loss", "predicted_loss_lo95_lat",
    "predicted_loss_hi95_lat", "sigma_latent_std", "sigma_predictive_std",
    "in_S_eps",
])

function params_fingerprint(params::AbstractDict)
    payload = join(
        [string(k, "=", round(Float64(params[k]), digits=6))
         for k in sort(collect(keys(params)))],
        ";",
    )
    return bytes2hex(sha256(payload))
end

function epsilon_tag()
    tol = parse(Float64, get(ENV, "PRIMARY_TOL", "0.10"))
    return string("eps", lpad(round(Int, 100 * tol), 2, '0'))
end

function archive_name(role::String, draw::Int)
    suffix = role == "surrogate_best" ? "" : "_" * epsilon_tag()
    draw_suffix = draw == 1 ? "" : "_draw$(draw)"
    return "backcheck_$(role)$(suffix)$(draw_suffix)"
end

function has_complete_uced_outputs(run_dir::AbstractString)
    return all(1:BackValidation.NUM_WEEKS) do week
        week_dir = joinpath(run_dir, string(week))
        isfile(joinpath(week_dir, "vGENDISPATCH_results.csv")) &&
            isfile(joinpath(week_dir, "vFLOW_results.csv"))
    end
end

function main()
    length(ARGS) == 1 || error(
        "Usage: RUNS_SAVE_DIR=runs_2021_100 julia --project=.. " *
        "05b_run_revalidation_candidate.jl <role>")

    role = ARGS[1]
    run_dir = BackValidation.PathConfig.RUN_DIR
    candidate_file = joinpath(run_dir, "revalidation_candidates_eps10.csv")
    isfile(candidate_file) || error("Candidate file not found: $candidate_file")

    candidates = CSV.read(candidate_file, DataFrame)
    rows = candidates[String.(candidates.role) .== role, :]
    nrow(rows) == 1 || error("Expected exactly one candidate for role '$role'; found $(nrow(rows))")
    row = rows[1, :]
    draw = Int(row.draw)

    params = Dict{String,Float64}()
    for column in names(candidates)
        column in METADATA_COLUMNS && continue
        params[column] = Float64(row[Symbol(column)])
    end

    archive_dir = joinpath(run_dir, "back_check_archive", archive_name(role, draw))
    cache_file = joinpath(archive_dir, "backcheck_result.json")
    fingerprint = params_fingerprint(params)
    force = lowercase(get(ENV, "REVAL_FORCE", "false")) == "true"
    recover_only = lowercase(get(ENV, "REVAL_RECOVER_ONLY", "false")) == "true"

    if isfile(cache_file) && !force
        cached = JSON.parsefile(cache_file)
        if get(cached, "params_fingerprint", "") == fingerprint
            println("Cached result already matches $role: $cache_file")
            return
        end
    end

    work_dir = abspath(get(
        ENV,
        "REVAL_WORK_DIR",
        joinpath(run_dir, "back_check_work", archive_name(role, draw)),
    ))

    println("Role:       $role")
    println("Dataset:    $run_dir")
    println("Work dir:   $work_dir")
    println("Archive:    $archive_dir")
    println("Parameters:")
    for key in sort(collect(keys(params)))
        println("  $key = $(params[key])")
    end

    if lowercase(get(ENV, "REVAL_DRY_RUN", "false")) == "true"
        println("Dry run complete; UCED was not started.")
        return
    end

    model_data = JLD2.load(joinpath(run_dir, "trained_metamodel_reduced.jld2"))
    metric_type = String(get(model_data, "metric_type", "NRMSE"))
    transform_params = model_data["transform_params"]

    completed_outputs = has_complete_uced_outputs(work_dir)
    result_dir = if completed_outputs && !force
        println("Completed UCED outputs already exist; recovering analysis without rerunning: $work_dir")
        work_dir
    elseif recover_only
        error("REVAL_RECOVER_ONLY=true, but complete UCED outputs were not found in: $work_dir")
    else
        BackValidation.prepare_and_run_validation(params; run_dir=work_dir)
    end
    result = BackValidation.analyze_validation_results(
        String(result_dir), transform_params, metric_type)

    mkpath(archive_dir)
    open(cache_file, "w") do io
        JSON.print(io, Dict(
            "role" => role,
            "draw" => draw,
            "params_fingerprint" => fingerprint,
            "params" => params,
            "result" => result,
            "metric_type" => metric_type,
            "work_dir" => work_dir,
            "run_at" => string(Dates.now()),
        ), 2)
    end

    println("Completed $role")
    original_loss = result["performance_score"]
    println("Original-UCED loss: $original_loss")
    println("Cache: $cache_file")
end

main()

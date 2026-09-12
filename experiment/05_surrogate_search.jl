# 05_surrogate_search.jl -- offline surrogate exploration of the reduced active space
#
# No expected improvement, Bayesian optimization, feasibility screening, or local refinement.
#
#     validated reduced GP
#       -> dense reduced-space exploration over the ex-ante design domain
#       -> S_eps  (defined on the ORIGINAL total-loss scale)
#       -> marginal recoverability diagnostics
#       -> targeted UCED revalidation candidates
#
# Search-domain priority:
#   1. <run_dir>/lhs_design_domain.json (frozen with the dataset)
#   2. a domain reconstructed from parameters.csv and then frozen
#   3. ExperimentConfig as a fallback
# ExperimentConfig is mutable and shared across datasets, while the 2016 and 2021
# designs use different CHP_Retrofitted_Time_* bounds.
#
# Historical naming
# The active features in older reduced models use names such as
#     Coal_Retrofitted_Min_Power_300-660 / Coal_Retrofitted_Time_0-300
# whereas the current ExperimentConfig.jl and stage 01 use
#     CHP_Retrofitted_* / CHP_NonRetrofitted_*
# Coal_* names are treated as aliases for CHP_* names when reading the model.
# All generated tables use canonical CHP_* names, and the summary records the
# complete model-feature-to-canonical-name mapping.
#
# Stage 01 looks up parameters with
#       haskey(params_for_run, "CHP_Retrofitted_Min_Power_<size>")
# Passing historical Coal_* columns directly to stage 01 would therefore leave
# the unit parameters unchanged. Revalidation must use canonical CHP_* names.
#
# Time dimensions are continuous by default. Stage 01 currently applies
# Int(round(param)) before writing generator data; remove that rounding there if
# UCED accepts continuous minimum up/down times. Set INTEGER_TIME=true to search
# on an integer grid.
#
# Environment variables
#   SEARCH_SAVE_DIR / N_SEARCH_SAMPLES / SEARCH_SEED
#   NEAR_OPT_TOLS (default "0.05,0.10,0.20,0.30") / PRIMARY_TOL (default 0.10)
#   SURROGATE_MODEL_FILE / METRIC_TYPE / INTEGER_TIME / ALLOW_FULL_MODEL_FALLBACK
#   SUPP_DENSITY / SUPP_GEOMETRY
#
#   UCED back-validation (via Back_validation.jl)
#     REVAL_AUTO          ask (default) | skip | best | all | role selection
#     REVAL_INACTIVE_REF  midpoint (default) | q25 | q75
#     REVAL_N_DRAWS       default 1; additional draws sample inactive parameters
#     REVAL_BV_NAMING     data (default) | canonical
#     REVAL_N_INTERIOR    number of additional interior candidates
#     REVAL_FORCE=true    ignore cached back-validation results
#
#   julia experiment/05_surrogate_search.jl 2>&1 | tee surrogate_search_$(date +%Y%m%d_%H%M).log
#
# ============================================================================
# Notes on interpretation
# ============================================================================
#
# sigma_latent_std is based on predict_f and is used for recoverability.
# sigma_predictive_std is based on predict_y and includes the nugget; use it when
# comparing surrogate predictions with UCED revalidation results. The fitted
# sigma_n may combine inactive-parameter variation, solver tolerance, and surface
# roughness, so it should not be assigned to one source without further evidence.
#
# L_min is the minimum of an estimated surface and may be biased downward. This
# can make S_eps too small and marginal recoverability intervals too narrow.
# Use UCED revalidation residuals, rather than global holdout RMSE, as the local
# error scale.
#
# Normalized IQR divides by each parameter's own ex-ante range. It is a marginal
# statistic and does not capture joint parameter constraints; SUPP_GEOMETRY adds
# pairwise diagnostics. Stability across epsilon values is more informative than
# any single result.
#
# Revalidation candidates are spot checks, not certification of the entire region.
# Legacy models without schema_version/gp_config may use an obsolete GP noise or
# ARD parameterization and should be regenerated with stage 04.
#
# Formal estimand and uncertainty statements are written to
# surrogate_search_summary.json.
# ============================================================================

using DataFrames, CSV, GaussianProcesses, Printf, Random, Statistics, JSON, JLD2,
      LinearAlgebra, Dates, SHA

include("PathConfig.jl")
include("FoldManager.jl")
include("ExperimentConfig.jl")
using .PathConfig
using .FoldManager
using .ExperimentConfig

# Back_validation provides prepare_and_run_validation and analyze_validation_results.
# The search remains usable when that optional file is absent.
if isfile(joinpath(@__DIR__, "Back_validation.jl"))
    include("Back_validation.jl")
    using .BackValidation
end

const SCRIPT_DIR = PathConfig.SCRIPT_DIR
const SAVE_DIR   = get(ENV, "SEARCH_SAVE_DIR", PathConfig.SAVE_DIR)

const N_SEARCH_SAMPLES = parse(Int,     get(ENV, "N_SEARCH_SAMPLES", "20000"))
const SEARCH_SEED      = parse(Int,     get(ENV, "SEARCH_SEED", "123"))
const PRIMARY_TOL      = parse(Float64, get(ENV, "PRIMARY_TOL", "0.10"))
const NEAR_OPT_TOLS    = sort(unique(vcat(
    [parse(Float64, s) for s in split(get(ENV, "NEAR_OPT_TOLS", "0.05,0.10,0.20,0.30"), ",")],
    PRIMARY_TOL)))

# Revalidation design
const REVAL_INACTIVE_REF = get(ENV, "REVAL_INACTIVE_REF", "midpoint")   # midpoint | q25 | q75
const REVAL_N_DRAWS      = parse(Int, get(ENV, "REVAL_N_DRAWS", "1"))
const REVAL_AUTO         = get(ENV, "REVAL_AUTO", "ask")               # ask | skip | best | all
const REVAL_BV_NAMING    = get(ENV, "REVAL_BV_NAMING", "data")         # data | canonical
const REVAL_FORCE        = get(ENV, "REVAL_FORCE", "false") == "true"  # Ignore cache.

const INTEGER_TIME      = get(ENV, "INTEGER_TIME", "false")      == "true"
const SUPP_DENSITY      = get(ENV, "SUPP_DENSITY", "true")      == "true"
const SUPP_GEOMETRY     = get(ENV, "SUPP_GEOMETRY", "true")     == "true"

const ROUNDTRIP_RTOL = 1e-8
const EPS_TAG = string("eps", lpad(round(Int, 100 * PRIMARY_TOL), 2, '0'))   # eps10 / eps20
const EPS_FREE_ROLES = Set(["surrogate_best"])

hr(c="=") = println(repeat(c, 78))
DOMAIN_PROVENANCE = "ExperimentConfig.jl (mutable, shared across datasets)"

# ============================================================================
# 2. Search domain
# ============================================================================

struct ParamSpec
    name::String        # Canonical name used by ExperimentConfig, stage 01, and outputs.
    data_name::String   # Name stored in the training data/model; may use Coal_*.
    lower::Float64
    upper::Float64
    isinteger::Bool
    unit::String
end

# Historical-data aliases for current configuration names.
const NAME_ALIASES = [
    "Coal_Retrofitted_"       => "CHP_Retrofitted_",
    "Coal_NonRetrofitted_"    => "CHP_NonRetrofitted_",
    "Coal_Non_Retrofitted_"   => "CHP_NonRetrofitted_",
]

function canonical_name(nm::AbstractString)
    for (old, new) in NAME_ALIASES
        startswith(nm, old) && return new * nm[nextind(nm, length(old)):end]
    end
    return String(nm)
end


"""
    load_frozen_domain(runs_path)

Load <run_dir>/lhs_design_domain.json when available. It is frozen with the
dataset and is unaffected by later ExperimentConfig changes.
"""
function load_frozen_domain(runs_path::String)
    p = joinpath(runs_path, "lhs_design_domain.json")
    isfile(p) || return nothing
    raw = JSON.parsefile(p)
    haskey(raw, "parameters") || return nothing
    bounds = Dict{String,Tuple{Float64,Float64,Bool,String}}()
    for q in raw["parameters"]
        nm = String(q["name"])
        isint = lowercase(String(get(q, "type", "continuous"))) == "integer"
        bounds[canonical_name(nm)] =
            (Float64(q["lower"]), Float64(q["upper"]),
             isint && INTEGER_TIME, occursin("Time", nm) ? "hours" : "fraction")
    end
    prov = haskey(raw, "recovered_from") ? "RECONSTRUCTED from parameters.csv" :
                                           "recorded at design time"
    println("\n  Design domain: $(basename(p))   design_id = $(get(raw, "design_id", "?"))" *
            "   n = $(get(raw, "n_design_points", "?"))")
    println("     provenance: $(prov)")
    global DOMAIN_PROVENANCE = prov
    return bounds
end

const GRIDS = [1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.025, 0.01, 0.005, 0.001]

"""Nicest value inside [lo, hi]: the coarsest grid that lands in the interval wins."""
function nicest(lo::Float64, hi::Float64)
    for g in GRIDS
        k0 = ceil(Int, lo / g - 1e-9); k1 = floor(Int, hi / g + 1e-9)
        k0 <= k1 || continue
        cands = [k * g for k in k0:k1]
        return cands[argmin(abs.(cands))]
    end
    return (lo + hi) / 2
end

"""
    recover_domain_from_parameters(runs_path)

Recover the ex-ante design domain from the realized LHS in parameters.csv, then freeze
it to lhs_design_domain.json.

Stratified LHS places exactly one sample per stratum, so u_min < 1/n and
u_max >= (n-1)/n. The true bounds are therefore pinned to an interval about one
range-percent wide at n = 100, and rounding to the simplest value inside that interval
recovers the design SPECIFICATION. This is not "use the realized min/max as the domain":
the realized range is only the evidence, the recovered grid value is the estimate.

ExperimentConfig cannot serve this role. It is mutable and shared across datasets, while
this project's 2016 and 2021 designs genuinely differ on the three retrofitted
minimum-run-time parameters.
"""
function recover_domain_from_parameters(runs_path::String)
    p = joinpath(runs_path, "parameters.csv")
    isfile(p) || return nothing
    df  = CSV.read(p, DataFrame)
    hdr = String.(names(df))
    i1 = findfirst(==("run_id"), hdr); i2 = findfirst(==("Description"), hdr)
    (i1 === nothing || i2 === nothing || i2 <= i1 + 1) && return nothing
    pcols = hdr[(i1+1):(i2-1)]

    lhs = hasproperty(df, :Description) ?
          df[.!ismissing.(df.Description) .& (strip.(string.(df.Description)) .!= ""), :] : df
    n = nrow(lhs)
    if n < 20
        @warn "Only $(n) LHS samples are available; domain reconstruction is too uncertain."
        return nothing
    end

    hr("-")
    println("DESIGN DOMAIN RECOVERED FROM parameters.csv  (LHS stratification inversion)")
    hr("-")
    @printf("  n = %d   reconstruction interval width = %.1f%% of range\n", n, 100 / n)
    @printf("  %-38s %20s   %s\n", "parameter", "realized [min,max]", "recovered [lower, upper]")

    bounds = Dict{String,Tuple{Float64,Float64,Bool,String}}()
    params = Vector{Dict{String,Any}}()
    for c in pcols
        v = Vector{Float64}(collect(skipmissing(lhs[!, Symbol(c)])))
        isempty(v) && continue
        vmin, vmax = minimum(v), maximum(v)
        w_hi = (vmax - vmin) * n / (n - 2)
        a = nicest(vmin - w_hi / n, vmin)
        b = nicest(vmax, vmax + w_hi / n)
        cn = canonical_name(c)
        is_time = occursin("Time", c)
        bounds[cn] = (a, b, is_time && INTEGER_TIME, is_time ? "hours" : "fraction")
        @printf("  %-38s [%7.3f,%7.3f]   [%g, %g]\n", cn, vmin, vmax, a, b)
        push!(params, Dict{String,Any}("name" => cn, "lower" => a, "upper" => b,
            "type" => is_time ? "integer" : "continuous",
            "realized_min" => vmin, "realized_max" => vmax))
    end
    isempty(bounds) && return nothing

    outp = joinpath(runs_path, "lhs_design_domain.json")
    open(outp, "w") do f
        JSON.print(f, Dict{String,Any}(
            "design_id" => basename(runs_path), "n_design_points" => n,
            "recovered_from" => "parameters.csv (LHS stratification inversion)",
            "recovered_at" => string(Dates.now()),
            "note" => string("Bounds were reconstructed from the realized LHS design, ",
                "exploiting the stratification property, then rounded to the simplest value ",
                "inside the feasible interval. This recovers the ex-ante design ",
                "specification; it is NOT the realized min/max used as a domain."),
            "parameters" => params), 2)
    end
    println("  Frozen domain written to $(basename(outp)); verify the table before reporting results.")
    global DOMAIN_PROVENANCE = "RECONSTRUCTED from parameters.csv, frozen to lhs_design_domain.json"
    return bounds
end

"""
Align model feature columns with the ex-ante domain, resolving historical aliases.
Unresolved names are rejected rather than assigned inferred bounds.
"""
function align_domain(bounds::Dict{String,Tuple{Float64,Float64,Bool,String}},
                      feature_names::Vector{String})
    specs = ParamSpec[]
    unresolved = String[]
    aliased = Tuple{String,String}[]

    for f in feature_names
        c = canonical_name(f)
        if haskey(bounds, c)
            lo, hi, isint, unit = bounds[c]
            push!(specs, ParamSpec(c, f, lo, hi, isint, unit))
            c != f && push!(aliased, (f, c))
        else
            push!(unresolved, f)
        end
    end

    isempty(unresolved) || error(string(
        "Active model features are missing from the design domain: ", join(unresolved, ", "), "\n",
        "Known canonical names: ", join(sort(collect(keys(bounds))), ", "), "\n",
        "Update NAME_ALIASES for a naming change, or restore the matching historical domain."))

    if !isempty(aliased)
        println("\n  Resolved historical feature aliases:")
        for (a, b) in aliased
            println("       $(a)  ->  $(b)")
        end
        println("     Outputs use canonical names.")
    end
    return specs, aliased
end

"""
Check that realized samples fall inside the selected ex-ante domain. Realized
minimum and maximum values are used only as a consistency check, never as search
bounds.
"""
function check_bounds_consistency(specs::Vector{ParamSpec}, runs_path::String)
    hr("-"); println("SEARCH DOMAIN CONSISTENCY CHECK  (config bounds vs. realized samples)"); hr("-")

    p = joinpath(runs_path, "parameters.csv")
    if !isfile(p)
        @warn "$(p) was not found; search-domain consistency could not be checked."
        return
    end
    df = CSV.read(p, DataFrame)

    # Exclude non-LHS rows, such as manually inserted baseline cases.
    has_desc = hasproperty(df, :Description)
    lhs_rows = has_desc ? df[.!ismissing.(df.Description) .& (strip.(string.(df.Description)) .!= ""), :] : df
    nrow(lhs_rows) == 0 && (lhs_rows = df)

    tol = 1e-6
    bad = NamedTuple[]
    for s in specs
        hasproperty(lhs_rows, Symbol(s.data_name)) || continue
        v = collect(skipmissing(lhs_rows[!, Symbol(s.data_name)]))
        isempty(v) && continue
        rlo, rhi = minimum(v), maximum(v)
        if rlo < s.lower - tol || rhi > s.upper + tol
            push!(bad, (name=s.name, data_name=s.data_name,
                       realized=(rlo, rhi), config=(s.lower, s.upper)))
        end
    end

    if isempty(bad)
        println("  All $(length(specs)) active-parameter ranges are inside the search domain.")
        return
    end

    println("  Realized ranges outside the search domain:\n")
    @printf("    %-38s %22s   %22s\n", "parameter", "realized [min, max]", "domain [lower, upper]")
    for b in bad
        @printf("    %-38s [%8.3f, %8.3f]   [%8.3f, %8.3f]\n",
                b.name, b.realized[1], b.realized[2], b.config[1], b.config[2])
    end

    if startswith(DOMAIN_PROVENANCE, "ExperimentConfig")
        @warn "ExperimentConfig does not match the realized design. Reconstruct and freeze the domain from parameters.csv."
    else
        error("""

            The frozen design domain conflicts with parameters.csv.
            Reconstruct lhs_design_domain.json or restore the matching input data.
            """)
    end
end

# ============================================================================
# 3. Surrogate loading and metadata validation
# ============================================================================

scaling_mean_std(p) = p isa AbstractDict ? (Float64(p["mean"]), Float64(p["std"])) :
                                           (Float64(p.mean),   Float64(p.std))
std_params_mean_std(p) = scaling_mean_std(p)

function fitted_noise_std(gp)
    ln = gp.logNoise
    v = isa(ln, Real) ? float(ln) : float(first(GaussianProcesses.get_params(ln)))
    return exp(v)
end

function fitted_signal_std(gp)
    k = gp.kernel
    if isa(k, GaussianProcesses.SumKernel)
        for sub in (k.kleft, k.kright)
            hasfield(typeof(sub), :σ2) && return sqrt(sub.σ2)
        end
    end
    return hasfield(typeof(k), :σ2) ? sqrt(k.σ2) : NaN
end

const ALLOW_FULL_MODEL_FALLBACK = get(ENV, "ALLOW_FULL_MODEL_FALLBACK", "false") == "true"

"""
Load a trained surrogate. Falling back from a reduced model to the full model
changes the search estimand and therefore requires explicit authorization through
ALLOW_FULL_MODEL_FALLBACK=true.
"""
function load_surrogate(runs_path::String)
    explicit = get(ENV, "SURROGATE_MODEL_FILE", "")

    if !isempty(explicit)
        model_path = isabspath(explicit) ? explicit : joinpath(runs_path, explicit)
        isfile(model_path) || error("SURROGATE_MODEL_FILE does not exist: $(model_path)")
    else
        reduced_path = joinpath(runs_path, "trained_metamodel_reduced.jld2")
        full_path    = joinpath(runs_path, "trained_metamodel_full.jld2")
        plain_path   = joinpath(runs_path, "trained_metamodel.jld2")

        if isfile(reduced_path)
            model_path = reduced_path
        elseif isfile(full_path)
            ALLOW_FULL_MODEL_FALLBACK || error(
                "Reduced surrogate not found. A full surrogate is available, but it changes " *
                "the search from the ARD-selected subspace to all feature dimensions. " *
                "Set ALLOW_FULL_MODEL_FALLBACK=true to use it, or rerun stage 04.")
            model_path = full_path
            @warn "Using the full surrogate because ALLOW_FULL_MODEL_FALLBACK=true; the search is not restricted to an ARD-selected subspace."
        elseif isfile(plain_path)
            model_path = plain_path
        else
            error("No trained surrogate was found in $(runs_path).")
        end
    end

    md = load(model_path)
    req = ["metamodel", "scaling_params", "feature_columns", "transform_params", "final_std_params"]
    mk = [k for k in req if !haskey(md, k)]
    isempty(mk) || error("Surrogate file is missing required fields: $(join(mk, ", "))")

    n_feat = length(md["feature_columns"])
    mtype  = get(md, "model_type", "?")
    hr("*")
    println("  Surrogate: $(basename(model_path))")
    println("     model_type = $(mtype)   |   search dimension d = $(n_feat)")
    if mtype == "full"
        println("     Full-model search over all $(n_feat) feature dimensions.")
    end
    hr("*")

    schema = get(md, "schema_version", nothing)
    gpcfg  = get(md, "gp_config", nothing)
    is_sum = isa(md["metamodel"].kernel, GaussianProcesses.SumKernel)

    if is_sum
        @warn "The surrogate uses SumKernel; its sigma and ARD results are not interpretable. Rerun stage 04."
    elseif schema === nothing || gpcfg === nothing
        @warn "The surrogate is missing schema_version or gp_config. Rerun stage 04 to record complete provenance."
    end
    return md, model_path
end

# ============================================================================
# 4. Prediction and inverse transformation
# ============================================================================

function standardize_matrix(X::AbstractMatrix{Float64}, scaling_params)
    n, d = size(X)
    Z = Matrix{Float64}(undef, n, d)
    for j in 1:d
        mu, sd = scaling_mean_std(scaling_params[j])
        Z[:, j] = sd > 1e-10 ? (X[:, j] .- mu) ./ sd : (X[:, j] .- mu)
    end
    return Z
end

"""
Return posterior mean, latent standard deviation, and predictive standard
deviation on the standardized scale.
"""
function predict_batch(gp, X::AbstractMatrix{Float64}, scaling_params)
    Xt = Matrix(standardize_matrix(X, scaling_params)')          # d x n
    mu_s, var_f = GaussianProcesses.predict_f(gp, Xt)
    _,    var_y = GaussianProcesses.predict_y(gp, Xt)
    return mu_s, sqrt.(max.(var_f, 0.0)), sqrt.(max.(var_y, 0.0))
end

"""
Map standardized predictions to the original total-loss scale using
FoldManager.inverse_y_transform. Invalid points in the inverse Box-Cox domain
are returned as NaN.
"""
function make_inverter(transform_params, final_std_params)
    m, s = std_params_mean_std(final_std_params)
    shim = Dict("fold_mean" => m, "fold_std" => s)
    applied = get(transform_params, "transformation_applied", false) === true
    lam = applied ? Float64(transform_params["lambda"]) : NaN

    function invert(z_std::AbstractVector{<:Real})
        zs = collect(Float64, z_std)
        ok = trues(length(zs))
        if applied && abs(lam) >= 1e-8
            z = zs .* s .+ m                      # Used only for domain validation.
            ok = (lam .* z .+ 1.0) .> 0.0
        end
        L = fill(NaN, length(zs))
        any(ok) && (L[ok] = FoldManager.inverse_y_transform(zs[ok], transform_params, shim))
        return L
    end
    invert(z::Real) = invert([z])[1]
    return invert
end

function metric_columns(metric_type::String)
    metric_type == "NRMSE" ? ["nrmse_coal","nrmse_wind","nrmse_solar","nrmse_mlt"] :
    metric_type == "RMSE"  ? ["raw_rmse_coal","raw_rmse_wind","raw_rmse_solar","raw_rmse_mlt"] :
                             ["raw_mae_coal","raw_mae_wind","raw_mae_solar","raw_mae_mlt"]
end

"""
Verify the complete raw -> transform -> standardize -> inverse -> raw path using
final_performance_summary.csv before producing search results.
"""
function validate_inverse(invert, transform_params, final_std_params,
                          runs_path::String, metric_type::String)
    hr("-"); println("INVERSE TRANSFORM IDENTITY CHECK"); hr("-")

    csv = joinpath(runs_path, "final_performance_summary.csv")
    isfile(csv) || error("Cannot validate the inverse transformation; file not found: $(csv)")
    df = CSV.read(csv, DataFrame)
    cols = metric_columns(metric_type)
    for c in cols
        hasproperty(df, Symbol(c)) ||
            error("Missing column $(c) for metric_type=$(metric_type).")
    end
    c_, w_, s_, m_ = (Vector{Float64}(df[!, c]) for c in cols)

    z, _ = FoldManager.apply_y_transform(c_, w_, s_, m_, transform_params;
                                         fold_train_indices=nothing,
                                         apply_standardization=false, verbose=false)
    mu, sd = std_params_mean_std(final_std_params)
    L_back = invert((Vector{Float64}(z) .- mu) ./ sd)

    W = PathConfig.WEIGHTS
    L_true = metric_type == "NRMSE" ?
        (W["coal_gen"].*c_.^2 .+ W["wind_gen"].*w_.^2 .+
         W["solar_gen"].*s_.^2 .+ W["mlt_flow"].*m_.^2) :
        (W["coal_gen"].*abs.(c_) .+ W["wind_gen"].*abs.(w_) .+
         W["solar_gen"].*abs.(s_) .+ W["mlt_flow"].*abs.(m_))

    all(isfinite, L_back) ||
        error("The inverse transformation produced NaN or Inf on the training data.")
    rel = maximum(abs.(L_back .- L_true)) / max(maximum(L_true) - minimum(L_true), eps())

    applied = get(transform_params, "transformation_applied", false) === true
    lam_str = applied ? "  (lambda = $(round(Float64(transform_params["lambda"]), digits=4)))" : ""
    println("  transformation_applied = $(applied)$(lam_str)" *
            "   data_shift = $(transform_params["data_shift"])")
    @printf("  n = %d   maximum relative deviation = %.3e\n", length(L_true), rel)
    @printf("  total_loss range = [%.6g, %.6g]\n", minimum(L_true), maximum(L_true))

    rel <= ROUNDTRIP_RTOL || error(string(
        "Inverse-transform identity check failed: relative deviation ", rel,
        " > ", ROUNDTRIP_RTOL, "."))
    println("  Identity check passed.")

    applied || println("  Box-Cox transformation was not applied.")
end

# ============================================================================
# 5. Nugget and noise representation
# ============================================================================

function report_noise(gp, sig_lat::Vector{Float64}, sig_prd::Vector{Float64})
    hr("-"); println("NOISE REPRESENTATION"); hr("-")
    @printf("  fitted sigma_n (standardized) : %.4f\n", fitted_noise_std(gp))
    @printf("  fitted sigma_f (standardized) : %.4f\n", fitted_signal_std(gp))
    frac = (sig_lat .^ 2) ./ max.(sig_prd .^ 2, eps())
    @printf("  var_latent / var_predictive over pool : min %.3f | median %.3f | max %.3f\n",
            minimum(frac), median(frac), maximum(frac))
end

# ============================================================================
# §6  Dense pool
# ============================================================================

function lhs_unit(n::Int, d::Int, rng::AbstractRNG)
    U = Matrix{Float64}(undef, n, d)
    for j in 1:d
        p = randperm(rng, n)
        @inbounds for i in 1:n
            U[i, j] = (p[i] - rand(rng)) / n
        end
    end
    return U
end

function build_pool(specs::Vector{ParamSpec}, n::Int, seed::Int)
    d = length(specs)
    U = lhs_unit(n, d, MersenneTwister(seed))
    X = Matrix{Float64}(undef, n, d)
    for (j, s) in enumerate(specs)
        if s.isinteger
            lv = collect(ceil(Int, s.lower):floor(Int, s.upper))
            isempty(lv) &&
                error("No integer values are available for $(s.name) in [$(s.lower), $(s.upper)].")
            idx = clamp.(floor.(Int, U[:, j] .* length(lv)) .+ 1, 1, length(lv))
            X[:, j] = Float64.(lv[idx])
        else
            X[:, j] = s.lower .+ U[:, j] .* (s.upper - s.lower)
        end
    end
    return X
end

# ============================================================================
# 7. S_eps on the original scale and normalized marginal IQR
# ============================================================================

near_optimal(L, L_min, eps_) = (findall(<=(L_min * (1 + eps_)), L), L_min * (1 + eps_))

function recoverability(X, idx::Vector{Int}, specs::Vector{ParamSpec})
    map(enumerate(specs)) do (j, s)
        v = X[idx, j]
        q25, q50, q75 = quantile(v, 0.25), quantile(v, 0.50), quantile(v, 0.75)
        rg = s.upper - s.lower
        Dict{String,Any}("parameter" => s.name, "median" => q50, "q25" => q25, "q75" => q75,
            "iqr" => q75 - q25, "normalized_iqr" => (q75 - q25) / rg,
            "min" => minimum(v), "max" => maximum(v),
            "normalized_range" => (maximum(v) - minimum(v)) / rg,
            "ex_ante_lower" => s.lower, "ex_ante_upper" => s.upper)
    end
end

function eps_sweep(X, L, specs, tols, N)
    L_min = minimum(L)
    hr(); println("LOW-MISMATCH SET  S_eps   (original total-loss scale)"); hr()
    @printf("  L_min (surrogate median prediction) = %.6g\n", L_min)

    out = Dict{String,Any}()
    for tol in tols
        idx, cut = near_optimal(L, L_min, tol)
        st = recoverability(X, idx, specs)
        out[string(tol)] = Dict{String,Any}("epsilon" => tol, "cutoff" => cut,
            "n_members" => length(idx), "set_fraction" => length(idx) / N, "parameters" => st)
        println()
        @printf("  eps = %4.0f%%   cutoff = %.6g   |S| = %d / %d  (%.3f%% of pool)\n",
                100 * tol, cut, length(idx), N, 100 * length(idx) / N)
        @printf("    %-36s %10s %10s %10s\n", "parameter", "median", "IQR", "norm.IQR")
        for x in st
            @printf("    %-36s %10.4f %10.4f %10.4f\n",
                    x["parameter"], x["median"], x["iqr"], x["normalized_iqr"])
        end
    end
    return out
end

# ============================================================================
# 8. UCED revalidation candidates
# ============================================================================

"""Reference value for an inactive parameter within its ex-ante interval."""
function inactive_reference(lo::Float64, hi::Float64, mode::String)
    mode == "midpoint" && return (lo + hi) / 2
    mode == "q25"      && return lo + 0.25 * (hi - lo)
    mode == "q75"      && return lo + 0.75 * (hi - lo)
    error("REVAL_INACTIVE_REF must be midpoint, q25, or q75; received $(mode).")
end

"""
Return (data_name, canonical_name) pairs in parameters.csv order, or
alphabetical order when that file is unavailable.
"""
function design_column_order(runs_path::String, bounds)
    p = joinpath(runs_path, "parameters.csv")
    if isfile(p)
        hdr = String.(names(CSV.read(p, DataFrame; limit=0)))
        i1 = findfirst(==("run_id"), hdr)
        i2 = findfirst(==("Description"), hdr)
        if i1 !== nothing && i2 !== nothing && i2 > i1 + 1
            cols = [(h, canonical_name(h)) for h in hdr[(i1+1):(i2-1)]]
            unknown = [c for (_, c) in cols if !haskey(bounds, c)]
            isempty(unknown) ||
                error("parameters.csv columns are missing from the design domain: $(join(unknown, ", "))")
            return cols
        end
    end
    return [(k, k) for k in sort(collect(keys(bounds)))]
end

"""
Complete a candidate vector with inactive parameters. Draw 1 uses the selected
reference; later draws sample a shared inactive-parameter scenario.
"""
function fill_inactive(cols, specs::Vector{ParamSpec}, bounds, draw::Int, seed::Int)
    act = Set(s.name for s in specs)
    vals = Dict{String,Float64}()
    rng = MersenneTwister(seed + 1000 * draw)
    for (_, c) in cols
        c in act && continue
        lo, hi, isint, _ = bounds[c]
        v = draw == 1 ? inactive_reference(lo, hi, REVAL_INACTIVE_REF) :
                        lo + rand(rng) * (hi - lo)
        vals[c] = isint ? round(v) : round(v, digits=3)
    end
    return vals
end

function select_candidates(X, L, idx::Vector{Int}, specs::Vector{ParamSpec}, cutoff::Float64,
                           gp, scaling_params, invert; rng_seed::Int=SEARCH_SEED)
    d = length(specs)
    picks = Tuple{String,Int}[("surrogate_best", idx[argmin(L[idx])])]

    W = Matrix{Float64}(undef, length(idx), d)
    for (j, s) in enumerate(specs)
        W[:, j] = (X[idx, j] .- s.lower) ./ (s.upper - s.lower)
    end
    sub = size(W, 1) > 4000 ?
          sort(randperm(MersenneTwister(rng_seed), size(W, 1))[1:4000]) : collect(1:size(W, 1))
    c = vec(mean(W[sub, :], dims=1))
    push!(picks, ("medoid", idx[sub[argmin(vec(sum((W[sub, :] .- c') .^ 2, dims=2)))]]))

    # k interior points, stratified by distance to the centroid of S_eps.
    #
    # Selection uses position only, never Lhat, so these residuals estimate the
    # surrogate error without the winner's-curse contamination that affects
    # surrogate_best. One draw from each equal-sized distance quartile keeps the
    # sample spread from the centre to the edge of S_eps; equal strata means the
    # estimator stays unbiased for the mean over S_eps while having lower variance
    # than four unrestricted draws.
    kint = parse(Int, get(ENV, "REVAL_N_INTERIOR", "4"))
    if kint > 0 && length(idx) >= kint
        dist  = vec(sum((W .- c') .^ 2, dims=2))
        order = sortperm(dist)
        rr    = MersenneTwister(rng_seed + 77)
        m     = length(order)
        for t in 1:kint
            lo = floor(Int, (t - 1) * m / kint) + 1
            hi = floor(Int, t * m / kint)
            hi >= lo || continue
            push!(picks, (string("interior_", t), idx[order[lo + rand(rr, 0:(hi - lo))]]))
        end
    end

    for (j, s) in enumerate(specs)
        v = X[idx, j]
        push!(picks, (string("axis_min_", s.name), idx[argmin(v)]))
        push!(picks, (string("axis_max_", s.name), idx[argmax(v)]))
    end

    seen = Set{Vector{Float64}}(); roles = String[]; rows = Vector{Float64}[]
    for (role, i) in picks
        th = [specs[j].isinteger ? round(X[i, j]) : X[i, j] for j in 1:d]
        key = round.(th, digits=8)
        key in seen && continue
        push!(seen, key); push!(roles, role); push!(rows, th)
    end

    Xc = permutedims(hcat(rows...))
    mu_s, sl, sp = predict_batch(gp, Xc, scaling_params)
    Lc   = invert(mu_s)
    L_lo = invert(mu_s .- 1.96 .* sl)      # Back-transformed bounds are asymmetric.
    L_hi = invert(mu_s .+ 1.96 .* sl)

    df = DataFrame(role = roles)
    for (j, s) in enumerate(specs)
        df[!, Symbol(s.name)] = s.isinteger ? Int.(Xc[:, j]) : round.(Xc[:, j], digits=3)
    end
    df.predicted_loss          = Lc
    df.predicted_loss_lo95_lat = L_lo
    df.predicted_loss_hi95_lat = L_hi
    df.sigma_latent_std        = sl
    df.sigma_predictive_std    = sp
    df.in_S_eps                = Lc .<= cutoff
    return df
end

"""
    revalidation_table(cand, cols, specs, bounds; draws)

Expand each candidate and draw to a complete active-plus-inactive parameter
vector suitable for UCED back-validation.
"""
function revalidation_table(cand::DataFrame, cols, specs::Vector{ParamSpec}, bounds;
                            draws::Int=1, seed::Int=SEARCH_SEED)
    act = Set(s.name for s in specs)
    out = DataFrame()
    for d in 1:draws
        inact = fill_inactive(cols, specs, bounds, d, seed)
        blk = DataFrame(role = cand.role, draw = fill(d, nrow(cand)))
        for (_, c) in cols
            blk[!, Symbol(c)] = c in act ? cand[!, Symbol(c)] : fill(inact[c], nrow(cand))
        end
        for col in (:predicted_loss, :predicted_loss_lo95_lat, :predicted_loss_hi95_lat,
                    :sigma_latent_std, :sigma_predictive_std, :in_S_eps)
            blk[!, col] = cand[!, col]
        end
        out = d == 1 ? blk : vcat(out, blk)
    end
    return out
end

"""
    resolve_roles(spec, labels)

Resolve a comma-separated list of candidate indices, role names, or role prefixes.
"""
function resolve_roles(spec::AbstractString, labels::Vector{String})
    out = Int[]
    for tok in split(spec, ',')
        t = String(strip(tok))
        isempty(t) && continue
        i = tryparse(Int, t)
        if i !== nothing
            1 <= i <= length(labels) ? push!(out, i) :
                @warn "Candidate index $(t) is outside 1:$(length(labels))."
            continue
        end
        hits = findall(r -> r == t || startswith(r, t), labels)
        isempty(hits) ? @warn("Unknown candidate role: $(t)") : append!(out, hits)
    end
    return unique(out)
end

"""
    ask_back_validation(labels)

Return the selected UCED back-validation candidate indices. Non-interactive
sessions skip this step unless REVAL_AUTO specifies a selection.
"""
function ask_back_validation(labels::Vector{String})
    n = length(labels)
    if REVAL_AUTO != "ask"
        REVAL_AUTO == "skip" && return Int[]
        REVAL_AUTO == "best" && return [1]
        REVAL_AUTO == "all"  && return collect(1:n)
        return resolve_roles(REVAL_AUTO, labels)
    end

    println()
    hr("-")
    println("OPTIONAL: BACK-VALIDATION WITH FULL UCED MODEL")
    hr("-")
    for (i, l) in enumerate(labels)
        @printf("   %2d  %s\n", i, l)
    end
    println()
    println("  [n]  skip (default)")
    println("  [a]  all $(n) candidates")
    println("  Enter comma-separated indices or roles (for example: medoid, 1,2, or axis_min).")
    print("  > ")

    resp = try
        strip(readline())
    catch
        ""                                   # Skip when stdin is unavailable.
    end

    lr = lowercase(resp)
    (isempty(lr) || lr in ("n", "no", "skip")) && return Int[]
    lr in ("a", "all") && return collect(1:n)
    return resolve_roles(resp, labels)
end

"""
Stable fingerprint of the parameter vector actually sent to UCED.

Rounded to 6 decimals so float formatting noise does not invalidate a cache entry,
but any real change to a candidate's coordinates (a different PRIMARY_TOL, a new
pool seed, a different inactive reference) produces a different digest and forces
a fresh run.
"""
function params_fingerprint(params::AbstractDict)
    s = join([string(k, "=", round(Float64(params[k]), digits=6))
              for k in sort(collect(keys(params)))], ";")
    return bytes2hex(sha256(s))
end

backcheck_dir(runs_path::String, role::AbstractString, draw::Integer) =
    joinpath(runs_path, "back_check_archive",
             string("backcheck_", role,
                    role in EPS_FREE_ROLES ? "" : "_" * EPS_TAG,
                    draw == 1 ? "" : "_draw$(draw)"))

"""
Set REVAL_BV_NAMING=canonical to pass canonical CHP_* parameter names.
"""
function back_validate(reval::DataFrame, cols, specs::Vector{ParamSpec}, idxs::Vector{Int},
                       transform_params, metric_type::String, L_min::Float64,
                       runs_path::String)
    isempty(idxs) && return nothing
    if !@isdefined(BackValidation)
        @warn "Back_validation.jl was not found; UCED back-validation was skipped."
        return nothing
    end

    use_canon = (REVAL_BV_NAMING == "canonical")
    rows = NamedTuple[]
    n_reused = 0

    hr(); println("UCED BACK-VALIDATION"); hr()
    @printf("  candidates = %d   parameter naming = %s\n", length(idxs), use_canon ? "canonical" : "data")

    for i in idxs
        r = reval[i, :]
        params = Dict{String,Float64}()
        for (dname, cname) in cols
            params[use_canon ? cname : dname] = Float64(r[Symbol(cname)])
        end

        @printf("\n  [%d/%d] %s (draw %d)  predicted = %.6g\n",
                findfirst(==(i), idxs), length(idxs), r.role, r.draw, r.predicted_loss)

                dst   = backcheck_dir(runs_path, r.role, r.draw)
        cache = joinpath(dst, "backcheck_result.json")
        fp    = params_fingerprint(params)
        vr    = nothing

        # ---- reuse ----------------------------------------------------------
        if isfile(cache) && !REVAL_FORCE
            c = try JSON.parsefile(cache) catch; nothing end
            if c === nothing
                @warn "Invalid cache file; rerunning $(basename(dst))."
            elseif get(c, "params_fingerprint", "") != fp
                @warn "Cached parameters do not match the candidate; rerunning $(basename(dst))."
            elseif get(c, "metric_type", metric_type) != metric_type
                @warn "Cached metric_type does not match; rerunning $(basename(dst))."
            else
                vr = Dict{String,Any}(c["result"])
                n_reused += 1
                @printf("        ⏭  reuse %s   (no UCED run)\n", basename(dst))
            end
        end

        # ---- run ------------------------------------------------------------
        if vr === nothing
            dir = nothing
            try
                dir = BackValidation.prepare_and_run_validation(params)
            catch e
                @warn "prepare_and_run_validation failed for $(r.role): $(e)"
                continue
            end
            dir === nothing && (@warn "No run directory was returned for $(r.role)."; continue)

            # Analyse from the directory BackValidation just produced, before moving it.
            try
                vr = BackValidation.analyze_validation_results(dir, transform_params, metric_type)
            catch e
                @warn "analyze_validation_results failed for $(r.role): $(e)"
                continue
            end

            try
                mkpath(dirname(dst))
                isdir(dst) && rm(dst; recursive=true, force=true)
                cp(String(dir), dst; force=true)
            catch e
                @warn "Could not archive $(dir) to $(dst): $(e). The result remains valid but is not cached."
            end

            if isdir(dst)
                open(cache, "w") do f
                    JSON.print(f, Dict(
                        "role"               => r.role,
                        "draw"               => r.draw,
                        "params_fingerprint" => fp,
                        "params"             => Dict(string(k) => v for (k, v) in params),
                        "result"             => vr,
                        "metric_type"        => metric_type,
                        "run_at"             => string(Dates.now())), 2)
                end
            end
        end

        # ---- record ----------------------------------------------------------
        actual = Float64(vr["performance_score"])
        push!(rows, (role = r.role, draw = r.draw, dir = string(dst),
                     predicted = r.predicted_loss, actual = actual,
                     residual = actual - r.predicted_loss,
                     coal  = Float64(get(vr, "raw_coal_error",  NaN)),
                     wind  = Float64(get(vr, "raw_wind_error",  NaN)),
                     solar = Float64(get(vr, "raw_solar_error", NaN)),
                     mlt   = Float64(get(vr, "raw_mlt_error",   NaN))))
        @printf("        actual = %.6g   residual = %+.3e\n", actual, actual - r.predicted_loss)
    end

    isempty(rows) && (println("\n  No successful back-validation results."); return nothing)

    bv = DataFrame(rows)
    println()
    n_reused > 0 && @printf("  Reused %d/%d results from back_check_archive/.\n",
                            n_reused, length(rows))
    @printf("  %-38s %12s %12s %12s\n", "role", "predicted", "actual", "residual")
    for r in eachrow(bv)
        @printf("  %-38s %12.6g %12.6g %+12.3e\n", r.role, r.predicted, r.actual, r.residual)
    end

    rms  = sqrt(mean(bv.residual .^ 2))
    band = PRIMARY_TOL * L_min
    println()
    @printf("  residual RMS      = %.4g\n", rms)
    @printf("  eps*L_min         = %.4g\n", band)
    @printf("  RMS / (eps*L_min) = %.2f  ->  %s\n", rms / band,
            rms / band < 0.5 ? "epsilon is adequate for reporting R_q" :
            rms / band < 1.0 ? "epsilon is tight; also report a larger value" :
                               "epsilon is within the surrogate-error scale; do not report R_q")

    if nrow(bv) > 1 && length(unique(bv.draw)) > 1
        println("\n  Residual decomposition across inactive-parameter draws:")
        for g in groupby(bv, :role)
            nrow(g) < 2 && continue
            @printf("    %-36s mean %+.3e   sd %.3e\n",
                    g[1, :role], mean(g.residual), std(g.residual))
        end
    end
    return bv
end


# ============================================================================
# 9. Optional diagnostics
# ============================================================================

function supp_density(specs, gp, scaling_params, invert)
    hr(); println("SUPPLEMENT — density convergence"); hr()
    res = Dict{String,Any}()
    for n in (10_000, 20_000, 50_000)
        X = build_pool(specs, n, SEARCH_SEED)
        mu_s, _, _ = predict_batch(gp, X, scaling_params)
        L = replace(invert(mu_s), NaN => Inf)
        i0 = argmin(L)
        th = [(X[i0, j] - s.lower) / (s.upper - s.lower) for (j, s) in enumerate(specs)]
        idx, _ = near_optimal(L, minimum(L), PRIMARY_TOL)
        Rq = [x["normalized_iqr"] for x in recoverability(X, idx, specs)]
        res[string(n)] = Dict("theta_hat_normalized" => th, "L_min" => minimum(L),
                              "set_fraction" => length(idx)/n, "normalized_iqr" => Rq)
        @printf("  n=%6d  L_min=%.6g  |S|/N=%.4f  R_q=[%s]\n",
                n, minimum(L), length(idx)/n, join(round.(Rq, digits=3), ", "))
    end
    ref = res["50000"]["theta_hat_normalized"]
    for n in ("10000", "20000")
        @printf("  theta_hat drift vs 50k (%s): %.4f (normalized)\n",
                n, maximum(abs.(res[n]["theta_hat_normalized"] .- ref)))
    end
    return res
end

function supp_geometry(X, idx, specs)
    hr(); println("SUPPLEMENT — joint recoverability (pairwise dependence in S_eps)"); hr()
    d = length(specs)
    W = hcat([(X[idx, j] .- s.lower) ./ (s.upper - s.lower) for (j, s) in enumerate(specs)]...)
    rk(v) = (p = sortperm(v); r = zeros(Float64, length(v)); r[p] = 1:length(v); r)
    C = cor(hcat([rk(W[:, j]) for j in 1:d]...))
    @printf("  %-36s", ""); for s in specs; @printf(" %10s", first(s.name, 10)); end; println()
    for (i, s) in enumerate(specs)
        @printf("  %-36s", s.name); for j in 1:d; @printf(" %10.3f", C[i, j]); end; println()
    end
    return Dict("order" => [s.name for s in specs], "spearman" => C)
end

# ============================================================================
# §10  Main
# ============================================================================

function main()
    hr(); println("05  OFFLINE SURROGATE SEARCH  (no EI, no BO, no local polish)"); hr()
    runs_path = joinpath(SCRIPT_DIR, SAVE_DIR)

    metric_type = get(ENV, "METRIC_TYPE", "")
    if isempty(metric_type)
        vs = joinpath(runs_path, "validation_summary.json")
        metric_type = isfile(vs) ? String(get(JSON.parsefile(vs), "metric_type", "NRMSE")) : "NRMSE"
    end
    println("  metric_type = $(metric_type)")

    model_data, model_path = load_surrogate(runs_path)
    gp             = model_data["metamodel"]
    scaling_params = model_data["scaling_params"]
    feature_names  = Vector{String}(model_data["feature_columns"])
    tp             = model_data["transform_params"]
    final_std      = model_data["final_std_params"]

    invert = make_inverter(tp, final_std)
    validate_inverse(invert, tp, final_std, runs_path, metric_type)

    # ---- domain ----
    bounds = load_frozen_domain(runs_path)
    bounds === nothing && (bounds = recover_domain_from_parameters(runs_path))
    bounds === nothing && (bounds = build_domain_from_config())
    specs, aliased = align_domain(bounds, feature_names)
    check_bounds_consistency(specs, runs_path)

    hr(); println("SEARCH DOMAIN   source: $(DOMAIN_PROVENANCE)"); hr()

    hr(); println("SEARCH DOMAIN  (ex-ante design domain from ExperimentConfig.jl)"); hr()
    println("  Full design: $(length(bounds)) parameters; active search: $(length(specs)).\n")
    for s in specs
        @printf("  %-36s [%8.3f, %8.3f]  %-9s %s\n", s.name, s.lower, s.upper,
                s.isinteger ? "integer" : "cont.", s.unit)
    end
    canon_active = Set(s.name for s in specs)
    inactive = sort([k for k in keys(bounds) if !(k in canon_active)])
    if !isempty(inactive)
        println("\n  inactive (not searched): $(join(inactive, ", "))")
    end

    # ---- pool ----
    hr(); println("DENSE EXPLORATION"); hr()
    @printf("  d = %d | n = %d | seed = %d | sampler = stratified LHS\n",
            length(specs), N_SEARCH_SAMPLES, SEARCH_SEED)
    X = build_pool(specs, N_SEARCH_SAMPLES, SEARCH_SEED)
    mu_s, sig_lat, sig_prd = predict_batch(gp, X, scaling_params)
    L    = invert(mu_s)
    L_lo = invert(mu_s .- 1.96 .* sig_lat)
    L_hi = invert(mu_s .+ 1.96 .* sig_lat)

    nbad = count(!isfinite, L)
    nbad > 0 && @warn "Excluded $(nbad) pool points outside the inverse Box-Cox domain."
    Lc = replace(L, NaN => Inf)

    L_min = minimum(Lc)
    (isfinite(L_min) && L_min > 0) || error(string(
        "Invalid L_min = ", L_min,
        "; relative threshold L_min*(1+epsilon) cannot be evaluated."))

    report_noise(gp, sig_lat, sig_prd)

    i0 = argmin(Lc)
    hr(); println("BEST SURROGATE CANDIDATE  (pool argmin; no local optimisation)"); hr()
    for (j, s) in enumerate(specs)
        @printf("  %-36s %.5f\n", s.name, X[i0, j])
    end
    @printf("  predicted loss (median)      : %.6g\n", Lc[i0])
    @printf("  95%% latent band (asymmetric) : [%.6g, %.6g]\n", L_lo[i0], L_hi[i0])

    eps_res = eps_sweep(X, Lc, specs, NEAR_OPT_TOLS, N_SEARCH_SAMPLES)
    idx_p, cut_p = near_optimal(Lc, L_min, PRIMARY_TOL)

    hr(); println("UCED REVALIDATION CANDIDATES  (eps = $(round(Int, 100 * PRIMARY_TOL))%)"); hr()
    cand_active = select_candidates(X, Lc, idx_p, specs, cut_p, gp, scaling_params, invert)
    cols  = design_column_order(runs_path, bounds)
    cand  = revalidation_table(cand_active, cols, specs, bounds;
                               draws=REVAL_N_DRAWS, seed=SEARCH_SEED)

    @printf("  %-38s %14s %10s\n", "role", "pred. loss", "in S_eps")
    for r in eachrow(cand_active)
        @printf("  %-38s %14.6g %10s\n", r.role, r.predicted_loss, r.in_S_eps ? "yes" : "no")
    end
    @printf("\n  Complete vectors: %d active + %d inactive (ex-ante %s).\n",
            length(specs), length(cols) - length(specs), REVAL_INACTIVE_REF)
    REVAL_N_DRAWS > 1 &&
        @printf("  inactive-parameter draws = %d (draw 1 is the reference).\n",
                REVAL_N_DRAWS)

    supp = Dict{String,Any}()
    SUPP_DENSITY  && (supp["density"]  = supp_density(specs, gp, scaling_params, invert))
    SUPP_GEOMETRY && (supp["geometry"] = supp_geometry(X, idx_p, specs))

    # ---- outputs ----
    hr(); println("OUTPUTS"); hr()
    pool = DataFrame()
    for (j, s) in enumerate(specs)
        pool[!, Symbol(s.name)] = X[:, j]
    end
    pool.predicted_loss          = L
    pool.predicted_loss_lo95_lat = L_lo
    pool.predicted_loss_hi95_lat = L_hi
    pool.sigma_latent_std        = sig_lat
    pool.sigma_predictive_std    = sig_prd
    for tol in NEAR_OPT_TOLS
        pool[!, Symbol("in_S_$(replace(string(tol), "." => "p"))")] = Lc .<= L_min * (1 + tol)
    end
    p1 = joinpath(runs_path, "dense_surrogate_search.csv"); CSV.write(p1, pool)
    p2 = joinpath(runs_path, "revalidation_candidates_$(EPS_TAG).csv"); CSV.write(p2, cand)


    summary = Dict{String,Any}(
        "model_file" => basename(model_path),
        "model_type" => get(model_data, "model_type", nothing),
        "metric_type" => metric_type,
        "domain_source" => DOMAIN_PROVENANCE,
        "active_policies" => ExperimentConfig.ACTIVE_POLICIES,
        "name_alias_applied" => [Dict("model_feature" => a, "canonical" => b) for (a, b) in aliased],
        "output_naming" => "canonical (CHP_*) — matches current ExperimentConfig / 01",
        "search_domain" => [Dict("name" => s.name, "model_feature" => s.data_name,
                                 "lower" => s.lower, "upper" => s.upper,
                                 "type" => s.isinteger ? "integer" : "continuous") for s in specs],
        "inactive_parameters" => inactive,
        "n_samples" => N_SEARCH_SAMPLES, "seed" => SEARCH_SEED, "sampler" => "stratified_LHS",
        "integer_time" => INTEGER_TIME,
        "transformation_applied" => get(tp, "transformation_applied", false),
        "boxcox_lambda" => get(tp, "lambda", nothing),
        "data_shift" => get(tp, "data_shift", nothing),
        "L_min" => L_min,
        "theta_hat" => Dict(s.name => X[i0, j] for (j, s) in enumerate(specs)),
        "primary_epsilon" => PRIMARY_TOL,
        "epsilon_sweep" => eps_res,
        "fitted_sigma_n_standardized" => fitted_noise_std(gp),
        "fitted_sigma_f_standardized" => fitted_signal_std(gp),
        "supplement" => supp,
        "estimand_note" => string(
            "The reduced GP is refitted using only the retained dimensions, so variation ",
            "associated with the omitted dimensions is absorbed into the reduced response ",
            "surface rather than conditioned on a single fixed reference value. It is an ",
            "empirical reduced-dimensional response over the original space-filling design, ",
            "NOT the loss conditional on a particular choice of inactive parameters, and NOT ",
            "an exact marginalization under a uniform distribution."),
        "uncertainty_note" => string(
            "predicted_loss is the median of the predictive distribution on the original loss ",
            "scale (monotone back-transform of the posterior mean), not the posterior mean. ",
            "The 95% band is asymmetric after back-transformation and is reported as separate ",
            "lo/hi columns; it must not be written as L +/- 1.96*sigma."),
        "revalidation_design" => Dict(
            "inactive_reference" => REVAL_INACTIVE_REF,
            "n_inactive_draws" => REVAL_N_DRAWS,
            "n_rows" => nrow(cand),
            "inactive_values" => Dict(c => cand[1, Symbol(c)] for (_, c) in cols
                                      if !(c in Set(s.name for s in specs))),
            "note" => string(
                "Inactive parameters are held at a fixed ex-ante reference, so a UCED run at ",
                "these settings returns a conditional loss while the reduced GP predicts a ",
                "marginal over the design distribution of those parameters. The revalidation ",
                "residual therefore upper-bounds the surrogate error: it also contains the ",
                "spread induced by the inactive parameters. Set REVAL_N_DRAWS > 1 to separate ",
                "the two components.")),
        "timestamp" => string(Dates.now()))
    p3 = joinpath(runs_path, "surrogate_search_summary.json")
    open(p3, "w") do f; JSON.print(f, summary, 2); end
    for p in (p1, p2, p3); println("  $(p)"); end

    # ---- optional UCED back-validation ----
    labels = REVAL_N_DRAWS > 1 ?
        [string(r.role, " (draw ", r.draw, ")") for r in eachrow(cand)] :
        Vector{String}(cand.role)
    bv = back_validate(cand, cols, specs, ask_back_validation(labels),
                       tp, metric_type, L_min, runs_path)
    if bv !== nothing
        p5 = joinpath(runs_path, "back_validation_results_$(EPS_TAG).csv"); CSV.write(p5, bv)
        println("\n  $(p5)")
        summary["back_validation"] = Dict(
            "n_runs" => nrow(bv),
            "residual_rms" => sqrt(mean(bv.residual .^ 2)),
            "eps_L_min" => PRIMARY_TOL * L_min,
            "rows" => [Dict(string(k) => v for (k, v) in pairs(r)) for r in eachrow(bv)])
        open(p3, "w") do f; JSON.print(f, summary, 2); end
    end

    @printf("\n  epsilon*L_min = %.4g   (revalidation residual RMS should be substantially smaller)\n",
            PRIMARY_TOL * L_min)
    hr()
    return (pool = pool, candidates = cand, summary = summary)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

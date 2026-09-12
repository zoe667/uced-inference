#!/usr/bin/env julia
# Reporting only. Re-evaluate existing parameter coordinates with current saved GPs.
# No fitting, design generation, or original UCED evaluations.
using CSV, DataFrames, GaussianProcesses, JLD2, JSON, Statistics
const ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT = normpath(joinpath(@__DIR__, "..", "data"))
getstat(p, k) = p isa AbstractDict ? p[k] : getproperty(p, Symbol(k))
audits = []
for year in (2016, 2021)
    r = joinpath(ROOT, "results", string(year))
    m = JLD2.load(joinpath(r,"trained_metamodel_reduced.jld2"))
    m["transform_params"]["transformation_applied"] == false || error("Expected untransformed loss")
    df = CSV.read(joinpath(r,"dense_surrogate_search.csv"), DataFrame)
    feats = String.(m["feature_columns"])
    inputcols = [hasproperty(df, Symbol(f)) ? f : replace(f,"Coal_Retrofitted"=>"CHP_Retrofitted") for f in feats]
    X = Matrix{Float64}(df[:, inputcols])
    for j in axes(X,2)
        p=m["scaling_params"][j]
        X[:,j] = (X[:,j] .- getstat(p,"mean")) ./ getstat(p,"std")
    end
    predicted = Float64[]
    # Batch to avoid forming an unnecessary large prediction covariance.
    for start in 1:200:nrow(df)
        stop=min(start+199,nrow(df))
        z,_ = predict_y(m["metamodel"], X[start:stop,:]')
        st=m["final_std_params"]
        append!(predicted,z .* getstat(st,"std") .+ getstat(st,"mean"))
    end
    maxdiff=maximum(abs.(predicted .- df.predicted_loss))
    out=DataFrame()
    for (f,c) in zip(feats,inputcols); out[!,f]=df[!,c]; end
    out.predicted_loss=predicted
    for e in [0.05,0.1,0.2,0.3]
        out[!, "in_S_"*replace(string(e),"."=>"p")] = predicted .<= (1+e)*minimum(predicted)
    end
    CSV.write(joinpath(OUTPUT,"dense_predictions_$(year).csv"),out)
    push!(audits,Dict("year"=>year,"source"=>dirname,"n_pool"=>nrow(df),
        "max_change_from_archived_predictions"=>maxdiff,"current_L_min"=>minimum(predicted),
        "model_features"=>feats,"refitted"=>false,"uced_evaluations"=>0))
    println(year," refreshed; max change = ",maxdiff,"; minimum = ",minimum(predicted))
end
open(joinpath(OUTPUT,"dense_prediction_audit.json"),"w") do io
    JSON.print(io,audits,2)
end

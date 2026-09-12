using CSV,DataFrames,GaussianProcesses,JLD2,JSON,Statistics
v(s,k)=s isa AbstractDict ? s[string(k)] : getproperty(s,k)
paramvalue(p,k)=haskey(p,k) ? p[k] : p[replace(k,"Coal_Retrofitted_"=>"CHP_Retrofitted_")]
rows=Dict[]
for y in ["2016_100","2021_100"]
 r=joinpath("results",first(split(y,"_")));m=JLD2.load(joinpath(r,"trained_metamodel_reduced.jld2"));f=String.(m["feature_columns"])
 for d in sort(readdir(joinpath(r,"back_check_archive")))
  p=joinpath(r,"back_check_archive",d,"backcheck_result.json");isfile(p)||continue
  j=JSON.parsefile(p);X=reshape([paramvalue(j["params"],k) for k in f],1,:)
  for k in eachindex(f);s=m["scaling_params"][k];X[:,k]=(X[:,k].-v(s,:mean))./v(s,:std);end
  z,_=predict_y(m["metamodel"],X');s=m["final_std_params"];pred=z[1]*v(s,:std)+v(s,:mean)
  push!(rows,Dict("year"=>first(split(y,"_")),"role"=>j["role"],"prediction"=>pred,"actual"=>j["result"]["aggregated_score"],"params"=>j["params"],"components"=>j["result"],"source"=>p))
  println(y," ",j["role"]," ",pred," ",j["result"]["aggregated_score"])
 end
end

output_path = normpath(joinpath(@__DIR__, "..", "data", "revalidation_audit.json"))
open(output_path,"w") do io;JSON.print(io,rows,2);end

module BackValidation
export prepare_and_run_validation, analyze_validation_results, InfeasibilityError
# new added function for active learning validation
# NRMSE version

using DataFrames, CSV, GaussianProcesses, Printf, Random, Statistics, LinearAlgebra, JSON
using Distributions, StatsBase
include("04_build_surrogate.jl")
include("03_aggregate_and_analyze.jl") 
include("ExperimentConfig.jl")
include("FoldManager.jl")
include("PathConfig.jl")
using .PathConfig: REGION_WEIGHTS, WEIGHTS
using .ValidateMetamodel: AUTO_TRANSFORM, NUM_FOLDS
using .FoldManager
using .ExperimentConfig: chp_parameter_ranges_retrofitted, chp_parameter_ranges_non_retrofitted, non_chp_parameter_ranges_non_retrofitted, ACTIVE_POLICIES

# CONFIGURATION
const RUN_DIR = PathConfig.RUN_DIR  # ← Use from PathConfig
println("✅ Retrieving saved settings from: $(RUN_DIR)")

const MODEL_YEAR = PathConfig.MODEL_YEAR  # ← Use from PathConfig
const BASE_DATA_PATH = PathConfig.BASE_DATA_PATH  # ← Use from PathConfig
const HISTORICAL_DATA_PATH = PathConfig.HISTORICAL_DATA_PATH  # ← Use from PathConfig
const BASE_GENERATORS_FILE = PathConfig.BASE_GENERATORS_FILE  # ← Use from PathConfig

println("📅 Auto-detected model year: $MODEL_YEAR")

const REGIONS = ["HL", "IME", "LN", "JL"]
const NUM_WEEKS = 52
const TOTAL_HOURS = NUM_WEEKS * 168
const ACTIVE_MONTHS_FULL = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
const ACTIVE_MONTHS_TEST = ["Sep", "Oct", "Nov"] 
const ACTIVE_MONTH_INDICES_FULL = collect(1:12)
const ACTIVE_MONTH_INDICES_TEST = [9, 10, 11]
const MONTH_COLS = Symbol.(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
const CAPACITY_DATA = load_capacity_data_for_validation(BASE_DATA_PATH, REGIONS)
const HM = 728  # monthly average
const FIXED_NONCHP_NONRETRO_MIN_POWER = 0.40
const FIXED_NONCHP_NONRETRO_TIME = 8

if isfile(BASE_GENERATORS_FILE)
    base_generators_df = CSV.read(BASE_GENERATORS_FILE, DataFrame)
    coal_technologies = ["cogen_conventional_steam_coal", "conventional_steam_coal"]
    coal_units_mask = in.(base_generators_df.technology, Ref(coal_technologies))
    retrofitted_flag = string.(base_generators_df.Retrofitted)
    valid_retrofitted_mask = coal_units_mask .& in.(retrofitted_flag, Ref(["0", "1", "0.0", "1.0"]))
    base_generators_df_filtered = base_generators_df[valid_retrofitted_mask, :]
    RETROFITTED_UNIT_IDS = base_generators_df_filtered[in.(string.(base_generators_df_filtered.Retrofitted), Ref(["1", "1.0"])), :R_ID]
    println("✅ Loaded $(length(RETROFITTED_UNIT_IDS)) retrofitted unit IDs in BackValidation module")
else
    error("❌ Base generators file not found: $base_generators_file")
end

# END CONFIGURATION

function calculate_nrmse_by_resource(
    all_hist_by_region::Dict{String, Vector{Float64}},
    all_sim_by_region::Dict{String, Vector{Float64}},
    tech_name::String,
    capacity_data::Dict{String, Dict{String, Float64}},
    region_weights::Dict{String, Float64}
)
    """
    计算单个资源的 NRMSE（与 03 脚本保持一致）
    
    步骤:
    1. 计算每个区域的 RMSE
    2. 归一化为区域 NRMSE: e_r = RMSE_r / (Cap_r × H_m)
    3. 加权平均: NRMSE = Σ (w_r × e_r)
    
    返回:
    - global_nrmse: 全局 NRMSE
    - global_rmse: 全局 RMSE（诊断用）
    - regional_nrmses: 各区域 NRMSE
    - regional_rmses: 各区域 RMSE
    """
    
    regional_rmses = Dict{String, Float64}()
    regional_nrmses = Dict{String, Float64}()
    
    # 计算每个区域的 RMSE 和 NRMSE
    for (region, hist_vals) in all_hist_by_region
        sim_vals = all_sim_by_region[region]
        
        # 确保长度匹配
        n = min(length(hist_vals), length(sim_vals))
        hist_vals = hist_vals[1:n]
        sim_vals = sim_vals[1:n]
        
        # 1. 区域 RMSE
        rmse_region = sqrt(sum((sim_vals .- hist_vals) .^ 2) / n)
        regional_rmses[region] = rmse_region
        
        # 2. 区域 NRMSE
        cap_region = capacity_data[tech_name][region]
        
        if cap_region > 1e-6
            denominator = cap_region * HM  # GW × h = GWh
            nrmse_region = rmse_region / denominator
        else
            @warn "Zero capacity for $tech_name in $region"
            nrmse_region = Inf
        end
        
        regional_nrmses[region] = nrmse_region
        
        println("   Region $region:")
        @printf("      RMSE:  %.4f GWh\n", rmse_region)
        @printf("      NRMSE: %.6f (= %.4f / (%.2f GW × %d h))\n",
                nrmse_region, rmse_region, cap_region, HM)
    end
    
    # 3. 全局 NRMSE（区域加权平均）
    global_nrmse = sum(region_weights[r] * regional_nrmses[r] 
                       for r in keys(regional_nrmses))
    
    # 4. 全局 RMSE（仅用于诊断）
    all_hist = vcat(values(all_hist_by_region)...)
    all_sim = vcat(values(all_sim_by_region)...)
    global_rmse = sqrt(mean((all_sim .- all_hist) .^ 2))
    
    return (
        global_nrmse = global_nrmse,
        global_rmse = global_rmse,
        regional_nrmses = regional_nrmses,
        regional_rmses = regional_rmses
    )
end

function check_uced_feasibility(run_dir::String)
    """
    检查UCED仿真的完整性和可行性
    返回: (is_feasible, failed_weeks, error_type)
    """
    println("\n🔍 CHECKING UCED SIMULATION FEASIBILITY:")
    println("="^50)
    
    error_type = "none"
    failed_weeks = Int[]
    is_feasible = true
    
    # 分析log文件
    log_file = joinpath(run_dir, "validation_run.log")
    if isfile(log_file)
        log_content = read(log_file, String)
        
        # 检查Gurobi的不可行性报告
        if contains(log_content, "is infeasible or unbounded")
            error_type = "model_infeasible_or_unbounded"
            is_feasible = false
            
            # 统计不可行性实例数量
            infeasible_count = length(split(log_content, "is infeasible or unbounded")) - 1
            
            println("   ❌ UCED infeasibility detected:")
            println("      Found $(infeasible_count) infeasible or unbounded subproblems")
            println("      This indicates parameter combination leads to infeasible solutions")
            
        else
            error_type = "none"
            is_feasible = true
            println("   ✅ No infeasibility detected - simulation appears feasible")
        end
        # 显示日志统计
        log_lines = count('\n', log_content)
        println("   📄 Log file: $(log_lines) lines")
        
    else
        error_type = "no_log_file"
        is_feasible = false
        println("   ❌ No log file found")
    end
    
    return is_feasible, failed_weeks, error_type
end

struct InfeasibilityError <: Exception
    msg::String
    failed_weeks::Vector{Int}
    error_type::String
end

function Base.showerror(io::IO, e::InfeasibilityError)
    print(io, "InfeasibilityError: ", e.msg)
end

function lookup_param(params_dict::Dict, canonical_key::AbstractString, aliases::AbstractString...)
    for key in (canonical_key, aliases...)
        if haskey(params_dict, key)
            return params_dict[key], key
        end

        symbol_key = Symbol(key)
        if haskey(params_dict, symbol_key)
            return params_dict[symbol_key], key
        end
    end

    return nothing, nothing
end

function prepare_and_run_validation(
    params_dict::Dict;
    run_dir::AbstractString=get(ENV, "REVAL_WORK_DIR", PathConfig.BACK_CHECK_DIR),
    uced_threads::Int=parse(Int, get(ENV, "REVAL_UCED_THREADS", "8")),
)
    println("\n🔍 RECEIVED PARAMETERS IN BACK_VALIDATION:")
    println("="^60)
    for (param_name, param_value) in params_dict
        println("   $(rpad(param_name, 30)): $param_value")
    end
    println("="^60)

    println("\n--- Preparing Input Files for Validation Run ---")
    run_dir = abspath(run_dir)

    # Back-validation reuses a fixed working directory. Clear it before every
    # run so stale weekly outputs or annual violation summaries cannot be
    # mistaken for results from the current candidate.
    if isdir(run_dir)
        for item in readdir(run_dir; join=true)
            rm(item; recursive=true, force=true)
        end
    end
    mkpath(run_dir)

    modified_generators_df = deepcopy(base_generators_df)

    println("🔧 PARAMETER APPLICATION:")
    println("="^50)
    println("   Total units to modify:")
    
    retrofitted_flag = string.(modified_generators_df.Retrofitted)
    valid_retrofitted_flag = .!ismissing.(modified_generators_df.Retrofitted)
    coal_technologies = ["cogen_conventional_steam_coal", "conventional_steam_coal"]
    coal_mask = in.(modified_generators_df.technology, Ref(coal_technologies))
    chp_mask = modified_generators_df.technology .== "cogen_conventional_steam_coal"
    non_chp_technology_mask = modified_generators_df.technology .== "conventional_steam_coal"

    # Match experiment/01_generate_inputs.jl: retrofit pilot parameters apply to
    # all Retrofitted==1 coal units, including rows whose CHP flag is 1 but whose
    # GEM-derived technology is conventional_steam_coal.
    retrofitted_mask = valid_retrofitted_flag .&
                       in.(retrofitted_flag, Ref(["1", "1.0"])) .&
                       coal_mask
    non_retrofitted_mask = valid_retrofitted_flag .&
                           in.(retrofitted_flag, Ref(["0", "0.0"])) .&
                           chp_mask
    non_chp_mask = valid_retrofitted_flag .&
                   in.(retrofitted_flag, Ref(["0", "0.0"])) .&
                   non_chp_technology_mask
    
    println("   Retrofitted coal pilot units: $(sum(retrofitted_mask)) units")
    println("   CHP Non-Retrofitted: $(sum(non_retrofitted_mask)) units")
    println("   Non-CHP Coal Non-Retrofitted: $(sum(non_chp_mask)) units")

    # Modify generators file using the dictionary of best parameters
    if !isempty(ExperimentConfig.chp_parameter_ranges_retrofitted)
        for size_class in keys(chp_parameter_ranges_retrofitted)
            min_cap, max_cap = parse.(Int, split(size_class, "-"))
            class_rows = retrofitted_mask .&
                         (modified_generators_df.Cap_Size .> min_cap) .&
                         (modified_generators_df.Cap_Size .<= max_cap)

            if sum(class_rows) > 0
                min_power_key = "CHP_Retrofitted_Min_Power_$(size_class)"
                time_key = "CHP_Retrofitted_Time_$(size_class)"
                min_power_val, min_power_used_key = lookup_param(
                    params_dict,
                    min_power_key,
                    "Coal_Retrofitted_Min_Power_$(size_class)",
                )
                time_val_raw, time_used_key = lookup_param(
                    params_dict,
                    time_key,
                    "Coal_Retrofitted_Time_$(size_class)",
                )
                
                if min_power_val !== nothing
                    modified_generators_df[class_rows, :Min_Power] .= min_power_val
                    println("   Applied $(min_power_used_key) = $(min_power_val) to $(sum(class_rows)) units")
                end
                if time_val_raw !== nothing
                    time_val = Int(round(time_val_raw))
                    modified_generators_df[class_rows, :Up_Time] .= time_val
                    modified_generators_df[class_rows, :Down_Time] .= time_val
                    println("   Applied $(time_used_key) = $(time_val) to $(sum(class_rows)) units")
                end
            end
        end
    end

    if !isempty(ExperimentConfig.chp_parameter_ranges_non_retrofitted)
        for size_class in keys(chp_parameter_ranges_non_retrofitted)
            min_cap, max_cap = parse.(Int, split(size_class, "-"))
            class_rows = non_retrofitted_mask .&
                         (modified_generators_df.Cap_Size .> min_cap) .&
                         (modified_generators_df.Cap_Size .<= max_cap)

            if sum(class_rows) > 0
                min_power_key = "CHP_NonRetrofitted_Min_Power_$(size_class)"
                time_key = "CHP_NonRetrofitted_Time_$(size_class)"
                
                if haskey(params_dict, min_power_key)
                    modified_generators_df[class_rows, :Min_Power] .= params_dict[min_power_key]
                    println("   Applied $(min_power_key) = $(params_dict[min_power_key]) to $(sum(class_rows)) units")
                end
                if haskey(params_dict, time_key)
                    time_val = Int(round(params_dict[time_key]))
                    modified_generators_df[class_rows, :Up_Time] .= time_val
                    modified_generators_df[class_rows, :Down_Time] .= time_val
                    println("   Applied $(time_key) = $(time_val) to $(sum(class_rows)) units")
                end
            end
        end
    end
    
    if !isempty(non_chp_parameter_ranges_non_retrofitted)
        for class_name in keys(non_chp_parameter_ranges_non_retrofitted)
            min_cap, max_cap = parse.(Int, split(class_name, "-"))

            class_rows = non_chp_mask .&
                         (modified_generators_df.Cap_Size .>= min_cap) .&
                         (modified_generators_df.Cap_Size .<= max_cap)

            if sum(class_rows) > 0
                min_power_key = "NonCHP_NonRetrofitted_Min_Power_$(class_name)"
                time_key = "NonCHP_NonRetrofitted_Time_$(class_name)"
                min_power_val, min_power_used_key = lookup_param(
                    params_dict,
                    min_power_key,
                    "NonCHP_Min_Power_$(class_name)",
                )
                time_val_raw, time_used_key = lookup_param(
                    params_dict,
                    time_key,
                    "NonCHP_Time_$(class_name)",
                )

                if min_power_val !== nothing
                    modified_generators_df[class_rows, :Min_Power] .= min_power_val
                    println("   Applied $(min_power_used_key) = $(min_power_val) to $(sum(class_rows)) units")
                end
                if time_val_raw !== nothing
                    time_val = Int(round(time_val_raw))
                    modified_generators_df[class_rows, :Up_Time] .= time_val
                    modified_generators_df[class_rows, :Down_Time] .= time_val
                    println("   Applied $(time_used_key) = $(time_val) to $(sum(class_rows)) units")
                end
            end
        end
    else
        modified_generators_df[non_chp_mask, :Min_Power] .= FIXED_NONCHP_NONRETRO_MIN_POWER
        modified_generators_df[non_chp_mask, :Up_Time] .= FIXED_NONCHP_NONRETRO_TIME
        modified_generators_df[non_chp_mask, :Down_Time] .= FIXED_NONCHP_NONRETRO_TIME
        println("   Non-CHP Non-Retrofitted parameter space is empty; applied fixed Min_Power=$(FIXED_NONCHP_NONRETRO_MIN_POWER), Time=$(FIXED_NONCHP_NONRETRO_TIME) to $(sum(non_chp_mask)) units")
    end
    
    CSV.write(joinpath(run_dir, "generators_modified.csv"), modified_generators_df)

    config_data = []
    if "quota_policies" in ACTIVE_POLICIES
        for size_class in ["0-300", "300-660", "660-1000"]
            quota_param_name = "Minimum_Quota_%_$(size_class)"
            if haskey(params_dict, quota_param_name)
                quota_val = params_dict[quota_param_name]
                push!(config_data, (parameter="Minimum_Quota_%_$(size_class)", value=quota_val))
            end
        end
    end

    if "mlt_policies" in ACTIVE_POLICIES
        mlt_param_name = "MLT_Band"
        if haskey(params_dict, mlt_param_name)
            mlt_band_val = params_dict[mlt_param_name]
            push!(config_data, (parameter="MLT_Band", value=mlt_band_val))
            println("   Applied $(mlt_param_name) = $(mlt_band_val)")
        end
    end

    config_df = DataFrame(parameter=[d.parameter for d in config_data], value=[d.value for d in config_data])
    CSV.write(joinpath(run_dir, "run_config.csv"), config_df)
    println("✅ Input files created in: $(run_dir)")

    println("\n--- Executing Full UCED Model for Validation ---")
    main_model_script = normpath(joinpath(@__DIR__, "..", "model", "Run.jl"))
    #julia_executable = `$(Base.julia_cmd()) -t 8`
    #cmd = `$julia_executable $main_model_script --run-dir $run_dir --config-file $(joinpath(run_dir, "run_config.csv"))`
    project_dir = dirname(Base.active_project())
    julia_executable = `$(Base.julia_cmd()) --project=$project_dir -t $uced_threads`
    cmd = `$julia_executable $main_model_script --run-dir $run_dir --config-file $(joinpath(run_dir, "run_config.csv"))`
    log_file = joinpath(run_dir, "validation_run.log")

    println("🔧 UCED MODEL CONFIGURATION:")
    println("="^50)
    println("   Model script: $main_model_script")
    println("   Run directory: $run_dir")
    if isfile(main_model_script)
        println("\n🔍 READING RUN.JL DEFAULT SETTINGS:")
        try
            content = read(main_model_script, String)
            
            # 查找常见的配置参数
            config_patterns = [
                (r"scenario\s*=\s*[\"']([^\"']+)[\"']", "runname"),
                (r"model_year\s*=\s*(\d+)", "model_year"),
                (r"quota_enabled\s*=\s*(\d+)", "quota_enabled"),
                (r"retrofit_scenario\s*=\s*(\d+)", "retrofit_scenario"),
            ]
            
            for (pattern, description) in config_patterns
                matches = collect(eachmatch(pattern, content))
                if !isempty(matches)
                    value = matches[end].captures[1]  # 取最后一个匹配（最可能是实际设置）
                    println("   $description: $value")
                end
            end
            
            # 检查是否有命令行参数处理
            if contains(content, "ArgParse") || contains(content, "ARGS")
                println("   ✅ Model supports command-line arguments")
            else
                println("   ⚠️  Model may use hardcoded parameters")
            end
            
        catch e
            println("   ⚠️  Could not read Run.jl: $e")
        end
    else
        println("   ❌ Run.jl not found at: $main_model_script")
    end
    
    try
        run(pipeline(cmd, stdout=log_file, stderr=log_file))
        println("✅ UCED model run completed.")
        
        # 检查仿真可行性
        is_feasible, failed_weeks, error_type = check_uced_feasibility(run_dir)
        
        if !is_feasible
            error_msg = "UCED simulation failed: $(error_type)"
            throw(InfeasibilityError(error_msg, failed_weeks, error_type))
        end
        
    catch e
        if isa(e, InfeasibilityError)
            println("❌ UCED feasibility check failed: $(e.msg)")
            throw(e)
        else
            println("❌ ERROR during UCED model execution.")
            throw(e)
        end
    end
    
    return run_dir
end

function analyze_validation_results(
    run_dir::String=PathConfig.BACK_CHECK_DIR,
    transform_params::Union{Dict, Nothing}=nothing,
    metric_type::String="NRMSE"  
    )
    println("\n--- Analyzing Validation Run Results using Z-Score Method ---")
    
    # 加载风电和光伏历史发电量数据
    hist_coal_gen = CSV.read(joinpath(HISTORICAL_DATA_PATH, "hist_coal_gen_monthly.csv"), DataFrame)
    hist_wind_gen = CSV.read(joinpath(HISTORICAL_DATA_PATH, "hist_wind_gen_monthly.csv"), DataFrame)  
    hist_solar_gen = CSV.read(joinpath(HISTORICAL_DATA_PATH, "hist_solar_gen_monthly.csv"), DataFrame) 
    hist_mlt_data = load_historical_mlt_data(HISTORICAL_DATA_PATH)

    # 确定分析期间（与GA脚本保持一致）
    if NUM_WEEKS == 12  
        active_months = ACTIVE_MONTHS_TEST
        active_month_indices = ACTIVE_MONTH_INDICES_TEST
        comparison_months = Symbol.(["Sep", "Oct", "Nov"])  # ✅ 新增
        println("🎯 TEST MODE: Only analyzing months $active_months")
    else
        active_months = ACTIVE_MONTHS_FULL
        active_month_indices = ACTIVE_MONTH_INDICES_FULL
        comparison_months = Symbol.(ACTIVE_MONTHS_FULL)  # ✅ 新增
        println("🎯 FULL YEAR MODE: Analyzing all 12 months")
    end

    # 计算三种发电类型的误差
    println("\n📊 GENERATION ERROR CALCULATION:")
    println("="^50)
    
    # 使用统一的 process_run_results 函数获取三种发电类型
    monthly_coal_gen_results_mwh, monthly_wind_gen_results_mwh, monthly_solar_gen_results_mwh, monthly_mlt_flow_results = 
        process_run_results(run_dir, NUM_WEEKS, TOTAL_HOURS, REGIONS; back_test=true)

    # === COAL GENERATION ERROR ===
    all_coal_hist = Float64[]
    all_coal_sim = Float64[]
    all_coal_hist_by_region = Dict{String, Vector{Float64}}()  
    all_coal_sim_by_region = Dict{String, Vector{Float64}}()   
    
    for region in REGIONS
        hist_row_coal = filter(row -> row.region == region, hist_coal_gen)
        if !isempty(hist_row_coal)
            hist_vector_full = [parse(Float64, string(x)) for x in hist_row_coal[1, MONTH_COLS]]
            
            if haskey(monthly_coal_gen_results_mwh, region)
                sim_data_mwh = monthly_coal_gen_results_mwh[region]
                sim_data_gwh = sim_data_mwh ./ 1000.0  # Convert to GWh
                
                # 根据模拟期间选择对应月份
                if NUM_WEEKS == 12  # 测试模式：3个月（Sep, Oct, Nov）
                    sim_vector_for_error = sim_data_gwh[active_month_indices]
                    hist_vector_for_error = hist_vector_full[active_month_indices]  # [9,10,11]对应Sep,Oct,Nov
                else  # 全年模式
                    sim_vector_for_error = sim_data_gwh[1:min(length(sim_data_gwh), 12)]
                    hist_vector_for_error = hist_vector_full[1:length(sim_vector_for_error)]
                end
                
                # 确保长度匹配
                min_length = min(length(sim_vector_for_error), length(hist_vector_for_error))
                sim_vector_for_error = sim_vector_for_error[1:min_length]
                hist_vector_for_error = hist_vector_for_error[1:min_length]

                println("\n📋 Region: $region (Coal Generation)")
                println("   Month    | Historical(GWh) | Simulated(GWh) | Error(GWh)")
                println("   ---------|----------------|----------------|----------")
                
                for (i, month) in enumerate(active_months[1:min_length])
                    hist_val = hist_vector_for_error[i]
                    sim_val = sim_vector_for_error[i]
                    signed_error = sim_val - hist_val  # ✅ 带符号误差（正=多发，负=少发）
                    @printf("   %-8s | %13.2f | %13.2f | %8.2f\n", month, hist_val, sim_val, signed_error)
                end

                all_coal_hist_by_region[region] = hist_vector_for_error
                all_coal_sim_by_region[region] = sim_vector_for_error
                
                # 添加到全局数据集
                append!(all_coal_hist, hist_vector_for_error)
                append!(all_coal_sim, sim_vector_for_error)
                
                println("Region $region (Coal): $(length(hist_vector_for_error)) months added to global calculation")
            else
                println("⚠️ No coal simulation data found for region $region")
            end
        end
    end
                
    if metric_type == "NRMSE"
        
        coal_metrics = calculate_nrmse_by_resource(
            all_coal_hist_by_region,
            all_coal_sim_by_region,
            "coal",
            CAPACITY_DATA,
            PathConfig.REGION_WEIGHTS
        )
        
        new_raw_error_coal = coal_metrics.global_nrmse
        println("🎯 GLOBAL COAL NRMSE: $(round(new_raw_error_coal, digits=6))")
        println("   Regional breakdown:")
        for (region, nrmse) in sort(collect(coal_metrics.regional_nrmses))
            @printf("      %5s: %.6f\n", region, nrmse)
        end
        
    elseif metric_type == "RMSE"
        new_raw_error_coal = sqrt(mean((all_coal_sim .- all_coal_hist) .^ 2))
        println("\n🎯 GLOBAL COAL RMSE: $(round(new_raw_error_coal, digits=4)) GWh")
    else #MAE
        new_raw_error_coal = mean(abs.(all_coal_sim .- all_coal_hist))
        println("\n🎯 GLOBAL COAL MAE: $(round(new_raw_error_coal, digits=4)) GWh")
    end

    global_coal_signed_error = sum(all_coal_sim) - sum(all_coal_hist)
    global_coal_bias_pct = (global_coal_signed_error / sum(all_coal_hist)) * 100

    println("\n📊 COAL Data Summary:")
    println("   Total data points: $(length(all_coal_hist))")
    println("   Historical total: $(round(sum(all_coal_hist), digits=1)) GWh")
    println("   Simulated total:  $(round(sum(all_coal_sim), digits=1)) GWh")
    println("   Global bias: $(global_coal_signed_error > 0 ? "+" : "")$(round(global_coal_signed_error, digits=1)) GWh ($(round(global_coal_bias_pct, digits=2))%)")
    if global_coal_signed_error > 0
        println("   ⚠️  Simulation OVER-generates coal (too much)")
    elseif global_coal_signed_error < 0
        println("   ⚠️  Simulation UNDER-generates coal (too little)")
    else
        println("   ✅ Simulation matches historical total")
    end


    # WIND GENERATION ERROR ===
    all_wind_hist = Float64[]
    all_wind_sim = Float64[]
    all_wind_hist_by_region = Dict{String, Vector{Float64}}() 
    all_wind_sim_by_region = Dict{String, Vector{Float64}}()
    
    for region in REGIONS
        hist_row_wind = filter(row -> row.region == region, hist_wind_gen)
        if !isempty(hist_row_wind)
            hist_vector_full = [parse(Float64, string(x)) for x in hist_row_wind[1, MONTH_COLS]]
            
            if haskey(monthly_wind_gen_results_mwh, region)
                sim_data_mwh = monthly_wind_gen_results_mwh[region]
                sim_data_gwh = sim_data_mwh ./ 1000.0  # Convert to GWh
                
                # 根据模拟期间选择对应月份
                if NUM_WEEKS == 12
                    sim_vector_for_error = sim_data_gwh[active_month_indices]
                    hist_vector_for_error = hist_vector_full[active_month_indices]
                else
                    sim_vector_for_error = sim_data_gwh[1:min(length(sim_data_gwh), 12)]
                    hist_vector_for_error = hist_vector_full[1:length(sim_vector_for_error)]
                end
                
                # 确保长度匹配
                min_length = min(length(sim_vector_for_error), length(hist_vector_for_error))
                sim_vector_for_error = sim_vector_for_error[1:min_length]
                hist_vector_for_error = hist_vector_for_error[1:min_length]

                println("\n📋 Region: $region (Wind Generation)")
                println("   Month    | Historical(GWh) | Simulated(GWh) | Error(GWh)")
                println("   ---------|----------------|----------------|----------")
                
                for (i, month) in enumerate(active_months[1:min_length])
                    hist_val = hist_vector_for_error[i]
                    sim_val = sim_vector_for_error[i]
                    signed_error = sim_val - hist_val
                    @printf("   %-8s | %13.2f | %13.2f | %8.2f\n", month, hist_val, sim_val, signed_error)
                end
                
                all_wind_hist_by_region[region] = hist_vector_for_error
                all_wind_sim_by_region[region] = sim_vector_for_error
                # 添加到全局数据集
                append!(all_wind_hist, hist_vector_for_error)
                append!(all_wind_sim, sim_vector_for_error)
                
                println("Region $region (Wind): $(length(hist_vector_for_error)) months added to global calculation")
            else
                println("⚠️ No wind simulation data found for region $region")
            end
        end
    end
    
    if metric_type == "NRMSE"
        wind_metrics = calculate_nrmse_by_resource(
            all_wind_hist_by_region,
            all_wind_sim_by_region,
            "wind",
            CAPACITY_DATA,
            PathConfig.REGION_WEIGHTS
        )
        new_raw_error_wind = wind_metrics.global_nrmse
        
        println("🎯 GLOBAL WIND NRMSE: $(round(new_raw_error_wind, digits=6))")
        
    elseif metric_type == "RMSE"
        new_raw_error_wind = sqrt(mean((all_wind_sim .- all_wind_hist) .^ 2))
        println("\n🎯 GLOBAL WIND RMSE: $(round(new_raw_error_wind, digits=4)) GWh")
    else
        new_raw_error_wind = mean(abs.(all_wind_sim .- all_wind_hist))
        println("\n🎯 GLOBAL WIND MAE: $(round(new_raw_error_wind, digits=4)) GWh")
    end

    global_wind_signed_error = sum(all_wind_sim) - sum(all_wind_hist)
    global_wind_bias_pct = (global_wind_signed_error / sum(all_wind_hist)) * 100
    
    println("\n📊 WIND Data Summary:")
    println("   Total data points: $(length(all_wind_hist))")
    println("   Historical total: $(round(sum(all_wind_hist), digits=1)) GWh")
    println("   Simulated total:  $(round(sum(all_wind_sim), digits=1)) GWh")
    println("   Global bias: $(global_wind_signed_error > 0 ? "+" : "")$(round(global_wind_signed_error, digits=1)) GWh ($(round(global_wind_bias_pct, digits=2))%)")

    # SOLAR GENERATION ERROR ===
    all_solar_hist = Float64[]
    all_solar_sim = Float64[]
    all_solar_hist_by_region = Dict{String, Vector{Float64}}()  
    all_solar_sim_by_region = Dict{String, Vector{Float64}}()
    
    for region in REGIONS
        hist_row_solar = filter(row -> row.region == region, hist_solar_gen)
        if !isempty(hist_row_solar)
            hist_vector_full = [parse(Float64, string(x)) for x in hist_row_solar[1, MONTH_COLS]]
            
            if haskey(monthly_solar_gen_results_mwh, region)
                sim_data_mwh = monthly_solar_gen_results_mwh[region]
                sim_data_gwh = sim_data_mwh ./ 1000.0  # Convert to GWh
                
                # 根据模拟期间选择对应月份
                if NUM_WEEKS == 12
                    sim_vector_for_error = sim_data_gwh[active_month_indices]
                    hist_vector_for_error = hist_vector_full[active_month_indices]
                else
                    sim_vector_for_error = sim_data_gwh[1:min(length(sim_data_gwh), 12)]
                    hist_vector_for_error = hist_vector_full[1:length(sim_vector_for_error)]
                end
                
                # 确保长度匹配
                min_length = min(length(sim_vector_for_error), length(hist_vector_for_error))
                sim_vector_for_error = sim_vector_for_error[1:min_length]
                hist_vector_for_error = hist_vector_for_error[1:min_length]

                println("\n📋 Region: $region (Solar Generation)")
                println("   Month    | Historical(GWh) | Simulated(GWh) | Error(GWh)")
                println("   ---------|----------------|----------------|----------")
                
                for (i, month) in enumerate(active_months[1:min_length])
                    hist_val = hist_vector_for_error[i]
                    sim_val = sim_vector_for_error[i]
                    signed_error = sim_val - hist_val
                    @printf("   %-8s | %13.2f | %13.2f | %8.2f\n", month, hist_val, sim_val, signed_error)
                end

                all_solar_hist_by_region[region] = hist_vector_for_error
                all_solar_sim_by_region[region] = sim_vector_for_error
                # 添加到全局数据集
                append!(all_solar_hist, hist_vector_for_error)
                append!(all_solar_sim, sim_vector_for_error)
                
                println("Region $region (Solar): $(length(hist_vector_for_error)) months added to global calculation")
            else
                println("⚠️ No solar simulation data found for region $region")
            end
        end
    end
    
    if metric_type == "NRMSE"
        solar_metrics = calculate_nrmse_by_resource(
            all_solar_hist_by_region,
            all_solar_sim_by_region,
            "solar",
            CAPACITY_DATA,
            PathConfig.REGION_WEIGHTS
        )
        
        new_raw_error_solar = solar_metrics.global_nrmse
        println("🎯 GLOBAL SOLAR NRMSE: $(round(new_raw_error_solar, digits=6))")
        
    elseif metric_type == "RMSE"
        new_raw_error_solar = sqrt(mean((all_solar_sim .- all_solar_hist) .^ 2))
        println("\n🎯 GLOBAL SOLAR RMSE: $(round(new_raw_error_solar, digits=4)) GWh")
    else
        new_raw_error_solar = mean(abs.(all_solar_sim .- all_solar_hist))
        println("\n🎯 GLOBAL SOLAR MAE: $(round(new_raw_error_solar, digits=4)) GWh")
    end

    global_solar_signed_error = sum(all_solar_sim) - sum(all_solar_hist)
    global_solar_bias_pct = (global_solar_signed_error / sum(all_solar_hist)) * 100
    
    println("\n📊 SOLAR Data Summary:")
    println("   Total data points: $(length(all_solar_hist))")
    println("   Historical total: $(round(sum(all_solar_hist), digits=1)) GWh")
    println("   Simulated total:  $(round(sum(all_solar_sim), digits=1)) GWh")
    println("   Global bias: $(global_solar_signed_error > 0 ? "+" : "")$(round(global_solar_signed_error, digits=1)) GWh ($(round(global_solar_bias_pct, digits=2))%)")

    # === MLT TRANSMISSION ERROR ===
    println("\n📊 MLT TRANSMISSION ERROR CALCULATION:")
    println("="^50)
    
    if !isempty(monthly_mlt_flow_results) && !isempty(hist_mlt_data)
        comparison_months = NUM_WEEKS == 12 ? Symbol.(["Sep", "Oct", "Nov"]) : Symbol.(ACTIVE_MONTHS_FULL)
        mlt_baseline_capacity = calculate_mlt_baseline_capacity(hist_mlt_data, comparison_months)
        println("   MLT baseline capacity: $(round(mlt_baseline_capacity, digits=2)) GW")

        mlt_metrics = calculate_mlt_metrics(hist_mlt_data, monthly_mlt_flow_results, comparison_months, mlt_baseline_capacity)
        
        if metric_type == "NRMSE"
            new_raw_error_mlt = mlt_metrics.global_nrmse
            println("🎯 GLOBAL MLT NRMSE: $(round(new_raw_error_mlt, digits=6)) (normalized)")
        elseif metric_type == "RMSE"
            new_raw_error_mlt = mlt_metrics.global_rmse
            println("🎯 GLOBAL MLT RMSE: $(round(new_raw_error_mlt, digits=4)) GWh")
        else
            new_raw_error_mlt = mlt_metrics.mae
            println("🎯 GLOBAL MLT MAE: $(round(new_raw_error_mlt, digits=4)) GWh")
        end
    else
        println("⚠️  MLT data unavailable, using default error value")
        new_raw_error_mlt = 0.0  # 或者其他默认值
    end


    # ================================================================
    # 🆕 NEW LOGIC: AGGREGATE THEN TRANSFORM
    # ================================================================
    println("\n📊 CALCULATING AGGREGATED SCORE & APPLYING TRANSFORM:")
    println("="^50)

    # 1. 加载参数
    if transform_params !== nothing
        train_y_params = transform_params
    else
        println("📁 Loading Y-transform parameters from: $(RUN_DIR)")
        y_params_path = joinpath(RUN_DIR, "master_transform_params.json")
        if !isfile(y_params_path)
            error("❌ Training Y-transform parameters not found: $(y_params_path)")
        end
        raw_y_params = JSON.parsefile(y_params_path)
        train_y_params = haskey(raw_y_params, "final_transform_params") ? raw_y_params["final_transform_params"] : raw_y_params
    end

    # 2. 聚合 (Aggregation)
    # 使用 PathConfig 中定义的权重
    println("   Using Weights: Coal=$(WEIGHTS["coal_gen"]), Wind=$(WEIGHTS["wind_gen"]), Solar=$(WEIGHTS["solar_gen"]), MLT=$(WEIGHTS["mlt_flow"])")
    
    metric_val_coal = new_raw_error_coal
    metric_val_wind = new_raw_error_wind
    metric_val_solar = new_raw_error_solar
    metric_val_mlt = new_raw_error_mlt

    aggregation_method = get(train_y_params, "aggregation_method", "L2")
    
    if aggregation_method == "L2"
        println("   Method: L2")
        raw_aggregated_score =
            WEIGHTS["coal_gen"] * metric_val_coal^2 +
            WEIGHTS["wind_gen"] * metric_val_wind^2 +
            WEIGHTS["solar_gen"] * metric_val_solar^2 +
            WEIGHTS["mlt_flow"] * metric_val_mlt^2
    else
        println("   Method: Weighted Sum")
        raw_aggregated_score = 
            WEIGHTS["coal_gen"] * metric_val_coal +
            WEIGHTS["wind_gen"] * metric_val_wind +
            WEIGHTS["solar_gen"] * metric_val_solar +
            WEIGHTS["mlt_flow"] * metric_val_mlt
    end
    println("   👉 Raw Aggregated Score: $(round(raw_aggregated_score, digits=6))")

    # 3. 全局变换 (Global Transform)
    global_mean = train_y_params["raw_mean"]
    global_std = train_y_params["raw_std"]
    lambda = get(train_y_params, "lambda", nothing)

    val_score_trans = raw_aggregated_score
    
    # Box-Cox
    if lambda !== nothing
        println("   Applying Box-Cox (λ=$(round(lambda, digits=4)))...")
        if abs(lambda) < 1e-6
            val_score_trans = log(val_score_trans)
        else
            val_score_trans = (val_score_trans^lambda - 1) / lambda
        end
    end

    # Z-Score
    validation_performance_score_transformed = (val_score_trans - global_mean) / global_std
    
    println("   Using Global Stats: μ=$(round(global_mean, digits=6)), σ=$(round(global_std, digits=6))")
    println("   👉 Raw Aggregated Score (original): $(round(raw_aggregated_score, digits=6))")
    println("   👉 Standardized Score (z-score):    $(round(validation_performance_score_transformed, digits=4))")
    
    # === 最终输出 ===
    println("\n🎯 FINAL PERFORMANCE SCORE:")
    unit_str = metric_type == "NRMSE" ? "(normalized)" : "(GWh)"
    
    @printf("New Run %s Coal:  %.4f %s\n", metric_type, new_raw_error_coal, unit_str)
    @printf("New Run %s Wind:  %.4f %s\n", metric_type, new_raw_error_wind, unit_str)
    @printf("New Run %s Solar: %.4f %s\n", metric_type, new_raw_error_solar, unit_str)
    @printf("New Run %s MLT:   %.4f %s\n", metric_type, new_raw_error_mlt, unit_str)
    @printf("Raw Aggregated Score (original): %.6f\n", raw_aggregated_score)
    @printf("Standardized Score (z-score):    %.4f (for reference only)\n", validation_performance_score_transformed)
    println("✅ Returning ORIGINAL-space loss for surrogate revalidation")
    println("="^50)

    # Return original-space loss for comparison with surrogate prediction
    return Dict(
        "performance_score" => raw_aggregated_score,  # ← 原始聚合分数
        "performance_score_transformed" => validation_performance_score_transformed,  # z-score（保留用于参考）
        "raw_coal_error" => new_raw_error_coal,
        "raw_wind_error" => new_raw_error_wind,
        "raw_solar_error" => new_raw_error_solar,
        "raw_mlt_error" => new_raw_error_mlt,
        "aggregated_score" => raw_aggregated_score
    )
end


end

# 03_aggregate_and_analyze.jl
# monthly-average version of 03_aggregate_and_analyze.jl
# Purpose:
# - Compute validation metrics after converting both simulated and historical
#   monthly generation / MLT totals to monthly-average power.
# - This avoids penalizing the 52-week UCED model year (8736 hours) against
#   calendar-month totals with 8760/8784-hour year coverage.
# - Outputs use *_monthly_average filenames and do not overwrite the original
#   final_performance_summary.csv.


using DataFrames
using CSV
using Statistics
using Printf
export load_historical_mlt_data, calculate_mlt_metrics, process_run_results
include(joinpath(@__DIR__, "PathConfig.jl"))
using .PathConfig
PathConfig.print_config()
PathConfig.validate_paths()

# --- Configuration ---

# 1. Define Paths & Experiment Settings

num_runs = PathConfig.NUM_RUNS
num_weeks = PathConfig.NUM_WEEKS
runs_path = PathConfig.RUN_DIR
historical_data_path = PathConfig.HISTORICAL_DATA_PATH
base_data_path = PathConfig.BASE_DATA_PATH
ANALYSIS_YEAR = parse(Int, PathConfig.MODEL_YEAR)
regions = PathConfig.REGIONS
total_hours = PathConfig.TOTAL_HOURS
const MONTH_COLS = [:Jan, :Feb, :Mar, :Apr, :May, :Jun, :Jul, :Aug, :Sep, :Oct, :Nov, :Dec]


function is_leap_year(year::Int)
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
end


function calendar_month_hours(year::Int)
    days = [31, is_leap_year(year) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return Dict(MONTH_COLS[i] => days[i] * 24 for i in eachindex(MONTH_COLS))
end


function simulated_month_hours(num_weeks::Int, total_hours::Int)
    month_hours = Dict(month => 0 for month in MONTH_COLS)

    if num_weeks < 52
        for month in [:Sep, :Oct, :Nov]
            month_hours[month] = 168 * 4
        end
        return month_hours
    end

    # Keep the same hour-to-month assignment as process_run_results:
    # a 52-week model year is 8736 hours, so December is clipped to 720 hours.
    days = [31, is_leap_year(year) ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month_end_hours = cumsum(days .* 24)
    month_end_hours = cumsum(nominal_hours)
    start_h = 1

    for (i, month) in enumerate(MONTH_COLS)
        end_h = min(month_end_hours[i], total_hours)
        month_hours[month] = start_h <= end_h ? end_h - start_h + 1 : 0
        start_h = end_h + 1
    end

    return month_hours
end


function normalize_monthly_dataframe!(df::DataFrame, hours_by_month::Dict{Symbol, Int})
    for month in MONTH_COLS
        if month in Symbol.(names(df))
            hours = get(hours_by_month, month, 0)
            if hours > 0
                df[!, month] = Float64.(df[!, month]) ./ hours
            end
        end
    end
    return df
end


function normalize_monthly_dict!(monthly_values::Dict{String, Vector{Float64}}, hours_by_month::Dict{Symbol, Int})
    for (_, values) in monthly_values
        for (i, month) in enumerate(MONTH_COLS)
            if i <= length(values)
                hours = get(hours_by_month, month, 0)
                if hours > 0
                    values[i] /= hours
                end
            end
        end
    end
    return monthly_values
end


function load_capacity_data(base_data_path::String, regions::Vector{String}; verbose::Bool=true)
    """
    从 Generators_data.csv 加载各技术的装机容量
    
    参数:
    - base_data_path: 数据目录路径
    - regions: 区域列表
    - verbose: 是否打印详细信息
    
    返回: Dict{String, Dict{String, Float64}}
         结构: capacity_data["coal"]["HL"] = 5000.0 (GW)
    """
    generators_file = joinpath(base_data_path, "Generators_data.csv")
    
    if !isfile(generators_file)
        error("❌ Generators_data.csv not found at: $generators_file")
    end
    
    gen_df = CSV.read(generators_file, DataFrame)
    
    # 检查必需列
    required_cols = [:technology, :region, :Cap_Size]
    missing_cols = setdiff(required_cols, Symbol.(names(gen_df)))
    if !isempty(missing_cols)
        error("❌ Missing columns in Generators_data.csv: $missing_cols")
    end
    
    # 技术资源映射
    tech_mapping = Dict(
        "coal" => ["cogen_conventional_steam_coal", "conventional_steam_coal"],
        "wind" => ["onshore_wind_turbine"],
        "solar" => ["solar_photovoltaic"]
    )
    
    # 初始化容量字典
    capacity_data = Dict{String, Dict{String, Float64}}()
    
    for (tech, tech_list) in tech_mapping
        capacity_data[tech] = Dict{String, Float64}()
        
        for region in regions
            # 筛选该技术和区域的发电机
            filtered = filter(row -> 
                row.technology in tech_list && row.region == region,
                gen_df
            )
            
            if isempty(filtered)
                if verbose
                    @warn "No generators found for $tech in $region"
                end
                capacity_data[tech][region] = 0.0
            else
                # 汇总装机容量（MW → GW）
                total_cap_mw = sum(filtered.Cap_Size)
                capacity_data[tech][region] = total_cap_mw / 1000.0  # 转换为GW
            end
        end
    end
    
    # 打印装机容量摘要
    if verbose
        println("\n📊 CAPACITY DATA LOADED:")
        println("="^60)
        for (tech, regional_caps) in capacity_data
            println("   $tech:")
            for (region, cap) in sort(collect(regional_caps))
                @printf("      %5s: %8.2f GW\n", region, cap)
            end
            total_cap = sum(values(regional_caps))
            @printf("      Total: %8.2f GW\n", total_cap)
        end
        println("="^60)
    end
    
    return capacity_data
end


# 为 Back_validation.jl 提供的别名函数
function load_capacity_data_for_validation(base_data_path::String, regions::Vector{String})
    """
    为验证模式加载装机容量数据（load_capacity_data 的别名）
    
    用途: 在 Back_validation.jl 中调用，避免代码重复
    """
    return load_capacity_data(base_data_path, regions; verbose=true)
end

# 计算MLT基准容量
function calculate_mlt_baseline_capacity(hist_mlt_df::DataFrame, comparison_months::Vector{Symbol})
    """
    使用历史平均传输功率作为MLT的基准容量
    
    返回: 平均传输功率（GW）
    """
    all_hist_flows = Float64[]
    
    for row in eachrow(hist_mlt_df)
        for month in comparison_months
            push!(all_hist_flows, abs(row[month]))  # 取绝对值
        end
    end
    
    baseline_capacity = mean(all_hist_flows)
    
    println("\n📊 MLT BASELINE CAPACITY:")
    println("="^60)
    @printf("   Historical average flow: %.2f GW\n", baseline_capacity)
    println("   (Used for NRMSE normalization)")
    println("="^60)
    
    return baseline_capacity
end

# --- Processing multiple runs results ---
function process_run_results(run_dir::String, num_weeks::Int, total_hours::Int, regions::Vector{String}; back_test::Bool=false)
    """
    统一的结果处理函数：
    - back_test=false (默认): 训练数据处理，使用 run_dir/1/, run_dir/2/, ... 结构
    - back_test=true: 验证数据处理，使用当前工作目录的 1/, 2/, ... 结构
    """
    
    if back_test
        println("\n🔍 BACK-TEST MODE: Processing validation data")
        println("="^50)
        # Use the validation directory supplied by the caller.  This is
        # required when several back checks run concurrently in separate
        # work directories.
        base_dir = run_dir
        println("   Back test directory: $base_dir")
    else
        println("\n🔍 TRAINING MODE: Processing training data")
        println("="^50)
        base_dir = run_dir
        println("   Run directory: $base_dir")
    end
    
    # 检查文件结构
    week_folders_exist = all(isdir(joinpath(base_dir, string(week))) for week in 1:num_weeks)
    println("   Weekly folder structure (1/, 2/, 3/...): $week_folders_exist")
    
    if !week_folders_exist
        if back_test
            println("   Available folders in BACK_CHECK_DIR: $(readdir(base_dir))")
            error("❌ Back-test mode requires weekly folders (1/, 2/, 3/, ...) in: $base_dir")
        else
            println("   Available folders in run_dir: $(readdir(base_dir))")
            error("❌ Training mode requires weekly folders in run directory: $base_dir")
        end
    end
    
    
    # --- Part 1: 拼接 vGENDISPATCH_results.csv ---
    all_dispatch_df = DataFrame()
    hours_per_week = 168
    
    for week in 1:num_weeks
        week_folder = joinpath(base_dir, string(week))
        dispatch_file = joinpath(week_folder, "vGENDISPATCH_results.csv")
        if isfile(dispatch_file)
            weekly_dispatch = CSV.read(dispatch_file, DataFrame)
            select!(weekly_dispatch, Not([:Zone]))
            rename!(weekly_dispatch, Dict(Symbol("x$i") => Symbol("h$((week-1)*hours_per_week + i)") for i in 1:hours_per_week))
            if isempty(all_dispatch_df)
                all_dispatch_df = weekly_dispatch
            else
                all_dispatch_df = innerjoin(all_dispatch_df, weekly_dispatch, on=[:Index, :Region, :Resource])
            end
        end
    end

    # Part 1.5: 拼接 vFLOW_results.csv (MLT传输数据)
    all_flow_df = DataFrame()
    
    for week in 1:num_weeks
        week_folder = joinpath(base_dir, string(week))
        flow_file = joinpath(week_folder, "vFLOW_results.csv")
        
        if isfile(flow_file)
            weekly_flow = CSV.read(flow_file, DataFrame)

            if :Index in names(weekly_flow)
                select!(weekly_flow, Not(:Index))  
            end
            
            first_col = names(weekly_flow)[1]
            if first_col != :Path
                other_cols = names(weekly_flow)[2:end]
                select!(weekly_flow, first_col => :Path, other_cols)
            end

            rename!(weekly_flow, Dict(Symbol("x$i") => Symbol("h$((week-1)*hours_per_week + i)") for i in 1:hours_per_week))
            if isempty(all_flow_df)
                all_flow_df = weekly_flow
            else
                all_flow_df = innerjoin(all_flow_df, weekly_flow; on=:Path)
            end
        else
            @warn "Missing flow file for week $week: $flow_file"
        end
    end

    if !isempty(all_flow_df)
        println("   ✅ Concatenated flow data: $(size(all_flow_df))")
    else
        println("   ⚠️  No flow data available - MLT metrics will be skipped")
    end
            

    # --- Part 2: 计算月度发电量（煤电、风电、光伏）---
    if num_weeks < 52
        hours_in_month = fill(168 * 4, 3) # 168 hours * 4 weeks
    else
        days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        hours_in_month = days_in_month .* 24
    end
    month_end_hours = cumsum(hours_in_month)
    num_months = length(hours_in_month)

    # 初始化三种发电类型的字典
    monthly_coal_gen_by_region = Dict(r => zeros(num_months) for r in regions)
    monthly_wind_gen_by_region = Dict(r => zeros(num_months) for r in regions)
    monthly_solar_gen_by_region = Dict(r => zeros(num_months) for r in regions)
    
    hour_cols = Symbol.(["h$i" for i in 1:total_hours])

    # 计算煤电发电量
    for region in regions
        coal_dispatch_df = filter(row -> row.Resource in ["cogen_conventional_steam_coal", "conventional_steam_coal"] && row.Region == region, all_dispatch_df)
        if !isempty(coal_dispatch_df)
            hourly_regional_coal = vec(sum(Matrix(coal_dispatch_df[!, hour_cols]), dims=1))
            start_h = 1
            for m in 1:num_months
                end_h = min(month_end_hours[m], total_hours)
                if start_h <= end_h
                    monthly_coal_gen_by_region[region][m] = sum(hourly_regional_coal[start_h:end_h])
                end
                start_h = end_h + 1
            end
        end
    end

    # 计算风电发电量
    for region in regions
        wind_dispatch_df = filter(row -> row.Resource == "onshore_wind_turbine" && row.Region == region, all_dispatch_df)
        if !isempty(wind_dispatch_df)
            hourly_regional_wind = vec(sum(Matrix(wind_dispatch_df[!, hour_cols]), dims=1))
            start_h = 1
            for m in 1:num_months
                end_h = min(month_end_hours[m], total_hours)
                if start_h <= end_h
                    monthly_wind_gen_by_region[region][m] = sum(hourly_regional_wind[start_h:end_h])
                end
                start_h = end_h + 1
            end
        end
    end

    # 计算光伏发电量
    for region in regions
        solar_dispatch_df = filter(row -> row.Resource == "solar_photovoltaic" && row.Region == region, all_dispatch_df)
        if !isempty(solar_dispatch_df)
            hourly_regional_solar = vec(sum(Matrix(solar_dispatch_df[!, hour_cols]), dims=1))
            start_h = 1
            for m in 1:num_months
                end_h = min(month_end_hours[m], total_hours)
                if start_h <= end_h
                    monthly_solar_gen_by_region[region][m] = sum(hourly_regional_solar[start_h:end_h])
                end
                start_h = end_h + 1
            end
        end
    end

    # Part 2.5: 计算月度 MLT 传输量 ✅ 新增
    # ========================================================================
    monthly_mlt_flow_by_path = Dict{String, Vector{Float64}}()
    
    if !isempty(all_flow_df)
        path_name_mapping = Dict(
            1 => "HL_to_IME",
            2 => "HL_to_JL",
            3 => "IME_to_JL",
            4 => "IME_to_LN",
            5 => "JL_to_LN",
            6 => "IME_to_SD",
            7 => "LN_to_JB"
        )
        
        for row in eachrow(all_flow_df)
            # 获取路径索引并转换
            path_idx = row.Path isa Integer ? row.Path : parse(Int, string(row.Path))
            
            # 跳过未知索引
            if !haskey(path_name_mapping, path_idx)
                @warn "Unknown transmission path index: $path_idx"
                continue
            end
            
            path_name = path_name_mapping[path_idx]
            
            # 提取逐时流量
            hourly_flow = Vector(row[hour_cols])
            
            # 按月汇总
            monthly_flows = zeros(num_months)
            start_h = 1
            
            for m in 1:num_months
                end_h = min(month_end_hours[m], total_hours)
                if start_h <= end_h
                    monthly_flows[m] = sum(hourly_flow[start_h:end_h]) / 1000.0
                end
                start_h = end_h + 1
            end
            
            monthly_mlt_flow_by_path[path_name] = monthly_flows
        end
        
        println("   ✅ Calculated monthly MLT flows for $(length(monthly_mlt_flow_by_path)) paths")
    else
        println("   ⚠️  Skipping MLT flow calculation (no flow data)")
    end

    # 如果是测试案例，将3个月数据放入9-11月的位置
    if num_weeks < 52
        full_year_coal_gen = Dict(r => zeros(12) for r in regions)
        full_year_wind_gen = Dict(r => zeros(12) for r in regions)
        full_year_solar_gen = Dict(r => zeros(12) for r in regions)
        full_year_mlt_flow = Dict{String, Vector{Float64}}()
        
        for r in regions
            full_year_coal_gen[r][9:11] = monthly_coal_gen_by_region[r]
            full_year_wind_gen[r][9:11] = monthly_wind_gen_by_region[r]
            full_year_solar_gen[r][9:11] = monthly_solar_gen_by_region[r]
        end

        for (path, flows) in monthly_mlt_flow_by_path
            full_year_flows = zeros(12)
            full_year_flows[9:11] = flows
            full_year_mlt_flow[path] = full_year_flows
        end
        
        monthly_coal_gen_by_region = full_year_coal_gen
        monthly_wind_gen_by_region = full_year_wind_gen
        monthly_solar_gen_by_region = full_year_solar_gen
        monthly_mlt_flow_by_path = full_year_mlt_flow
    end

    # 返回三种发电类型的结果
    return monthly_coal_gen_by_region, monthly_wind_gen_by_region, monthly_solar_gen_by_region, monthly_mlt_flow_by_path
end


function load_historical_mlt_data(historical_data_path::String)
    """
    加载历史MLT传输数据（月度格式，保留流向符号）
    
    返回: DataFrame包含 (Path, Jan, Feb, ..., Dec) 列，单位 GWh
    
    说明:
    - 正值: 正向流动（按路径名称定义的方向）
    - 负值: 反向流动（与路径名称相反的方向）
    """
    mlt_file = joinpath(historical_data_path, "hist_mlt_trans_monthly.csv")
    
    if !isfile(mlt_file)
        error("❌ hist_mlt_trans_monthly.csv not found in: $historical_data_path")
    end
    
    # 直接读取，不做任何修改
    hist_mlt_df = CSV.read(mlt_file, DataFrame)
    
    println("\n📊 HISTORICAL MLT DATA LOADED:")
    println("="^60)
    println("   File: $(basename(mlt_file))")
    println("   Dimensions: $(size(hist_mlt_df))")
    println("   Raw column names: $(names(hist_mlt_df))")  
    
    first_col_name = names(hist_mlt_df)[1]
    if first_col_name != :Path && first_col_name != "Path"
        error("❌ Expected first column to be 'Path', got '$first_col_name'")
    end
    if first_col_name == "Path"
        rename!(hist_mlt_df, "Path" => :Path)
    end
    month_cols = [:Jan, :Feb, :Mar, :Apr, :May, :Jun, :Jul, :Aug, :Sep, :Oct, :Nov, :Dec]
    for month in month_cols
        hist_mlt_df[!, month] = hist_mlt_df[!, month] ./ 1000.0
    end
    println("   ✅ Unit conversion: MWh → GWh (÷ 1000)")
    println("   Transmission paths: $(length(hist_mlt_df.Path))")
    println("   Paths: $(hist_mlt_df.Path)")
    
    # 统计正负值分布
    for month in month_cols
        n_positive = sum(hist_mlt_df[!, month] .> 0)
        n_negative = sum(hist_mlt_df[!, month] .< 0)
        n_zero = sum(hist_mlt_df[!, month] .== 0)
        
        if n_negative > 0
            println("   $(month): $(n_positive) positive, $(n_negative) negative, $(n_zero) zero")
        end
    end
    
    # 显示样本数据（保留符号）
    println("\n   Sample data (with direction):")
    for (i, row) in enumerate(eachrow(hist_mlt_df))
        if i <= 3  # 只显示前3条路径
            @printf("   %15s: Jan=%.1f, Jun=%.1f, Sep=%.1f GWh\n",
                row.Path, row.Jan, row.Jun, row.Sep)
        end
    end
    
    println("   ℹ️  Flow direction preserved (negative = reverse flow)")
    println("="^60)
    
    return hist_mlt_df
end


function calculate_mlt_metrics(
    hist_mlt_df::DataFrame, 
    sim_mlt_monthly::Dict{String, Vector{Float64}}, 
    comparison_months::Vector{Symbol},
    mlt_baseline_capacity::Float64
)
    """
    计算MLT传输的NRMSE和其他性能指标
    
    参数:
    - mlt_baseline_capacity: MLT基准容量（历史平均传输功率，GW）
    
    公式:
    - RMSE_mlt = mean([RMSE_path1, RMSE_path2, ..., RMSE_path7])
    - NRMSE_mlt = RMSE_mlt / mlt_baseline_capacity
    """
    
    # 定义需要排除的路径
    excluded_paths = if ANALYSIS_YEAR == 2016
        ["IME_to_SD"]
    else
        String[]
    end
    
    # 计算每条路径的RMSE
    path_rmses = Float64[]
    
    # 收集所有匹配的数据（用于其他指标）
    matched_hist = Float64[]
    matched_sim = Float64[]
    
    n_months = length(comparison_months)
    
    for (sim_path, sim_monthly_flows) in sim_mlt_monthly
        if sim_path in excluded_paths
            continue
        end
        
        hist_row = filter(row -> row.Path == sim_path, hist_mlt_df)
        
        if !isempty(hist_row)
            # 提取历史值
            hist_values = [hist_row[1, month] for month in comparison_months]
            
            # 提取模拟值
            month_indices = if comparison_months == Symbol.(["Sep", "Oct", "Nov"])
                [9, 10, 11]
            elseif length(comparison_months) == 12
                1:12
            else
                error("❌ Unsupported comparison_months: $comparison_months")
            end
            
            sim_values = sim_monthly_flows[month_indices]
            
            # 计算该路径的RMSE
            path_rmse = sqrt(sum((sim_values .- hist_values) .^ 2) / n_months)
            push!(path_rmses, path_rmse)
            
            # 收集数据（用于其他指标）
            append!(matched_hist, hist_values)
            append!(matched_sim, sim_values)
        end
    end
    
    if isempty(path_rmses)
        error("❌ No matching transmission paths found")
    end
    
    # 计算全局RMSE（所有路径的平均）
    global_rmse_mlt = mean(path_rmses)
    
    # 计算NRMSE
    global_nrmse_mlt = global_rmse_mlt / mlt_baseline_capacity
    
    # 其他指标
    mae = mean(abs.(matched_sim .- matched_hist))
    mape = mean(abs.((matched_sim .- matched_hist) ./ (abs.(matched_hist) .+ 1e-6))) * 100
    direction_match_rate = mean(sign.(matched_sim) .== sign.(matched_hist)) * 100
    mean_signed_error = mean(matched_sim .- matched_hist)
    
    return (
        global_nrmse = global_nrmse_mlt,  
        global_rmse = global_rmse_mlt,    
        mae = mae,
        mape = mape,
        matched_paths = length(path_rmses),
        direction_match_rate = direction_match_rate,
        mean_signed_error = mean_signed_error
    )
end



function calculate_comprehensive_metrics(
    hist_df, 
    sim_df, 
    comparison_months, 
    regions,
    capacity_data::Dict{String, Dict{String, Float64}},
    tech_name::String
)
    """
    计算容量标准化的NRMSE和其他性能指标。
    本 monthly-average 版本要求 hist_df 和 sim_df 的月度值都已经从
    monthly energy total (GWh) 转成 monthly-average power (GW)。
    
    参数:
    - capacity_data: 装机容量数据 (GW)
    - tech_name: 技术名称 ("coal", "wind", "solar")
    
    公式:
    - e_{i,r} = RMSE_{i,r} / Cap_{r,i}
    - NRMSE_i = Σ_r (w_r × e_{i,r})
    """
    
    # 1. 全局RMSE计算（用于诊断）
    all_hist_values = Float64[]
    all_sim_values = Float64[]
    
    # 2. 区域级别指标
    regional_rmses = Dict{String, Float64}()
    regional_nrmses = Dict{String, Float64}() 
    regional_maes = Dict{String, Float64}()
    regional_smapes = Dict{String, Float64}()
    
    # 3. 月份指标
    monthly_data = Dict{Symbol, Tuple{Vector{Float64}, Vector{Float64}}}()
    for month in comparison_months
        monthly_data[month] = (Float64[], Float64[])
    end
    
    # 4. 月度数量（用于RMSE分母）
    n_months = length(comparison_months)
    
    # 收集所有数据并计算区域指标
    for region in regions
        hist_row = filter(row -> row.region == region, hist_df)
        sim_row = filter(row -> row.Region == region, sim_df)
        
        if !isempty(hist_row) && !isempty(sim_row)
            hist_vector = Vector(hist_row[1, comparison_months])
            sim_vector = Vector(sim_row[1, comparison_months])
            
            # 添加到全局数据
            append!(all_hist_values, hist_vector)
            append!(all_sim_values, sim_vector)
            
            # 计算区域RMSE
            regional_rmse = sqrt(sum((sim_vector .- hist_vector) .^ 2) / n_months)
            regional_rmses[region] = regional_rmse
            
            # 计算区域NRMSE
            cap_region = capacity_data[tech_name][region]
            
            if cap_region > 1e-6  
                denominator = cap_region  # GW, because monthly totals have been converted to average GW
                regional_nrmse = regional_rmse / denominator
            else
                @warn "Zero capacity for $tech_name in $region, setting NRMSE=Inf"
                regional_nrmse = Inf
            end
            
            regional_nrmses[region] = regional_nrmse
            
            # 计算区域MAE
            regional_maes[region] = mean(abs.(sim_vector .- hist_vector))
            
            # 计算区域sMAPE
            regional_smapes[region] = mean(
                2.0 * abs.(sim_vector .- hist_vector) ./ 
                (abs.(sim_vector) .+ abs.(hist_vector) .+ 1e-9)
            )
            
            # 按月份分组
            for (i, month) in enumerate(comparison_months)
                push!(monthly_data[month][1], hist_vector[i])
                push!(monthly_data[month][2], sim_vector[i])
            end
        else
            @warn "Missing data for $tech_name in region $region"
        end
    end
    
    # 计算月度指标
    monthly_rmses = Dict{Symbol, Float64}()
    monthly_maes = Dict{Symbol, Float64}()
    monthly_smapes = Dict{Symbol, Float64}()
    
    for month in comparison_months
        hist_vals, sim_vals = monthly_data[month]
        if !isempty(hist_vals)
            monthly_rmses[month] = sqrt(mean((sim_vals .- hist_vals) .^ 2))
            monthly_maes[month] = mean(abs.(sim_vals .- hist_vals))
            monthly_smapes[month] = mean(
                2.0 * abs.(sim_vals .- hist_vals) ./ 
                (abs.(sim_vals) .+ abs.(hist_vals) .+ 1e-9)
            )
        end
    end
    
    # 计算全局NRMSE（区域加权平均）
    global_nrmse = sum(PathConfig.REGION_WEIGHTS[r] * regional_nrmses[r] for r in regions)
    
    # 全局RMSE（仅用于诊断）
    global_rmse = sqrt(sum((all_sim_values .- all_hist_values) .^ 2) / length(all_sim_values))
    
    # 全局MAE, MAPE
    mae = mean(abs.(all_sim_values .- all_hist_values))
    mape = mean(abs.((all_sim_values .- all_hist_values) ./ (abs.(all_hist_values) .+ 1e-9))) * 100
    
    return (
        global_nrmse = global_nrmse,  
        global_rmse = global_rmse,    
        regional_rmses = regional_rmses,
        regional_nrmses = regional_nrmses,  
        regional_maes = regional_maes,
        regional_smapes = regional_smapes,
        monthly_rmses = monthly_rmses,
        monthly_maes = monthly_maes,
        monthly_smapes = monthly_smapes,
        mape = mape,
        mae = mae
    )
end


# --- Main Script Logic ---
function main()
    println("--- Starting Results Aggregation and Analysis (Monthly-Average NRMSE Mode) ---")

    # 加载数据
    parameters_df = CSV.read(joinpath(runs_path, "parameters.csv"), DataFrame)
    hist_coal_gen = CSV.read(joinpath(historical_data_path, "hist_coal_gen_monthly.csv"), DataFrame)
    hist_wind_gen = CSV.read(joinpath(historical_data_path, "hist_wind_gen_monthly.csv"), DataFrame)
    hist_solar_gen = CSV.read(joinpath(historical_data_path, "hist_solar_gen_monthly.csv"), DataFrame)
    hist_mlt_data = load_historical_mlt_data(historical_data_path)
    hist_month_hours = calendar_month_hours(ANALYSIS_YEAR)
    sim_month_hours = simulated_month_hours(num_weeks, total_hours)

    hist_coal_gen_avg = normalize_monthly_dataframe!(copy(hist_coal_gen), hist_month_hours)
    hist_wind_gen_avg = normalize_monthly_dataframe!(copy(hist_wind_gen), hist_month_hours)
    hist_solar_gen_avg = normalize_monthly_dataframe!(copy(hist_solar_gen), hist_month_hours)
    hist_mlt_data_avg = normalize_monthly_dataframe!(copy(hist_mlt_data), hist_month_hours)
    
    # 加载装机容量数据
    capacity_data = load_capacity_data(base_data_path, regions)
    
    println("✅ Supporting data loaded successfully.")
    println("Parameters DataFrame columns: $(names(parameters_df))")
    println("ℹ️  Monthly-average metric mode:")
    println("   Historical month hours: $(hist_month_hours)")
    println("   Simulated month hours:  $(sim_month_hours)")

    # 确定对比月份
    comparison_months = if num_weeks < 52
        Symbol.(["Sep", "Oct", "Nov"])
    else
        Symbol.(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    end
    println("ℹ️  Comparing results for months: $(comparison_months)")
    
    # 计算MLT基准容量
    mlt_baseline_capacity = calculate_mlt_baseline_capacity(hist_mlt_data_avg, comparison_months)

    results_list = []

    for i in 1:num_runs
        run_id_str = @sprintf("run_%03d", i)
        run_dir = joinpath(runs_path, run_id_str)

        if !isdir(run_dir)
            continue
        end
        
        is_run_complete = all(
            isfile(joinpath(run_dir, string(week), "vGENDISPATCH_results.csv")) && 
            isfile(joinpath(run_dir, string(week), "vFLOW_results.csv"))
            for week in 1:num_weeks
        )
        
        if !is_run_complete
            println("⚠️ WARNING: Skipping $(run_id_str) due to missing weekly results.")
            continue
        end
        
        println("Processing results for: $(run_id_str)...")

        # 处理模拟结果
        monthly_coal_gen_results, monthly_wind_gen_results, monthly_solar_gen_results, monthly_mlt_flow_results = 
            process_run_results(run_dir, num_weeks, total_hours, regions; back_test=false)
        
        # 转换单位: MWh to GWh
        for region in keys(monthly_coal_gen_results)
            monthly_coal_gen_results[region] = monthly_coal_gen_results[region] ./ 1000.0
            monthly_wind_gen_results[region] = monthly_wind_gen_results[region] ./ 1000.0
            monthly_solar_gen_results[region] = monthly_solar_gen_results[region] ./ 1000.0
        end

        # Convert simulated monthly totals (GWh) to monthly-average power (GW).
        normalize_monthly_dict!(monthly_coal_gen_results, sim_month_hours)
        normalize_monthly_dict!(monthly_wind_gen_results, sim_month_hours)
        normalize_monthly_dict!(monthly_solar_gen_results, sim_month_hours)
        normalize_monthly_dict!(monthly_mlt_flow_results, sim_month_hours)
        
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        
        # 保存煤电结果
        df_coal_sim = DataFrame(Region=regions)
        for (j, m_name) in enumerate(month_names)
            df_coal_sim[!, Symbol(m_name)] = [monthly_coal_gen_results[r][j] for r in regions]
        end
        CSV.write(joinpath(run_dir, "simulated_monthly_average_coal_gen.csv"), df_coal_sim)

        # 保存风电结果
        df_wind_sim = DataFrame(Region=regions)
        for (j, m_name) in enumerate(month_names)
            df_wind_sim[!, Symbol(m_name)] = [monthly_wind_gen_results[r][j] for r in regions]
        end
        CSV.write(joinpath(run_dir, "simulated_monthly_average_wind_gen.csv"), df_wind_sim)

        # 保存光伏结果
        df_solar_sim = DataFrame(Region=regions)
        for (j, m_name) in enumerate(month_names)
            df_solar_sim[!, Symbol(m_name)] = [monthly_solar_gen_results[r][j] for r in regions]
        end
        CSV.write(joinpath(run_dir, "simulated_monthly_average_solar_gen.csv"), df_solar_sim)

        # 保存MLT结果
        df_mlt_sim = DataFrame(Path = collect(keys(monthly_mlt_flow_results)))
        for (j, m_name) in enumerate(month_names)
            df_mlt_sim[!, Symbol(m_name)] = [monthly_mlt_flow_results[path][j] 
                                                for path in df_mlt_sim.Path]
        end
        sort!(df_mlt_sim, :Path)
        CSV.write(joinpath(run_dir, "simulated_monthly_average_mlt_trans.csv"), df_mlt_sim)
        
        # 计算NRMSE指标（传入capacity_data）
        coal_metrics = calculate_comprehensive_metrics(
            hist_coal_gen_avg, df_coal_sim, comparison_months, regions, 
            capacity_data, "coal")

        wind_metrics = calculate_comprehensive_metrics(
            hist_wind_gen_avg, df_wind_sim, comparison_months, regions,
            capacity_data, "wind")

        solar_metrics = calculate_comprehensive_metrics(
            hist_solar_gen_avg, df_solar_sim, comparison_months, regions,
            capacity_data, "solar")

        mlt_metrics = calculate_mlt_metrics(
            hist_mlt_data_avg, monthly_mlt_flow_results, comparison_months,
            mlt_baseline_capacity)

        # 保存结果（使用NRMSE）
        push!(results_list, (
            run_id = i,
            # 主要指标 - NRMSE
            nrmse_coal = coal_metrics.global_nrmse,
            nrmse_wind = wind_metrics.global_nrmse,
            nrmse_solar = solar_metrics.global_nrmse,
            nrmse_mlt = mlt_metrics.global_nrmse,
            # 诊断指标 - RMSE
            raw_rmse_coal = coal_metrics.global_rmse,
            raw_rmse_wind = wind_metrics.global_rmse,
            raw_rmse_solar = solar_metrics.global_rmse,
            raw_rmse_mlt = mlt_metrics.global_rmse,
            # 额外指标 - MAPE
            mape_coal = coal_metrics.mape,
            mape_wind = wind_metrics.mape,
            mape_solar = solar_metrics.mape,
            mape_mlt = mlt_metrics.mape,
            # 额外指标 - MAE
            raw_mae_coal = coal_metrics.mae,
            raw_mae_wind = wind_metrics.mae,
            raw_mae_solar = solar_metrics.mae,
            raw_mae_mlt = mlt_metrics.mae,
            # MLT特定指标
            mlt_direction_match_rate = mlt_metrics.direction_match_rate,
            mlt_mean_signed_error = mlt_metrics.mean_signed_error,
            # 详细分析（保存为字符串）
            regional_rmse_coal_str = string(coal_metrics.regional_rmses),
            regional_rmse_wind_str = string(wind_metrics.regional_rmses),
            regional_rmse_solar_str = string(solar_metrics.regional_rmses),
            regional_nrmse_coal_str = string(coal_metrics.regional_nrmses),  
            regional_nrmse_wind_str = string(wind_metrics.regional_nrmses),
            regional_nrmse_solar_str = string(solar_metrics.regional_nrmses),
            regional_mae_coal_str = string(coal_metrics.regional_maes),
            regional_mae_wind_str = string(wind_metrics.regional_maes),
            regional_mae_solar_str = string(solar_metrics.regional_maes),
            regional_smape_coal_str = string(coal_metrics.regional_smapes),
            regional_smape_wind_str = string(wind_metrics.regional_smapes),
            regional_smape_solar_str = string(solar_metrics.regional_smapes),
            # 月度分析
            monthly_rmse_coal_str = string(coal_metrics.monthly_rmses),
            monthly_rmse_wind_str = string(wind_metrics.monthly_rmses),
            monthly_rmse_solar_str = string(solar_metrics.monthly_rmses),
            monthly_mae_coal_str = string(coal_metrics.monthly_maes),
            monthly_mae_wind_str = string(wind_metrics.monthly_maes),
            monthly_mae_solar_str = string(solar_metrics.monthly_maes),
            monthly_smape_coal_str = string(coal_metrics.monthly_smapes),
            monthly_smape_wind_str = string(wind_metrics.monthly_smapes),
            monthly_smape_solar_str = string(solar_metrics.monthly_smapes)
        ))
    end

    results_df = DataFrame(results_list)

    # 检查是否有有效结果
    if isempty(results_df)
        println("\n❌ ERROR: No valid simulation results found!")
        println("   All runs were skipped due to missing weekly results.")
        println("   Please check that your simulation runs completed successfully.")
        return
    end

    println("✅ Found $(nrow(results_df)) valid simulation results to analyze.")

    # 计算TotalLoss（方案A：先聚合区域，再平方求和）
    results_df.total_loss = 
        WEIGHTS["coal_gen"] .* (results_df.nrmse_coal .^ 2) .+
        WEIGHTS["wind_gen"] .* (results_df.nrmse_wind .^ 2) .+
        WEIGHTS["solar_gen"] .* (results_df.nrmse_solar .^ 2) .+
        WEIGHTS["mlt_flow"] .* (results_df.nrmse_mlt .^ 2)
    

    # 合并参数和结果
    final_summary_df = innerjoin(parameters_df, results_df, on=:run_id)
    sort!(final_summary_df, :total_loss)

    # 检查 MLT slack violations
    println("\n🔍 Checking for MLT slack violations...")
    infeasible_tags = String[]
    
    for i in 1:nrow(final_summary_df)
        run_id = final_summary_df.run_id[i]
        run_id_str = @sprintf("run_%03d", run_id)
        run_dir = joinpath(runs_path, run_id_str)
        
        # 检查所有周文件夹
        has_violation = false
        if isdir(run_dir)
            for week in 1:num_weeks
                week_folder = joinpath(run_dir, string(week))
                violation_file = joinpath(week_folder, "mlt_slack_violations.csv")
                
                if isfile(violation_file)
                    has_violation = true
                    break  
                end
            end
        end
        
        push!(infeasible_tags, has_violation ? "infeasible" : "")
    end

    # 添加 infeasible 列
    final_summary_df.infeasible = infeasible_tags
    
    # 统计
    n_infeasible = count(x -> x == "infeasible", infeasible_tags)
    println("   Found $(n_infeasible) / $(nrow(final_summary_df)) runs with MLT violations")
    
    # 保存结果
    final_summary_filepath = joinpath(runs_path, "final_performance_summary.csv")
    CSV.write(final_summary_filepath, final_summary_df)

    println("\n--- Analysis Complete (NRMSE Mode)! ---")
    println("✅ Final summary file saved to: $(final_summary_filepath)")
    println("\n📊 NRMSE Statistics:")
    println("="^60)
    @printf("   Coal NRMSE:  mean=%.6f, std=%.6f\n", 
            mean(results_df.nrmse_coal), std(results_df.nrmse_coal))
    @printf("   Wind NRMSE:  mean=%.6f, std=%.6f\n",
            mean(results_df.nrmse_wind), std(results_df.nrmse_wind))
    @printf("   Solar NRMSE: mean=%.6f, std=%.6f\n",
            mean(results_df.nrmse_solar), std(results_df.nrmse_solar))
    @printf("   MLT NRMSE:   mean=%.6f, std=%.6f\n",
            mean(results_df.nrmse_mlt), std(results_df.nrmse_mlt))
    @printf("   TotalLoss:   mean=%.6f, std=%.6f\n",
            mean(results_df.total_loss), std(results_df.total_loss))
    println("="^60)
    
    println("\nTop 5 best-performing parameter sets (lowest TotalLoss):")
    print(first(final_summary_df[:, [:run_id, :nrmse_coal, :nrmse_wind, :nrmse_solar, :nrmse_mlt, :total_loss]], 5))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

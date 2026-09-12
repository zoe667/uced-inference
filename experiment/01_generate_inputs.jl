using DataFrames, CSV, PyCall, Printf, Random

# --- Configuration ---
include(joinpath(@__DIR__, "PathConfig.jl"))
using .PathConfig
PathConfig.print_config()
PathConfig.validate_paths()

num_runs = PathConfig.NUM_RUNS # using Latin Hypercude Sampling (LHS) to generate #UNM_RUNS parameter combinations
runs_path = PathConfig.RUN_DIR
base_data_path = PathConfig.BASE_DATA_PATH
base_generators_file_template = PathConfig.BASE_GENERATORS_FILE


# Baseline Parameters for Classification
BASELINE_CHP_MIN_POWER = 64 # as a percentage
BASELINE_NONCHP_MIN_POWER = 43 # as a percentage, for non-CHP units
BASELINE_TIME = 8          # in hours, for both types
BASELINE_QUOTA = 64        # as a percentage, 64% is the value that has no infeasibility

# Fixed assumptions for non-retrofitted non-CHP coal units. This group is held
# outside the inferred parameter space, but its technical parameters are still
# standardized when generating each run.
FIXED_NONCHP_NONRETRO_MIN_POWER = 0.40
FIXED_NONCHP_NONRETRO_TIME = 8

# --- Parameter Space Definition (Using Integers) ---
include(joinpath(@__DIR__,"ExperimentConfig.jl"))
using .ExperimentConfig: ACTIVE_POLICIES, get_active_policies, 
                         chp_parameter_ranges_retrofitted,
                         chp_parameter_ranges_non_retrofitted,
                         non_chp_parameter_ranges_non_retrofitted

function analyze_coal_units(filtered_generators_df::DataFrame)
    """
    分析煤电机组的两个组别：改造机组(1)、非改造机组(0)
    输入的 DataFrame 应该已经过滤，只包含煤电机组且 Retrofitted 列无缺失值
    """
    retrofitted_flag = string.(filtered_generators_df.Retrofitted)

    # 改造机组（Retrofitted == 1）
    retrofitted_mask = retrofitted_flag .== "1"
    
    # 按技术类型进一步细分改造机组
    retrofitted_chp_mask = retrofitted_mask .& (filtered_generators_df.technology .== "cogen_conventional_steam_coal")
    retrofitted_non_chp_mask = retrofitted_mask .& (filtered_generators_df.technology .== "conventional_steam_coal")
    
    # 非改造CHP机组（Retrofitted == 0 且是CHP）
    chp_non_retrofitted_mask = (retrofitted_flag .== "0") .& 
                               (filtered_generators_df.technology .== "cogen_conventional_steam_coal")
    
    # 非改造非CHP机组（Retrofitted == 0 且是非CHP）
    non_chp_non_retrofitted_mask = (retrofitted_flag .== "0") .& 
                                   (filtered_generators_df.technology .== "conventional_steam_coal")
    
    # 按尺寸分类统计 - 所有改造机组（不区分CHP/非CHP）
    retrofitted_size_breakdown = Dict()
    for size_class in ["0-300", "300-660", "660-1000"]
        min_cap, max_cap = parse.(Int, split(size_class, "-"))
        class_count = sum(retrofitted_mask .& 
                         (filtered_generators_df.Cap_Size .> min_cap) .& 
                         (filtered_generators_df.Cap_Size .<= max_cap))
        retrofitted_size_breakdown[size_class] = class_count
    end
    
    # 按尺寸分类统计 - 非改造CHP
    chp_non_retrofitted_size_breakdown = Dict()
    for size_class in ["0-300", "300-660", "660-1000"]
        min_cap, max_cap = parse.(Int, split(size_class, "-"))
        class_count = sum(chp_non_retrofitted_mask .& 
                         (filtered_generators_df.Cap_Size .> min_cap) .& 
                         (filtered_generators_df.Cap_Size .<= max_cap))
        chp_non_retrofitted_size_breakdown[size_class] = class_count
    end
    
    # 按尺寸分类统计 - 非改造非CHP
    non_chp_non_retrofitted_size_breakdown = Dict()
    for size_class in ["0-300", "300-660", "660-1000"]
        min_cap, max_cap = parse.(Int, split(size_class, "-"))
        class_count = sum(non_chp_non_retrofitted_mask .& 
                         (filtered_generators_df.Cap_Size .> min_cap) .& 
                         (filtered_generators_df.Cap_Size .<= max_cap))
        non_chp_non_retrofitted_size_breakdown[size_class] = class_count
    end
    
    return (
        total_retrofitted = sum(retrofitted_mask),
        total_retrofitted_chp = sum(retrofitted_chp_mask),
        total_retrofitted_non_chp = sum(retrofitted_non_chp_mask),
        retrofitted_breakdown = retrofitted_size_breakdown,
        total_chp_non_retrofitted = sum(chp_non_retrofitted_mask),
        chp_non_retrofitted_breakdown = chp_non_retrofitted_size_breakdown,
        total_non_chp_non_retrofitted = sum(non_chp_non_retrofitted_mask),
        non_chp_non_retrofitted_breakdown = non_chp_non_retrofitted_size_breakdown
    )
end


# Flexible parameter collection function
function collect_all_parameters()
    param_names = String[]
    param_bounds = Vector{Tuple{Float64, Float64}}()
    
    # 改造煤电试点参数：按 Retrofitted flag 定义，包含 cogen 与 conventional coal
    if !isempty(chp_parameter_ranges_retrofitted)
        for (size_class, params) in chp_parameter_ranges_retrofitted
            for (param_type, bounds) in params
                push!(param_names, "Coal_Retrofitted_$(param_type)_$(size_class)")
                push!(param_bounds, bounds)
            end
        end
        println("Added $(length(chp_parameter_ranges_retrofitted) * 2) Retrofitted Coal parameters")
    end

    # 非改造CHP机组参数
    if !isempty(chp_parameter_ranges_non_retrofitted)
        for (size_class, params) in chp_parameter_ranges_non_retrofitted
            for (param_type, bounds) in params
                push!(param_names, "CHP_NonRetrofitted_$(param_type)_$(size_class)")
                push!(param_bounds, bounds)
            end
        end
        println("Added $(length(chp_parameter_ranges_non_retrofitted) * 2) CHP Non-Retrofitted parameters")
    end
    
    # 非改造非CHP机组参数
    if @isdefined(non_chp_parameter_ranges_non_retrofitted) && !isempty(non_chp_parameter_ranges_non_retrofitted)
        for (size_class, params) in non_chp_parameter_ranges_non_retrofitted
            for (param_type, bounds) in params
                push!(param_names, "NonCHP_NonRetrofitted_$(param_type)_$(size_class)")
                push!(param_bounds, bounds)
            end
        end
        println("Added $(length(non_chp_parameter_ranges_non_retrofitted) * 2) Non-CHP Non-Retrofitted parameters")
    else
        println("Skipped Non-CHP Non-Retrofitted parameters (empty or undefined)")
    end
    
    # Policy parameters (保持不变)
    if !isempty(ACTIVE_POLICIES)
        active_policy_params = get_active_policies(ACTIVE_POLICIES)
        for (param_name, bounds) in active_policy_params
            push!(param_names, param_name)
            push!(param_bounds, bounds)
        end
        println("Added $(length(active_policy_params)) policy parameters")
    end
    
    return param_names, param_bounds
end

# Apply modifications and count them (updated for new policy structure)
function apply_coal_modifications(base_generators_df::DataFrame, params_for_run::DataFrameRow)
    modified_generators_df = deepcopy(base_generators_df)
    retrofitted_flag = string.(modified_generators_df.Retrofitted)
    
    # 初始化计数字典
    retrofitted_counts = Dict("0-300" => 0, "300-660" => 0, "660-1000" => 0)
    chp_non_retrofitted_counts = Dict("0-300" => 0, "300-660" => 0, "660-1000" => 0)
    non_chp_non_retrofitted_counts = Dict("0-300" => 0, "300-660" => 0, "660-1000" => 0)
    
    # 对每个尺寸类别处理
    for size_class in ["0-300", "300-660", "660-1000"]
        min_cap, max_cap = parse.(Int, split(size_class, "-"))
        
        # 改造机组（所有煤电类型）
        retrofitted_mask = .!ismissing.(modified_generators_df.Retrofitted) .&
                          (retrofitted_flag .== "1") .&
                          in.(modified_generators_df.technology, Ref(["cogen_conventional_steam_coal", "conventional_steam_coal"])) .&
                          (modified_generators_df.Cap_Size .> min_cap) .&
                          (modified_generators_df.Cap_Size .<= max_cap)
        
        # 非改造CHP机组
        chp_non_retrofitted_mask = .!ismissing.(modified_generators_df.Retrofitted) .&
                                   (retrofitted_flag .== "0") .&
                                   (modified_generators_df.technology .== "cogen_conventional_steam_coal") .&
                                   (modified_generators_df.Cap_Size .> min_cap) .&
                                   (modified_generators_df.Cap_Size .<= max_cap)
        
        # 非改造非CHP机组
        non_chp_non_retrofitted_mask = .!ismissing.(modified_generators_df.Retrofitted) .&
                                       (retrofitted_flag .== "0") .&
                                       (modified_generators_df.technology .== "conventional_steam_coal") .&
                                       (modified_generators_df.Cap_Size .> min_cap) .&
                                       (modified_generators_df.Cap_Size .<= max_cap)
        
        retrofitted_counts[size_class] = sum(retrofitted_mask)
        chp_non_retrofitted_counts[size_class] = sum(chp_non_retrofitted_mask)
        non_chp_non_retrofitted_counts[size_class] = sum(non_chp_non_retrofitted_mask)
        
        # 应用改造机组参数
        if sum(retrofitted_mask) > 0
            retro_min_power_param = "Coal_Retrofitted_Min_Power_$(size_class)"
            retro_time_param = "Coal_Retrofitted_Time_$(size_class)"
            
            if haskey(params_for_run, retro_min_power_param)
                modified_generators_df[retrofitted_mask, :Min_Power] .= params_for_run[retro_min_power_param]
            end
            if haskey(params_for_run, retro_time_param)
                time_val = Int(round(params_for_run[retro_time_param]))
                modified_generators_df[retrofitted_mask, :Up_Time] .= time_val
                modified_generators_df[retrofitted_mask, :Down_Time] .= time_val
            end
        end

        # 应用非改造CHP机组参数
        if sum(chp_non_retrofitted_mask) > 0
            chp_min_power_param = "CHP_NonRetrofitted_Min_Power_$(size_class)"
            chp_time_param = "CHP_NonRetrofitted_Time_$(size_class)"
            
            if haskey(params_for_run, chp_min_power_param)
                modified_generators_df[chp_non_retrofitted_mask, :Min_Power] .= params_for_run[chp_min_power_param]
            end
            if haskey(params_for_run, chp_time_param)
                time_val = Int(round(params_for_run[chp_time_param]))
                modified_generators_df[chp_non_retrofitted_mask, :Up_Time] .= time_val
                modified_generators_df[chp_non_retrofitted_mask, :Down_Time] .= time_val
            end
        end

        # 非改造非CHP煤电不参与参数推断；统一使用固定技术假设。
        if sum(non_chp_non_retrofitted_mask) > 0
            modified_generators_df[non_chp_non_retrofitted_mask, :Min_Power] .=
                FIXED_NONCHP_NONRETRO_MIN_POWER
            modified_generators_df[non_chp_non_retrofitted_mask, :Up_Time] .=
                FIXED_NONCHP_NONRETRO_TIME
            modified_generators_df[non_chp_non_retrofitted_mask, :Down_Time] .=
                FIXED_NONCHP_NONRETRO_TIME
        end
    end
    
    return modified_generators_df, retrofitted_counts, chp_non_retrofitted_counts, non_chp_non_retrofitted_counts
end


# Generate policy config files (updated for new structure)
function generate_policy_configs(params_for_run::DataFrameRow, run_dir::String)
    config_data = []
    
    # Handle quota policies by size class
    if "quota_policies" in ACTIVE_POLICIES
        for size_class in ["0-300", "300-660", "660-1000"]
            quota_param_name = "Minimum_Quota_%_$(size_class)"
            if haskey(params_for_run, quota_param_name)
                quota_val = params_for_run[quota_param_name]
                push!(config_data, (parameter="Minimum_Quota_%_$(size_class)", value=quota_val))
            end
        end
    end

    # Handle MLT policies
    if "mlt_policies" in ACTIVE_POLICIES
        mlt_param_name = "MLT_Band"
        if haskey(params_for_run, mlt_param_name)
            mlt_band_val = params_for_run[mlt_param_name]
            push!(config_data, (parameter="MLT_Band", value=mlt_band_val))
        end
    end

    # Handle other policy types if they become active
    if "carbon_tax_policies" in ACTIVE_POLICIES
        for size_class in ["0-300", "300-660", "660-1000"]
            tax_param_name = "Carbon_Tax_Rate_$(size_class)"
            if haskey(params_for_run, tax_param_name)
                tax_val = params_for_run[tax_param_name]
                push!(config_data, (parameter="Carbon_Tax_Rate_$(size_class)", value=tax_val))
            end
        end
    end
    
    # Save config file
    if !isempty(config_data)
        config_df = DataFrame(parameter=[d.parameter for d in config_data], 
                             value=[d.value for d in config_data])
        CSV.write(joinpath(run_dir, "run_config.csv"), config_df)
    end
end

function classify_run(params_for_run::DataFrameRow)
    # Use representative size class for comparison - fix parameter names
    # Check both retrofitted and non-retrofitted, use retrofitted as primary if available
    retro_min_power_param = "Coal_Retrofitted_Min_Power_300-660"
    non_retro_chp_min_power_param = "CHP_NonRetrofitted_Min_Power_300-660"
    retro_time_param = "Coal_Retrofitted_Time_300-660"
    non_retro_chp_time_param = "CHP_NonRetrofitted_Time_300-660"
    
    # Use retrofitted values if available, otherwise use non-retrofitted
    is_high_chp_min_power = if haskey(params_for_run, retro_min_power_param)
        params_for_run[retro_min_power_param] > (BASELINE_CHP_MIN_POWER / 100.0)
    elseif haskey(params_for_run, non_retro_chp_min_power_param)
        params_for_run[non_retro_chp_min_power_param] > (BASELINE_CHP_MIN_POWER / 100.0)
    else
        false
    end
    
    is_high_time = if haskey(params_for_run, retro_time_param)
        params_for_run[retro_time_param] > BASELINE_TIME
    elseif haskey(params_for_run, non_retro_chp_time_param)
        params_for_run[non_retro_chp_time_param] > BASELINE_TIME
    else
        false
    end
    
    # Only handle quota policy if it's active
    is_high_quota = false
    if "quota_policies" in ACTIVE_POLICIES
        quota_param_name = "Minimum_Quota_%_300-660"
        if haskey(params_for_run, quota_param_name)
            is_high_quota = params_for_run[quota_param_name] > (BASELINE_QUOTA / 100.0)
        end
    end

    is_tight_mlt = false
    BASELINE_MLT_BAND = 0.15  # 定义基准MLT band为15%
    if "mlt_policies" in ACTIVE_POLICIES
        mlt_param_name = "MLT_Band"
        if haskey(params_for_run, mlt_param_name)
            is_tight_mlt = params_for_run[mlt_param_name] < BASELINE_MLT_BAND
        end
    end

    # Only include active policies in description
    desc_parts = [
        is_high_chp_min_power ? "High_MP" : "Low_MP",
        is_high_time ? "High_Time" : "Low_Time"
    ]
    
    # Add policy descriptions only if they are active
    if "quota_policies" in ACTIVE_POLICIES
        push!(desc_parts, is_high_quota ? "High_Quota" : "Low_Quota")
    end
    
    if "mlt_policies" in ACTIVE_POLICIES
        push!(desc_parts, is_tight_mlt ? "Tight_MLT" : "Loose_MLT")
    end
    
    description_string = join(desc_parts, ", ")

    # Count only active policy parameters for classification
    low_param_conditions = [
        !is_high_chp_min_power,
        !is_high_time
    ]
    
    if "quota_policies" in ACTIVE_POLICIES
        push!(low_param_conditions, !is_high_quota)
    end
    
    if "mlt_policies" in ACTIVE_POLICIES
        push!(low_param_conditions, !is_tight_mlt)
    end
    
    low_param_count = sum(low_param_conditions)
    total_params = length(low_param_conditions)

    classification = if low_param_count == total_params
        "Fully More Flexible"
    elseif low_param_count >= total_params ÷ 2
        "Partially More Flexible"
    elseif low_param_count == 0
        "Fully Less Flexible"
    else
        "Partially Less Flexible"
    end
    
    return description_string, classification
end

# --- Main Script Logic ---

function main()
    println("--- Starting Experiment Setup ---")
    println("📋 Active policy types: $(ACTIVE_POLICIES)")

    if !isfile(base_generators_file_template)
        error("Base generators file template not found at: $(base_generators_file_template). Please check path.")
    end

    actual_num_samples = num_runs

    param_names, param_bounds = collect_all_parameters()

    master_param_file = joinpath(runs_path, "parameters.csv")
    use_existing_params = isfile(master_param_file)

    if use_existing_params
        println("\n🔍 Found existing parameters.csv file")
        println("📥 Loading parameters from: $(master_param_file)")
        parameters_df = CSV.read(master_param_file, DataFrame)
        actual_num_samples = nrow(parameters_df)
        println("✅ Loaded $(actual_num_samples) existing parameter sets")
    else
        println("\n🆕 No existing parameters.csv found, generating new parameters")
        param_names, param_bounds = collect_all_parameters()

        if isempty(param_names)
            error("No parameters defined! Check your ExperimentConfig.jl")
        end
        
        println("\n📊 Experiment Summary:")
        println("   Total parameters: $(length(param_names))")
        println("   Parameter names: $(join(param_names, ", "))")

        println("\n📂 Loading base generators from: $(base_generators_file_template)")
        base_generators_df = CSV.read(base_generators_file_template, DataFrame)
        coal_units_df = base_generators_df[
            in.(base_generators_df.technology, Ref(["cogen_conventional_steam_coal", "conventional_steam_coal"])) .&
            .!ismissing.(base_generators_df.Retrofitted),
            :
        ]
        println("   Filtered to $(nrow(coal_units_df)) coal units with valid Retrofitted info")
        # Read and analyze coal units first
        coal_stats = analyze_coal_units(coal_units_df)
    
        println("\n📊 COAL UNITS IN DATASET (RETROFITTED vs NON-RETROFITTED):")
        println("="^60)
        
        # 改造机组统计（包括详细的CHP/非CHP分解）
        println("  Retrofitted Coal Units (All): $(coal_stats.total_retrofitted)")
        println("    ├── CHP Retrofitted: $(coal_stats.total_retrofitted_chp)")
        println("    ├── Non-CHP Retrofitted: $(coal_stats.total_retrofitted_non_chp)")
        for (class_name, count) in coal_stats.retrofitted_breakdown
            println("    ├── $(class_name) MW: $count units")
        end
        
        # 非改造CHP机组统计
        println("  Non-Retrofitted CHP Units: $(coal_stats.total_chp_non_retrofitted)")
        for (class_name, count) in coal_stats.chp_non_retrofitted_breakdown
            println("   ├── $(class_name) MW: $count units")
        end
        
        # 非改造非CHP机组统计
        println("  Non-Retrofitted Non-CHP Units: $(coal_stats.total_non_chp_non_retrofitted)")
        for (class_name, count) in coal_stats.non_chp_non_retrofitted_breakdown
            println("   ├── $(class_name) MW: $count units")
        end

        # 计算预期的参数修改数量
        total_retrofitted_params = coal_stats.total_retrofitted * 3  
        total_chp_non_retrofitted_params = coal_stats.total_chp_non_retrofitted * 3
        total_non_chp_non_retrofitted_params = 0
        if !isempty(non_chp_parameter_ranges_non_retrofitted)
            total_non_chp_non_retrofitted_params = coal_stats.total_non_chp_non_retrofitted * 3
        end
        
        total_params = total_retrofitted_params + total_chp_non_retrofitted_params + total_non_chp_non_retrofitted_params
        
        println("📊 Each run modifies:")
        println("   - $(coal_stats.total_retrofitted) Retrofitted units × 3 parameters = $(coal_stats.total_retrofitted * 3) Retrofitted parameter changes")
        println("   - $(coal_stats.total_chp_non_retrofitted) CHP Non-Retrofitted units × 3 parameters = $(coal_stats.total_chp_non_retrofitted * 3) CHP Non-Retrofitted parameter changes")
        if total_non_chp_non_retrofitted_params > 0
            println("   - $(coal_stats.total_non_chp_non_retrofitted) Non-CHP Non-Retrofitted units × 3 parameters = $(total_non_chp_non_retrofitted_params) Non-CHP Non-Retrofitted parameter changes")
        else
            println("   - $(coal_stats.total_non_chp_non_retrofitted) Non-CHP Non-Retrofitted units × 0 parameters = 0 Non-CHP Non-Retrofitted parameter changes (skipped)")
        end
        println("   - Total: $(coal_stats.total_retrofitted + coal_stats.total_chp_non_retrofitted + (total_non_chp_non_retrofitted_params > 0 ? coal_stats.total_non_chp_non_retrofitted : 0)) units, $(total_params) parameter changes per run")
        println("="^60)

        # Generate LHS samples
        qmc = pyimport("scipy.stats.qmc")
        l_bounds = [bounds[1] for bounds in param_bounds]
        u_bounds = [bounds[2] for bounds in param_bounds]
        num_dimensions = length(param_names)

        sampler = qmc.LatinHypercube(d=num_dimensions, seed=7)
        samples_scaled = qmc.scale(sampler.random(n=num_runs), l_bounds, u_bounds)
        samples_rounded = round.(samples_scaled, digits=3)

        parameters_df = DataFrame(samples_rounded, param_names)
        insertcols!(parameters_df, 1, :run_id => 1:num_runs)

        # Classification
        classifications = []
        descriptions = []
        println("\nClassifying generated runs:")
        for row in eachrow(parameters_df)
            description_string, classification = classify_run(row)
            push!(descriptions, description_string)
            push!(classifications, classification)
            @printf("  run_%03d: [%s] -> %s\n", row.run_id, description_string, classification)
        end
        
        parameters_df.Description = descriptions
        parameters_df.Classification = classifications

    end # end of else for generating new parameters
    
    # Generate files (clean up runs directory except parameters.csv if it exists)
    if !use_existing_params
        rm(runs_path, force=true, recursive=true)
        mkpath(runs_path)
    else
        # Clean up existing run folders but keep parameters.csv
        for item in readdir(runs_path)
            item_path = joinpath(runs_path, item)
            if isdir(item_path)
                rm(item_path, force=true, recursive=true)
            end
        end
    end

    if !@isdefined(base_generators_df)
        println("\n📂 Loading base generators from: $(base_generators_file_template)")
        base_generators_df = CSV.read(base_generators_file_template, DataFrame)
        println("✅ Loaded $(nrow(base_generators_df)) generators")
    end
    
    # Save master parameters (only if newly generated)
    if !use_existing_params
        CSV.write(master_param_file, parameters_df)
        println("✅ Master parameters file saved to: $(master_param_file)")
    end
    
    println("\n📝 GENERATING RUNS AND VERIFYING COUNTS:")
    println("="^60)
    
    for i in 1:actual_num_samples
        run_id_str = @sprintf("run_%03d", i)
        run_dir = joinpath(runs_path, run_id_str)
        mkpath(run_dir)
        
        params_for_run = parameters_df[i, :]
        
        # Apply modifications and get counts
        modified_generators_df, retrofitted_counts, chp_non_retrofitted_counts, non_chp_non_retrofitted_counts = apply_coal_modifications(base_generators_df, params_for_run)
        
        # Save files
        CSV.write(joinpath(run_dir, "generators_modified.csv"), modified_generators_df)
        
        # Generate policy config files (updated)
        generate_policy_configs(params_for_run, run_dir)
        
        # --- Run Summary Printout ---
        total_retrofitted = sum(values(retrofitted_counts))
        total_chp_non_retrofitted = sum(values(chp_non_retrofitted_counts))
        total_non_chp_non_retrofitted = sum(values(non_chp_non_retrofitted_counts))
        total_modified = total_retrofitted + total_chp_non_retrofitted #+ total_non_chp_non_retrofitted
        actual_non_chp_modified = 0
        if !isempty(non_chp_parameter_ranges_non_retrofitted)
            actual_non_chp_modified = total_non_chp_non_retrofitted
            total_modified += actual_non_chp_modified
        end

        println("$run_id_str: Modified $(total_modified) units ($(total_retrofitted) retrofitted + $(total_chp_non_retrofitted) CHP non-retro$(actual_non_chp_modified > 0 ? " + $(actual_non_chp_modified) non-CHP non-retro" : ""))")
        
        if i <= 3 # Show size breakdown for first few runs
            print("   Retrofitted breakdown: "); println(join(["$(k):$(v)" for (k,v) in retrofitted_counts], ", "))
            print("   CHP non-retrofitted breakdown: "); println(join(["$(k):$(v)" for (k,v) in chp_non_retrofitted_counts], ", "))
            print("   Non-CHP non-retrofitted breakdown: "); println(join(["$(k):$(v)" for (k,v) in non_chp_non_retrofitted_counts], ", "))
        end
        # --- End of Updated Run Summary Printout ---
    end
    println("\n✅ Successfully created $(num_runs) input scenarios.")
    println("📊 Active policies: $(ACTIVE_POLICIES)")
end

main()

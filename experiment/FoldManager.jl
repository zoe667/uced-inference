module FoldManager
# 三步处理逻辑：
# 1. 聚合各技术的指标 (L2 for NRMSE, L1 for others)，L2不开方以保证平滑性
# 2. 判断 TotalLoss 偏度，决定是否 Box-Cox
# 3. CV Fold 内部进行均值平移和方差缩放

export create_shared_stratified_folds, save_fold_assignments, load_fold_assignments, 
       learn_y_transform_params, apply_y_transform, inverse_y_transform,
       apply_y_standardization

using DataFrames, CSV, Statistics, StatsBase, Random
include("PathConfig.jl")
using .PathConfig: WEIGHTS
using Discretizers

# ==============================================================================
# Step 1: 聚合函数（L1/L2 范数）
# ==============================================================================

function _compute_total_loss(coal::Vector{Float64}, wind::Vector{Float64}, 
                             solar::Vector{Float64}, mlt::Vector{Float64};
                             metric_type::String="NRMSE")
    """
    聚合4个指标为 TotalLoss
    
    参数:
    - metric_type: "NRMSE", "RMSE", or "MAE"
    
    聚合方式:
    - NRMSE: L2范数 = w1*x1² + w2*x2² + w3*x3² + w4*x4² (不开方，保持平滑性)
    - RMSE/MAE: L1范数 = w1*|x1| + w2*|x2| + w3*|x3| + w4*|x4|
    """
    w_coal = WEIGHTS["coal_gen"]
    w_wind = WEIGHTS["wind_gen"]
    w_solar = WEIGHTS["solar_gen"]
    w_mlt = WEIGHTS["mlt_flow"]
    
    if metric_type == "NRMSE"
        # L2 范数（平方和，不开方以保证平滑性）
        total_loss = 
            w_coal .* coal.^2 .+ 
            w_wind .* wind.^2 .+ 
            w_solar .* solar.^2 .+ 
            w_mlt .* mlt.^2
        
    elseif metric_type in ["RMSE", "MAE"]
        # L1 范数（绝对值和）
        total_loss = (
            w_coal .* abs.(coal) .+ 
            w_wind .* abs.(wind) .+ 
            w_solar .* abs.(solar) .+ 
            w_mlt .* abs.(mlt)
        )
    else
        error("Unknown metric_type: $metric_type. Must be 'NRMSE', 'RMSE', or 'MAE'")
    end
    
    return total_loss
end

# ==============================================================================
# Step 2: Box-Cox 变换（基于偏度判断）
# ==============================================================================

function _box_cox_transform(data::Vector{Float64}, lambda::Float64)
    """Box-Cox transformation for positive data"""
    if any(data .<= 0)
        error("Box-Cox requires all data > 0. Found minimum: $(minimum(data))")
    end
    
    if abs(lambda) < 1e-8
        return log.(data)
    else
        return (data .^ lambda .- 1) ./ lambda
    end
end

function _box_cox_inverse(transformed::Vector{Float64}, lambda::Float64)
    """Inverse Box-Cox transformation"""
    if abs(lambda) < 1e-8
        return exp.(transformed)
    else
        return (lambda .* transformed .+ 1) .^ (1 / lambda)
    end
end

function _find_optimal_boxcox_lambda(data::Vector{Float64}; 
                                     lambda_range::StepRangeLen=range(-2.0, 2.0, length=17),
                                     verbose::Bool=false)
    """Find optimal Box-Cox lambda by minimizing skewness"""
    
    # 确保数据为正
    if any(data .<= 0)
        if verbose
            println("      ⚠️  Data contains non-positive values, shifting...")
        end
        data = data .- minimum(data) .+ 1e-6
    end
    
    best_lambda = 1.0
    best_skewness = Inf
    
    for lambda in lambda_range
        try
            transformed = _box_cox_transform(data, lambda)
            current_skew = abs(skewness(transformed))
            
            if current_skew < best_skewness
                best_skewness = current_skew
                best_lambda = lambda
            end
        catch e
            if verbose
                println("      ⚠️  Lambda $lambda failed: $e")
            end
            continue
        end
    end
    
    if verbose
        println("      🔍 Box-Cox: λ=$(round(best_lambda, digits=3)), |skew|=$(round(best_skewness, digits=4))")
    end
    
    return best_lambda
end

# ==============================================================================
# Step 3: 学习全局变换参数
# ==============================================================================

function learn_y_transform_params(
    coal_raw::Vector, wind_raw::Vector, solar_raw::Vector, mlt_raw::Vector; 
    metric_type::String="NRMSE",
    apply_transform::Bool=true, 
    verbose::Bool=true,
    skew_threshold::Float64=1.0
)
    """
    三步法学习变换参数：
    1. 聚合4个指标为 TotalLoss (NRMSE用L2, RMSE/MAE用L1)
    2. 判断 TotalLoss 偏度，决定是否 Box-Cox
    3. 返回全局变换参数（fold内标准化在apply时处理）
    
    参数:
    - metric_type: "NRMSE", "RMSE", or "MAE"
    - apply_transform: 是否应用 Box-Cox 变换
    - skew_threshold: 偏度阈值，超过则进行变换
    
    返回:
    - Dict 包含变换参数和聚合方式
    """
    if verbose
        println("\n🔄 Learning Y-Transform Parameters (3-Step Pipeline)")
        println("="^60)
    end
    
    # ============== Step 1: 聚合为 TotalLoss ==============
    norm_type = metric_type == "NRMSE" ? "L2" : "L1"
    
    total_loss = _compute_total_loss(
        Vector{Float64}(coal_raw), 
        Vector{Float64}(wind_raw), 
        Vector{Float64}(solar_raw), 
        Vector{Float64}(mlt_raw),
        metric_type=metric_type
    )
    
    if verbose
        println("\n📊 Step 1: $norm_type Aggregation → TotalLoss (metric: $metric_type)")
        println("-"^50)
        if metric_type == "NRMSE"
            println("   Formula: w_coal*coal² + w_wind*wind² + w_solar*solar² + w_mlt*mlt²")
        else
            println("   Formula: w_coal*|coal| + w_wind*|wind| + w_solar*|solar| + w_mlt*|mlt|")
        end
        println("   Weights: coal=$(WEIGHTS["coal_gen"]), wind=$(WEIGHTS["wind_gen"]), " *
                "solar=$(WEIGHTS["solar_gen"]), mlt=$(WEIGHTS["mlt_flow"])")
        println("   TotalLoss range: [$(round(minimum(total_loss), digits=4)), " *
                "$(round(maximum(total_loss), digits=4))]")
    end
    
    # ============== Step 2: 判断是否需要 Box-Cox ==============
    loss_skew = skewness(total_loss)
    loss_kurt = kurtosis(total_loss)
    loss_min = minimum(total_loss)
    loss_mean = mean(total_loss)
    loss_std = std(total_loss)
    
    if verbose
        println("\n📊 Step 2: Assess TotalLoss Distribution")
        println("-"^50)
        println("   Mean:     $(round(loss_mean, digits=4))")
        println("   Std:      $(round(loss_std, digits=4))")
        println("   Skewness: $(round(loss_skew, digits=4))")
        println("   Kurtosis: $(round(loss_kurt, digits=4))")
        println("   Min:      $(round(loss_min, digits=4))")
    end
    
    # 处理非正值
    data_shift = 0.0
    if loss_min <= 0
        data_shift = -loss_min + 1e-6
        if verbose
            println("      ⚠️  TotalLoss contains non-positive values, shifting by $(round(data_shift, digits=6))")
        end
    end
    
    total_loss_shifted = total_loss .+ data_shift
    
    # 判断是否需要变换
    needs_transform = apply_transform && abs(loss_skew) > skew_threshold
    
    transform_params = Dict{String, Any}(
        "metric_type" => metric_type,
        "aggregation_method" => norm_type,
        "data_shift" => data_shift,
        "raw_mean" => loss_mean,
        "raw_std" => loss_std,
        "skewness_before" => loss_skew,
        "kurtosis_before" => loss_kurt
    )
    
    if needs_transform
        if verbose
            println("\n   🔧 TRANSFORMATION NEEDED:")
            println("      Reason: |skewness| = $(round(abs(loss_skew), digits=4)) > $skew_threshold")
        end
        
        # 寻找最优 lambda
        optimal_lambda = _find_optimal_boxcox_lambda(total_loss_shifted, verbose=verbose)
        
        # 应用变换
        transformed_data = _box_cox_transform(total_loss_shifted, optimal_lambda)
        transformed_skew = skewness(transformed_data)
        transformed_kurt = kurtosis(transformed_data)
        
        if verbose
            println("\n   ✅ Transformation Applied:")
            println("      Optimal λ: $(round(optimal_lambda, digits=4))")
            println("      Data shift: $(round(data_shift, digits=6))")
            println("      Skewness: $(round(loss_skew, digits=4)) → $(round(transformed_skew, digits=4))")
            println("      Kurtosis: $(round(loss_kurt, digits=4)) → $(round(transformed_kurt, digits=4))")
            println("      Improvement: Δskew=$(round(abs(loss_skew) - abs(transformed_skew), digits=4))")
            
            if abs(transformed_skew) > 1.5
                println("\n      ⚠️  WARNING: Post-transformation |skewness| = $(round(abs(transformed_skew), digits=4)) > 1.5")
                println("      📊 RECOMMENDATION: Check GP residuals after training")
            end
        end
        
        # 更新参数
        merge!(transform_params, Dict(
            "lambda" => optimal_lambda,
            "skewness_after" => transformed_skew,
            "kurtosis_after" => transformed_kurt,
            "transformation_applied" => true,
            "needs_residual_check" => abs(transformed_skew) > 1.5
        ))
    else
        if verbose
            if !apply_transform
                println("\n   ⏭️  TRANSFORMATION DISABLED (by user setting)")
            else
                println("\n   ✅ NO TRANSFORMATION NEEDED:")
                println("      |skewness| = $(round(abs(loss_skew), digits=4)) ≤ $skew_threshold")
            end
        end
        
        merge!(transform_params, Dict(
            "lambda" => nothing,
            "transformation_applied" => false
        ))
    end
    
    # ============== Summary ==============
    if verbose
        println("\n" * "="^60)
        println("📋 TRANSFORMATION SUMMARY:")
        println("   Metric: $metric_type")
        println("   Aggregation: $norm_type norm")
        println("   Data shift: $(round(data_shift, digits=6))")
        if transform_params["transformation_applied"]
            println("   Box-Cox: λ=$(round(transform_params["lambda"], digits=3))")
            println("   Skewness improvement: $(round(abs(loss_skew) - abs(transform_params["skewness_after"]), digits=4))")
        else
            println("   Box-Cox: Not applied")
        end
        
        if get(transform_params, "needs_residual_check", false)
            println("\n   ⚠️  Residual check recommended")
        end
        println("="^60)
    end
    
    return transform_params
end

# ==============================================================================
# Step 3: 应用变换（含 Fold 内标准化）
# ==============================================================================

function apply_y_transform(
    coal_raw::Vector, wind_raw::Vector, solar_raw::Vector, mlt_raw::Vector,
    params::Union{Dict, Nothing}=nothing;
    fold_train_indices::Union{Vector{Int}, Nothing}=nothing,
    apply_standardization::Bool=true,
    verbose::Bool=false
    )
    """
    应用学习的变换参数
    
    参数:
    - params: 全局变换参数（包含 metric_type）
    - fold_train_indices: 如果提供，则基于训练集计算均值/方差（Step 3）
    - apply_standardization: 是否应用标准化
    
    返回:
    - Vector{Float64}: 变换后的 TotalLoss
    - Dict: Fold内的标准化参数（用于逆变换）
    """
    
    # 如果没有参数，仅做基本标准化
    if isnothing(params)
        if verbose
            println("⚠️  No transformation parameters, using raw L1 aggregation + standardization")
        end
        
        total_loss = _compute_total_loss(
            Vector{Float64}(coal_raw),
            Vector{Float64}(wind_raw),
            Vector{Float64}(solar_raw),
            Vector{Float64}(mlt_raw),
            metric_type="RMSE"  # 默认使用L1
        )
        
        # Fold内标准化
        if !isnothing(fold_train_indices)
            train_mean = mean(total_loss[fold_train_indices])
            train_std = std(total_loss[fold_train_indices])
            train_std = train_std > 1e-9 ? train_std : 1.0
        else
            train_mean = mean(total_loss)
            train_std = std(total_loss)
            train_std = train_std > 1e-9 ? train_std : 1.0
        end
        
        normalized_loss = (total_loss .- train_mean) ./ train_std
        
        fold_params = Dict(
            "fold_mean" => train_mean,
            "fold_std" => train_std
        )
        
        return normalized_loss, fold_params
    end
    
    # ============== Step 1: 聚合 ==============
    metric_type = get(params, "metric_type", "NRMSE")
    
    total_loss = _compute_total_loss(
        Vector{Float64}(coal_raw),
        Vector{Float64}(wind_raw),
        Vector{Float64}(solar_raw),
        Vector{Float64}(mlt_raw),
        metric_type=metric_type
    )
    
    # ============== Step 2: Box-Cox（如果需要）==============
    data_shift = params["data_shift"]
    total_loss_shifted = total_loss .+ data_shift
    
    if params["transformation_applied"]
        lambda = params["lambda"]
        transformed_loss = _box_cox_transform(total_loss_shifted, lambda)
        
        if verbose
            println("   🔄 Applied Box-Cox: λ=$(round(lambda, digits=3)), shift=$(round(data_shift, digits=6))")
        end
    else
        transformed_loss = total_loss_shifted
        
        if verbose && data_shift > 0
            println("   📊 Applied shift: $(round(data_shift, digits=6))")
        end
    end

    if !apply_standardization
        # 只返回变换后的值，不标准化
        return transformed_loss, Dict(
            "mean" => nothing,
            "std" => nothing,
            "standardization_applied" => false
        )
    end
    
    # ============== Step 3: Fold内标准化 ==============
    if !isnothing(fold_train_indices)
        # 仅使用训练集计算均值/方差
        fold_mean = mean(transformed_loss[fold_train_indices])
        fold_std = std(transformed_loss[fold_train_indices])
        fold_std = fold_std > 1e-9 ? fold_std : 1.0
        
        if verbose
            println("   📊 Fold-wise standardization: mean=$(round(fold_mean, digits=4)), std=$(round(fold_std, digits=4))")
        end
    else
        # 全局标准化
        fold_mean = mean(transformed_loss)
        fold_std = std(transformed_loss)
        fold_std = fold_std > 1e-9 ? fold_std : 1.0
    end
    
    normalized_loss = (transformed_loss .- fold_mean) ./ fold_std
    
    # 保存fold参数用于逆变换
    fold_params = Dict(
        "fold_mean" => fold_mean,
        "fold_std" => fold_std,
        "standardization_applied" => true
    )
    
    return normalized_loss, fold_params
end

# 在 export 列表中添加
export create_shared_stratified_folds, save_fold_assignments, load_fold_assignments, 
       learn_y_transform_params, apply_y_transform, inverse_y_transform,
       apply_y_standardization, standardize_y  # ✅ 添加这个


function standardize_y(
    y_raw::Vector{Float64},
    fold_train_indices::Union{Vector{Int}, Nothing};
    verbose::Bool=false
)
    """
    学习目标变量的标准化参数
    
    参数:
    - y_raw: 原始（或已变换）的目标变量
    - fold_train_indices: 训练集索引（如果提供，仅用训练集计算均值/方差）
    - verbose: 是否打印详细信息
    
    返回:
    - y_standardized: 标准化后的值
    - std_params: 标准化参数字典
    """
    
    # 计算均值和标准差
    if fold_train_indices !== nothing
        # 仅使用训练集
        y_train = y_raw[fold_train_indices]
        fold_mean = mean(y_train)
        fold_std = std(y_train)
    else
        # 使用全部数据
        fold_mean = mean(y_raw)
        fold_std = std(y_raw)
    end
    
    # 数值稳定性检查
    if fold_std < 1e-9
        @warn "Standard deviation is very small ($(fold_std)), using 1.0 to avoid division by zero"
        fold_std = 1.0
    end
    
    # 标准化
    y_standardized = (y_raw .- fold_mean) ./ fold_std
    
    if verbose
        println("   📊 Target standardization:")
        println("      Mean: $(round(fold_mean, digits=4))")
        println("      Std:  $(round(fold_std, digits=4))")
        println("      Range (standardized): [$(round(minimum(y_standardized), digits=3)), $(round(maximum(y_standardized), digits=3))]")
        
        # 验证
        post_mean = mean(y_standardized)
        post_std = std(y_standardized)
        println("      Verification:")
        println("        Post-standardization mean: $(round(post_mean, digits=6)) (should ≈ 0)")
        println("        Post-standardization std:  $(round(post_std, digits=6)) (should ≈ 1)")
        
        if abs(post_mean) > 1e-6
            @warn "Post-standardization mean is not close to 0: $post_mean"
        end
        if abs(post_std - 1.0) > 1e-6
            @warn "Post-standardization std is not close to 1: $post_std"
        end
    end
    
    # 保存标准化参数
    std_params = Dict(
        "mean" => fold_mean,
        "std" => fold_std
    )
    
    return y_standardized, std_params
end

function apply_y_standardization(
    y_raw::Vector{Float64},
    std_params::Dict
)
    """
    应用已学习的标准化参数
    
    参数:
    - y_raw: 原始（或已变换）的目标变量
    - std_params: 标准化参数字典
    
    返回:
    - 标准化后的值
    """
    fold_mean = std_params["mean"]
    fold_std = std_params["std"]
    
    return (y_raw .- fold_mean) ./ fold_std
end

# ==============================================================================
# 逆变换函数
# ==============================================================================

function inverse_y_transform(
    normalized_loss::Vector{Float64},
    global_params::Dict,
    fold_params::Dict
)
    """
    逆变换：normalized → original TotalLoss
    
    步骤:
    1. 逆标准化 (fold内)
    2. 逆 Box-Cox
    3. 逆数据偏移
    """
    
    # Step 1: 逆标准化
    fold_mean = fold_params["fold_mean"]
    fold_std = fold_params["fold_std"]
    transformed_loss = normalized_loss .* fold_std .+ fold_mean
    
    # Step 2: 逆 Box-Cox
    if global_params["transformation_applied"]
        lambda = global_params["lambda"]
        total_loss_shifted = _box_cox_inverse(transformed_loss, lambda)
    else
        total_loss_shifted = transformed_loss
    end
    
    # Step 3: 逆数据偏移
    data_shift = global_params["data_shift"]
    total_loss = total_loss_shifted .- data_shift
    
    return total_loss
end

# ==============================================================================
# Fold 创建函数
# ==============================================================================

function create_shared_stratified_folds(
    df::DataFrame, 
    metric_type::String="NRMSE",
    n_folds::Int=5, 
    apply_transform::Bool=true, 
    save_path::String=""
)
    """
    创建分层 fold 分配（基于 TotalLoss）
    
    参数:
    - df: 数据框
    - metric_type: "NRMSE", "RMSE", or "MAE"
    - n_folds: fold数量
    - apply_transform: 是否应用Box-Cox变换
    - save_path: 保存路径
    
    ⚠️  注意：分层使用原始 TotalLoss（未变换），确保样本均衡分布
    
    返回:
    - fold_assignments: fold分配向量
    - transform_params: 变换参数字典
    """
    println("--- Creating Stratified Folds (3-Step Pipeline) ---")
    Random.seed!(12345)
    
    # 根据 metric_type 提取相应的列
    if metric_type == "NRMSE"
        coal_raw = Vector{Float64}(df[!, "nrmse_coal"])
        wind_raw = Vector{Float64}(df[!, "nrmse_wind"])
        solar_raw = Vector{Float64}(df[!, "nrmse_solar"])
        mlt_raw = Vector{Float64}(df[!, "nrmse_mlt"])
    elseif metric_type == "RMSE"
        coal_raw = Vector{Float64}(df[!, "raw_rmse_coal"])
        wind_raw = Vector{Float64}(df[!, "raw_rmse_wind"])
        solar_raw = Vector{Float64}(df[!, "raw_rmse_solar"])
        mlt_raw = Vector{Float64}(df[!, "raw_rmse_mlt"])
    elseif metric_type == "MAE"
        coal_raw = Vector{Float64}(df[!, "raw_mae_coal"])
        wind_raw = Vector{Float64}(df[!, "raw_mae_wind"])
        solar_raw = Vector{Float64}(df[!, "raw_mae_solar"])
        mlt_raw = Vector{Float64}(df[!, "raw_mae_mlt"])
    else
        error("Unknown metric_type: $metric_type. Must be 'NRMSE', 'RMSE', or 'MAE'")
    end
    
    println("📊 Using metric: $metric_type")
    
    # 学习全局变换参数
    transform_params = learn_y_transform_params(
        coal_raw, wind_raw, solar_raw, mlt_raw, 
        metric_type=metric_type,
        apply_transform=apply_transform, 
        verbose=true
    )
    
    println("\n🔧 STRATIFICATION METHOD:")
    println("   Using L1 aggregation (no ^2 to avoid distribution bias)")
    
    # 分层时始终使用L1（绝对值和），避免分布偏差
    stratification_score = (
        PathConfig.WEIGHTS["coal_gen"] .* abs.(coal_raw) .+ 
        PathConfig.WEIGHTS["wind_gen"] .* abs.(wind_raw) .+ 
        PathConfig.WEIGHTS["solar_gen"] .* abs.(solar_raw) .+ 
        PathConfig.WEIGHTS["mlt_flow"] .* abs.(mlt_raw)
    )
    
    n_samples = length(stratification_score)
    
    # 使用更细粒度的桶（10个桶 = 每个fold约有2个桶）
    n_bins = n_folds * 2  # 例如 5 folds → 10 bins
    
    println("   Using Quantile Discretization:")
    println("      Bins: $n_bins (ensures balanced sample distribution)")
    
    algo = DiscretizeQuantile(n_bins)
    edges = binedges(algo, stratification_score)
    
    discretizer = LinearDiscretizer(edges)
    
    strat_labels = encode(discretizer, stratification_score)

    fold_assignments = Vector{Int}(undef, n_samples)
    
    rotation_counter = 0
    for bin_id in 1:n_bins
        bin_indices = findall(strat_labels .== bin_id)
        for idx in bin_indices
            fold_assignments[idx] = (rotation_counter % n_folds) + 1
            rotation_counter += 1
        end
    end
    
    # 验证
    fold_counts = [sum(fold_assignments .== k) for k in 1:n_folds]
    
    println("\n   Total samples: $n_samples")
    println("   Fold size range: [$(minimum(fold_counts)), $(maximum(fold_counts))]")
    println("   Fold size CV: $(round(std(fold_counts) / mean(fold_counts), digits=4))")

    # 显示均衡性
    println("\n   📊 Stratification Quality (by TotalLoss):")
    for k in 1:n_folds
        fold_mask = fold_assignments .== k
        fold_score = stratification_score[fold_mask]
        avg_loss = mean(fold_score)
        std_loss = std(fold_score)
        min_loss = minimum(fold_score)
        max_loss = maximum(fold_score)
        
        println("   Fold $k: $(sum(fold_mask)) samples")
        println("      TotalLoss: mean=$(round(avg_loss, digits=4)), std=$(round(std_loss, digits=4))")
        println("      Range: [$(round(min_loss, digits=4)), $(round(max_loss, digits=4))]")
    end

    # 验证每个 fold 都覆盖了全范围
    overall_min = minimum(stratification_score)
    overall_max = maximum(stratification_score)
    println("\n   ✅ Overall range: [$(round(overall_min, digits=4)), $(round(overall_max, digits=4))]")
    
    for k in 1:n_folds
        fold_mask = fold_assignments .== k
        fold_min = minimum(stratification_score[fold_mask])
        fold_max = maximum(stratification_score[fold_mask])
        coverage = (fold_max - fold_min) / (overall_max - overall_min) * 100
        println("   Fold $k coverage: $(round(coverage, digits=1))% of overall range")
    end
    
    if !isempty(save_path)
        save_fold_assignments(fold_assignments, save_path)
    end
    
    return fold_assignments, transform_params
end

# ==============================================================================
# 保存/加载函数
# ==============================================================================

function save_fold_assignments(fold_assignments::Vector{Int}, filepath::String)
    df_folds = DataFrame(
        sample_id = 1:length(fold_assignments),
        fold_assignment = fold_assignments
    )
    CSV.write(filepath, df_folds)
    println("📁 Fold assignments saved to: $filepath")
end

function load_fold_assignments(filepath::String)
    if !isfile(filepath)
        error("Fold assignments file not found: $filepath")
    end
    
    df_folds = CSV.read(filepath, DataFrame)
    fold_assignments = Vector{Int}(df_folds[!, "fold_assignment"])
    
    println("📁 Loaded $(length(fold_assignments)) fold assignments from: $filepath")
    
    return fold_assignments
end

end # module
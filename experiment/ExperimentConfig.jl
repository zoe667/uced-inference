# ExperimentConfig.jl
#
# This file serves as the single source of truth for the parameter
# ranges used in both the input generation and the optimization scripts.
module ExperimentConfig
export chp_parameter_ranges_retrofitted, chp_parameter_ranges_non_retrofitted, 
       non_chp_parameter_ranges_non_retrofitted, ACTIVE_POLICIES, get_active_policies

const ACTIVE_POLICIES = ["mlt_policies"]  # Modify for different experiments

# Retrofitted parameters are policy-defined, not technology-defined:
# they apply to all official flexibility-retrofit coal pilot units
# (Retrofitted == 1 and coal technology), including both cogen and
# conventional coal rows in the generator data. In generated parameter
# tables this group is named Coal_Retrofitted_*.
#
# Non-retrofitted CHP parameters are technology-defined:
# they apply only to non-retrofitted cogen_conventional_steam_coal rows.
# In generated parameter tables this group is named CHP_NonRetrofitted_*.
#
# For a clean inference design, keep comparable ex-ante ranges across
# retrofitted coal and non-retrofitted CHP unless intentionally running
# a sensitivity experiment with different prior bounds.
const chp_parameter_ranges_retrofitted = Dict(
    "0-300" => Dict(
        "Min_Power" => (0.30, 0.75),  # 0.30, 0.75 
        "Time" => (4.0, 12.0)             
    ),    
    "300-660" => Dict(
        "Min_Power" => (0.30, 0.75),  # 0.30, 0.75
        "Time" => (4.0, 16.0)         
    ),
    "660-1000" => Dict(
        "Min_Power" => (0.30, 0.75),  
        "Time" => (4.0, 16.0)         
    ),
)

const chp_parameter_ranges_non_retrofitted = Dict(
    "0-300" => Dict(
        "Min_Power" => (0.30, 0.75),  
        "Time" => (4.0, 12.0)         
    ),    
    "300-660" => Dict(
        "Min_Power" => (0.30, 0.75),  
        "Time" => (4.0, 16.0)         
    ),
    "660-1000" => Dict(
        "Min_Power" => (0.300, 0.750),  
        "Time" => (4.0, 16.0)         
    ),
)

const non_chp_parameter_ranges_non_retrofitted = Dict(
    #=
    "0-300" => Dict(
        "Min_Power" => (40.0, 65.0),  
        "Time" => (4.0, 12.0)         
    ),    
    "300-660" => Dict(
        "Min_Power" => (30.0, 60.0),  
        "Time" => (4.0, 16.0)         
    ),
    "660-1000" => Dict(
        "Min_Power" => (30.0, 60.0),  
        "Time" => (4.0, 16.0)         
    ),
    =#
)

# Policy Parameter Space Definition (Extensible Structure)
const policy_parameter_ranges = Dict(
    # Quota policies by size category
    "quota_policies" => Dict(  # for CHP in non-heating season (due to must run policy in winter months) 
        "0-300" => (0, 0.60),
        "300-660" => (0, 0.60),
        "660-1000" => (0, 0.60)
    ),
    
    "mlt_policies" => Dict(
        "band" => (0.05, 0.40)  # band范围从5%到40%，对应约束范围(1-band, 1+band)
    ),
    
    # Regional policies (example)
    "regional_policies" => Dict(
        "HL" => (20, 40),
        "IME" => (25, 45),
        "JL" => (15, 35),
        "LN" => (18, 38)
    )
)


# Helper function to get active policies (for selective experimentation)
function get_active_policies(active_policy_types::Vector{String}=["quota_policies"])
    param_dict = Dict{String, Tuple{Float64, Float64}}()  # Or Dict{String, Vector{Float64}}
    # Define parameter names for each policy type
    policy_param_names = Dict(
        "quota_policies" => "Minimum_Quota_%", # add if needed
        "mlt_policies" => "MLT_Band",
    )
    for policy_type in active_policy_types
        if haskey(policy_parameter_ranges, policy_type)
            param_name = get(policy_param_names, policy_type, policy_type)
            
            # 特殊处理MLT policies（因为它不是按size分类的）
            if policy_type == "mlt_policies"
                for (category, bounds) in policy_parameter_ranges[policy_type]
                    full_param_name = param_name  # 直接使用MLT_Band作为参数名
                    param_dict[full_param_name] = (Float64(bounds[1]), Float64(bounds[2]))
                end
            else
                # 其他policies按原来的逻辑处理
                for (category, bounds) in policy_parameter_ranges[policy_type]
                    full_param_name = "$(param_name)_$(category)"
                    param_dict[full_param_name] = (Float64(bounds[1]), Float64(bounds[2]))
                end
            end
        else
            @warn "Policy type '$policy_type' not found in policy_parameter_ranges"
        end
    end
    
    return param_dict
end


println("Loaded experiment parameter ranges from ExperimentConfig.jl")
end # module

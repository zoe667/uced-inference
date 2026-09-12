using Printf, CSV, DataFrames, Dates
include(joinpath(@__DIR__, "PathConfig.jl"))
using .PathConfig
PathConfig.print_config()
PathConfig.validate_paths()

# --- Configuration ---
script_dir = @__DIR__

function resolve_model_script()
    if haskey(ENV, "UCED_MODEL_SCRIPT")
        candidate = abspath(ENV["UCED_MODEL_SCRIPT"])
        isfile(candidate) || error("UCED_MODEL_SCRIPT does not exist: $candidate")
        return candidate
    end

    candidate = joinpath(PathConfig.PROJECT_ROOT, "model", "Run.jl")
    isfile(candidate) || error("Main model script not found at: $candidate")
    return candidate
end

main_model_script = resolve_model_script()
num_runs = PathConfig.NUM_RUNS
runs_path = PathConfig.RUN_DIR

println("=== Batch Runner for Power System Optimization ===")
println("Number of runs to execute: $(num_runs)")
println("Available threads: $(Threads.nthreads())")
println("Project root: $(PathConfig.PROJECT_ROOT)")
println("Runs path: $(runs_path)")
println("Model script: $(main_model_script)")

# Validate setup
if !isdir(runs_path)
    error("Runs directory not found at: $runs_path")
end

# --- Core Functions ---

struct RunInfo
    id::Int
    dir::String
    status::Symbol
    start_time::Float64
    end_time::Float64
    error_msg::String
end

RunInfo(id::Int, dir::String) = RunInfo(id, dir, :pending, 0.0, 0.0, "")

function test_single_run(run_id::Int = 1)
    """Test a single run with full terminal output for debugging"""
    
    run_id_str = @sprintf("run_%03d", run_id)
    run_dir = joinpath(runs_path, run_id_str)
    
    println("🧪 Testing single run: $run_id_str")
    println("Directory: $run_dir")
    
    if !isdir(run_dir)
        error("Run directory not found: $run_dir")
    end
    
    # Validate required files
    generators_file = joinpath(run_dir, "generators_modified.csv")
    config_file = joinpath(run_dir, "run_config.csv")
    
    if !isfile(generators_file)
        error("generators_modified.csv not found in $run_dir")
    end
    
    if !isfile(config_file)
        error("run_config.csv not found in $run_dir")
    end
    
    # Show config parameters for debugging
    try
        config_df = CSV.read(config_file, DataFrame)
        println("\n📋 Config parameters for this run:")
        for row in eachrow(config_df)
            println("   $(row.parameter): $(row.value)")
        end
        println()
    catch e
        println("⚠️ Could not read config file: $e")
    end
    
    # Construct and execute command with full terminal output
    cmd = `julia --project=$(PathConfig.PROJECT_ROOT) $main_model_script
        --run-dir $(abspath(run_dir))
        --config-file $(abspath(config_file))`
    
    println("Executing command: $cmd")
    println("="^60)
    
    # Run with full terminal output (no file redirection)
    try
        result = run(cmd, wait=true)
        
        if result.exitcode == 0
            println("="^60)
            println("✅ Test run completed successfully!")
        else
            println("="^60)
            println("❌ Test run failed with exit code: $(result.exitcode)")
        end
        
        return result.exitcode == 0
    catch e
        println("💥 Error during test run: $e")
        return false
    end
end

function execute_single_run(run_info::RunInfo)
    """Execute a single optimization run"""
    
    run_id_str = @sprintf("run_%03d", run_info.id)
    
    # Validate run directory
    if !isdir(run_info.dir)
        return RunInfo(run_info.id, run_info.dir, :missing_dir, 0.0, 0.0, "Directory not found")
    end
    
    # Check required files
    generators_file = joinpath(run_info.dir, "generators_modified.csv")
    config_file = joinpath(run_info.dir, "run_config.csv")
    
    if !isfile(generators_file)
        return RunInfo(run_info.id, run_info.dir, :missing_files, 0.0, 0.0, "generators_modified.csv not found")
    end
    
    if !isfile(config_file)
        return RunInfo(run_info.id, run_info.dir, :missing_files, 0.0, 0.0, "run_config.csv not found")
    end

    # Validate config file has expected parameters (new validation)
    try
        config_df = CSV.read(config_file, DataFrame)
        println("   Config file loaded, $(nrow(config_df)) parameters found")
    catch e
        println("   Warning: Could not validate config file: $e")
    end
    
    start_time = time()
    thread_id = Threads.threadid()
    
    println("Thread $thread_id: Starting $run_id_str")
    
    try
        # Construct command for this specific run
        cmd = `julia --project=$(PathConfig.PROJECT_ROOT) $main_model_script
            --run-dir $(abspath(run_info.dir))
            --config-file $(abspath(joinpath(run_info.dir, "run_config.csv")))`

        # Set up logging
        log_file = joinpath(run_info.dir, "run.log")
        error_file = joinpath(run_info.dir, "run_errors.log")

        # Execute the model
        result = run(pipeline(cmd, stdout=log_file, stderr=error_file))
        
        end_time = time()
        duration = round(end_time - start_time, digits=1)
        #=
        if result.exitcode == 0
            println("✅ Thread $thread_id: Completed $run_id_str ($(duration)s)")
            return RunInfo(run_info.id, run_info.dir, :success, start_time, end_time, "")
        else
            println("❌ Thread $thread_id: Failed $run_id_str (exit code: $(result.exitcode))")
            return RunInfo(run_info.id, run_info.dir, :failed, start_time, end_time, "Exit code: $(result.exitcode)")
        end
        =#
        # === 新增:检查是否因infeasibility而终止 ===
        infeasible_marker = joinpath(run_info.dir, "INFEASIBLE.txt")
        success_marker = joinpath(run_info.dir, "SUCCESS.txt")
        
        if isfile(infeasible_marker)
            # 读取infeasibility详情
            infeasible_info = read(infeasible_marker, String)
            println("⚠️  Thread $thread_id: $run_id_str is INFEASIBLE ($(duration)s)")
            return RunInfo(run_info.id, run_info.dir, :infeasible, 
                          start_time, end_time, infeasible_info)
        
        elseif isfile(success_marker) && result.exitcode == 0
            println("✅ Thread $thread_id: Completed $run_id_str ($(duration)s)")
            return RunInfo(run_info.id, run_info.dir, :success, 
                          start_time, end_time, "")
        else
            println("❌ Thread $thread_id: Failed $run_id_str (exit code: $(result.exitcode))")
            return RunInfo(run_info.id, run_info.dir, :failed, 
                          start_time, end_time, "Exit code: $(result.exitcode)")
        end
        
    catch e
        end_time = time()
        error_msg = string(e)
        println("💥 Thread $thread_id: Error in $run_id_str - $error_msg")
        
        # Log error details
        try
            error_file = joinpath(run_info.dir, "run_errors.log")
            open(error_file, "a") do f
                println(f, "\n=== Julia Exception ===")
                println(f, "Time: $(Dates.now())")
                println(f, "Error: $e")
                println(f, "Backtrace:")
                Base.show_backtrace(f, catch_backtrace())
            end
        catch
            # If we can't write to error file, continue anyway
        end
        
        return RunInfo(run_info.id, run_info.dir, :error, start_time, end_time, error_msg)
    end
end

function prepare_run_list()
    """Prepare list of all runs to execute"""
    
    run_list = RunInfo[]
    
    batch_start = parse(Int, get(ENV, "BATCH_START", "1"))
    batch_end = parse(Int, get(ENV, "BATCH_END", string(num_runs)))

    batch_start >= 1 || error("BATCH_START must be >= 1")
    batch_end <= num_runs || error("BATCH_END must be <= NUM_RUNS")
    batch_start <= batch_end || error("BATCH_START must be <= BATCH_END")

    println("Running batch range: run_$(lpad(batch_start, 3, '0')) to run_$(lpad(batch_end, 3, '0'))")

    for i in batch_start:batch_end
        run_id_str = @sprintf("run_%03d", i)
        run_dir = joinpath(runs_path, run_id_str)
        push!(run_list, RunInfo(i, run_dir))
    end
    
    return run_list
end

function execute_all_runs()
    """Execute all runs in parallel"""
    
    println("🚀 Preparing to execute $num_runs optimization runs...")
    
    # Prepare run list
    run_list = prepare_run_list()
    
    # Check available runs
    valid_runs = filter(r -> isdir(r.dir), run_list)
    println("Found $(length(valid_runs)) valid run directories")
    
    if length(valid_runs) == 0
        error("No valid run directories found!")
    end
    
    # Execute runs in parallel
    println("⚡ Executing runs in parallel with $(Threads.nthreads()) threads...")
    
    results = Vector{RunInfo}(undef, length(valid_runs))
    
    # Use @threads for parallel execution
    # Each thread will pick up the next available run
    Threads.@threads for i in 1:length(valid_runs)
        results[i] = execute_single_run(valid_runs[i])
    end
    
    return results
end

function report_results(results::Vector{RunInfo})
    """Generate comprehensive results summary"""
    
    println("\n" * "="^60)
    println("BATCH EXECUTION SUMMARY")
    println("="^60)
    
    # Count results by status
    status_counts = Dict{Symbol, Int}()
    for result in results
        status_counts[result.status] = get(status_counts, result.status, 0) + 1
    end
    
    total_runs = length(results)
    success_count = get(status_counts, :success, 0)
    
    println("📊 Overall Results:")
    println("   Total runs attempted: $total_runs")
    println("   ✅ Successful: $(get(status_counts, :success, 0))")
    println("   ❌ Failed: $(get(status_counts, :failed, 0))")
    println("   💥 Errors: $(get(status_counts, :error, 0))")
    
    # Timing analysis for successful runs
    successful_runs = filter(r -> r.status == :success, results)
    if !isempty(successful_runs)
        durations = [r.end_time - r.start_time for r in successful_runs]
        avg_duration = round(sum(durations) / length(durations), digits=1)
        min_duration = round(minimum(durations), digits=1)
        max_duration = round(maximum(durations), digits=1)
        total_duration = round(sum(durations), digits=1)
        
        println("\n⏱️  Timing Analysis:")
        println("   Average run time: $(avg_duration)s")
        println("   Fastest run: $(min_duration)s")
        println("   Slowest run: $(max_duration)s")
        println("   Total compute time: $(total_duration)s ($(round(total_duration/60, digits=1)) minutes)")
    end
    
    # List failed runs
    failed_runs = filter(r -> r.status in [:failed, :error], results)
    if !isempty(failed_runs)
        println("\n⚠️  Failed/Error Runs:")
        for run in failed_runs
            run_id_str = @sprintf("run_%03d", run.id)
            println("   $run_id_str: $(run.status) - $(run.error_msg)")
        end
        println("\n💡 Check individual run.log and run_errors.log files for details.")
    end
    
    return success_count == total_runs
end

# --- Main Execution ---

function main()
    start_time = time()
    
    try
        # Execute all runs
        results = execute_all_runs()
        
        # Report results
        all_successful = report_results(results)
        
        end_time = time()
        total_time = round((end_time - start_time) / 60, digits=2)
        
        println("\n🏁 Batch execution completed in $total_time minutes")
        
        if all_successful
            println("🎉 All runs completed successfully!")
            exit(0)
        else
            println("⚠️  Some runs failed - check logs for details")
            exit(1)
        end
        
    catch e
        println("\n💥 Fatal error in batch runner:")
        println(e)
        Base.show_backtrace(stdout, catch_backtrace())
        exit(2)
    end
end

# --- Entry Point ---
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

#=
if abspath(PROGRAM_FILE) == @__FILE__
    # Check for test mode argument
    if length(ARGS) > 0 && ARGS[1] == "test"
        test_run_id = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 1
        test_single_run(test_run_id)
    else
        main()
    end
end
=#

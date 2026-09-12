module PathConfig

export SAVE_DIR, MODEL_YEAR,
       SCRIPT_DIR, PROJECT_ROOT,
       RUN_DIR, BASE_DATA_PATH, HISTORICAL_DATA_PATH,
       BASE_GENERATORS_FILE, BACK_CHECK_DIR,
       NUM_RUNS, NUM_WEEKS, HOURS_PER_WEEK, TOTAL_HOURS,
       REGIONS, WEIGHTS, REGION_WEIGHTS,
       print_config, validate_paths

# =============================================================================
# User settings
# Modify these values before running the workflow.
# =============================================================================

const SAVE_DIR = get(ENV, "RUNS_SAVE_DIR", "runs_2021_work")

# Number of parameter combinations to generate and UCED runs to execute.
const NUM_RUNS = parse(Int, get(ENV, "UCED_NUM_RUNS", "100"))

# Number of simulated weeks per UCED run.
# Use 52 for a full-year run.
const NUM_WEEKS = parse(Int, get(ENV, "UCED_NUM_WEEKS", "52"))

# =============================================================================
# Derived paths and settings
# Do not modify below unless the repository structure changes.
# =============================================================================

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = normpath(joinpath(SCRIPT_DIR, ".."))

const MODEL_YEAR = begin
    m = match(r"_(\d{4})(?:_|$)", SAVE_DIR)
    if m !== nothing
        m.captures[1]
    else
        error("Could not detect MODEL_YEAR from SAVE_DIR = $SAVE_DIR. Use names like runs_btmup_2016.")
    end
end

const UCED_DATA_ROOT = joinpath(PROJECT_ROOT, "data")
const BASE_DATA_PATH = joinpath(UCED_DATA_ROOT, "ne_$(MODEL_YEAR)_maininput")
const HISTORICAL_DATA_PATH = joinpath(UCED_DATA_ROOT, "hist_data_$(MODEL_YEAR)")

const RUN_DIR = joinpath(SCRIPT_DIR, SAVE_DIR)
const BACK_CHECK_DIR = joinpath(RUN_DIR, "back_check")
const BASE_GENERATORS_FILE = joinpath(BASE_DATA_PATH, "Generators_data.csv")

const HOURS_PER_WEEK = 168
const TOTAL_HOURS = NUM_WEEKS * HOURS_PER_WEEK

const REGIONS = ["HL", "IME", "LN", "JL"]

const WEIGHTS = Dict(
    "coal_gen" => 0.25,
    "wind_gen" => 0.25,
    "solar_gen" => 0.25,
    "mlt_flow" => 0.25
)

const REGION_WEIGHTS = Dict(
    "HL" => 0.25,
    "IME" => 0.25,
    "LN" => 0.25,
    "JL" => 0.25
)

function validate_paths()
    if NUM_RUNS <= 0
        error("NUM_RUNS must be positive. Got: $NUM_RUNS")
    end

    if NUM_WEEKS <= 0 || NUM_WEEKS > 52
        error("NUM_WEEKS must be between 1 and 52. Got: $NUM_WEEKS")
    end

    required_dirs = [BASE_DATA_PATH, HISTORICAL_DATA_PATH]
    for d in required_dirs
        if !isdir(d)
            error("Required directory not found: $d")
        end
    end

    if !isfile(BASE_GENERATORS_FILE)
        error("Generators_data.csv not found: $BASE_GENERATORS_FILE")
    end

    mkpath(RUN_DIR)
    mkpath(BACK_CHECK_DIR)

    return nothing
end

function print_config()
    println("=== Experiment Configuration ===")
    println("SAVE_DIR:             ", SAVE_DIR)
    println("MODEL_YEAR:           ", MODEL_YEAR)
    println("RUN_DIR:              ", RUN_DIR)
    println("BASE_DATA_PATH:       ", BASE_DATA_PATH)
    println("HISTORICAL_DATA_PATH: ", HISTORICAL_DATA_PATH)
    println("NUM_RUNS:             ", NUM_RUNS)
    println("NUM_WEEKS:            ", NUM_WEEKS)
    println("TOTAL_HOURS:          ", TOTAL_HOURS)
    println("REGIONS:              ", join(REGIONS, ", "))
    println("================================")
end

end

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
runs_dir="${RUNS_SAVE_DIR:-runs_2021_100}"
julia_bin="${JULIA_BIN:-julia}"
threads="${REVAL_UCED_THREADS:-8}"
roles=(surrogate_best medoid interior_1 interior_2 interior_3 interior_4)
log_dir="$script_dir/$runs_dir/revalidation_logs"

mkdir -p "$log_dir"

echo "Launching ${#roles[@]} UCED revalidations"
echo "Dataset: $runs_dir"
echo "Threads per process: $threads"
echo "Logs: $log_dir"

pids=()
for role in "${roles[@]}"; do
    echo "Starting $role"
    env \
        RUNS_SAVE_DIR="$runs_dir" \
        REVAL_UCED_THREADS="$threads" \
        "$julia_bin" --project="$repo_dir" \
        "$script_dir/05b_run_revalidation_candidate.jl" "$role" \
        >"$log_dir/$role.log" 2>&1 &
    pids+=("$!")
done

failed=0
for i in "${!roles[@]}"; do
    if wait "${pids[$i]}"; then
        echo "Completed ${roles[$i]}"
    else
        echo "FAILED ${roles[$i]} -- see $log_dir/${roles[$i]}.log" >&2
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    exit 1
fi

if [[ "${REVAL_DRY_RUN:-false}" == "true" ]]; then
    echo "Dry run complete; skipping cache verification and combined-result assembly"
    exit 0
fi

# A zero process exit is not enough: require the six reusable result caches
# before reporting that the parallel stage succeeded.
missing_cache=0
for role in "${roles[@]}"; do
    if [[ "$role" == "surrogate_best" ]]; then
        cache="$script_dir/$runs_dir/back_check_archive/backcheck_$role/backcheck_result.json"
    else
        cache="$script_dir/$runs_dir/back_check_archive/backcheck_${role}_eps10/backcheck_result.json"
    fi
    if [[ -s "$cache" ]]; then
        echo "Result cache: $cache"
    else
        echo "MISSING result cache: $cache" >&2
        missing_cache=1
    fi
done

if [[ "$missing_cache" -ne 0 ]]; then
    echo "Parallel UCED processes exited, but one or more result caches were not created." >&2
    exit 1
fi

role_list="surrogate_best,medoid,interior_1,interior_2,interior_3,interior_4"
echo "All UCED runs completed; assembling the combined validation table"
env \
    RUNS_SAVE_DIR="$runs_dir" \
    SEARCH_SAVE_DIR="$runs_dir" \
    REVAL_AUTO="$role_list" \
    REVAL_BV_NAMING=canonical \
    REVAL_FORCE=false \
    "$julia_bin" --project="$repo_dir" \
    "$script_dir/05_surrogate_search.jl" \
    >"$log_dir/aggregate.log" 2>&1

combined_file="$script_dir/$runs_dir/back_validation_results_eps10.csv"
if [[ ! -s "$combined_file" ]]; then
    echo "Combined validation table was not created: $combined_file" >&2
    exit 1
fi
echo "Done. Combined results: $combined_file"

#!/usr/bin/env bash
set -euo pipefail
year="${1:-}"
if [[ "$year" != "2016" && "$year" != "2021" ]]; then
  echo "Usage: $0 {2016|2021}" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="smoke_${year}"
export RUNS_SAVE_DIR="$run_name"
export UCED_NUM_RUNS=1
export UCED_NUM_WEEKS=1
export UCED_SELECTED_WEEKS=1
export UCED_RUNNAME="ne_${year}_SpotOnly"
export BATCH_START=1
export BATCH_END=1
julia --project="$root" "$root/experiment/01_generate_inputs.jl"
julia --project="$root" -t 1 "$root/experiment/02_run_batch.jl"
test -s "$root/experiment/$run_name/run_001/1/vGENDISPATCH_results.csv"
test -s "$root/experiment/$run_name/run_001/1/vFLOW_results.csv"
echo "PASS: one-design, one-week UCED smoke test completed for $year."

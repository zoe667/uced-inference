#!/usr/bin/env bash
set -euo pipefail
year="${1:-}"
if [[ "$year" != "2016" && "$year" != "2021" ]]; then
  echo "Usage: $0 {2016|2021}" >&2
  exit 2
fi
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_name="reproduction_${year}"
python3 "$root/reproduction/prepare_analysis.py" "$year" --force
export RUNS_SAVE_DIR="$run_name"
export SAVE_DIR_OVERRIDE="$run_name"
export SEARCH_SAVE_DIR="$run_name"
julia --project="$root" "$root/experiment/04_build_surrogate.jl"
julia --project="$root" "$root/experiment/04b_active_parameter_robustness.jl"
julia --project="$root" "$root/experiment/05_surrogate_search.jl"
echo "Reproduction outputs: $root/experiment/$run_name"

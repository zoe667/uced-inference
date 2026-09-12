#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MPLBACKEND=Agg
mkdir -p "$root/build/matplotlib-cache" "$root/build/xdg-cache"
export MPLCONFIGDIR="$root/build/matplotlib-cache"
export XDG_CACHE_HOME="$root/build/xdg-cache"
cd "$root"
julia --project=. paper/supplementary/scripts/extract_gp_validation.jl
julia --project=. paper/supplementary/scripts/refresh_dense_predictions.jl
python3 paper/supplementary/scripts/build_tables_figures.py
julia --project=. paper/supplementary/scripts/extract_revalidation.jl
python3 paper/supplementary/scripts/build_revalidation.py

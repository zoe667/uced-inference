#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root/build/matplotlib-cache" "$root/build/xdg-cache"
export MPLCONFIGDIR="$root/build/matplotlib-cache"
export XDG_CACHE_HOME="$root/build/xdg-cache"
export MPLBACKEND=Agg
python3 "$root/paper/figures/cross_year_parameter_concentration/generate_cross_year_parameter_concentration.py"
python3 "$root/paper/figures/component_nrmse_spotonly/replot_from_frozen_data.py"
if command -v Rscript >/dev/null 2>&1; then
  Rscript "$root/paper/figures/surrogate_performance/plot_surrogate_performance.R"
else
  echo "Rscript not found; retained the checked-in surrogate performance figure." >&2
fi

#!/usr/bin/env python3
"""Recreate the published NRMSE panel from its frozen plotting table."""
from pathlib import Path
import importlib.util
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("nrmse_builder", HERE / "build_spotonly_nrmse_figure.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
data = pd.read_csv(HERE / "tech_nrmse_relative_change.csv")
module.plot_cleveland(data, HERE / "fig_tech_nrmse_cleveland_spotonly.png")
print("Rebuilt NRMSE PDF and PNG from frozen plotting data.")

#!/usr/bin/env python3
"""Copy a frozen result set to a writable experiment directory."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("year", choices=("2016", "2021"))
p.add_argument("--force", action="store_true")
a = p.parse_args()
source = ROOT / "results" / a.year
target = ROOT / "experiment" / f"reproduction_{a.year}"
if target.exists():
    if not a.force:
        raise SystemExit(f"{target} exists; use --force to replace it")
    shutil.rmtree(target)
shutil.copytree(source, target)
print(target)
